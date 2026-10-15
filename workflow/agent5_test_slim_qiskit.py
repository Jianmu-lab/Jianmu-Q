# -*- coding: utf-8 -*-
"""Agent 5: faithfully translate either Agent-4 design into a thin Qiskit adapter.

This file normally calls the language model once per explicit invocation.  A
state adapter that fails the existing physical smoke test receives at most one
targeted translation repair; no ansatz redesign is permitted.  This file does
not optimize states, simulate observables, export QPY, or submit hardware.
Those deterministic responsibilities belong to the runtime modules.
"""

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import logging
import os
import pprint
import re
import sys
import time

import numpy as np
from openai import OpenAI
from llm_support import api_key_for_model, llm_connection_options
from vqe_experience_store import DEFAULT_DATABASE_PATH

from circuit_design_support import (
    format_observable_contract,
    format_state_preparation_contract,
    load_observable_contract,
    load_state_preparation_contract,
)
from parameter_contract import (
    ParameterContractError,
    format_parameter_contract,
    load_parameter_contract,
    load_parameter_overrides,
)
from observable_specification import (
    expand_observable_samples,
    load_observable_specification,
)
from machine_contracts import (
    format_machine_contract,
    load_machine_contract,
    validate_mapping_contract,
)
from pipeline_support import (
    categorized_output_directory,
    resolve_output_tag,
    write_stage_output_manifest,
)


MODE_STATE_PREPARATION = "state_preparation"
MODE_OBSERVABLE = "observable"

REQUIRED_FUNCTIONS = {
    MODE_STATE_PREPARATION: {
        "configure_runtime",
        "build_hamiltonian",
        "build_state_preparation",
    },
    MODE_OBSERVABLE: {
        "build_observable_measurements",
        "reconstruct_observable",
    },
}


def qiskit_environment_summary():
    """Return a short, runtime-verified API note for Agent 5's translator."""
    import qiskit

    probes = (
        ("qiskit", "QuantumCircuit"),
        ("qiskit", "transpile"),
        ("qiskit.circuit", "ParameterVector"),
        ("qiskit.circuit.library", "PauliEvolutionGate"),
        ("qiskit.quantum_info", "Operator"),
        ("qiskit.quantum_info", "SparsePauliOp"),
        ("qiskit.quantum_info", "Statevector"),
        ("qiskit.synthesis", "LieTrotter"),
        ("qiskit.synthesis", "MatrixExponential"),
        ("qiskit.synthesis", "SuzukiTrotter"),
        ("qiskit.synthesis.unitary.qsd", "qs_decomposition"),
        ("qiskit.synthesis", "synth_qsd"),
    )
    available, unavailable = [], []
    for module_name, attribute in probes:
        try:
            module = importlib.import_module(module_name)
            exists = hasattr(module, attribute)
        except (ImportError, AttributeError):
            exists = False
        entry = "from {} import {}".format(module_name, attribute)
        (available if exists else unavailable).append(entry)
    return "\n".join(
        [
            "Installed Qiskit version: {}".format(qiskit.__version__),
            "Runtime-verified available imports:",
            *("- " + entry for entry in available),
            "Runtime-verified unavailable imports (do not use):",
            *("- " + entry for entry in unavailable),
            "Use these verified names for version-sensitive synthesis; do not "
            "invent or rename a Qiskit API from memory.",
        ]
    )


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def extract_single_python_block(text):
    blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if len(blocks) != 1:
        raise ValueError(
            "Agent 5 must return exactly one fenced Python block; found {}.".format(
                len(blocks)
            )
        )
    return blocks[0].strip()


def _assignment_nodes(tree, name):
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                matches.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                matches.append(node)
    return matches


def _assignment_node(tree, name):
    matches = _assignment_nodes(tree, name)
    if len(matches) != 1:
        raise ValueError(
            "Generated adapter must define {} exactly once; found {}.".format(
                name, len(matches)
            )
        )
    return matches[0]


def extract_embedded_constant(source, name):
    """Read one top-level literal or json.loads(<literal string>) assignment."""
    node = _assignment_node(ast.parse(source), name)
    value_node = node.value
    if (
        isinstance(value_node, ast.Call)
        and isinstance(value_node.func, ast.Attribute)
        and isinstance(value_node.func.value, ast.Name)
        and value_node.func.value.id == "json"
        and value_node.func.attr == "loads"
        and len(value_node.args) == 1
        and not value_node.keywords
    ):
        encoded = ast.literal_eval(value_node.args[0])
        if not isinstance(encoded, str):
            raise ValueError("json.loads assignment for {} must contain a string.".format(name))
        return json.loads(encoded)
    return ast.literal_eval(value_node)


def inject_authoritative_constants(source, values, insert_missing=False):
    """Replace authoritative constants, optionally restoring omitted assignments."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    missing = []
    for name, value in values.items():
        replacement = "{} = {}\n".format(
            name,
            pprint.pformat(value, width=96, sort_dicts=True),
        )
        matches = _assignment_nodes(tree, name)
        if not matches and insert_missing:
            missing.append(replacement)
            continue
        node = _assignment_node(tree, name)
        replacements.append((node.lineno - 1, node.end_lineno, replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    normalized = "".join(lines)
    if missing:
        normalized_tree = ast.parse(normalized)
        insertion_line = 0
        for node in normalized_tree.body:
            is_docstring = (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and insertion_line == 0
            )
            if is_docstring or isinstance(node, (ast.Import, ast.ImportFrom)):
                insertion_line = node.end_lineno
                continue
            break
        normalized_lines = normalized.splitlines(keepends=True)
        normalized_lines[insertion_line:insertion_line] = missing + ["\n"]
        normalized = "".join(normalized_lines)
    ast.parse(normalized)
    return normalized


def ensure_runtime_configuration(source):
    call = "configure_runtime(PARAMETER_CONTRACT, OBSERVABLE_SPECIFICATION)"
    tree = ast.parse(source)
    configured = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "configure_runtime"
        for node in tree.body
    )
    if not configured:
        source = source.rstrip() + "\n\n" + call + "\n"
    ast.parse(source)
    return source


def ensure_state_runtime_configuration(source):
    call = "configure_runtime(PARAMETER_CONTRACT)"
    tree = ast.parse(source)
    configured = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "configure_runtime"
        for node in tree.body
    )
    if not configured:
        source = source.rstrip() + "\n\n" + call + "\n"
    ast.parse(source)
    return source


def validate_generated_adapter(
    source, mode,
):
    """Reject missing interfaces and runtime responsibilities before publishing."""
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(REQUIRED_FUNCTIONS[mode] - function_names)
    if missing:
        raise ValueError(
            "Generated {} adapter is missing: {}.".format(mode, ", ".join(missing))
        )
    if "main" in function_names:
        raise ValueError("Generated adapters must not define main().")

    forbidden_for_both = (
        "OpenAI(",
        "chat.completions",
        "scipy.optimize",
        "ReverseEstimatorGradient",
        "minimize(",
        "QiskitRuntimeService",
        "SamplerV2",
        "EstimatorV2",
        "qpy.dump",
        "os.replace(",
    )
    present = [token for token in forbidden_for_both if token in source]
    if present:
        raise ValueError(
            "Generated adapter took runtime responsibilities: {}.".format(
                ", ".join(present)
            )
        )

    if mode == MODE_STATE_PREPARATION:
        if not ({"physical_basis_indices", "physical_basis_vectors"} & function_names):
            raise ValueError(
                "State-preparation adapter must define physical_basis_indices() "
                "or physical_basis_vectors()."
            )
        if "ParameterVector" not in source:
            raise ValueError("State-preparation adapter must use ParameterVector.")
        for forbidden in ("build_observable_measurements", "reconstruct_observable"):
            if forbidden in function_names:
                raise ValueError("State-preparation adapter contains observable logic.")
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        if len(functions["configure_runtime"].args.args) != 1:
            raise ValueError(
                "configure_runtime must accept exactly one positional argument."
            )
    else:
        if "OBSERVABLE_SPECIFICATION" not in source:
            raise ValueError(
                "Observable adapter must embed OBSERVABLE_SPECIFICATION."
            )
        for forbidden in ("build_state_preparation", "build_ansatz"):
            if forbidden in function_names:
                raise ValueError("Observable adapter contains state-preparation logic.")
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        expected_arguments = {
            "build_observable_measurements": 3,
            "reconstruct_observable": 2,
        }
        if "configure_runtime" in functions:
            expected_arguments["configure_runtime"] = 2
        for name, count in expected_arguments.items():
            if len(functions[name].args.args) != count:
                raise ValueError(
                    "{} must accept exactly {} positional arguments.".format(name, count)
                )
    return source


def smoke_test_adapter(
    path, mode, mapping_contract, parameter_contract,
    observable_contract=None, observable_specification=None, vqe_config=None,
):
    """Construct the smallest real circuit; do not revalidate JSON contracts."""
    spec = importlib.util.spec_from_file_location("agent5_smoke_adapter", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import generated adapter.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qubit_count = mapping_contract["qubits"]["count"]
    labels = parameter_contract["external_states"]["labels"]
    if mode == MODE_STATE_PREPARATION:
        if module.build_hamiltonian().num_qubits != qubit_count:
            raise ValueError("Generated Hamiltonian has the wrong qubit count.")
        vector_basis = hasattr(module, "physical_basis_vectors")
        basis_function = (
            module.physical_basis_vectors if vector_basis
            else module.physical_basis_indices
        )
        accepts_label = bool(inspect.signature(basis_function).parameters)
        for label in labels:
            basis = basis_function(label) if accepts_label else basis_function()
            basis_array = np.asarray(basis)
            if basis_array.size == 0:
                raise ValueError(
                    "Generated physical basis is empty for {}.".format(label)
                )
            if vector_basis:
                if basis_array.ndim != 2 or basis_array.shape[0] != 2 ** qubit_count:
                    raise ValueError("Generated physical basis vectors have the wrong shape.")
                gram = basis_array.conj().T @ basis_array
                if not np.allclose(gram, np.eye(basis_array.shape[1]), atol=1e-8):
                    raise ValueError("Generated physical basis vectors are not orthonormal.")
            circuit, ordered = module.build_state_preparation(label, 1)
            if not isinstance(circuit, QuantumCircuit):
                raise TypeError("State builder did not return a QuantumCircuit.")
            if circuit.num_qubits != qubit_count or set(ordered) != set(circuit.parameters):
                raise ValueError("State builder returned an inconsistent D=1 circuit.")
            unsupported = sorted({
                item.operation.name for item in circuit.data
                if item.operation.name in {"cry", "crz"}
            })
            if unsupported:
                raise ValueError(
                    "State adapter must decompose reverse-gradient-incompatible "
                    "gates before returning: {}.".format(", ".join(unsupported))
                )
            # A user-requested quantum-number sector is a physical requirement,
            # not metadata.  Check the reference point and one deterministic
            # nonzero point so a falsely symmetry-preserving layer fails here,
            # before the expensive VQE run.
            parameter_points = [np.zeros(len(ordered), dtype=float)]
            if ordered:
                rng = np.random.default_rng(1729)
                parameter_points.append(
                    rng.uniform(-0.37, 0.37, size=len(ordered))
                )
            for point_index, values in enumerate(parameter_points):
                binding = {
                    parameter: float(value)
                    for parameter, value in zip(ordered, values)
                }
                state = np.asarray(
                    Statevector.from_instruction(
                        circuit.assign_parameters(binding, inplace=False)
                    ).data,
                    dtype=complex,
                )
                if vector_basis:
                    sector_weight = float(np.real(np.sum(
                        np.abs(basis_array.conj().T @ state) ** 2
                    )))
                else:
                    indices = np.asarray(basis_array, dtype=int).reshape(-1)
                    if np.any(indices < 0) or np.any(indices >= 2 ** qubit_count):
                        raise ValueError("Generated physical-basis index is out of range.")
                    sector_weight = float(np.real(np.sum(np.abs(state[indices]) ** 2)))
                if sector_weight < 1.0 - 1.0e-8:
                    point_name = "reference" if point_index == 0 else "nonzero"
                    detail = ""
                    if point_index != 0 and ordered:
                        # Reuse this smoke test to identify the first primitive
                        # that leaks.  This runs only after the combined point
                        # has already failed and gives Agent5 a bounded local
                        # repair target instead of asking for a full rewrite.
                        for parameter_index, parameter in enumerate(ordered):
                            isolated = np.zeros(len(ordered), dtype=float)
                            isolated[parameter_index] = 0.211
                            isolated_binding = {
                                item: float(value)
                                for item, value in zip(ordered, isolated)
                            }
                            isolated_state = np.asarray(
                                Statevector.from_instruction(
                                    circuit.assign_parameters(
                                        isolated_binding, inplace=False
                                    )
                                ).data,
                                dtype=complex,
                            )
                            isolated_weight = float(np.real(np.sum(
                                np.abs(
                                    basis_array.conj().T @ isolated_state
                                ) ** 2
                            ))) if vector_basis else float(np.real(np.sum(
                                np.abs(isolated_state[indices]) ** 2
                            )))
                            if isolated_weight < 1.0 - 1.0e-8:
                                detail = (
                                    "; first individually leaking parameter={}"
                                    "; isolated_sector_weight={:.12g}"
                                ).format(str(parameter), isolated_weight)
                                break
                    raise ValueError(
                        "State-preparation circuit leaves the declared target "
                        "sector at the {} parameter point for {} "
                        "(sector_weight={:.12g}){}.".format(
                            point_name, label, sector_weight, detail
                        )
                    )
        target_specs = (
            vqe_config.get("target_spec_by_state", {})
            if isinstance(vqe_config, dict) else {}
        )
        for label in labels:
            target = target_specs.get(label, {})
            if (
                target.get("kind") != "sector_eigenstate"
                or target.get("level_index", 0) <= 0
            ):
                continue
            level_count = int(target["level_index"]) + 1
            references = []
            parameter_counts = set()
            for level in range(level_count):
                try:
                    circuit, ordered = module.build_state_preparation(
                        label, 1, reference_level=level
                    )
                except TypeError as error:
                    raise ValueError(
                        "Excited-state adapter must support "
                        "build_state_preparation(state_label, depth, "
                        "reference_level=None)."
                    ) from error
                if not isinstance(circuit, QuantumCircuit):
                    raise TypeError("SSVQE reference builder did not return a circuit.")
                if circuit.num_qubits != qubit_count or set(ordered) != set(circuit.parameters):
                    raise ValueError("SSVQE reference returned an inconsistent circuit.")
                parameter_counts.add(len(ordered))
                zero_binding = {parameter: 0.0 for parameter in ordered}
                references.append(np.asarray(
                    Statevector.from_instruction(
                        circuit.assign_parameters(zero_binding, inplace=False)
                    ).data,
                    dtype=complex,
                ))
            if len(parameter_counts) != 1:
                raise ValueError(
                    "All SSVQE references must use the same logical parameter count."
                )
            gram = np.asarray(references) @ np.asarray(references).conj().T
            if not np.allclose(gram, np.eye(level_count), atol=1.0e-8):
                raise ValueError(
                    "SSVQE levels 0 through {} require mutually orthogonal "
                    "reference states for {}.".format(level_count - 1, label)
                )
        return

    sample_values = dict(
        expand_observable_samples(observable_specification)[0]["values"]
    )
    if hasattr(module, "configure_runtime"):
        module.configure_runtime(parameter_contract, observable_specification)
    evolution = observable_contract["time_evolution"]
    if evolution.get("required"):
        steps = evolution["recommended_test_steps"][0]
    else:
        steps = 0
    # Exercise adapters with a real nonempty unitary state circuit. An empty
    # circuit previously hid generated-code assumptions about gate attributes
    # until the full measurement run.
    states = {}
    for label in labels:
        state = QuantumCircuit(qubit_count)
        if qubit_count:
            state.x(0)
        states[label] = state
    recipes = module.build_observable_measurements(states, steps, sample_values)
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("Observable builder returned no circuit recipes.")
    expectations = {}
    for recipe in recipes:
        circuit = recipe.get("circuit")
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Observable recipe did not contain a QuantumCircuit.")
        if any(
            item.operation.name in {"measure", "reset", "initialize"}
            or item.clbits
            or getattr(item.operation, "condition", None) is not None
            for item in circuit.data
        ):
            raise ValueError(
                "Observable recipe contains a nonunitary or classically "
                "controlled operation."
            )
        expectations[recipe["recipe_id"]] = 0.0
    complex(module.reconstruct_observable(expectations, sample_values))


def build_vqe_run_config(
    state_contract, parameter_contract, override_document, timestamp,
    runtime_directory="outputs",
):
    """Translate user-owned runtime values without asking Agent 4 to choose them."""
    runtime = override_document.get("vqe_runtime")
    if not isinstance(runtime, dict):
        raise ParameterContractError("User override must contain a vqe_runtime object.")
    external_states = parameter_contract["external_states"]
    labels = external_states["labels"]
    targets = runtime.get("target_kind_by_state")
    if not isinstance(targets, dict) or set(targets) != set(labels):
        raise ParameterContractError(
            "vqe_runtime.target_kind_by_state must define every external-state label."
        )
    ansatz = state_contract["ansatz"]
    scan_all_depths = runtime.get("scan_all_depths", True)
    if not isinstance(scan_all_depths, bool):
        raise ParameterContractError("vqe_runtime.scan_all_depths must be boolean.")
    stop_restarts = runtime.get("stop_restarts_on_convergence", True)
    if not isinstance(stop_restarts, bool):
        raise ParameterContractError(
            "vqe_runtime.stop_restarts_on_convergence must be boolean."
        )
    record_experience = runtime.get("record_experience", True)
    if not isinstance(record_experience, bool):
        raise ParameterContractError("vqe_runtime.record_experience must be boolean.")
    experience_database = runtime.get(
        "experience_database", DEFAULT_DATABASE_PATH
    )
    if not isinstance(experience_database, str) or not experience_database:
        raise ParameterContractError(
            "vqe_runtime.experience_database must be a nonempty path."
        )
    config = {
        "state_labels": labels,
        "equivalent_state_groups": external_states.get("equivalent_groups", []),
        "target_kind_by_state": targets,
        "depth_schedule": ansatz["recommended_test_depths"],
        "nested_by_depth": ansatz["nested_by_depth"],
        "scan_all_depths": scan_all_depths,
        "stop_restarts_on_convergence": stop_restarts,
        "record_experience": record_experience,
        "experience_database": experience_database,
        "restart_count": runtime["restart_count"],
        "max_iterations": runtime["max_iterations"],
        "max_same_depth_continuations": runtime.get(
            "max_same_depth_continuations", 1
        ),
        "first_depth_lower_bound": runtime["first_depth_lower_bound"],
        "first_depth_upper_bound": runtime["first_depth_upper_bound"],
        "new_layer_value": runtime["new_layer_value"],
        "perturbation_scale": runtime["perturbation_scale"],
        "variance_tolerance": runtime["variance_tolerance"],
        "energy_tolerance": runtime["energy_tolerance"],
        "energy_tie_tolerance": runtime["energy_tie_tolerance"],
        "gradient_check_step": runtime["gradient_check_step"],
        "gradient_check_tolerance": runtime["gradient_check_tolerance"],
        "checkpoint_file": os.path.join(
            runtime_directory, "VQE_Runtime_Checkpoint_{}.json".format(timestamp)
        ),
        "result_file": os.path.join(
            runtime_directory, "VQE_Runtime_Result_{}.json".format(timestamp)
        ),
        "parameter_bundle_file": os.path.join(
            runtime_directory, "VQE_ParameterBundle_{}.json".format(timestamp)
        ),
        "random_seed": runtime["random_seed"],
    }
    if "target_spec_by_state" in runtime:
        config["target_spec_by_state"] = runtime["target_spec_by_state"]
    if "adaptive_depth" in runtime:
        if not isinstance(runtime["adaptive_depth"], dict):
            raise ParameterContractError("vqe_runtime.adaptive_depth must be an object.")
        config["adaptive_depth"] = runtime["adaptive_depth"]
    return config


def validate_runtime_settings_identity(parameter_contract, override_document):
    """Allow VQE changes while rejecting a different physical run identity."""
    if override_document.get("schema_version") != "1.0":
        raise ParameterContractError("Runtime-settings schema_version must be '1.0'.")
    if override_document.get("source") not in ("user", "pipeline_resolved"):
        raise ParameterContractError(
            "Runtime settings source must be 'user' or 'pipeline_resolved'."
        )
    values = override_document.get("resolved_values")
    if not isinstance(values, dict):
        raise ParameterContractError(
            "Runtime settings must contain the user-owned resolved_values object."
        )
    selected = {
        item["name"]: item["selected_value"]
        for item in parameter_contract["parameters"]
    }
    unknown = sorted(set(values) - set(selected))
    if unknown:
        raise ParameterContractError(
            "Runtime settings reference unknown physical parameters: {}.".format(
                ", ".join(unknown)
            )
        )
    mismatched = [
        name for name, value in values.items()
        if selected[name] != value
    ]
    if mismatched:
        raise ParameterContractError(
            "Runtime settings differ from the Agent-4 physical parameters: {}.".format(
                ", ".join(sorted(mismatched))
            )
        )
    external_states = override_document.get("external_states")
    if external_states is not None and external_states != parameter_contract["external_states"]:
        raise ParameterContractError(
            "Runtime settings differ from the Agent-4 external-state definitions."
        )


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class QiskitAdapterAgent:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", model_name="gpt-5.6-sol"):
        self.client = OpenAI(**llm_connection_options(model_name, api_key=api_key, base_url=base_url))
        self.model_name = model_name
        self.logger = setup_logger("Agent5_QiskitTranslator")

    def _call_model_once(self, system, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        return content

    def translate_state_preparation(
        self, mapping_contract, parameter_contract, state_contract
    ):
        qiskit_environment = qiskit_environment_summary()
        prompt = f"""
Return exactly one fenced Python block containing a thin model-specific Qiskit
state-preparation adapter and no other text.

[LOCAL QISKIT ENVIRONMENT]
{qiskit_environment}

[AUTHORITATIVE AGENT 3 QUBIT-MAPPING CONTRACT]
```json
{format_machine_contract(mapping_contract, validate_mapping_contract, expected_parameter_contract=parameter_contract)}
```

[AUTHORITATIVE PARAMETER CONTRACT]
```json
{format_parameter_contract(parameter_contract)}
```

[AUTHORITATIVE STATE-PREPARATION CONTRACT]
```json
{format_state_preparation_contract(state_contract, parameter_contract)}
```

Translate literally. Embed the two JSON contracts as PARAMETER_CONTRACT and
STATE_PREPARATION_CONTRACT. Define only these model-specific interfaces:

Every Qiskit gate parameter must implement Agent4.1's stated mathematical gate
action exactly. Do not infer an angle sign, phase convention, qubit order, or
complex-conjugation convention from a Qiskit gate name. This is a translation
requirement, not permission to redesign or scientifically review the circuit.
Do not force a synthesis result through a hand-selected basis_gates list unless
that exact conversion is supported by the displayed local Qiskit environment;
standard parameter-bound Qiskit gates may remain for downstream transpilation.

Hamiltonian coefficients in the mapping contract may be exact rational strings
such as `"-1/4"`. Parse them with `fractions.Fraction` before converting to a
numeric scalar; Python `complex("-1/4")` is invalid. Preserve their sign and
value exactly. A parameter-dependent coefficient is supplied by Agent3 as an
algebraic expression using exact Parameter Contract names. Translate that
expression literally into deterministic Python evaluated from the current
selected parameter values inside configure_runtime. Do not replace it by the
generation-run number, and do not infer a missing coefficient formula from
physics prose or from another run.

1. configure_runtime(parameter_contract): bind all numerical Hamiltonian
   coefficients and numerical model inputs from the supplied current Parameter
   Contract. Keep circuit structure and conventions fixed; reject a genuinely
   unsupported structural size instead of silently using embedded old values.
   Treat lattice size `L`, qubit count and site/link loops as runtime-derived
   values when Agent3 supplies the corresponding parameterized mapping and
   Agent4.1's design is a scalable repeated local construction;
   do not freeze the example contract's current `L` into literal Pauli strings
   or fixed loop bounds. Instantiate Agent3's declared index/support rules and
   verify that they reproduce its current-size expanded Pauli list; never infer
   a new-size Hamiltonian from that list alone. A fixed-size mapping or design may reject a
   new `L` explicitly, but it must never reuse the old Hamiltonian silently.
   Call configure_runtime(PARAMETER_CONTRACT) once at module import.
   build_hamiltonian(): return the currently bound Hamiltonian as one Qiskit
   SparsePauliOp with the exact qubit and Pauli-label convention.
2. Define one exact sector interface. Use
   physical_basis_indices(state_label=None) when the requested sector is spanned
   by a subset of computational-basis states. Otherwise define
   physical_basis_vectors(state_label=None), returning a two-dimensional complex
   array whose orthonormal columns span the requested sector. Read each state's
   target quantum numbers and sector-level request from the authoritative
   contracts. For an unrestricted target return the complete basis. Reject an
   unknown non-null label; never invent a penalty approximation.
3. build_state_preparation(state_label, depth, reference_level=None): return exactly
   (QuantumCircuit, ordered_parameters). Use ParameterVector, preserve Agent 4.1's
   reference state, layer order, logical parameter order, parameter sharing and
   exact physical-sector preservation. Support every declared state label and
   every positive integer depth; reject unknown labels, booleans, non-integers,
   and integers less than one. Agent 4.1's recommended_test_depths are only a
   default runtime scan schedule, never a whitelist of supported depths.
   With reference_level=None, build the requested external target as before.
   For a sector target with zero-based level k>0, accept every integer
   reference_level from 0 through k and prepare Agent4.1's corresponding
   orthogonal reference followed by the same shared parameterized unitary.
   Reject reference_level for an unrestricted ground target and reject values
   outside the designed range. Missing lower levels are runtime-internal SSVQE
   references; do not add them as external state labels.
   Equivalent states may reuse gate structure only when the Parameter Contract
   explicitly permits it.
   Distinct requested energy levels in the same quantum-number sector are a
   weighted-SSVQE group: implement mutually orthogonal reference states for all
   levels 0 through the largest requested k, including unrequested internal
   lower levels, followed by exactly the same parameterized unitary, with
   identical logical parameter count, order, gate meaning and sharing for every
   level. Separate Qiskit Parameter objects may be used per circuit, but binding
   the same numeric vector must apply the same unitary after the reference
   preparation. Do not add VQD projectors, overlap penalties or level-dependent
   ansatz layers.
   Translate Agent4.1's local symmetry-preserving reference operations and any
   required translation-orbit momentum projection literally. Do not replace
   them with convenient complementary computational-basis bitstrings. Preserve
   Agent3's translation-eigenvalue and momentum-phase convention, including
   normalization and relative phases. When translation is an exact symmetry,
   preserve Agent4.1's parameter sharing across translated generator copies so
   the shared unitary remains in the requested momentum sector. Do not invent
   a momentum projection for OBC or for a state without an exact momentum
   requirement.
   Follow every entry in Agent 4.1's ansatz.symmetry_preservation literally.
   For `strategy: exact_circuit`, the reference state and every implemented
   layer must stay in the declared target sector. A `diagnostic_only` entry is
   metadata, not a hard basis restriction: do not invent gates, penalties, or
   projection for it. The chosen physical-basis function must contain only the
   mandatory gauge/physical sector and quantum numbers explicitly defining the
   requested target state.
   An abstract fermionic rotation between nonadjacent Jordan--Wigner modes is
   not a bare two-qubit rotation: preserve the required parity string or use the
   fermionic-swap construction specified by Agent 4.1. Never silently drop it.
   Preserve the stated time order inside every composite basis transform. If
   Agent 4.1 specifies a mode permutation followed by a block transform, append
   the permutation first and the block transform second; do not reverse them
   while translating mathematical composition notation into Qiskit order.
   Return a circuit that Qiskit's reverse-gradient implementation can consume
   directly: the VQE runtime will not decompose or rewrite it. Do not return
   opaque fixed UnitaryGate instructions or unsupported parameterized
   controlled rotations such as CRY/CRZ inside the ansatz. Expand those gates
   here into an exactly equivalent sequence of supported scalar-parameter
   Qiskit gates; this translation must not change the designed unitary,
   parameter sharing, logical parameter order, or physical-sector preservation.
   For a custom Fourier, orbital, Bogoliubov, or other fermionic basis change,
   translate Agent4.1's mathematical mode matrix and mode ordering exactly. The
   compiled primitive must induce that one-particle transformation up to one
   common global phase, its stated inverse must undo it, and inserting any one
   parameterized generator must preserve every `exact_circuit` symmetry. Do
   not guess a QR row/column convention, silently reverse mathematical
   composition order, or reinterpret spectral indices. If the supplied gate
   recipe is inconsistent with the supplied mathematical matrix, implement the
   matrix convention faithfully with a standard decomposed construction rather
   than preserving the inconsistent recipe.
   Return the circuit in one consistent global-phase convention with
   circuit.global_phase equal to numeric zero. Never leave a variational
   ParameterExpression in QuantumCircuit.global_phase. This removes only the
   physically common phase; preserve every relative phase inside gate
   operations.

The adapter must contain no main(), optimizer, gradient, depth loop, checkpoint,
observable, time evolution, measurement, QPY export, simulator, run manifest or
IBM code. Do not import vqe_runtime. Do not improve or redesign Agent 4.1.
Do not define validate_adapter_contract(). configure_runtime may read only the
small set of numerical parameter values required by executable builders; do not
repeat full contract validation. Convert design data into simple executable
module constants used directly by the builder functions.
"""
        system = (
            "You are a literal Qiskit circuit translator. Implement the supplied "
            "state-preparation design only; make no physics or execution decisions."
        )
        source = extract_single_python_block(self._call_model_once(system, prompt))
        source = inject_authoritative_constants(
            source,
            {
                "PARAMETER_CONTRACT": parameter_contract,
                "STATE_PREPARATION_CONTRACT": state_contract,
            },
        )
        return validate_generated_adapter(source, MODE_STATE_PREPARATION)

    def repair_state_preparation(
        self, source, smoke_error, mapping_contract, parameter_contract,
        state_contract,
    ):
        qiskit_environment = qiskit_environment_summary()
        prompt = f"""
The Qiskit state-preparation adapter below passed structural validation but
failed the existing physical smoke test.

[LOCAL QISKIT ENVIRONMENT]
{qiskit_environment}

[SMOKE-TEST ERROR]
{smoke_error}

[CURRENT ADAPTER]
```python
{source}
```

[AUTHORITATIVE AGENT 3 QUBIT-MAPPING CONTRACT]
```json
{format_machine_contract(mapping_contract, validate_mapping_contract, expected_parameter_contract=parameter_contract)}
```

[AUTHORITATIVE STATE-PREPARATION CONTRACT]
```json
{format_state_preparation_contract(state_contract, parameter_contract)}
```

Return exactly one fenced Python block containing the COMPLETE repaired adapter,
including every import, embedded contract constant, helper, and public function
from the current adapter. Do not return a diff, patch, excerpt, isolated helper,
or replacement fragment. Make one targeted translation repair: preserve the
Hamiltonian, references, ansatz family, parameter order, public functions, and
all unrelated working code, and change only the primitive and its direct
helpers responsible for the reported failure. Do not redesign the ansatz,
remove a generator, weaken the physical basis, relax a tolerance, or delete a
symmetry check.

As in the initial translation, every Qiskit gate parameter must implement
Agent4.1's stated mathematical action exactly; never guess angle signs, phase
conventions, qubit ordering, or complex-conjugation conventions from gate names.

For a custom fermionic basis transform, make the compiled one-particle action
match Agent4.1's mathematical matrix and mode-to-qubit ordering up to a common
global phase; make the inverse exact; and keep Jordan--Wigner/FSWAP signs. The
individually leaking parameter named by the error must remain a real variational
degree of freedom while its implemented generator preserves every declared
`exact_circuit` sector. Do not alter embedded authoritative contracts; they are
re-injected by the controller.
Hamiltonian coefficients may be exact rational strings such as `"-1/4"`;
parse them with `fractions.Fraction`, never `complex("a/b")`.
Preserve the three-argument
build_state_preparation(state_label, depth, reference_level=None) interface.
For a sector level k, reference_level=0,...,k must select orthogonal references
followed by one identical shared parameterized unitary; an unrequested lower
level remains internal and must not be added as an external state label.
"""
        system = (
            "You are repairing one concrete Qiskit translation error. Preserve "
            "the approved physics design and every unrelated implementation detail."
        )
        repaired = extract_single_python_block(
            self._call_model_once(system, prompt)
        )
        repaired = inject_authoritative_constants(
            repaired,
            {
                "PARAMETER_CONTRACT": parameter_contract,
                "STATE_PREPARATION_CONTRACT": state_contract,
            },
            insert_missing=True,
        )
        return validate_generated_adapter(repaired, MODE_STATE_PREPARATION)

    def translate_observable(
        self, mapping_contract, parameter_contract,
        observable_contract, observable_specification,
    ):
        qiskit_environment = qiskit_environment_summary()
        prompt = f"""
Return exactly one fenced Python block containing a thin model-specific Qiskit
observable adapter and no other text.

[LOCAL QISKIT ENVIRONMENT]
{qiskit_environment}

[AUTHORITATIVE AGENT 3 QUBIT-MAPPING CONTRACT]
```json
{format_machine_contract(mapping_contract, validate_mapping_contract, expected_parameter_contract=parameter_contract)}
```

[AUTHORITATIVE PARAMETER CONTRACT]
```json
{format_parameter_contract(parameter_contract)}
```

[AUTHORITATIVE OBSERVABLE CONTRACT]
```json
{format_observable_contract(observable_contract, parameter_contract, observable_specification)}
```

[AUTHORITATIVE USER OBSERVABLE SPECIFICATION]
```json
{json.dumps(observable_specification, ensure_ascii=False, indent=2, sort_keys=True)}
```

Translate literally. Embed the two JSON contracts as PARAMETER_CONTRACT and
OBSERVABLE_CONTRACT, and embed the immutable user input as
OBSERVABLE_SPECIFICATION. Define only these interfaces:

The supplied JSON objects are the only authoritative data model. Access only
top-level keys that actually occur in the displayed contracts; never invent a
shortcut/alias field such as `trajectory_table`. Resolve reusable definitions
by their existing `operator_recipes[].id` and executable measurements by their
existing `measurement.circuit_recipes[].recipe_id`, then convert the required
content into ordinary module constants. If a required design is not expressible
from those existing entries, fail clearly instead of assuming another field.
Do not force a synthesis result through a hand-selected basis_gates list unless
that exact conversion is supported by the displayed local Qiskit environment;
standard parameter-bound Qiskit gates may remain for downstream transpilation.
Every circuit returned by build_observable_measurements must be serializable by
the installed qiskit.qpy implementation. Do not leave an abstract synthesis
interface or a synthesis strategy with nonserializable settings attached to a
high-level gate. When such a strategy is required, call its concrete synthesize
method and compose the resulting QuantumCircuit before returning the recipe.

Preserve useful operator structure until downstream transpilation. When several
Qiskit implementations realize the same Agent-4B-designed unitary, choose the
structure-preserving implementation expected to use fewer logical two-qubit
gates. In particular:
- keep Pauli sums as SparsePauliOp or another locally verified structured
  representation and use a locally verified evolution-synthesis API when it
  implements the contract's declared product formula;
- preserve commuting groups, number/charge-preserving quadratic structure, and
  palindromic product-formula structure so Qiskit can synthesize and simplify
  them; expose adjacent rotations without barriers or unnecessary opaque gates;
- do not first convert a structured local generator to a dense matrix or a
  generic UnitaryGate. Matrix exponential, generic unitary synthesis, and QSD
  are last resorts when the contract requires an exact block exponential and no
  exact structure-preserving construction is available;
- rely on the downstream backend-aware transpiler for native-basis translation,
  layout, routing, inverse cancellation, and ordinary gate optimization. Do not
  duplicate a general compiler or encode assumptions about one backend here.

These are translation choices, not permission to redesign the algorithm. Keep
the contract's Hamiltonian terms, ordering, evolution time, product-formula
order and repetition count exactly. Never introduce an additional Trotter or
other approximation unless Agent 4.2 explicitly declares it. Never substitute a
default Qiskit synthesis whose approximation differs from the declared one.

1. configure_runtime(parameter_contract, observable_specification): validate
   that the supplied model, state labels and observable form match this
   generated program, then recompute every
   coupling-dependent coefficient, angle, normalization and reconstruction
   weight from the supplied selected parameter values. Numerical values such as
   masses, couplings, lattice spacing, theta and lattice size `L` must not be
   frozen from the generation run. Derive qubit count, site/link loops and Pauli
   labels from the current runtime values; do not reject a different valid `L`
   merely because the generating run used another size. Likewise, validate the
   sampling variable names and physical domain, but never require the exact old
   list of grid values. This function may update simple module constants and must be
   called once at import using the embedded PARAMETER_CONTRACT and
   OBSERVABLE_SPECIFICATION, so a deterministically rebound copy also works in
   measurement and hardware runtimes.
   Obtain Hamiltonian parameter dependence only from Agent3's explicit
   coefficient expressions. Never reverse-engineer a formula from numerical
   coefficients or silently carry numerical Hamiltonian values from another
   parameter point.
2. build_observable_measurements(state_circuits, trotter_steps, sample_values):
   accept a dict
   mapping every state label to a fully parameter-bound QuantumCircuit. Treat
   those circuits as immutable black boxes. Return a list of dict recipes with
   exactly these fields:
      recipe_id: unique nonempty string
      component: "real" or "imag"
      circuit: parameter-free QuantumCircuit with no measurements/classical bits
      readout_pauli: one Qiskit-order Pauli label string of circuit.num_qubits
   `readout_pauli` is the Pauli observable applied by the runtime Estimator after
   `circuit`. For a direct Pauli expectation, return the unrotated logical circuit
   and set `readout_pauli` to that Pauli string; do not append H or Sdg-H gates
   merely to convert X/Y readout to Z. Keep H, Sdg-H, controlled operations, or
   other gates that are intrinsic to a Hadamard test, SWAP test, or another
   interferometric algorithm, and in that case use the protocol's actual final
   Pauli readout (normally Z on the ancilla). Never represent the same readout
   basis change in both `circuit` and `readout_pauli`.
   Build the exact Agent-4B operator, evolution and interferometric circuit. The
   function chooses no ansatz depth and performs no simulation. Validate that
   trotter_steps is a positive integer when time_evolution.required is true. If
   time_evolution.required is false, accept the existing integer sentinel 0 and
   do not use it to alter any recipe. Validate that
   sample_values is one exact supported assignment and substitute it literally
   into insertion positions, transporter paths and evolution parameters. Preserve
   unitary logical circuits: never add measure, reset, initialize, classical
   bits, or classically controlled operations. Newly allocated Qiskit qubits
   already start in |0>, so do not reset them. After requiring
   `state_circuit.num_clbits == 0`, do not access the version-sensitive
   `instruction.operation.condition` attribute; ordinary gates such as X do not
   define it in supported Qiskit versions.
   Preserve
   Agent 4.2's declared imaginary-part sign convention exactly: for the standard
   (|0>|psi>+|1>V|psi>)/sqrt(2) branch convention, Sdg then H before Z readout
   returns +Im(<psi|V|psi>), while S then H returns its negative. Never label the
   latter as +Im unless reconstruct_observable compensates with an explicit minus.
3. reconstruct_observable(expectations, sample_values): accept a dict recipe_id
   -> real float and the same immutable sampling assignment,
   apply Agent 4.2's exact coefficients, normalization, signs and trajectory
   weights, including all observable.matrix_elements terms, and return their one
   final Python complex linear combination.

The adapter must contain no state-preparation implementation, optimizer, depth
scan, simulation, QPY export, shot allocation, run manifest, backend selection,
or IBM submission. It may compose/invert/control the supplied state circuits
only as required by Agent 4.2. Do not redesign the protocol or silently omit an
unsupported term; fail clearly if the design cannot be translated exactly.
Do not define validate_adapter_contract() and do not traverse the embedded JSON
contracts at runtime to re-check them. Convert Hamiltonian terms, trajectory
weights and other required design data once into simple executable module
constants used directly by the builder functions.
"""
        system = (
            "You are a literal Qiskit circuit translator. Implement the supplied "
            "observable protocol only and treat state preparation as a black box."
        )
        source = extract_single_python_block(self._call_model_once(system, prompt))
        source = inject_authoritative_constants(
            source,
            {
                "PARAMETER_CONTRACT": parameter_contract,
                "OBSERVABLE_CONTRACT": observable_contract,
                "OBSERVABLE_SPECIFICATION": observable_specification,
            },
        )
        source = ensure_runtime_configuration(source)
        return validate_generated_adapter(source, MODE_OBSERVABLE)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Translate one focused Agent-4 design.")
    parser.add_argument("--mode", choices=(MODE_STATE_PREPARATION, MODE_OBSERVABLE), required=True)
    parser.add_argument("--mapping-contract", required=True)
    parser.add_argument("--parameter-contract", required=True)
    parser.add_argument("--design-contract", required=True)
    parser.add_argument(
        "--runtime-settings",
        help="User-owned JSON containing vqe_runtime settings (state mode only).",
    )
    parser.add_argument(
        "--observable-spec",
        help="User observable specification (required in observable mode).",
    )
    parser.add_argument(
        "--reuse-adapter",
        help="Existing reusable adapter of the selected mode; skips the Agent 5 call.",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--output-manifest")
    parser.add_argument("--output-tag")
    parser.add_argument("--output-root")
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_tag = resolve_output_tag(args.output_tag)
    adapter_directory = categorized_output_directory(args.output_root, "adapters")
    runtime_directory = categorized_output_directory(args.output_root, "runtime")
    kind = MODE_STATE_PREPARATION if args.mode == MODE_STATE_PREPARATION else "observable"
    parameter_path, design_contract_path = args.parameter_contract, args.design_contract
    parameter_contract = load_parameter_contract(parameter_path)
    mapping_contract = load_machine_contract(
        args.mapping_contract,
        validate_mapping_contract,
        expected_parameter_contract=parameter_contract,
    )
    timestamp = output_tag
    api_key = api_key_for_model(args.model, required=False)
    if not args.reuse_adapter and not api_key:
        api_key_for_model(args.model)
    agent = (
        QiskitAdapterAgent(api_key=api_key, model_name=args.model)
        if not args.reuse_adapter else None
    )
    logger = setup_logger("Agent5_QiskitTranslator")

    if args.mode == MODE_STATE_PREPARATION:
        if not args.runtime_settings:
            raise SystemExit("State-preparation mode requires --runtime-settings.")
        state_contract = load_state_preparation_contract(
            design_contract_path, parameter_contract
        )
        if args.reuse_adapter:
            with open(args.reuse_adapter, "r", encoding="utf-8") as handle:
                source = handle.read()
            source = inject_authoritative_constants(
                source,
                {
                    "PARAMETER_CONTRACT": parameter_contract,
                    "STATE_PREPARATION_CONTRACT": state_contract,
                },
            )
            source = ensure_state_runtime_configuration(source)
            source = validate_generated_adapter(source, MODE_STATE_PREPARATION)
        else:
            source = agent.translate_state_preparation(
                mapping_contract, parameter_contract, state_contract
            )
        adapter_path = os.path.join(
            adapter_directory, "StatePreparationAdapter_{}.py".format(timestamp)
        )
        override = load_parameter_overrides(args.runtime_settings)
        validate_runtime_settings_identity(parameter_contract, override)
        run_config = build_vqe_run_config(
            state_contract, parameter_contract, override, timestamp,
            runtime_directory,
        )
        config_path = os.path.join(
            runtime_directory, "VQE_RunConfig_{}.json".format(timestamp)
        )
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(run_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        logger.info("VQE run config saved to %s", config_path)
    else:
        if not args.observable_spec:
            raise SystemExit("Observable mode requires --observable-spec.")
        observable_specification = load_observable_specification(args.observable_spec)
        observable_contract = load_observable_contract(
            design_contract_path, parameter_contract, observable_specification
        )
        if args.reuse_adapter:
            with open(args.reuse_adapter, "r", encoding="utf-8") as handle:
                source = handle.read()
            source = inject_authoritative_constants(
                source,
                {
                    "PARAMETER_CONTRACT": parameter_contract,
                    "OBSERVABLE_CONTRACT": observable_contract,
                    "OBSERVABLE_SPECIFICATION": observable_specification,
                },
            )
            source = ensure_runtime_configuration(source)
            source = validate_generated_adapter(source, MODE_OBSERVABLE)
        else:
            source = agent.translate_observable(
                mapping_contract, parameter_contract, observable_contract,
                observable_specification,
            )
        adapter_path = os.path.join(
            adapter_directory, "ObservableAdapter_{}.py".format(timestamp)
        )

    with open(adapter_path, "w", encoding="utf-8") as handle:
        handle.write(source)
        handle.write("\n")
    try:
        smoke_test_adapter(
            adapter_path, args.mode, mapping_contract, parameter_contract,
            observable_contract=(
                observable_contract if args.mode == MODE_OBSERVABLE else None
            ),
            observable_specification=(
                observable_specification if args.mode == MODE_OBSERVABLE else None
            ),
            vqe_config=(
                run_config if args.mode == MODE_STATE_PREPARATION else None
            ),
        )
    except Exception as error:
        if args.mode != MODE_STATE_PREPARATION:
            raise
        logger.warning(
            "State adapter failed its physical smoke test; requesting one "
            "targeted Agent5 translation repair: %s", error,
        )
        source = agent.repair_state_preparation(
            source,
            str(error),
            mapping_contract,
            parameter_contract,
            state_contract,
        )
        with open(adapter_path, "w", encoding="utf-8") as handle:
            handle.write(source)
            handle.write("\n")
        # A second failure is final for this invocation.  Never loop model
        # calls or weaken a physical requirement to force acceptance.
        smoke_test_adapter(
            adapter_path,
            args.mode,
            mapping_contract,
            parameter_contract,
            vqe_config=run_config,
        )
    logger.info("%s adapter saved to %s", args.mode, adapter_path)
    outputs = {"adapter": adapter_path}
    if args.mode == MODE_STATE_PREPARATION:
        outputs["vqe_config"] = config_path
    write_stage_output_manifest(
        args.output_manifest,
        "agent5_state" if args.mode == MODE_STATE_PREPARATION else "agent5_observable",
        outputs,
    )


if __name__ == "__main__":
    main()
