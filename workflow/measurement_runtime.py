# -*- coding: utf-8 -*-
"""Deterministically merge state preparation and observable measurement.

This module does not call a language model, optimize a state, choose a backend,
or submit a job.  It evaluates the full logical ``(D_ans, N_T)`` grid on an
exact statevector and exports the same parameter-bound recipe circuits as QPY.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

from circuit_design_support import (
    contract_sha256,
    validate_observable_contract,
    validate_state_preparation_contract,
)
from parameter_contract import parameter_contract_digest, validate_parameter_contract
from observable_specification import (
    expand_observable_samples,
    load_observable_specification,
    observable_specification_digest,
    resolve_observable_sampling,
    validate_observable_specification,
)
from vqe_experience_store import DEFAULT_DATABASE_PATH, record_measurement_result


MEASUREMENT_SCHEMA_VERSION = "1.0"


class MeasurementRuntimeError(RuntimeError):
    """Raised when adapters, contracts, or measurement recipes disagree."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementRuntimeError(message)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_path(module: Any) -> str:
    path = inspect.getsourcefile(module)
    _require(path is not None and os.path.isfile(path), "Cannot locate adapter source.")
    return os.path.abspath(path)


def _module_hash(module: Any) -> str:
    return _sha256_file(_module_path(module))


def _atomic_json_write(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".measurement_runtime_", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _load_module(path: str, module_name: str) -> Any:
    absolute = os.path.abspath(path)
    _require(os.path.isfile(absolute), "Adapter file does not exist: {}".format(path))
    spec = importlib.util.spec_from_file_location(module_name, absolute)
    _require(spec is not None and spec.loader is not None,
             "Cannot import adapter: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_contracts(
    state_adapter: Any, observable_adapter: Any,
    observable_specification: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for module, names in (
        (state_adapter, ("PARAMETER_CONTRACT", "STATE_PREPARATION_CONTRACT")),
        (
            observable_adapter,
            ("PARAMETER_CONTRACT", "OBSERVABLE_CONTRACT", "OBSERVABLE_SPECIFICATION"),
        ),
    ):
        for name in names:
            _require(hasattr(module, name), "Adapter is missing {}.".format(name))

    state_parameter = state_adapter.PARAMETER_CONTRACT
    observable_parameter = observable_adapter.PARAMETER_CONTRACT
    validate_parameter_contract(state_parameter)
    validate_parameter_contract(observable_parameter)
    # Agent2 is the sole parameter authority. Agent5 injects the same document
    # into both adapters, so measurement uses the state adapter's copy without
    # creating another compatibility protocol.
    state_contract = validate_state_preparation_contract(
        state_adapter.STATE_PREPARATION_CONTRACT, state_parameter
    )
    observable_contract = validate_observable_contract(
        observable_adapter.OBSERVABLE_CONTRACT, state_parameter,
        observable_specification,
    )
    embedded_observable = validate_observable_specification(
        observable_adapter.OBSERVABLE_SPECIFICATION
    )
    _require(
        embedded_observable == observable_specification,
        "Observable adapter embeds a different observable specification.",
    )
    _require(
        state_contract["state_labels"] == observable_contract["state_labels"],
        "State and observable contracts disagree on state-label order.",
    )
    required_operations = " ".join(
        observable_contract["state_preparation_interface"]["required_operations"]
    ).lower()
    if "inverse" in required_operations:
        _require(state_contract["adapter_interface"]["supports_inverse"],
                 "Observable protocol requires an unsupported inverse state circuit.")
    if "control" in required_operations or "multiplex" in required_operations:
        _require(state_contract["adapter_interface"]["supports_controlled"],
                 "Observable protocol requires an unsupported controlled state circuit.")
    return state_parameter, state_contract, observable_contract


def _build_bound_states(
    state_adapter: Any,
    depth_entry: Mapping[str, Any],
    state_labels: Sequence[str],
) -> Dict[str, QuantumCircuit]:
    depth = depth_entry.get("depth")
    states = depth_entry.get("states")
    _require(isinstance(depth, int) and depth > 0, "VQE depth entry is invalid.")
    _require(isinstance(states, dict) and set(states) == set(state_labels),
             "VQE depth entry does not contain every state label exactly once.")
    bound: Dict[str, QuantumCircuit] = {}
    qubit_count = None
    for label in state_labels:
        value = state_adapter.build_state_preparation(label, depth)
        _require(isinstance(value, tuple) and len(value) == 2,
                 "build_state_preparation() returned the wrong interface.")
        circuit, ordered_parameters = value
        ordered = list(ordered_parameters)
        _require(isinstance(circuit, QuantumCircuit),
                 "State-preparation builder must return QuantumCircuit.")
        _require(len(ordered) == len(set(ordered)) and set(ordered) == set(circuit.parameters),
                 "State-preparation parameter order is inconsistent with the circuit.")
        parameters = states[label].get("parameters")
        _require(isinstance(parameters, list) and len(parameters) == len(ordered),
                 "Saved VQE parameter vector has the wrong length.")
        _require(all(isinstance(value, (int, float)) and math.isfinite(value)
                     for value in parameters),
                 "Saved VQE parameters must be finite numbers.")
        binding = {parameter: float(value) for parameter, value in zip(ordered, parameters)}
        prepared = circuit.assign_parameters(binding, inplace=False)
        _require(not prepared.parameters, "State-preparation circuit is not fully bound.")
        _require(prepared.num_clbits == 0, "State-preparation circuit must be unitary.")
        _require(
            all(
                item.operation.name not in {"measure", "reset", "initialize"}
                and not item.clbits
                and getattr(item.operation, "condition", None) is None
                for item in prepared.data
            ),
            "State-preparation circuit contains a nonunitary operation.",
        )
        if qubit_count is None:
            qubit_count = prepared.num_qubits
        _require(prepared.num_qubits == qubit_count,
                 "External-state circuits use different qubit counts.")
        bound[label] = prepared
    return bound


def _validate_recipes(
    recipes: Any,
    observable_contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    _require(isinstance(recipes, list) and recipes,
             "Observable adapter must return a nonempty recipe list.")
    expected_ids = observable_contract["measurement"]["circuit_recipe_ids"]
    allowed_components = set(observable_contract["measurement"]["components"])
    validated = []
    seen = set()
    for recipe in recipes:
        _require(isinstance(recipe, dict), "Every measurement recipe must be a dict.")
        _require(set(recipe) == {"recipe_id", "component", "circuit", "readout_pauli"},
                 "Measurement recipes contain missing or extra fields.")
        recipe_id = recipe["recipe_id"]
        component = recipe["component"]
        circuit = recipe["circuit"]
        pauli = recipe["readout_pauli"]
        _require(isinstance(recipe_id, str) and recipe_id and recipe_id not in seen,
                 "recipe_id values must be unique nonempty strings.")
        seen.add(recipe_id)
        _require(component in allowed_components,
                 "Recipe component is absent from the Observable Contract.")
        _require(isinstance(circuit, QuantumCircuit),
                 "Recipe circuit must be a QuantumCircuit.")
        _require(not circuit.parameters, "Recipe circuit must be parameter-free.")
        _require(circuit.num_clbits == 0, "Logical recipes must contain no classical bits.")
        _require(all(item.operation.name != "measure" for item in circuit.data),
                 "Logical recipes must contain no measurements.")
        _require(
            all(
                item.operation.name not in {"measure", "reset", "initialize"}
                and getattr(item.operation, "condition", None) is None
                for item in circuit.data
            ),
            "Logical measurement recipe contains a nonunitary operation.",
        )
        _require(
            isinstance(pauli, str)
            and len(pauli) == circuit.num_qubits
            and set(pauli).issubset(set("IXYZ")),
            "readout_pauli must be a Qiskit-order I/X/Y/Z label of circuit width.",
        )
        validated.append(dict(recipe))
    _require(set(seen).issubset(set(expected_ids)),
             "Observable adapter returned a recipe ID absent from the contract.")
    return validated


def _resource_metrics(circuit: QuantumCircuit) -> Dict[str, int]:
    # Agent-5 adapters deliberately package state preparation and Trotter
    # evolution as composite gates.  Counting only ``circuit.data`` therefore
    # measures wrapper boxes instead of the logical circuit.  Expand onto a
    # fixed, all-to-all ideal basis before counting.  Backend connectivity and
    # native-ISA costs remain the responsibility of hardware_runtime.py.
    expanded = transpile(
        circuit,
        basis_gates=["u", "cx"],
        optimization_level=0,
    )
    two_qubit = 0
    multi_qubit = 0
    for instruction in expanded.data:
        width = len(instruction.qubits)
        if width == 2:
            two_qubit += 1
        elif width > 2:
            multi_qubit += 1
    return {
        "num_qubits": expanded.num_qubits,
        "logical_depth": expanded.depth(),
        "logical_two_qubit_gates": two_qubit,
        "logical_gates_wider_than_two_qubits": multi_qubit,
        "logical_operation_count": len(expanded.data),
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "recipe"


def _write_qpy(path: str, circuit: QuantumCircuit) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        qpy.dump(circuit, handle)
    return _sha256_file(path)


def _trotter_schedule(observable_contract: Mapping[str, Any]) -> List[int]:
    evolution = observable_contract["time_evolution"]
    if evolution["required"]:
        return list(evolution["recommended_test_steps"])
    return [0]


def run_measurements(
    state_adapter: Any,
    observable_adapter: Any,
    vqe_result: Mapping[str, Any],
    output_path: str,
    circuit_directory: str,
    *,
    observable_specification: Mapping[str, Any],
    vqe_result_path: str | None = None,
    database_path: str | None = None,
) -> Dict[str, Any]:
    """Evaluate and export every logical measurement grid point."""
    observable_specification = validate_observable_specification(
        observable_specification
    )
    parameter_contract, state_contract, observable_contract = _load_contracts(
        state_adapter, observable_adapter, observable_specification
    )
    execution_specification = resolve_observable_sampling(
        observable_specification, parameter_contract
    )
    samples = expand_observable_samples(execution_specification)
    state_adapter_hash = _module_hash(state_adapter)
    observable_adapter_hash = _module_hash(observable_adapter)
    tested_depths = vqe_result.get("tested_depths")
    _require(isinstance(tested_depths, list) and tested_depths,
             "VQE result contains no tested depths.")
    state_labels = state_contract["state_labels"]
    trotter_steps = _trotter_schedule(observable_contract)
    grid = []

    for depth_entry in tested_depths:
        depth = depth_entry["depth"]
        bound_states = _build_bound_states(
            state_adapter, depth_entry, state_labels
        )
        for steps in trotter_steps:
            for sample in samples:
                sample_id = sample["sample_id"]
                sample_values = dict(sample["values"])
                snapshots = {label: circuit.copy() for label, circuit in bound_states.items()}
                recipes = observable_adapter.build_observable_measurements(
                    dict(bound_states), steps, dict(sample_values)
                )
                _require(all(bound_states[label] == snapshots[label] for label in state_labels),
                         "Observable adapter modified a supplied state circuit in place.")
                recipes = _validate_recipes(recipes, observable_contract)
                expectations: Dict[str, float] = {}
                exported = []
                for recipe in recipes:
                    statevector = Statevector.from_instruction(recipe["circuit"])
                    operator = SparsePauliOp.from_list([(recipe["readout_pauli"], 1.0)])
                    expectation = complex(statevector.expectation_value(operator))
                    _require(abs(expectation.imag) <= 1.0e-9,
                             "Hermitian Pauli expectation has a non-negligible imaginary part.")
                    real_expectation = float(expectation.real)
                    expectations[recipe["recipe_id"]] = real_expectation
                    point_directory = os.path.join(
                        circuit_directory,
                        _safe_name(sample_id),
                        "D{}".format(depth),
                        "NT{}".format(steps),
                    )
                    filename = "{}.qpy".format(_safe_name(recipe["recipe_id"]))
                    qpy_path = os.path.join(point_directory, filename)
                    qpy_hash = _write_qpy(qpy_path, recipe["circuit"])
                    exported.append({
                        "recipe_id": recipe["recipe_id"],
                        "component": recipe["component"],
                        "readout_pauli": recipe["readout_pauli"],
                        "noiseless_expectation": real_expectation,
                        "qpy_path": os.path.abspath(qpy_path),
                        "qpy_sha256": qpy_hash,
                        "resources": _resource_metrics(recipe["circuit"]),
                    })

                observable = observable_adapter.reconstruct_observable(
                    dict(expectations), dict(sample_values)
                )
                _require(isinstance(observable, (int, float, complex)),
                         "reconstruct_observable() must return a number.")
                observable = complex(observable)
                _require(math.isfinite(observable.real) and math.isfinite(observable.imag),
                         "Reconstructed observable is not finite.")
                state_quality = {
                    label: {
                        key: depth_entry["states"][label].get(key)
                        for key in ("status", "energy", "energy_error", "variance")
                    }
                    for label in state_labels
                }
                all_states_converged = all(
                    quality["status"] == "converged_state"
                    for quality in state_quality.values()
                )
                grid.append({
                    "candidate_id": "{}_D{}_NT{}".format(
                        _safe_name(sample_id), depth, steps
                    ),
                    "sample_id": sample_id,
                    "sample_values": sample_values,
                    "D_ans": depth,
                    "N_T": steps,
                    "state_preparation_validated": all_states_converged,
                    "state_quality": state_quality,
                    "observable_real": float(observable.real),
                    "observable_imag": float(observable.imag),
                    "recipes": exported,
                })

    result = {
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "status": "noiseless_validation_complete",
        "parameter_contract_sha256": parameter_contract_digest(parameter_contract),
        "observable_specification_sha256": observable_specification_digest(
            observable_specification
        ),
        "sampling_points": samples,
        "state_preparation_contract_sha256": contract_sha256(
            state_contract, validate_state_preparation_contract, parameter_contract
        ),
        "observable_contract_sha256": contract_sha256(
            observable_contract, validate_observable_contract, parameter_contract
        ),
        "state_adapter_sha256": state_adapter_hash,
        "state_adapter_path": _module_path(state_adapter),
        "observable_adapter_sha256": observable_adapter_hash,
        "observable_adapter_path": _module_path(observable_adapter),
        "vqe_result_path": os.path.abspath(vqe_result_path) if vqe_result_path else None,
        "vqe_result_sha256": (
            _sha256_file(vqe_result_path) if vqe_result_path is not None else None
        ),
        "depth_schedule": [entry["depth"] for entry in tested_depths],
        "trotter_schedule": trotter_steps,
        "validation_grid": grid,
        "hardware_eligible_candidates": [],
        "hardware_eligibility_status": "pending_agent8_crosscheck",
    }
    if database_path is not None:
        identities = record_measurement_result(
            database_path,
            state_adapter,
            observable_adapter,
            vqe_result,
            result,
        )
        result.update(identities)
    _atomic_json_write(output_path, result)
    if database_path is not None:
        record_measurement_result(
            database_path,
            state_adapter,
            observable_adapter,
            vqe_result,
            result,
            result_path=output_path,
        )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the logical (D_ans, N_T) measurement grid."
    )
    parser.add_argument("--state-adapter", required=True)
    parser.add_argument("--observable-adapter", required=True)
    parser.add_argument("--vqe-result", required=True)
    parser.add_argument("--observable-spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--circuit-directory", required=True)
    parser.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    state_adapter = _load_module(args.state_adapter, "state_preparation_adapter")
    observable_adapter = _load_module(args.observable_adapter, "observable_adapter")
    with open(args.vqe_result, "r", encoding="utf-8") as handle:
        vqe_result = json.load(handle)
    observable_specification = load_observable_specification(args.observable_spec)
    result = run_measurements(
        state_adapter,
        observable_adapter,
        vqe_result,
        args.output,
        args.circuit_directory,
        observable_specification=observable_specification,
        vqe_result_path=args.vqe_result,
        database_path=args.database,
    )
    print(
        "Measurement grid completed: {} point(s), campaign {}.".format(
            len(result["validation_grid"]), result["campaign_id"]
        )
    )


if __name__ == "__main__":
    main()
