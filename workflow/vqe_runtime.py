# -*- coding: utf-8 -*-
"""Deterministic VQE execution engine used by thin Agent-5 adapters.

Agent 5 owns only model-specific Qiskit construction.  This module owns the
optimizer, reverse-gradient binding order, depth scan, warm start, convergence
test, and atomic checkpoint. Ground states use ordinary VQE; multiple energy
levels in one explicitly defined quantum-number sector use weighted SSVQE.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.gradients import ReverseEstimatorGradient


RUNTIME_SCHEMA_VERSION = "1.0"
PARAMETER_BUNDLE_SCHEMA_VERSION = "1.0"
ORTHOGONALITY_TOLERANCE = 1.0e-4
# The equal-weight stage only locates the low-energy subspace.  Most of the
# budget must remain for the ordered stage that assigns each reference to its
# level without using any exact target energy in the loss.
SSVQE_SUBSPACE_BUDGET_FRACTION = 0.3


class VQERuntimeError(RuntimeError):
    """Raised when an adapter, configuration, or numerical check is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VQERuntimeError(message)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".vqe_runtime_", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def default_parameter_bundle_path(result_path: str) -> str:
    """Return the compact reusable-parameter path beside a full VQE result."""
    directory = os.path.dirname(result_path)
    filename = os.path.basename(result_path)
    if "Runtime_Result" in filename:
        filename = filename.replace("Runtime_Result", "ParameterBundle", 1)
    else:
        stem, _ = os.path.splitext(filename)
        filename = stem + "_ParameterBundle.json"
    return os.path.join(directory, filename)


def build_parameter_bundle(
    adapter: Any,
    result: Mapping[str, Any],
    *,
    source_result_path: str | None = None,
) -> Dict[str, Any]:
    """Extract only the selected state parameters needed to rebuild circuits."""
    selected = result.get("selected_states")
    _require(isinstance(selected, dict) and selected,
             "VQE result has no selected_states to export.")
    adapter_digest = _adapter_hash(adapter)

    states: Dict[str, Dict[str, Any]] = {}
    preparations: Dict[str, Dict[str, Any]] = {}
    preparation_ids: Dict[Tuple[int, Tuple[float, ...]], str] = {}
    for label in sorted(selected):
        state = selected[label]
        _require(isinstance(state, dict),
                 "Selected state {} must be an object.".format(label))
        depth = state.get("depth")
        values = state.get("parameters")
        _require(isinstance(depth, int) and not isinstance(depth, bool) and depth > 0,
                 "Selected state {} has an invalid depth.".format(label))
        _require(
            isinstance(values, list)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in values
            ),
            "Selected state {} has invalid parameter values.".format(label),
        )
        circuit, ordered_parameters = adapter.build_state_preparation(label, depth)
        _require(isinstance(circuit, QuantumCircuit),
                 "Adapter did not return a QuantumCircuit for {}.".format(label))
        _require(len(ordered_parameters) == len(values),
                 "Selected parameters do not match adapter order for {}.".format(label))
        numeric_values = tuple(float(value) for value in values)
        preparation_key = (depth, numeric_values)
        preparation_id = preparation_ids.get(preparation_key)
        if preparation_id is None:
            preparation_id = "preparation_{}".format(len(preparations) + 1)
            preparation_ids[preparation_key] = preparation_id
            preparations[preparation_id] = {
                "depth": depth,
                "parameter_binding": "ordered_parameters_returned_by_adapter",
                "parameter_count": len(values),
                "parameter_values": list(numeric_values),
            }
        states[label] = {
            "preparation_id": preparation_id,
            "status": state.get("status"),
            "target_spec": state.get("target_spec"),
            "quality": {
                key: state.get(key)
                for key in (
                    "energy", "energy_error", "relative_energy_error",
                    "variance", "sector_weight",
                )
            },
        }

    source = None
    if source_result_path is not None:
        absolute = os.path.abspath(source_result_path)
        _require(os.path.isfile(absolute),
                 "Source VQE result does not exist: {}".format(absolute))
        source = {"path": absolute, "sha256": _sha256_file(absolute)}
    return {
        "parameter_bundle_schema_version": PARAMETER_BUNDLE_SCHEMA_VERSION,
        "adapter_sha256": adapter_digest,
        "source_vqe_result": source,
        "selection_policy": "vqe_result.selected_states",
        "preparations": preparations,
        "states": states,
    }


def write_parameter_bundle(
    adapter: Any,
    result: Mapping[str, Any],
    output_path: str,
    *,
    source_result_path: str | None = None,
) -> Dict[str, Any]:
    bundle = build_parameter_bundle(
        adapter, result, source_result_path=source_result_path
    )
    _atomic_json_write(output_path, bundle)
    return bundle


def bind_state_from_parameter_bundle(
    adapter: Any, bundle: Mapping[str, Any], label: str
) -> QuantumCircuit:
    """Rebuild one selected, parameter-free state circuit from a compact bundle."""
    _require(
        bundle.get("parameter_bundle_schema_version")
        == PARAMETER_BUNDLE_SCHEMA_VERSION,
        "Unsupported VQE parameter-bundle schema.",
    )
    states = bundle.get("states")
    preparations = bundle.get("preparations")
    _require(isinstance(states, dict) and label in states,
             "Parameter bundle has no state label {}.".format(label))
    _require(isinstance(preparations, dict),
             "Parameter bundle preparations must be an object.")
    preparation_id = states[label].get("preparation_id")
    _require(preparation_id in preparations,
             "State {} refers to a missing preparation.".format(label))
    preparation = preparations[preparation_id]
    depth = preparation.get("depth")
    values = preparation.get("parameter_values")
    _require(isinstance(depth, int) and not isinstance(depth, bool) and depth > 0,
             "Bundle preparation depth must be a positive integer.")
    _require(isinstance(values, list),
             "Bundle parameter_values must be a list.")
    circuit, ordered_parameters = adapter.build_state_preparation(label, depth)
    _require(len(ordered_parameters) == len(values),
             "Bundle parameter count does not match adapter for {}.".format(label))
    bound = circuit.assign_parameters(
        dict(zip(ordered_parameters, values)), inplace=False
    )
    _require(not bound.parameters,
             "Bundle failed to bind every parameter for {}.".format(label))
    return bound


def validate_run_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the deliberately small, model-independent runtime config."""
    validated = dict(config)
    required = (
        "state_labels",
        "equivalent_state_groups",
        "target_kind_by_state",
        "depth_schedule",
        "nested_by_depth",
        "scan_all_depths",
        "restart_count",
        "max_iterations",
        "first_depth_lower_bound",
        "first_depth_upper_bound",
        "new_layer_value",
        "perturbation_scale",
        "variance_tolerance",
        "energy_tolerance",
        "energy_tie_tolerance",
        "gradient_check_step",
        "gradient_check_tolerance",
        "checkpoint_file",
        "result_file",
        "random_seed",
    )
    missing = [key for key in required if key not in config]
    _require(not missing, "VQE run config is missing: {}".format(", ".join(missing)))

    labels = config["state_labels"]
    _require(
        isinstance(labels, list)
        and labels
        and all(isinstance(label, str) and label for label in labels)
        and len(labels) == len(set(labels)),
        "state_labels must contain unique nonempty strings.",
    )

    targets = config["target_kind_by_state"]
    _require(isinstance(targets, dict) and set(targets) == set(labels),
             "target_kind_by_state must define every state label exactly once.")
    unsupported = [
        label for label, kind in targets.items()
        if kind not in {"ground", "sector_eigenstate"}
    ]
    _require(
        not unsupported,
        "Unsupported target kind for: {}".format(", ".join(unsupported)),
    )
    supplied_specs = config.get("target_spec_by_state", {})
    _require(isinstance(supplied_specs, dict),
             "target_spec_by_state must be an object when supplied.")
    target_specs: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        kind = targets[label]
        raw = supplied_specs.get(label, {})
        _require(isinstance(raw, dict),
                 "target_spec_by_state.{} must be an object.".format(label))
        _require(
            raw.get("kind", kind) == kind,
            "target kind and target spec disagree for {}.".format(label),
        )
        level_index = raw.get("level_index", 0)
        quantum_numbers = raw.get("quantum_numbers", {})
        _require(
            isinstance(level_index, int)
            and not isinstance(level_index, bool)
            and level_index >= 0,
            "target level_index for {} must be a nonnegative integer.".format(label),
        )
        _require(isinstance(quantum_numbers, dict),
                 "target quantum_numbers for {} must be an object.".format(label))
        if kind == "sector_eigenstate":
            _require(
                bool(quantum_numbers),
                "sector_eigenstate {} requires explicit quantum_numbers.".format(label),
            )
        spec = {
            "kind": kind,
            "level_index": level_index,
            "quantum_numbers": dict(quantum_numbers),
        }
        if isinstance(raw.get("name"), str) and raw["name"].strip():
            spec["name"] = raw["name"].strip()
        target_specs[label] = spec
    validated["target_spec_by_state"] = target_specs

    groups = config["equivalent_state_groups"]
    _require(isinstance(groups, list), "equivalent_state_groups must be a list.")
    grouped: List[str] = []
    for group in groups:
        _require(
            isinstance(group, list)
            and len(group) >= 2
            and len(group) == len(set(group))
            and all(label in labels for label in group),
            "Every equivalent-state group must contain distinct known labels.",
        )
        grouped.extend(group)
    _require(len(grouped) == len(set(grouped)),
             "A state label cannot occur in two equivalent-state groups.")
    for group in groups:
        reference = target_specs[group[0]]
        _require(
            all(target_specs[label] == reference for label in group[1:]),
            "Equivalent states must have identical target specifications.",
        )

    adaptive = config.get("adaptive_depth", {"enabled": False})
    _require(isinstance(adaptive, dict), "adaptive_depth must be an object.")
    _require(isinstance(adaptive.get("enabled", False), bool),
             "adaptive_depth.enabled must be boolean.")
    if adaptive.get("enabled", False):
        for key in ("plateau_window", "minimum_relative_improvement"):
            _require(key in adaptive,
                     "adaptive_depth is missing {}.".format(key))
        explicit_range = {
            key for key in ("start_depth", "depth_increment", "maximum_depth")
            if key in adaptive
        }
        _require(
            not explicit_range or explicit_range == {
                "start_depth", "depth_increment", "maximum_depth"
            },
            "adaptive_depth must provide all or none of start_depth, "
            "depth_increment, and maximum_depth.",
        )
        for key in explicit_range:
            value = adaptive[key]
            _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                     "adaptive_depth.{} must be a positive integer.".format(key))
        if explicit_range:
            _require(adaptive["start_depth"] <= adaptive["maximum_depth"],
                     "adaptive start_depth exceeds maximum_depth.")
        window = adaptive["plateau_window"]
        _require(isinstance(window, int) and not isinstance(window, bool) and window >= 2,
                 "adaptive_depth.plateau_window must be an integer >= 2.")
        improvement = adaptive["minimum_relative_improvement"]
        _require(
            isinstance(improvement, (int, float))
            and not isinstance(improvement, bool)
            and math.isfinite(improvement)
            and 0.0 <= improvement < 1.0,
            "adaptive minimum_relative_improvement must be in [0, 1).",
        )
        _require(config["nested_by_depth"] is True,
                 "Adaptive depth requires an ansatz nested_by_depth.")
        if explicit_range:
            schedule = list(range(
                adaptive["start_depth"],
                adaptive["maximum_depth"] + 1,
                adaptive["depth_increment"],
            ))
            validated["depth_schedule"] = schedule
        else:
            # Keep Agent4.1's nonuniform test schedule and add one bounded
            # warm-start depth.  This gives an improving family more budget
            # without creating an unbounded VQE loop or another controller.
            schedule = list(config["depth_schedule"])
            if schedule:
                schedule.append(2 * schedule[-1])
            validated["depth_schedule"] = schedule
    else:
        schedule = config["depth_schedule"]
    _require(
        isinstance(schedule, list)
        and schedule
        and all(isinstance(depth, int) and not isinstance(depth, bool) and depth > 0
                for depth in schedule)
        and schedule == sorted(set(schedule)),
        "depth_schedule must contain increasing unique positive integers.",
    )
    for key in ("nested_by_depth", "scan_all_depths"):
        _require(
            isinstance(config[key], bool),
            "{} must be boolean.".format(key),
        )
    if "stop_restarts_on_convergence" in config:
        _require(
            isinstance(config["stop_restarts_on_convergence"], bool),
            "stop_restarts_on_convergence must be boolean.",
        )
    max_same_depth_continuations = config.get("max_same_depth_continuations", 1)
    _require(
        isinstance(max_same_depth_continuations, int)
        and not isinstance(max_same_depth_continuations, bool)
        and max_same_depth_continuations >= 0,
        "max_same_depth_continuations must be a nonnegative integer.",
    )
    validated["max_same_depth_continuations"] = max_same_depth_continuations
    if "record_experience" in config:
        _require(
            isinstance(config["record_experience"], bool),
            "record_experience must be boolean.",
        )
    if config.get("record_experience", False):
        _require(
            isinstance(config.get("experience_database"), str)
            and bool(config["experience_database"]),
            "experience_database must be a nonempty path when recording is enabled.",
        )

    for key in ("restart_count", "max_iterations", "random_seed"):
        value = config[key]
        minimum = 0 if key == "random_seed" else 1
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
                 "{} must be an integer >= {}.".format(key, minimum))

    numeric = (
        "first_depth_lower_bound",
        "first_depth_upper_bound",
        "new_layer_value",
        "perturbation_scale",
        "variance_tolerance",
        "energy_tolerance",
        "energy_tie_tolerance",
        "gradient_check_step",
        "gradient_check_tolerance",
    )
    for key in numeric:
        value = config[key]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                 "{} must be numeric.".format(key))
    _require(config["first_depth_lower_bound"] < config["first_depth_upper_bound"],
             "The first-depth initialization interval is empty.")
    for key in (
        "perturbation_scale",
        "variance_tolerance",
        "energy_tolerance",
        "energy_tie_tolerance",
        "gradient_check_step",
        "gradient_check_tolerance",
    ):
        _require(config[key] > 0, "{} must be positive.".format(key))
    for key in ("checkpoint_file", "result_file"):
        _require(isinstance(config[key], str) and config[key],
                 "{} must be a nonempty path.".format(key))
    parameter_bundle_file = config.get(
        "parameter_bundle_file", default_parameter_bundle_path(config["result_file"])
    )
    _require(isinstance(parameter_bundle_file, str) and parameter_bundle_file,
             "parameter_bundle_file must be a nonempty path.")
    validated["parameter_bundle_file"] = parameter_bundle_file
    refinement_seeds = config.get("initial_parameters_by_depth", {})
    _require(isinstance(refinement_seeds, dict),
             "initial_parameters_by_depth must be an object when supplied.")
    for depth_key, by_state in refinement_seeds.items():
        _require(isinstance(depth_key, str) and depth_key.isdigit(),
                 "Refinement depth keys must be positive-integer strings.")
        _require(int(depth_key) in schedule,
                 "A refinement depth is absent from depth_schedule.")
        _require(isinstance(by_state, dict) and by_state,
                 "Every refinement depth must contain state parameters.")
        for label, values in by_state.items():
            _require(label in labels,
                     "Refinement parameters use an unknown state label.")
            _require(
                isinstance(values, list)
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in values
                ),
                "Refinement parameters must be a finite numeric list.",
            )
    return validated


def _adapter_path(adapter: Any) -> str:
    path = inspect.getsourcefile(adapter)
    _require(path is not None and os.path.isfile(path),
             "Cannot determine the Agent-5 adapter source file.")
    return os.path.abspath(path)


def _adapter_hash(adapter: Any) -> str:
    return _sha256_file(_adapter_path(adapter))


def _build_hamiltonian(adapter: Any) -> SparsePauliOp:
    _require(hasattr(adapter, "build_hamiltonian"),
             "Agent-5 adapter must define build_hamiltonian().")
    hamiltonian = adapter.build_hamiltonian()
    _require(isinstance(hamiltonian, SparsePauliOp),
             "build_hamiltonian() must return SparsePauliOp.")
    return hamiltonian.simplify(atol=1.0e-14)


def _build_state_preparation(
    adapter: Any, label: str, depth: int, reference_level: Optional[int] = None
) -> Tuple[QuantumCircuit, List[Parameter]]:
    _require(hasattr(adapter, "build_state_preparation"),
             "Adapter must define build_state_preparation(label, depth).")
    if reference_level is None:
        value = adapter.build_state_preparation(label, depth)
    else:
        try:
            value = adapter.build_state_preparation(
                label, depth, reference_level=reference_level
            )
        except TypeError as error:
            raise VQERuntimeError(
                "Excited-state SSVQE needs the Agent-5 adapter interface "
                "build_state_preparation(label, depth, reference_level=None) "
                "so missing lower levels can be internal reference states."
            ) from error
    _require(isinstance(value, tuple) and len(value) == 2,
             "build_state_preparation() must return (QuantumCircuit, ordered_parameters).")
    circuit, ordered_parameters = value
    ordered = list(ordered_parameters)
    _require(isinstance(circuit, QuantumCircuit),
             "build_state_preparation() did not return a QuantumCircuit.")
    _require(len(ordered) == len(set(ordered)),
             "The ordered logical parameter list contains duplicates.")
    _require(set(ordered) == set(circuit.parameters),
             "The ordered logical parameters must equal circuit.parameters as a set.")
    forbidden = {"measure", "reset", "initialize"}
    _require(
        all(
            item.operation.name not in forbidden
            and not item.clbits
            and getattr(item.operation, "condition", None) is None
            for item in circuit.data
        ),
        "State-preparation circuit contains a nonunitary or classically controlled operation.",
    )
    return circuit, ordered


def _physical_basis(adapter: Any, dimension: int, state_label: Optional[str] = None) -> np.ndarray:
    if hasattr(adapter, "physical_basis_vectors"):
        function = adapter.physical_basis_vectors
        parameters = inspect.signature(function).parameters
        vectors = np.asarray(
            function(state_label) if parameters else function(), dtype=complex
        )
        _require(
            vectors.ndim == 2 and vectors.shape[0] == dimension
            and vectors.shape[1] > 0,
            "physical_basis_vectors() returned an empty or invalid basis.",
        )
        _require(
            np.allclose(
                vectors.conj().T @ vectors,
                np.eye(vectors.shape[1]), atol=1.0e-8,
            ),
            "physical basis vectors must be orthonormal.",
        )
        return vectors
    if not hasattr(adapter, "physical_basis_indices"):
        return np.arange(dimension, dtype=int)
    function = adapter.physical_basis_indices
    parameters = inspect.signature(function).parameters
    indices = np.asarray(
        function(state_label) if parameters else function(),
        dtype=int,
    )
    _require(indices.ndim == 1 and indices.size > 0,
             "physical_basis_indices() returned an empty or invalid basis.")
    _require(np.all(indices >= 0) and np.all(indices < dimension),
             "physical basis contains an out-of-range index.")
    _require(len(indices) == len(np.unique(indices)),
             "physical basis indices must be unique.")
    return indices


def _exact_ground_energy(matrix: np.ndarray, basis: np.ndarray) -> float:
    restricted = (
        matrix[np.ix_(basis, basis)] if basis.ndim == 1
        else basis.conj().T @ matrix @ basis
    )
    return float(np.real(np.linalg.eigvalsh(restricted)[0]))


def _exact_sector_energy(
    matrix: np.ndarray, basis: np.ndarray, level_index: int
) -> float:
    restricted = (
        matrix[np.ix_(basis, basis)] if basis.ndim == 1
        else basis.conj().T @ matrix @ basis
    )
    eigenvalues = np.linalg.eigvalsh(restricted)
    _require(
        level_index < len(eigenvalues),
        "Requested sector level {} but the selected sector has dimension {}.".format(
            level_index, len(eigenvalues)
        ),
    )
    return float(np.real(eigenvalues[level_index]))


def _state_vector(
    circuit: QuantumCircuit,
    ordered_parameters: Sequence[Parameter],
    values: Sequence[float],
) -> np.ndarray:
    binding = {
        parameter: float(value)
        for parameter, value in zip(ordered_parameters, values)
    }
    return np.asarray(
        Statevector.from_instruction(
            circuit.assign_parameters(binding, inplace=False)
        ).data,
        dtype=complex,
    )


def _orthogonality_overlap(
    state: np.ndarray, lower_states: Sequence[np.ndarray]
) -> float:
    return float(sum(abs(np.vdot(lower, state)) ** 2 for lower in lower_states))


def _state_energy_variance(
    circuit: QuantumCircuit,
    ordered_parameters: Sequence[Parameter],
    values: Sequence[float],
    matrix: np.ndarray,
) -> Tuple[float, float, float]:
    values_array = np.asarray(values, dtype=float)
    _require(values_array.shape == (len(ordered_parameters),),
             "Variational parameter vector has the wrong shape.")
    binding = {
        parameter: float(value)
        for parameter, value in zip(ordered_parameters, values_array)
    }
    bound = circuit.assign_parameters(binding, inplace=False)
    state = np.asarray(Statevector.from_instruction(bound).data, dtype=complex)
    h_state = matrix @ state
    energy = float(np.real(np.vdot(state, h_state)))
    raw_variance = float(np.real(np.vdot(h_state, h_state) - energy * energy))
    variance = 0.0 if -1.0e-10 < raw_variance < 0.0 else raw_variance
    return energy, raw_variance, variance


def _state_sector_weight(
    circuit: QuantumCircuit,
    ordered_parameters: Sequence[Parameter],
    values: Sequence[float],
    basis: np.ndarray,
) -> float:
    state = _state_vector(circuit, ordered_parameters, values)
    if basis.ndim == 1:
        return float(np.real(np.sum(np.abs(state[basis]) ** 2)))
    return float(np.real(np.sum(np.abs(basis.conj().T @ state) ** 2)))


def reverse_gradient(
    engine: ReverseEstimatorGradient,
    circuit: QuantumCircuit,
    ordered_parameters: Sequence[Parameter],
    logical_values: Sequence[float],
    hamiltonian: SparsePauliOp,
) -> np.ndarray:
    """Evaluate one reverse gradient with explicit Qiskit binding order."""
    logical_values_array = np.asarray(logical_values, dtype=float)
    _require(logical_values_array.shape == (len(ordered_parameters),),
             "Gradient input has the wrong logical-vector shape.")
    value_by_parameter = {
        parameter: float(value)
        for parameter, value in zip(ordered_parameters, logical_values_array)
    }
    # Agent 5 owns gate-level translation.  The runtime uses the adapter circuit
    # directly and must not silently expand it again inside every optimizer step.
    gradient_parameter_order = list(circuit.parameters)
    parameter_values = [value_by_parameter[p] for p in gradient_parameter_order]
    result = engine.run(
        [circuit],
        [hamiltonian],
        [parameter_values],
        parameters=[gradient_parameter_order],
    ).result()
    circuit_order_gradient = np.asarray(
        np.real(result.gradients[0]), dtype=float
    )
    gradient_by_parameter = dict(zip(
        gradient_parameter_order, circuit_order_gradient
    ))
    gradient = np.asarray(
        [gradient_by_parameter[p] for p in ordered_parameters], dtype=float
    )
    _require(gradient.shape == logical_values_array.shape,
             "Reverse gradient returned the wrong shape.")
    return gradient


def validate_reverse_gradient(
    engine: ReverseEstimatorGradient,
    circuit: QuantumCircuit,
    ordered_parameters: Sequence[Parameter],
    test_values: Sequence[float],
    hamiltonian: SparsePauliOp,
    matrix: np.ndarray,
    step: float,
    tolerance: float,
) -> float:
    """Compare every logical reverse-gradient component to central differences."""
    values = np.asarray(test_values, dtype=float)
    reverse = reverse_gradient(
        engine, circuit, ordered_parameters, values, hamiltonian
    )
    finite = np.empty_like(reverse)
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        plus_energy = _state_energy_variance(
            circuit, ordered_parameters, plus, matrix
        )[0]
        minus_energy = _state_energy_variance(
            circuit, ordered_parameters, minus, matrix
        )[0]
        finite[index] = (plus_energy - minus_energy) / (2.0 * step)
    maximum_error = float(np.max(np.abs(reverse - finite)))
    _require(
        np.allclose(reverse, finite, atol=tolerance, rtol=tolerance),
        "Reverse-gradient self-test failed: max error {:.6e}.".format(maximum_error),
    )
    return maximum_error


def _parameters_per_layer(parameter_count: int, depth: int) -> int:
    _require(parameter_count % depth == 0,
             "Ansatz parameter count is not divisible by its depth.")
    return parameter_count // depth


def make_initial_values(
    *,
    config: Mapping[str, Any],
    depth: int,
    parameter_count: int,
    restart_index: int,
    seed: int,
    previous_depth: Optional[int],
    previous_values: Optional[Sequence[float]],
) -> np.ndarray:
    """Construct a reproducible first-depth or arbitrary-jump warm start."""
    rng = np.random.default_rng(seed)
    if previous_values is not None and not config["nested_by_depth"]:
        previous_values = None
        previous_depth = None
    if previous_values is None:
        _require(previous_depth is None, "previous_depth was supplied without values.")
        return rng.uniform(
            config["first_depth_lower_bound"],
            config["first_depth_upper_bound"],
            size=parameter_count,
        )

    _require(previous_depth is not None and previous_depth < depth,
             "Warm start requires a smaller completed depth.")
    previous = np.asarray(previous_values, dtype=float)
    target_per_layer = _parameters_per_layer(parameter_count, depth)
    _require(previous.size == previous_depth * target_per_layer,
             "Warm-start vector does not contain exactly the completed layers.")
    initial = np.full(
        parameter_count, float(config["new_layer_value"]), dtype=float
    )
    initial[:previous.size] = previous
    if restart_index > 0:
        initial += rng.uniform(
            -config["perturbation_scale"],
            config["perturbation_scale"],
            size=parameter_count,
        )
    return initial


def make_refinement_initial_values(
    *,
    saved_values: Sequence[float],
    parameter_count: int,
    restart_index: int,
    seed: int,
    perturbation_scale: float,
) -> np.ndarray:
    """Continue at one depth from saved parameters, with optional restarts."""
    initial = np.asarray(saved_values, dtype=float).copy()
    _require(initial.shape == (parameter_count,),
             "Saved refinement vector has the wrong parameter count.")
    _require(np.all(np.isfinite(initial)),
             "Saved refinement vector contains a non-finite value.")
    if restart_index > 0:
        rng = np.random.default_rng(seed)
        initial += rng.uniform(
            -perturbation_scale,
            perturbation_scale,
            size=parameter_count,
        )
    return initial


def _run_candidate(
    *,
    config: Mapping[str, Any],
    circuit: QuantumCircuit,
    ordered_parameters: Sequence[Parameter],
    initial_values: np.ndarray,
    objective_hamiltonian: SparsePauliOp,
    objective_matrix: np.ndarray,
    physical_matrix: np.ndarray,
    target_basis: np.ndarray,
    lower_states: Sequence[np.ndarray],
    gradient_engine: ReverseEstimatorGradient,
    restart_index: int,
    seed: int,
) -> Dict[str, Any]:
    def objective(values: np.ndarray) -> float:
        return _state_energy_variance(
            circuit, ordered_parameters, values, objective_matrix
        )[0]

    def jacobian(values: np.ndarray) -> np.ndarray:
        return reverse_gradient(
            gradient_engine,
            circuit,
            ordered_parameters,
            values,
            objective_hamiltonian,
        )

    initial_objective = objective(initial_values)
    initial_energy, initial_raw_variance, initial_variance = _state_energy_variance(
        circuit, ordered_parameters, initial_values, physical_matrix
    )
    initial_sector_weight = _state_sector_weight(
        circuit, ordered_parameters, initial_values, target_basis
    )
    initial_overlap = _orthogonality_overlap(
        _state_vector(circuit, ordered_parameters, initial_values), lower_states
    )
    if not ordered_parameters:
        energy, raw_variance, variance = _state_energy_variance(
            circuit, ordered_parameters, initial_values, physical_matrix
        )
        return {
            "restart_index": restart_index,
            "seed": seed,
            "initial_energy": float(initial_energy),
            "initial_objective_value": float(initial_objective),
            "objective_value": float(initial_objective),
            "sector_weight": initial_sector_weight,
            "orthogonality_overlap": initial_overlap,
            "energy": energy,
            "raw_variance": raw_variance,
            "variance": variance,
            "parameters": [],
            "optimizer_success": True,
            "optimizer_status": 0,
            "optimizer_message": "No variational parameters.",
            "function_evaluations": 1,
            "gradient_evaluations": 0,
        }
    result = minimize(
        objective,
        initial_values,
        method="BFGS",
        jac=jacobian,
        options={"maxiter": config["max_iterations"], "gtol": 1.0e-8},
    )
    optimized_energy, optimized_raw_variance, optimized_variance = _state_energy_variance(
        circuit, ordered_parameters, result.x, physical_matrix
    )
    optimized_objective = objective(result.x)
    optimized_sector_weight = _state_sector_weight(
        circuit, ordered_parameters, result.x, target_basis
    )
    optimized_overlap = _orthogonality_overlap(
        _state_vector(circuit, ordered_parameters, result.x), lower_states
    )
    retain_initial = (
        optimized_objective > initial_objective + config["energy_tie_tolerance"]
        or (
            optimized_objective <= initial_objective + config["energy_tie_tolerance"]
            and initial_objective <= optimized_objective + config["energy_tie_tolerance"]
            and initial_variance < optimized_variance
        )
    )
    if retain_initial:
        selected_values = np.asarray(initial_values, dtype=float)
        energy = initial_energy
        raw_variance = initial_raw_variance
        variance = initial_variance
        objective_value = initial_objective
        sector_weight = initial_sector_weight
        orthogonality_overlap = initial_overlap
    else:
        selected_values = np.asarray(result.x, dtype=float)
        energy = optimized_energy
        raw_variance = optimized_raw_variance
        variance = optimized_variance
        objective_value = optimized_objective
        sector_weight = optimized_sector_weight
        orthogonality_overlap = optimized_overlap
    return {
        "restart_index": restart_index,
        "seed": seed,
        "initial_energy": float(initial_energy),
        "initial_objective_value": float(initial_objective),
        "objective_value": float(objective_value),
        "sector_weight": float(sector_weight),
        "orthogonality_overlap": float(orthogonality_overlap),
        "energy": energy,
        "raw_variance": raw_variance,
        "variance": variance,
        "parameters": selected_values.tolist(),
        "retained_initial_point": bool(retain_initial),
        "optimized_energy": optimized_energy,
        "optimized_raw_variance": optimized_raw_variance,
        "optimized_variance": optimized_variance,
        "optimized_objective_value": float(optimized_objective),
        "optimized_sector_weight": float(optimized_sector_weight),
        "optimized_orthogonality_overlap": float(optimized_overlap),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
    }


def _ssvqe_groups(
    representatives: Sequence[str], config: Mapping[str, Any]
) -> List[List[str]]:
    """Group requested states that must share one weighted-SSVQE unitary."""
    groups: List[List[str]] = []
    used = set()
    for representative in representatives:
        if representative in used:
            continue
        spec = config["target_spec_by_state"][representative]
        if spec["kind"] != "sector_eigenstate":
            groups.append([representative])
            used.add(representative)
            continue
        sector = json.dumps(
            spec["quantum_numbers"], sort_keys=True, separators=(",", ":")
        )
        members = [
            label for label in representatives
            if config["target_spec_by_state"][label]["kind"] == "sector_eigenstate"
            and json.dumps(
                config["target_spec_by_state"][label]["quantum_numbers"],
                sort_keys=True,
                separators=(",", ":"),
            ) == sector
        ]
        if all(
            config["target_spec_by_state"][label]["level_index"] == 0
            for label in members
        ):
            groups.extend([[label] for label in members])
            used.update(members)
            continue
        by_level: Dict[int, str] = {}
        for label in members:
            level = config["target_spec_by_state"][label]["level_index"]
            _require(
                level not in by_level,
                "Weighted SSVQE requires one independent state label per "
                "level in a quantum-number sector.",
            )
            by_level[level] = label
        group = [by_level[level] for level in sorted(by_level)]
        groups.append(group)
        used.update(group)
    return groups


def _expand_ssvqe_groups(
    groups: Sequence[Sequence[str]],
    config: Mapping[str, Any],
    adapter: Any,
    matrix: np.ndarray,
    basis_by_state: Mapping[str, np.ndarray],
    target_energy_by_state: Mapping[str, float],
) -> Tuple[
    List[List[str]],
    Dict[str, Dict[str, Any]],
    Dict[str, np.ndarray],
    Dict[str, float],
    Dict[str, Tuple[str, Optional[int]]],
]:
    """Add unrequested lower levels as private SSVQE reference states."""
    expanded_specs = {
        label: dict(spec)
        for label, spec in config["target_spec_by_state"].items()
    }
    expanded_basis = dict(basis_by_state)
    expanded_targets = dict(target_energy_by_state)
    builder_requests: Dict[str, Tuple[str, Optional[int]]] = {
        label: (label, None) for group in groups for label in group
    }
    occupied = set(expanded_specs)
    expanded_groups: List[List[str]] = []

    for requested_group in groups:
        group = list(requested_group)
        first_spec = expanded_specs[group[0]]
        if first_spec["kind"] != "sector_eigenstate":
            expanded_groups.append(group)
            continue

        by_level = {
            expanded_specs[label]["level_index"]: label for label in group
        }
        maximum_level = max(by_level)
        if maximum_level == 0:
            expanded_groups.append(group)
            continue

        # The highest requested state owns the shared ansatz family.  Missing
        # lower levels use only a different orthogonal reference preparation.
        anchor = by_level[maximum_level]
        full_group: List[str] = []
        for level in range(maximum_level + 1):
            if level in by_level:
                full_group.append(by_level[level])
                continue
            auxiliary = "__ssvqe_aux__{}__level_{}".format(anchor, level)
            suffix = 2
            while auxiliary in occupied:
                auxiliary = "__ssvqe_aux__{}__level_{}_{}".format(
                    anchor, level, suffix
                )
                suffix += 1
            occupied.add(auxiliary)
            spec = dict(first_spec)
            spec["level_index"] = level
            expanded_specs[auxiliary] = spec
            expanded_basis[auxiliary] = _physical_basis(
                adapter, matrix.shape[0], anchor
            )
            expanded_targets[auxiliary] = _exact_sector_energy(
                matrix, expanded_basis[auxiliary], level
            )
            builder_requests[auxiliary] = (anchor, level)
            full_group.append(auxiliary)
        expanded_groups.append(full_group)

    return (
        expanded_groups,
        expanded_specs,
        expanded_basis,
        expanded_targets,
        builder_requests,
    )


def _ssvqe_weights(group: Sequence[str]) -> Dict[str, float]:
    """Return distinctly decreasing, target-energy-independent level weights."""
    # Groups are ordered by increasing level index.  The integer ladder gives
    # 2:1 for two states and n:(n-1):...:1 for a general n-level request.
    raw = np.arange(len(group), 0, -1, dtype=float)
    raw /= float(np.sum(raw))
    return {label: float(raw[index]) for index, label in enumerate(group)}


def _equal_ssvqe_weights(group: Sequence[str]) -> Dict[str, float]:
    """Return equal weights for low-energy-subspace optimization."""
    weight = 1.0 / float(len(group))
    return {label: weight for label in group}


def _ssvqe_state_metrics(
    circuits: Mapping[str, Tuple[QuantumCircuit, Sequence[Parameter]]],
    values: Sequence[float],
    physical_matrix: np.ndarray,
    basis_by_state: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    vectors = {
        label: _state_vector(circuit, ordered, values)
        for label, (circuit, ordered) in circuits.items()
    }
    metrics: Dict[str, Dict[str, float]] = {}
    for label, (circuit, ordered) in circuits.items():
        energy, raw_variance, variance = _state_energy_variance(
            circuit, ordered, values, physical_matrix
        )
        overlap = sum(
            abs(np.vdot(other, vectors[label])) ** 2
            for other_label, other in vectors.items()
            if other_label != label
        )
        metrics[label] = {
            "energy": energy,
            "raw_variance": raw_variance,
            "variance": variance,
            "sector_weight": _state_sector_weight(
                circuit, ordered, values, basis_by_state[label]
            ),
            "orthogonality_overlap": float(overlap),
        }
    return metrics


def _run_weighted_ssvqe_candidate(
    *,
    config: Mapping[str, Any],
    circuits: Mapping[str, Tuple[QuantumCircuit, Sequence[Parameter]]],
    weights: Mapping[str, float],
    initial_values: np.ndarray,
    hamiltonian: SparsePauliOp,
    physical_matrix: np.ndarray,
    basis_by_state: Mapping[str, np.ndarray],
    gradient_engine: ReverseEstimatorGradient,
    restart_index: int,
    seed: int,
) -> Dict[str, Any]:
    """Optimize the low-energy subspace, then resolve its ordered levels."""
    parameter_count = len(next(iter(circuits.values()))[1])
    equal_weights = _equal_ssvqe_weights(list(circuits))

    def objective(values: np.ndarray, stage_weights: Mapping[str, float]) -> float:
        metrics = _ssvqe_state_metrics(circuits, values, physical_matrix, basis_by_state)
        return float(sum(
            stage_weights[label] * metrics[label]["energy"] for label in circuits
        ))

    def jacobian(
        values: np.ndarray, stage_weights: Mapping[str, float]
    ) -> np.ndarray:
        total = np.zeros(parameter_count, dtype=float)
        for label, (circuit, ordered) in circuits.items():
            total += stage_weights[label] * reverse_gradient(
                gradient_engine, circuit, ordered, values, hamiltonian
            )
        return total

    def optimize_stage(
        start: np.ndarray,
        stage_weights: Mapping[str, float],
        max_iterations: int,
    ) -> Tuple[np.ndarray, Any, float, bool]:
        start_objective = objective(start, stage_weights)
        if not parameter_count or max_iterations <= 0:
            return np.asarray(start, dtype=float), None, start_objective, False
        result = minimize(
            lambda values: objective(values, stage_weights),
            start,
            method="BFGS",
            jac=lambda values: jacobian(values, stage_weights),
            options={"maxiter": max_iterations, "gtol": 1.0e-8},
        )
        optimized = np.asarray(result.x, dtype=float)
        optimized_objective = objective(optimized, stage_weights)
        retain_start = optimized_objective > (
            start_objective + config["energy_tie_tolerance"]
        )
        return (
            np.asarray(start if retain_start else optimized, dtype=float),
            result,
            float(start_objective if retain_start else optimized_objective),
            bool(retain_start),
        )

    initial_objective = objective(initial_values, weights)
    initial_metrics = _ssvqe_state_metrics(
        circuits, initial_values, physical_matrix, basis_by_state
    )
    if parameter_count:
        total_budget = int(config["max_iterations"])
        if len(circuits) > 1 and total_budget > 1:
            subspace_budget = max(
                1, int(math.floor(total_budget * SSVQE_SUBSPACE_BUDGET_FRACTION))
            )
            resolution_budget = total_budget - subspace_budget
        else:
            subspace_budget = 0
            resolution_budget = total_budget
        subspace_values, subspace_result, subspace_objective, retained_before = (
            optimize_stage(
                np.asarray(initial_values, dtype=float),
                equal_weights,
                subspace_budget,
            )
        )
        selected_values, result, optimized_objective, retain_subspace = optimize_stage(
            subspace_values,
            weights,
            resolution_budget,
        )
        selected_metrics = _ssvqe_state_metrics(
            circuits, selected_values, physical_matrix, basis_by_state
        )
        retain_initial = bool(retained_before and retain_subspace)
    else:
        subspace_budget = 0
        resolution_budget = 0
        subspace_result = None
        subspace_objective = objective(initial_values, equal_weights)
        result = None
        optimized_objective = initial_objective
        selected_values = np.asarray(initial_values, dtype=float)
        selected_metrics = initial_metrics
        retain_initial = False
    stage_results = [item for item in (subspace_result, result) if item is not None]
    return {
        "restart_index": restart_index,
        "seed": seed,
        "initial_objective_value": float(initial_objective),
        "objective_value": float(optimized_objective),
        "subspace_objective_value": float(subspace_objective),
        "subspace_weights": dict(equal_weights),
        "subspace_iteration_budget": int(subspace_budget),
        "level_resolution_iteration_budget": int(resolution_budget),
        "parameters": np.asarray(selected_values, dtype=float).tolist(),
        "state_metrics": selected_metrics,
        "variance": max(item["variance"] for item in selected_metrics.values()),
        "energy": selected_metrics[next(iter(circuits))]["energy"],
        "sector_weight": min(
            item["sector_weight"] for item in selected_metrics.values()
        ),
        "orthogonality_overlap": max(
            item["orthogonality_overlap"] for item in selected_metrics.values()
        ),
        "retained_initial_point": bool(retain_initial),
        "optimizer_success": True if result is None else bool(result.success),
        "optimizer_status": 0 if result is None else int(result.status),
        "optimizer_message": (
            "No variational parameters." if result is None else str(result.message)
        ),
        "function_evaluations": (
            1 if not stage_results else sum(int(item.nfev) for item in stage_results)
        ),
        "gradient_evaluations": sum(int(item.njev) for item in stage_results),
        "subspace_optimizer_success": (
            True if subspace_result is None else bool(subspace_result.success)
        ),
        "subspace_optimizer_status": (
            0 if subspace_result is None else int(subspace_result.status)
        ),
    }


def _validate_weighted_ssvqe_gradient(
    *,
    circuits: Mapping[str, Tuple[QuantumCircuit, Sequence[Parameter]]],
    weights: Mapping[str, float],
    values: np.ndarray,
    hamiltonian: SparsePauliOp,
    physical_matrix: np.ndarray,
    gradient_engine: ReverseEstimatorGradient,
    step: float,
    tolerance: float,
) -> float:
    reverse = np.zeros(values.size, dtype=float)
    for label, (circuit, ordered) in circuits.items():
        reverse += weights[label] * reverse_gradient(
            gradient_engine, circuit, ordered, values, hamiltonian
        )

    def objective(point: np.ndarray) -> float:
        return float(sum(
            weights[label] * _state_energy_variance(
                circuit, ordered, point, physical_matrix
            )[0]
            for label, (circuit, ordered) in circuits.items()
        ))

    finite = np.empty_like(reverse)
    for index in range(values.size):
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        finite[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    maximum_error = float(np.max(np.abs(reverse - finite)))
    _require(
        np.allclose(reverse, finite, atol=tolerance, rtol=tolerance),
        "Weighted-SSVQE gradient self-test failed: max error {:.6e}.".format(
            maximum_error
        ),
    )
    return maximum_error


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]], energy_tie_tolerance: float
) -> Dict[str, Any]:
    _require(bool(candidates), "No VQE candidate was produced.")
    minimum_energy = min(
        float(item.get("objective_value", item["energy"])) for item in candidates
    )
    tied = [
        item for item in candidates
        if float(item.get("objective_value", item["energy"]))
        <= minimum_energy + energy_tie_tolerance
    ]
    selected = min(
        tied,
        key=lambda item: (
            float(item["variance"]),
            float(item.get("objective_value", item["energy"])),
            int(item["restart_index"]),
        ),
    )
    return dict(selected)


def _circuit_signature(circuit: QuantumCircuit) -> List[Tuple[Any, ...]]:
    signature = []
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        clbits = tuple(circuit.find_bit(bit).index for bit in instruction.clbits)
        signature.append((operation.name, qubits, clbits, len(operation.params)))
    return signature


def _representatives(
    labels: Sequence[str], groups: Sequence[Sequence[str]]
) -> Tuple[List[str], Dict[str, str]]:
    representative_by_label = {label: label for label in labels}
    for group in groups:
        representative = group[0]
        for label in group:
            representative_by_label[label] = representative
    representatives = []
    for label in labels:
        representative = representative_by_label[label]
        if representative not in representatives:
            representatives.append(representative)
    return representatives, representative_by_label


def _checkpoint_header(adapter: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    del adapter
    return {
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "depth_schedule": config["depth_schedule"],
        "state_labels": config["state_labels"],
    }


def _load_checkpoint(
    adapter: Any,
    config: Mapping[str, Any],
    matrix: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    path = config["checkpoint_file"]
    if not os.path.exists(path):
        return [], None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_header = _checkpoint_header(adapter, config)
    stored_schedule = payload.get("depth_schedule")
    expected_schedule = expected_header["depth_schedule"]
    _require(
        isinstance(stored_schedule, list)
        and stored_schedule == expected_schedule[:len(stored_schedule)],
        "Checkpoint depth_schedule is not a prefix of the current schedule.",
    )
    for key, expected in expected_header.items():
        if key == "depth_schedule":
            continue
        _require(payload.get(key) == expected,
                 "Checkpoint is incompatible at field {}.".format(key))
    completed = payload.get("completed_depths")
    _require(isinstance(completed, list), "Checkpoint completed_depths is invalid.")
    stored_depths = [entry.get("depth") for entry in completed]
    _require(stored_depths == config["depth_schedule"][:len(stored_depths)],
             "Checkpoint depths are not an ordered schedule prefix.")
    for entry in completed:
        depth = entry["depth"]
        for label, state in entry["states"].items():
            circuit, ordered = _build_state_preparation(adapter, label, depth)
            energy, raw_variance, variance = _state_energy_variance(
                circuit, ordered, state["parameters"], matrix
            )
            _require(np.isclose(energy, state["energy"], atol=1.0e-9, rtol=1.0e-9),
                     "Checkpoint energy does not reproduce.")
            _require(np.isclose(raw_variance, state["raw_variance"], atol=1.0e-9,
                                rtol=1.0e-8),
                     "Checkpoint raw variance does not reproduce.")
            _require(np.isclose(variance, state["variance"], atol=1.0e-9,
                                rtol=1.0e-8),
                     "Checkpoint variance does not reproduce.")
            # Status is derived data.  Recompute it so checkpoints written by
            # older runtimes cannot preserve a weaker excited-state gate.
            validation_energy = state.get("target_energy")
            if validation_energy is not None:
                state["energy_error"] = abs(energy - validation_energy)
                state["relative_energy_error"] = _relative_energy_error(
                    energy, validation_energy
                )
            state["status"] = (
                "converged_state"
                if _candidate_meets_thresholds(
                    state, validation_energy, config
                )
                else "best_effort_state"
            )
        entry["status"] = (
            "converged_state"
            if all(
                state["status"] == "converged_state"
                for state in entry["states"].values()
            )
            else "best_effort_state"
        )
    in_progress = payload.get("in_progress_depth")
    if in_progress is not None:
        _require(isinstance(in_progress, dict),
                 "Checkpoint in_progress_depth must be an object or null.")
        next_index = len(completed)
        _require(next_index < len(config["depth_schedule"]),
                 "Checkpoint has in-progress data after the schedule completed.")
        _require(in_progress.get("depth") == config["depth_schedule"][next_index],
                 "Checkpoint in-progress depth is not the next scheduled depth.")
        candidates = in_progress.get("candidates_by_representative")
        _require(isinstance(candidates, dict),
                 "Checkpoint in-progress candidates must be an object.")
    return completed, in_progress


def _write_checkpoint(
    adapter: Any,
    config: Mapping[str, Any],
    completed_depths: Sequence[Mapping[str, Any]],
    in_progress_depth: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = _checkpoint_header(adapter, config)
    payload["completed_depths"] = list(completed_depths)
    payload["in_progress_depth"] = (
        dict(in_progress_depth) if in_progress_depth is not None else None
    )
    _atomic_json_write(config["checkpoint_file"], payload)


def _candidate_meets_thresholds(
    candidate: Mapping[str, Any],
    validation_energy: Optional[float],
    config: Mapping[str, Any],
) -> bool:
    return (
        float(candidate["variance"]) < config["variance_tolerance"]
        and float(candidate.get("sector_weight", 1.0)) >= 1.0 - 1.0e-8
        and float(candidate.get("orthogonality_overlap", 0.0))
        < ORTHOGONALITY_TOLERANCE
        and (
            validation_energy is None
            or _relative_energy_error(
                float(candidate["energy"]), validation_energy
            ) < config["energy_tolerance"]
        )
    )


def _candidate_residual(
    candidate: Mapping[str, Any],
    validation_energy: Optional[float],
    config: Mapping[str, Any],
) -> float:
    """Return the normalized energy/variance/orthogonality residual."""
    residuals = [
        float(candidate["variance"]) / config["variance_tolerance"],
        float(candidate.get("orthogonality_overlap", 0.0))
        / ORTHOGONALITY_TOLERANCE,
    ]
    if validation_energy is not None:
        residuals.append(
            _relative_energy_error(float(candidate["energy"]), validation_energy)
            / config["energy_tolerance"]
        )
    return max(residuals)


def _optimizer_reached_iteration_limit(candidate: Mapping[str, Any]) -> bool:
    """Recognize SciPy's bounded-iteration stop without relying on one message."""
    if bool(candidate.get("optimizer_success", False)):
        return False
    message = str(candidate.get("optimizer_message", "")).lower()
    return int(candidate.get("optimizer_status", -1)) == 1 or (
        "maximum" in message and "iteration" in message
    )


def _relative_energy_error(energy: float, target_energy: float) -> float:
    """Use a stable relative error, with unit scale when the target is near zero."""
    return abs(energy - target_energy) / max(abs(target_energy), 1.0)


def _depth_residual(entry: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    """Return the convergence residual used for each requested state."""
    residuals = []
    for state in entry["states"].values():
        values = [
            float(state["variance"]) / config["variance_tolerance"],
            float(state.get("orthogonality_overlap", 0.0))
            / ORTHOGONALITY_TOLERANCE,
        ]
        values.append(
            float(state["relative_energy_error"])
            / config["energy_tolerance"]
        )
        residuals.append(max(values))
    return max(residuals)


def _depth_trend_summary(
    completed: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the compact trend Agent4.1 needs, without the full VQE payload."""
    history = [
        {"depth": entry["depth"], "residual": _depth_residual(entry, config)}
        for entry in completed
    ]
    if len(history) < 2:
        improvement = None
    else:
        first = history[0]["residual"]
        last = history[-1]["residual"]
        improvement = (first - last) / max(abs(first), 1.0e-300)
    return {
        "classification": (
            "converged"
            if any(entry["status"] == "converged_state" for entry in completed)
            else "stalled"
            if _adaptive_plateau_diagnostic(completed, config)["plateau"]
            else "promising"
        ),
        "residual_history": history,
        "overall_relative_improvement": improvement,
    }


def _adaptive_plateau_diagnostic(
    completed: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    adaptive = config.get("adaptive_depth", {})
    if not adaptive.get("enabled", False):
        return {"applicable": False, "plateau": False, "recent": []}
    window = adaptive["plateau_window"]
    recent_entries = list(completed[-window:])
    recent = [
        {"depth": entry["depth"], "residual": _depth_residual(entry, config)}
        for entry in recent_entries
    ]
    if len(recent) < window:
        return {"applicable": True, "plateau": False, "recent": recent}
    first = recent[0]["residual"]
    last = recent[-1]["residual"]
    relative_improvement = (first - last) / max(abs(first), 1.0e-300)
    plateau = (
        last > 1.0
        and relative_improvement
        < adaptive["minimum_relative_improvement"]
    )
    return {
        "applicable": True,
        "plateau": plateau,
        "recent": recent,
        "relative_improvement": relative_improvement,
        "minimum_relative_improvement": adaptive[
            "minimum_relative_improvement"
        ],
    }


def _ssvqe_candidate_meets_thresholds(
    candidate: Mapping[str, Any],
    group: Sequence[str],
    target_energy_by_state: Mapping[str, float],
    config: Mapping[str, Any],
) -> bool:
    return all(
        _candidate_meets_thresholds(
            candidate["state_metrics"][label],
            target_energy_by_state[label],
            config,
        )
        for label in group
    )


def _ssvqe_candidate_residual(
    candidate: Mapping[str, Any],
    group: Sequence[str],
    target_energy_by_state: Mapping[str, float],
    config: Mapping[str, Any],
) -> float:
    return max(
        _candidate_residual(
            candidate["state_metrics"][label],
            target_energy_by_state[label],
            config,
        )
        for label in group
    )


def _run_weighted_ssvqe(
    *,
    adapter: Any,
    config: Mapping[str, Any],
    hamiltonian: SparsePauliOp,
    matrix: np.ndarray,
    labels: Sequence[str],
    representatives: Sequence[str],
    representative_by_label: Mapping[str, str],
    basis_by_state: Mapping[str, np.ndarray],
    target_energy_by_state: Mapping[str, float],
) -> Dict[str, Any]:
    """Run weighted SSVQE for requested excited levels and ordinary VQE groups."""
    public_config = config
    groups = _ssvqe_groups(representatives, config)
    (
        groups,
        expanded_specs,
        basis_by_state,
        target_energy_by_state,
        builder_requests,
    ) = _expand_ssvqe_groups(
        groups,
        config,
        adapter,
        matrix,
        basis_by_state,
        target_energy_by_state,
    )
    config = dict(config)
    config["target_spec_by_state"] = expanded_specs
    requested_labels = set(labels)

    def requested_leader(group: Sequence[str]) -> str:
        return next(label for label in group if label in requested_labels)

    gradient_engine = ReverseEstimatorGradient()
    completed, in_progress = _load_checkpoint(adapter, config, matrix)
    compatible = all(
        state.get("optimization_objective") in {"energy", "weighted_ssvqe"}
        for entry in completed
        for state in entry.get("states", {}).values()
    )
    # Do not mix checkpoints optimized with the former near-equal ordering
    # weights into the new distinctly ordered SSVQE policy.
    for entry in completed:
        for state in entry.get("states", {}).values():
            stored_weights = state.get("ssvqe_weights")
            if not stored_weights or len(stored_weights) <= 1:
                continue
            ordered_weight_labels = sorted(
                stored_weights,
                key=lambda label: config["target_spec_by_state"][label][
                    "level_index"
                ],
            )
            expected_weights = _ssvqe_weights(ordered_weight_labels)
            if set(stored_weights) != set(expected_weights) or any(
                not np.isclose(
                    float(stored_weights[label]),
                    float(expected_weights[label]),
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )
                for label in expected_weights
            ):
                compatible = False
                break
        if not compatible:
            break
    if not compatible or (
        in_progress is not None
        and in_progress.get("optimization_method") != "weighted_ssvqe"
    ):
        completed, in_progress = [], None

    previous_depth: Optional[int] = None
    previous_by_group: Dict[str, List[float]] = {}
    if completed:
        previous_depth = completed[-1]["depth"]
        for group in groups:
            leader = requested_leader(group)
            previous_by_group[leader] = list(
                completed[-1]["states"][leader]["parameters"]
            )

    adaptive_enabled = config.get("adaptive_depth", {}).get("enabled", False)
    already_converged = any(entry["status"] == "converged_state" for entry in completed)
    plateau_diagnostic = _adaptive_plateau_diagnostic(completed, config)
    termination_reason: Optional[str] = None
    if already_converged and adaptive_enabled:
        should_continue = False
        termination_reason = "accuracy_threshold_reached"
    elif plateau_diagnostic["plateau"]:
        should_continue = False
        termination_reason = "accuracy_plateau"
    else:
        should_continue = config["scan_all_depths"] or not already_converged

    if should_continue:
        for depth in config["depth_schedule"][len(completed):]:
            candidates_by_group: Dict[str, List[Dict[str, Any]]] = {}
            continuations_by_group: Dict[str, List[Dict[str, Any]]] = {}
            if in_progress is not None:
                _require(in_progress["depth"] == depth,
                         "Loaded weighted-SSVQE checkpoint has the wrong depth.")
                candidates_by_group = {
                    label: [dict(item) for item in items]
                    for label, items in in_progress.get(
                        "candidates_by_representative", {}
                    ).items()
                }
                continuations_by_group = {
                    label: [dict(item) for item in items]
                    for label, items in in_progress.get(
                        "continuations_by_representative", {}
                    ).items()
                }

            representative_results: Dict[str, Dict[str, Any]] = {}
            for group in groups:
                leader = requested_leader(group)
                circuits = {}
                for label in group:
                    builder_label, reference_level = builder_requests[label]
                    circuits[label] = _build_state_preparation(
                        adapter,
                        builder_label,
                        depth,
                        reference_level=reference_level,
                    )
                counts = {len(ordered) for _, ordered in circuits.values()}
                _require(
                    len(counts) == 1,
                    "Weighted SSVQE requires the shared ansatz to have the same "
                    "logical parameter count for every level.",
                )
                parameter_count = counts.pop()
                weights = _ssvqe_weights(group)

                if len(group) > 1:
                    test_points = [np.zeros(parameter_count, dtype=float)]
                    if parameter_count:
                        test_points.append(np.full(parameter_count, 0.173, dtype=float))
                    for point in test_points:
                        metrics = _ssvqe_state_metrics(
                            circuits, point, matrix, basis_by_state
                        )
                        _require(
                            max(item["orthogonality_overlap"] for item in metrics.values())
                            < ORTHOGONALITY_TOLERANCE,
                            "Weighted SSVQE requires orthogonal reference states "
                            "acted on by one shared unitary ansatz.",
                        )

                gradient_error = None
                if parameter_count:
                    rng = np.random.default_rng(config["random_seed"] + depth * 1000)
                    gradient_error = _validate_weighted_ssvqe_gradient(
                        circuits=circuits,
                        weights=weights,
                        values=rng.uniform(-0.2, 0.2, size=parameter_count),
                        hamiltonian=hamiltonian,
                        physical_matrix=matrix,
                        gradient_engine=gradient_engine,
                        step=config["gradient_check_step"],
                        tolerance=config["gradient_check_tolerance"],
                    )

                candidates = candidates_by_group.setdefault(leader, [])
                continuations = continuations_by_group.setdefault(leader, [])
                stop_restarts = config.get("stop_restarts_on_convergence", False)
                for restart_index in range(len(candidates), config["restart_count"]):
                    if stop_restarts and any(
                        _ssvqe_candidate_meets_thresholds(
                            item, group, target_energy_by_state, config
                        )
                        for item in candidates
                    ):
                        break
                    seed = (
                        config["random_seed"] + depth * 10000 + restart_index
                        + labels.index(leader) * 1000000
                    )
                    saved_by_state = config.get(
                        "initial_parameters_by_depth", {}
                    ).get(str(depth), {})
                    saved = next(
                        (saved_by_state[label] for label in group
                         if label in saved_by_state),
                        None,
                    )
                    if saved is not None:
                        initial = make_refinement_initial_values(
                            saved_values=saved,
                            parameter_count=parameter_count,
                            restart_index=restart_index,
                            seed=seed,
                            perturbation_scale=config["perturbation_scale"],
                        )
                    else:
                        initial = make_initial_values(
                            config=config,
                            depth=depth,
                            parameter_count=parameter_count,
                            restart_index=restart_index,
                            seed=seed,
                            previous_depth=previous_depth,
                            previous_values=previous_by_group.get(leader),
                        )
                    candidate = _run_weighted_ssvqe_candidate(
                        config=config,
                        circuits=circuits,
                        weights=weights,
                        initial_values=initial,
                        hamiltonian=hamiltonian,
                        physical_matrix=matrix,
                        basis_by_state=basis_by_state,
                        gradient_engine=gradient_engine,
                        restart_index=restart_index,
                        seed=seed,
                    )
                    candidates.append(candidate)
                    _write_checkpoint(
                        adapter,
                        config,
                        completed,
                        {
                            "depth": depth,
                            "optimization_method": "weighted_ssvqe",
                            "candidates_by_representative": candidates_by_group,
                            "continuations_by_representative": continuations_by_group,
                        },
                    )
                    if stop_restarts and _ssvqe_candidate_meets_thresholds(
                        candidate, group, target_energy_by_state, config
                    ):
                        break

                selected = _select_candidate(
                    candidates + continuations, config["energy_tie_tolerance"]
                )
                for continuation_round in range(
                    len(continuations) + 1,
                    config["max_same_depth_continuations"] + 1,
                ):
                    if _ssvqe_candidate_meets_thresholds(
                        selected, group, target_energy_by_state, config
                    ) or not _optimizer_reached_iteration_limit(selected):
                        break
                    seed = (
                        config["random_seed"] + depth * 10000 + 500000
                        + continuation_round + labels.index(leader) * 1000000
                    )
                    refined = _run_weighted_ssvqe_candidate(
                        config=config,
                        circuits=circuits,
                        weights=weights,
                        initial_values=make_refinement_initial_values(
                            saved_values=selected["parameters"],
                            parameter_count=parameter_count,
                            restart_index=0,
                            seed=seed,
                            perturbation_scale=config["perturbation_scale"],
                        ),
                        hamiltonian=hamiltonian,
                        physical_matrix=matrix,
                        basis_by_state=basis_by_state,
                        gradient_engine=gradient_engine,
                        restart_index=int(selected["restart_index"]),
                        seed=seed,
                    )
                    refined["continuation_round"] = continuation_round
                    continuations.append(refined)
                    selected = _select_candidate(
                        candidates + continuations, config["energy_tie_tolerance"]
                    )
                    _write_checkpoint(
                        adapter,
                        config,
                        completed,
                        {
                            "depth": depth,
                            "optimization_method": "weighted_ssvqe",
                            "candidates_by_representative": candidates_by_group,
                            "continuations_by_representative": continuations_by_group,
                        },
                    )

                auxiliary_converged = all(
                    _candidate_meets_thresholds(
                        selected["state_metrics"][label],
                        target_energy_by_state[label],
                        config,
                    )
                    for label in group
                    if label not in requested_labels
                )
                for label in group:
                    metric = dict(selected["state_metrics"][label])
                    target_energy = target_energy_by_state[label]
                    state = {
                        **metric,
                        "depth": depth,
                        "label": label,
                        "parameters": list(selected["parameters"]),
                        "target_energy": target_energy,
                        "target_spec": config["target_spec_by_state"][label],
                        "energy_error": abs(metric["energy"] - target_energy),
                        "relative_energy_error": _relative_energy_error(
                            metric["energy"], target_energy
                        ),
                        "objective_value": selected["objective_value"],
                        "optimization_objective": (
                            "weighted_ssvqe" if len(group) > 1 else "energy"
                        ),
                        "ssvqe_group": list(group),
                        "ssvqe_weights": dict(weights),
                        "gradient_check_max_error": gradient_error,
                        "restart_index": selected["restart_index"],
                        "seed": selected["seed"],
                        "optimizer_success": selected["optimizer_success"],
                        "optimizer_status": selected["optimizer_status"],
                        "optimizer_message": selected["optimizer_message"],
                        "function_evaluations": selected["function_evaluations"],
                        "gradient_evaluations": selected["gradient_evaluations"],
                        "restart_records": candidates,
                        "continuation_records": continuations,
                        "shared_from": None,
                    }
                    state["status"] = (
                        "converged_state"
                        if auxiliary_converged and _candidate_meets_thresholds(
                            state,
                            target_energy,
                            config,
                        )
                        else "best_effort_state"
                    )
                    representative_results[label] = state

            states: Dict[str, Dict[str, Any]] = {}
            for label in labels:
                representative = representative_by_label[label]
                source = representative_results[representative]
                if label == representative:
                    states[label] = source
                    continue
                circuit, ordered = _build_state_preparation(adapter, label, depth)
                energy, raw_variance, variance = _state_energy_variance(
                    circuit, ordered, source["parameters"], matrix
                )
                state = dict(source)
                state.update({
                    "label": label,
                    "energy": energy,
                    "raw_variance": raw_variance,
                    "variance": variance,
                    "target_energy": target_energy_by_state[label],
                    "target_spec": config["target_spec_by_state"][label],
                    "energy_error": abs(energy - target_energy_by_state[label]),
                    "relative_energy_error": _relative_energy_error(
                        energy, target_energy_by_state[label]
                    ),
                    "sector_weight": _state_sector_weight(
                        circuit, ordered, source["parameters"], basis_by_state[label]
                    ),
                    "shared_from": representative,
                })
                states[label] = state

            status = (
                "converged_state"
                if all(state["status"] == "converged_state" for state in states.values())
                else "best_effort_state"
            )
            completed.append({"depth": depth, "status": status, "states": states})
            _write_checkpoint(adapter, config, completed)
            in_progress = None
            previous_depth = depth
            previous_by_group = {
                requested_leader(group): list(
                    representative_results[requested_leader(group)]["parameters"]
                )
                for group in groups
            }
            if status == "converged_state" and (
                adaptive_enabled or not config["scan_all_depths"]
            ):
                termination_reason = "accuracy_threshold_reached"
                break
            plateau_diagnostic = _adaptive_plateau_diagnostic(completed, config)
            if plateau_diagnostic["plateau"]:
                termination_reason = "accuracy_plateau"
                break

    if termination_reason is None:
        if any(entry["status"] == "converged_state" for entry in completed):
            termination_reason = "schedule_complete_with_converged_state"
        elif adaptive_enabled:
            termination_reason = "maximum_depth_reached"
        else:
            termination_reason = "fixed_schedule_complete_without_convergence"

    selected_by_state = {
        label: _select_candidate(
            [entry["states"][label] for entry in completed],
            config["energy_tie_tolerance"],
        )
        for label in labels
    }
    converged_entries = [
        entry for entry in completed if entry["status"] == "converged_state"
    ]
    validated_depths = [entry["depth"] for entry in converged_entries]
    shallowest = converged_entries[0]["depth"] if converged_entries else None
    plateau_diagnostic = _adaptive_plateau_diagnostic(completed, config)
    if converged_entries:
        run_status = "converged_state"
    elif adaptive_enabled and termination_reason == "accuracy_plateau":
        run_status = "ansatz_insufficient"
    elif adaptive_enabled:
        run_status = "promising_state"
    else:
        run_status = "best_effort_state"
    if run_status == "converged_state":
        next_action = {"action": "proceed_to_measurement", "requires_user_approval": False}
    elif run_status == "ansatz_insufficient":
        next_action = {
            "action": "redesign_state_preparation_ansatz",
            "requires_user_approval": True,
            "reason": "The weighted-SSVQE residual plateaued above the thresholds.",
        }
    elif run_status == "promising_state":
        next_action = {
            "action": "increase_vqe_budget",
            "requires_user_approval": False,
            "reason": "The weighted-SSVQE residual is still improving.",
            "recommended_follow_up": "extend_depth_with_warm_start",
        }
    else:
        next_action = {
            "action": "crosscheck_best_effort_or_enable_adaptive_depth",
            "requires_user_approval": False,
        }
    result = {
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "status": run_status,
        "termination_reason": termination_reason,
        "next_action": next_action,
        "adapter_sha256": _adapter_hash(adapter),
        "config": public_config,
        "target_ground_energy": None,
        "target_energies_by_state": {
            label: target_energy_by_state[label] for label in labels
        },
        "target_specs_by_state": public_config["target_spec_by_state"],
        "tested_depths": completed,
        "selected_states": selected_by_state,
        "validated_depths": validated_depths,
        "shallowest_validated_depth": shallowest,
        "hardware_eligible_depths": [shallowest] if shallowest is not None else [],
        "adaptive_depth_diagnostic": {
            "enabled": adaptive_enabled,
            "resolved_depth_schedule": config["depth_schedule"],
            "plateau": plateau_diagnostic,
            "residual_history": [
                {"depth": entry["depth"], "residual": _depth_residual(entry, config)}
                for entry in completed
            ],
        },
        "trend_summary": _depth_trend_summary(completed, config),
    }
    _atomic_json_write(config["result_file"], result)
    return result


def run_vqe(adapter: Any, run_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a depth scan for ground or quantum-number-sector eigenstates."""
    config = validate_run_config(run_config)
    hamiltonian = _build_hamiltonian(adapter)
    matrix = np.asarray(hamiltonian.to_matrix(), dtype=complex)
    labels = config["state_labels"]
    representatives, representative_by_label = _representatives(
        labels, config["equivalent_state_groups"]
    )
    basis_by_state = {
        label: _physical_basis(adapter, matrix.shape[0], label) for label in labels
    }
    # Independent exact references validate the final state but never enter
    # the variational objective.
    target_energy_by_state = {
        label: _exact_sector_energy(
            matrix,
            basis_by_state[label],
            config["target_spec_by_state"][label]["level_index"],
        )
        for label in labels
    }
    if any(
        spec["level_index"] > 0
        for spec in config["target_spec_by_state"].values()
    ):
        return _run_weighted_ssvqe(
            adapter=adapter,
            config=config,
            hamiltonian=hamiltonian,
            matrix=matrix,
            labels=labels,
            representatives=representatives,
            representative_by_label=representative_by_label,
            basis_by_state=basis_by_state,
            target_energy_by_state=target_energy_by_state,
        )
    gradient_engine = ReverseEstimatorGradient()

    completed, in_progress = _load_checkpoint(adapter, config, matrix)
    gradient_checked = set()
    previous_depth: Optional[int] = None
    previous_by_representative: Dict[str, List[float]] = {}
    if completed:
        previous_depth = completed[-1]["depth"]
        for representative in representatives:
            previous_by_representative[representative] = list(
                completed[-1]["states"][representative]["parameters"]
            )

    adaptive_enabled = config.get("adaptive_depth", {}).get("enabled", False)
    already_converged = any(entry["status"] == "converged_state" for entry in completed)
    plateau_diagnostic = _adaptive_plateau_diagnostic(completed, config)
    termination_reason: Optional[str] = None
    if already_converged and adaptive_enabled:
        should_continue = False
        termination_reason = "accuracy_threshold_reached"
    elif plateau_diagnostic["plateau"]:
        should_continue = False
        termination_reason = "accuracy_plateau"
    else:
        should_continue = config["scan_all_depths"] or not already_converged
    if should_continue:
        for depth in config["depth_schedule"][len(completed):]:
            representative_results: Dict[str, Dict[str, Any]] = {}
            candidates_by_representative: Dict[str, List[Dict[str, Any]]] = {}
            continuations_by_representative: Dict[str, List[Dict[str, Any]]] = {}
            if in_progress is not None:
                _require(in_progress["depth"] == depth,
                         "Loaded in-progress checkpoint has the wrong depth.")
                candidates_by_representative = {
                    label: [dict(candidate) for candidate in candidates]
                    for label, candidates in in_progress[
                        "candidates_by_representative"
                    ].items()
                }
                continuations_by_representative = {
                    label: [dict(candidate) for candidate in candidates]
                    for label, candidates in in_progress.get(
                        "continuations_by_representative", {}
                    ).items()
                }
            for representative in representatives:
                target_energy = target_energy_by_state[representative]
                circuit, ordered = _build_state_preparation(adapter, representative, depth)
                parameter_count = len(ordered)
                level_index = config["target_spec_by_state"][representative][
                    "level_index"
                ]
                _require(level_index == 0,
                         "Excited targets must use the weighted-SSVQE path.")
                validation_energy = target_energy
                lower_states: List[np.ndarray] = []
                objective_operator = hamiltonian
                objective_kind = "energy"
                objective_matrix = matrix
                gradient_key = (representative, depth)
                if parameter_count and gradient_key not in gradient_checked:
                    seed = config["random_seed"] + depth * 1000
                    test_rng = np.random.default_rng(seed)
                    test_values = test_rng.uniform(-0.2, 0.2, size=parameter_count)
                    error = validate_reverse_gradient(
                        gradient_engine,
                        circuit,
                        ordered,
                        test_values,
                        objective_operator,
                        objective_matrix,
                        config["gradient_check_step"],
                        config["gradient_check_tolerance"],
                    )
                    gradient_checked.add(gradient_key)
                else:
                    error = None

                candidates = candidates_by_representative.setdefault(
                    representative, []
                )
                continuations = continuations_by_representative.setdefault(
                    representative, []
                )
                _require(
                    [candidate.get("restart_index") for candidate in candidates]
                    == list(range(len(candidates))),
                    "Saved restart records are not a consecutive prefix.",
                )
                _require(len(candidates) <= config["restart_count"],
                         "Checkpoint contains too many restart records.")
                for candidate in candidates:
                    energy, raw_variance, variance = _state_energy_variance(
                        circuit, ordered, candidate.get("parameters", []), matrix
                    )
                    _require(np.isclose(energy, candidate.get("energy"), atol=1.0e-9,
                                        rtol=1.0e-9),
                             "Saved restart energy does not reproduce.")
                    _require(np.isclose(raw_variance,
                                        candidate.get("raw_variance"), atol=1.0e-9,
                                        rtol=1.0e-8),
                             "Saved restart raw variance does not reproduce.")
                    _require(np.isclose(variance, candidate.get("variance"), atol=1.0e-9,
                                        rtol=1.0e-8),
                             "Saved restart variance does not reproduce.")
                _require(
                    [candidate.get("continuation_round") for candidate in continuations]
                    == list(range(1, len(continuations) + 1)),
                    "Saved continuation records are not a consecutive prefix.",
                )
                _require(
                    len(continuations) <= config["max_same_depth_continuations"],
                    "Checkpoint contains too many same-depth continuations.",
                )
                for candidate in continuations:
                    energy, raw_variance, variance = _state_energy_variance(
                        circuit, ordered, candidate.get("parameters", []), matrix
                    )
                    _require(np.isclose(energy, candidate.get("energy"), atol=1.0e-9,
                                        rtol=1.0e-9),
                             "Saved continuation energy does not reproduce.")
                    _require(np.isclose(raw_variance,
                                        candidate.get("raw_variance"), atol=1.0e-9,
                                        rtol=1.0e-8),
                             "Saved continuation raw variance does not reproduce.")
                    _require(np.isclose(variance, candidate.get("variance"), atol=1.0e-9,
                                        rtol=1.0e-8),
                             "Saved continuation variance does not reproduce.")

                stop_restarts = config.get("stop_restarts_on_convergence", False)
                for restart_index in range(len(candidates), config["restart_count"]):
                    if stop_restarts and any(
                        _candidate_meets_thresholds(
                            candidate, validation_energy, config
                        )
                        for candidate in candidates
                    ):
                        break
                    seed = (
                        config["random_seed"]
                        + depth * 10000
                        + restart_index
                        + labels.index(representative) * 1000000
                    )
                    saved_by_state = config.get(
                        "initial_parameters_by_depth", {}
                    ).get(str(depth), {})
                    if representative in saved_by_state:
                        initial = make_refinement_initial_values(
                            saved_values=saved_by_state[representative],
                            parameter_count=parameter_count,
                            restart_index=restart_index,
                            seed=seed,
                            perturbation_scale=config["perturbation_scale"],
                        )
                    else:
                        initial = make_initial_values(
                            config=config,
                            depth=depth,
                            parameter_count=parameter_count,
                            restart_index=restart_index,
                            seed=seed,
                            previous_depth=previous_depth,
                            previous_values=previous_by_representative.get(representative),
                        )
                    candidate = _run_candidate(
                        config=config,
                        circuit=circuit,
                        ordered_parameters=ordered,
                        initial_values=initial,
                        objective_hamiltonian=objective_operator,
                        objective_matrix=objective_matrix,
                        physical_matrix=matrix,
                        target_basis=basis_by_state[representative],
                        lower_states=lower_states,
                        gradient_engine=gradient_engine,
                        restart_index=restart_index,
                        seed=seed,
                    )
                    candidates.append(candidate)
                    _write_checkpoint(
                        adapter,
                        config,
                        completed,
                        {
                            "depth": depth,
                            "candidates_by_representative": (
                                candidates_by_representative
                            ),
                            "continuations_by_representative": (
                                continuations_by_representative
                            ),
                        },
                    )
                    if stop_restarts and _candidate_meets_thresholds(
                        candidate, validation_energy, config
                    ):
                        break
                selected = _select_candidate(
                    candidates + continuations, config["energy_tie_tolerance"]
                )
                for continuation_round in range(
                    len(continuations) + 1,
                    config["max_same_depth_continuations"] + 1,
                ):
                    if _candidate_meets_thresholds(selected, validation_energy, config):
                        break
                    if not _optimizer_reached_iteration_limit(selected):
                        break
                    residual_before = _candidate_residual(
                        selected, validation_energy, config
                    )
                    seed = (
                        config["random_seed"]
                        + depth * 10000
                        + 500000
                        + continuation_round
                        + labels.index(representative) * 1000000
                    )
                    refined = _run_candidate(
                        config=config,
                        circuit=circuit,
                        ordered_parameters=ordered,
                        initial_values=make_refinement_initial_values(
                            saved_values=selected["parameters"],
                            parameter_count=parameter_count,
                            restart_index=0,
                            seed=seed,
                            perturbation_scale=config["perturbation_scale"],
                        ),
                        objective_hamiltonian=objective_operator,
                        objective_matrix=objective_matrix,
                        physical_matrix=matrix,
                        target_basis=basis_by_state[representative],
                        lower_states=lower_states,
                        gradient_engine=gradient_engine,
                        restart_index=int(selected["restart_index"]),
                        seed=seed,
                    )
                    refined["continuation_round"] = continuation_round
                    refined["continued_from_residual"] = residual_before
                    refined["resulting_residual"] = _candidate_residual(
                        refined, validation_energy, config
                    )
                    continuations.append(refined)
                    selected = _select_candidate(
                        candidates + continuations,
                        config["energy_tie_tolerance"],
                    )
                    _write_checkpoint(
                        adapter,
                        config,
                        completed,
                        {
                            "depth": depth,
                            "candidates_by_representative": (
                                candidates_by_representative
                            ),
                            "continuations_by_representative": (
                                continuations_by_representative
                            ),
                        },
                    )
                selected["depth"] = depth
                selected["label"] = representative
                selected["target_energy"] = target_energy
                selected["target_spec"] = config["target_spec_by_state"][representative]
                selected["energy_error"] = abs(selected["energy"] - target_energy)
                selected["relative_energy_error"] = _relative_energy_error(
                    selected["energy"], target_energy
                )
                selected["optimization_objective"] = objective_kind
                selected["status"] = (
                    "converged_state"
                    if _candidate_meets_thresholds(
                        selected, validation_energy, config
                    )
                    else "best_effort_state"
                )
                selected["gradient_check_max_error"] = error
                selected["restart_records"] = candidates
                selected["continuation_records"] = continuations
                representative_results[representative] = selected

            states: Dict[str, Dict[str, Any]] = {}
            for label in labels:
                representative = representative_by_label[label]
                target_energy = target_energy_by_state[label]
                source = representative_results[representative]
                circuit, ordered = _build_state_preparation(adapter, label, depth)
                representative_circuit, representative_ordered = _build_state_preparation(
                    adapter, representative, depth
                )
                _require(len(ordered) == len(representative_ordered),
                         "Equivalent-state ansatz parameter counts differ.")
                _require(_circuit_signature(circuit) == _circuit_signature(representative_circuit),
                         "Equivalent-state ansatz gate structures differ.")
                energy, raw_variance, variance = _state_energy_variance(
                    circuit, ordered, source["parameters"], matrix
                )
                sector_weight = _state_sector_weight(
                    circuit, ordered, source["parameters"], basis_by_state[label]
                )
                state = dict(source)
                state.update({
                    "label": label,
                    "energy": energy,
                    "raw_variance": raw_variance,
                    "variance": variance,
                    "target_energy": target_energy,
                    "target_spec": config["target_spec_by_state"][label],
                    "energy_error": abs(energy - target_energy),
                    "relative_energy_error": _relative_energy_error(
                        energy, target_energy
                    ),
                    "sector_weight": sector_weight,
                    "shared_from": representative if label != representative else None,
                })
                state["status"] = (
                    "converged_state"
                    if _candidate_meets_thresholds(
                        state,
                        target_energy,
                        config,
                    )
                    else "best_effort_state"
                )
                states[label] = state

            status = (
                "converged_state"
                if all(state["status"] == "converged_state" for state in states.values())
                else "best_effort_state"
            )
            completed.append({"depth": depth, "status": status, "states": states})
            _write_checkpoint(adapter, config, completed)
            in_progress = None
            previous_depth = depth
            previous_by_representative = {
                representative: representative_results[representative]["parameters"]
                for representative in representatives
            }
            if status == "converged_state" and (
                adaptive_enabled or not config["scan_all_depths"]
            ):
                termination_reason = "accuracy_threshold_reached"
                break
            plateau_diagnostic = _adaptive_plateau_diagnostic(completed, config)
            if plateau_diagnostic["plateau"]:
                termination_reason = "accuracy_plateau"
                break

    if termination_reason is None:
        if any(entry["status"] == "converged_state" for entry in completed):
            termination_reason = "schedule_complete_with_converged_state"
        elif adaptive_enabled:
            termination_reason = "maximum_depth_reached"
        else:
            termination_reason = "fixed_schedule_complete_without_convergence"

    selected_by_state: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        candidates = [entry["states"][label] for entry in completed]
        selected_by_state[label] = _select_candidate(
            candidates, config["energy_tie_tolerance"]
        )
    converged_entries = [entry for entry in completed if entry["status"] == "converged_state"]
    validated_depths = [entry["depth"] for entry in converged_entries]
    shallowest_validated_depth = (
        converged_entries[0]["depth"] if converged_entries else None
    )
    plateau_diagnostic = _adaptive_plateau_diagnostic(completed, config)
    if converged_entries:
        run_status = "converged_state"
    elif adaptive_enabled and termination_reason == "accuracy_plateau":
        run_status = "ansatz_insufficient"
    elif adaptive_enabled:
        # Exhausting the approved depth budget while R_D is still falling is
        # evidence for the family, not evidence against it.
        run_status = "promising_state"
    else:
        run_status = "best_effort_state"
    if run_status == "converged_state":
        next_action = {
            "action": "proceed_to_measurement",
            "requires_user_approval": False,
        }
    elif run_status == "ansatz_insufficient":
        next_action = {
            "action": "redesign_state_preparation_ansatz",
            "requires_user_approval": True,
            "reason": (
                "A paid Agent4.1/Agent5 redesign is never launched automatically. "
                "The recorded VQE history will be supplied on the next explicit run."
            ),
        }
    elif run_status == "promising_state":
        next_action = {
            "action": "increase_vqe_budget",
            "requires_user_approval": False,
            "reason": (
                "The bounded depth schedule ended while the normalized VQE "
                "residual was still improving, but the accuracy thresholds were "
                "not met. Keep this ansatz family and continue from its saved "
                "parameters; do not enter measurement or cross-check yet."
            ),
            "recommended_follow_up": "extend_depth_with_warm_start",
        }
    else:
        next_action = {
            "action": "crosscheck_best_effort_or_enable_adaptive_depth",
            "requires_user_approval": False,
        }
    trend_summary = _depth_trend_summary(completed, config)
    result = {
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "status": run_status,
        "termination_reason": termination_reason,
        "next_action": next_action,
        "adapter_sha256": _adapter_hash(adapter),
        "config": config,
        # Retained for readers of the former ground-only result schema.
        "target_ground_energy": (
            target_energy_by_state[labels[0]]
            if all(config["target_kind_by_state"][label] == "ground" for label in labels)
            else None
        ),
        "target_energies_by_state": target_energy_by_state,
        "target_specs_by_state": config["target_spec_by_state"],
        "tested_depths": completed,
        "selected_states": selected_by_state,
        "validated_depths": validated_depths,
        "shallowest_validated_depth": shallowest_validated_depth,
        "hardware_eligible_depths": (
            [shallowest_validated_depth]
            if shallowest_validated_depth is not None else []
        ),
        "adaptive_depth_diagnostic": {
            "enabled": adaptive_enabled,
            "resolved_depth_schedule": config["depth_schedule"],
            "plateau": plateau_diagnostic,
            "residual_history": [
                {
                    "depth": entry["depth"],
                    "residual": _depth_residual(entry, config),
                }
                for entry in completed
            ],
        },
        "trend_summary": trend_summary,
    }
    _atomic_json_write(config["result_file"], result)
    return result


def _load_adapter(path: str) -> Any:
    absolute = os.path.abspath(path)
    _require(os.path.isfile(absolute), "Adapter file does not exist: {}".format(path))
    spec = importlib.util.spec_from_file_location("state_preparation_adapter", absolute)
    _require(spec is not None and spec.loader is not None,
             "Cannot import adapter: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic VQE depth scans for an Agent-5 adapter."
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    adapter = _load_adapter(args.adapter)
    result = run_vqe(adapter, config)
    run_id = None
    if result["config"].get("record_experience", False):
        from vqe_experience_store import record_vqe_result

        run_id = record_vqe_result(
            result["config"]["experience_database"],
            adapter,
            result["config"],
            result,
            result_path=result["config"]["result_file"],
        )
        result["vqe_run_id"] = run_id
        _atomic_json_write(result["config"]["result_file"], result)
        record_vqe_result(
            result["config"]["experience_database"],
            adapter,
            result["config"],
            result,
            result_path=result["config"]["result_file"],
        )
    bundle_path = result["config"].get(
        "parameter_bundle_file",
        default_parameter_bundle_path(result["config"]["result_file"]),
    )
    write_parameter_bundle(
        adapter,
        result,
        bundle_path,
        source_result_path=result["config"]["result_file"],
    )
    print(
        "VQE completed: {} depth(s), status {}, result {}, parameter bundle {}{}".format(
            len(result["tested_depths"]),
            result["status"],
            result["config"]["result_file"],
            bundle_path,
            ", experience run {}".format(run_id) if run_id else "",
        )
    )


if __name__ == "__main__":
    main()
