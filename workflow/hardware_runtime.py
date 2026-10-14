# -*- coding: utf-8 -*-
"""Deterministic IBM Quantum planning, submission, and retrieval runtime.

Planning queries current backend state and records transpiled hardware evidence.
Submission is a separate command that requires Agent 9's explicit approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
from qiskit_ibm_runtime.options import EstimatorOptions

from vqe_experience_store import (
    DEFAULT_DATABASE_PATH,
    record_hardware_plan,
    record_hardware_results,
    record_hardware_run,
)


HARDWARE_PLAN_SCHEMA_VERSION = "1.0"
HARDWARE_RUN_SCHEMA_VERSION = "1.0"
DEFAULT_MAX_ISA_DEPTH_PER_RECIPE = 5000
DEFAULT_MAX_TOTAL_ISA_TWO_QUBIT_GATES = 20000
DEFAULT_MAX_TWO_QUBIT_ERROR_SUM = 1.0
MIN_INSTANCE_REMAINING_SECONDS = 600
MAX_ZNE_STANDARD_ERROR_RATIO = 3.0


class HardwareRuntimeError(RuntimeError):
    """Raised when an unapproved or incompatible hardware operation is requested."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HardwareRuntimeError(message)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), "Expected JSON object: {}".format(path))
    return value


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".hardware_runtime_", suffix=".tmp", dir=directory
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


def validate_crosscheck_manifest(
    manifest: Mapping[str, Any],
    candidate_class: str = "hardware_demonstration_candidate",
) -> List[Dict[str, Any]]:
    _require(manifest.get("crosscheck_schema_version") == "1.0",
             "Hardware runtime requires crosscheck schema 1.0.")
    candidate_fields = {
        "physics_hardware_candidate": "hardware_eligible_candidates",
        "hardware_demonstration_candidate": "hardware_demonstration_candidates",
    }
    _require(candidate_class in candidate_fields, "Unknown hardware candidate class.")
    if candidate_class == "physics_hardware_candidate":
        _require(manifest.get("status") == "approved",
                 "Cross-check has not approved physics hardware submission.")
    candidates = manifest.get(candidate_fields[candidate_class])
    _require(isinstance(candidates, list) and candidates,
             "No {} entries are available.".format(candidate_fields[candidate_class]))
    seen = set()
    result = []
    for raw in candidates:
        _require(isinstance(raw, dict), "Hardware candidate must be an object.")
        candidate = dict(raw)
        candidate_id = candidate.get("candidate_id")
        _require(isinstance(candidate_id, str) and candidate_id and candidate_id not in seen,
                 "Hardware candidate IDs must be unique nonempty strings.")
        _require(isinstance(candidate.get("sample_values"), dict),
                 "Hardware candidate must include sample_values.")
        seen.add(candidate_id)
        if candidate_class == "physics_hardware_candidate":
            _require(candidate.get("state_preparation_validated") is True,
                     "Physics candidate uses an unvalidated state preparation.")
        else:
            _require(
                candidate.get("interpretation")
                == "same_circuit_noiseless_comparison_only",
                "Hardware demo must be limited to same-circuit comparison.",
            )
        recipes = candidate.get("recipes")
        _require(isinstance(recipes, list) and recipes,
                 "Hardware candidate has no measurement recipes.")
        recipe_ids = set()
        for recipe in recipes:
            recipe_id = recipe.get("recipe_id")
            path = recipe.get("qpy_path")
            _require(isinstance(recipe_id, str) and recipe_id and recipe_id not in recipe_ids,
                     "Hardware recipe IDs must be unique.")
            recipe_ids.add(recipe_id)
            _require(isinstance(path, str) and os.path.isfile(path),
                     "Hardware QPY file is missing: {}".format(path))
            _require(_sha256_file(path) == recipe.get("qpy_sha256"),
                     "Hardware QPY hash mismatch: {}".format(path))
            pauli = recipe.get("readout_pauli")
            _require(isinstance(pauli, str) and pauli and set(pauli) <= set("IXYZ"),
                     "Hardware recipe has an invalid readout Pauli.")
        result.append(candidate)
    return result


def _load_one_qpy(path: str) -> QuantumCircuit:
    with open(path, "rb") as handle:
        circuits = qpy.load(handle)
    _require(len(circuits) == 1, "Every recipe QPY must contain exactly one circuit.")
    circuit = circuits[0]
    _require(isinstance(circuit, QuantumCircuit) and not circuit.parameters,
             "Hardware recipe must be one parameter-free QuantumCircuit.")
    return circuit


def _backend_name(backend: Any) -> str:
    value = getattr(backend, "name", None)
    return str(value() if callable(value) else value)


def _backend_pending_jobs(backend: Any) -> int:
    try:
        status = backend.status()
        return int(getattr(status, "pending_jobs", 0))
    except Exception:
        return 0


def _backend_calibration_time(backend: Any) -> str | None:
    try:
        properties = backend.properties()
        value = getattr(properties, "last_update_date", None)
        return value.isoformat() if value is not None else None
    except Exception:
        return None


def _two_qubit_error(backend: Any, circuit: QuantumCircuit) -> Tuple[float, int]:
    total = 0.0
    missing = 0
    for instruction in circuit.data:
        if (len(instruction.qubits) != 2
                or instruction.operation.name in ("barrier", "delay")):
            continue
        indices = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        try:
            properties = backend.target[instruction.operation.name][indices]
            error = getattr(properties, "error", None)
            if error is None or not math.isfinite(float(error)):
                missing += 1
            else:
                total += float(error)
        except Exception:
            missing += 1
    return total, missing


def _one_qubit_error(backend: Any, circuit: QuantumCircuit) -> Tuple[float, int]:
    total = 0.0
    missing = 0
    for instruction in circuit.data:
        if (len(instruction.qubits) != 1
                or instruction.operation.name in ("barrier", "delay")):
            continue
        indices = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        try:
            properties = backend.target[instruction.operation.name][indices]
            error = getattr(properties, "error", None)
            if error is None or not math.isfinite(float(error)):
                missing += 1
            else:
                total += float(error)
        except Exception:
            missing += 1
    return total, missing


def _timing_and_readout_evidence(backend: Any, circuit: QuantumCircuit) -> Dict[str, Any]:
    active = sorted({
        circuit.find_bit(qubit).index
        for instruction in circuit.data
        if instruction.operation.name != "barrier"
        for qubit in instruction.qubits
    })
    try:
        duration = float(circuit.estimate_duration(backend.target, unit="s"))
        if not math.isfinite(duration):
            duration = None
    except Exception:
        duration = None
    t1_values = []
    t2_values = []
    readout_errors = []
    readout_missing = 0
    for qubit in active:
        try:
            properties = backend.qubit_properties(qubit)
            if properties is not None:
                if properties.t1 is not None and math.isfinite(float(properties.t1)):
                    t1_values.append(float(properties.t1))
                if properties.t2 is not None and math.isfinite(float(properties.t2)):
                    t2_values.append(float(properties.t2))
        except Exception:
            pass
        try:
            error = getattr(backend.target["measure"][(qubit,)], "error", None)
            if error is None or not math.isfinite(float(error)):
                readout_missing += 1
            else:
                readout_errors.append(float(error))
        except Exception:
            readout_missing += 1
    minimum_t1 = min(t1_values) if t1_values else None
    minimum_t2 = min(t2_values) if t2_values else None
    return {
        "active_physical_qubits": active,
        "estimated_duration_seconds": duration,
        "minimum_t1_seconds": minimum_t1,
        "minimum_t2_seconds": minimum_t2,
        "duration_over_minimum_t1": (
            duration / minimum_t1 if duration is not None and minimum_t1 else None
        ),
        "duration_over_minimum_t2": (
            duration / minimum_t2 if duration is not None and minimum_t2 else None
        ),
        "mean_readout_error": (
            sum(readout_errors) / len(readout_errors) if readout_errors else None
        ),
        "readout_errors_missing": readout_missing,
    }


def _isa_resources(backend: Any, circuit: QuantumCircuit) -> Dict[str, Any]:
    one_qubit = sum(
        len(item.qubits) == 1 and item.operation.name not in ("barrier", "delay")
        for item in circuit.data
    )
    two_qubit = sum(
        len(item.qubits) == 2 and item.operation.name not in ("barrier", "delay")
        for item in circuit.data
    )
    swaps = sum(item.operation.name == "swap" for item in circuit.data)
    one_error_sum, one_missing = _one_qubit_error(backend, circuit)
    two_error_sum, two_missing = _two_qubit_error(backend, circuit)
    resources = {
        "isa_depth": circuit.depth(),
        "isa_operation_count": len(circuit.data),
        "isa_one_qubit_gates": one_qubit,
        "isa_two_qubit_gates": two_qubit,
        "isa_swaps": swaps,
        "one_qubit_error_sum": one_error_sum,
        "one_qubit_errors_missing": one_missing,
        "two_qubit_error_sum": two_error_sum,
        "two_qubit_errors_missing": two_missing,
    }
    resources.update(_timing_and_readout_evidence(backend, circuit))
    return resources


def _save_isa(path: str, circuit: QuantumCircuit) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        qpy.dump(circuit, handle)
    return _sha256_file(path)


def _resource_gate(
    deployment: Mapping[str, Any],
    *,
    max_isa_depth_per_recipe: int,
    max_total_isa_two_qubit_gates: int,
    max_two_qubit_error_sum: float,
) -> Dict[str, Any]:
    """Record reference-limit comparisons as evidence, never as submission authority."""
    _require(max_isa_depth_per_recipe > 0,
             "max_isa_depth_per_recipe must be positive.")
    _require(max_total_isa_two_qubit_gates > 0,
             "max_total_isa_two_qubit_gates must be positive.")
    _require(math.isfinite(max_two_qubit_error_sum) and max_two_qubit_error_sum > 0,
             "max_two_qubit_error_sum must be positive and finite.")
    recipes = deployment.get("recipes")
    _require(isinstance(recipes, list) and recipes,
             "A deployment must contain transpiled recipe resources.")
    resources = [item["resources"] for item in recipes]
    maximum_depth = max(int(item["isa_depth"]) for item in resources)
    total_two_qubit = sum(int(item["isa_two_qubit_gates"]) for item in resources)
    total_error = sum(float(item["two_qubit_error_sum"]) for item in resources)
    maximum_recipe_error = max(
        float(item["two_qubit_error_sum"]) for item in resources
    )
    total_missing = sum(int(item["two_qubit_errors_missing"]) for item in resources)
    reasons = []
    if maximum_depth > max_isa_depth_per_recipe:
        reasons.append("maximum recipe ISA depth exceeds limit")
    if total_two_qubit > max_total_isa_two_qubit_gates:
        reasons.append("total ISA two-qubit gate count exceeds limit")
    if maximum_recipe_error > max_two_qubit_error_sum:
        reasons.append("maximum recipe two-qubit calibration error exceeds limit")
    if total_missing:
        reasons.append("two-qubit calibration errors are incomplete")
    return {
        "advisory_only": True,
        "passed": not reasons,
        "rejection_reasons": reasons,
        "observed": {
            "maximum_isa_depth_per_recipe": maximum_depth,
            "total_isa_two_qubit_gates": total_two_qubit,
            "two_qubit_error_sum": total_error,
            "maximum_recipe_two_qubit_error_sum": maximum_recipe_error,
            "two_qubit_errors_missing": total_missing,
        },
        "limits": {
            "maximum_isa_depth_per_recipe": max_isa_depth_per_recipe,
            "total_isa_two_qubit_gates": max_total_isa_two_qubit_gates,
            "maximum_recipe_two_qubit_error_sum": max_two_qubit_error_sum,
            "require_complete_two_qubit_errors": True,
        },
    }


def _pareto_frontier(deployments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return nondominated accuracy/resource deployments without hidden weights."""
    def metrics(item: Mapping[str, Any]) -> Tuple[float, float, float, float]:
        observed = item["resource_gate"]["observed"]
        return (
            float(item["algorithm_error_vs_scipy"]),
            float(observed["maximum_isa_depth_per_recipe"]),
            float(observed["total_isa_two_qubit_gates"]),
            float(observed["two_qubit_error_sum"]),
        )

    frontier = []
    for candidate in deployments:
        candidate_metrics = metrics(candidate)
        dominated = False
        for other in deployments:
            if other is candidate:
                continue
            other_metrics = metrics(other)
            if (all(left <= right for left, right in zip(other_metrics, candidate_metrics))
                    and any(left < right for left, right in zip(other_metrics, candidate_metrics))):
                dominated = True
                break
        if not dominated:
            frontier.append(dict(candidate))
    return frontier


def _screen_deployments(
    deployments: Sequence[Mapping[str, Any]],
    *,
    max_isa_depth_per_recipe: int,
    max_total_isa_two_qubit_gates: int,
    max_two_qubit_error_sum: float,
    max_selections: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    screened = []
    for raw in deployments:
        item = dict(raw)
        item["resource_gate"] = _resource_gate(
            item,
            max_isa_depth_per_recipe=max_isa_depth_per_recipe,
            max_total_isa_two_qubit_gates=max_total_isa_two_qubit_gates,
            max_two_qubit_error_sum=max_two_qubit_error_sum,
        )
        screened.append(item)
    # These comparisons are useful evidence for Agent 9, but fixed thresholds
    # cannot decide whether a backend will preserve a scientifically meaningful
    # signal.  Agent 9 receives every transpiled deployment.
    frontier = _pareto_frontier(screened)
    frontier.sort(key=lambda item: (
        item["score"], item["algorithm_error_vs_scipy"],
        item["D_ans"], item["N_T"], item["backend"],
    ))
    return screened, frontier, frontier[:max_selections]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "item"


def _discard_unselected_isa(
    deployments: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> int:
    """Keep executable QPY only for deployments that may actually be submitted."""
    selected_keys = {
        (item.get("candidate_id"), item.get("backend")) for item in selected
    }
    removed = 0
    for deployment in deployments:
        if (deployment.get("candidate_id"), deployment.get("backend")) in selected_keys:
            continue
        for recipe in deployment.get("recipes", []):
            path = recipe.pop("isa_qpy_path", None)
            recipe.pop("isa_qpy_sha256", None)
            recipe["isa_retained"] = False
            if isinstance(path, str) and os.path.isfile(path):
                os.unlink(path)
                removed += 1
    return removed


def _measurement_metadata(
    crosscheck_manifest: Mapping[str, Any], crosscheck_path: str
) -> Dict[str, Any]:
    sources = crosscheck_manifest.get("source_files", {})
    measurement_path = sources.get("measurement")
    _require(isinstance(measurement_path, str) and os.path.isfile(measurement_path),
             "Cross-check does not point to its measurement result.")
    _require(_sha256_file(measurement_path) == sources.get("measurement_sha256"),
             "Measurement-result hash changed after cross-check.")
    measurement = _load_json(measurement_path)
    observable_path = measurement.get("observable_adapter_path")
    _require(isinstance(observable_path, str) and os.path.isfile(observable_path),
             "Measurement result does not identify the observable adapter.")
    _require(_sha256_file(observable_path) == measurement.get("observable_adapter_sha256"),
             "Observable adapter changed after noiseless validation.")
    return {
        "crosscheck_path": os.path.abspath(crosscheck_path),
        "crosscheck_sha256": _sha256_file(crosscheck_path),
        "measurement_path": os.path.abspath(measurement_path),
        "measurement_sha256": _sha256_file(measurement_path),
        "observable_adapter_path": os.path.abspath(observable_path),
        "observable_adapter_sha256": measurement["observable_adapter_sha256"],
    }


def _choose_backends(
    service: QiskitRuntimeService,
    minimum_qubits: int,
    backend_names: Sequence[str],
    max_backends: int,
) -> List[Tuple[Any, Mapping[str, Any], Mapping[str, Any]]]:
    """Return operational backends only from instances with execution time.

    IBM allocation is attached to an account instance, not to an individual
    backend.  Query every accessible instance explicitly so a backend on an
    exhausted instance cannot be selected merely because its queue is short.
    """
    requested = set(backend_names)
    candidates = []
    for instance in service.instances():
        crn = instance.get("crn")
        if not isinstance(crn, str) or not crn:
            continue
        try:
            backends = service.backends(
                min_num_qubits=minimum_qubits,
                instance=crn,
                operational=True,
                simulator=False,
            )
            if requested:
                backends = [
                    backend for backend in backends
                    if _backend_name(backend) in requested
                ]
            if not backends:
                continue
            usage = service.usage()
        except Exception:
            # An inaccessible/stale instance is not a safe submission target.
            continue
        remaining = usage.get("usage_remaining_seconds")
        has_safe_time = (
            usage.get("usage_limit_reached") is not True
            and isinstance(remaining, (int, float))
            and remaining > MIN_INSTANCE_REMAINING_SECONDS
        )
        if not has_safe_time:
            continue
        for backend in backends:
            candidates.append((backend, instance, usage))

    candidates.sort(key=lambda item: _backend_pending_jobs(item[0]))
    candidates = candidates[:max_backends]
    _require(
        candidates,
        "No requested operational hardware backend has available instance time.",
    )
    return candidates


def build_hardware_plan(
    crosscheck_manifest: Mapping[str, Any],
    crosscheck_path: str,
    service: QiskitRuntimeService,
    output_path: str,
    isa_directory: str,
    *,
    candidate_class: str = "hardware_demonstration_candidate",
    max_demo_recipes: int = 2,
    demo_mode: str = "representative",
    backend_names: Sequence[str] = (),
    max_backends: int = 5,
    max_selections: int = 1,
    seed_transpiler: int = 17,
    max_isa_depth_per_recipe: int = DEFAULT_MAX_ISA_DEPTH_PER_RECIPE,
    max_total_isa_two_qubit_gates: int = DEFAULT_MAX_TOTAL_ISA_TWO_QUBIT_GATES,
    max_two_qubit_error_sum: float = DEFAULT_MAX_TWO_QUBIT_ERROR_SUM,
    database_path: str | None = None,
) -> Dict[str, Any]:
    _require(isinstance(max_backends, int) and max_backends > 0,
             "max_backends must be positive.")
    _require(isinstance(max_selections, int) and max_selections > 0,
             "max_selections must be positive.")
    _require(demo_mode in ("representative", "full_d1_nt1"),
             "demo_mode must be representative or full_d1_nt1.")
    _require(isinstance(max_demo_recipes, int) and max_demo_recipes > 0,
             "max_demo_recipes must be positive.")
    candidates = validate_crosscheck_manifest(crosscheck_manifest, candidate_class)
    full_demo = (
        candidate_class == "hardware_demonstration_candidate"
        and demo_mode == "full_d1_nt1"
    )
    if candidate_class == "hardware_demonstration_candidate" and not full_demo:
        for candidate in candidates:
            recipes = sorted(
                candidate["recipes"],
                key=lambda recipe: abs(float(recipe["noiseless_expectation"])),
                reverse=True,
            )
            candidate["recipes"] = recipes[:max_demo_recipes]
    metadata = _measurement_metadata(crosscheck_manifest, crosscheck_path)
    minimum_qubits = max(
        int(recipe["resources"]["num_qubits"])
        for candidate in candidates for recipe in candidate["recipes"]
    )
    backends = _choose_backends(
        service, minimum_qubits, backend_names, max_backends
    )
    deployments = []
    for backend, instance, usage in backends:
        backend_name = _backend_name(backend)
        pass_manager = generate_preset_pass_manager(
            optimization_level=3, backend=backend, seed_transpiler=seed_transpiler
        )
        for candidate in candidates:
            recipe_records = []
            total_depth = 0
            total_two_qubit = 0
            total_error = 0.0
            total_missing = 0
            for recipe in candidate["recipes"]:
                logical = _load_one_qpy(recipe["qpy_path"])
                _require(logical.num_qubits == len(recipe["readout_pauli"]),
                         "Readout Pauli width does not match logical circuit.")
                isa = pass_manager.run(logical)
                logical_observable = SparsePauliOp.from_list(
                    [(recipe["readout_pauli"], 1.0)]
                )
                isa_observable = logical_observable.apply_layout(isa.layout)
                filename = "{}_{}_{}.qpy".format(
                    _safe_name(candidate["candidate_id"]),
                    _safe_name(backend_name),
                    _safe_name(recipe["recipe_id"]),
                )
                isa_path = os.path.join(isa_directory, filename)
                isa_hash = _save_isa(isa_path, isa)
                resources = _isa_resources(backend, isa)
                total_depth += resources["isa_depth"]
                total_two_qubit += resources["isa_two_qubit_gates"]
                total_error += resources["two_qubit_error_sum"]
                total_missing += resources["two_qubit_errors_missing"]
                recipe_records.append({
                    "recipe_id": recipe["recipe_id"],
                    "component": recipe["component"],
                    "isa_qpy_path": os.path.abspath(isa_path),
                    "isa_qpy_sha256": isa_hash,
                    "isa_observable": [
                        [label, [float(complex(coefficient).real), float(complex(coefficient).imag)]]
                        for label, coefficient in isa_observable.to_list()
                    ],
                    "resources": resources,
                    "ideal_expectation": float(recipe["noiseless_expectation"]),
                })
            pending = _backend_pending_jobs(backend)
            score = (
                total_error
                + total_missing
                + 1.0e-4 * total_two_qubit
                + 1.0e-6 * total_depth
                + 1.0e-8 * pending
            )
            deployments.append({
                "candidate_id": candidate["candidate_id"],
                "sample_values": dict(candidate["sample_values"]),
                "D_ans": candidate["D_ans"],
                "N_T": candidate["N_T"],
                "ideal_observable_real": candidate["ideal_observable_real"],
                "ideal_observable_imag": candidate["ideal_observable_imag"],
                "algorithm_error_vs_scipy": candidate["algorithm_error_vs_scipy"],
                "candidate_class": candidate_class,
                "interpretation": candidate.get("interpretation"),
                "backend": backend_name,
                "backend_instance": instance["crn"],
                "backend_instance_name": instance.get("name"),
                "backend_instance_plan": instance.get("plan"),
                "instance_usage": {
                    key: usage.get(key) for key in (
                        "usage_limit_reached",
                        "usage_remaining_seconds",
                        "usage_consumed_seconds",
                        "usage_limit_seconds",
                        "usage_allocation_seconds",
                        "time_available_at",
                    )
                },
                "backend_num_qubits": int(backend.num_qubits),
                "backend_pending_jobs": pending,
                "backend_calibration_time": _backend_calibration_time(backend),
                "score": score,
                "error_metric_complete": total_missing == 0,
                "recipes": recipe_records,
            })
    deployments.sort(
        key=lambda item: (
            item["score"], item["D_ans"], item["N_T"], item["backend"]
        )
    )
    _require(deployments, "No hardware deployment could be transpiled.")
    deployments, pareto, selected = _screen_deployments(
        deployments,
        max_isa_depth_per_recipe=max_isa_depth_per_recipe,
        max_total_isa_two_qubit_gates=max_total_isa_two_qubit_gates,
        max_two_qubit_error_sum=max_two_qubit_error_sum,
        max_selections=max_selections,
    )
    if full_demo:
        candidate_ids = {item["candidate_id"] for item in deployments}
        complete_backends = []
        for backend_name in {item["backend"] for item in deployments}:
            group = [
                item for item in deployments
                if item["backend"] == backend_name
            ]
            if {item["candidate_id"] for item in group} == candidate_ids:
                complete_backends.append(group)
        if complete_backends:
            selected = min(
                complete_backends,
                key=lambda group: sum(float(item["score"]) for item in group),
            )
            selected.sort(key=lambda item: item["candidate_id"])
            pareto = list(selected)
        else:
            selected = []
    # Keep all evidence circuits until Agent 9 has chosen the actual deployment.
    discarded_isa_files = 0
    plan = {
        "hardware_plan_schema_version": HARDWARE_PLAN_SCHEMA_VERSION,
        "status": "awaiting_agent9_decision",
        "parameter_contract_sha256": crosscheck_manifest["parameter_contract_sha256"],
        "candidate_class": candidate_class,
        "plan_kind": (
            "full_demo_observable" if full_demo else
            "recipe_pilot" if candidate_class == "hardware_demonstration_candidate"
            else "full_observable"
        ),
        "full_observable_reconstruction_available": (
            candidate_class == "physics_hardware_candidate" or full_demo
        ),
        "source": metadata,
        "planning_policy": {
            "decision_authority": "agent9",
            "optimization_level": 3,
            "seed_transpiler": seed_transpiler,
            "max_backends": max_backends,
            "max_selections": max_selections,
            "automatic_demo_recipe_rule": (
                "largest absolute noiseless expectation"
                if candidate_class == "hardware_demonstration_candidate"
                and not full_demo else None
            ),
            "max_demo_recipes": max_demo_recipes,
            "demo_mode": demo_mode,
            "minimum_instance_remaining_seconds": MIN_INSTANCE_REMAINING_SECONDS,
            "score_rule": "error_sum + missing_error_count + 1e-4*n2q + 1e-6*depth + 1e-8*queue",
            "selection_rule": (
                "recommend one backend covering every D1_NT1 sample by lowest hardware score; Agent9 decides"
                if full_demo else
                "advisory Pareto shortlist only; Agent9 decides whether and what to submit"
            ),
            "resource_limits": {
                "maximum_isa_depth_per_recipe": max_isa_depth_per_recipe,
                "total_isa_two_qubit_gates": max_total_isa_two_qubit_gates,
                "maximum_recipe_two_qubit_error_sum": max_two_qubit_error_sum,
                "require_complete_two_qubit_errors": True,
            },
        },
        "all_deployments": deployments,
        "pareto_deployments": pareto,
        "selected_deployments": selected,
        "discarded_unselected_isa_files": discarded_isa_files,
    }
    if "campaign_id" in crosscheck_manifest:
        plan["campaign_id"] = crosscheck_manifest["campaign_id"]
    if "crosscheck_run_id" in crosscheck_manifest:
        plan["crosscheck_run_id"] = crosscheck_manifest["crosscheck_run_id"]
    if database_path is not None:
        _require(
            "campaign_id" in plan and "crosscheck_run_id" in plan,
            "Cross-check must be archived before database-backed hardware planning.",
        )
        plan["hardware_plan_id"] = record_hardware_plan(database_path, plan)
    _atomic_json_write(output_path, plan)
    if database_path is not None:
        record_hardware_plan(database_path, plan, plan_path=output_path)
    return plan


def audit_hardware_plan(
    plan: Mapping[str, Any], output_path: str, *,
    max_isa_depth_per_recipe: int = DEFAULT_MAX_ISA_DEPTH_PER_RECIPE,
    max_total_isa_two_qubit_gates: int = DEFAULT_MAX_TOTAL_ISA_TWO_QUBIT_GATES,
    max_two_qubit_error_sum: float = DEFAULT_MAX_TWO_QUBIT_ERROR_SUM,
    max_selections: int = 1,
    database_path: str | None = None,
) -> Dict[str, Any]:
    """Re-screen an existing transpiled plan without querying IBM or transpiling again."""
    _require(plan.get("hardware_plan_schema_version") == "1.0",
             "Hardware plan schema must be 1.0.")
    deployments = [dict(item) for item in plan.get("all_deployments", [])]
    _require(deployments, "Hardware plan contains no deployments to audit.")
    if any("algorithm_error_vs_scipy" not in item for item in deployments):
        crosscheck_path = plan.get("source", {}).get("crosscheck_path")
        _require(isinstance(crosscheck_path, str) and os.path.isfile(crosscheck_path),
                 "Legacy plan needs its cross-check file to recover accuracy metrics.")
        crosscheck = _load_json(crosscheck_path)
        by_id = {
            item["candidate_id"]: item["algorithm_error_vs_scipy"]
            for item in crosscheck.get("hardware_eligible_candidates", [])
        }
        for item in deployments:
            _require(item["candidate_id"] in by_id,
                     "Cross-check lacks accuracy for {}.".format(item["candidate_id"]))
            item["algorithm_error_vs_scipy"] = by_id[item["candidate_id"]]
    screened, pareto, selected = _screen_deployments(
        deployments,
        max_isa_depth_per_recipe=max_isa_depth_per_recipe,
        max_total_isa_two_qubit_gates=max_total_isa_two_qubit_gates,
        max_two_qubit_error_sum=max_two_qubit_error_sum,
        max_selections=max_selections,
    )
    audited = dict(plan)
    audited.pop("hardware_plan_id", None)
    audited["status"] = "awaiting_agent9_decision"
    policy = dict(plan.get("planning_policy", {}))
    policy.update({
        "max_selections": max_selections,
        "decision_authority": "agent9",
        "selection_rule": "advisory evidence only; Agent9 decides whether and what to submit",
        "resource_limits": {
            "maximum_isa_depth_per_recipe": max_isa_depth_per_recipe,
            "total_isa_two_qubit_gates": max_total_isa_two_qubit_gates,
            "maximum_recipe_two_qubit_error_sum": max_two_qubit_error_sum,
            "require_complete_two_qubit_errors": True,
        },
    })
    audited["planning_policy"] = policy
    audited["all_deployments"] = screened
    audited["pareto_deployments"] = pareto
    audited["selected_deployments"] = selected
    if database_path is not None:
        audited["hardware_plan_id"] = record_hardware_plan(database_path, audited)
    _atomic_json_write(output_path, audited)
    if database_path is not None:
        record_hardware_plan(database_path, audited, plan_path=output_path)
    return audited


def _observable_from_record(record: Sequence[Sequence[Any]]) -> SparsePauliOp:
    return SparsePauliOp.from_list([
        (label, complex(float(parts[0]), float(parts[1])))
        for label, parts in record
    ])


def submit_hardware_plan(
    plan: Mapping[str, Any],
    plan_path: str,
    agent9_decision: Mapping[str, Any],
    agent9_decision_path: str,
    service: QiskitRuntimeService,
    output_path: str,
    *,
    shots: int = 4096,
    resilience_level: int | None = None,
    database_path: str | None = None,
) -> Dict[str, Any]:
    _require(plan.get("hardware_plan_schema_version") == "1.0",
             "Hardware plan schema must be 1.0.")
    _require(agent9_decision.get("decision") == "submit",
             "Agent 9 did not approve hardware submission.")
    _require(isinstance(shots, int) and shots > 0, "shots must be positive.")
    if resilience_level is None:
        resilience_level = 1 if plan.get("plan_kind") == "recipe_pilot" else 2
    _require(resilience_level in (0, 1, 2),
             "resilience_level must be 0, 1, or 2.")
    requested = agent9_decision.get("selected_deployments")
    _require(isinstance(requested, list) and requested,
             "Agent 9 approved no hardware deployment.")
    available = {
        (item.get("candidate_id"), item.get("backend")): item
        for item in plan.get("all_deployments", [])
    }
    keys = []
    for item in requested:
        _require(isinstance(item, dict), "Invalid Agent 9 deployment selection.")
        key = (item.get("candidate_id"), item.get("backend"))
        _require(key in available and key not in keys,
                 "Agent 9 selected an unavailable or duplicate deployment.")
        keys.append(key)
    deployments = [available[key] for key in keys]
    if plan.get("plan_kind") == "full_demo_observable":
        _require(
            len({item["backend"] for item in deployments}) == 1
            and {item["candidate_id"] for item in deployments}
            == {item.get("candidate_id") for item in plan.get("all_deployments", [])},
            "Agent 9 must approve every requested sample on one backend for a full demo.",
        )
    submitted = []
    for deployment in deployments:
        instance = deployment.get("backend_instance")
        _require(
            isinstance(instance, str) and instance,
            "Hardware plan does not pin the selected IBM instance; rebuild the plan.",
        )
        backend = service.backend(deployment["backend"], instance=instance)
        usage = service.usage()
        remaining = usage.get("usage_remaining_seconds")
        _require(
            usage.get("usage_limit_reached") is not True
            and isinstance(remaining, (int, float))
            and remaining > MIN_INSTANCE_REMAINING_SECONDS,
            "Selected IBM instance does not have more than ten minutes of execution "
            "time remaining; rebuild the "
            "hardware plan to select another backend.",
        )
        status = backend.status()
        _require(
            bool(status.operational),
            "Selected backend {} is no longer operational; rebuild the hardware plan."
            .format(deployment["backend"]),
        )
        pubs = []
        index_map = []
        for recipe in deployment["recipes"]:
            _require(_sha256_file(recipe["isa_qpy_path"]) == recipe["isa_qpy_sha256"],
                     "ISA QPY changed after planning.")
            circuit = _load_one_qpy(recipe["isa_qpy_path"])
            observable = _observable_from_record(recipe["isa_observable"])
            pubs.append((circuit, observable))
            index_map.append(recipe["recipe_id"])
        options = EstimatorOptions(
            default_shots=shots, resilience_level=resilience_level
        )
        options.resilience.measure_mitigation = resilience_level >= 1
        options.resilience.zne_mitigation = resilience_level >= 2
        if resilience_level >= 2:
            options.resilience.zne.noise_factors = [1.0, 3.0, 5.0]
            options.resilience.zne.extrapolator = [
                "exponential", "linear", "fallback"
            ]
        options.environment.job_tags = [
            "Jianmu-Q", str(plan.get("plan_kind", "full_observable")),
            str(deployment["candidate_id"]),
        ]
        estimator = EstimatorV2(mode=backend, options=options)
        job = estimator.run(pubs)
        job_id = job.job_id()
        submitted.append({
            "candidate_id": deployment["candidate_id"],
            "sample_values": dict(deployment["sample_values"]),
            "D_ans": deployment["D_ans"],
            "N_T": deployment["N_T"],
            "ideal_observable_real": deployment["ideal_observable_real"],
            "ideal_observable_imag": deployment["ideal_observable_imag"],
            "backend": deployment["backend"],
            "backend_instance": instance,
            "backend_instance_name": deployment.get("backend_instance_name"),
            "instance_usage_at_submission": {
                key: usage.get(key) for key in (
                    "usage_limit_reached",
                    "usage_remaining_seconds",
                    "usage_consumed_seconds",
                    "usage_limit_seconds",
                    "usage_allocation_seconds",
                    "time_available_at",
                )
            },
            "backend_calibration_time": deployment["backend_calibration_time"],
            "job_id": job_id,
            "recipe_index_map": index_map,
            "shots": shots,
            "candidate_class": deployment.get("candidate_class"),
            "interpretation": deployment.get("interpretation"),
            "ideal_recipe_expectations": {
                recipe["recipe_id"]: recipe["ideal_expectation"]
                for recipe in deployment["recipes"]
            },
        })
    record = {
        "hardware_run_schema_version": HARDWARE_RUN_SCHEMA_VERSION,
        "status": "submitted",
        "plan_path": os.path.abspath(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "agent9_decision_path": os.path.abspath(agent9_decision_path),
        "agent9_decision_sha256": _sha256_file(agent9_decision_path),
        "parameter_contract_sha256": plan["parameter_contract_sha256"],
        "observable_adapter_path": plan["source"]["observable_adapter_path"],
        "observable_adapter_sha256": plan["source"]["observable_adapter_sha256"],
        "run_kind": plan.get("plan_kind", "full_observable"),
        "candidate_class": plan.get("candidate_class", "physics_hardware_candidate"),
        "resilience": {
            "level": resilience_level,
            "measurement_mitigation": resilience_level >= 1,
            "zne_mitigation": resilience_level >= 2,
            "zne_noise_factors": (
                [1.0, 3.0, 5.0] if resilience_level >= 2 else None
            ),
            "zne_extrapolator": (
                ["exponential", "linear", "fallback"]
                if resilience_level >= 2 else None
            ),
        },
        "jobs": submitted,
    }
    for key in ("campaign_id", "crosscheck_run_id", "hardware_plan_id"):
        if key in plan:
            record[key] = plan[key]
    if database_path is not None:
        _require(
            all(key in record for key in ("campaign_id", "hardware_plan_id")),
            "Hardware plan must be archived before database-backed submission.",
        )
        record["hardware_run_id"] = record_hardware_run(database_path, record)
    _atomic_json_write(output_path, record)
    if database_path is not None:
        record_hardware_run(
            database_path, record, run_record_path=output_path
        )
    return record


def _load_observable_adapter(path: str, expected_hash: str) -> Any:
    _require(os.path.isfile(path) and _sha256_file(path) == expected_hash,
             "Observable adapter changed after validation.")
    spec = importlib.util.spec_from_file_location("hardware_observable_adapter", path)
    _require(spec is not None and spec.loader is not None,
             "Cannot import observable adapter.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    _require(array.size == 1, "Estimator returned a non-scalar recipe result.")
    result = float(array.reshape(-1)[0])
    _require(math.isfinite(result), "Estimator returned a non-finite result.")
    return result


def _estimator_recipe_value(pub_result: Any) -> Tuple[float, float, str]:
    """Use stable ZNE; otherwise retain the measured scale-1 result."""
    try:
        zne = (
            _scalar(pub_result.data.evs),
            _scalar(pub_result.data.stds),
        )
    except HardwareRuntimeError:
        zne = None
    values = np.asarray(
        getattr(pub_result.data, "evs_noise_factors", []), dtype=float
    ).reshape(-1)
    errors = np.asarray(
        getattr(pub_result.data, "stds_noise_factors", []), dtype=float
    ).reshape(-1)
    if not values.size or errors.size != values.size:
        _require(zne is not None, "Estimator returned no usable result.")
        return zne[0], zne[1], "zne_extrapolated"
    _require(math.isfinite(float(values[0])) and math.isfinite(float(errors[0])),
             "Estimator returned neither finite ZNE nor finite scale-1 data.")
    scale_one = (float(values[0]), float(errors[0]))
    if zne is None:
        return scale_one[0], scale_one[1], "noise_factor_1_fallback_nonfinite_zne"
    if zne[1] > MAX_ZNE_STANDARD_ERROR_RATIO * max(scale_one[1], 1.0e-12):
        return scale_one[0], scale_one[1], "noise_factor_1_fallback_unstable_zne"
    return zne[0], zne[1], "zne_extrapolated"


def _propagate_reconstruction_uncertainty(
    adapter: Any, expectations: Mapping[str, float], standard_errors: Mapping[str, float],
    sample_values: Mapping[str, Any],
) -> Tuple[float, float]:
    real_variance = 0.0
    imag_variance = 0.0
    for recipe_id, value in expectations.items():
        step = max(1.0e-6, standard_errors[recipe_id])
        plus = dict(expectations)
        minus = dict(expectations)
        plus[recipe_id] = value + step
        minus[recipe_id] = value - step
        derivative = (
            complex(adapter.reconstruct_observable(plus, dict(sample_values)))
            - complex(adapter.reconstruct_observable(minus, dict(sample_values)))
        ) / (2.0 * step)
        real_variance += (derivative.real * standard_errors[recipe_id]) ** 2
        imag_variance += (derivative.imag * standard_errors[recipe_id]) ** 2
    return math.sqrt(real_variance), math.sqrt(imag_variance)


def retrieve_hardware_results(
    run_record: Mapping[str, Any],
    run_record_path: str,
    service: QiskitRuntimeService,
    output_path: str,
    *,
    database_path: str | None = None,
) -> Dict[str, Any]:
    _require(run_record.get("hardware_run_schema_version") == "1.0",
             "Hardware run schema must be 1.0.")
    pilot = run_record.get("run_kind") == "recipe_pilot"
    adapter = None if pilot else _load_observable_adapter(
        run_record["observable_adapter_path"],
        run_record["observable_adapter_sha256"],
    )
    results = []
    for job_record in run_record["jobs"]:
        job = service.job(job_record["job_id"])
        primitive_result = job.result()
        _require(len(primitive_result) == len(job_record["recipe_index_map"]),
                 "Estimator result count does not match recipe index map.")
        expectations = {}
        standard_errors = {}
        mitigation_status = {}
        noise_factor_data = {}
        for recipe_id, pub_result in zip(
            job_record["recipe_index_map"], primitive_result
        ):
            value, error, status = _estimator_recipe_value(pub_result)
            expectations[recipe_id] = value
            standard_errors[recipe_id] = error
            mitigation_status[recipe_id] = status
            raw_values = np.asarray(
                getattr(pub_result.data, "evs_noise_factors", []), dtype=float
            ).reshape(-1)
            raw_errors = np.asarray(
                getattr(pub_result.data, "stds_noise_factors", []), dtype=float
            ).reshape(-1)
            if raw_values.size and raw_values.size == raw_errors.size:
                noise_factor_data[recipe_id] = {
                    "noise_factors": run_record.get("resilience", {}).get(
                        "zne_noise_factors"
                    ),
                    "expectations": [
                        float(item) if math.isfinite(float(item)) else None
                        for item in raw_values
                    ],
                    "standard_errors": [
                        float(item) if math.isfinite(float(item)) else None
                        for item in raw_errors
                    ],
                }
        if pilot:
            recipe_results = []
            ideals = job_record["ideal_recipe_expectations"]
            for recipe_id in job_record["recipe_index_map"]:
                difference = expectations[recipe_id] - float(ideals[recipe_id])
                sigma = standard_errors[recipe_id]
                recipe_results.append({
                    "recipe_id": recipe_id,
                    "ideal_noiseless_expectation": float(ideals[recipe_id]),
                    "hardware_expectation": expectations[recipe_id],
                    "hardware_standard_error": sigma,
                    "hardware_minus_noiseless": difference,
                    "within_two_standard_errors": abs(difference) <= 2.0 * sigma,
                    "mitigation_status": mitigation_status[recipe_id],
                    "noise_factor_data": noise_factor_data.get(recipe_id),
                })
            results.append({
                "candidate_id": job_record["candidate_id"],
                "D_ans": job_record["D_ans"],
                "N_T": job_record["N_T"],
                "backend": job_record["backend"],
                "job_id": job_record["job_id"],
                "interpretation": "same_circuit_noiseless_comparison_only",
                "recipe_results": recipe_results,
            })
            continue
        sample_values = dict(job_record["sample_values"])
        observable = complex(adapter.reconstruct_observable(
            dict(expectations), sample_values
        ))
        sigma_real, sigma_imag = _propagate_reconstruction_uncertainty(
            adapter, expectations, standard_errors, sample_values
        )
        ideal = complex(
            job_record["ideal_observable_real"], job_record["ideal_observable_imag"]
        )
        results.append({
            "candidate_id": job_record["candidate_id"],
            "D_ans": job_record["D_ans"],
            "N_T": job_record["N_T"],
            "backend": job_record["backend"],
            "job_id": job_record["job_id"],
            "sample_values": sample_values,
            "expectations": expectations,
            "standard_errors": standard_errors,
            "mitigation_status": mitigation_status,
            "noise_factor_data": noise_factor_data,
            "hardware_observable_real": observable.real,
            "hardware_observable_imag": observable.imag,
            "hardware_standard_error_real": sigma_real,
            "hardware_standard_error_imag": sigma_imag,
            "ideal_observable_real": ideal.real,
            "ideal_observable_imag": ideal.imag,
            "hardware_error_real": observable.real - ideal.real,
            "hardware_error_imag": observable.imag - ideal.imag,
        })
    output = {
        "hardware_result_schema_version": "1.0",
        "status": "retrieved",
        "run_record_path": os.path.abspath(run_record_path),
        "run_record_sha256": _sha256_file(run_record_path),
        "parameter_contract_sha256": run_record["parameter_contract_sha256"],
        "resilience": run_record.get("resilience", {}),
        "results": results,
    }
    for key in (
        "campaign_id", "crosscheck_run_id", "hardware_plan_id", "hardware_run_id"
    ):
        if key in run_record:
            output[key] = run_record[key]
    if database_path is not None and not pilot:
        _require(
            all(key in output for key in ("campaign_id", "hardware_run_id")),
            "Hardware run must be archived before database-backed retrieval.",
        )
    _atomic_json_write(output_path, output)
    if database_path is not None and not pilot:
        record_hardware_results(
            database_path, run_record, output, result_path=output_path
        )
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IBM Quantum hardware runtime.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--crosscheck", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--isa-directory", required=True)
    plan.add_argument("--backend", action="append", default=[])
    plan.add_argument(
        "--candidate-class",
        choices=("hardware_demonstration_candidate", "physics_hardware_candidate"),
        default="hardware_demonstration_candidate",
    )
    plan.add_argument("--max-demo-recipes", type=int, default=2)
    plan.add_argument(
        "--demo-mode", choices=("representative", "full_d1_nt1"),
        default="representative",
    )
    plan.add_argument("--max-backends", type=int, default=5)
    plan.add_argument("--max-selections", type=int, default=1)
    plan.add_argument("--seed-transpiler", type=int, default=17)
    plan.add_argument("--max-isa-depth-per-recipe", type=int,
                      default=DEFAULT_MAX_ISA_DEPTH_PER_RECIPE)
    plan.add_argument("--max-total-isa-two-qubit-gates", type=int,
                      default=DEFAULT_MAX_TOTAL_ISA_TWO_QUBIT_GATES)
    plan.add_argument("--max-two-qubit-error-sum", type=float,
                      default=DEFAULT_MAX_TWO_QUBIT_ERROR_SUM)
    plan.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--plan", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--max-selections", type=int, default=1)
    audit.add_argument("--max-isa-depth-per-recipe", type=int,
                       default=DEFAULT_MAX_ISA_DEPTH_PER_RECIPE)
    audit.add_argument("--max-total-isa-two-qubit-gates", type=int,
                       default=DEFAULT_MAX_TOTAL_ISA_TWO_QUBIT_GATES)
    audit.add_argument("--max-two-qubit-error-sum", type=float,
                       default=DEFAULT_MAX_TWO_QUBIT_ERROR_SUM)
    audit.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--plan", required=True)
    submit.add_argument("--agent9-decision", required=True)
    submit.add_argument("--output", required=True)
    submit.add_argument("--shots", type=int, default=4096)
    submit.add_argument("--resilience-level", type=int, choices=(0, 1, 2))
    submit.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--run-record", required=True)
    retrieve.add_argument("--output", required=True)
    retrieve.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    for child in (plan, audit, submit, retrieve):
        child.add_argument(
            "--pipeline-manifest",
            help="Register the output path in an existing pipeline manifest.",
        )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.mode == "plan":
        service = QiskitRuntimeService()
        manifest = _load_json(args.crosscheck)
        result = build_hardware_plan(
            manifest,
            args.crosscheck,
            service,
            args.output,
            args.isa_directory,
            candidate_class=args.candidate_class,
            max_demo_recipes=args.max_demo_recipes,
            demo_mode=args.demo_mode,
            backend_names=args.backend,
            max_backends=args.max_backends,
            max_selections=args.max_selections,
            seed_transpiler=args.seed_transpiler,
            max_isa_depth_per_recipe=args.max_isa_depth_per_recipe,
            max_total_isa_two_qubit_gates=args.max_total_isa_two_qubit_gates,
            max_two_qubit_error_sum=args.max_two_qubit_error_sum,
            database_path=args.database,
        )
        print("Hardware plan status {}; {} selection(s); nothing submitted.".format(
            result["status"], len(result["selected_deployments"])))
    elif args.mode == "audit":
        result = audit_hardware_plan(
            _load_json(args.plan), args.output,
            max_isa_depth_per_recipe=args.max_isa_depth_per_recipe,
            max_total_isa_two_qubit_gates=args.max_total_isa_two_qubit_gates,
            max_two_qubit_error_sum=args.max_two_qubit_error_sum,
            max_selections=args.max_selections,
            database_path=args.database,
        )
        print("Audited hardware plan status {}; {} selection(s).".format(
            result["status"], len(result["selected_deployments"])))
    elif args.mode == "submit":
        service = QiskitRuntimeService()
        result = submit_hardware_plan(
            _load_json(args.plan), args.plan,
            _load_json(args.agent9_decision), args.agent9_decision,
            service, args.output, shots=args.shots,
            resilience_level=args.resilience_level,
            database_path=args.database,
        )
        print("Submitted {} IBM Runtime job(s).".format(len(result["jobs"])))
    else:
        service = QiskitRuntimeService()
        result = retrieve_hardware_results(
            _load_json(args.run_record), args.run_record, service, args.output,
            database_path=args.database,
        )
        print("Retrieved {} hardware result(s).".format(len(result["results"])))
    if args.pipeline_manifest:
        from manifest_artifact import register_external_artifact

        names = {
            "plan": "hardware_plan",
            "audit": "hardware_audit",
            "submit": "hardware_run",
            "retrieve": "hardware_result",
        }
        register_external_artifact(
            args.pipeline_manifest,
            logical_name=names[args.mode],
            payload_path=args.output,
        )


if __name__ == "__main__":
    main()
