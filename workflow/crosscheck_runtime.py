# -*- coding: utf-8 -*-
"""Deterministic classical agreement, convergence, and candidate selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from vqe_experience_store import DEFAULT_DATABASE_PATH, record_crosscheck_result
from validation_policy import provenance_warning


CROSSCHECK_SCHEMA_VERSION = "1.0"


class CrosscheckRuntimeError(RuntimeError):
    """Raised when reference or measurement records cannot be compared safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrosscheckRuntimeError(message)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), "Expected a JSON object: {}".format(path))
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
        prefix=".crosscheck_runtime_", suffix=".tmp", dir=directory
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


def _finite_number(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             "{} must be numeric.".format(name))
    value = float(value)
    _require(math.isfinite(value), "{} must be finite.".format(name))
    return value


def validate_reference(
    value: Mapping[str, Any],
    expected_method: str,
    parameter_digest: str | None,
) -> Dict[str, Any]:
    _require(value.get("reference_schema_version") == "1.0",
             "Reference schema must be 1.0.")
    _require(value.get("method") == expected_method,
             "Expected {} reference.".format(expected_method))
    provenance_warning(
        "parameter_contract_hash_mismatch",
        "Reference provenance hash differs; numerical grids and results will still be cross-checked.",
        parameter_digest,
        value.get("parameter_contract_sha256"),
    )
    result = dict(value)
    for key in ("ground_state_energy", "energy_variance"):
        result[key] = _finite_number(value.get(key), "reference.{}".format(key))
    _require(result["energy_variance"] >= 0.0,
             "reference.energy_variance must be nonnegative.")
    raw_samples = value.get("observable_samples")
    if raw_samples is None:
        legacy_sample = {
            "sample_id": "default",
            "sample_values": {},
        }
        legacy_sample.update({
            key: value.get(key) for key in (
                "observable_real", "observable_imag",
                "uncertainty_real", "uncertainty_imag",
            )
        })
        raw_samples = [legacy_sample]
    _require(isinstance(raw_samples, list) and raw_samples,
             "Reference observable_samples must be a nonempty list.")
    samples = []
    seen = set()
    for raw in raw_samples:
        _require(isinstance(raw, dict), "Every reference sample must be an object.")
        sample_id = raw.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id and sample_id not in seen,
                 "Reference sample IDs must be unique nonempty strings.")
        seen.add(sample_id)
        sample_values = raw.get("sample_values")
        _require(isinstance(sample_values, dict), "Reference sample_values must be an object.")
        sample = {"sample_id": sample_id, "sample_values": sample_values}
        for key in (
            "observable_real", "observable_imag", "uncertainty_real", "uncertainty_imag"
        ):
            sample[key] = _finite_number(raw.get(key), "reference.{}".format(key))
        for key in ("uncertainty_real", "uncertainty_imag"):
            _require(sample[key] >= 0.0, "reference.{} must be nonnegative.".format(key))
        samples.append(sample)
    result["observable_samples"] = samples
    # Retain the first sample in legacy scalar columns while the complete list
    # remains authoritative in observable_samples/result_json.
    for key in (
        "observable_real", "observable_imag", "uncertainty_real", "uncertainty_imag"
    ):
        result[key] = samples[0][key]
    convergence = value.get("convergence")
    _require(
        isinstance(convergence, dict)
        and isinstance(convergence.get("passed"), bool)
        and isinstance(convergence.get("details"), dict),
        "Reference convergence must contain passed and details.",
    )
    return result


def _complex_from(record: Mapping[str, Any], prefix: str = "observable") -> complex:
    return complex(
        _finite_number(record.get(prefix + "_real"), prefix + "_real"),
        _finite_number(record.get(prefix + "_imag"), prefix + "_imag"),
    )


def _validate_measurement(measurement: Mapping[str, Any]) -> List[Dict[str, Any]]:
    _require(measurement.get("measurement_schema_version") == "1.0",
             "Measurement schema must be 1.0.")
    grid = measurement.get("validation_grid")
    _require(isinstance(grid, list) and grid, "Measurement validation grid is empty.")
    seen = set()
    validated = []
    for raw in grid:
        _require(isinstance(raw, dict), "Every grid point must be an object.")
        point = dict(raw)
        point.setdefault("sample_id", "default")
        point.setdefault("sample_values", {})
        _require(isinstance(point["sample_id"], str) and point["sample_id"],
                 "Grid sample_id must be a nonempty string.")
        _require(isinstance(point["sample_values"], dict),
                 "Grid sample_values must be an object.")
        candidate_id = point.get("candidate_id")
        _require(isinstance(candidate_id, str) and candidate_id and candidate_id not in seen,
                 "Grid candidate IDs must be unique nonempty strings.")
        seen.add(candidate_id)
        for key in ("D_ans", "N_T"):
            _require(isinstance(point.get(key), int) and not isinstance(point[key], bool),
                     "{} must be an integer.".format(key))
        _require(point["D_ans"] > 0 and point["N_T"] >= 0,
                 "D_ans must be positive and N_T nonnegative.")
        _require(isinstance(point.get("state_preparation_validated"), bool),
                 "Grid point lacks state-preparation status.")
        point["observable_real"] = _finite_number(
            point.get("observable_real"), "grid.observable_real"
        )
        point["observable_imag"] = _finite_number(
            point.get("observable_imag"), "grid.observable_imag"
        )
        recipes = point.get("recipes")
        _require(isinstance(recipes, list) and recipes, "Grid point has no QPY recipes.")
        for recipe in recipes:
            path = recipe.get("qpy_path")
            digest_value = recipe.get("qpy_sha256")
            _require(isinstance(path, str) and os.path.isfile(path),
                     "Missing QPY recipe file: {}".format(path))
            _require(_sha256_file(path) == digest_value,
                     "QPY hash mismatch: {}".format(path))
        validated.append(point)
    coordinates = [
        (point["sample_id"], point["D_ans"], point["N_T"])
        for point in validated
    ]
    _require(len(coordinates) == len(set(coordinates)),
             "Measurement grid contains duplicate coordinates.")
    return validated


def _series_diagnostics(
    points: Sequence[Mapping[str, Any]],
    fixed_key: str,
    scan_key: str,
    tail_tolerance: float,
    time_evolution_required: bool,
) -> Dict[str, Any]:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for point in points:
        groups.setdefault(
            (str(point.get("sample_id", "default")), int(point[fixed_key])), []
        ).append(point)
    records = []
    for (sample_id, fixed), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item[scan_key])
        values = [_complex_from(item) for item in ordered]
        deltas = [abs(right - left) for left, right in zip(values, values[1:])]
        sufficient = len(ordered) >= 2
        stable = sufficient and deltas[-1] <= tail_tolerance
        records.append({
            "sample_id": sample_id,
            "fixed_value": fixed,
            "scan_values": [item[scan_key] for item in ordered],
            "adjacent_deltas": deltas,
            "sufficient_points": sufficient,
            "tail_stable": stable,
        })
    applicable = not (scan_key == "N_T" and not time_evolution_required)
    passed = True if not applicable else bool(records) and all(
        record["tail_stable"] for record in records
    )
    return {"applicable": applicable, "passed": passed, "series": records}


def _resource_score(point: Mapping[str, Any]) -> Tuple[int, int, int, int, int, int]:
    resources = [recipe["resources"] for recipe in point["recipes"]]
    maximum_depth = max(int(item["logical_depth"]) for item in resources)
    total_two_qubit = sum(int(item["logical_two_qubit_gates"]) for item in resources)
    maximum_qubits = max(int(item["num_qubits"]) for item in resources)
    return (
        maximum_depth,
        total_two_qubit,
        maximum_qubits,
        int(point["D_ans"]),
        int(point["N_T"]),
    )


def run_crosscheck(
    measurement: Mapping[str, Any],
    scipy_reference: Mapping[str, Any],
    itensor_reference: Mapping[str, Any],
    output_path: str,
    *,
    measurement_path: str | None = None,
    scipy_path: str | None = None,
    itensor_path: str | None = None,
    classical_tolerance: float = 1.0e-4,
    algorithm_tolerance: float = 1.0e-3,
    tail_tolerance: float = 1.0e-3,
    max_hardware_candidates: int = 4,
    database_path: str | None = None,
) -> Dict[str, Any]:
    for name, value in (
        ("classical_tolerance", classical_tolerance),
        ("algorithm_tolerance", algorithm_tolerance),
        ("tail_tolerance", tail_tolerance),
    ):
        _require(isinstance(value, (int, float)) and value > 0,
                 "{} must be positive.".format(name))
    _require(isinstance(max_hardware_candidates, int) and max_hardware_candidates > 0,
             "max_hardware_candidates must be positive.")

    grid = _validate_measurement(measurement)
    parameter_digest = measurement.get("parameter_contract_sha256")
    scipy = validate_reference(scipy_reference, "scipy_exact", parameter_digest)
    itensor = validate_reference(itensor_reference, "itensor_mps", parameter_digest)
    observable_digest = measurement.get("observable_specification_sha256")
    provenance_warning(
        "observable_specification_hash_mismatch",
        "SciPy observable provenance differs; sample definitions and numerical results will still be checked.",
        observable_digest,
        scipy.get("observable_specification_sha256"),
    )
    provenance_warning(
        "observable_specification_hash_mismatch",
        "ITensor observable provenance differs; sample definitions and numerical results will still be checked.",
        observable_digest,
        itensor.get("observable_specification_sha256"),
    )
    scipy_by_sample = {item["sample_id"]: item for item in scipy["observable_samples"]}
    itensor_by_sample = {item["sample_id"]: item for item in itensor["observable_samples"]}
    measurement_values = {}
    for point in grid:
        previous = measurement_values.setdefault(point["sample_id"], point["sample_values"])
        _require(previous == point["sample_values"],
                 "Measurement reused a sample_id with different values.")
    _require(set(measurement_values) == set(scipy_by_sample) == set(itensor_by_sample),
             "Measurement, SciPy and ITensor sample IDs differ.")
    classical_sample_records = []
    for sample_id in sorted(measurement_values):
        scipy_sample = scipy_by_sample[sample_id]
        itensor_sample = itensor_by_sample[sample_id]
        _require(
            scipy_sample["sample_values"] == measurement_values[sample_id]
            and itensor_sample["sample_values"] == measurement_values[sample_id],
            "Classical reference sample values differ from the measurement request.",
        )
        allowance = math.hypot(
            scipy_sample["uncertainty_real"] + itensor_sample["uncertainty_real"],
            scipy_sample["uncertainty_imag"] + itensor_sample["uncertainty_imag"],
        )
        difference = abs(_complex_from(scipy_sample) - _complex_from(itensor_sample))
        classical_sample_records.append({
            "sample_id": sample_id,
            "sample_values": measurement_values[sample_id],
            "observable_difference": difference,
            "combined_observable_uncertainty": allowance,
            "passed": difference <= classical_tolerance + allowance,
        })
    observable_difference = max(
        item["observable_difference"] for item in classical_sample_records
    )
    uncertainty_allowance = max(
        item["combined_observable_uncertainty"] for item in classical_sample_records
    )
    energy_difference = abs(
        scipy["ground_state_energy"] - itensor["ground_state_energy"]
    )
    classical_agreement = (
        scipy["convergence"]["passed"]
        and itensor["convergence"]["passed"]
        and all(item["passed"] for item in classical_sample_records)
        and energy_difference <= classical_tolerance
    )

    point_diagnostics = []
    for point in grid:
        scipy_observable = _complex_from(scipy_by_sample[point["sample_id"]])
        error = abs(_complex_from(point) - scipy_observable)
        point_diagnostics.append({
            "candidate_id": point["candidate_id"],
            "sample_id": point["sample_id"],
            "sample_values": point["sample_values"],
            "D_ans": point["D_ans"],
            "N_T": point["N_T"],
            "algorithm_error_vs_scipy": error,
            "within_algorithm_tolerance": error <= algorithm_tolerance,
            "state_preparation_validated": point["state_preparation_validated"],
        })

    time_evolution_required = measurement.get("trotter_schedule") != [0]
    depth_convergence = _series_diagnostics(
        grid, "N_T", "D_ans", tail_tolerance, time_evolution_required
    )
    trotter_convergence = _series_diagnostics(
        grid, "D_ans", "N_T", tail_tolerance, time_evolution_required
    )
    endpoints = []
    for sample_id in sorted(measurement_values):
        sample_grid = [point for point in grid if point["sample_id"] == sample_id]
        endpoint = max(sample_grid, key=lambda point: (point["D_ans"], point["N_T"]))
        endpoint_error = abs(
            _complex_from(endpoint) - _complex_from(scipy_by_sample[sample_id])
        )
        endpoints.append({
            "sample_id": sample_id,
            "candidate_id": endpoint["candidate_id"],
            "algorithm_error_vs_scipy": endpoint_error,
            "within_algorithm_tolerance": endpoint_error <= algorithm_tolerance,
        })
    overall_passed = (
        classical_agreement
        and depth_convergence["passed"]
        and trotter_convergence["passed"]
        and all(item["within_algorithm_tolerance"] for item in endpoints)
    )

    error_by_id = {
        item["candidate_id"]: item["algorithm_error_vs_scipy"]
        for item in point_diagnostics
    }
    eligible = []
    if overall_passed:
        eligible = [
            dict(point)
            for point in grid
            if point["state_preparation_validated"]
            and error_by_id[point["candidate_id"]] <= algorithm_tolerance
        ]
        eligible.sort(key=_resource_score)
        eligible = eligible[:max_hardware_candidates]
        for point in eligible:
            point["ideal_observable_real"] = point["observable_real"]
            point["ideal_observable_imag"] = point["observable_imag"]
            point["algorithm_error_vs_scipy"] = error_by_id[point["candidate_id"]]
            point["candidate_class"] = "physics_hardware_candidate"
            point["interpretation"] = "physics_observable_candidate"

    # Retain the shallowest scanned circuit for every observable sample even
    # when physics validation fails.  These records support only a real-device
    # versus identical-noiseless-circuit demonstration and are never promoted
    # to physics-accurate candidates by this fallback.
    demonstration = []
    for sample_id in sorted(measurement_values):
        sample_grid = [point for point in grid if point["sample_id"] == sample_id]
        shallowest = min(
            sample_grid, key=lambda point: (point["D_ans"], point["N_T"])
        )
        demo = dict(shallowest)
        demo["ideal_observable_real"] = demo["observable_real"]
        demo["ideal_observable_imag"] = demo["observable_imag"]
        demo["algorithm_error_vs_scipy"] = error_by_id[demo["candidate_id"]]
        demo["candidate_class"] = "hardware_demonstration_candidate"
        demo["interpretation"] = "same_circuit_noiseless_comparison_only"
        demonstration.append(demo)

    result = {
        "crosscheck_schema_version": CROSSCHECK_SCHEMA_VERSION,
        "status": "approved" if overall_passed and eligible else "not_approved",
        "parameter_contract_sha256": parameter_digest,
        "observable_specification_sha256": observable_digest,
        "source_files": {
            "measurement": measurement_path,
            "measurement_sha256": _sha256_file(measurement_path) if measurement_path else None,
            "scipy": scipy_path,
            "scipy_sha256": _sha256_file(scipy_path) if scipy_path else None,
            "itensor": itensor_path,
            "itensor_sha256": _sha256_file(itensor_path) if itensor_path else None,
        },
        "thresholds": {
            "classical_tolerance": classical_tolerance,
            "algorithm_tolerance": algorithm_tolerance,
            "tail_tolerance": tail_tolerance,
            "max_hardware_candidates": max_hardware_candidates,
        },
        "classical_agreement": {
            "passed": classical_agreement,
            "observable_difference": observable_difference,
            "energy_difference": energy_difference,
            "combined_observable_uncertainty": uncertainty_allowance,
            "scipy_converged": scipy["convergence"]["passed"],
            "itensor_converged": itensor["convergence"]["passed"],
            "samples": classical_sample_records,
        },
        "reference": {
            "observable_real": scipy["observable_samples"][0]["observable_real"],
            "observable_imag": scipy["observable_samples"][0]["observable_imag"],
            "ground_state_energy": scipy["ground_state_energy"],
        },
        "references": scipy["observable_samples"],
        "depth_convergence": depth_convergence,
        "trotter_convergence": trotter_convergence,
        "endpoint": endpoints[0],
        "endpoints": endpoints,
        "point_diagnostics": point_diagnostics,
        "hardware_eligible_candidates": eligible,
        "hardware_demonstration_candidates": demonstration,
    }
    if database_path is not None:
        _require(
            isinstance(measurement.get("campaign_id"), str)
            and isinstance(measurement.get("measurement_run_id"), str),
            "Measurement must be archived before database-backed cross-check.",
        )
        identities = record_crosscheck_result(
            database_path,
            measurement,
            scipy,
            itensor,
            result,
            scipy_path=scipy_path,
            itensor_path=itensor_path,
        )
        result.update(identities)
    _atomic_json_write(output_path, result)
    if database_path is not None:
        record_crosscheck_result(
            database_path,
            measurement,
            scipy,
            itensor,
            result,
            result_path=output_path,
            scipy_path=scipy_path,
            itensor_path=itensor_path,
        )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-check classical references and logical convergence."
    )
    parser.add_argument("--measurement", required=True)
    parser.add_argument("--scipy", required=True)
    parser.add_argument("--itensor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classical-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--algorithm-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--tail-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-hardware-candidates", type=int, default=4)
    parser.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result = run_crosscheck(
        _load_json(args.measurement),
        _load_json(args.scipy),
        _load_json(args.itensor),
        args.output,
        measurement_path=args.measurement,
        scipy_path=args.scipy,
        itensor_path=args.itensor,
        classical_tolerance=args.classical_tolerance,
        algorithm_tolerance=args.algorithm_tolerance,
        tail_tolerance=args.tail_tolerance,
        max_hardware_candidates=args.max_hardware_candidates,
        database_path=args.database,
    )
    print(
        "Cross-check {} with {} physics candidate(s) and {} automatic demo(s).".format(
            result["status"], len(result["hardware_eligible_candidates"]),
            len(result["hardware_demonstration_candidates"]),
        )
    )


if __name__ == "__main__":
    main()
