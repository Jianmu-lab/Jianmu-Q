#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp


METHOD_ID = "scipy_exact"
REFERENCE_SCHEMA_VERSION = "1.0"
EXPECTED_STATE_LABELS = {"vacuum", "glueball"}


def fail(message):
    raise RuntimeError(message)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path, description):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except Exception as exc:
        fail(f"Could not load {description} from {path!r}: {exc}")
    if not isinstance(value, dict):
        fail(f"{description} must be one JSON object")
    return value


def require(condition, message):
    if not condition:
        fail(message)


def require_dict(value, path):
    require(isinstance(value, dict), f"{path} must be an object")
    return value


def require_list(value, path):
    require(isinstance(value, list), f"{path} must be an array")
    return value


def finite_real(value, path):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path} must be a real number",
    )
    result = float(value)
    require(math.isfinite(result), f"{path} must be finite")
    return result


def find_unique(items, predicate, description):
    matches = [item for item in items if isinstance(item, dict) and predicate(item)]
    require(len(matches) == 1, f"Expected exactly one {description}, found {len(matches)}")
    return matches[0]


def parameter_by_name(contract, name):
    parameters = require_list(contract.get("parameters"), "parameters")
    return find_unique(
        parameters,
        lambda item: item.get("name") == name,
        f"parameter named {name!r}",
    )


def validate_parameter_contract(contract):
    require(contract.get("schema_version") == "1.1",
            "Unsupported Parameter Contract schema_version")

    model = require_dict(contract.get("model"), "model")
    require(model.get("name") ==
            "2+1 dimensional pure Z2 gauge theory in Kogut-Susskind formalism",
            "Unsupported model")
    require(model.get("gauge_structure") == "local Z2",
            "Unsupported gauge structure")
    require(model.get("spacetime_dimension") == "2+1",
            "Unsupported spacetime dimension")

    g_parameter = parameter_by_name(contract, "bare_coupling_g")
    a_parameter = parameter_by_name(contract, "lattice_spacing_a")
    g = finite_real(g_parameter.get("selected_value"),
                    "bare_coupling_g.selected_value")
    a = finite_real(a_parameter.get("selected_value"),
                    "lattice_spacing_a.selected_value")
    require(g > 0.0, "bare_coupling_g must be strictly positive")
    require(a > 0.0, "lattice_spacing_a must be strictly positive")

    immutable_g = parameter_by_name(contract, "bare copuling: g")
    require(immutable_g.get("immutable") is True,
            "The immutable bare-coupling input is not marked immutable")
    require(immutable_g.get("source_ref") ==
            "/user_input/parameters/bare copuling: g",
            "Unexpected immutable bare-coupling source_ref")
    immutable_g_text = immutable_g.get("selected_value")
    require(isinstance(immutable_g_text, str),
            "Immutable bare-coupling input must be a string")
    match = re.fullmatch(
        r"\s*g\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*",
        immutable_g_text,
    )
    require(match is not None,
            "Immutable bare-coupling input must have the form 'g = NUMBER'")
    immutable_g_value = float(match.group(1))
    require(math.isfinite(immutable_g_value) and immutable_g_value > 0.0,
            "Immutable bare-coupling value must be finite and positive")
    require(immutable_g_value == g,
            "Numerical bare_coupling_g contradicts the immutable user input")

    immutable_shape = parameter_by_name(
        contract, "number of lattice site [L,L]:"
    )
    require(immutable_shape.get("immutable") is True,
            "The immutable lattice-shape input is not marked immutable")
    require(immutable_shape.get("source_ref") ==
            "/user_input/parameters/number of lattice site [L,L]:",
            "Unexpected immutable lattice-shape source_ref")
    require(immutable_shape.get("selected_value") == [4, 4],
            "This implementation supports only the contracted 4x4 vertex lattice")

    discretizations = require_list(
        contract.get("discretizations"), "discretizations"
    )
    spacetime = find_unique(
        discretizations,
        lambda item: item.get("sector") == "spacetime",
        "spacetime discretization",
    )
    gauge = find_unique(
        discretizations,
        lambda item: item.get("sector") == "gauge",
        "gauge discretization",
    )

    require(
        spacetime.get("scheme") ==
        "finite square spatial lattice with continuous Hamiltonian time",
        "Unsupported spacetime discretization scheme",
    )
    st = require_dict(spacetime.get("settings"),
                      "spacetime discretization settings")
    expected_settings = {
        "spatial_boundary_condition": "open",
        "spatial_shape": [4, 4],
        "time_realization": "continuous",
        "vertex_count": 16,
        "link_count": 24,
        "plaquette_count": 9,
        "independent_gauss_constraint_count": 15,
        "vertex_indices": {"x": [0, 3], "y": [0, 3]},
        "horizontal_link_indices": {"x": [0, 2], "y": [0, 3]},
        "vertical_link_indices": {"x": [0, 3], "y": [0, 2]},
        "plaquette_indices": {"x": [0, 2], "y": [0, 2]},
    }
    for key, expected in expected_settings.items():
        require(st.get(key) == expected,
                f"Unsupported spacetime setting {key!r}")

    st_truncation = require_dict(
        spacetime.get("truncation"), "spacetime truncation"
    )
    require(st_truncation.get("scheme") == "none",
            "Spacetime truncation is unsupported")

    require(
        gauge.get("scheme") ==
        "exact Gauss-law reduction to open-boundary plaquette-dual variables",
        "Unsupported gauge discretization scheme",
    )
    gs = require_dict(gauge.get("settings"), "gauge discretization settings")
    expected_gauge_settings = {
        "effective_qubit_count": 9,
        "original_kinematic_dimension": 16777216,
        "original_link_local_dimension": 2,
        "physical_dimension": 512,
        "maximum_reduced_term_support": 2,
        "gauge_group": "Z2",
        "gauge_fixing": "none",
        "gauss_sector": "G_v=+1 at every lattice vertex",
        "plaquette_mapping": "B_p maps to tau_p^z",
        "electric_mapping":
            "X_l maps to the product of tau_p^x over plaquettes incident on link l",
    }
    for key, expected in expected_gauge_settings.items():
        require(gs.get(key) == expected,
                f"Unsupported gauge setting {key!r}")

    gauge_truncation = require_dict(
        gauge.get("truncation"), "gauge truncation"
    )
    require(gauge_truncation.get("scheme") == "none",
            "Gauge truncation is unsupported")
    gauge_truncation_settings = require_dict(
        gauge_truncation.get("settings"), "gauge truncation settings"
    )
    require(gauge_truncation_settings.get("reduction_exact") is True,
            "The plaquette-dual reduction must be exact")
    require(
        gauge_truncation_settings.get("finite_group_representation") ==
        "complete two-dimensional Z2 link representation",
        "Unsupported finite-group representation",
    )

    external = require_dict(contract.get("external_states"),
                            "external_states")
    labels = require_list(external.get("labels"), "external_states.labels")
    require(
        len(labels) == 2 and set(labels) == EXPECTED_STATE_LABELS,
        "This implementation requires independent vacuum and glueball states",
    )
    require(external.get("equivalent_groups") == [],
            "Equivalent-state grouping is unsupported for these independent states")
    definitions = require_dict(external.get("definitions"),
                               "external_states.definitions")
    require(
        definitions.get("vacuum") == "Ground state / vacuum",
        "The vacuum target must be the ground state",
    )
    glueball_definition = definitions.get("glueball")
    require(isinstance(glueball_definition, str),
            "The glueball definition must be a string")
    require("level_index=1" in glueball_definition,
            "The glueball must use zero-based level_index=1")
    require("\"Gauss's law\": 0" in glueball_definition,
            "The glueball must target zero gauge charge")

    obligations = require_list(
        contract.get("validation_obligations"), "validation_obligations"
    )
    obligation_ids = {
        item.get("id") for item in obligations if isinstance(item, dict)
    }
    required_obligations = {
        "validate_physical_sector",
        "validate_dual_operator_algebra",
        "validate_glueball_level",
        "validate_unshifted_hamiltonian",
    }
    require(required_obligations.issubset(obligation_ids),
            "Required validation obligations are missing")

    return g, a, labels


def validate_observable_spec(spec, available_states):
    require(spec.get("observable_schema_version") == "1.0",
            "Unsupported Observable Specification schema version")
    require(spec.get("expression") == "Hamiltonian: H",
            "Only the contracted Hamiltonian observable is supported")
    require(spec.get("ordering") == "as_written",
            "Only as-written observable ordering is supported")

    insertions = require_list(spec.get("insertions"), "insertions")
    require(len(insertions) == 1, "Exactly one Hamiltonian insertion is required")
    insertion = require_dict(insertions[0], "insertions[0]")
    require(
        insertion.get("id") == "Hamiltonian"
        and insertion.get("operator") == "H"
        and insertion.get("position") == "global",
        "Unsupported observable insertion structure",
    )

    components = require_list(
        spec.get("requested_components"), "requested_components"
    )
    require(
        len(components) == 2 and set(components) == {"real", "imag"},
        "The contracted observable requires real and imag components",
    )

    matrix_elements = require_list(
        spec.get("matrix_elements"), "matrix_elements"
    )
    require(len(matrix_elements) > 0,
            "At least one matrix element is required")
    validated_elements = []
    referenced_states = set()
    for index, element_value in enumerate(matrix_elements):
        element = require_dict(
            element_value, f"matrix_elements[{index}]"
        )
        bra = element.get("bra")
        ket = element.get("ket")
        require(isinstance(bra, str) and bra in available_states,
                f"Unsupported bra state in matrix_elements[{index}]")
        require(isinstance(ket, str) and ket in available_states,
                f"Unsupported ket state in matrix_elements[{index}]")
        coefficient = finite_real(
            element.get("coefficient"),
            f"matrix_elements[{index}].coefficient",
        )
        require(isinstance(element.get("id"), str),
                f"matrix_elements[{index}].id must be a string")
        referenced_states.update((bra, ket))
        validated_elements.append((bra, ket, coefficient))

    require(
        referenced_states == EXPECTED_STATE_LABELS,
        "The observable must preserve and reference both contracted states",
    )

    sampling = require_dict(spec.get("sampling"), "sampling")
    require(sampling.get("mode") == "explicit_points",
            "Only explicit_points sampling is supported")
    points = require_list(sampling.get("points"), "sampling.points")
    require(len(points) > 0, "At least one sampling point is required")
    seen_ids = set()
    validated_points = []
    for index, point_value in enumerate(points):
        point = require_dict(point_value, f"sampling.points[{index}]")
        sample_id = point.get("sample_id")
        values = point.get("values")
        require(isinstance(sample_id, str) and sample_id,
                f"sampling.points[{index}].sample_id must be a nonempty string")
        require(sample_id not in seen_ids,
                f"Duplicate sample_id {sample_id!r}")
        require(isinstance(values, dict),
                f"sampling.points[{index}].values must be an object")
        seen_ids.add(sample_id)
        validated_points.append((sample_id, copy.deepcopy(values)))

    return validated_elements, validated_points


def plaquette_index(x, y):
    require(0 <= x <= 2 and 0 <= y <= 2,
            f"Invalid plaquette coordinate ({x}, {y})")
    return x + 3 * y


def construct_link_masks():
    links = []

    for x in range(3):
        for y in range(4):
            incident = []
            if y <= 2:
                incident.append((x, y))
            if y >= 1:
                incident.append((x, y - 1))
            mask = 0
            for px, py in incident:
                mask |= 1 << plaquette_index(px, py)
            links.append({
                "kind": "h",
                "x": x,
                "y": y,
                "endpoints": ((x, y), (x + 1, y)),
                "incident": tuple(incident),
                "mask": mask,
            })

    for x in range(4):
        for y in range(3):
            incident = []
            if x <= 2:
                incident.append((x, y))
            if x >= 1:
                incident.append((x - 1, y))
            mask = 0
            for px, py in incident:
                mask |= 1 << plaquette_index(px, py)
            links.append({
                "kind": "v",
                "x": x,
                "y": y,
                "endpoints": ((x, y), (x, y + 1)),
                "incident": tuple(incident),
                "mask": mask,
            })

    require(len(links) == 24, "Internal error: expected 24 links")
    require(all(link["mask"] != 0 for link in links),
            "An open-boundary link has no incident plaquette")
    require(all(link["mask"].bit_count() in (1, 2) for link in links),
            "A reduced electric term has unsupported support")
    return links


def validate_reduced_algebra(links):
    for link in links:
        expected_mask = 0
        for coordinate in link["incident"]:
            expected_mask |= 1 << plaquette_index(*coordinate)
        require(link["mask"] == expected_mask,
                "Incident-plaquette mapping mismatch")

        incident_indices = {
            plaquette_index(px, py) for px, py in link["incident"]
        }
        for p in range(9):
            anticommutes = ((link["mask"] >> p) & 1) == 1
            require(
                anticommutes == (p in incident_indices),
                "Reduced X_l/B_p commutation algebra mismatch",
            )

    for vx in range(4):
        for vy in range(4):
            gauss_mask = 0
            incident_link_count = 0
            for link in links:
                if (vx, vy) in link["endpoints"]:
                    gauss_mask ^= link["mask"]
                    incident_link_count += 1
            require(incident_link_count in (2, 3, 4),
                    "Invalid open-square vertex incidence")
            require(
                gauss_mask == 0,
                f"Gauss operator at vertex ({vx},{vy}) is not identity "
                "in the reduced representation",
            )

    require(2 ** (24 - 15) == 512,
            "Physical Hilbert-space dimension check failed")


def construct_hamiltonian(g, a, links):
    dimension = 1 << 9
    electric_coefficient = -(g * g) / a
    magnetic_coefficient = -(g ** -2) / a

    rows = []
    cols = []
    data = []

    basis = np.arange(dimension, dtype=np.int64)

    for link in links:
        mask = link["mask"]
        rows.extend((basis ^ mask).tolist())
        cols.extend(basis.tolist())
        data.extend([electric_coefficient] * dimension)

    diagonal = np.zeros(dimension, dtype=np.float64)
    for p in range(9):
        z_eigenvalues = 1.0 - 2.0 * ((basis >> p) & 1)
        diagonal += magnetic_coefficient * z_eigenvalues

    rows.extend(basis.tolist())
    cols.extend(basis.tolist())
    data.extend(diagonal.tolist())

    hamiltonian = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(dimension, dimension),
        dtype=np.float64,
    ).tocsr()
    hamiltonian.sum_duplicates()
    hamiltonian.eliminate_zeros()

    require(hamiltonian.shape == (512, 512),
            "Hamiltonian has the wrong physical-space dimension")
    asymmetry = hamiltonian - hamiltonian.transpose()
    asymmetry.eliminate_zeros()
    require(asymmetry.nnz == 0,
            "Constructed Hamiltonian is not exactly real symmetric")
    require(np.all(np.isfinite(hamiltonian.data)),
            "Hamiltonian contains non-finite coefficients")
    return hamiltonian


def diagonalize_and_build_states(hamiltonian):
    dense = hamiltonian.toarray()
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        dense,
        lower=True,
        overwrite_a=True,
        check_finite=True,
        driver="evd",
    )
    require(eigenvalues.shape == (512,),
            "Exact diagonalization returned the wrong spectrum size")
    require(np.all(np.isfinite(eigenvalues)),
            "Exact diagonalization returned non-finite eigenvalues")
    require(np.all(np.diff(eigenvalues) >= -1e-11),
            "Eigenvalues are not ordered")

    states = {
        "vacuum": eigenvectors[:, 0].astype(np.complex128, copy=False),
        "glueball": eigenvectors[:, 1].astype(np.complex128, copy=False),
    }
    levels = {"vacuum": 0, "glueball": 1}
    return eigenvalues, states, levels


def state_diagnostics(hamiltonian, state, target_energy):
    norm = float(np.linalg.norm(state))
    require(math.isfinite(norm) and abs(norm - 1.0) <= 1e-10,
            "Eigensolver returned an unnormalized state")

    h_state = hamiltonian @ state
    energy_complex = np.vdot(state, h_state)
    require(abs(float(energy_complex.imag)) <= 1e-10,
            "State energy has an unexpectedly large imaginary part")
    energy = float(energy_complex.real)

    residual = h_state - target_energy * state
    variance = float(np.vdot(residual, residual).real)
    require(math.isfinite(energy), "State energy is non-finite")
    require(math.isfinite(variance) and variance >= 0.0,
            "State energy variance is invalid")
    require(abs(energy - target_energy) <=
            1e-10 * max(1.0, abs(target_energy)),
            "State energy does not match its targeted exact level")
    return energy, variance, float(np.linalg.norm(residual))


def finite_output_number(value, path):
    result = float(value)
    require(math.isfinite(result), f"Non-finite output value at {path}")
    return result


def atomic_write_json(path, result):
    output_path = Path(path)
    parent = output_path.parent
    require(parent.exists() and parent.is_dir(),
            f"Output directory does not exist: {parent}")

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent),
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                result,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="SciPy exact reference for the contracted 4x4 open Z2 lattice"
    )
    parser.add_argument("--parameter-contract", required=True)
    parser.add_argument("--observable-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parameter_contract = load_json(
        args.parameter_contract, "Parameter Contract"
    )
    observable_spec = load_json(
        args.observable_spec, "Observable Specification"
    )

    g, a, state_label_order = validate_parameter_contract(parameter_contract)
    matrix_elements, sample_points = validate_observable_spec(
        observable_spec, EXPECTED_STATE_LABELS
    )

    links = construct_link_masks()
    validate_reduced_algebra(links)
    hamiltonian = construct_hamiltonian(g, a, links)

    eigenvalues, states, levels = diagonalize_and_build_states(hamiltonian)

    state_results = {}
    residual_norms = {}
    for label in state_label_order:
        target_level = levels[label]
        energy, variance, residual_norm = state_diagnostics(
            hamiltonian, states[label], float(eigenvalues[target_level])
        )
        state_results[label] = {
            "energy": finite_output_number(
                energy, f"state_results.{label}.energy"
            ),
            "energy_variance": finite_output_number(
                variance, f"state_results.{label}.energy_variance"
            ),
        }
        residual_norms[label] = residual_norm

    observable = 0.0 + 0.0j
    roundoff_weight = 0.0
    for bra, ket, coefficient in matrix_elements:
        matrix_element = np.vdot(states[bra], hamiltonian @ states[ket])
        require(
            math.isfinite(float(matrix_element.real))
            and math.isfinite(float(matrix_element.imag)),
            f"Non-finite matrix element <{bra}|H|{ket}>",
        )
        observable += coefficient * matrix_element
        roundoff_weight += abs(coefficient) * (
            abs(matrix_element) + np.linalg.norm(hamiltonian @ states[ket])
        )

    observable_real = finite_output_number(
        observable.real, "observable_real"
    )
    observable_imag = finite_output_number(
        observable.imag, "observable_imag"
    )
    roundoff_uncertainty = finite_output_number(
        np.finfo(np.float64).eps
        * 128.0
        * max(1.0, roundoff_weight, abs(observable)),
        "roundoff_uncertainty",
    )

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    residual_tolerance = 1e-10 * scale
    convergence_passed = all(
        value <= residual_tolerance for value in residual_norms.values()
    )
    require(convergence_passed,
            "Exact diagonalization residual tolerance was not met")

    observable_samples = []
    for sample_id, sample_values in sample_points:
        observable_samples.append({
            "sample_id": sample_id,
            "sample_values": sample_values,
            "observable_real": observable_real,
            "observable_imag": observable_imag,
            "uncertainty_real": roundoff_uncertainty,
            "uncertainty_imag": roundoff_uncertainty,
        })

    ground = state_results["vacuum"]
    result = {
        "reference_schema_version": REFERENCE_SCHEMA_VERSION,
        "method": METHOD_ID,
        "ground_state_energy": ground["energy"],
        "energy_variance": ground["energy_variance"],
        "state_results": state_results,
        "observable_samples": observable_samples,
        "convergence": {
            "passed": convergence_passed,
            "details": {
                "physical_dimension": 512,
                "effective_qubit_count": 9,
                "link_count": 24,
                "plaquette_count": 9,
                "boundary_condition": "open",
                "gauss_sector": "G_v=+1 at every lattice vertex",
                "gauss_reduction_exact": True,
                "hamiltonian_unshifted": True,
                "vacuum_level_index": 0,
                "glueball_level_index": 1,
                "residual_norms": {
                    key: finite_output_number(
                        value, f"convergence.residual_norms.{key}"
                    )
                    for key, value in residual_norms.items()
                },
                "residual_tolerance": finite_output_number(
                    residual_tolerance,
                    "convergence.residual_tolerance",
                ),
            },
        },
    }

    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
