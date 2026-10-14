#!/usr/bin/env python3
import argparse
import itertools
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
SUPPORTED_SITE_COUNT = 5
SUPPORTED_MODES_PER_SITE = 2
SUPPORTED_MODE_COUNT = 10
NUMERICAL_TOLERANCE = 5.0e-11


def fail(message):
    raise ValueError(message)


def require(condition, message):
    if not condition:
        fail(message)


def finite_real(value, name):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be a real number",
    )
    value = float(value)
    require(math.isfinite(value), f"{name} must be finite")
    return value


def load_json(path, description):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Cannot load {description} from {path}: {exc}") from exc
    require(isinstance(value, dict), f"{description} must be one JSON object")
    return value


def parameter_map(contract):
    parameters = contract.get("parameters")
    require(isinstance(parameters, list), "Parameter Contract parameters must be a list")
    result = {}
    for entry in parameters:
        require(isinstance(entry, dict), "Every parameter entry must be an object")
        name = entry.get("name")
        require(isinstance(name, str) and name, "Every parameter must have a name")
        require(name not in result, f"Duplicate parameter name: {name}")
        require("selected_value" in entry, f"Parameter {name} has no selected_value")
        result[name] = entry
    return result


def selected_parameter(parameters, name):
    require(name in parameters, f"Missing required parameter: {name}")
    return parameters[name]["selected_value"]


def find_discretization(contract, sector):
    matches = [
        item
        for item in contract.get("discretizations", [])
        if isinstance(item, dict) and item.get("sector") == sector
    ]
    require(len(matches) == 1, f"Expected exactly one {sector} discretization")
    return matches[0]


def parse_state_definition(definition, label):
    if isinstance(definition, dict):
        level = definition.get("level_index")
        quantum_numbers = definition.get("quantum_numbers")
        require(isinstance(quantum_numbers, dict), f"{label} quantum_numbers must be an object")
        charge = quantum_numbers.get("charge")
        return level, charge

    require(isinstance(definition, str), f"Unsupported state definition for {label}")
    level_match = re.search(r"level_index\s*=\s*(-?\d+)", definition)
    charge_match = re.search(r'["\']?charge["\']?\s*:\s*(-?\d+)', definition)
    require(level_match is not None, f"Cannot parse level_index for {label}")
    require(charge_match is not None, f"Cannot parse charge for {label}")
    return int(level_match.group(1)), int(charge_match.group(1))


def validate_parameter_contract(contract):
    require(
        contract.get("schema_version") == "1.1",
        "Unsupported Parameter Contract schema_version",
    )
    parameters = parameter_map(contract)

    g = finite_real(selected_parameter(parameters, "bare_gauge_coupling"), "g")
    m = finite_real(selected_parameter(parameters, "bare_fermion_mass"), "m")
    a = finite_real(selected_parameter(parameters, "lattice_spacing"), "a")
    at = finite_real(
        selected_parameter(parameters, "temporal_lattice_spacing"), "a_t"
    )
    r = finite_real(selected_parameter(parameters, "wilson_parameter"), "r")
    n_ref = selected_parameter(parameters, "charge_reference_occupancy")
    lightcone_normalization = finite_real(
        selected_parameter(parameters, "lightcone_normalization"),
        "lightcone_normalization",
    )

    require(g >= 0.0, "bare_gauge_coupling must be nonnegative")
    require(a > 0.0 and at > 0.0, "Lattice spacings must be positive")
    require(r != 0.0, "wilson_parameter must be nonzero")
    require(
        abs(at - a) <= 1.0e-14 * max(1.0, abs(a)),
        "temporal_lattice_spacing must equal lattice_spacing",
    )
    require(n_ref == 1, "Only charge_reference_occupancy=1 is implemented")
    require(
        lightcone_normalization == 1.0,
        "Only the authoritative light-cone normalization 1 is implemented",
    )

    aliases = {
        "bare coupling: g": g,
        "bare quark mass: m": m,
        "lattice spacing: a": a,
    }
    for name, expected in aliases.items():
        actual = finite_real(selected_parameter(parameters, name), name)
        require(
            actual == expected,
            f"Contradictory canonical and immutable values for {name}",
        )

    lattice_size = selected_parameter(parameters, "number of lattice site: L")
    require(
        isinstance(lattice_size, int) and not isinstance(lattice_size, bool),
        "number of lattice site: L must be an integer",
    )
    require(
        lattice_size == SUPPORTED_SITE_COUNT,
        "This implementation supports exactly five spatial sites",
    )

    model = contract.get("model")
    require(isinstance(model, dict), "Missing model declaration")
    require(model.get("name") == "Schwinger model", "Unsupported model")
    require(model.get("gauge_structure") == "local U(1)", "Unsupported gauge structure")
    require(model.get("spacetime_dimension") == "1+1", "Unsupported spacetime dimension")

    spacetime = find_discretization(contract, "spacetime")
    st_settings = spacetime.get("settings", {})
    require(
        spacetime.get("scheme")
        == "centered one-dimensional spatial lattice with equal temporal spacing",
        "Unsupported spacetime discretization",
    )
    require(st_settings.get("spatial_boundary_condition") == "open", "OBC required")
    require(st_settings.get("spatial_site_count") == 5, "Five sites are required")
    require(st_settings.get("spatial_link_count") == 4, "Four links are required")
    require(st_settings.get("spatial_dimension") == 1, "One spatial dimension is required")
    require(st_settings.get("observable_site_map") == "s(z)=z+2", "Unsupported site map")
    require(st_settings.get("plaquette_count") == 0, "Unexpected plaquettes")
    require(st_settings.get("temporal_spacing_relation") == "a_t=a", "a_t=a required")

    fermion = find_discretization(contract, "fermion")
    f_settings = fermion.get("settings", {})
    require(fermion.get("scheme") == "two-component Wilson fermions", "Unsupported fermions")
    require(f_settings.get("alpha") == "sigma_3", "alpha must be sigma_3")
    require(f_settings.get("beta") == "sigma_1", "beta must be sigma_1")
    require(f_settings.get("gamma_0") == "sigma_1", "gamma_0 must be sigma_1")
    require(f_settings.get("gamma_1") == "-i*sigma_2", "gamma_1 must be -i*sigma_2")
    require(f_settings.get("component_count_per_site") == 2, "Two spinor components required")
    require(f_settings.get("fermionic_mode_count") == 10, "Ten fermion modes required")
    if "wilson_parameter" in f_settings:
        require(
            float(f_settings["wilson_parameter"]) == r,
            "Contradictory Wilson-parameter declarations",
        )
    f_trunc = fermion.get("truncation", {})
    require(
        f_trunc.get("scheme") == "exact fermionic occupation",
        "Approximate fermion truncation is unsupported",
    )

    gauge = find_discretization(contract, "gauge")
    g_settings = gauge.get("settings", {})
    require(
        gauge.get("scheme")
        == "spatial axial gauge with exact open-boundary Gauss-law elimination",
        "Exact axial-gauge Gauss-law elimination is required",
    )
    require(g_settings.get("gauge_condition") == "A_1=0", "A_1=0 gauge is required")
    require(g_settings.get("independent_gauge_degree_count") == 0, "Gauge links must be eliminated")
    require(g_settings.get("kinematic_dimension_after_reduction") == 1024, "Unexpected Hilbert dimension")
    require(g_settings.get("external_neutral_sector_dimension") == 252, "Unexpected neutral dimension")
    require(g_settings.get("physical_dimension_with_all_required_charge_sectors") == 1024,
            "All charge sectors must remain available")
    require(
        g_settings.get("electric_field_reconstruction")
        == "E_-1=-(g/2)*Q_lat; E_ell=g*(S_ell-Q_lat/2) for ell=0,...,3; E_4=(g/2)*Q_lat",
        "Unsupported boundary-flux reconstruction",
    )
    g_trunc = gauge.get("truncation", {})
    require(g_trunc.get("scheme") == "none", "Gauge-flux truncation is forbidden")
    trunc_settings = g_trunc.get("settings", {})
    require(trunc_settings.get("electric_flux_cutoff") == "none", "Gauge-flux cutoff is forbidden")
    require(trunc_settings.get("gauge_links") == "exactly eliminated", "Gauge links must remain eliminated")

    external = contract.get("external_states")
    require(isinstance(external, dict), "Missing external_states")
    labels = external.get("labels")
    require(labels == ["phi", "psi"], "Exactly the independent states phi and psi are required")
    require(external.get("equivalent_groups") == [], "phi and psi must not be identified")
    definitions = external.get("definitions")
    require(isinstance(definitions, dict), "Missing external state definitions")

    state_targets = {}
    for label in labels:
        require(label in definitions, f"Missing definition for {label}")
        level, charge = parse_state_definition(definitions[label], label)
        require(isinstance(level, int) and level >= 0, f"Invalid level for {label}")
        require(charge == 0, f"{label} must be in the Q_lat=0 sector")
        state_targets[label] = {"level_index": level, "charge": charge}

    require(
        state_targets["phi"]["level_index"] == 0,
        "phi must be zero-based level 0",
    )
    require(
        state_targets["psi"]["level_index"] == 1,
        "psi must be zero-based level 1",
    )

    return {
        "g": g,
        "m": m,
        "a": a,
        "at": at,
        "r": r,
        "n_ref": int(n_ref),
        "lightcone_normalization": lightcone_normalization,
        "state_targets": state_targets,
    }


def validate_and_expand_observable(spec):
    require(
        spec.get("observable_schema_version") == "1.0",
        "Unsupported Observable Specification schema version",
    )
    require(spec.get("ordering") == "as_written", "Only as-written operator ordering is supported")
    require(
        spec.get("requested_components") == ["real", "imag"],
        "This run must request real and imag components in canonical order",
    )

    expression = spec.get("expression")
    require(isinstance(expression, str), "Observable expression must be a string")
    compact = re.sub(r"\s+", "", expression)
    require(r"\bar{\psi}(zn)" in compact, "Observable must contain bar-psi at zn")
    require(r"\gamma^-" in compact, "Observable must use gamma-minus")
    require(r"W(zn,0)" in compact, "Observable must contain W(zn,0)")
    require(r"\psi(0)" in compact, "Observable must contain psi(0)")
    require(
        r"\gamma^-=\gamma^0-\gamma^1" in compact,
        "gamma-minus must literally equal gamma-zero minus gamma-one",
    )

    insertions = spec.get("insertions")
    require(isinstance(insertions, list) and len(insertions) == 2, "Expected two insertions")
    require(insertions[0].get("id") == "psibar", "First insertion must be psibar")
    require(insertions[0].get("operator") == r"\bar{\psi}", "Unsupported first insertion")
    require(insertions[0].get("position") == {"t1": "z1"}, "Unsupported psibar position")
    require(insertions[1].get("id") == "psi", "Second insertion must be psi")
    require(insertions[1].get("operator") == r"\psi", "Unsupported second insertion")
    require(insertions[1].get("position") == {"t2": "z2"}, "Unsupported psi position")

    transporters = spec.get("gauge_transporters")
    require(isinstance(transporters, list) and len(transporters) == 1,
            "Exactly one Wilson transporter is required")
    transporter = transporters[0]
    require(transporter.get("from") == r"\psi", "Wilson line must start at psi")
    require(transporter.get("to") == r"\bar{\psi}", "Wilson line must end at psibar")
    require(transporter.get("path") == "straight lattice path", "Only the straight path is supported")

    require("matrix_elements" not in spec, "Unsupported explicit matrix_elements structure")

    coordinate = spec.get("coordinate_convention")
    require(isinstance(coordinate, dict), "Missing coordinate convention")
    require(coordinate.get("mode") == "lattice_indices", "Lattice-index coordinates required")
    require(coordinate.get("temporal_spacing") == "same_as_spatial", "Equal spacings required")

    sampling = spec.get("sampling")
    require(isinstance(sampling, dict), "Missing sampling declaration")
    require(sampling.get("mode") == "cartesian_product", "Only cartesian_product sampling is supported")
    maximum_points = sampling.get("maximum_points")
    require(isinstance(maximum_points, int) and maximum_points > 0, "Invalid maximum_points")
    variables = sampling.get("variables")
    require(isinstance(variables, dict), "sampling.variables must be an object")
    require(list(variables.keys()) == ["z1", "z2"], "Sampling variables must be ordered z1, z2")

    z1_values = variables["z1"]
    z2_values = variables["z2"]
    require(isinstance(z1_values, list) and z1_values, "z1 sampling list must be nonempty")
    require(isinstance(z2_values, list) and z2_values, "z2 sampling list must be nonempty")
    require(z2_values == [0], "This observable implementation requires z2=0")

    for z in z1_values:
        require(isinstance(z, int) and not isinstance(z, bool), "Every z1 must be an integer")
        require(-2 <= z <= 2, "Every z1 must map to a valid site through s(z)=z+2")

    assignments = []
    for index, (z1, z2) in enumerate(itertools.product(z1_values, z2_values)):
        assignments.append(
            {
                "sample_id": f"sample_{index:04d}",
                "values": {"z1": z1, "z2": z2},
            }
        )
    require(len(assignments) <= maximum_points, "Expanded sampling exceeds maximum_points")
    return assignments


def popcount(value):
    return int(value.bit_count())


def sector_basis(particle_number):
    return np.array(
        [state for state in range(1 << SUPPORTED_MODE_COUNT)
         if popcount(state) == particle_number],
        dtype=np.int64,
    )


def one_body_matrix(m, a, r):
    sigma1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma3 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    h = np.zeros((SUPPORTED_MODE_COUNT, SUPPORTED_MODE_COUNT), dtype=np.complex128)
    onsite = (m + r / a) * sigma1
    forward = -1j * sigma3 / (2.0 * a) - r * sigma1 / (2.0 * a)

    for site in range(SUPPORTED_SITE_COUNT):
        sl = slice(2 * site, 2 * site + 2)
        h[sl, sl] += onsite

    for site in range(SUPPORTED_SITE_COUNT - 1):
        left = slice(2 * site, 2 * site + 2)
        right = slice(2 * (site + 1), 2 * (site + 1) + 2)
        h[left, right] += forward
        h[right, left] += forward.conj().T

    require(
        np.max(np.abs(h - h.conj().T)) < 1.0e-14,
        "Constructed one-body matrix is not Hermitian",
    )
    return h


def site_charges(state, n_ref=1):
    return np.array(
        [
            ((state >> (2 * site)) & 1)
            + ((state >> (2 * site + 1)) & 1)
            - n_ref
            for site in range(SUPPORTED_SITE_COUNT)
        ],
        dtype=np.float64,
    )


def coulomb_energy(state, g, a, n_ref):
    q = site_charges(state, n_ref)
    total_charge = float(np.sum(q))
    partial = np.cumsum(q)[:4]
    return 0.5 * a * g * g * float(np.sum((partial - 0.5 * total_charge) ** 2))


def construct_hamiltonian(basis, h_one_body, g, a, n_ref):
    dimension = len(basis)
    index = {int(state): row for row, state in enumerate(basis)}
    rows = []
    cols = []
    data = []

    nonzero_h = np.argwhere(np.abs(h_one_body) > 0.0)
    for col, state_value in enumerate(basis):
        state = int(state_value)

        diagonal = coulomb_energy(state, g, a, n_ref)
        if diagonal != 0.0:
            rows.append(col)
            cols.append(col)
            data.append(complex(diagonal))

        for i, j in nonzero_h:
            i = int(i)
            j = int(j)
            if ((state >> j) & 1) == 0:
                continue

            sign_ann = -1.0 if popcount(state & ((1 << j) - 1)) % 2 else 1.0
            intermediate = state ^ (1 << j)
            if ((intermediate >> i) & 1) != 0:
                continue

            sign_cre = -1.0 if popcount(intermediate & ((1 << i) - 1)) % 2 else 1.0
            target = intermediate | (1 << i)
            row = index.get(target)
            if row is None:
                fail("Hamiltonian unexpectedly left its particle-number sector")

            rows.append(row)
            cols.append(col)
            data.append(h_one_body[i, j] * sign_ann * sign_cre)

    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, cols)),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    defect = matrix - matrix.getH()
    max_defect = 0.0 if defect.nnz == 0 else float(np.max(np.abs(defect.data)))
    require(max_defect < NUMERICAL_TOLERANCE, "Hamiltonian is not Hermitian")
    return matrix


def construct_annihilation(source_basis, target_basis, mode):
    target_index = {int(state): row for row, state in enumerate(target_basis)}
    rows = []
    cols = []
    data = []

    for col, state_value in enumerate(source_basis):
        state = int(state_value)
        if ((state >> mode) & 1) == 0:
            continue
        target = state ^ (1 << mode)
        require(target in target_index, "Annihilation operator left its target sector")
        sign = -1.0 if popcount(state & ((1 << mode) - 1)) % 2 else 1.0
        rows.append(target_index[target])
        cols.append(col)
        data.append(sign)

    return sp.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, cols)),
        shape=(len(target_basis), len(source_basis)),
        dtype=np.complex128,
    ).tocsr()


def a0_diagonal(basis, site, g, a, n_ref):
    require(0 <= site < SUPPORTED_SITE_COUNT, "A0 site is out of range")
    values = np.empty(len(basis), dtype=np.float64)
    for row, state_value in enumerate(basis):
        q = site_charges(int(state_value), n_ref)
        total_charge = float(np.sum(q))
        partial = np.cumsum(q)[:4]
        coefficient = np.array(
            [(1.0 if site <= ell else 0.0) - 0.5 for ell in range(4)],
            dtype=np.float64,
        )
        values[row] = a * g * float(
            np.dot(coefficient, partial - 0.5 * total_charge)
        )
    return values


def validate_gauss_identity(g, n_ref):
    max_residual = 0.0
    observed_intermediate_sector = False

    for state in range(1 << SUPPORTED_MODE_COUNT):
        q = site_charges(state, n_ref)
        total_charge = float(np.sum(q))
        if total_charge == -1.0:
            observed_intermediate_sector = True

        electric = {-1: -0.5 * g * total_charge, 4: 0.5 * g * total_charge}
        partial = np.cumsum(q)
        for ell in range(4):
            electric[ell] = g * (partial[ell] - 0.5 * total_charge)

        for site in range(5):
            residual = electric[site] - electric[site - 1] - g * q[site]
            max_residual = max(max_residual, abs(float(residual)))

    require(observed_intermediate_sector, "The Q=-1 intermediate sector is missing")
    require(max_residual < 1.0e-13, "Gauss-law reconstruction failed")
    return max_residual


def canonicalize_eigenvector_phases(vectors):
    vectors = np.array(vectors, dtype=np.complex128, copy=True)
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        magnitude = abs(vector[pivot])
        require(magnitude > 0.0, "Encountered a zero eigenvector")
        phase = vector[pivot] / magnitude
        vectors[:, column] /= phase
        if vectors[pivot, column].real < 0.0:
            vectors[:, column] *= -1.0
    return vectors


def diagonalize(matrix):
    dense = matrix.toarray()
    values, vectors = scipy.linalg.eigh(
        dense,
        lower=True,
        overwrite_a=True,
        check_finite=True,
        driver="evd",
    )
    vectors = canonicalize_eigenvector_phases(vectors)
    return values, vectors


def evolution_operator(eigenvalues, eigenvectors, time):
    phases = np.exp(-1j * eigenvalues * time)
    return (eigenvectors * phases[np.newaxis, :]) @ eigenvectors.conj().T


def state_energy_and_variance(matrix, state):
    applied = matrix @ state
    energy_complex = np.vdot(state, applied)
    require(abs(energy_complex.imag) < 1.0e-10, "State energy is unexpectedly complex")
    energy = float(energy_complex.real)
    residual = applied - energy * state
    variance = float(max(0.0, np.vdot(residual, residual).real))
    require(math.isfinite(energy) and math.isfinite(variance), "Non-finite state result")
    return energy, variance, float(np.linalg.norm(residual))


def atomic_write_json(path, payload):
    destination = Path(path)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Independent SciPy exact reference for the five-site axial-gauge Schwinger model"
    )
    parser.add_argument("--parameter-contract", required=True)
    parser.add_argument("--observable-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = load_json(args.parameter_contract, "Parameter Contract")
    observable_spec = load_json(args.observable_spec, "Observable Specification")

    runtime = validate_parameter_contract(contract)
    assignments = validate_and_expand_observable(observable_spec)

    g = runtime["g"]
    m = runtime["m"]
    a = runtime["a"]
    r = runtime["r"]
    n_ref = runtime["n_ref"]

    h_one_body = one_body_matrix(m, a, r)

    full_basis = np.arange(1 << SUPPORTED_MODE_COUNT, dtype=np.int64)
    neutral_basis = sector_basis(5)
    minus_one_basis = sector_basis(4)

    require(len(full_basis) == 1024, "Unexpected full reduced-space dimension")
    require(len(neutral_basis) == 252, "Unexpected Q=0 sector dimension")
    require(len(minus_one_basis) == 210, "Unexpected Q=-1 sector dimension")

    full_hamiltonian = construct_hamiltonian(
        full_basis, h_one_body, g, a, n_ref
    )
    h_neutral = construct_hamiltonian(
        neutral_basis, h_one_body, g, a, n_ref
    )
    h_minus_one = construct_hamiltonian(
        minus_one_basis, h_one_body, g, a, n_ref
    )

    # Full-space charge preservation check.
    full_charges = np.array(
        [popcount(int(state)) - 5 for state in full_basis], dtype=np.float64
    )
    charge_operator = sp.diags(full_charges, format="csr")
    commutator = full_hamiltonian @ charge_operator - charge_operator @ full_hamiltonian
    charge_commutator_norm = (
        0.0 if commutator.nnz == 0 else float(np.max(np.abs(commutator.data)))
    )
    require(
        charge_commutator_norm < NUMERICAL_TOLERANCE,
        "H_lat does not preserve Q_lat in the full reduced Fock space",
    )

    gauss_residual = validate_gauss_identity(g, n_ref)

    evals_neutral, evecs_neutral = diagonalize(h_neutral)
    evals_minus_one, evecs_minus_one = diagonalize(h_minus_one)

    state_vectors = {}
    state_results = {}
    state_residuals = {}

    for label in ["phi", "psi"]:
        target = runtime["state_targets"][label]
        require(target["charge"] == 0, f"Unsupported charge target for {label}")
        level = target["level_index"]
        require(level < len(evals_neutral), f"Level index for {label} is out of range")
        vector = np.array(evecs_neutral[:, level], copy=True)
        state_vectors[label] = vector
        energy, variance, residual_norm = state_energy_and_variance(h_neutral, vector)
        state_results[label] = {
            "energy": energy,
            "energy_variance": variance,
        }
        state_residuals[label] = residual_norm

    phi = state_vectors["phi"]
    psi = state_vectors["psi"]

    # The right-to-left action is Q=0 --c--> Q=-1 --W--> Q=-1 --c†--> Q=0.
    ordered_sector_path = [0, -1, -1, 0]
    require(
        ordered_sector_path == [0, -1, -1, 0],
        "Ordered observable sector path is invalid",
    )

    evolution_neutral_cache = {}
    evolution_minus_cache = {}
    wilson_step_cache = {}

    def neutral_evolution(time):
        key = float(time)
        if key not in evolution_neutral_cache:
            evolution_neutral_cache[key] = evolution_operator(
                evals_neutral, evecs_neutral, key
            )
        return evolution_neutral_cache[key]

    def minus_evolution(time):
        key = float(time)
        if key not in evolution_minus_cache:
            evolution_minus_cache[key] = evolution_operator(
                evals_minus_one, evecs_minus_one, key
            )
        return evolution_minus_cache[key]

    def wilson_step(k):
        if k not in wilson_step_cache:
            site = k + 2
            require(0 <= site < 5, f"Wilson step k={k} maps outside the lattice")
            time = k * a
            a0_values = a0_diagonal(minus_one_basis, site, g, a, n_ref)
            diagonal_phase = np.exp(-1j * g * a * a0_values)
            r_minus = minus_evolution(time)
            step = r_minus.conj().T @ (
                diagonal_phase[:, np.newaxis] * r_minus
            )
            wilson_step_cache[k] = step
        return wilson_step_cache[k]

    def wilson_line(z):
        dimension = len(minus_one_basis)
        result = np.eye(dimension, dtype=np.complex128)
        if z > 0:
            for k in range(0, z):
                result = wilson_step(k) @ result
        elif z < 0:
            for k in range(z, 0):
                result = result @ wilson_step(k).conj().T
        return result

    observable_samples = []
    machine_epsilon = np.finfo(np.float64).eps

    # gamma^-=gamma^0-gamma^1 and bar(psi)=psi† gamma^0 give
    # gamma^0 gamma^-=I-sigma_3=diag(0,2).
    spinor_weight = 2.0 * runtime["lightcone_normalization"]

    for assignment in assignments:
        z = assignment["values"]["z1"]
        require(assignment["values"]["z2"] == 0, "Only z2=0 is implemented")
        endpoint_site = z + 2
        endpoint_mode = 2 * endpoint_site + 1
        origin_mode = 2 * 2 + 1

        annihilate_origin = construct_annihilation(
            neutral_basis, minus_one_basis, origin_mode
        ).toarray()
        annihilate_endpoint = construct_annihilation(
            neutral_basis, minus_one_basis, endpoint_mode
        ).toarray()
        create_endpoint = annihilate_endpoint.conj().T

        time = z * a
        r_neutral = neutral_evolution(time)
        r_minus = minus_evolution(time)
        create_endpoint_time = (
            r_neutral.conj().T @ create_endpoint @ r_minus
        )

        transporter = wilson_line(z)
        observable_action = (
            (spinor_weight / a)
            * create_endpoint_time
            @ transporter
            @ annihilate_origin
        )
        matrix_element = np.vdot(phi, observable_action @ psi)

        require(
            math.isfinite(float(matrix_element.real))
            and math.isfinite(float(matrix_element.imag)),
            "Observable matrix element is non-finite",
        )

        roundoff_scale = (
            256.0
            * machine_epsilon
            * max(1.0, float(abs(matrix_element)))
            * max(1.0, len(minus_one_basis))
        )

        observable_samples.append(
            {
                "sample_id": assignment["sample_id"],
                "sample_values": dict(assignment["values"]),
                "observable_real": float(matrix_element.real),
                "observable_imag": float(matrix_element.imag),
                "uncertainty_real": float(roundoff_scale),
                "uncertainty_imag": float(roundoff_scale),
            }
        )

    max_state_residual = max(state_residuals.values())
    max_variance = max(
        result["energy_variance"] for result in state_results.values()
    )
    convergence_passed = bool(
        charge_commutator_norm < NUMERICAL_TOLERANCE
        and gauss_residual < NUMERICAL_TOLERANCE
        and max_state_residual < 1.0e-9
        and max_variance < 1.0e-16
    )

    output = {
        "reference_schema_version": REFERENCE_SCHEMA_VERSION,
        "method": METHOD_ID,
        "ground_state_energy": state_results["phi"]["energy"],
        "energy_variance": state_results["phi"]["energy_variance"],
        "state_results": state_results,
        "observable_samples": observable_samples,
        "convergence": {
            "passed": convergence_passed,
            "details": {
                "full_reduced_dimension": 1024,
                "neutral_sector_dimension": 252,
                "intermediate_charge_minus_one_dimension": 210,
                "charge_commutator_max_abs": charge_commutator_norm,
                "gauss_identity_max_abs": gauss_residual,
                "maximum_state_residual_norm": max_state_residual,
                "maximum_state_energy_variance": max_variance,
                "ordered_sector_path": ordered_sector_path,
                "boundary_condition": "open_reflection_symmetric_flux",
                "energy_shift_applied": False,
                "independent_gauge_hilbert_space_added": False,
                "lightcone_convention": {
                    "gamma_minus": "gamma^0-gamma^1",
                    "coordinate": "x^minus=x^0-x^1",
                    "normalization": runtime["lightcone_normalization"],
                },
            },
        },
    }

    require(convergence_passed, "Exact-reference convergence checks failed")
    atomic_write_json(args.output, output)


if __name__ == "__main__":
    main()
