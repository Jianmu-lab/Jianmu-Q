#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import tempfile
from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


METHOD_ID = "scipy_exact"
REFERENCE_SCHEMA_VERSION = "1.0"
IMPLEMENTED_SITES = (-2, -1, 0, 1, 2)
IMPLEMENTED_COMPONENTS_PER_SITE = 2
IMPLEMENTED_BOUNDARY = "open"
TOL = 1.0e-10


def fail(message):
    raise ValueError(message)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        fail(f"{path}: top-level JSON value must be an object")
    return obj


def require(condition, message):
    if not condition:
        fail(message)


def finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        fail(f"{name} must be finite")
    return value


def integer_value(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{name} must be an integer")
    return int(value)


def close_scalar(left, right, name, atol=1.0e-13):
    left = finite_real(left, name)
    right = finite_real(right, name)
    require(
        abs(left - right) <= atol * max(1.0, abs(left), abs(right)),
        f"inconsistent values for {name}: {left!r} versus {right!r}",
    )


def parameter_map(contract):
    parameters = contract.get("parameters")
    require(isinstance(parameters, list), "Parameter Contract parameters must be a list")
    result = {}
    for entry in parameters:
        require(isinstance(entry, dict), "each parameter entry must be an object")
        name = entry.get("name")
        require(isinstance(name, str) and name, "each parameter must have a name")
        require(name not in result, f"duplicate Parameter Contract parameter {name!r}")
        require("selected_value" in entry, f"parameter {name!r} lacks selected_value")
        result[name] = entry
    return result


def selected(parameters, name):
    require(name in parameters, f"missing required parameter {name!r}")
    return parameters[name]["selected_value"]


def locate_fermion_settings(contract):
    discretizations = contract.get("discretizations")
    require(isinstance(discretizations, list), "discretizations must be a list")
    matches = [
        d for d in discretizations
        if isinstance(d, dict)
        and d.get("sector") == "fermion"
        and d.get("scheme") == "two-component Wilson fermions"
    ]
    require(len(matches) == 1, "exactly one two-component Wilson-fermion discretization is required")
    return matches[0]


def locate_spacetime_settings(contract):
    discretizations = contract.get("discretizations")
    matches = [
        d for d in discretizations
        if isinstance(d, dict)
        and d.get("sector") == "spacetime"
        and d.get("scheme") == "centered one-dimensional spatial lattice with continuous real time"
    ]
    require(len(matches) == 1, "exactly one supported spacetime discretization is required")
    return matches[0]


def validate_parameter_contract(contract):
    require(contract.get("schema_version") == "1.1", "unsupported Parameter Contract schema_version")

    model = contract.get("model")
    require(isinstance(model, dict), "model must be an object")
    require(model.get("name") == "one flavor NJL model", "unsupported model")
    require(model.get("spacetime_dimension") == "1+1", "only the declared 1+1-dimensional model is supported")
    require(model.get("gauge_structure") == "none", "this implementation supports no explicit gauge sector")

    params = parameter_map(contract)
    mass = finite_real(selected(params, "bare_quark_mass"), "bare_quark_mass")
    coupling = finite_real(
        selected(params, "bare_four_fermion_coupling"),
        "bare_four_fermion_coupling",
    )
    spacing = finite_real(selected(params, "lattice_spacing"), "lattice_spacing")
    site_count = integer_value(selected(params, "spatial_site_count"), "spatial_site_count")
    wilson = finite_real(selected(params, "wilson_parameter"), "wilson_parameter")
    mode_count = integer_value(selected(params, "fermionic_mode_count"), "fermionic_mode_count")

    require(spacing > 0.0, "lattice_spacing must be positive")
    require(wilson >= 0.0, "wilson_parameter must be nonnegative")
    require(site_count == len(IMPLEMENTED_SITES), "runtime lattice size differs from the implemented five-site model")
    require(mode_count == IMPLEMENTED_COMPONENTS_PER_SITE * site_count, "fermionic_mode_count must equal 2*spatial_site_count")

    close_scalar(selected(params, "bare coupling: g"), coupling, "immutable bare coupling: g")
    close_scalar(selected(params, "bare quark mass: m"), mass, "immutable bare quark mass: m")
    close_scalar(selected(params, "lattice spacing: a"), spacing, "immutable lattice spacing: a")
    require(
        integer_value(selected(params, "number of lattice site: L"), "number of lattice site: L")
        == site_count,
        "immutable lattice-site count is inconsistent",
    )

    require(site_count == 5, "only the authoritative five-site geometry is implemented")
    require(abs(spacing - 1.0) <= 1.0e-13, "the authoritative lattice spacing must remain 1")

    spacetime = locate_spacetime_settings(contract)
    ss = spacetime.get("settings")
    require(isinstance(ss, dict), "spacetime settings must be an object")
    require(ss.get("spatial_boundary_condition") == IMPLEMENTED_BOUNDARY, "boundary condition must remain open")
    require(ss.get("spatial_dimension") == 1, "spatial_dimension must be one")
    require(ss.get("spatial_site_indices") == list(IMPLEMENTED_SITES), "authoritative site indices changed")
    require(ss.get("time_representation") == "continuous Hamiltonian real time", "unsupported time representation")
    require(ss.get("link_count") == 4, "open five-site chain must have four links")
    require(
        ss.get("nearest_neighbor_link_indices")
        == [[-2, -1], [-1, 0], [0, 1], [1, 2]],
        "nearest-neighbor link set changed",
    )
    trunc = spacetime.get("truncation")
    require(isinstance(trunc, dict), "spacetime truncation must be an object")
    endpoint = trunc.get("settings", {}).get("endpoint_prescription")
    require(
        endpoint == "open Wilson second-difference with full onsite Wilson coefficient at every retained site",
        "unsupported Wilson endpoint prescription",
    )

    fermion = locate_fermion_settings(contract)
    fs = fermion.get("settings")
    require(isinstance(fs, dict), "fermion settings must be an object")
    require(fs.get("components_per_site") == 2, "two spinor components per site are required")
    require(fs.get("fermionic_mode_count") == mode_count, "fermion discretization mode count is inconsistent")
    require(fs.get("finite_fock_dimension") == 2 ** mode_count, "finite Fock dimension is inconsistent")
    require(fs.get("charge_zero_sector_dimension") == math.comb(mode_count, site_count),
            "charge-zero sector dimension is inconsistent")
    require(fs.get("intermediate_charge_minus_one_sector_dimension") == math.comb(mode_count, site_count - 1),
            "charge-minus-one sector dimension is inconsistent")
    require(fs.get("alpha_1") == "sigma_3", "unsupported alpha_1 convention")
    require(fs.get("beta") == "sigma_1", "unsupported beta convention")
    require(fs.get("gamma_0") == "sigma_1", "unsupported gamma_0 convention")
    require(fs.get("gamma_1") == "-i*sigma_2", "unsupported gamma_1 convention")
    require(
        fs.get("wilson_second_derivative_prescription")
        == "onsite r/a at every site and hopping -r*beta/(2a) on every retained nearest-neighbor link",
        "unsupported Wilson prescription",
    )
    occupations = fermion.get("truncation", {}).get("settings", {}).get("occupation_per_mode")
    require(occupations == [0, 1], "canonical fermion occupations must be exactly [0,1]")

    external = contract.get("external_states")
    require(isinstance(external, dict), "external_states must be an object")
    labels = external.get("labels")
    require(isinstance(labels, list) and all(isinstance(x, str) for x in labels),
            "external state labels must be a string list")
    require(len(labels) == len(set(labels)), "external state labels must be unique")
    definitions = external.get("definitions")
    require(isinstance(definitions, dict), "external state definitions must be an object")
    require(set(labels) == set(definitions), "external state labels and definitions differ")
    require(external.get("equivalent_groups") == [], "equivalent-state groups are unsupported for this run")

    return {
        "mass": mass,
        "coupling": coupling,
        "spacing": spacing,
        "site_count": site_count,
        "wilson": wilson,
        "mode_count": mode_count,
        "external_labels": labels,
        "external_definitions": definitions,
    }


def parse_state_definition(label, definition):
    if isinstance(definition, dict):
        level = definition.get("level_index")
        quantum_numbers = definition.get("quantum_numbers")
    elif isinstance(definition, str):
        level_match = re.search(r"(?:^|;)\s*level_index\s*=\s*(\d+)\s*(?:;|$)", definition)
        qn_match = re.search(r"(?:^|;)\s*quantum_numbers\s*=\s*(\{.*\})\s*$", definition)
        require(level_match is not None, f"cannot parse level_index for state {label!r}")
        require(qn_match is not None, f"cannot parse quantum_numbers for state {label!r}")
        level = int(level_match.group(1))
        try:
            quantum_numbers = json.loads(qn_match.group(1))
        except json.JSONDecodeError as exc:
            fail(f"invalid quantum_numbers for state {label!r}: {exc}")
    else:
        fail(f"unsupported state definition for {label!r}")

    require(isinstance(level, int) and not isinstance(level, bool) and level >= 0,
            f"state {label!r} has invalid level_index")
    require(isinstance(quantum_numbers, dict), f"state {label!r} quantum_numbers must be an object")
    require(set(quantum_numbers) == {"charge"}, f"state {label!r} must specify only charge")
    charge = quantum_numbers["charge"]
    require(isinstance(charge, int) and not isinstance(charge, bool),
            f"state {label!r} charge must be an integer")
    require(charge == 0, f"state {label!r} must remain in the charge-zero sector")
    return {"level_index": level, "charge": charge}


def parse_coefficient(value, name):
    if isinstance(value, dict):
        require(set(value).issubset({"real", "imag"}), f"{name} has unsupported fields")
        real = finite_real(value.get("real", 0.0), f"{name}.real")
        imag = finite_real(value.get("imag", 0.0), f"{name}.imag")
        return complex(real, imag)
    return complex(finite_real(value, name), 0.0)


def validate_observable_spec(spec, state_labels):
    require(spec.get("observable_schema_version") == "1.0", "unsupported Observable Specification version")
    require(spec.get("name") == "user_observable", "unsupported observable name")
    require(spec.get("ordering") == "as_written", "observable ordering must remain as_written")
    require(
        spec.get("expression") == r"\bar{\psi}(z n) \gamma^- \psi(0), n^mu=(1,1), \gamma^- = \gamma^0 - \gamma^1",
        "observable operator structure or gamma-minus sign changed",
    )

    convention = spec.get("coordinate_convention")
    require(isinstance(convention, dict), "coordinate_convention must be an object")
    require(convention.get("mode") == "lattice_indices", "only lattice-index sampling is supported")
    require(convention.get("spacing_parameter") == "lattice_spacing_a", "unsupported spacing parameter")
    require(convention.get("temporal_spacing") == "same_as_spatial", "temporal pullback changed")
    require(convention.get("units") == "natural_units_c_1", "unsupported coordinate units")

    insertions = spec.get("insertions")
    require(isinstance(insertions, list) and len(insertions) == 2,
            "the ordered bilocal must contain exactly two insertions")
    require(
        insertions[0] == {
            "id": "bar_psi_zn",
            "operator": r"\bar{\psi}",
            "position": {"t_index": "z_index", "x_index": "z_index"},
        },
        "left observable insertion changed",
    )
    require(
        insertions[1] == {
            "id": "psi_0",
            "operator": r"\psi",
            "position": {"t_index": 0, "x_index": 0},
        },
        "right observable insertion changed",
    )

    requested = spec.get("requested_components")
    require(isinstance(requested, list), "requested_components must be a list")
    require(set(requested) == {"real", "imag"} and len(requested) == 2,
            "this reference requires exactly real and imag components")

    matrix_elements = spec.get("matrix_elements")
    require(isinstance(matrix_elements, list) and matrix_elements,
            "observable.matrix_elements must be a nonempty list")
    parsed_terms = []
    ids = set()
    for index, term in enumerate(matrix_elements):
        require(isinstance(term, dict), "each matrix-element term must be an object")
        term_id = term.get("id")
        require(isinstance(term_id, str) and term_id and term_id not in ids,
                "matrix-element IDs must be unique nonempty strings")
        ids.add(term_id)
        bra = term.get("bra")
        ket = term.get("ket")
        require(bra in state_labels, f"unknown matrix-element bra state {bra!r}")
        require(ket in state_labels, f"unknown matrix-element ket state {ket!r}")
        require("coefficient" in term, f"matrix-element term {term_id!r} lacks coefficient")
        parsed_terms.append({
            "id": term_id,
            "bra": bra,
            "ket": ket,
            "coefficient": parse_coefficient(term["coefficient"], f"matrix_elements[{index}].coefficient"),
        })

    sampling = spec.get("sampling")
    require(isinstance(sampling, dict), "sampling must be an object")
    require(sampling.get("mode") == "cartesian_product", "only cartesian_product sampling is supported")
    variables = sampling.get("variables")
    require(isinstance(variables, dict) and list(variables) == ["z_index"],
            "this model supports exactly the z_index sampling variable")
    z_values = variables["z_index"]
    require(z_values == [-2, -1, 0, 1, 2], "authoritative z_index sampling assignments changed")
    maximum = sampling.get("maximum_points")
    require(isinstance(maximum, int) and not isinstance(maximum, bool) and maximum >= len(z_values),
            "sampling.maximum_points is invalid")

    assignments = [
        {"sample_id": f"sample_{index:04d}", "values": {"z_index": z}}
        for index, z in enumerate(z_values)
    ]
    return parsed_terms, assignments


def annihilate(state, mode):
    if ((state >> mode) & 1) == 0:
        return None
    sign = -1 if ((state & ((1 << mode) - 1)).bit_count() & 1) else 1
    return state ^ (1 << mode), sign


def create(state, mode):
    if ((state >> mode) & 1) != 0:
        return None
    sign = -1 if ((state & ((1 << mode) - 1)).bit_count() & 1) else 1
    return state | (1 << mode), sign


def bilinear_matrix(basis, mode_count, one_body):
    index = {state: i for i, state in enumerate(basis)}
    rows, cols, data = [], [], []
    nz_p, nz_q = np.nonzero(np.abs(one_body) > 0.0)
    for col, state in enumerate(basis):
        for p, q in zip(nz_p.tolist(), nz_q.tolist()):
            first = annihilate(state, q)
            if first is None:
                continue
            middle, sign1 = first
            second = create(middle, p)
            if second is None:
                continue
            final, sign2 = second
            row = index.get(final)
            if row is not None:
                rows.append(row)
                cols.append(col)
                data.append(one_body[p, q] * sign1 * sign2)
    return sp.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (rows, cols)),
        shape=(len(basis), len(basis)),
        dtype=np.complex128,
    ).tocsr()


def build_full_hamiltonian(mass, coupling, spacing, wilson, mode_count):
    sigma1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma3 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    beta = sigma1
    alpha1 = sigma3

    one_body = np.zeros((mode_count, mode_count), dtype=np.complex128)

    def mode(site_offset, component):
        return 2 * site_offset + component

    onsite = beta * (mass + wilson / spacing)
    for site_offset in range(len(IMPLEMENTED_SITES)):
        for a in range(2):
            for b in range(2):
                one_body[mode(site_offset, a), mode(site_offset, b)] += onsite[a, b]

    forward = -1j * alpha1 / (2.0 * spacing) - wilson * beta / (2.0 * spacing)
    backward = 1j * alpha1 / (2.0 * spacing) - wilson * beta / (2.0 * spacing)
    for site_offset in range(len(IMPLEMENTED_SITES) - 1):
        for a in range(2):
            for b in range(2):
                one_body[mode(site_offset, a), mode(site_offset + 1, b)] += forward[a, b]
                one_body[mode(site_offset + 1, a), mode(site_offset, b)] += backward[a, b]

    full_basis = list(range(1 << mode_count))
    hamiltonian = bilinear_matrix(full_basis, mode_count, one_body)

    # Construct each direct finite S_x = c_x^\dagger beta c_x and square it
    # as an operator, rather than replacing it by an expectation value.
    for site_offset in range(len(IMPLEMENTED_SITES)):
        local = np.zeros((mode_count, mode_count), dtype=np.complex128)
        for a in range(2):
            for b in range(2):
                local[mode(site_offset, a), mode(site_offset, b)] = beta[a, b]
        scalar_density = bilinear_matrix(full_basis, mode_count, local)
        hamiltonian = hamiltonian - (coupling / spacing) * (scalar_density @ scalar_density)

    return hamiltonian.tocsr(), beta, alpha1


def sector_indices(mode_count, particle_number):
    return np.asarray(
        [state for state in range(1 << mode_count) if state.bit_count() == particle_number],
        dtype=np.int64,
    )


def transition_operator(source_basis, target_basis, mode, operation):
    target_index = {int(state): i for i, state in enumerate(target_basis)}
    rows, cols, data = [], [], []
    for col, raw_state in enumerate(source_basis):
        state = int(raw_state)
        result = annihilate(state, mode) if operation == "annihilate" else create(state, mode)
        if result is None:
            continue
        final, sign = result
        row = target_index.get(final)
        if row is not None:
            rows.append(row)
            cols.append(col)
            data.append(float(sign))
    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(target_basis), len(source_basis)),
        dtype=np.complex128,
    ).tocsr()


def hermiticity_error(matrix):
    delta = matrix - matrix.getH()
    return 0.0 if delta.nnz == 0 else float(np.max(np.abs(delta.data)))


def state_variance(matrix, vector, energy):
    residual = matrix @ vector - energy * vector
    value = float(np.vdot(residual, residual).real)
    return max(0.0, value)


def complex_matrix_element(
    bra_vector,
    bra_energy,
    ket_vector,
    t,
    h_minus,
    annihilators,
    creators,
    spin_matrix,
    spacing,
):
    value = 0.0j
    for a in range(2):
        for b in range(2):
            coefficient = spin_matrix[a, b]
            if coefficient == 0.0:
                continue
            intermediate = annihilators[b] @ ket_vector
            evolved = spla.expm_multiply((-1j * t) * h_minus, intermediate)
            restored = creators[a] @ evolved
            value += coefficient * np.vdot(bra_vector, restored)
    return np.exp(1j * bra_energy * t) * value / spacing


def atomic_write_json(path, result):
    target = os.path.abspath(path)
    directory = os.path.dirname(target) or "."
    require(os.path.isdir(directory), f"output directory does not exist: {directory}")
    fd, temporary = tempfile.mkstemp(prefix=".scipy_exact_", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Independent SciPy exact reference")
    parser.add_argument("--parameter-contract", required=True)
    parser.add_argument("--observable-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = load_json(args.parameter_contract)
    observable_spec = load_json(args.observable_spec)

    config = validate_parameter_contract(contract)
    state_targets = {
        label: parse_state_definition(label, config["external_definitions"][label])
        for label in config["external_labels"]
    }
    matrix_terms, assignments = validate_observable_spec(
        observable_spec, set(config["external_labels"])
    )

    h_full, beta, alpha1 = build_full_hamiltonian(
        config["mass"],
        config["coupling"],
        config["spacing"],
        config["wilson"],
        config["mode_count"],
    )

    full_dim = 1 << config["mode_count"]
    require(h_full.shape == (full_dim, full_dim), "full Fock-space Hamiltonian has wrong shape")
    h_hermiticity = hermiticity_error(h_full)
    require(h_hermiticity <= TOL, f"Hamiltonian is not Hermitian: error={h_hermiticity}")

    charges = np.asarray(
        [state.bit_count() - config["site_count"] for state in range(full_dim)],
        dtype=np.float64,
    )
    q_op = sp.diags(charges, format="csr")
    commutator = h_full @ q_op - q_op @ h_full
    charge_commutator_error = (
        0.0 if commutator.nnz == 0 else float(np.max(np.abs(commutator.data)))
    )
    require(
        charge_commutator_error <= TOL,
        f"Hamiltonian does not preserve Q_lat: commutator error={charge_commutator_error}",
    )

    q0_basis = sector_indices(config["mode_count"], config["site_count"])
    qm1_basis = sector_indices(config["mode_count"], config["site_count"] - 1)
    require(len(q0_basis) == math.comb(config["mode_count"], config["site_count"]),
            "charge-zero physical sector is empty or has wrong dimension")
    require(len(qm1_basis) == math.comb(config["mode_count"], config["site_count"] - 1),
            "charge-minus-one intermediate sector is empty or has wrong dimension")

    h0 = h_full[q0_basis, :][:, q0_basis].tocsr()
    hm1 = h_full[qm1_basis, :][:, qm1_basis].tocsr()
    require(hermiticity_error(h0) <= TOL, "charge-zero Hamiltonian block is not Hermitian")
    require(hermiticity_error(hm1) <= TOL, "charge-minus-one Hamiltonian block is not Hermitian")

    dense_h0 = h0.toarray()
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        dense_h0, overwrite_a=False, check_finite=True, driver="evr"
    )

    max_level = max(target["level_index"] for target in state_targets.values())
    require(max_level < len(eigenvalues), "requested state level exceeds charge-zero sector dimension")

    states = {}
    max_residual = 0.0
    for label in config["external_labels"]:
        level = state_targets[label]["level_index"]
        vector = np.asarray(eigenvectors[:, level], dtype=np.complex128)
        energy = float(eigenvalues[level])
        variance = state_variance(h0, vector, energy)
        residual = math.sqrt(variance)
        max_residual = max(max_residual, residual)
        states[label] = {
            "vector": vector,
            "energy": energy,
            "energy_variance": variance,
            "level_index": level,
        }

    require("vacuum" in states and states["vacuum"]["level_index"] == 0,
            "vacuum must be charge-zero level_index 0")
    require("hadron" in states and states["hadron"]["level_index"] == 1,
            "hadron must be charge-zero level_index 1")
    require(max_residual <= TOL, f"eigensolver residual exceeds tolerance: {max_residual}")

    site_to_offset = {site: i for i, site in enumerate(IMPLEMENTED_SITES)}
    origin_offset = site_to_offset[0]

    annihilator_cache = {}
    creator_cache = {}

    def annihilator(site, component):
        key = (site, component)
        if key not in annihilator_cache:
            mode = 2 * site_to_offset[site] + component
            annihilator_cache[key] = transition_operator(
                q0_basis, qm1_basis, mode, "annihilate"
            )
        return annihilator_cache[key]

    def creator(site, component):
        key = (site, component)
        if key not in creator_cache:
            mode = 2 * site_to_offset[site] + component
            creator_cache[key] = transition_operator(
                qm1_basis, q0_basis, mode, "create"
            )
        return creator_cache[key]

    origin_annihilators = [annihilator(0, component) for component in range(2)]
    spin_matrix = np.eye(2, dtype=np.complex128) - alpha1

    observable_samples = []
    eps = np.finfo(float).eps
    h_scale = float(spla.norm(hm1))

    for assignment in assignments:
        z_index = assignment["values"]["z_index"]
        require(z_index in site_to_offset, f"sample site {z_index} is outside the lattice")
        t = float(z_index) * config["spacing"]
        endpoint_creators = [creator(z_index, component) for component in range(2)]

        combined = 0.0j
        for term in matrix_terms:
            bra = states[term["bra"]]
            ket = states[term["ket"]]
            element = complex_matrix_element(
                bra["vector"],
                bra["energy"],
                ket["vector"],
                t,
                hm1,
                origin_annihilators,
                endpoint_creators,
                spin_matrix,
                config["spacing"],
            )
            combined += term["coefficient"] * element

        require(math.isfinite(combined.real) and math.isfinite(combined.imag),
                f"non-finite observable result for {assignment['sample_id']}")

        uncertainty = float(
            128.0 * eps * (1.0 + abs(combined)) * (1.0 + abs(t) * h_scale)
        )
        observable_samples.append({
            "sample_id": assignment["sample_id"],
            "sample_values": dict(assignment["values"]),
            "observable_real": float(combined.real),
            "observable_imag": float(combined.imag),
            "uncertainty_real": uncertainty,
            "uncertainty_imag": uncertainty,
        })

    state_results = {
        label: {
            "energy": float(states[label]["energy"]),
            "energy_variance": float(states[label]["energy_variance"]),
        }
        for label in config["external_labels"]
    }

    result = {
        "reference_schema_version": REFERENCE_SCHEMA_VERSION,
        "method": METHOD_ID,
        "ground_state_energy": float(states["vacuum"]["energy"]),
        "energy_variance": float(states["vacuum"]["energy_variance"]),
        "state_results": state_results,
        "observable_samples": observable_samples,
        "convergence": {
            "passed": True,
            "details": {
                "full_fock_dimension": int(full_dim),
                "charge_zero_dimension": int(len(q0_basis)),
                "charge_minus_one_dimension": int(len(qm1_basis)),
                "hamiltonian_hermiticity_error": h_hermiticity,
                "charge_commutator_error": charge_commutator_error,
                "maximum_eigenstate_residual_norm": float(max_residual),
                "boundary_condition": IMPLEMENTED_BOUNDARY,
                "gauge_structure": "none",
                "observable_sector_path": "Q=0 -> Q=-1 -> Q=0",
                "lightcone_convention": {
                    "coordinate_definition": "x^plus/minus=x^0 plus/minus x^1",
                    "gamma_definition": "gamma^plus/minus=gamma^0 plus/minus gamma^1",
                    "normalization": 1.0,
                    "applied_gamma_minus": "gamma^0-gamma^1",
                    "beta_gamma_minus": "I-alpha1",
                },
            },
        },
    }

    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
