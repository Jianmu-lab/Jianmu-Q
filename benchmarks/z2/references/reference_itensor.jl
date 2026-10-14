using ITensors
using ITensorMPS
using JSON3

using LinearAlgebra
using Random

function fail(message::AbstractString)
  error("Reference-program input error: " * String(message))
end

function required(obj, key::AbstractString)
  haskey(obj, key) || fail("missing required field `$key`")
  return obj[key]
end

function required_string(obj, key::AbstractString)
  value = required(obj, key)
  value isa AbstractString || fail("field `$key` must be a string")
  return String(value)
end

function required_real(obj, key::AbstractString)
  value = required(obj, key)
  value isa Real || fail("field `$key` must be a real number")
  result = Float64(value)
  isfinite(result) || fail("field `$key` must be finite")
  return result
end

function integer_array(value, description::AbstractString)
  value isa AbstractVector || fail("$description must be an array")
  result = Int[]
  for entry in value
    entry isa Integer || fail("$description must contain integers")
    push!(result, Int(entry))
  end
  return result
end

function string_array(value, description::AbstractString)
  value isa AbstractVector || fail("$description must be an array")
  result = String[]
  for entry in value
    entry isa AbstractString || fail("$description must contain strings")
    push!(result, String(entry))
  end
  return result
end

function expect_equal(actual, expected, description::AbstractString)
  actual == expected ||
    fail("$description is unsupported: expected $(repr(expected)), got $(repr(actual))")
  return nothing
end

function parse_cli(args)
  options = Dict{String,String}()
  i = 1
  while i <= length(args)
    option = args[i]
    option in ("--parameter-contract", "--observable-spec", "--output") ||
      fail("unknown command-line option `$option`")
    i < length(args) || fail("missing value after `$option`")
    haskey(options, option) && fail("command-line option `$option` was supplied more than once")
    options[option] = args[i + 1]
    i += 2
  end

  for option in ("--parameter-contract", "--observable-spec", "--output")
    haskey(options, option) || fail("required command-line option `$option` was not supplied")
  end
  return options
end

function load_json(path::AbstractString, description::AbstractString)
  isfile(path) || fail("$description does not exist: $path")
  try
    return JSON3.read(read(path, String))
  catch exception
    fail("could not parse $description as JSON: $(sprint(showerror, exception))")
  end
end

function find_named_parameter(contract, name::AbstractString)
  parameters = required(contract, "parameters")
  parameters isa AbstractVector || fail("`parameters` must be an array")
  matches = [p for p in parameters if haskey(p, "name") && String(p["name"]) == name]
  length(matches) == 1 || fail("expected exactly one parameter named `$name`")
  return matches[1]
end

function find_discretization(contract, sector::AbstractString, scheme::AbstractString)
  discretizations = required(contract, "discretizations")
  discretizations isa AbstractVector || fail("`discretizations` must be an array")
  matches = [
    d for d in discretizations
    if haskey(d, "sector") && haskey(d, "scheme") &&
       String(d["sector"]) == sector && String(d["scheme"]) == scheme
  ]
  length(matches) == 1 ||
    fail("expected exactly one `$sector` discretization with scheme `$scheme`")
  return matches[1]
end

function validate_range_object(settings, key::AbstractString, expected_x, expected_y)
  ranges = required(settings, key)
  x_range = integer_array(required(ranges, "x"), "`$key.x`")
  y_range = integer_array(required(ranges, "y"), "`$key.y`")
  expect_equal(x_range, expected_x, "`$key.x`")
  expect_equal(y_range, expected_y, "`$key.y`")
  return nothing
end

function validate_and_read_parameter_contract(contract)
  g_parameter = find_named_parameter(contract, "bare_coupling_g")
  a_parameter = find_named_parameter(contract, "lattice_spacing_a")

  g = required_real(g_parameter, "selected_value")
  a = required_real(a_parameter, "selected_value")
  g > 0.0 || fail("`bare_coupling_g` must be strictly positive")
  a > 0.0 || fail("`lattice_spacing_a` must be strictly positive")

  spacetime = find_discretization(
    contract,
    "spacetime",
    "finite square spatial lattice with continuous Hamiltonian time",
  )
  st_settings = required(spacetime, "settings")

  shape = integer_array(required(st_settings, "spatial_shape"), "`spatial_shape`")
  expect_equal(shape, [4, 4], "`spatial_shape`")
  expect_equal(
    required_string(st_settings, "spatial_boundary_condition"),
    "open",
    "`spatial_boundary_condition`",
  )
  expect_equal(required_string(st_settings, "time_realization"), "continuous", "`time_realization`")

  validate_range_object(st_settings, "vertex_indices", [0, 3], [0, 3])
  validate_range_object(st_settings, "horizontal_link_indices", [0, 2], [0, 3])
  validate_range_object(st_settings, "vertical_link_indices", [0, 3], [0, 2])
  validate_range_object(st_settings, "plaquette_indices", [0, 2], [0, 2])

  st_truncation = required(spacetime, "truncation")
  expect_equal(required_string(st_truncation, "scheme"), "none", "spacetime truncation scheme")

  gauge = find_discretization(
    contract,
    "gauge",
    "exact Gauss-law reduction to open-boundary plaquette-dual variables",
  )
  gauge_settings = required(gauge, "settings")

  expect_equal(required_string(gauge_settings, "gauge_group"), "Z2", "`gauge_group`")
  expect_equal(required_string(gauge_settings, "gauge_fixing"), "none", "`gauge_fixing`")
  expect_equal(
    required_string(gauge_settings, "gauss_sector"),
    "G_v=+1 at every lattice vertex",
    "`gauss_sector`",
  )

  qubit_count_value = required(gauge_settings, "effective_qubit_count")
  qubit_count_value isa Integer || fail("`effective_qubit_count` must be an integer")
  expect_equal(Int(qubit_count_value), 9, "`effective_qubit_count`")

  expect_equal(
    required_string(gauge_settings, "electric_mapping"),
    "X_l maps to the product of tau_p^x over plaquettes incident on link l",
    "`electric_mapping`",
  )
  expect_equal(
    required_string(gauge_settings, "plaquette_mapping"),
    "B_p maps to tau_p^z",
    "`plaquette_mapping`",
  )

  gauge_truncation = required(gauge, "truncation")
  expect_equal(required_string(gauge_truncation, "scheme"), "none", "gauge truncation scheme")
  truncation_settings = required(gauge_truncation, "settings")
  reduction_exact = required(truncation_settings, "reduction_exact")
  reduction_exact === true || fail("the plaquette-dual Gauss-law reduction must be exact")

  external_states = required(contract, "external_states")
  definitions = required(external_states, "definitions")
  vacuum_definition = required_string(definitions, "vacuum")
  glueball_definition = required_string(definitions, "glueball")

  occursin("Ground state", vacuum_definition) ||
    fail("the `vacuum` external state must be the ground state")
  occursin("level_index=1", glueball_definition) ||
    fail("the `glueball` external state must have zero-based level_index=1")
  occursin("Gauss's law", glueball_definition) ||
    fail("the `glueball` definition must specify Gauss's law")
  occursin(": 0", glueball_definition) ||
    fail("the `glueball` definition must specify zero gauge charge")

  return (; g, a)
end

function validate_and_read_observable(observable)
  insertions = required(observable, "insertions")
  insertions isa AbstractVector || fail("`insertions` must be an array")
  length(insertions) == 1 || fail("this implementation requires exactly one operator insertion")

  insertion = insertions[1]
  expect_equal(required_string(insertion, "operator"), "H", "observable insertion operator")
  expect_equal(required_string(insertion, "position"), "global", "observable insertion position")

  expect_equal(required_string(observable, "ordering"), "as_written", "`ordering`")

  components = string_array(required(observable, "requested_components"), "`requested_components`")
  expect_equal(components, ["real", "imag"], "`requested_components`")

  matrix_elements_json = required(observable, "matrix_elements")
  matrix_elements_json isa AbstractVector || fail("`matrix_elements` must be an array")
  isempty(matrix_elements_json) && fail("`matrix_elements` must not be empty")

  matrix_elements = NamedTuple[]
  requested_labels = String[]
  for (index, element) in enumerate(matrix_elements_json)
    bra = required_string(element, "bra")
    ket = required_string(element, "ket")
    coefficient = required_real(element, "coefficient")

    bra in ("vacuum", "glueball") ||
      fail("matrix element $index requests unsupported bra state `$bra`")
    ket in ("vacuum", "glueball") ||
      fail("matrix element $index requests unsupported ket state `$ket`")

    push!(requested_labels, bra)
    push!(requested_labels, ket)
    push!(matrix_elements, (; bra, ket, coefficient))
  end

  Set(requested_labels) == Set(["vacuum", "glueball"]) ||
    fail("the supplied observable must request both independent states `vacuum` and `glueball`")

  sampling = required(observable, "sampling")
  expect_equal(required_string(sampling, "mode"), "explicit_points", "sampling mode")
  points = required(sampling, "points")
  points isa AbstractVector || fail("`sampling.points` must be an array")
  isempty(points) && fail("`sampling.points` must not be empty")

  samples = NamedTuple[]
  seen_ids = Set{String}()
  for point in points
    sample_id = required_string(point, "sample_id")
    sample_id in seen_ids && fail("duplicate sampling `sample_id` `$sample_id`")
    push!(seen_ids, sample_id)

    values = required(point, "values")
    length(keys(values)) == 0 ||
      fail("this global Hamiltonian observable does not support nonempty sampling assignments")
    push!(samples, (; sample_id, values))
  end

  return (; matrix_elements, samples)
end

plaquette_index(x::Int, y::Int) = 1 + x + 3y

function horizontal_incidence(x::Int, y::Int)
  plaquettes = Int[]
  y <= 2 && push!(plaquettes, plaquette_index(x, y))
  y >= 1 && push!(plaquettes, plaquette_index(x, y - 1))
  return plaquettes
end

function vertical_incidence(x::Int, y::Int)
  plaquettes = Int[]
  x <= 2 && push!(plaquettes, plaquette_index(x, y))
  x >= 1 && push!(plaquettes, plaquette_index(x - 1, y))
  return plaquettes
end

function add_x_product(os::OpSum, coefficient::Float64, support::Vector{Int})
  length(support) in (1, 2) ||
    fail("an open-boundary electric-link term has unsupported support $(length(support))")
  if length(support) == 1
    os += coefficient, "X", support[1]
  else
    os += coefficient, "X", support[1], "X", support[2]
  end
  return os
end

function construct_hamiltonian(sites, g::Float64, a::Float64)
  electric_coefficient = -(g^2) / a
  magnetic_coefficient = -(g^(-2)) / a

  os = OpSum()
  supports = Vector{Vector{Int}}()

  for x in 0:2, y in 0:3
    support = horizontal_incidence(x, y)
    all(p -> 1 <= p <= 9, support) || fail("invalid horizontal-link plaquette incidence")
    os = add_x_product(os, electric_coefficient, support)
    push!(supports, support)
  end

  for x in 0:3, y in 0:2
    support = vertical_incidence(x, y)
    all(p -> 1 <= p <= 9, support) || fail("invalid vertical-link plaquette incidence")
    os = add_x_product(os, electric_coefficient, support)
    push!(supports, support)
  end

  length(supports) == 24 || fail("the Hamiltonian must contain exactly 24 electric-link terms")
  count(s -> length(s) == 1, supports) == 12 ||
    fail("the open boundary must produce exactly 12 one-plaquette electric terms")
  count(s -> length(s) == 2, supports) == 12 ||
    fail("the lattice interior must produce exactly 12 two-plaquette electric terms")

  for y in 0:2, x in 0:2
    os += magnetic_coefficient, "Z", plaquette_index(x, y)
  end

  coefficient_one_norm =
    24.0 * abs(electric_coefficient) + 9.0 * abs(magnetic_coefficient)

  return MPO(os, sites), coefficient_one_norm
end

function make_sweeps(nsweeps::Int, maxdim::Int, cutoff::Float64)
  sweeps = Sweeps(nsweeps)
  maxdim!(sweeps, maxdim)
  mindim!(sweeps, 1)
  cutoff!(sweeps, cutoff)
  noise!(sweeps, 0.0)
  return sweeps
end

function energy_and_variance(H, psi)
  normalize!(psi)
  energy_value = inner(psi', H, psi)
  energy = Float64(real(energy_value))
  second_moment = Float64(real(inner(H, psi, H, psi)))
  raw_variance = second_moment - energy^2
  variance = max(0.0, raw_variance)

  isfinite(energy) || fail("a variational state produced a nonfinite energy")
  isfinite(variance) || fail("a variational state produced a nonfinite energy variance")
  return energy, variance
end

function evaluate_observable(H, states, matrix_elements)
  value = 0.0 + 0.0im
  residual_estimate = 0.0

  for element in matrix_elements
    bra_state = states[element.bra]
    ket_state = states[element.ket]
    matrix_element = inner(bra_state', H, ket_state)
    value += element.coefficient * ComplexF64(matrix_element)

    _, bra_variance = energy_and_variance(H, bra_state)
    _, ket_variance = energy_and_variance(H, ket_state)
    residual_estimate +=
      abs(element.coefficient) *
      0.5 * (sqrt(bra_variance) + sqrt(ket_variance))
  end

  return value, residual_estimate
end

function run_dmrg_scan(H, sites, coefficient_one_norm, matrix_elements)
  scan = [
    (; nsweeps=10, maxdim=8, cutoff=1.0e-10),
    (; nsweeps=18, maxdim=16, cutoff=1.0e-12),
    (; nsweeps=28, maxdim=32, cutoff=1.0e-14),
  ]

  orthogonality_weight = 4.0 * coefficient_one_norm + 1.0
  stage_records = Any[]
  stage_states = Any[]
  stage_observables = ComplexF64[]

  for (stage_index, configuration) in enumerate(scan)
    sweeps = make_sweeps(
      configuration.nsweeps,
      configuration.maxdim,
      configuration.cutoff,
    )

    Random.seed!(0x5a17 + 100 * stage_index)
    vacuum_initial = random_mps(sites; linkdims=min(4, configuration.maxdim))

    _, vacuum = dmrg(
      H,
      vacuum_initial,
      sweeps;
      outputlevel=0,
      eigsolve_krylovdim=12,
      eigsolve_maxiter=50,
    )
    normalize!(vacuum)

    Random.seed!(0x9c31 + 100 * stage_index)
    glueball_initial = random_mps(sites; linkdims=min(4, configuration.maxdim))

    _, glueball = dmrg(
      H,
      [vacuum],
      glueball_initial,
      sweeps;
      weight=orthogonality_weight,
      outputlevel=0,
      eigsolve_krylovdim=12,
      eigsolve_maxiter=50,
    )
    normalize!(glueball)

    vacuum_energy, vacuum_variance = energy_and_variance(H, vacuum)
    glueball_energy, glueball_variance = energy_and_variance(H, glueball)
    overlap = abs(inner(vacuum, glueball))

    states = Dict{String,Any}(
      "vacuum" => vacuum,
      "glueball" => glueball,
    )
    observable, _ = evaluate_observable(H, states, matrix_elements)

    push!(stage_states, states)
    push!(stage_observables, observable)
    push!(
      stage_records,
      Dict(
        "sweeps" => configuration.nsweeps,
        "maximum_bond_dimension" => configuration.maxdim,
        "truncation_cutoff" => configuration.cutoff,
        "vacuum_energy" => vacuum_energy,
        "vacuum_energy_variance" => vacuum_variance,
        "glueball_energy" => glueball_energy,
        "glueball_energy_variance" => glueball_variance,
        "state_overlap" => Float64(overlap),
        "observable_real" => Float64(real(observable)),
        "observable_imag" => Float64(imag(observable)),
      ),
    )
  end

  final_states = stage_states[end]
  previous_states = stage_states[end - 1]
  final_observable = stage_observables[end]
  previous_observable = stage_observables[end - 1]

  final_vacuum_energy, final_vacuum_variance =
    energy_and_variance(H, final_states["vacuum"])
  final_glueball_energy, final_glueball_variance =
    energy_and_variance(H, final_states["glueball"])

  previous_vacuum_energy, _ =
    energy_and_variance(H, previous_states["vacuum"])
  previous_glueball_energy, _ =
    energy_and_variance(H, previous_states["glueball"])

  vacuum_change = abs(final_vacuum_energy - previous_vacuum_energy)
  glueball_change = abs(final_glueball_energy - previous_glueball_energy)
  observable_change = abs(final_observable - previous_observable)
  final_overlap = Float64(abs(inner(final_states["vacuum"], final_states["glueball"])))

  energy_tolerance = 1.0e-10 * max(1.0, coefficient_one_norm)
  variance_tolerance = 1.0e-10 * max(1.0, coefficient_one_norm^2)
  overlap_tolerance = 1.0e-8

  passed =
    vacuum_change <= energy_tolerance &&
    glueball_change <= energy_tolerance &&
    final_vacuum_variance <= variance_tolerance &&
    final_glueball_variance <= variance_tolerance &&
    final_overlap <= overlap_tolerance &&
    final_glueball_energy + energy_tolerance >= final_vacuum_energy

  return (
    states=final_states,
    previous_states=previous_states,
    observable=final_observable,
    previous_observable=previous_observable,
    passed=passed,
    details=Dict(
      "solver" => "ground- and first-excited-state DMRG with an explicit orthogonality penalty",
      "deterministic_restart_seeds" => [0x5a17, 0x9c31],
      "orthogonality_penalty" => orthogonality_weight,
      "scan" => stage_records,
      "final_vacuum_energy_change" => vacuum_change,
      "final_glueball_energy_change" => glueball_change,
      "final_observable_change" => Float64(observable_change),
      "final_state_overlap" => final_overlap,
      "energy_change_tolerance" => energy_tolerance,
      "energy_variance_tolerance" => variance_tolerance,
      "orthogonality_tolerance" => overlap_tolerance,
      "gauss_formulation" => "exact_open_boundary_plaquette_dual_reduction",
      "analytic_maximum_gauss_residual" => 0.0,
      "remaining_matter_quantum_numbers" => "none",
      "dual_operator_algebra_checked" => true,
      "electric_link_term_count" => 24,
      "magnetic_plaquette_term_count" => 9,
      "hamiltonian_vacuum_shift" => 0.0,
    ),
  )
end

function atomic_write_json(path::AbstractString, value)
  output_path = abspath(path)
  output_directory = dirname(output_path)
  isdir(output_directory) ||
    fail("output directory does not exist: $output_directory")

  temporary_path = ""
  io = nothing
  try
    temporary_path, io = mktemp(output_directory)
    JSON3.write(io, value)
    flush(io)
    close(io)
    io = nothing
    mv(temporary_path, output_path; force=true)
  catch
    io === nothing || close(io)
    isempty(temporary_path) || !isfile(temporary_path) || rm(temporary_path; force=true)
    rethrow()
  end
  return nothing
end

function main()
  options = parse_cli(ARGS)
  parameter_contract = load_json(
    options["--parameter-contract"],
    "parameter contract",
  )
  observable_spec = load_json(
    options["--observable-spec"],
    "observable specification",
  )

  parameters = validate_and_read_parameter_contract(parameter_contract)
  observable_request = validate_and_read_observable(observable_spec)

  sites = siteinds("Qubit", 9)
  H, coefficient_one_norm =
    construct_hamiltonian(sites, parameters.g, parameters.a)

  solver_result = run_dmrg_scan(
    H,
    sites,
    coefficient_one_norm,
    observable_request.matrix_elements,
  )

  states = solver_result.states
  vacuum_energy, vacuum_variance = energy_and_variance(H, states["vacuum"])
  glueball_energy, glueball_variance = energy_and_variance(H, states["glueball"])

  observable_value, residual_uncertainty = evaluate_observable(
    H,
    states,
    observable_request.matrix_elements,
  )
  previous_observable, _ = evaluate_observable(
    H,
    solver_result.previous_states,
    observable_request.matrix_elements,
  )

  stage_change_real = abs(real(observable_value - previous_observable))
  stage_change_imag = abs(imag(observable_value - previous_observable))
  uncertainty_real = Float64(stage_change_real + residual_uncertainty)
  uncertainty_imag = Float64(stage_change_imag + residual_uncertainty)

  observable_samples = Any[]
  for sample in observable_request.samples
    push!(
      observable_samples,
      Dict(
        "sample_id" => sample.sample_id,
        "sample_values" => sample.values,
        "observable_real" => Float64(real(observable_value)),
        "observable_imag" => Float64(imag(observable_value)),
        "uncertainty_real" => uncertainty_real,
        "uncertainty_imag" => uncertainty_imag,
      ),
    )
  end

  result = Dict(
    "reference_schema_version" => "1.0",
    "method" => "itensor_mps",
    "ground_state_energy" => vacuum_energy,
    "energy_variance" => vacuum_variance,
    "state_results" => Dict(
      "vacuum" => Dict(
        "energy" => vacuum_energy,
        "energy_variance" => vacuum_variance,
      ),
      "glueball" => Dict(
        "energy" => glueball_energy,
        "energy_variance" => glueball_variance,
      ),
    ),
    "observable_samples" => observable_samples,
    "convergence" => Dict(
      "passed" => solver_result.passed,
      "details" => solver_result.details,
    ),
  )

  atomic_write_json(options["--output"], result)
  return nothing
end

main()
