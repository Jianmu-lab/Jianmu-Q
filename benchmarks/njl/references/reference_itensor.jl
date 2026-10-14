using ITensors
using ITensorMPS
using JSON3
using LinearAlgebra
using Random
using Printf

function fail(message::AbstractString)
  error("Reference-program input error: " * message)
end

function parse_cli(args)
  values = Dict{String,String}()
  i = 1
  while i <= length(args)
    option = args[i]
    option in ("--parameter-contract", "--observable-spec", "--output") ||
      fail("unknown command-line option '$option'")
    i < length(args) || fail("missing value after '$option'")
    haskey(values, option) && fail("duplicate command-line option '$option'")
    values[option] = args[i + 1]
    i += 2
  end
  for option in ("--parameter-contract", "--observable-spec", "--output")
    haskey(values, option) || fail("required option '$option' was not supplied")
  end
  return values
end

function load_json(path::AbstractString)
  isfile(path) || fail("JSON input file does not exist: $path")
  return JSON3.read(read(path, String))
end

function require_property(object, name::Symbol, context::AbstractString)
  hasproperty(object, name) || fail("missing $context.$name")
  return getproperty(object, name)
end

function selected_parameter(contract, name::AbstractString)
  parameters = require_property(contract, :parameters, "parameter contract")
  matches = [p for p in parameters if String(require_property(p, :name, "parameter")) == name]
  length(matches) == 1 ||
    fail("expected exactly one parameter named '$name', found $(length(matches))")
  return require_property(only(matches), :selected_value, "parameter '$name'")
end

function find_discretization(contract, sector::AbstractString)
  discretizations = require_property(contract, :discretizations, "parameter contract")
  matches = [
    d for d in discretizations
    if String(require_property(d, :sector, "discretization")) == sector
  ]
  length(matches) == 1 ||
    fail("expected exactly one '$sector' discretization, found $(length(matches))")
  return only(matches)
end

function parse_state_definition(definition, label::AbstractString)
  definition isa AbstractString ||
    fail("state definition for '$label' must use the supported contract string form")
  text = String(definition)

  level_match = match(r"level_index\s*=\s*([0-9]+)", text)
  level_match === nothing && fail("state '$label' has no parseable level_index")

  charge_match = match(r"\"charge\"\s*:\s*(-?[0-9]+)", text)
  charge_match === nothing && fail("state '$label' has no parseable charge")

  return (
    level = parse(Int, level_match.captures[1]),
    charge = parse(Int, charge_match.captures[1]),
  )
end

function read_runtime_model(contract)
  m = Float64(selected_parameter(contract, "bare_quark_mass"))
  g = Float64(selected_parameter(contract, "bare_four_fermion_coupling"))
  a = Float64(selected_parameter(contract, "lattice_spacing"))
  L = Int(selected_parameter(contract, "spatial_site_count"))
  r = Float64(selected_parameter(contract, "wilson_parameter"))
  mode_count = Int(selected_parameter(contract, "fermionic_mode_count"))

  all(isfinite, (m, g, a, r)) || fail("all numerical couplings must be finite")
  a > 0 || fail("lattice spacing must be positive")
  L == 5 || fail("this implementation supports exactly five spatial sites")
  mode_count == 2L ||
    fail("fermionic_mode_count must equal two times spatial_site_count")

  spacetime = find_discretization(contract, "spacetime")
  st_settings = require_property(spacetime, :settings, "spacetime discretization")
  String(require_property(st_settings, :spatial_boundary_condition, "spacetime settings")) ==
    "open" || fail("only the authoritative open boundary condition is supported")

  spatial_dimension = Int(require_property(st_settings, :spatial_dimension, "spacetime settings"))
  spatial_dimension == 1 || fail("only one spatial dimension is supported")

  site_indices_raw =
    require_property(st_settings, :spatial_site_indices, "spacetime settings")
  site_coordinates = Int[x for x in site_indices_raw]
  site_coordinates == [-2, -1, 0, 1, 2] ||
    fail("spatial site indices must be exactly [-2,-1,0,1,2]")

  links_raw =
    require_property(st_settings, :nearest_neighbor_link_indices, "spacetime settings")
  links = [(Int(link[1]), Int(link[2])) for link in links_raw]
  links == [(-2, -1), (-1, 0), (0, 1), (1, 2)] ||
    fail("nearest-neighbor links do not match the supported open chain")

  fermion = find_discretization(contract, "fermion")
  String(require_property(fermion, :scheme, "fermion discretization")) ==
    "two-component Wilson fermions" ||
    fail("unsupported fermion discretization")
  fsettings = require_property(fermion, :settings, "fermion discretization")
  Int(require_property(fsettings, :components_per_site, "fermion settings")) == 2 ||
    fail("exactly two Dirac components per site are required")
  Int(require_property(fsettings, :fermionic_mode_count, "fermion settings")) == mode_count ||
    fail("inconsistent fermionic mode count")

  model = require_property(contract, :model, "parameter contract")
  String(require_property(model, :gauge_structure, "model")) == "none" ||
    fail("this matter-only implementation does not support gauge links or constraints")

  external_states = require_property(contract, :external_states, "parameter contract")
  labels = String[x for x in require_property(external_states, :labels, "external states")]
  isempty(labels) && fail("at least one named state is required")
  length(unique(labels)) == length(labels) || fail("state labels must be unique")

  definitions = require_property(external_states, :definitions, "external states")
  state_targets = Dict{String,NamedTuple{(:level, :charge),Tuple{Int,Int}}}()
  for label in labels
    symbol = Symbol(label)
    hasproperty(definitions, symbol) || fail("missing definition for state '$label'")
    target = parse_state_definition(getproperty(definitions, symbol), label)
    target.charge == 0 ||
      fail("this implementation supports the requested named states only in Q_lat=0")
    target.level in (0, 1) ||
      fail("this implementation supports Q_lat=0 level indices 0 and 1")
    state_targets[label] = target
  end

  levels = [state_targets[label].level for label in labels]
  length(unique(levels)) == length(levels) ||
    fail("independent named states may not silently share a level index")

  return (
    m = m,
    g = g,
    a = a,
    L = L,
    r = r,
    mode_count = mode_count,
    coordinates = site_coordinates,
    links = links,
    labels = labels,
    state_targets = state_targets,
  )
end

function read_observable(spec, coordinates)
  String(require_property(spec, :ordering, "observable specification")) == "as_written" ||
    fail("only the authoritative as-written operator ordering is supported")

  expression = String(require_property(spec, :expression, "observable specification"))
  occursin("\\gamma^-", expression) ||
    fail("observable must contain the authoritative gamma-minus insertion")

  insertions = require_property(spec, :insertions, "observable specification")
  length(insertions) == 2 || fail("the supported bilocal observable requires two insertions")
  String(require_property(insertions[1], :operator, "first insertion")) == "\\bar{\\psi}" ||
    fail("first insertion must be bar-psi")
  String(require_property(insertions[2], :operator, "second insertion")) == "\\psi" ||
    fail("second insertion must be psi")

  first_position = require_property(insertions[1], :position, "first insertion")
  String(require_property(first_position, :t_index, "first insertion position")) == "z_index" ||
    fail("first insertion time index must be z_index")
  String(require_property(first_position, :x_index, "first insertion position")) == "z_index" ||
    fail("first insertion spatial index must be z_index")

  second_position = require_property(insertions[2], :position, "second insertion")
  Int(require_property(second_position, :t_index, "second insertion position")) == 0 ||
    fail("second insertion time index must be zero")
  Int(require_property(second_position, :x_index, "second insertion position")) == 0 ||
    fail("second insertion spatial index must be zero")

  requested_components =
    String[x for x in require_property(spec, :requested_components, "observable specification")]
  Set(requested_components) == Set(["real", "imag"]) ||
    fail("this output implementation requires real and imag observable components")

  sampling = require_property(spec, :sampling, "observable specification")
  String(require_property(sampling, :mode, "sampling")) == "cartesian_product" ||
    fail("only Cartesian-product sampling is supported")
  variables = require_property(sampling, :variables, "sampling")
  z_values = Int[x for x in require_property(variables, :z_index, "sampling variables")]
  z_values == coordinates ||
    fail("z_index samples must be exactly the authoritative lattice coordinates")

  matrix_elements_raw =
    require_property(spec, :matrix_elements, "observable specification")
  isempty(matrix_elements_raw) &&
    fail("observable.matrix_elements is required for this implementation")

  matrix_elements = NamedTuple[]
  for term in matrix_elements_raw
    push!(
      matrix_elements,
      (
        id = String(require_property(term, :id, "matrix-element term")),
        bra = String(require_property(term, :bra, "matrix-element term")),
        ket = String(require_property(term, :ket, "matrix-element term")),
        coefficient = Float64(require_property(term, :coefficient, "matrix-element term")),
      ),
    )
  end
  all(isfinite(term.coefficient) for term in matrix_elements) ||
    fail("matrix-element coefficients must be finite")

  return (z_values = z_values, matrix_elements = matrix_elements)
end

function add_hamiltonian_terms!(
  os::OpSum,
  coordinates,
  coordinate_to_site,
  m::Float64,
  g::Float64,
  a::Float64,
  r::Float64,
)
  onsite_beta = m + r / a
  interaction = -g / a

  for x in coordinates
    i = coordinate_to_site[x]

    # beta = sigma_1.
    os += onsite_beta, "Cdagup", i, "Cdn", i
    os += onsite_beta, "Cdagdn", i, "Cup", i

    # Direct finite operator
    #   -(g/a) (c† beta c)^2
    # = -(g/a) (Nup + Ndn - 2 Nup*Ndn).
    os += interaction, "Nup", i
    os += interaction, "Ndn", i
    os += -2interaction, "Nupdn", i
  end

  for left_x in coordinates[1:(end - 1)]
    right_x = left_x + 1
    i = coordinate_to_site[left_x]
    j = coordinate_to_site[right_x]

    # A = -i*alpha1/(2a) - r*beta/(2a), alpha1 = sigma_3.
    A11 = -1im / (2a)
    A22 = 1im / (2a)
    A12 = -r / (2a)
    A21 = -r / (2a)

    os += A11, "Cdagup", i, "Cup", j
    os += A12, "Cdagup", i, "Cdn", j
    os += A21, "Cdagdn", i, "Cup", j
    os += A22, "Cdagdn", i, "Cdn", j

    os += conj(A11), "Cdagup", j, "Cup", i
    os += conj(A12), "Cdagdn", j, "Cup", i
    os += conj(A21), "Cdagup", j, "Cdn", i
    os += conj(A22), "Cdagdn", j, "Cdn", i
  end

  return os
end

function number_mpo(sites)
  os = OpSum()
  for i in eachindex(sites)
    os += 1.0, "Nup", i
    os += 1.0, "Ndn", i
  end
  return MPO(os, sites)
end

function deterministic_initial_mps(sites, variant::Int; linkdims::Int = 8)
  patterns = [
    ["Up", "Dn", "Up", "Dn", "Up"],
    ["Dn", "Up", "Dn", "Up", "Dn"],
    ["UpDn", "Emp", "Up", "Dn", "Up"],
    ["Emp", "UpDn", "Dn", "Up", "Dn"],
  ]
  pattern = patterns[mod1(variant, length(patterns))]
  return random_mps(sites, pattern; linkdims = linkdims)
end

function energy_and_variance(H::MPO, psi::MPS)
  norm2 = real(inner(psi, psi))
  norm2 > 0 || error("encountered a zero-norm variational state")
  energy = real(inner(psi', H, psi)) / norm2
  second = real(inner(H, psi, H, psi)) / norm2
  variance = max(0.0, second - energy^2)
  return energy, variance
end

function solve_states(H::MPO, sites)
  scan = [
    (nsweeps = 8, maxdim = 48, cutoff = 1.0e-9),
    (nsweeps = 12, maxdim = 96, cutoff = 1.0e-11),
    (nsweeps = 16, maxdim = 160, cutoff = 1.0e-12),
  ]

  previous_ground = nothing
  previous_excited = nothing
  previous_energies = nothing
  final_ground = nothing
  final_excited = nothing
  final_energy_changes = (Inf, Inf)

  for (stage, cfg) in enumerate(scan)
    sweeps = Sweeps(cfg.nsweeps)
    maxdim!(sweeps, cfg.maxdim)
    mindim!(sweeps, 1)
    cutoff!(sweeps, cfg.cutoff)
    noise!(sweeps, 1.0e-7, 1.0e-8, 0.0)

    ground_initial = if previous_ground === nothing
      deterministic_initial_mps(sites, 1; linkdims = min(8, cfg.maxdim))
    else
      copy(previous_ground)
    end

    _, ground = dmrg(
      H,
      ground_initial,
      sweeps;
      eigsolve_maxiter = 8,
      eigsolve_krylovdim = 5,
      outputlevel = 0,
    )
    normalize!(ground)

    excited_initial = if previous_excited === nothing
      deterministic_initial_mps(sites, 2; linkdims = min(8, cfg.maxdim))
    else
      copy(previous_excited)
    end

    _, excited = dmrg(
      H,
      [ground],
      excited_initial,
      sweeps;
      weight = 20.0,
      eigsolve_maxiter = 8,
      eigsolve_krylovdim = 5,
      outputlevel = 0,
    )
    normalize!(excited)

    overlap = abs(inner(ground, excited))
    overlap < 1.0e-6 ||
      error("excited-state DMRG failed its orthogonality check: overlap=$overlap")

    Eg, _ = energy_and_variance(H, ground)
    Ee, _ = energy_and_variance(H, excited)

    if previous_energies !== nothing
      final_energy_changes = (
        abs(Eg - previous_energies[1]),
        abs(Ee - previous_energies[2]),
      )
    end

    previous_ground = ground
    previous_excited = excited
    previous_energies = (Eg, Ee)
    final_ground = ground
    final_excited = excited
  end

  return (
    level_states = Dict(0 => final_ground, 1 => final_excited),
    energy_changes = final_energy_changes,
    scan = scan,
  )
end

function apply_jw_local(
  psi::MPS,
  sites,
  target::Int,
  operator_name::AbstractString,
)
  1 <= target <= length(sites) || error("Jordan-Wigner target is out of range")
  result = copy(psi)

  # The physical Electron sites contain both spinor components. The parity
  # string over all preceding physical sites is the exact Jordan-Wigner string.
  for j in 1:(target - 1)
    F = op("F", sites[j])
    result[j] = replaceprime(product(F, result[j]), 1 => 0)
  end

  local_operator = op(operator_name, sites[target])
  result[target] =
    replaceprime(product(local_operator, result[target]), 1 => 0)
  return result
end

function normalized_number_check(N::MPO, psi::MPS, expected_N::Int; tolerance = 1.0e-8)
  norm2 = real(inner(psi, psi))
  if norm2 <= 1.0e-28
    return (zero_branch = true, mean = nothing, variance = nothing, passed = true)
  end

  mean_N = real(inner(psi', N, psi)) / norm2
  second_N = real(inner(N, psi, N, psi)) / norm2
  variance_N = max(0.0, second_N - mean_N^2)
  passed = abs(mean_N - expected_N) <= tolerance && variance_N <= tolerance
  return (
    zero_branch = false,
    mean = mean_N,
    variance = variance_N,
    passed = passed,
  )
end

function scale_mps!(psi::MPS, coefficient::Number)
  psi[1] *= coefficient
  return psi
end

function compressed_add(x::MPS, y::MPS; cutoff::Float64, maxdim::Int)
  return add(x, y; cutoff = cutoff, maxdim = maxdim)
end

function taylor_step(
  H::MPO,
  psi::MPS,
  dt::Float64;
  order::Int,
  cutoff::Float64,
  maxdim::Int,
)
  accumulated = copy(psi)
  term = copy(psi)

  for k in 1:order
    term = apply(H, term; cutoff = cutoff, maxdim = maxdim)
    scale_mps!(term, (-1im * dt) / k)
    accumulated =
      compressed_add(accumulated, term; cutoff = cutoff, maxdim = maxdim)
  end

  return accumulated
end

function evolve_mps(
  H::MPO,
  psi::MPS,
  time::Float64;
  nsteps::Int,
  order::Int = 6,
  cutoff::Float64 = 1.0e-12,
  maxdim::Int = 256,
)
  nsteps > 0 || error("nsteps must be positive")
  time == 0 && return copy(psi)

  dt = time / nsteps
  result = copy(psi)
  for _ in 1:nsteps
    result = taylor_step(
      H,
      result,
      dt;
      order = order,
      cutoff = cutoff,
      maxdim = maxdim,
    )
  end
  return result
end

function bilocal_term(
  H::MPO,
  sites,
  coordinate_to_site,
  bra_state::MPS,
  ket_state::MPS,
  z::Int,
  a::Float64;
  nsteps::Int,
)
  origin_site = coordinate_to_site[0]
  z_site = coordinate_to_site[z]

  # (I-alpha1) = diag(0,2), so the ordered bilocal is
  # (2/a) c†_{z,dn}(t) c_{0,dn}.
  inserted_ket = apply_jw_local(ket_state, sites, origin_site, "Cdn")
  time = z * a

  evolved_bra = evolve_mps(H, bra_state, time; nsteps = nsteps)
  evolved_inserted_ket = evolve_mps(H, inserted_ket, time; nsteps = nsteps)
  restored = apply_jw_local(evolved_inserted_ket, sites, z_site, "Cdagdn")

  return (2.0 / a) * inner(evolved_bra, restored)
end

function evaluate_observable(
  H::MPO,
  N::MPO,
  sites,
  coordinate_to_site,
  states::Dict{String,MPS},
  matrix_elements,
  z_values,
  a::Float64,
)
  samples = Any[]
  maximum_time_scan_change = 0.0
  intermediate_checks = Any[]

  origin_site = coordinate_to_site[0]
  for (label, state) in states
    inserted = apply_jw_local(state, sites, origin_site, "Cdn")
    check = normalized_number_check(N, inserted, 4)
    check.passed ||
      error("the c_0 intermediate branch for '$label' is not in particle number N=4")
    push!(
      intermediate_checks,
      Dict(
        "state" => label,
        "expected_particle_number" => 4,
        "zero_branch" => check.zero_branch,
        "mean_particle_number" => check.mean,
        "particle_number_variance" => check.variance,
      ),
    )
  end

  for (sample_index, z) in enumerate(z_values)
    absolute_time = abs(z * a)
    coarse_steps = max(1, Int(ceil(absolute_time / 0.10)))
    fine_steps = max(1, 2coarse_steps)

    coarse_value = 0.0 + 0.0im
    fine_value = 0.0 + 0.0im

    for term in matrix_elements
      haskey(states, term.bra) ||
        fail("matrix-element bra state '$(term.bra)' is not defined")
      haskey(states, term.ket) ||
        fail("matrix-element ket state '$(term.ket)' is not defined")

      coarse_value += term.coefficient * bilocal_term(
        H,
        sites,
        coordinate_to_site,
        states[term.bra],
        states[term.ket],
        z,
        a;
        nsteps = coarse_steps,
      )

      fine_value += term.coefficient * bilocal_term(
        H,
        sites,
        coordinate_to_site,
        states[term.bra],
        states[term.ket],
        z,
        a;
        nsteps = fine_steps,
      )
    end

    difference = fine_value - coarse_value
    uncertainty_real = abs(real(difference))
    uncertainty_imag = abs(imag(difference))
    maximum_time_scan_change = max(
      maximum_time_scan_change,
      uncertainty_real,
      uncertainty_imag,
    )

    all(
      isfinite,
      (
        real(fine_value),
        imag(fine_value),
        uncertainty_real,
        uncertainty_imag,
      ),
    ) || error("non-finite observable result at z_index=$z")

    push!(
      samples,
      Dict(
        "sample_id" => @sprintf("sample_%04d", sample_index - 1),
        "sample_values" => Dict("z_index" => z),
        "observable_real" => Float64(real(fine_value)),
        "observable_imag" => Float64(imag(fine_value)),
        "uncertainty_real" => Float64(uncertainty_real),
        "uncertainty_imag" => Float64(uncertainty_imag),
      ),
    )
  end

  return (
    samples = samples,
    maximum_time_scan_change = maximum_time_scan_change,
    intermediate_checks = intermediate_checks,
  )
end

function atomic_write_json(path::AbstractString, object)
  directory = dirname(abspath(path))
  isdir(directory) || mkpath(directory)

  temporary_path, io = mktemp(directory)
  try
    JSON3.write(io, object)
    write(io, '\n')
    flush(io)
    close(io)
    mv(temporary_path, path; force = true)
  catch
    isopen(io) && close(io)
    isfile(temporary_path) && rm(temporary_path; force = true)
    rethrow()
  end
end

function main()
  cli = parse_cli(ARGS)
  contract = load_json(cli["--parameter-contract"])
  observable_spec = load_json(cli["--observable-spec"])

  model = read_runtime_model(contract)
  observable = read_observable(observable_spec, model.coordinates)

  Random.seed!(0x5a17_2026)

  # One four-state Electron site represents both Dirac components at a spatial
  # site. Nf QNs exactly restrict all variational states to total occupation 5,
  # which is Q_lat = Nf - 5 = 0.
  sites = siteinds("Electron", model.L; conserve_nf = true)
  coordinate_to_site =
    Dict(x => i for (i, x) in enumerate(model.coordinates))

  hamiltonian_sum = OpSum()
  hamiltonian_sum = add_hamiltonian_terms!(
    hamiltonian_sum,
    model.coordinates,
    coordinate_to_site,
    model.m,
    model.g,
    model.a,
    model.r,
  )
  H = MPO(hamiltonian_sum, sites)
  N = number_mpo(sites)

  # Every Hamiltonian monomial has equal creation and annihilation count.
  # Therefore [H,N]=0 analytically, without independently compressed HN/NH MPOs.
  analytic_number_commutator_residual = 0.0

  solver = solve_states(H, sites)

  named_states = Dict{String,MPS}()
  for label in model.labels
    target = model.state_targets[label]
    named_states[label] = copy(solver.level_states[target.level])
  end

  state_results = Dict{String,Any}()
  maximum_variance = 0.0
  for label in model.labels
    energy, variance = energy_and_variance(H, named_states[label])
    isfinite(energy) || error("non-finite energy for state '$label'")
    isfinite(variance) && variance >= 0 ||
      error("invalid energy variance for state '$label'")
    state_results[label] = Dict(
      "energy" => Float64(energy),
      "energy_variance" => Float64(variance),
    )
    maximum_variance = max(maximum_variance, variance)
  end

  vacuum_labels = [
    label for label in model.labels
    if model.state_targets[label].level == 0
  ]
  length(vacuum_labels) == 1 ||
    fail("exactly one named level_index=0 state is required")
  ground_label = only(vacuum_labels)
  ground_energy = Float64(state_results[ground_label]["energy"])
  ground_variance = Float64(state_results[ground_label]["energy_variance"])

  observable_result = evaluate_observable(
    H,
    N,
    sites,
    coordinate_to_site,
    named_states,
    observable.matrix_elements,
    observable.z_values,
    model.a,
  )

  maximum_energy_change = maximum(solver.energy_changes)
  energy_tolerance = 1.0e-7
  variance_tolerance = 1.0e-7
  time_scan_tolerance = 1.0e-6

  convergence_passed =
    maximum_energy_change <= energy_tolerance &&
    maximum_variance <= variance_tolerance &&
    observable_result.maximum_time_scan_change <= time_scan_tolerance

  scan_details = [
    Dict(
      "nsweeps" => cfg.nsweeps,
      "maxdim" => cfg.maxdim,
      "cutoff" => cfg.cutoff,
    ) for cfg in solver.scan
  ]

  output = Dict(
    "reference_schema_version" => "1.0",
    "method" => "itensor_mps",
    "ground_state_energy" => ground_energy,
    "energy_variance" => ground_variance,
    "state_results" => state_results,
    "observable_samples" => observable_result.samples,
    "convergence" => Dict(
      "passed" => convergence_passed,
      "details" => Dict(
        "dmrg_scan" => scan_details,
        "final_ground_energy_change" => solver.energy_changes[1],
        "final_excited_energy_change" => solver.energy_changes[2],
        "maximum_energy_variance" => maximum_variance,
        "maximum_observable_time_scan_change" =>
          observable_result.maximum_time_scan_change,
        "energy_change_tolerance" => energy_tolerance,
        "energy_variance_tolerance" => variance_tolerance,
        "observable_time_scan_tolerance" => time_scan_tolerance,
        "charge_sector" => Dict(
          "named_state_particle_number" => 5,
          "named_state_Q_lat" => 0,
          "intermediate_particle_number" => 4,
          "intermediate_Q_lat" => -1,
          "analytic_H_number_commutator_residual" =>
            analytic_number_commutator_residual,
          "intermediate_checks" => observable_result.intermediate_checks,
        ),
        "gauge_formulation" => Dict(
          "type" => "non_gauge_matter_only",
          "gauss_law_constraints" => "not_applicable",
        ),
        "lightcone_convention" => Dict(
          "coordinate_definition" => "x^plus/minus=x^0 plus/minus x^1",
          "gamma_definition" => "gamma^plus/minus=gamma^0 plus/minus gamma^1",
          "normalization" => 1.0,
        ),
        "time_evolution" => Dict(
          "algorithm" => "compressed_MPS_Taylor",
          "Taylor_order" => 6,
          "coarse_max_step_magnitude" => 0.10,
          "fine_step_refinement_factor" => 2,
          "cutoff" => 1.0e-12,
          "maxdim" => 256,
        ),
      ),
    ),
  )

  atomic_write_json(cli["--output"], output)
  return nothing
end

main()
