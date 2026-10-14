using ITensors
using ITensorMPS
using JSON3
using LinearAlgebra
using Random
using Printf

# Independent ITensor/MPS reference implementation for the five-site,
# two-component Wilson-fermion discretization with exact Gauss-law elimination.

const METHOD_ID = "itensor_mps"
const REF_SCHEMA = "1.0"

fail(msg) = throw(ArgumentError(msg))

function required(obj, key::AbstractString)
    haskey(obj, key) || fail("Missing required JSON field: $key")
    return obj[key]
end

function as_float(x, name)
    x isa Number || fail("$name must be numeric")
    y = Float64(x)
    isfinite(y) || fail("$name must be finite")
    return y
end

function as_int(x, name)
    x isa Integer || fail("$name must be an integer")
    return Int(x)
end

function as_string(x, name)
    x isa AbstractString || fail("$name must be a string")
    return String(x)
end

function parameter_by_name(pc, name::String)
    pars = required(pc, "parameters")
    for p in pars
        if String(required(p, "name")) == name
            return p
        end
    end
    fail("Required parameter '$name' was not found")
end

function parameter_value(pc, name::String)
    p = parameter_by_name(pc, name)
    return required(p, "selected_value")
end

function discretization_by_sector(pc, sector::String)
    ds = required(pc, "discretizations")
    for d in ds
        if String(required(d, "sector")) == sector
            return d
        end
    end
    fail("Missing discretization for sector '$sector'")
end

function parse_cli(args)
    values = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--parameter-contract", "--observable-spec", "--output")
            i == length(args) && fail("Missing value after $arg")
            haskey(values, arg) && fail("CLI option $arg was supplied more than once")
            values[arg] = args[i + 1]
            i += 2
        else
            fail("Unknown CLI option: $arg")
        end
    end
    for key in ("--parameter-contract", "--observable-spec", "--output")
        haskey(values, key) || fail("Required CLI option $key was not supplied")
    end
    return values
end

function read_json(path::String)
    isfile(path) || fail("Input JSON file does not exist: $path")
    return JSON3.read(read(path, String))
end

function validate_and_extract(pc, obs)
    g = as_float(parameter_value(pc, "bare_gauge_coupling"),
                 "bare_gauge_coupling")
    m = as_float(parameter_value(pc, "bare_fermion_mass"),
                 "bare_fermion_mass")
    a = as_float(parameter_value(pc, "lattice_spacing"),
                 "lattice_spacing")
    at = as_float(parameter_value(pc, "temporal_lattice_spacing"),
                  "temporal_lattice_spacing")
    r = as_float(parameter_value(pc, "wilson_parameter"),
                 "wilson_parameter")
    nref = as_int(parameter_value(pc, "charge_reference_occupancy"),
                  "charge_reference_occupancy")
    lc = as_float(parameter_value(pc, "lightcone_normalization"),
                  "lightcone_normalization")

    g >= 0 || fail("bare_gauge_coupling must be nonnegative")
    a > 0 || fail("lattice_spacing must be positive")
    at > 0 || fail("temporal_lattice_spacing must be positive")
    r != 0 || fail("wilson_parameter must be nonzero")
    at == a || fail("This implementation requires temporal_lattice_spacing == lattice_spacing")
    r == 1.0 || fail("This implementation requires Wilson parameter r=1")
    nref == 1 || fail("This implementation requires charge reference occupancy n_ref=1")
    lc == 1.0 || fail("This implementation requires light-cone normalization 1")

    spacetime = discretization_by_sector(pc, "spacetime")
    stsettings = required(spacetime, "settings")
    as_int(required(stsettings, "spatial_site_count"),
           "spatial_site_count") == 5 ||
        fail("This implementation supports exactly five spatial sites")
    String(required(stsettings, "spatial_boundary_condition")) == "open" ||
        fail("This implementation supports only open spatial boundary conditions")
    as_int(required(stsettings, "spatial_link_count"),
           "spatial_link_count") == 4 ||
        fail("Expected exactly four spatial links")
    String(required(stsettings, "observable_site_map")) == "s(z)=z+2" ||
        fail("Unsupported observable site map")

    fermion = discretization_by_sector(pc, "fermion")
    fsettings = required(fermion, "settings")
    as_int(required(fsettings, "component_count_per_site"),
           "component_count_per_site") == 2 ||
        fail("Expected two fermion components per spatial site")
    as_int(required(fsettings, "fermionic_mode_count"),
           "fermionic_mode_count") == 10 ||
        fail("Expected ten fermionic modes")
    String(required(fsettings, "alpha")) == "sigma_3" ||
        fail("Unsupported alpha-matrix convention")
    String(required(fsettings, "beta")) == "sigma_1" ||
        fail("Unsupported beta-matrix convention")

    gauge = discretization_by_sector(pc, "gauge")
    String(required(gauge, "scheme")) ==
        "spatial axial gauge with exact open-boundary Gauss-law elimination" ||
        fail("This program implements only exact axial-gauge Gauss-law elimination")
    gsettings = required(gauge, "settings")
    as_int(required(gsettings, "independent_gauge_degree_count"),
           "independent_gauge_degree_count") == 0 ||
        fail("Independent gauge links are not supported")
    gtrunc = required(gauge, "truncation")
    String(required(gtrunc, "scheme")) == "none" ||
        fail("Gauge-flux truncation is forbidden in this implementation")

    states_obj = required(required(pc, "external_states"), "definitions")
    state_specs = Dict{String,NamedTuple{(:level, :charge),Tuple{Int,Int}}}()
    for (label_raw, definition_raw) in pairs(states_obj)
        label = String(label_raw)
        definition = as_string(definition_raw, "state definition for $label")
        lm = match(r"level_index\s*=\s*(\d+)", definition)
        cm = match(r"\"charge\"\s*:\s*(-?\d+)", definition)
        lm === nothing && fail("Could not parse level_index for state '$label'")
        cm === nothing && fail("Could not parse charge for state '$label'")
        state_specs[label] = (level=parse(Int, lm.captures[1]),
                              charge=parse(Int, cm.captures[1]))
    end
    isempty(state_specs) && fail("No external states were requested")
    all(v.charge == 0 for v in values(state_specs)) ||
        fail("Only the required neutral external-state sector is implemented")
    levels = sort(unique(v.level for v in values(state_specs)))
    levels == collect(0:maximum(levels)) ||
        fail("Requested levels must form a contiguous sequence starting at zero")

    expression = as_string(required(obs, "expression"), "observable expression")
    occursin("\\gamma^-", expression) ||
        fail("Observable must explicitly use gamma^-")
    occursin("\\bar{\\psi}", expression) ||
        fail("Unsupported observable operator structure")
    occursin("W", expression) ||
        fail("The required Wilson transporter is absent")

    ordering = as_string(required(obs, "ordering"), "observable ordering")
    ordering == "as_written" || fail("Only as-written operator ordering is supported")

    insertions = required(obs, "insertions")
    insertion_ids = Set(String(required(x, "id")) for x in insertions)
    insertion_ids == Set(["psibar", "psi"]) ||
        fail("Unsupported observable insertion set")

    if haskey(obs, "matrix_elements")
        mes = obs["matrix_elements"]
        length(mes) == 0 ||
            fail("Explicit observable.matrix_elements is not supported by this legacy phi/psi observable implementation")
    end
    haskey(state_specs, "phi") || fail("Legacy matrix element requires state 'phi'")
    haskey(state_specs, "psi") || fail("Legacy matrix element requires state 'psi'")

    requested_components = Set(
        String(x) for x in required(obs, "requested_components")
    )
    requested_components == Set(["real", "imag"]) ||
        fail("This output interface requires exactly real and imag components")

    sampling = required(obs, "sampling")
    String(required(sampling, "mode")) == "cartesian_product" ||
        fail("Only cartesian_product sampling is implemented")
    variables = required(sampling, "variables")
    z1_values = collect(required(variables, "z1"))
    z2_values = collect(required(variables, "z2"))
    isempty(z1_values) && fail("z1 sampling list is empty")
    isempty(z2_values) && fail("z2 sampling list is empty")

    samples = NamedTuple[]
    sample_index = 0
    for z1_raw in z1_values
        z1 = as_int(z1_raw, "sampling z1")
        for z2_raw in z2_values
            z2 = as_int(z2_raw, "sampling z2")
            z2 == 0 || fail("This Agent-2 observable requires z2=0")
            -2 <= z1 <= 2 || fail("z1=$z1 does not fit the five-site map s(z)=z+2")
            push!(samples,
                  (sample_id=@sprintf("sample_%04d", sample_index),
                   z1=z1,
                   z2=z2))
            sample_index += 1
        end
    end

    length(samples) <= as_int(required(sampling, "maximum_points"),
                              "sampling maximum_points") ||
        fail("Expanded sampling exceeds maximum_points")

    return (
        g=g, m=m, a=a, at=at, r=r, nref=nref, lc=lc,
        state_specs=state_specs, samples=samples,
    )
end

mode_index(site0::Int, component::Int) = 2 * site0 + component

function add_bilinear(os, coefficient, p::Int, q::Int)
    iszero(coefficient) && return os
    os += coefficient, "Cdag", p, "C", q
    return os
end

function coulomb_weights(link::Int)
    # X_link = S_link-Q/2 = sum_N w_N q_N.
    return [((N <= link) ? 1.0 : 0.0) - 0.5 for N in 0:4]
end

function build_hamiltonian(sites, pars)
    g, m, a, r = pars.g, pars.m, pars.a, pars.r
    os = OpSum()

    # On-site (m+r/a) beta, with beta=sigma_1.
    onsite = m + r / a
    for N in 0:4
        p1 = mode_index(N, 1)
        p2 = mode_index(N, 2)
        os = add_bilinear(os, onsite, p1, p2)
        os = add_bilinear(os, onsite, p2, p1)
    end

    # T = -i alpha/(2a)-r beta/(2a), alpha=sigma_3.
    T = ComplexF64[
        -im / (2a)   -r / (2a)
        -r / (2a)     im / (2a)
    ]
    for N in 0:3
        for α in 1:2, β in 1:2
            left = mode_index(N, α)
            right = mode_index(N + 1, β)
            os = add_bilinear(os, T[α, β], left, right)
            os = add_bilinear(os, conj(T[α, β]), right, left)
        end
    end

    # Exact diagonal Coulomb interaction:
    # (a g^2/2) sum_link (S_link-Q/2)^2.
    prefactor = a * g^2 / 2
    for link in 0:3
        wsite = coulomb_weights(link)
        d = Float64[]
        for N in 0:4, α in 1:2
            push!(d, wsite[N + 1])
        end
        constant = -sum(wsite) # q_N=n_N-1

        c0 = prefactor * constant^2
        if !iszero(c0)
            os += c0, "Id", 1
        end

        for p in 1:10
            cp = prefactor * (d[p]^2 + 2constant * d[p])
            if !iszero(cp)
                os += cp, "N", p
            end
        end

        for p in 1:9, q in (p + 1):10
            cpq = prefactor * 2d[p] * d[q]
            if !iszero(cpq)
                os += cpq, "N", p, "N", q
            end
        end
    end

    return MPO(os, sites)
end

function build_number_mpo(sites)
    os = OpSum()
    for p in eachindex(sites)
        os += 1.0, "N", p
    end
    return MPO(os, sites)
end

function deterministic_product_state(sites, which::Int)
    occupied_patterns = [
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10],
        [1, 2, 5, 8, 9],
        [2, 3, 6, 7, 10],
    ]
    pattern = occupied_patterns[mod1(which, length(occupied_patterns))]
    occupied = Set(pattern)
    labels = [p in occupied ? "1" : "0" for p in eachindex(sites)]
    return MPS(sites, labels)
end

function configured_sweeps(nsweep::Int, maxdim::Int, cutoff::Float64)
    sweeps = Sweeps(nsweep)
    maxdim!(sweeps, maxdim)
    mindim!(sweeps, 1)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, 1e-7, 1e-8, 1e-9, 0.0)
    return sweeps
end

function expectation(A, psi)
    return inner(psi', A, psi)
end

function energy_and_variance(H, psi)
    n2 = real(inner(psi, psi))
    n2 > 0 || fail("Encountered a zero-norm variational state")
    e = real(inner(psi', H, psi)) / n2
    e2 = real(inner(H, psi, H, psi)) / n2
    variance = max(0.0, e2 - e^2)
    isfinite(e) || fail("Non-finite state energy")
    isfinite(variance) || fail("Non-finite state energy variance")
    return e, variance
end

function normalized_number_check(Nop, psi, target::Int; tolerance=1e-8)
    n2 = real(inner(psi, psi))
    n2 <= 1e-28 && return (zero_branch=true, mean=NaN, variance=NaN)

    mean_n = real(inner(psi', Nop, psi)) / n2
    second_n = real(inner(Nop, psi, Nop, psi)) / n2
    var_n = max(0.0, second_n - mean_n^2)

    abs(mean_n - target) <= tolerance ||
        fail("Intermediate MPS has particle number $mean_n, expected $target")
    var_n <= tolerance ||
        fail("Intermediate MPS particle-number variance $var_n exceeds tolerance")

    return (zero_branch=false, mean=mean_n, variance=var_n)
end

function solve_state_scan(H, sites, max_level::Int)
    configs = [
        (nsweep=8,  maxdim=64,  cutoff=1e-9),
        (nsweep=14, maxdim=128, cutoff=1e-11),
        (nsweep=20, maxdim=256, cutoff=1e-12),
    ]

    previous = [deterministic_product_state(sites, level + 1)
                for level in 0:max_level]
    energy_history = Vector{Vector{Float64}}()

    for cfg in configs
        solved = MPS[]
        energies = Float64[]
        sweeps = configured_sweeps(cfg.nsweep, cfg.maxdim, cfg.cutoff)

        for level in 0:max_level
            initial = previous[level + 1]
            if level == 0
                energy, state = dmrg(
                    H, initial, sweeps;
                    outputlevel=0,
                    eigsolve_maxiter=20,
                )
            else
                energy, state = dmrg(
                    H, solved, initial, sweeps;
                    weight=20.0,
                    outputlevel=0,
                    eigsolve_maxiter=30,
                )
            end
            normalize!(state)
            push!(solved, state)
            push!(energies, Float64(real(energy)))
        end

        previous = solved
        push!(energy_history, energies)
    end

    final_states = previous
    final_energies = energy_history[end]
    previous_energies = energy_history[end - 1]
    energy_changes = abs.(final_energies .- previous_energies)

    return final_states, final_energies, energy_changes, energy_history
end

function local_fermion_insertion(psi, sites, p::Int, opname::String;
                                  cutoff=1e-13, maxdim=512)
    1 <= p <= length(sites) || fail("Fermion insertion index is out of range")
    opname in ("C", "Cdag") || fail("Unsupported odd fermion operator: $opname")

    # A parity-odd operator is applied directly, never converted to an MPO.
    # The explicit F string realizes the Jordan-Wigner convention.
    ops = ITensor[]
    for j in 1:(p - 1)
        push!(ops, op("F", sites[j]))
    end
    push!(ops, op(opname, sites[p]))

    return product(ops, psi; cutoff=cutoff, maxdim=maxdim)
end

function a0_coefficients(target_site0::Int, a::Float64, g::Float64)
    0 <= target_site0 <= 4 || fail("A0 target site is out of range")

    site_coeff = zeros(Float64, 5)
    constant = 0.0
    for link in 0:3
        outer_weight = ((target_site0 <= link) ? 1.0 : 0.0) - 0.5
        inner_weights = coulomb_weights(link)
        for N in 0:4
            site_coeff[N + 1] += a * g * outer_weight * inner_weights[N + 1]
        end
        constant += a * g * outer_weight * (-sum(inner_weights))
    end

    mode_coeff = Float64[]
    for N in 0:4, α in 1:2
        push!(mode_coeff, site_coeff[N + 1])
    end
    return constant, mode_coeff
end

function apply_a0_exponential(psi, sites, target_site0::Int,
                              sign::Int, pars;
                              cutoff=1e-13, maxdim=512)
    sign in (-1, 1) || fail("Wilson-line orientation sign must be ±1")
    constant, coefficients =
        a0_coefficients(target_site0, pars.a, pars.g)

    # U = exp[-i sign*g*a*A0]. Since A0 is diagonal and linear in N_p,
    # this is an exact product of one-mode diagonal gates.
    theta = sign * pars.g * pars.a
    gates = ITensor[]
    for p in eachindex(sites)
        phase = exp(-im * theta * coefficients[p])
        gate = op("Id", sites[p]) + (phase - 1) * op("N", sites[p])
        push!(gates, gate)
    end

    result = product(gates, psi; cutoff=cutoff, maxdim=maxdim)
    result[1] = exp(-im * theta * constant) * result[1]
    return result
end

function evolve(H, psi, time::Float64, dt::Float64;
                cutoff=1e-12, maxdim=512)
    abs(time) <= eps(Float64) && return copy(psi)
    dt > 0 || fail("TDVP time step must be positive")
    nsteps = max(1, ceil(Int, abs(time) / dt))
    return tdvp(
        H, -im * time, psi;
        nsteps=nsteps,
        cutoff=cutoff,
        maxdim=maxdim,
        normalize=false,
        outputlevel=0,
    )
end

function apply_heisenberg_a0_unitary(H, psi, sites, k::Int,
                                     orientation::Int, pars, dt, Nop)
    t = k * pars.a
    x = evolve(H, psi, t, dt)
    normalized_number_check(Nop, x, 4)
    x = apply_a0_exponential(x, sites, k + 2, orientation, pars)
    normalized_number_check(Nop, x, 4)
    x = evolve(H, x, -t, dt)
    normalized_number_check(Nop, x, 4)
    return x
end

function observable_value(H, sites, phi, psi, z::Int, pars, dt, Nop)
    # (I-sigma_3)=diag(0,2), so only spinor component alpha=2 remains.
    center_mode = mode_index(2, 2)
    endpoint_mode = mode_index(z + 2, 2)

    # Rightmost psi(0) insertion: Q=0 (N=5) -> Q=-1 (N=4).
    x = local_fermion_insertion(psi, sites, center_mode, "C")
    if real(inner(x, x)) <= 1e-28
        return 0.0 + 0.0im
    end
    normalized_number_check(Nop, x, 4)

    # Apply the oriented Wilson transporter right-to-left.
    if z > 0
        for k in 0:(z - 1)
            x = apply_heisenberg_a0_unitary(
                H, x, sites, k, 1, pars, dt, Nop
            )
        end
    elseif z < 0
        for k in -1:-1:z
            x = apply_heisenberg_a0_unitary(
                H, x, sites, k, -1, pars, dt, Nop
            )
        end
    end
    normalized_number_check(Nop, x, 4)

    # cdag_endpoint(t), t=z*a:
    # exp(+iHt) cdag exp(-iHt), acting from the right.
    t = z * pars.a
    x = evolve(H, x, t, dt)
    normalized_number_check(Nop, x, 4)
    x = local_fermion_insertion(x, sites, endpoint_mode, "Cdag")
    if real(inner(x, x)) <= 1e-28
        return 0.0 + 0.0im
    end
    normalized_number_check(Nop, x, 5)
    x = evolve(H, x, -t, dt)
    normalized_number_check(Nop, x, 5)

    value = (2 / pars.a) * inner(phi, x)
    isfinite(real(value)) && isfinite(imag(value)) ||
        fail("Observable evaluation produced a non-finite value")
    return ComplexF64(value)
end

function atomic_write_json(path::String, object)
    directory = dirname(abspath(path))
    isdir(directory) || fail("Output directory does not exist: $directory")
    temporary = tempname(directory)
    try
        open(temporary, "w") do io
            JSON3.write(io, object)
            write(io, '\n')
            flush(io)
        end
        mv(temporary, path; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
end

function main()
    cli = parse_cli(ARGS)
    pc = read_json(cli["--parameter-contract"])
    obs = read_json(cli["--observable-spec"])
    pars = validate_and_extract(pc, obs)

    Random.seed!(0x51A7E)

    # Ten exact spinless occupation modes represent five two-component sites.
    sites = siteinds("Fermion", 10; conserve_qns=true)
    H = build_hamiltonian(sites, pars)
    Nop = build_number_mpo(sites)

    max_level = maximum(v.level for v in values(pars.state_specs))
    states, variational_energies, energy_changes, energy_history =
        solve_state_scan(H, sites, max_level)

    state_data = Dict{String,Any}()
    state_variances = Dict{String,Float64}()
    for label in sort(collect(keys(pars.state_specs)))
        spec = pars.state_specs[label]
        state = states[spec.level + 1]

        # Q_lat=N_total-5. QN flux fixes N_total=5 exactly; no annihilating
        # constraint MPO is applied to the state.
        normalized_number_check(Nop, state, 5)

        energy, variance = energy_and_variance(H, state)
        state_variances[label] = variance
        state_data[label] = Dict(
            "energy" => energy,
            "energy_variance" => variance,
        )
    end

    phi = states[pars.state_specs["phi"].level + 1]
    psi = states[pars.state_specs["psi"].level + 1]

    # Required time-step convergence scan.
    coarse_dt = pars.a / 8
    fine_dt = pars.a / 16
    observable_rows = Any[]
    observable_changes = Float64[]

    for sample in pars.samples
        z = sample.z1
        coarse = observable_value(
            H, sites, phi, psi, z, pars, coarse_dt, Nop
        )
        fine = observable_value(
            H, sites, phi, psi, z, pars, fine_dt, Nop
        )

        difference = abs(fine - coarse)
        push!(observable_changes, difference)

        uncertainty_real = abs(real(fine) - real(coarse))
        uncertainty_imag = abs(imag(fine) - imag(coarse))

        push!(observable_rows, Dict(
            "sample_id" => sample.sample_id,
            "sample_values" => Dict(
                "z1" => sample.z1,
                "z2" => sample.z2,
            ),
            "observable_real" => real(fine),
            "observable_imag" => imag(fine),
            "uncertainty_real" => uncertainty_real,
            "uncertainty_imag" => uncertainty_imag,
        ))
    end

    energy_tolerance = 1e-8
    variance_tolerance = 1e-8
    observable_tolerance = 1e-6

    maximum_energy_change = maximum(energy_changes)
    maximum_variance = maximum(values(state_variances))
    maximum_observable_change =
        isempty(observable_changes) ? 0.0 : maximum(observable_changes)

    convergence_passed =
        maximum_energy_change <= energy_tolerance &&
        maximum_variance <= variance_tolerance &&
        maximum_observable_change <= observable_tolerance

    phi_energy, phi_variance = energy_and_variance(H, phi)

    # Exact-reduction diagnostics:
    # E_N-E_{N-1}-g*q_N vanishes algebraically in every charge sector,
    # including Q=-1. Every Hamiltonian monomial contains equal Cdag/C
    # counts (or only number operators), hence [H,N_total]=[H,Q_lat]=0.
    convergence_details = Dict(
        "dmrg_scan" => Dict(
            "sweep_counts" => [8, 14, 20],
            "maximum_bond_dimensions" => [64, 128, 256],
            "cutoffs" => [1e-9, 1e-11, 1e-12],
            "energy_history" => energy_history,
            "maximum_final_energy_change" => maximum_energy_change,
            "energy_change_tolerance" => energy_tolerance,
        ),
        "tdvp_scan" => Dict(
            "coarse_time_step" => coarse_dt,
            "fine_time_step" => fine_dt,
            "maximum_observable_change" => maximum_observable_change,
            "observable_change_tolerance" => observable_tolerance,
        ),
        "maximum_energy_variance" => maximum_variance,
        "energy_variance_tolerance" => variance_tolerance,
        "exact_reduction_checks" => Dict(
            "gauge_links_eliminated" => true,
            "gauge_flux_truncation" => false,
            "analytic_gauss_residual" => 0.0,
            "analytic_charge_commutator_residual" => 0.0,
            "external_charge_sector" => 0,
            "external_total_occupation" => 5,
            "ordered_intermediate_charge" => -1,
            "ordered_intermediate_total_occupation" => 4,
            "energy_shift_applied" => false,
        ),
        "applied_convention" => Dict(
            "gamma_0" => "sigma_1",
            "gamma_1" => "-i*sigma_2",
            "gamma_minus" => "gamma_0-gamma_1",
            "lightcone_normalization" => pars.lc,
        ),
    )

    output = Dict(
        "reference_schema_version" => REF_SCHEMA,
        "method" => METHOD_ID,
        "ground_state_energy" => phi_energy,
        "energy_variance" => phi_variance,
        "state_results" => state_data,
        "observable_samples" => observable_rows,
        "convergence" => Dict(
            "passed" => convergence_passed,
            "details" => convergence_details,
        ),
    )

    atomic_write_json(cli["--output"], output)
    return nothing
end

main()
