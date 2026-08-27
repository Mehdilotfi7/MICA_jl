#!/usr/bin/env julia
# Profile-likelihood analysis for the wind-turbine application using BreakpointProfiles.jl.
# Upgraded to support Hybrid Optimizers, Adaptive D2D stepping, and Joint Profiling.
#
# Usage: julia turbine_ple_hybrid.jl <label> <cps_csv> <params_csv> <out_dir> [selection] [mode] [cp_selection]

using Pkg
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "codes", "BreakpointProfiles.jl"))
Pkg.instantiate()

using Mica
using BreakpointProfiles
using CSV, DataFrames, Statistics, Random
using Evolutionary, OrdinaryDiffEq
using Optim
using Printf
using Dates

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "..", "codes", "Mica.jl", "examples", "Wind_Turbine_model")

# Adaptive profiling options (D2D style)
const PLE_D2D_SAMPLESIZE       = parse(Int, get(ENV, "TURBINE_PLE_D2D_SAMPLESIZE", "100"))
const PLE_D2D_REL_STEP         = parse(Float64, get(ENV, "TURBINE_PLE_D2D_REL_STEP", "0.1"))
const PLE_D2D_STEP_FACTOR      = parse(Float64, get(ENV, "TURBINE_PLE_D2D_STEP_FACTOR", "1.5"))
const PLE_D2D_STOP_MARGIN      = parse(Float64, get(ENV, "TURBINE_PLE_D2D_STOP_MARGIN", "1.2"))
const PLE_D2D_POLISH           = parse(Bool, get(ENV, "TURBINE_PLE_D2D_POLISH", "true"))
const PLE_D2D_SMOOTH_JUMPS     = parse(Bool, get(ENV, "TURBINE_PLE_D2D_SMOOTH_JUMPS", "true"))
const PLE_D2D_ALLOW_BETTER     = parse(Bool, get(ENV, "TURBINE_PLE_D2D_ALLOW_BETTER", "false"))
const PLE_NPOINTS              = parse(Int, get(ENV, "TURBINE_PLE_NPOINTS", "20"))

# Optimizer selection
const PLE_MULTISTART      = parse(Int, get(ENV, "TURBINE_PLE_MULTISTART", "1"))
const PLE_MULTISTART_PERT = parse(Float64, get(ENV, "TURBINE_PLE_MULTISTART_PERTURBATION", "0.5"))
const PLE_REF_OPTIMIZER   = lowercase(get(ENV, "TURBINE_PLE_REF_OPTIMIZER", "hybrid"))
const PLE_PROF_OPTIMIZER  = lowercase(get(ENV, "TURBINE_PLE_OPTIMIZER", "neldermead"))

# Joint settings
const PLE_JOINT_WINDOW      = parse(Int, get(ENV, "TURBINE_PLE_JOINT_WINDOW", "20"))
const PLE_JOINT_CP_STEP     = parse(Int, get(ENV, "TURBINE_PLE_JOINT_CP_STEP", "5"))
const PLE_JOINT_NPOINTS     = parse(Int, get(ENV, "TURBINE_PLE_JOINT_NPOINTS", "10"))
const PLE_JOINT_OPTIMIZER   = lowercase(get(ENV, "TURBINE_PLE_JOINT_OPTIMIZER", "neldermead"))
const PLE_JOINT_REF_SEARCH  = parse(Bool, get(ENV, "TURBINE_PLE_JOINT_REF_SEARCH", "false"))
const PLE_JOINT_PARALLEL_OVER_PARAMS = parse(Bool, get(ENV, "TURBINE_PLE_JOINT_PARALLEL_OVER_PARAMS", "true"))
const PLE_JOINT_PARALLELIZE_GRID     = parse(Bool, get(ENV, "TURBINE_PLE_JOINT_PARALLELIZE_GRID", "true"))

# GA Settings
const PLE_POPULATION = parse(Int, get(ENV, "TURBINE_PLE_POPULATION", "80"))
const PLE_ITERATIONS = parse(Int, get(ENV, "TURBINE_PLE_ITERATIONS", "100"))
const PLE_PARALLEL   = Symbol(get(ENV, "TURBINE_PLE_PARALLEL", "serial"))
const PLE_SELECTION  = get(ENV, "TURBINE_PLE_SELECTION", "tournament(2)")
const PLE_CROSSOVER  = get(ENV, "TURBINE_PLE_CROSSOVER", "SBX(0.7, 1)")
const PLE_MUTATION   = get(ENV, "TURBINE_PLE_MUTATION", "gaussian(0.0001)")
const PLE_MUTATION_RATE = parse(Float64, get(ENV, "TURBINE_PLE_MUTATION_RATE", "0.7"))
const PLE_CROSSOVER_RATE = parse(Float64, get(ENV, "TURBINE_PLE_CROSSOVER_RATE", "0.7"))

const PLE_REF_POPULATION = parse(Int, get(ENV, "TURBINE_PLE_REF_POPULATION", "150"))
const PLE_REF_ITERATIONS = parse(Int, get(ENV, "TURBINE_PLE_REF_ITERATIONS", "1000"))
const PLE_PROF_POPULATION = parse(Int, get(ENV, "TURBINE_PLE_PROF_POPULATION", "80"))
const PLE_PROF_ITERATIONS = parse(Int, get(ENV, "TURBINE_PLE_PROF_ITERATIONS", "100"))

const BOUND_SCALE = parse(Float64, get(ENV, "TURBINE_PLE_BOUND_SCALE", "1.0"))
const CHISQ_95 = 3.8414588206941285

function build_optimizer(opt_name::String=PLE_PROF_OPTIMIZER; ref::Bool=false)
    pop = ref ? PLE_REF_POPULATION : PLE_PROF_POPULATION
    iters = ref ? PLE_REF_ITERATIONS : PLE_PROF_ITERATIONS
    if opt_name == "hybrid"
        ga = EvolutionaryPLEConfig(
            population_size=pop, iterations=iters, selection=PLE_SELECTION,
            crossover=PLE_CROSSOVER, mutation=PLE_MUTATION,
            mutation_rate=PLE_MUTATION_RATE, crossover_rate=PLE_CROSSOVER_RATE,
            parallelization=PLE_PARALLEL
        )
        local_opt = OptimPLEConfig(Fminbox(LBFGS()), Optim.Options(show_trace=false, iterations=200))
        return HybridOptimizerConfig(global_optimizer=ga, local_optimizer=local_opt)
    elseif opt_name == "evolutionary"
        return EvolutionaryPLEConfig(
            population_size=pop, iterations=iters, selection=PLE_SELECTION,
            crossover=PLE_CROSSOVER, mutation=PLE_MUTATION,
            mutation_rate=PLE_MUTATION_RATE, crossover_rate=PLE_CROSSOVER_RATE,
            parallelization=PLE_PARALLEL
        )
    elseif opt_name == "neldermead"
        return OptimPLEConfig(Fminbox(NelderMead()), Optim.Options(show_trace=false, iterations=2000))
    elseif opt_name == "lbfgs"
        return OptimPLEConfig(Fminbox(LBFGS()), Optim.Options(show_trace=false, iterations=200))
    elseif opt_name == "bobyqa"
        return NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1.0e-6, maxeval=10_000)
    else
        error("Unknown optimizer: $opt_name.")
    end
end

function example_difference_model(θ, T_initial, num_steps, extra_data)
    θ1, θ2, θ3, θ4, θ5, θ6, θ7 = θ.θ1, θ.θ2, θ.θ3, θ.θ4, θ.θ5, θ.θ6, θ.θ7
    wind_speeds, ambient_temperatures = extra_data
    generator_temperatures_sim = zeros(num_steps)
    generator_temperatures_sim[1] = T_initial
    for k in 2:num_steps
        u1, u2 = wind_speeds[k], ambient_temperatures[k]
        y_prev = generator_temperatures_sim[k-1]
        generator_temperatures_sim[k] = ((θ1*u1^3 + θ2*u1^2 + θ3*u1 + y_prev - u2) /
                                         (θ4*u1^3 + θ5*u1^2 + θ6*u1 + θ7)) + u2
    end
    return reshape(Float64.(generator_temperatures_sim), 1, :)
end

function loss_function(observed, simulated)
    return sum((observed .- simulated).^2)
end

function load_turbine_data()
    df = CSV.read(joinpath(EXAMPLE_DIR, "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv"), DataFrame)
    wind_speeds = df[:, "Wind speed (m/s)"][1:2500]
    ambient_temperatures = df[:, "Ambient temperature (converter) (°C)"][1:2500]
    generator_temperatures = df[:, "Generator bearing front temperature (°C)"][1:2500]
    return (wind_speeds, ambient_temperatures, generator_temperatures)
end

const parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7)
const n_global = 3
const n_segment_specific = 4
const initial_chromosome = [1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5]
const _lower_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
const _upper_base = [100.0, 100.0, 100.0, 466.0, 466.0, 466.0, 1466.0]

const lower = max.(1e-12, _lower_base ./ BOUND_SCALE)
const upper = _upper_base .* BOUND_SCALE

const wind_speeds, ambient_temperatures, generator_temperatures = load_turbine_data()
const u0 = generator_temperatures[1]
const num_steps = 2500
const data_M = reshape(Float64.(generator_temperatures), 1, :)
const n = length(data_M)

const de_spec = DifferenceModelSpec(example_difference_model, initial_chromosome, u0, num_steps, (wind_speeds, ambient_temperatures))
const model_manager = ModelManager(de_spec)

function make_parameter_labels(n_cps)
    labs = String[]
    for i in 1:n_global
        push!(labs, string(parnames[i]) * "_global")
    end
    for s in 1:(n_cps+1)
        for i in 1:n_segment_specific
            push!(labs, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    return labs
end

length(ARGS) >= 4 || error("Usage: julia turbine_ple_hybrid.jl <label> <cps_csv> <params_csv> <out_dir> [param_selection] [mode] [cp_selection]")
const LABEL = ARGS[1]
const CPS_CSV = ARGS[2]
const PARAMS_CSV = ARGS[3]
const OUT_DIR = ARGS[4]
mkpath(OUT_DIR)

const cps_best = sort(unique(Int.(CSV.read(CPS_CSV, DataFrame).cp)))
const params_best = Float64.(CSV.read(PARAMS_CSV, DataFrame).value)
const param_labels = make_parameter_labels(length(cps_best))[1:length(params_best)]

const MODE = length(ARGS) >= 6 ? lowercase(ARGS[6]) : "parameter"
const CP_SELECTION = length(ARGS) >= 7 ? ARGS[7] : "all"

const n_segments = length(cps_best) + 1
const n_params = n_global + n_segments * n_segment_specific
const lb_full = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
const ub_full = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]

function mica_objective(params, cps)
    return Mica.objective_function(
        params, cps, parnames, n_global, n_segment_specific,
        model_manager, loss_function, data_M
    )
end

function selected_indices(selection::String)
    sel = strip(lowercase(selection))
    if sel == "all"
        return collect(1:n_params)
    elseif sel == "global"
        return collect(1:n_global)
    else
        wanted = strip.(split(selection, ","))
        idx = Int[]
        for w in wanted
            pos = findfirst(==(w), param_labels)
            pos === nothing && error("Unknown parameter label: $w")
            push!(idx, pos)
        end
        return idx
    end
end

function main()
    selection = length(ARGS) >= 5 ? ARGS[5] : "all"
    idxs = selected_indices(selection)

    println("[$LABEL] Turbine PLE on $(length(idxs)) parameter(s) out of $n_params")
    println("[$LABEL] Mode: $MODE")
    println("[$LABEL] CPs: $cps_best")
    println("[$LABEL] D2D adaptive profiling: samplesize=$PLE_D2D_SAMPLESIZE, rel_step=$PLE_D2D_REL_STEP, step_factor=$PLE_D2D_STEP_FACTOR")
    println("[$LABEL] Threads: $(Threads.nthreads())")
    flush(stdout)

    params_init = params_best
    cps_ref = cps_best

    l2_loss = mica_objective(params_best, cps_best)
    println("[$LABEL] Base loss (MICA original) = $(round(l2_loss, digits=4))")

    prob_init = ODEChangepointPLEProblem(
        objective = mica_objective,
        data = data_M,
        loss_fn = loss_function,
        changepoints = cps_ref,
        best_params = params_init,
        best_loss = l2_loss,
        lb = lb_full,
        ub = ub_full,
        param_names = param_labels,
        n_global = n_global,
        n_segment_specific = n_segment_specific,
        n_obs = n
    )

    best_params = params_init
    best_loss_val = l2_loss

    println("[$LABEL] Re-optimising reference...")
    ref_opt = build_optimizer(PLE_REF_OPTIMIZER; ref=true)
    
    for s in 1:PLE_MULTISTART
        x_init = if s == 1
            copy(params_init)
        else
            range = ub_full .- lb_full
            x_pert = copy(params_init)
            for j in 1:n_params
                δ = PLE_MULTISTART_PERT * range[j] * (2.0 * rand() - 1.0)
                x_pert[j] = clamp(x_pert[j] + δ, lb_full[j], ub_full[j])
            end
            x_pert
        end
        
        try
            loss, x_opt = optimize_ple(
                (p) -> mica_objective(p, cps_ref), x_init, lb_full, ub_full, ref_opt
            )
            if isfinite(loss) && loss < best_loss_val
                best_loss_val = loss
                best_params = copy(x_opt)
            end
            println("[$LABEL] Multi-start $s: loss = $(round(loss, digits=4))")
        catch e
            println("[$LABEL] Multi-start $s failed: $e")
        end
    end

    println("[$LABEL] Best re-optimised loss = $(round(best_loss_val, digits=4))")
    threshold = best_loss_val + CHISQ_95
    println("[$LABEL] Threshold (L2 loss + 3.84) = $(round(threshold, digits=4))")

    prob = ODEChangepointPLEProblem(
        objective = mica_objective,
        data = data_M,
        loss_fn = loss_function,
        changepoints = cps_ref,
        best_params = best_params,
        best_loss = best_loss_val,
        lb = lb_full,
        ub = ub_full,
        param_names = param_labels,
        n_global = n_global,
        n_segment_specific = n_segment_specific,
        n_obs = n
    )

    prof_opt = build_optimizer(PLE_PROF_OPTIMIZER; ref=false)

    if MODE == "parameter" || MODE == "both"
        println("\n[$LABEL] === Profiling $(length(idxs)) parameters (conditional) ===")
        profiles = Vector{ProfileResult}(undef, length(idxs))
        
        d2d_opts = ProfileOptions(
            samplesize = PLE_D2D_SAMPLESIZE,
            rel_step_increase = PLE_D2D_REL_STEP,
            step_factor = PLE_D2D_STEP_FACTOR,
            stop_margin_factor = PLE_D2D_STOP_MARGIN,
            polish = PLE_D2D_POLISH,
            smooth_jumps = PLE_D2D_SMOOTH_JUMPS,
            allow_better_optimum = PLE_D2D_ALLOW_BETTER
        )
        
        Base.Threads.@threads for k in 1:length(idxs)
            idx = idxs[k]
            profiles[k] = profile_parameter(prob, idx;
                threshold = threshold,
                optimizer = prof_opt,
                options = d2d_opts
            )
        end
        sort!(profiles, by=p -> p.index)
        write_profiles(joinpath(OUT_DIR, "ple_results.csv"), profiles)
        CSV.write(joinpath(OUT_DIR, "ple_summary.csv"), ple_summary(profiles))
    end

    if MODE == "joint" || MODE == "both"
        println("\n[$LABEL] === Profiling $(length(idxs)) parameters (joint) ===")
        joint_opt = build_optimizer(PLE_JOINT_OPTIMIZER; ref=false)
        joint_ref_opt = build_optimizer(PLE_REF_OPTIMIZER; ref=true)
        
        # Note: do_reference_search and reference_optimizer are not in the base package signature
        # We'll pass the supported arguments.
        joint_profiles = profile_all_parameters_joint(
            prob;
            indices = idxs,
            window = PLE_JOINT_WINDOW,
            step = PLE_JOINT_CP_STEP,
            threshold = threshold,
            optimizer = joint_opt,
            parallel_over_parameters = PLE_JOINT_PARALLEL_OVER_PARAMS,
            parallelize_grid = PLE_JOINT_PARALLELIZE_GRID
        )
        sort!(joint_profiles, by=p -> p.index)
        write_profiles(joinpath(OUT_DIR, "joint_ple_results.csv"), joint_profiles)
        CSV.write(joinpath(OUT_DIR, "joint_ple_summary.csv"), ple_summary(joint_profiles))
    end

    println("[$LABEL] Done. Outputs in $OUT_DIR")
end

main()
