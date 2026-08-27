#!/usr/bin/env julia
# Profile-likelihood (PLE) for a COVID-19 MICA winner using threaded Evolutionary GA.
#
# Usage:
#   julia --threads=auto covid_ple_bobyqa.jl <label> <cps_csv> <params_csv> <out_dir> [param_selection] [mode] [cp_selection]
#
# param_selection: "all" (default), "global", "segment", range "a-b", or comma-separated labels.
# mode:            "parameter" (conditional), "joint" (parameter + CP joint PLE),
#                  "cp", or "both" (parameter + cp, default: "both").
# cp_selection:    "all" (default) or comma-separated 1-based CP indices.
#
# This script defaults to L2 loss (proper Gaussian negative log-likelihood on the
# log scale, per-channel σ estimated from the reference-fit residuals) with the
# standard χ²(1, 0.95) = 3.84 threshold.  L1 loss (MICA default) with a moving-block
# bootstrap threshold is still available via COVID_PLE_LOSS=l1.  Conditional parameter profiling
# uses the D2D-style adaptive stepping implemented in BreakpointProfiles.jl
# (fixed change points, flexible thresholds).  The GA settings match those used
# for the MICA change-point analysis, wrapped in a hybrid global-then-local optimizer
# and threaded population evaluation so that all 40 cores on one node are used.
#
# Requires: Mica.jl project environment (for CovModel!, data loading, etc.)
#           BreakpointProfiles.jl

using Pkg
# Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "..", "codes", "Mica.jl"))
# Make BreakpointProfiles available
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "..", "codes", "BreakpointProfiles.jl"))

Pkg.instantiate()
using Mica
using BreakpointProfiles
using CSV, DataFrames, Statistics, Random
using OrdinaryDiffEq, Smoothers
using Optim
using LabelledArrays
using Printf
using Dates

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")

const CP_WINDOW      = parse(Int, get(ENV, "COVID_CP_WINDOW", "7"))
const LOSS_TYPE      = lowercase(get(ENV, "COVID_PLE_LOSS", "l2"))  # "l1" or "l2"
const BOOTSTRAP_SAMPLES = parse(Int, get(ENV, "COVID_PLE_BOOTSTRAP_SAMPLES", "50"))
const BOOTSTRAP_BLOCK_SIZE = parse(Int, get(ENV, "COVID_PLE_BOOTSTRAP_BLOCK_SIZE", "14"))  # match 14-day moving-average window

# D2D-style adaptive profiling options
const PLE_D2D_SAMPLESIZE       = parse(Int, get(ENV, "COVID_PLE_D2D_SAMPLESIZE", "100"))
const PLE_D2D_REL_STEP         = parse(Float64, get(ENV, "COVID_PLE_D2D_REL_STEP", "0.1"))
const PLE_D2D_STEP_FACTOR      = parse(Float64, get(ENV, "COVID_PLE_D2D_STEP_FACTOR", "1.5"))
const PLE_D2D_STOP_MARGIN      = parse(Float64, get(ENV, "COVID_PLE_D2D_STOP_MARGIN", "1.2"))
const PLE_D2D_POLISH           = parse(Bool, get(ENV, "COVID_PLE_D2D_POLISH", "true"))
const PLE_D2D_SMOOTH_JUMPS     = parse(Bool, get(ENV, "COVID_PLE_D2D_SMOOTH_JUMPS", "true"))
const PLE_D2D_ALLOW_BETTER     = parse(Bool, get(ENV, "COVID_PLE_D2D_ALLOW_BETTER", "false"))

# Optimizer selection for PLE. We use the same GA settings that were used for the
# MICA change-point analysis, wrapped in a hybrid global-then-local optimizer.
const PLE_MULTISTART      = parse(Int, get(ENV, "COVID_PLE_MULTISTART", "1"))
const PLE_MULTISTART_PERT = parse(Float64, get(ENV, "COVID_PLE_MULTISTART_PERTURBATION", "0.5"))
const PLE_REF_OPTIMIZER   = lowercase(get(ENV, "COVID_PLE_REF_OPTIMIZER", "hybrid"))   # optimizer for reference search
const PLE_PROF_OPTIMIZER  = lowercase(get(ENV, "COVID_PLE_OPTIMIZER", "neldermead"))   # optimizer for profiling

# Joint adaptive PLE settings: when MODE == "joint", parameters are profiled while
# changepoint locations are treated as nuisance variables and searched jointly.
const PLE_JOINT_WINDOW      = parse(Int, get(ENV, "COVID_PLE_JOINT_WINDOW", "20"))
const PLE_JOINT_CP_STEP     = parse(Int, get(ENV, "COVID_PLE_JOINT_CP_STEP", "5"))
const PLE_JOINT_NPOINTS     = parse(Int, get(ENV, "COVID_PLE_JOINT_NPOINTS", "10"))  # fewer points because each evaluation is much more expensive
const PLE_JOINT_OPTIMIZER   = lowercase(get(ENV, "COVID_PLE_JOINT_OPTIMIZER", "neldermead"))
const PLE_JOINT_REF_SEARCH  = parse(Bool, get(ENV, "COVID_PLE_JOINT_REF_SEARCH", "false"))  # true = full joint reference search; false = use conditional reference (hybrid mode)
const PLE_JOINT_PARALLEL_OVER_PARAMS = parse(Bool, get(ENV, "COVID_PLE_JOINT_PARALLEL_OVER_PARAMS", "true"))  # thread over parameters in profile_all_parameters_joint
const PLE_JOINT_PARALLELIZE_GRID     = parse(Bool, get(ENV, "COVID_PLE_JOINT_PARALLELIZE_GRID", "true"))     # thread over CP grid inside each parameter profile

# Bootstrap threshold settings
const BOOTSTRAP_PER_PARAM = parse(Bool, get(ENV, "COVID_PLE_BOOTSTRAP_PER_PARAM", "true"))  # per-parameter thresholds instead of single global
const BOOTSTRAP_OPTIMIZER = lowercase(get(ENV, "COVID_PLE_BOOTSTRAP_OPTIMIZER", "neldermead"))   # optimizer for bootstrap inner loop
const BOOTSTRAP_BOBYQA_STARTS = parse(Int, get(ENV, "COVID_PLE_BOOTSTRAP_BOBYQA_STARTS", "3"))
const BOOTSTRAP_BOBYQA_MAXEVAL = parse(Int, get(ENV, "COVID_PLE_BOOTSTRAP_BOBYQA_MAXEVAL", "2000"))

# Metaheuristics DE settings
const PLE_DE_POPULATION = parse(Int, get(ENV, "COVID_PLE_DE_POPULATION", "150"))
const PLE_DE_F          = parse(Float64, get(ENV, "COVID_PLE_DE_F", "0.9"))
const PLE_DE_CR         = parse(Float64, get(ENV, "COVID_PLE_DE_CR", "0.9"))
const PLE_DE_MAX_EVALS  = parse(Int, get(ENV, "COVID_PLE_DE_MAX_EVALS", "150000"))
const PLE_DE_PARALLEL   = parse(Bool, get(ENV, "COVID_PLE_DE_PARALLEL", "true"))
const BOOTSTRAP_DE_MAX_EVALS = parse(Int, get(ENV, "COVID_PLE_BOOTSTRAP_DE_MAX_EVALS", "75000"))

# Evolutionary GA settings (kept as an alternative)
const PLE_POPULATION = parse(Int, get(ENV, "COVID_PLE_POPULATION", "40"))
const PLE_ITERATIONS = parse(Int, get(ENV, "COVID_PLE_ITERATIONS", "200"))
const PLE_PARALLEL   = Symbol(get(ENV, "COVID_PLE_PARALLEL", "serial"))  # :serial lets the CP grid thread
const PLE_SELECTION  = get(ENV, "COVID_PLE_SELECTION", "tournament(2)")
const PLE_CROSSOVER  = get(ENV, "COVID_PLE_CROSSOVER", "SBX(0.7, 1)")
const PLE_MUTATION   = get(ENV, "COVID_PLE_MUTATION", "gaussian(0.0001)")
const PLE_MUTATION_RATE = parse(Float64, get(ENV, "COVID_PLE_MUTATION_RATE", "0.7"))
const PLE_CROSSOVER_RATE = parse(Float64, get(ENV, "COVID_PLE_CROSSOVER_RATE", "0.7"))

# Separate budgets for reference search (strong) and profiling (fast)
const PLE_REF_POPULATION = parse(Int, get(ENV, "COVID_PLE_REF_POPULATION", "150"))
const PLE_REF_ITERATIONS = parse(Int, get(ENV, "COVID_PLE_REF_ITERATIONS", "1000"))
const PLE_PROF_POPULATION = parse(Int, get(ENV, "COVID_PLE_PROF_POPULATION", "40"))
const PLE_PROF_ITERATIONS = parse(Int, get(ENV, "COVID_PLE_PROF_ITERATIONS", "200"))

function build_optimizer(opt_name::String=PLE_PROF_OPTIMIZER; ref::Bool=false)
    pop = ref ? PLE_REF_POPULATION : PLE_PROF_POPULATION
    iters = ref ? PLE_REF_ITERATIONS : PLE_PROF_ITERATIONS
    if opt_name == "hybrid"
        ga = EvolutionaryPLEConfig(
            population_size=pop,
            iterations=iters,
            selection=PLE_SELECTION,
            crossover=PLE_CROSSOVER,
            mutation=PLE_MUTATION,
            mutation_rate=PLE_MUTATION_RATE,
            crossover_rate=PLE_CROSSOVER_RATE,
            parallelization=PLE_PARALLEL
        )
        local_opt = OptimPLEConfig(Fminbox(LBFGS()), Optim.Options(show_trace=false, iterations=200))
        return HybridOptimizerConfig(global_optimizer=ga, local_optimizer=local_opt)
    elseif opt_name == "de"
        return MetaheuristicsDEConfig(
            population=PLE_DE_POPULATION,
            F=PLE_DE_F,
            CR=PLE_DE_CR,
            max_evals=PLE_DE_MAX_EVALS,
            parallel=PLE_DE_PARALLEL
        )
    elseif opt_name == "evolutionary"
        return EvolutionaryPLEConfig(
            population_size=pop,
            iterations=iters,
            selection=PLE_SELECTION,
            crossover=PLE_CROSSOVER,
            mutation=PLE_MUTATION,
            mutation_rate=PLE_MUTATION_RATE,
            crossover_rate=PLE_CROSSOVER_RATE,
            parallelization=PLE_PARALLEL
        )
    elseif opt_name == "neldermead"
        # Derivative-free simplex optimizer, appropriate for local polishing after a
        # strong global reference search.  It requires no manual perturbation and
        # handles the log-clip/abs(L1) loss better than gradient-based methods.
        return OptimPLEConfig(
            Fminbox(NelderMead()),
            Optim.Options(show_trace=false, iterations=2000)
        )
    elseif opt_name == "lbfgs"
        # Gradient-based local optimizer.  Fast on smooth problems; may fail on
        # stiff ODEs with integration errors.
        return OptimPLEConfig(
            Fminbox(LBFGS()),
            Optim.Options(show_trace=false, iterations=200)
        )
    elseif opt_name == "bobyqa"
        # Derivative-free bound-constrained direct search via NLopt.
        return NLoptPLEConfig(
            algorithm=:LN_BOBYQA,
            xtol_rel=1.0e-6,
            maxeval=10_000
        )
    else
        error("Unknown optimizer: $opt_name. Use 'hybrid', 'de', 'evolutionary', 'neldermead', 'lbfgs', or 'bobyqa'.")
    end
end

function build_bootstrap_optimizer()
    if BOOTSTRAP_OPTIMIZER == "bobyqa"
        return MultiStartBOBYQAConfig(
            n_starts=BOOTSTRAP_BOBYQA_STARTS,
            maxeval=BOOTSTRAP_BOBYQA_MAXEVAL,
            xtol_rel=1.0e-6,
            perturbation=0.1
        )
    elseif BOOTSTRAP_OPTIMIZER == "de"
        # Use a small warm-start perturbation for the bootstrap refit: the bootstrap
        # MLE should be close to the original MLE, and the default 0.5 relative
        # perturbation is too large for the wide COVID bounds (e.g. [0,8]).
        # Polishing with LBFGS is important: pure DE on bootstrap data can get stuck
        # at the warm-start, producing a zero profile-likelihood-ratio threshold.
        de = MetaheuristicsDEConfig(
            population=PLE_DE_POPULATION,
            F=PLE_DE_F,
            CR=PLE_DE_CR,
            max_evals=BOOTSTRAP_DE_MAX_EVALS,
            parallel=PLE_DE_PARALLEL,
            warm_start_perturbation=0.05
        )
        lbfgs = OptimPLEConfig(Fminbox(LBFGS()), Optim.Options(show_trace=false, iterations=200))
        return HybridOptimizerConfig(global_optimizer=de, local_optimizer=lbfgs)
    elseif BOOTSTRAP_OPTIMIZER == "neldermead"
        return OptimPLEConfig(
            Fminbox(NelderMead()),
            Optim.Options(show_trace=false, iterations=2000)
        )
    elseif BOOTSTRAP_OPTIMIZER == "lbfgs"
        return OptimPLEConfig(
            Fminbox(LBFGS()),
            Optim.Options(show_trace=false, iterations=200)
        )
    elseif BOOTSTRAP_OPTIMIZER == "bobyqa"
        return NLoptPLEConfig(
            algorithm=:LN_BOBYQA,
            xtol_rel=1.0e-6,
            maxeval=10_000
        )
    else
        return build_optimizer(BOOTSTRAP_OPTIMIZER)
    end
end

function optimizer_summary(opt_name::String=PLE_PROF_OPTIMIZER; ref::Bool=false)
    pop = ref ? PLE_REF_POPULATION : PLE_PROF_POPULATION
    iters = ref ? PLE_REF_ITERATIONS : PLE_PROF_ITERATIONS
    if opt_name == "hybrid"
        return "Hybrid Evolutionary GA + LBFGS (GA: population=$pop, iterations=$iters, selection=$PLE_SELECTION, crossover=$PLE_CROSSOVER, mutation=$PLE_MUTATION, mutation_rate=$PLE_MUTATION_RATE, crossover_rate=$PLE_CROSSOVER_RATE, parallel=$PLE_PARALLEL)"
    elseif opt_name == "de"
        return "Metaheuristics DE (population=$PLE_DE_POPULATION, F=$PLE_DE_F, CR=$PLE_DE_CR, max_evals=$PLE_DE_MAX_EVALS, parallel=$PLE_DE_PARALLEL)"
    elseif opt_name == "bobyqa"
        return "NLopt BOBYQA (maxeval=10000)"
    elseif opt_name == "neldermead"
        return "Optim Nelder-Mead with bounds (Fminbox, iterations=2000)"
    elseif opt_name == "lbfgs"
        return "Optim LBFGS with bounds (Fminbox, iterations=200)"
    elseif opt_name == "evolutionary"
        return "Evolutionary GA (population=$pop, iterations=$iters, selection=$PLE_SELECTION, crossover=$PLE_CROSSOVER, mutation=$PLE_MUTATION, mutation_rate=$PLE_MUTATION_RATE, crossover_rate=$PLE_CROSSOVER_RATE, parallel=$PLE_PARALLEL)"
    else
        error("Unknown optimizer for summary: $opt_name")
    end
end

# ---------- model / data (same as other COVID tasks) ----------
function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ * cos(2 * π * ((t - t₀) / 365))
end

function log_transform_data(data, threshold=1)
    return [val >= threshold ? log(val) : 0 for val in data]
end

function CovModel!(du, u, p, t)
    (ᴺS, ᴺE₀, ᴺE₁, ᴺI₀, ᴺI₁, ᴺI₂, ᴺI₃, ᴺR, D, Cases, V) = u
    N = ᴺS + ᴺE₀ + ᴺE₁ + ᴺI₀ + ᴺI₁ + ᴺI₂ + ᴺI₃ + ᴺR + D
    ᴺε₀  = p.ᴺε₀; ᴺε₁ = p.ᴺε₁; ᴺγ₀ = p.ᴺγ₀; ᴺγ₁ = p.ᴺγ₁
    ᴺγ₂  = p.ᴺγ₂; ᴺγ₃ = p.ᴺγ₃; ᴺp₁ = p.ᴺp₁; ᴺp₁₂ = p.ᴺp₁₂
    ᴺp₂₃ = p.ᴺp₂₃; ᴺp₁D = p.ᴺp₁D; ᴺp₂D = p.ᴺp₂D; ᴺp₃D = p.ᴺp₃D
    δ    = p.δ; δₜ = fδ(t, δ); ᴺβ = p.ᴺβ; ω = p.ω
    ν    = t < 330 ? 0.0 : p.ν
    ᴺβᴺSI = ᴺβ * δₜ * ᴺS * (ᴺE₁ + ᴺI₀ + ᴺI₁)
    du[1]  = -(ᴺβᴺSI) / N + ω * ᴺR - ν * ᴺS
    du[2]  =  (ᴺβᴺSI / N) - (ᴺε₀ * ᴺE₀)
    du[3]  =  (ᴺε₀ * ᴺE₀) - (ᴺε₁ * ᴺE₁)
    du[4]  =  ((1 - ᴺp₁) * ᴺε₁ * ᴺE₁) - (ᴺγ₀ * ᴺI₀)
    du[5]  =  (ᴺp₁ * ᴺε₁ * ᴺE₁) - (ᴺγ₁ * ᴺI₁)
    du[6]  =  (ᴺp₁₂ * ᴺγ₁ * ᴺI₁) - (ᴺγ₂ * ᴺI₂)
    du[7]  =  (ᴺp₂₃ * ᴺγ₂ * ᴺI₂) - (ᴺγ₃ * ᴺI₃)
    du[8]  =  ᴺγ₀ * ᴺI₀ +
              (1 - ᴺp₁₂ - ᴺp₁D) * ᴺγ₁ * ᴺI₁ +
              (1 - ᴺp₂₃ - ᴺp₂D) * ᴺγ₂ * ᴺI₂ +
              (1 - ᴺp₃D) * ᴺγ₃ * ᴺI₃ - ω * ᴺR + ν * ᴺS
    du[9]  =  (ᴺp₁D * ᴺγ₁ * ᴺI₁) + (ᴺp₂D * ᴺγ₂ * ᴺI₂) + (ᴺp₃D * ᴺγ₃ * ᴺI₃)
    du[10] =  ᴺp₁ * ᴺε₁ * ᴺE₁
    du[11] =  ν * ᴺS
end

function ple_ode_model(params, tspan::Tuple{Float64, Float64}, u0)
    if tspan[2] < 330
        params[:ν] = 0.0
    end
    prob = ODEProblem(CovModel!, u0, tspan, params)
    try
        sol = solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-6, reltol=1.0e-6,
                    maxiters=100_000,
                    isoutofdomain=(u, p, t) -> any(x -> x < 0, u),
                    verbose=false)
        if SciMLBase.successful_retcode(sol)
            return sol[:, :]
        else
            return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
        end
    catch e
        # Stiff or ill-conditioned parameter sets can make Tsit5 throw before
        # returning an unsuccessful retcode.  Return a NaN filler so the loss
        # function can penalise the proposal gracefully.
        @debug "ODE solve failed for params=$params, tspan=$tspan: $e"
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

function load_covid_data(n_days=parse(Int, get(ENV, "COVID_PLE_NDAYS", "400")))
    cases_CP = CSV.read(joinpath(EXAMPLE_DIR, "case_rki_daily.csv"), DataFrame).total
    hospital_CP = CSV.read(joinpath(EXAMPLE_DIR, "Hospitalization_rki_daily.csv"), DataFrame).total
    death_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "death_rki_daily.csv"), DataFrame).Todesfaelle_neu)
    icu_CP = CSV.read(joinpath(EXAMPLE_DIR, "icu_rki_daily.csv"), DataFrame).total
    vacc_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "vaccination_rki_daily_allShots.csv"), DataFrame).Total)

    data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
    max_length = maximum(length, data_CP)
    data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
    data_CP = [vector[1:n_days] for vector in data_CP]
    data_CP[1] = hma(data_CP[1], 14)
    data_CP[4] = hma(data_CP[4], 14)
    data_CP[5] = hma(data_CP[5], 14)
    return Matrix(reduce(hcat, data_CP)')
end

const PLE_NDAYS = parse(Int, get(ENV, "COVID_PLE_NDAYS", "400"))
const data_CP = load_covid_data(PLE_NDAYS)
const n_days = size(data_CP, 2)

const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                   :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
const n_global = 8
const n_segment_specific = 8
const u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const data_indices = [5, 6, 7, 9, 11]

const BOUND_SCALE = parse(Float64, get(ENV, "COVID_PLE_BOUND_SCALE", "1.0"))
const _lower_base = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
                     0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
const _upper_base = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
                     0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]
# Widen bounds by BOUND_SCALE.  Keep lower bounds non-negative and add a tiny
# floor so that Fminbox/NelderMead does not receive an exact zero-width box.
const lower = max.(1e-12, _lower_base ./ BOUND_SCALE)
const upper = _upper_base .* BOUND_SCALE

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

# ---------- CLI ----------
length(ARGS) >= 4 || error("Usage: julia covid_ple_bobyqa.jl <label> <cps_csv> <params_csv> <out_dir> [param_selection] [mode] [cp_selection]")
const LABEL = ARGS[1]
const CPS_CSV = ARGS[2]
const PARAMS_CSV = ARGS[3]
const OUT_DIR = ARGS[4]
mkpath(OUT_DIR)

const cps_best = sort(unique(Int.(CSV.read(CPS_CSV, DataFrame).cp)))
const params_best = Float64.(CSV.read(PARAMS_CSV, DataFrame).value)
const param_labels = make_parameter_labels(length(cps_best))[1:length(params_best)]

const MODE = length(ARGS) >= 6 ? lowercase(ARGS[6]) : "both"  # "parameter", "cp", or "both"
const CP_SELECTION = length(ARGS) >= 7 ? ARGS[7] : "all"        # used when MODE == "cp"

const n_segments = length(cps_best) + 1
const n_params = n_global + n_segments * n_segment_specific
const lb_full = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
const ub_full = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]

const ode_spec = ODEModelSpec(ple_ode_model, params_best, Float64.(u0_full), (0.0, Float64(n_days)))
const model_manager = ModelManager(ode_spec)

# ---------- L1 objective (MICA default) ----------
const BAD_LOSS_PENALTY = 1.0e12

function loss_abs_log(obs, sim)
    # If the ODE solver failed or produced non-finite states, return a large but
    # finite penalty so the optimizer can continue instead of crashing on NaN/Inf.
    if any(!isfinite, sim)
        return BAD_LOSS_PENALTY
    end
    total = 0.0
    for (k, r) in enumerate(data_indices)
        total += sum(abs, log_transform_data(sim[r, :]) .- log_transform_data(obs[k, :]))
    end
    return total
end

function mica_objective_l1(params, cps, data=data_CP)
    return Mica.objective_function(
        params, cps, parnames, n_global, n_segment_specific,
        model_manager, loss_abs_log, data
    )
end

# ---------- L2 objective (proper Gaussian NLL) ----------
# Flatten a MICA simulate_full_model result to a plain Matrix{Float64}.
# When the result contains DiffEqArray segments, hcat of segments can produce a
# Matrix{DiffEqArray} instead of a numeric Matrix.
function flatten_simulation(sim)
    if sim isa Matrix && !(eltype(sim) <: Number)
        segments = Matrix{Float64}[hcat(seg.u...) for seg in sim]
        return reduce(hcat, segments)
    elseif !(sim isa AbstractMatrix{<:Number})
        # Single DiffEqArray case: convert to Matrix
        return hcat(sim.u...)
    end
    return sim
end

# Estimate per-channel σ from the residuals of a reference simulation on the log
# scale.  The resulting Gaussian NLL is on the -2 log L scale so that the standard
function boxcox(data, lambda=0.25)
    return [val > 0.0 ? (val^lambda - 1.0)/lambda : 0.0 for val in data]
end

# χ²(1, 0.95) = 3.8415 threshold is valid for 95% profile-likelihood CIs.
function estimate_channel_sigma_from_reference(params, cps; data=data_CP)
    sim_raw, _ = Mica.simulate_full_model(
        params, cps, parnames, n_global, n_segment_specific,
        model_manager, data; data_indices=data_indices
    )
    sim_full = flatten_simulation(sim_raw)
    
    if LOSS_TYPE == "boxcox"
        sim_full_bc = boxcox(sim_full)
        data_bc = boxcox(data)
        return estimate_channel_sigma(sim_full_bc, data_bc; data_indices=data_indices)
    else
        return estimate_channel_sigma(sim_full, data; data_indices=data_indices)
    end
end

# Build MICA-compatible (obs, sim) and PLE-package-compatible (sim, data) losses
# that map the five observed channels to their model-compartment rows.
function build_l2_losses(sigmas)
    ll = GaussianLogNLL(sigmas)
    function loss_l2_mica(obs, sim)
        if any(!isfinite, sim)
            return BAD_LOSS_PENALTY
        end
        return ll(sim[data_indices, :], obs)
    end
    function loss_l2_ple(sim, data)
        if any(!isfinite, sim)
            return BAD_LOSS_PENALTY
        end
        return ll(sim[data_indices, :], data)
    end
    return loss_l2_mica, loss_l2_ple
end

function build_boxcox_losses(sigmas)
    ll = GaussianLogNLL(sigmas)
    function loss_boxcox_mica(obs, sim)
        if any(!isfinite, sim)
            return BAD_LOSS_PENALTY
        end
        return ll(boxcox(sim[data_indices, :]), boxcox(obs))
    end
    function loss_boxcox_ple(sim, data)
        if any(!isfinite, sim)
            return BAD_LOSS_PENALTY
        end
        return ll(boxcox(sim[data_indices, :]), boxcox(data))
    end
    return loss_boxcox_mica, loss_boxcox_ple
end


# Placeholders; replaced in main() once the reference σ is estimated.
loss_l2_mica = loss_abs_log
loss_l2_ple  = loss_abs_log

function mica_objective_l2(params, cps, data=data_CP)
    return Mica.objective_function(
        params, cps, parnames, n_global, n_segment_specific,
        model_manager, loss_l2_mica, data
    )
end

# ---------- parameter selection ----------
function selected_indices(selection::String)
    sel = strip(lowercase(selection))
    if sel == "all"
        return collect(1:n_params)
    elseif sel == "global"
        return collect(1:n_global)
    elseif sel == "segment"
        return collect((n_global+1):n_params)
    elseif occursin(r"^\d+\-\d+$", sel) || occursin(r"^\d+\:\d+$", sel)
        # Range selection, e.g. "1-11" or "1:11"
        sep = occursin('-', sel) ? '-' : ':'
        parts = split(sel, sep)
        a = parse(Int, parts[1])
        b = parse(Int, parts[2])
        1 <= a <= b <= n_params || error("Range $sel out of bounds 1:$n_params")
        return collect(a:b)
    else
        wanted = split(selection, ",")
        wanted = strip.(wanted)
        idx = Int[]
        for w in wanted
            pos = findfirst(==(w), param_labels)
            if pos === nothing
                error("Unknown parameter label: $w. Available: $(join(param_labels, ", "))")
            end
            push!(idx, pos)
        end
        return idx
    end
end

# ---------- bootstrap data generator for L1 threshold ----------
# Moving-block bootstrap of residuals on the log scale.  The block length defaults
# to 14 days to match the 14-day moving-average smoothing applied to the case,
# death, and vaccination channels.  This preserves short-range temporal
# correlation that an i.i.d. bootstrap would destroy.
function make_bootstrap_data_generator(ref_params, ref_cps; block_size::Int=BOOTSTRAP_BLOCK_SIZE)
    sim_full, _ = Mica.simulate_full_model(
        ref_params, ref_cps, parnames, n_global, n_segment_specific,
        model_manager, data_CP; data_indices=data_indices
    )
    # simulate_full_model may return a plain Matrix{<:Number} or a Matrix whose
    # elements are DiffEqArray segment results (DiffEqArray is not an AbstractArray,
    # so hcat of segments does not flatten). Concatenate to a plain Matrix{Float64}.
    if sim_full isa Matrix && !(eltype(sim_full) <: Number)
        segments = [hcat(seg.u...) for seg in sim_full]
        sim_full = reduce(hcat, segments)
    end
    log_sim = log_transform_data(sim_full[data_indices, :])
    log_data = log_transform_data(data_CP)
    residuals = log_data .- log_sim

    function data_generator()
        nt = size(residuals, 2)
        boot_residuals = similar(residuals)
        # Block bootstrap: choose random starting indices and copy blocks of length
        # `block_size`, wrapping around at the end of the series.
        t = 1
        while t <= nt
            start_idx = rand(1:nt)
            for b in 0:(block_size - 1)
                t > nt && break
                idx = mod1(start_idx + b, nt)
                for k in 1:size(residuals, 1)
                    boot_residuals[k, t] = residuals[k, idx]
                end
                t += 1
            end
        end
        return exp.(log_sim .+ boot_residuals)
    end
    return data_generator
end

"""    joint_reference_search(mica_objective, params_init, cps_init, lb, ub, n_obs, optimizer;
                           window=7, max_combinations=5000)

Find the best joint maximum-likelihood estimate (params, changepoints) before running
joint adaptive PLE.  The conditional MICA reference `(params_init, cps_init)` is a good
warm-start, but it is not the joint MLE because the CPs were estimated separately.  This
function exhaustively searches a discrete grid of CP locations around `cps_init`
(±`window` days) and re-optimises the parameters for every CP set in parallel.  The best
`(loss, params, cps)` found is returned and used as the reference for joint profiling.

Using a *fast* local optimiser (Nelder-Mead or LBFGS) is deliberate: the grid already
covers the main CP uncertainty, and a full global search at every CP set would be
prohibitively expensive."""
function joint_reference_search(mica_objective, params_init, cps_init, lb, ub, n_obs,
                                optimizer; window::Int=7, step::Int=1, max_combinations::Int=5000)
    grid = BreakpointProfiles._joint_cp_grid(cps_init, window, n_obs; step=step, max_combinations=max_combinations)
    println("[$LABEL] Joint reference search: $(length(grid)) CP grid combinations (window=±$window, step=$step, n_obs=$n_obs)")
    flush(stdout)

    best_loss = Inf
    best_params = copy(params_init)
    best_cps = copy(cps_init)

    losses = Vector{Float64}(undef, length(grid))
    params_list = Vector{Vector{Float64}}(undef, length(grid))
    cps_list = Vector{Vector{Int}}(undef, length(grid))

    # Parallelize over CP grid combinations.  Each combination runs a local optimizer
    # starting from the conditional best parameters; the inner optimizer is serial.
    if BreakpointProfiles.uses_internal_threads(optimizer)
        @warn "Joint reference search optimizer uses internal threads; falling back to serial CP-grid loop to avoid nested threading."
        for (i, cps) in enumerate(grid)
            cps_list[i] = cps
            try
                loss, x = optimize_ple(x -> mica_objective(x, cps), params_init, lb, ub, optimizer)
                losses[i] = loss
                params_list[i] = x
            catch e
                @warn "Joint reference search failed for cps=$cps: $e"
                losses[i] = Inf
                params_list[i] = copy(params_init)
            end
        end
    else
        Base.Threads.@threads for i in 1:length(grid)
            cps = grid[i]
            cps_list[i] = cps
            try
                loss, x = optimize_ple(x -> mica_objective(x, cps), params_init, lb, ub, optimizer)
                losses[i] = loss
                params_list[i] = x
            catch e
                @warn "Joint reference search failed for cps=$cps: $e"
                losses[i] = Inf
                params_list[i] = copy(params_init)
            end
        end
    end

    for i in 1:length(grid)
        if isfinite(losses[i]) && losses[i] < best_loss
            best_loss = losses[i]
            best_params = copy(params_list[i])
            best_cps = copy(cps_list[i])
        end
    end

    if !isfinite(best_loss)
        @warn "Joint reference search failed for all CP combinations; falling back to conditional reference."
        best_loss = mica_objective(params_init, cps_init)
        best_params = copy(params_init)
        best_cps = copy(cps_init)
    end

    println("[$LABEL] Joint reference search best loss = $(round(best_loss, digits=4)) at CPs = $best_cps")
    flush(stdout)
    return best_loss, best_params, best_cps
end

# ---------- changepoint selection ----------
function selected_cp_indices(selection::String)
    sel = strip(lowercase(selection))
    if sel == "all"
        return collect(1:length(cps_best))
    else
        wanted = split(selection, ",")
        wanted = strip.(wanted)
        idx = Int[]
        for w in wanted
            cp = tryparse(Int, w)
            if cp === nothing
                # Try to interpret as 1-based CP index
                error("Invalid CP selection: $w. Use 'all' or comma-separated 1-based CP indices (e.g. '1,2').")
            end
            1 <= cp <= length(cps_best) || error("CP index $cp out of range 1:$(length(cps_best))")
            push!(idx, cp)
        end
        return idx
    end
end

# ---------- main ----------
function main()
    selection = length(ARGS) >= 5 ? ARGS[5] : "all"
    idxs = selected_indices(selection)

    # Use local copies inside main so that joint-reference updates do not shadow
    # the global consts.
    cps_ref = cps_best
    param_labels_ref = param_labels

    loss_label = LOSS_TYPE == "l1" ? "L1" : "L2"
    threshold_label = LOSS_TYPE == "l1" ? "bootstrap" : "chi-squared"
    println("[$LABEL] Loss type: $loss_label")
    println("[$LABEL] Threshold method: $threshold_label")
    println("[$LABEL] PLE for $(length(idxs)) parameter(s) out of $n_params")
    println("[$LABEL] CPs: $cps_ref")
    println("[$LABEL] Reference-search optimizer: $(optimizer_summary(PLE_REF_OPTIMIZER; ref=true))")
    println("[$LABEL] Profiling optimizer: $(optimizer_summary(PLE_PROF_OPTIMIZER; ref=false))")
    println("[$LABEL] Multi-start reference: $PLE_MULTISTART starts (perturbation=$PLE_MULTISTART_PERT)")
    println("[$LABEL] D2D adaptive profiling: samplesize=$PLE_D2D_SAMPLESIZE, rel_step=$PLE_D2D_REL_STEP, step_factor=$PLE_D2D_STEP_FACTOR, stop_margin=$PLE_D2D_STOP_MARGIN")
    println("[$LABEL] D2D polish/smooth/better: polish=$PLE_D2D_POLISH, smooth_jumps=$PLE_D2D_SMOOTH_JUMPS, allow_better=$PLE_D2D_ALLOW_BETTER")
    if LOSS_TYPE == "l1"
        println("[$LABEL] Bootstrap: $BOOTSTRAP_SAMPLES replicates, block_size=$BOOTSTRAP_BLOCK_SIZE days")
    end
    println("[$LABEL] Threads: $(Threads.nthreads())")
    flush(stdout)

    params_init = params_best

    # Step 0: Verify L1 loss matches the MICA output (always reported for reference)
    l1_loss = mica_objective_l1(params_best, cps_ref)
    println("[$LABEL] L1 loss (MICA original) = $(round(l1_loss, digits=4))")

    # Choose objective and package loss function with correct signatures
    mica_objective, loss_fn = if LOSS_TYPE == "l1"
        loss_abs_log_ple(sim, data) = loss_abs_log(data, sim)
        mica_objective_l1, loss_abs_log_ple
    else
        println("[$LABEL] Estimating per-channel σ from reference residuals...")
        sigmas = estimate_channel_sigma_from_reference(params_best, cps_ref)
        println("[$LABEL] Per-channel σ: $(round.(sigmas, digits=4))")
        CSV.write(joinpath(OUT_DIR, "l2_channel_sigmas.csv"),
                  DataFrame(channel=1:length(sigmas), sigma=sigmas))
        if LOSS_TYPE == "boxcox"
            loss_l2_mica_local, loss_l2_ple_local = build_boxcox_losses(sigmas)
        else
            loss_l2_mica_local, loss_l2_ple_local = build_l2_losses(sigmas)
        end
        global loss_l2_mica = loss_l2_mica_local
        global loss_l2_ple  = loss_l2_ple_local
        function mica_objective_l2_local(params, cps, data=data_CP)
            return Mica.objective_function(
                params, cps, parnames, n_global, n_segment_specific,
                model_manager, loss_l2_mica, data
            )
        end
        l2_loss_init = mica_objective_l2_local(params_best, cps_ref)
        println("[$LABEL] L2 loss (at L1-fit params) = $(round(l2_loss_init, digits=4))")
        mica_objective_l2_local, loss_l2_ple_local
    end

    # Step 1: Multi-start reference search (if requested) or single re-optimisation
    println("[$LABEL] Re-optimising under $loss_label loss..."); flush(stdout)
    obj = x -> mica_objective(x, cps_ref)
    if PLE_MULTISTART > 1
        println("[$LABEL] Running multi-start reference search with $PLE_MULTISTART starts...")
        ref_optimizer = build_optimizer(PLE_REF_OPTIMIZER; ref=true)
        best_loss, best_params = multi_start_reference_search(
            obj, params_init, lb_full, ub_full, ref_optimizer, PLE_MULTISTART;
            perturbation=PLE_MULTISTART_PERT
        )
        println("[$LABEL] Multi-start best $loss_label loss = $(round(best_loss, digits=4))")
    else
        ple_optimizer = build_optimizer(PLE_PROF_OPTIMIZER)
        best_loss, best_params = optimize_ple(obj, params_init, lb_full, ub_full, ple_optimizer)
    end
    println("[$LABEL] $loss_label loss (re-optimised) = $(round(best_loss, digits=4))")
    flush(stdout)

    # Save the re-optimised parameters
    refit_file = LOSS_TYPE == "l1" ? "l1_refit_params.csv" : "l2_refit_params.csv"
    CSV.write(joinpath(OUT_DIR, refit_file),
              DataFrame(parameter=param_labels_ref, value=best_params))

    # Step 1b: Joint reference search for joint adaptive PLE (optional).
    # In the hybrid mode (PLE_JOINT_REF_SEARCH=false) we keep the conditional reference
    # (CPs fixed at the MICA-detected values) and only marginalise over CP locations
    # during profiling.  This avoids the broken full joint reference search that could
    # find a worse optimum than the conditional reference.
    if MODE == "joint"
        println("[$LABEL] Joint profiling mode: window=±$(PLE_JOINT_WINDOW), CP step=$(PLE_JOINT_CP_STEP), points per direction=$(PLE_JOINT_NPOINTS)")
        flush(stdout)
        if PLE_JOINT_REF_SEARCH
            println("[$LABEL] === Full joint reference search (CPs as nuisance variables) ===")
            flush(stdout)
            joint_ref_optimizer = build_optimizer(PLE_JOINT_OPTIMIZER; ref=true)
            joint_loss, joint_params, joint_cps = joint_reference_search(
                mica_objective, best_params, cps_ref, lb_full, ub_full, n_days,
                joint_ref_optimizer;
                window=PLE_JOINT_WINDOW, step=PLE_JOINT_CP_STEP, max_combinations=5000
            )
            println("[$LABEL] Upgrading reference to joint MLE: loss $(round(best_loss, digits=4)) -> $(round(joint_loss, digits=4)), CPs $(cps_ref) -> $(joint_cps)")
            flush(stdout)
            best_loss = joint_loss
            best_params = joint_params
            cps_ref = sort(unique(joint_cps))
            # Rebuild parameter labels in case the number of segments changed (unlikely for a small window)
            param_labels_ref = make_parameter_labels(length(cps_ref))[1:length(best_params)]
            # Save the joint reference parameters
            CSV.write(joinpath(OUT_DIR, "joint_refit_params.csv"),
                      DataFrame(parameter=param_labels_ref, value=best_params))
            CSV.write(joinpath(OUT_DIR, "joint_refit_cps.csv"),
                      DataFrame(cp=cps_ref))
        else
            println("[$LABEL] === Hybrid mode: keeping conditional reference (CPs fixed at $(cps_ref)) ===")
            println("[$LABEL] CP locations will be treated as nuisance variables only during joint profiling.")
            flush(stdout)
        end
    end

    # Step 2: Compute threshold
    if LOSS_TYPE == "l2"
        threshold = best_loss + chi2_threshold(1, 0.95)
        println("[$LABEL] Threshold (L2 loss + 3.84) = $(round(threshold, digits=4))")
        thresholds_per_param = fill(threshold, n_params)
    else
        println("[$LABEL] Computing bootstrap threshold ($BOOTSTRAP_SAMPLES replicates, block_size=$BOOTSTRAP_BLOCK_SIZE)..."); flush(stdout)
        data_gen = make_bootstrap_data_generator(best_params, cps_ref; block_size=BOOTSTRAP_BLOCK_SIZE)
        objective_for_boot(params, data) = mica_objective_l1(params, cps_ref, data)
        threshold_optimizer = build_bootstrap_optimizer()
        if BOOTSTRAP_PER_PARAM
            println("[$LABEL] Per-parameter bootstrap thresholds using $(optimizer_summary(BOOTSTRAP_OPTIMIZER)) ..."); flush(stdout)
            threshold_margins = bootstrap_thresholds(data_gen, objective_for_boot, best_params;
                                                   n_boot=BOOTSTRAP_SAMPLES, alpha=0.95,
                                                   lb=lb_full, ub=ub_full, optimizer=threshold_optimizer,
                                                   indices=idxs)
            thresholds_per_param = best_loss .+ threshold_margins
            println("[$LABEL] Threshold margin range: [$(round(minimum(threshold_margins), digits=4)), $(round(maximum(threshold_margins), digits=4))]")
            println("[$LABEL] Per-parameter absolute thresholds written to report")
        else
            println("[$LABEL] Single global bootstrap threshold for parameter 1 ..."); flush(stdout)
            threshold_margin = bootstrap_threshold(data_gen, objective_for_boot, best_params, 1;
                                                   n_boot=BOOTSTRAP_SAMPLES, alpha=0.95,
                                                   lb=lb_full, ub=ub_full, optimizer=threshold_optimizer)
            threshold = best_loss + threshold_margin
            thresholds_per_param = fill(threshold, n_params)
            println("[$LABEL] Bootstrap threshold margin = $(round(threshold_margin, digits=4))")
            println("[$LABEL] Absolute threshold = $(round(threshold, digits=4))")
        end
    end
    flush(stdout)

    # For CP profiling we need a single scalar threshold.  In L2 mode this is the
    # global χ² threshold; in L1 per-parameter mode we use the mean per-parameter
    # bootstrap threshold as a pragmatic global threshold for the changepoint CIs.
    cp_threshold = if LOSS_TYPE == "l2"
        threshold
    elseif BOOTSTRAP_PER_PARAM
        mean(thresholds_per_param)
    else
        threshold
    end
    println("[$LABEL] CP-profile threshold = $(round(cp_threshold, digits=4))")
    flush(stdout)

    # Step 3: Construct the PLE problem
    prob = ODEChangepointPLEProblem(
        objective = mica_objective,
        data = data_CP,
        loss_fn = loss_fn,
        changepoints = cps_ref,
        best_params = best_params,
        best_loss = best_loss,
        lb = lb_full,
        ub = ub_full,
        param_names = param_labels_ref,
        n_global = n_global,
        n_segment_specific = n_segment_specific,
        n_obs = n_days
    )

    opt = build_optimizer(PLE_PROF_OPTIMIZER)

    ple_options = ProfileOptions(
        samplesize=PLE_D2D_SAMPLESIZE,
        rel_step_increase=PLE_D2D_REL_STEP,
        step_factor=PLE_D2D_STEP_FACTOR,
        stop_margin_factor=PLE_D2D_STOP_MARGIN,
        polish=PLE_D2D_POLISH,
        smooth_jumps=PLE_D2D_SMOOTH_JUMPS,
        allow_better_optimum=PLE_D2D_ALLOW_BETTER
    )

    # Step 4: Profile parameters (conditional, CPs fixed)
    profiles = ProfileResult[]
    elapsed_params = 0.0
    if MODE in ("parameter", "both")
        println("\n[$LABEL] === Profiling $(length(idxs)) parameters (conditional) ==="); flush(stdout)
        t0 = time()
        profiles = profile_all_parameters(prob;
            options=ple_options,
            optimizer=opt, indices=idxs, threshold=thresholds_per_param
        )
        elapsed_params = time() - t0
        println("[$LABEL] Conditional parameter profiling completed in $(round(elapsed_params/60, digits=1)) minutes")

        # Report quality
        n_identifiable = count(p -> p.identifiable, profiles)
        n_better = count(p -> p.best_found_loss < p.best_loss - 0.1, profiles)
        n_with_failures = count(p -> p.n_failed > 0, profiles)
        println("[$LABEL] Identifiable: $n_identifiable / $(length(profiles))")
        println("[$LABEL] Found better optimum: $n_better / $(length(profiles))")
        println("[$LABEL] Profiles with optimizer failures: $n_with_failures / $(length(profiles))")
        flush(stdout)

        # Step 4c: Reference improvement.  If profiling found a substantially better
        # loss than the reference, re-optimise from the best profile point and use the
        # improved reference for a second profiling pass.  This is the standard PLE
        # iteration: the reference must be the MLE before the CIs are trustworthy.
        if n_better > 0
            best_profile = argmin(p -> p.best_found_loss, profiles)
            if best_profile.best_found_loss < prob.best_loss - 0.1
                println("\n[$LABEL] === Reference improvement ===")
                println("[$LABEL] Best loss found during profiling: $(round(best_profile.best_found_loss, digits=4))")
                println("[$LABEL] Current reference loss: $(round(prob.best_loss, digits=4))")
                println("[$LABEL] Re-optimising from the best profile point...")
                flush(stdout)
                ref_improve_opt = HybridOptimizerConfig(
                    global_optimizer=build_optimizer(PLE_PROF_OPTIMIZER; ref=true),
                    local_optimizer=OptimPLEConfig(Fminbox(LBFGS()),
                                                   Optim.Options(show_trace=false, iterations=200))
                )
                t0 = time()
                improved_loss, improved_params = optimize_ple(
                    x -> mica_objective(x, cps_ref),
                    best_profile.best_found_params, prob.lb, prob.ub, ref_improve_opt
                )
                elapsed_ref_improve = time() - t0
                println("[$LABEL] Reference improvement completed in $(round(elapsed_ref_improve/60, digits=1)) minutes")
                println("[$LABEL] Improved reference loss = $(round(improved_loss, digits=4))")
                flush(stdout)

                if improved_loss < prob.best_loss - 0.1
                    best_params = improved_params
                    best_loss = improved_loss

                    # Recompute bootstrap thresholds with the improved reference
                    if LOSS_TYPE == "l1"
                        println("[$LABEL] Recomputing bootstrap threshold with improved reference...")
                        flush(stdout)
                        data_gen = make_bootstrap_data_generator(best_params, cps_ref; block_size=BOOTSTRAP_BLOCK_SIZE)
                        objective_for_boot(params, data) = mica_objective_l1(params, cps_ref, data)
                        threshold_optimizer = build_bootstrap_optimizer()
                        if BOOTSTRAP_PER_PARAM
                            threshold_margins = bootstrap_thresholds(data_gen, objective_for_boot, best_params;
                                                                   n_boot=BOOTSTRAP_SAMPLES, alpha=0.95,
                                                                   lb=lb_full, ub=ub_full, optimizer=threshold_optimizer,
                                                                   indices=idxs)
                            thresholds_per_param = best_loss .+ threshold_margins
                            println("[$LABEL] Improved threshold margin range: [$(round(minimum(threshold_margins), digits=4)), $(round(maximum(threshold_margins), digits=4))]")
                        else
                            threshold_margin = bootstrap_threshold(data_gen, objective_for_boot, best_params, 1;
                                                                   n_boot=BOOTSTRAP_SAMPLES, alpha=0.95,
                                                                   lb=lb_full, ub=ub_full, optimizer=threshold_optimizer)
                            threshold = best_loss + threshold_margin
                            thresholds_per_param = fill(threshold, n_params)
                            println("[$LABEL] Improved bootstrap threshold margin = $(round(threshold_margin, digits=4))")
                        end
                        cp_threshold = mean(thresholds_per_param)
                        println("[$LABEL] Improved CP-profile threshold = $(round(cp_threshold, digits=4))")
                        flush(stdout)
                    else
                        threshold = best_loss + chi2_threshold(1, 0.95)
                        thresholds_per_param = fill(threshold, n_params)
                        cp_threshold = threshold
                    end

                    # Re-build problem with improved reference
                    prob = ODEChangepointPLEProblem(
                        objective = mica_objective,
                        data = data_CP,
                        loss_fn = loss_fn,
                        changepoints = cps_ref,
                        best_params = best_params,
                        best_loss = best_loss,
                        lb = lb_full,
                        ub = ub_full,
                        param_names = param_labels_ref,
                        n_global = n_global,
                        n_segment_specific = n_segment_specific,
                        n_obs = n_days
                    )

                    println("[$LABEL] Re-profiling $(length(idxs)) parameters with improved reference...")
                    flush(stdout)
                    t0 = time()
                    profiles = profile_all_parameters(prob;
                        options=ple_options,
                        optimizer=opt, indices=idxs, threshold=thresholds_per_param
                    )
                    elapsed_params += time() - t0
                    println("[$LABEL] Second-pass conditional parameter profiling completed in $(round(elapsed_params/60, digits=1)) minutes total")
                    n_identifiable = count(p -> p.identifiable, profiles)
                    n_better = count(p -> p.best_found_loss < p.best_loss - 0.1, profiles)
                    n_with_failures = count(p -> p.n_failed > 0, profiles)
                    println("[$LABEL] Second-pass identifiable: $n_identifiable / $(length(profiles))")
                    println("[$LABEL] Second-pass found better optimum: $n_better / $(length(profiles))")
                    println("[$LABEL] Second-pass profiles with optimizer failures: $n_with_failures / $(length(profiles))")
                    flush(stdout)
                else
                    println("[$LABEL] Reference improvement did not beat current reference; keeping original reference.")
                    flush(stdout)
                end
            end
        end
    end

    # Step 4b: Joint adaptive parameter PLE (CPs treated as nuisance variables)
    joint_profiles = ProfileResult[]
    elapsed_joint = 0.0
    if MODE == "joint"
        println("\n[$LABEL] === Joint adaptive PLE for $(length(idxs)) parameters ==="); flush(stdout)
        println("[$LABEL] Joint CP window: ±$(PLE_JOINT_WINDOW), points per direction: $(PLE_JOINT_NPOINTS)")
        println("[$LABEL] Joint optimizer: $(optimizer_summary(PLE_JOINT_OPTIMIZER))")
        flush(stdout)
        t0 = time()
        joint_opt = build_optimizer(PLE_JOINT_OPTIMIZER)
        joint_profiles = profile_all_parameters_joint(prob;
            n_points=PLE_JOINT_NPOINTS, window=PLE_JOINT_WINDOW, step=PLE_JOINT_CP_STEP,
            optimizer=joint_opt, indices=idxs, threshold=thresholds_per_param,
            parallel_over_parameters=PLE_JOINT_PARALLEL_OVER_PARAMS,
            parallelize_grid=PLE_JOINT_PARALLELIZE_GRID
        )
        elapsed_joint = time() - t0
        println("[$LABEL] Joint parameter profiling completed in $(round(elapsed_joint/60, digits=1)) minutes")

        n_joint_identifiable = count(p -> p.identifiable, joint_profiles)
        n_joint_better = count(p -> p.best_found_loss < p.best_loss - 0.1, joint_profiles)
        n_joint_failures = count(p -> p.n_failed > 0, joint_profiles)
        println("[$LABEL] Joint identifiable: $n_joint_identifiable / $(length(joint_profiles))")
        println("[$LABEL] Joint found better optimum: $n_joint_better / $(length(joint_profiles))")
        println("[$LABEL] Joint profiles with optimizer failures: $n_joint_failures / $(length(joint_profiles))")
        flush(stdout)
    end

    # Step 5: Profile changepoint locations
    cp_profiles = CPProfileResult[]
    elapsed_cp = 0.0
    if MODE in ("cp", "both")
        cp_idxs = selected_cp_indices(CP_SELECTION)
        println("\n[$LABEL] === Profiling $(length(cp_idxs)) changepoint location(s) ==="); flush(stdout)
        t0 = time()
        if length(cp_idxs) == length(cps_ref)
            cp_profiles = profile_all_changepoints(prob; window=CP_WINDOW, optimizer=opt)
        else
            cp_profiles = [profile_changepoint(prob, j; window=CP_WINDOW, optimizer=opt) for j in cp_idxs]
        end
        elapsed_cp = time() - t0
        println("[$LABEL] CP profiling completed in $(round(elapsed_cp/60, digits=1)) minutes")
        flush(stdout)
    end

    # Step 6: Write outputs
    if MODE in ("parameter", "both")
        write_profiles(joinpath(OUT_DIR, "ple_results.csv"), profiles)
        summary_df = ple_summary(profiles)
        CSV.write(joinpath(OUT_DIR, "ple_summary.csv"), summary_df)
    end
    if MODE == "joint"
        write_profiles(joinpath(OUT_DIR, "joint_ple_results.csv"), joint_profiles)
        joint_summary_df = ple_summary(joint_profiles)
        CSV.write(joinpath(OUT_DIR, "joint_ple_summary.csv"), joint_summary_df)
    end
    if MODE in ("cp", "both")
        write_cp_profiles(joinpath(OUT_DIR, "cp_profile_loss.csv"), cp_profiles)
        cp_df = cp_summary(cp_profiles, cp_threshold)
        CSV.write(joinpath(OUT_DIR, "cp_profile_ci.csv"), cp_df)
    end

    # Step 7: Write report
    lines = String[]
    push!(lines, "# Profile-likelihood analysis for `$(LABEL)` ($loss_label loss, $(MODE == "joint" ? "joint adaptive" : "conditional") PLE)")
    push!(lines, "")
    push!(lines, "**Date:** $(Dates.now())")
    push!(lines, "**Parameter selection:** $(selection)")
    push!(lines, "**Mode:** $(MODE)")
    if MODE in ("cp", "both")
        push!(lines, "**CP selection:** $(CP_SELECTION)")
    end
    if MODE == "joint"
        push!(lines, "**Joint CP search window:** ±$(PLE_JOINT_WINDOW), step $(PLE_JOINT_CP_STEP)")
        push!(lines, "**Joint points per direction:** $(PLE_JOINT_NPOINTS)")
        push!(lines, "**Joint optimizer:** $(optimizer_summary(PLE_JOINT_OPTIMIZER))")
        push!(lines, "**Joint reference search:** $(PLE_JOINT_REF_SEARCH ? "full" : "hybrid (conditional reference)")")
    end
    push!(lines, "**Original CPs:** $(cps_ref)")
    push!(lines, "**L1 loss (MICA fit):** $(round(l1_loss, digits=4))")
    if PLE_MULTISTART > 1
        push!(lines, "**Reference search:** $PLE_MULTISTART starts with $(optimizer_summary(PLE_REF_OPTIMIZER; ref=true)) (perturbation=$(PLE_MULTISTART_PERT))")
    else
        push!(lines, "**Reference search:** single run with $(optimizer_summary(PLE_REF_OPTIMIZER; ref=true))")
    end
    if LOSS_TYPE == "l2"
        push!(lines, "**L2 loss (re-optimised):** $(round(best_loss, digits=4))")
        push!(lines, "**Threshold (Δloss ≤ $(round(chi2_threshold(1, 0.95), digits=4))):** $(round(thresholds_per_param[1], digits=4))")
    else
        push!(lines, "**L1 loss (re-optimised):** $(round(best_loss, digits=4))")
        if BOOTSTRAP_PER_PARAM
            push!(lines, "**Threshold (per-parameter moving-block bootstrap, $BOOTSTRAP_SAMPLES replicates, block_size=$BOOTSTRAP_BLOCK_SIZE):** see table below")
        else
            push!(lines, "**Threshold (single global moving-block bootstrap, $BOOTSTRAP_SAMPLES replicates, block_size=$BOOTSTRAP_BLOCK_SIZE):** $(round(thresholds_per_param[1], digits=4))")
        end
    end
    push!(lines, "**Profiling optimizer:** $(optimizer_summary(PLE_PROF_OPTIMIZER))")
    push!(lines, "**D2D adaptive profiling:** samplesize=$PLE_D2D_SAMPLESIZE, rel_step_increase=$PLE_D2D_REL_STEP, step_factor=$PLE_D2D_STEP_FACTOR, stop_margin=$PLE_D2D_STOP_MARGIN")
    push!(lines, "**D2D options:** polish=$PLE_D2D_POLISH, smooth_jumps=$PLE_D2D_SMOOTH_JUMPS, allow_better_optimum=$PLE_D2D_ALLOW_BETTER")
    if MODE in ("parameter", "both")
        push!(lines, "**Parameter wall time:** $(round(elapsed_params/60, digits=1)) minutes")
    end
    if MODE == "joint"
        push!(lines, "**Joint parameter wall time:** $(round(elapsed_joint/60, digits=1)) minutes")
    end
    if MODE in ("cp", "both")
        push!(lines, "**CP wall time:** $(round(elapsed_cp/60, digits=1)) minutes")
    end
    push!(lines, "")
    if LOSS_TYPE == "l1" && BOOTSTRAP_PER_PARAM
        push!(lines, "## Per-parameter bootstrap thresholds")
        push!(lines, "")
        push!(lines, "| parameter | threshold margin | absolute threshold |")
        push!(lines, "|---|---|---|")
        for (k, i) in enumerate(idxs)
            margin = thresholds_per_param[k] - best_loss
            push!(lines, @sprintf("| %s | %.5g | %.5g |", param_labels_ref[i], margin, thresholds_per_param[k]))
        end
        push!(lines, "")
    end
    if MODE in ("parameter", "both")
        push!(lines, "## Conditional parameter profiles")
        push!(lines, "")
        push!(lines, "| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |")
        push!(lines, "|---|---|---|---|---|---|---|")
        for r in eachrow(summary_df)
            better = r.best_found_loss < r.best_loss - 0.1 ? "⚠" : ""
            push!(lines, @sprintf("| %s | %.5g | %.5g | %.5g | %s | %d | %s |",
                                  r.parameter, r.best_value, r.ci_lower, r.ci_upper,
                                  r.identifiable ? "yes" : "no", r.n_failed, better))
        end
        push!(lines, "")
    end

    if MODE == "joint"
        push!(lines, "## Joint adaptive parameter profiles (CPs marginalised)")
        push!(lines, "")
        push!(lines, "| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |")
        push!(lines, "|---|---|---|---|---|---|---|")
        for r in eachrow(joint_summary_df)
            better = r.best_found_loss < r.best_loss - 0.1 ? "⚠" : ""
            push!(lines, @sprintf("| %s | %.5g | %.5g | %.5g | %s | %d | %s |",
                                  r.parameter, r.best_value, r.ci_lower, r.ci_upper,
                                  r.identifiable ? "yes" : "no", r.n_failed, better))
        end
        push!(lines, "")
    end

    if MODE in ("cp", "both")
        push!(lines, "## Changepoint profile intervals")
        push!(lines, "")
        push!(lines, "| cp # | original | CI lower | CI upper | identifiable |")
        push!(lines, "|---|---|---|---|---|")
        for r in eachrow(cp_df)
            push!(lines, @sprintf("| %d | %d | %d | %d | %s |",
                                  r.cp_index, r.original_cp, r.ci_lower, r.ci_upper,
                                  r.identifiable ? "yes" : "no"))
        end
        push!(lines, "")
    end

    push!(lines, "## Quality diagnostics")
    push!(lines, "")
    if MODE in ("parameter", "both")
        push!(lines, "- **Identifiable parameters (conditional):** $n_identifiable / $(length(profiles))")
        push!(lines, "- **Conditional profiles that found a better optimum:** $n_better (indicates the $loss_label re-fit may not have converged)")
        push!(lines, "- **Conditional profiles with optimizer failures:** $n_with_failures")
    end
    if MODE == "joint"
        push!(lines, "- **Identifiable parameters (joint):** $n_joint_identifiable / $(length(joint_profiles))")
        push!(lines, "- **Joint profiles that found a better optimum:** $n_joint_better (indicates the reference may not be the joint MLE)")
        push!(lines, "- **Joint profiles with optimizer failures:** $n_joint_failures")
    end
    if MODE in ("cp", "both")
        n_cp_identifiable = count(r -> r.identifiable, eachrow(cp_df))
        push!(lines, "- **Identifiable changepoints:** $n_cp_identifiable / $(nrow(cp_df))")
    end
    push!(lines, "")
    push!(lines, "## Files")
    if MODE in ("parameter", "both")
        push!(lines, "- `ple_results.csv` — conditional profile summary")
        push!(lines, "- `ple_results_curves.csv` — conditional full profile curves")
        push!(lines, "- `ple_summary.csv` — conditional identifiability summary")
    end
    if MODE == "joint"
        push!(lines, "- `joint_ple_results.csv` — joint profile summary")
        push!(lines, "- `joint_ple_results_curves.csv` — joint full profile curves")
        push!(lines, "- `joint_ple_summary.csv` — joint identifiability summary")
    end
    if MODE in ("cp", "both")
        push!(lines, "- `cp_profile_loss.csv` — changepoint profile curves")
        push!(lines, "- `cp_profile_ci.csv` — changepoint profile CIs")
    end
    refit_label = LOSS_TYPE == "l1" ? "L1" : "L2"
    push!(lines, "- `$(refit_label)_refit_params.csv` — $(refit_label) re-optimised parameters")
    open(joinpath(OUT_DIR, "ple_report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end

    println("\n[$LABEL] PLE done. Outputs in $OUT_DIR")
end

main()
