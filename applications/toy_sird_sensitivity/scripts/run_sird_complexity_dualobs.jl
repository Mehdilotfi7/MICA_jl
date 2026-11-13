#!/usr/bin/env julia
# ============================================================
# Toy-SIRD complexity / SP-NSP experiment with DUAL observables
# (I and D both observed) — control experiment.
#
# Identical setup to run_sird_complexity.jl (same true params, CPs,
# seeds, noise levels, MICA setting), except:
#   - two observed channels: I(t) and D(t)
#   - I channel noise is IDENTICAL to the single-observable
#     experiment (same RNG sequence), so old vs new is controlled
#   - D channel noise uses the same absolute sigma (second RNG stream,
#     truncated at 0)
#   - the loss is the raw pooled sum over both channels (same scale as
#     the single-observable experiment; no per-channel normalization):
#       L = sum((yI - I).^2) + sum((yD - D).^2)
#   - detection feeds L itself (pooled RSS, not sqrt) into the MDL
#     objective: with Mica's internal n = 80 time points, the MDL evidence
#     term then equals the correct n = 160 (two-channel) evidence up to a
#     model-independent constant.
#
# Usage (one job per configuration):
#   julia --threads=8 run_sird_complexity_dualobs.jl <config> [out_dir]
#
# config : M1, M2, M3, or M4
# Output: ../results/data/sird_dualobs_<config>.json
# ============================================================

using Pkg
const _MICA_PROJECT = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
if Base.active_project() === nothing || abspath(Base.active_project()) != abspath(joinpath(_MICA_PROJECT, "Project.toml"))
    Pkg.activate(_MICA_PROJECT)
end

using Mica
using JSON
using Random
using Statistics
using OrdinaryDiffEq
using Evolutionary
using Optim
using Printf

const CONFIG_NAME = length(ARGS) >= 1 ? ARGS[1] : error("config required: M1|M2|M3|M4")
const OUT_DIR  = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results", "data")
mkpath(OUT_DIR)

# -------------------------------------------------------------
# Experimental settings (identical to the single-observable run)
# -------------------------------------------------------------
const BETA_VALUES   = [0.4, 0.1, 0.3]
const GAMMA_TRUE    = 0.1
const DELTA_TRUE    = 0.02
const TRUE_CPS      = [20, 40]
const N             = 80
const N_POP         = 200
const N_ITER        = 400
const MIN_LENGTH    = 5
const STEP          = 1
const TOL           = 5
const OBJECTIVE     = :mdl
const SCALE_PENALTY = false

# -------------------------------------------------------------
# SIRD model
# -------------------------------------------------------------
function sird_model!(du, u, p, t)
    S, I, R, D = u
    Npop = 1000.0
    β = p.β
    γ = p.γ
    δ = p.δ
    du[1] = -β * S * I / Npop
    du[2] =  β * S * I / Npop - γ * I - δ * I
    du[3] =  γ * I
    du[4] =  δ * I
end

function sird_solver(params, tspan::Tuple{Float64,Float64}, u0::Vector{Float64})
    prob = ODEProblem(sird_model!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
    return hcat(sol.u...)
end

# -------------------------------------------------------------
# Data generation: I channel identical to the single-observable
# experiment (same RNG sequence), D channel amplitude-matched.
# -------------------------------------------------------------
function generate_sird_data_dualobs(seed::Int, sigma::Float64)
    rng = MersenneTwister(seed)
    Npop = 1000.0
    u0 = [990.0, 10.0, 0.0, 0.0]
    n = N
    tspan_full = (0.0, Float64(n - 1))

    function beta_piecewise(t)
        t_int = floor(Int, t)
        if t_int < TRUE_CPS[1]
            return BETA_VALUES[1]
        elseif t_int < TRUE_CPS[2]
            return BETA_VALUES[2]
        else
            return BETA_VALUES[3]
        end
    end

    function sird_true!(du, u, p, t)
        S, I, R, D = u
        β = beta_piecewise(t)
        γ = p.γ
        δ = p.δ
        du[1] = -β * S * I / Npop
        du[2] =  β * S * I / Npop - γ * I - δ * I
        du[3] =  γ * I
        du[4] =  δ * I
    end

    prob = ODEProblem(sird_true!, u0, tspan_full, (γ=GAMMA_TRUE, δ=DELTA_TRUE))
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
    I_true = [sol.u[i][2] for i in 1:length(sol.u)]
    D_true = [sol.u[i][4] for i in 1:length(sol.u)]

    # I channel: EXACTLY the same noise as the single-observable experiment
    yI = I_true .+ sigma .* randn(rng, n)
    yI = max.(yI, 0.0)
    s_I = max(maximum(yI), eps())

    # D channel: same noise scheme as the I channel (same absolute sigma,
    # truncated at 0), drawn as a second RNG stream so the I channel stays
    # byte-identical to the single-observable experiment.
    noise_D = sigma .* randn(rng, n)
    yD = max.(D_true .+ noise_D, 0.0)
    s_D = max(maximum(yD), eps())

    return Dict(
        "y" => yI,              # kept for compatibility (I channel)
        "yI" => yI,
        "yD" => yD,
        "I_true" => I_true,
        "D_true" => D_true,
        "s_I" => s_I,
        "s_D" => s_D,
        "true_cps" => TRUE_CPS,
        "true_betas" => BETA_VALUES,
        "true_gamma" => GAMMA_TRUE,
        "true_delta" => DELTA_TRUE,
        "sigma" => sigma,
        "seed" => seed,
        "n" => n
    )
end

# Two-channel loss: raw pooled RSS over both channels (same scale as the
# single-observable experiment; no normalization).  Detection uses sqrt(RSS)
# (same sqrt-RSS + MDL setting as the old experiment).
function loss_dualobs(observed::AbstractMatrix{Float64}, simulated::AbstractMatrix{Float64})
    rI = observed[1, :] .- simulated[2, :]
    rD = observed[2, :] .- simulated[4, :]
    return sqrt(sum(rI .^ 2) + sum(rD .^ 2))
end

# Raw (non-sqrt) pooled RSS for PLE (Wilks chi2 applies)
function loss_rss_dualobs(observed::AbstractMatrix{Float64}, simulated::AbstractMatrix{Float64})
    rI = observed[1, :] .- simulated[2, :]
    rD = observed[2, :] .- simulated[4, :]
    return sum(rI .^ 2) + sum(rD .^ 2)
end

# -------------------------------------------------------------
# Metrics (same as single-observable experiment + gamma/delta MAE)
# -------------------------------------------------------------
function f1_score(detected::Vector{Int}, true_cps::Vector{Int}; tolerance::Int=TOL)
    tp = 0
    used = falses(length(true_cps))
    for d in detected
        for (i, t) in enumerate(true_cps)
            if abs(d - t) <= tolerance && !used[i]
                tp += 1
                used[i] = true
                break
            end
        end
    end
    precision = length(detected) > 0 ? tp / length(detected) : (length(true_cps) == 0 ? 1.0 : 0.0)
    recall    = length(true_cps) > 0 ? tp / length(true_cps) : 1.0
    f1        = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    return precision, recall, f1
end

function cp_location_error(detected::Vector{Int}, true_cps::Vector{Int})
    length(detected) == 0 && return NaN
    errs = Float64[]
    used = falses(length(true_cps))
    for d in detected
        best_err = Inf
        best_i = 0
        for (i, t) in enumerate(true_cps)
            if !used[i]
                err = abs(d - t)
                if err < best_err
                    best_err = err
                    best_i = i
                end
            end
        end
        if best_i > 0
            push!(errs, best_err)
            used[best_i] = true
        end
    end
    return length(errs) > 0 ? mean(errs) : NaN
end

# Per-segment MAE for a parameter (SP: aligned to true segments; NSP: global
# value against each true segment value).
function param_mae(detected::Vector{Int}, params::Vector{Float64}, parnames::Tuple,
                   n_global::Int, n_seg::Int, pname::Symbol, true_seg_vals::Vector{Float64})
    length(detected) == 0 && return NaN
    # global parameter?
    gidx = findfirst(==(pname), parnames[1:n_global])
    if gidx !== nothing
        return mean(abs.(params[gidx] .- true_seg_vals))
    end
    seg_names = parnames[n_global+1:n_global+n_seg]
    bidx = findfirst(==(pname), seg_names)
    bidx === nothing && return NaN
    n_segments = length(detected) + 1
    seg_vals = [params[n_global + (i - 1) * n_seg + bidx] for i in 1:n_segments]
    aligned = Float64[]
    used = falses(length(true_seg_vals))
    for b in seg_vals
        best_i = 0
        best_err = Inf
        for (i, tb) in enumerate(true_seg_vals)
            if !used[i]
                err = abs(b - tb)
                if err < best_err
                    best_err = err
                    best_i = i
                end
            end
        end
        if best_i > 0
            push!(aligned, abs(b - true_seg_vals[best_i]))
            used[best_i] = true
        end
    end
    return length(aligned) > 0 ? mean(aligned) : NaN
end

function param_std_across_segments(params::Vector{Float64}, parnames::Tuple, n_global::Int, n_seg::Int, pname::Symbol)
    seg_names = parnames[n_global+1:n_global+n_seg]
    idx_in_block = findfirst(==(pname), seg_names)
    idx_in_block === nothing && return NaN
    n_segments = div(length(params) - n_global, n_seg)
    vals = [params[n_global + (i - 1) * n_seg + idx_in_block] for i in 1:n_segments]
    return std(vals)
end

# -------------------------------------------------------------
# MICA setup builders (identical to the single-observable run)
# -------------------------------------------------------------
function build_optimizer(seed::Int)
    ga = Evolutionary.GA(
        populationSize=N_POP,
        selection=Evolutionary.uniformranking(15),
        crossover=Evolutionary.MILX(0.01, 0.17, 0.5),
        mutationRate=0.3,
        crossoverRate=0.6,
        mutation=Evolutionary.gaussian(0.02)
    )
    return EvolutionaryOptimizer(ga, options=Evolutionary.Options(show_trace=false, iterations=N_ITER), seed=seed)
end

function build_config(::Val{:M1}, n::Int, seed::Int)
    n_global = 2
    n_seg = 1
    parnames = (:γ, :δ, :β)
    chrom = [0.1, 0.02, 0.2, 0.2, 0.2]
    lower = [0.01, 0.001, 0.05, 0.05, 0.05]
    upper = [0.5, 0.1, 1.0, 1.0, 1.0]
    spec = ODEModelSpec(sird_solver, Dict(:γ=>0.1, :δ=>0.02, :β=>0.2), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames, build_optimizer(seed)
end

function build_config(::Val{:M2}, n::Int, seed::Int)
    n_global = 1
    n_seg = 2
    parnames = (:δ, :β, :γ)
    chrom = [0.02, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]
    lower = [0.001, 0.05, 0.01, 0.05, 0.01, 0.05, 0.01]
    upper = [0.1, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
    spec = ODEModelSpec(sird_solver, Dict(:δ=>0.02, :β=>0.2, :γ=>0.1), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames, build_optimizer(seed)
end

function build_config(::Val{:M3}, n::Int, seed::Int)
    n_global = 0
    n_seg = 3
    parnames = (:β, :γ, :δ)
    chrom = [0.2, 0.1, 0.02, 0.2, 0.1, 0.02, 0.2, 0.1, 0.02]
    lower = [0.05, 0.01, 0.001, 0.05, 0.01, 0.001, 0.05, 0.01, 0.001]
    upper = [1.0, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0, 0.5, 0.1]
    spec = ODEModelSpec(sird_solver, Dict(:β=>0.2, :γ=>0.1, :δ=>0.02), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames, build_optimizer(seed)
end

function build_config(::Val{:M4}, n::Int, seed::Int)
    n_global = 2
    n_seg = 1
    parnames = (:β, :δ, :γ)
    chrom = [0.3, 0.02, 0.1, 0.1, 0.1]
    lower = [0.05, 0.001, 0.01, 0.01, 0.01]
    upper = [1.0, 0.1, 0.5, 0.5, 0.5]
    spec = ODEModelSpec(sird_solver, Dict(:β=>0.3, :δ=>0.02, :γ=>0.1), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames, build_optimizer(seed)
end

# -------------------------------------------------------------
# Single run
# -------------------------------------------------------------
function run_config(ds::Dict{String,Any}, config_name::String, seed_opt::Int)
    yI = Float64.(ds["yI"])
    yD = Float64.(ds["yD"])
    n = length(yI)
    data = Matrix(hcat(yI, yD)')          # 2 x n observed matrix
    s_I = Float64(ds["s_I"])
    s_D = Float64(ds["s_D"])
    loss_fn = loss_rss_dualobs
    true_cps = Int.(ds["true_cps"])
    true_betas  = Float64.(ds["true_betas"])
    true_gammas = fill(Float64(ds["true_gamma"]), 3)
    true_deltas = fill(Float64(ds["true_delta"]), 3)

    manager, n_global, n_seg, chrom, bounds, parnames, optimizer =
        build_config(Val(Symbol(config_name)), n, seed_opt)

    t0 = time()
    cps, params = detect_changepoints(
        objective_function,
        n, n_global, n_seg,
        manager,
        loss_fn,
        data,
        copy(chrom),
        parnames,
        (copy(bounds[1]), copy(bounds[2])),
        optimizer,
        MIN_LENGTH, STEP;
        objective_type=OBJECTIVE,
        scale_penalty=SCALE_PENALTY,
        penalty_fn=(p, n) -> 0.0,
        verbose=false,
        animate=false
    )
    runtime = time() - t0

    cps = filter(c -> c > MIN_LENGTH && c < n - MIN_LENGTH, cps)
    cps = sort(unique(cps))

    obj_val = objective_function(params, cps, parnames, n_global, n_seg, manager, loss_fn, data)

    sim_mat, _ = simulate_full_model(params, cps, parnames, n_global, n_seg, manager, data)

    p, r, f1 = f1_score(cps, true_cps)
    cperr = cp_location_error(cps, true_cps)
    b_err = param_mae(cps, params, parnames, n_global, n_seg, :β, true_betas)
    g_err = param_mae(cps, params, parnames, n_global, n_seg, :γ, true_gammas)
    d_err = param_mae(cps, params, parnames, n_global, n_seg, :δ, true_deltas)
    gamma_std = param_std_across_segments(params, parnames, n_global, n_seg, :γ)
    delta_std = param_std_across_segments(params, parnames, n_global, n_seg, :δ)

    nan2nothing(x) = x isa Real && isnan(x) ? nothing : x

    return Dict(
        "config" => config_name,
        "sigma" => ds["sigma"],
        "seed" => ds["seed"],
        "n" => n,
        "true_cps" => true_cps,
        "detected_cps" => cps,
        "n_cps" => length(cps),
        "precision" => p,
        "recall" => r,
        "f1" => f1,
        "cp_location_error" => nan2nothing(cperr),
        "beta_mae" => nan2nothing(b_err),
        "gamma_mae" => nan2nothing(g_err),
        "delta_mae" => nan2nothing(d_err),
        "gamma_std" => nan2nothing(gamma_std),
        "delta_std" => nan2nothing(delta_std),
        "obj_value" => obj_val,
        "runtime" => runtime,
        "params" => params,
        "parnames" => string.(parnames),
        "simulation" => vec(sim_mat[2:2, :]),
        "simulation_D" => vec(sim_mat[4:4, :]),
        "s_I" => s_I,
        "s_D" => s_D,
        "y" => yI,
        "yI" => yI,
        "yD" => yD
    )
end

# -------------------------------------------------------------
# Main: one configuration, all 9 (sigma, seed) combos
# -------------------------------------------------------------
function main()
    sigmas = [0.5, 1.0, 2.0]
    seeds = [1, 2, 3]

    out_path = joinpath(OUT_DIR, "sird_dualobs_$(CONFIG_NAME).json")

    datasets = Dict((σ, s) => generate_sird_data_dualobs(s, σ) for σ in sigmas, s in seeds)
    jobs = [(σ, s) for σ in sigmas for s in seeds]
    n_jobs = length(jobs)

    results = Vector{Dict{String,Any}}(undef, n_jobs)

    println("[$CONFIG_NAME] Running $n_jobs dual-observable SIRD fits on $(Threads.nthreads()) threads")
    println("  beta=$BETA_VALUES  objective=$OBJECTIVE  scale_penalty=$SCALE_PENALTY  loss=pooled RSS (n=160-consistent MDL evidence), channels I+D")
    flush(stdout)

    Threads.@threads for k in 1:n_jobs
        σ, s = jobs[k]
        opt_seed = 1000 + Int(round(σ)) * 10 + s + k
        try
            ds = datasets[(σ, s)]
            rec = run_config(ds, CONFIG_NAME, opt_seed)
            results[k] = rec
            println("  [$k/$n_jobs] σ=$σ seed=$s $CONFIG_NAME -> CPs=$(rec["detected_cps"]) F1=$(round(rec["f1"], digits=2))")
        catch e
            println("  [$k/$n_jobs] σ=$σ seed=$s $CONFIG_NAME FAILED: $e")
            results[k] = Dict(
                "config" => CONFIG_NAME,
                "sigma" => σ,
                "seed" => s,
                "error" => string(e)
            )
        end
        flush(stdout)
    end

    open(out_path, "w") do f
        JSON.print(f, results, 2)
    end

    println("[$CONFIG_NAME] Done. Results saved to $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
