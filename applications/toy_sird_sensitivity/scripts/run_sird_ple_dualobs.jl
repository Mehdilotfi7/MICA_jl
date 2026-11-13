#!/usr/bin/env julia
# ============================================================
# Profile-likelihood identifiability analysis for the toy-SIRD
# DUAL-observable control experiment (I and D both observed).
#
# Usage:
#   julia --threads=10 run_sird_ple_dualobs.jl <config> <mode> [out_dir]
#
# config : M1, M2, M3, or M4
# mode   : conditional or joint
# out_dir: defaults to ../results/ple/<config>_<mode>
#
# Dataset: sigma=0.5, seed=1 from results/data/sird_dualobs_<config>.json
# (written by run_sird_complexity_dualobs.jl).  Changepoints are fixed
# at the true locations [20, 40].
#
# Loss: raw pooled RSS over both observed channels (I and D), same scale
# and chi^2_1 threshold convention as the single-observable PLE:
# threshold = best_loss + 3.8415.
# ============================================================

using Pkg
const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(joinpath(ROOT, "codes", "Mica.jl"))
push!(LOAD_PATH, joinpath(ROOT, "codes", "BreakpointProfiles_v2.jl"))
Pkg.instantiate()

using Mica
using BreakpointProfiles
using JSON
using CSV, DataFrames
using Statistics, Random
using OrdinaryDiffEq
using Evolutionary, Optim
using Printf
using Dates

# -------------------------------------------------------------
# Settings
# -------------------------------------------------------------
const CONFIG_NAME   = length(ARGS) >= 1 ? ARGS[1] : error("config required: M1|M2|M3|M4")
const MODE          = lowercase(length(ARGS) >= 2 ? ARGS[2] : "conditional")
@assert MODE in ("conditional", "joint") "mode must be 'conditional' or 'joint'"
const CPSET         = lowercase(length(ARGS) >= 3 ? ARGS[3] : "true")
@assert CPSET in ("true", "detected") "cpset must be 'true' or 'detected'"
const PLE_BASE      = CPSET == "true" ? "ple" : "ple_detected"
const OUT_DIR       = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "..", "results", PLE_BASE, "$(CONFIG_NAME)_$(MODE)")
mkpath(OUT_DIR)

const RESULTS_JSON  = joinpath(@__DIR__, "..", "results", "data", "sird_dualobs_$(CONFIG_NAME).json")
const SIGMA_EXAMPLE = 1.0
const SEED_EXAMPLE  = 2
const FIXED_CPS     = Int[20, 40]

const CHISQ_95      = 3.8414588206941285

const REF_POP       = parse(Int, get(ENV, "SIRD_PLE_REF_POP", "150"))
const REF_ITER      = parse(Int, get(ENV, "SIRD_PLE_REF_ITER", "1000"))
const REF_MULTISTART= parse(Int, get(ENV, "SIRD_PLE_REF_MULTISTART", "10"))
const REF_PERT      = parse(Float64, get(ENV, "SIRD_PLE_REF_PERT", "0.3"))

const D2D_SAMPLESIZE  = parse(Int, get(ENV, "SIRD_PLE_D2D_SAMPLESIZE", "100"))
const D2D_REL_STEP    = parse(Float64, get(ENV, "SIRD_PLE_D2D_REL_STEP", "0.1"))
const D2D_STEP_FACTOR = parse(Float64, get(ENV, "SIRD_PLE_D2D_STEP_FACTOR", "1.5"))
const D2D_STOP_MARGIN = parse(Float64, get(ENV, "SIRD_PLE_D2D_STOP_MARGIN", "1.2"))
const D2D_POLISH      = parse(Bool, get(ENV, "SIRD_PLE_D2D_POLISH", "true"))
const D2D_SMOOTH      = parse(Bool, get(ENV, "SIRD_PLE_D2D_SMOOTH", "true"))
const D2D_ALLOW_BETTER= parse(Bool, get(ENV, "SIRD_PLE_D2D_ALLOW_BETTER", "true"))

const CP_WINDOW       = parse(Int, get(ENV, "SIRD_PLE_CP_WINDOW", "7"))

const JOINT_WINDOW    = parse(Int, get(ENV, "SIRD_PLE_JOINT_WINDOW", "7"))
const JOINT_STEP      = parse(Int, get(ENV, "SIRD_PLE_JOINT_STEP", "1"))
const JOINT_PARALLEL_OVER_PARAMS = parse(Bool, get(ENV, "SIRD_PLE_JOINT_PARALLEL_OVER_PARAMS", "true"))
const JOINT_PARALLELIZE_GRID     = parse(Bool, get(ENV, "SIRD_PLE_JOINT_PARALLELIZE_GRID", "true"))

# -------------------------------------------------------------
# SIRD model
# -------------------------------------------------------------
function sird_model!(du, u, p, t)
    S, I, R, D = u
    N = 1000.0
    β = p.β
    γ = p.γ
    δ = p.δ
    du[1] = -β * S * I / N
    du[2] =  β * S * I / N - γ * I - δ * I
    du[3] =  γ * I
    du[4] =  δ * I
end

function sird_solver(params, tspan::Tuple{Float64,Float64}, u0::Vector{Float64})
    prob = ODEProblem(sird_model!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
    return hcat(sol.u...)
end

# -------------------------------------------------------------
# Config builders (identical to the single-observable PLE)
# -------------------------------------------------------------
function build_config(::Val{:M1}, n::Int)
    n_global = 2
    n_seg    = 1
    parnames = (:γ, :δ, :β)
    chrom    = [0.1, 0.02, 0.4]
    lower    = [0.01, 0.001, 0.05]
    upper    = [0.5, 0.1, 1.0]
    spec = ODEModelSpec(sird_solver, Dict(:γ=>0.1, :δ=>0.02, :β=>0.2), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames
end

function build_config(::Val{:M2}, n::Int)
    n_global = 1
    n_seg    = 2
    parnames = (:δ, :β, :γ)
    chrom    = [0.02, 0.4, 0.1]
    lower    = [0.001, 0.05, 0.01]
    upper    = [0.1, 1.0, 0.5]
    spec = ODEModelSpec(sird_solver, Dict(:δ=>0.02, :β=>0.2, :γ=>0.1), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames
end

function build_config(::Val{:M3}, n::Int)
    n_global = 0
    n_seg    = 3
    parnames = (:β, :γ, :δ)
    chrom    = [0.4, 0.1, 0.02]
    lower    = [0.05, 0.01, 0.001]
    upper    = [1.0, 0.5, 0.1]
    spec = ODEModelSpec(sird_solver, Dict(:β=>0.2, :γ=>0.1, :δ=>0.02), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames
end

function build_config(::Val{:M4}, n::Int)
    n_global = 2
    n_seg    = 1
    parnames = (:β, :δ, :γ)
    chrom    = [0.4, 0.02, 0.1]
    lower    = [0.05, 0.001, 0.01]
    upper    = [1.0, 0.1, 0.5]
    spec = ODEModelSpec(sird_solver, Dict(:β=>0.3, :δ=>0.02, :γ=>0.1), [990.0, 10.0, 0.0, 0.0], (0.0, Float64(n)))
    manager = ModelManager(spec)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames
end

# -------------------------------------------------------------
# Optimizers
# -------------------------------------------------------------
function build_hybrid_optimizer_serial()
    ga = EvolutionaryPLEConfig(
        population_size=REF_POP, iterations=REF_ITER,
        selection="tournament(2)", crossover="SBX(0.7, 1)",
        mutation="gaussian(0.0001)", mutation_rate=0.7, crossover_rate=0.7,
        parallelization=:serial
    )
    local_opt = OptimPLEConfig(
        Fminbox(LBFGS()),
        Optim.Options(show_trace=false, iterations=200)
    )
    return HybridOptimizerConfig(global_optimizer=ga, local_optimizer=local_opt)
end

function build_neldermead_optimizer()
    return OptimPLEConfig(
        Fminbox(NelderMead()),
        Optim.Options(show_trace=false, iterations=2000)
    )
end

function run_multistart_ref(mica_objective, x0, lb_full, ub_full, cps_ref)
    opt = build_hybrid_optimizer_serial()
    range = ub_full .- lb_full
    results = Vector{Tuple{Float64, Vector{Float64}}}(undef, REF_MULTISTART)
    inits = Vector{Vector{Float64}}(undef, REF_MULTISTART)
    inits[1] = copy(x0)
    for s in 2:REF_MULTISTART
        x_pert = copy(x0)
        for j in 1:length(x0)
            δ = REF_PERT * range[j] * (2.0 * rand() - 1.0)
            x_pert[j] = clamp(x_pert[j] + δ, lb_full[j], ub_full[j])
        end
        inits[s] = x_pert
    end

    Base.Threads.@threads for s in 1:REF_MULTISTART
        Random.seed!(s + Threads.threadid() * 1000)
        try
            loss, x_opt = optimize_ple((p) -> mica_objective(p, cps_ref), inits[s], lb_full, ub_full, opt)
            results[s] = (loss, x_opt)
            println("  Reference multi-start $s: loss = $(round(loss, sigdigits=6))")
        catch e
            println("  Reference multi-start $s failed: $e")
            results[s] = (Inf, copy(inits[s]))
        end
    end

    best_loss = Inf
    best_x = copy(x0)
    for (loss, x_opt) in results
        if isfinite(loss) && loss < best_loss
            best_loss = loss
            best_x = copy(x_opt)
        end
    end
    return best_loss, best_x
end

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
function main()
    println("[$CONFIG_NAME] Loading dual-observable toy dataset from $RESULTS_JSON")
    records = JSON.parsefile(RESULTS_JSON)
    rec = findfirst(r -> r["config"] == CONFIG_NAME && r["sigma"] == SIGMA_EXAMPLE && r["seed"] == SEED_EXAMPLE && !haskey(r, "error"), records)
    rec === nothing && error("Record not found for $CONFIG_NAME, sigma=$SIGMA_EXAMPLE, seed=$SEED_EXAMPLE")
    ds = records[rec]

    yI = Float64.(ds["yI"])
    yD = Float64.(ds["yD"])
    n = length(yI)
    data = Matrix(hcat(yI, yD)')
    s_I = Float64(ds["s_I"])
    s_D = Float64(ds["s_D"])

    # PLE loss: raw (non-sqrt) pooled RSS over both channels
    function loss_rss_dualobs(observed::AbstractMatrix{Float64}, simulated::AbstractMatrix{Float64})
        rI = observed[1, :] .- simulated[2, :]
        rD = observed[2, :] .- simulated[4, :]
        return sum(rI .^ 2) + sum(rD .^ 2)
    end

    true_cps = Int.(ds["true_cps"])
    true_betas = haskey(ds, "true_betas") ? Float64.(ds["true_betas"]) : [0.4, 0.1, 0.3]
    true_gamma = haskey(ds, "true_gamma") ? Float64(ds["true_gamma"]) : 0.1
    true_delta = haskey(ds, "true_delta") ? Float64(ds["true_delta"]) : 0.02

    cps = if CPSET == "true"
        sort(copy(FIXED_CPS))
    else
        # anchor at the CPs detected for this config on this dataset
        sort(Int.(ds["detected_cps"]))
    end
    isempty(cps) && error("No detected CPs available for $CONFIG_NAME (sigma=$SIGMA_EXAMPLE, seed=$SEED_EXAMPLE)")
    println("[$CONFIG_NAME] SIRD dual-observable toy data: n=$n, sigma=$SIGMA_EXAMPLE, seed=$SEED_EXAMPLE, true CPs=$true_cps, fixed CPs=$cps")
    println("[$CONFIG_NAME] channel scales: s_I=$(round(s_I, digits=2)), s_D=$(round(s_D, digits=2))")
    println("[$CONFIG_NAME] Running on $(Threads.nthreads()) thread(s), $REF_MULTISTART reference multi-starts")

    manager, n_global, n_segment_specific, chrom, bounds, parnames = build_config(Val(Symbol(CONFIG_NAME)), n)
    lb, ub = bounds

    n_segments = length(cps) + 1
    n_params = n_global + n_segments * n_segment_specific
    lb_full = [lb[1:n_global]; repeat(lb[n_global+1:end], n_segments)]
    ub_full = [ub[1:n_global]; repeat(ub[n_global+1:end], n_segments)]

    param_labels = String[]
    for i in 1:n_global
        push!(param_labels, string(parnames[i]) * "_global")
    end
    for s in 1:n_segments
        for i in 1:n_segment_specific
            push!(param_labels, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end

    x0 = [chrom[1:n_global]; repeat(chrom[n_global+1:end], n_segments)]

    # For the detected anchor, the reference search starts from the detection
    # run's fitted parameters (truncated to the actual number of segments) —
    # the GA alone can miss the good basin at off-truth CP locations.
    if CPSET == "detected" && haskey(ds, "params")
        det_params = Float64.(ds["params"])
        n_need = n_global + n_segments * n_segment_specific
        if length(det_params) >= n_need
            x0 = det_params[1:n_need]
            println("[$CONFIG_NAME] Detected anchor: reference starts from detection-run params")
        end
    end

    function mica_objective(pars, cps_)
        return objective_function(pars, cps_, parnames, n_global, n_segment_specific, manager, loss_rss_dualobs, data)
    end

    base_loss = mica_objective(x0, cps)
    println("[$CONFIG_NAME] Initial weighted RSS = $(round(base_loss, sigdigits=6))")

    println("[$CONFIG_NAME] Re-optimising reference with hybrid GA+LBFGS ($REF_MULTISTART starts on $(Threads.nthreads()) threads)")
    t0_ref = time()
    best_loss, best_params = run_multistart_ref(mica_objective, x0, lb_full, ub_full, cps)
    best_params_original = copy(best_params)
    println("[$CONFIG_NAME] Best re-optimised weighted RSS = $(round(best_loss, sigdigits=6))")

    # Guard against the stochastic GA landing in a worse basin: if a
    # conditional run for this config/set already exists, adopt its best fit
    # (same data, same loss, same anchor CPs) when it beats the fresh GA.
    cond_params_file = joinpath(@__DIR__, "..", "results", PLE_BASE, "inputs", "$(CONFIG_NAME)_refit_params.csv")
    if MODE == "joint" && isfile(cond_params_file)
        x_cond = Float64.(CSV.read(cond_params_file, DataFrame).value)
        cond_loss = mica_objective(x_cond, cps)
        println("[$CONFIG_NAME] Conditional-run reference RSS = $(round(cond_loss, sigdigits=6))")
        if cond_loss < best_loss
            println("[$CONFIG_NAME] Adopting conditional-run reference (better than this run's GA result)")
            best_loss = cond_loss
            best_params = x_cond
            best_params_original = copy(x_cond)
        end
    end
    println("[$CONFIG_NAME] Reference multi-start walltime = $(round((time()-t0_ref)/60, digits=1)) min")

    # chi^2_1 threshold on the raw pooled RSS (same convention as the
    # single-observable PLE): threshold = best_loss + 3.8415.
    threshold = best_loss + CHISQ_95
    println("[$CONFIG_NAME] Threshold (RSS + $(round(CHISQ_95, digits=4))) = $(round(threshold, sigdigits=6))")

    prob = ODEChangepointPLEProblem(
        objective = mica_objective,
        data = data,
        loss_fn = loss_rss_dualobs,
        changepoints = cps,
        best_params = best_params,
        best_loss = best_loss,
        lb = lb_full,
        ub = ub_full,
        param_names = param_labels,
        n_global = n_global,
        n_segment_specific = n_segment_specific,
        n_obs = n
    )

    prof_opt = build_neldermead_optimizer()
    d2d_opts = ProfileOptions(
        samplesize = D2D_SAMPLESIZE,
        rel_step_increase = D2D_REL_STEP,
        step_factor = D2D_STEP_FACTOR,
        stop_margin_factor = D2D_STOP_MARGIN,
        polish = D2D_POLISH,
        smooth_jumps = D2D_SMOOTH,
        allow_better_optimum = D2D_ALLOW_BETTER
    )

    idxs = collect(1:n_params)

    if MODE == "conditional"
        println("[$CONFIG_NAME] Profiling $(length(idxs)) parameters (conditional)...")
        t0 = time()
        param_profiles = Vector{ProfileResult}(undef, length(idxs))
        Base.Threads.@threads for k in 1:length(idxs)
            idx = idxs[k]
            param_profiles[k] = profile_parameter(prob, idx;
                threshold = threshold,
                optimizer = prof_opt,
                options = d2d_opts
            )
        end
        elapsed_params = time() - t0
        sort!(param_profiles, by=p -> p.index)
        write_profiles(joinpath(OUT_DIR, "ple_results.csv"), param_profiles)
        summary_df = ple_summary(param_profiles)
        CSV.write(joinpath(OUT_DIR, "ple_summary.csv"), summary_df)
        println("[$CONFIG_NAME] Parameter profiling completed in $(round(elapsed_params/60, digits=1)) minutes")

        println("[$CONFIG_NAME] Profiling $(length(cps)) change points...")
        t0 = time()
        cp_profiles = Vector{CPProfileResult}(undef, length(cps))
        Base.Threads.@threads for j in 1:length(cps)
            cp_profiles[j] = profile_changepoint(prob, j; window=CP_WINDOW, optimizer=prof_opt)
        end
        elapsed_cp = time() - t0
        write_cp_profiles(joinpath(OUT_DIR, "cp_profile_loss.csv"), cp_profiles)
        cp_df = cp_summary(cp_profiles, threshold)
        CSV.write(joinpath(OUT_DIR, "cp_profile_ci.csv"), cp_df)
        println("[$CONFIG_NAME] CP profiling completed in $(round(elapsed_cp/60, digits=1)) minutes")

        n_identifiable = count(r -> r.identifiable, eachrow(summary_df))
        n_better = count(r -> r.best_found_loss < r.best_loss - 0.1 * max(1.0, abs(r.best_loss)), eachrow(summary_df))
        n_with_failures = count(r -> r.n_failed > 0, eachrow(summary_df))
        n_cp_identifiable = count(r -> r.identifiable, eachrow(cp_df))

        mkpath(joinpath(OUT_DIR, "..", "inputs"))
        CSV.write(joinpath(OUT_DIR, "..", "inputs", "$(CONFIG_NAME)_detected_cps.csv"), DataFrame(cp=cps))
        CSV.write(joinpath(OUT_DIR, "..", "inputs", "$(CONFIG_NAME)_refit_params.csv"), DataFrame(parameter=param_labels, value=best_params_original))

        lines = String[]
        push!(lines, "# Profile-likelihood analysis for dual-observable toy-SIRD $CONFIG_NAME")
        push!(lines, "")
        push!(lines, "**Date:** $(Dates.now())")
        push!(lines, "**Config:** $CONFIG_NAME")
        push!(lines, "**Mode:** conditional")
        push!(lines, "**Model:** SIRD, n=$n, sigma=$SIGMA_EXAMPLE, seed=$SEED_EXAMPLE, true CPs=$true_cps")
        push!(lines, "**Channels:** I and D observed, scales s_I=$(round(s_I, digits=2)), s_D=$(round(s_D, digits=2))")
        push!(lines, "**Fixed CPs:** $cps")
        push!(lines, "**Loss:** raw pooled RSS over both observed channels (I and D)")
        push!(lines, "**Reference search:** hybrid GA+LBFGS, $REF_MULTISTART starts on $(Threads.nthreads()) threads (pop=$REF_POP, iter=$REF_ITER)")
        push!(lines, "**RSS (re-optimised):** $(round(best_loss, sigdigits=8))")
        push!(lines, "**Threshold (RSS + $(round(CHISQ_95, digits=4))):** $(round(threshold, sigdigits=8))")
        push!(lines, "**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)")
        push!(lines, "**D2D adaptive profiling:** samplesize=$D2D_SAMPLESIZE, rel_step=$D2D_REL_STEP, step_factor=$D2D_STEP_FACTOR, stop_margin=$D2D_STOP_MARGIN")
        push!(lines, "**Parameter wall time:** $(round(elapsed_params/60, digits=1)) minutes")
        push!(lines, "**CP wall time:** $(round(elapsed_cp/60, digits=1)) minutes")
        push!(lines, "")
        push!(lines, "## Conditional parameter profiles")
        push!(lines, "")
        push!(lines, "| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |")
        push!(lines, "|---|---|---|---|---|---|---|")
        for r in eachrow(summary_df)
            better = r.best_found_loss < r.best_loss - 0.1 * max(1.0, abs(r.best_loss)) ? "WARNING" : ""
            push!(lines, @sprintf("| %s | %.5g | %.5g | %.5g | %s | %d | %s |",
                                  r.parameter, r.best_value, r.ci_lower, r.ci_upper,
                                  r.identifiable ? "yes" : "no", r.n_failed, better))
        end
        push!(lines, "")
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
        push!(lines, "## Quality diagnostics")
        push!(lines, "- **Identifiable parameters (conditional):** $n_identifiable / $(length(param_profiles))")
        push!(lines, "- **Conditional profiles that found a better optimum:** $n_better")
        push!(lines, "- **Conditional profiles with optimizer failures:** $n_with_failures")
        push!(lines, "- **Identifiable changepoints:** $n_cp_identifiable / $(nrow(cp_df))")
        open(joinpath(OUT_DIR, "ple_report.md"), "w") do f
            write(f, join(lines, "\n") * "\n")
        end

    else
        println("[$CONFIG_NAME] Profiling $(length(idxs)) parameters (joint, CPs marginalised)...")
        joint_seed_file = joinpath(@__DIR__, "..", "results", PLE_BASE, "inputs", "$(CONFIG_NAME)_joint_seed_params.csv")
        extra_starts = Vector{Vector{Float64}}()
        if isfile(joint_seed_file)
            x_seed = Float64.(CSV.read(joint_seed_file, DataFrame).value)
            push!(extra_starts, x_seed)
            println("[$CONFIG_NAME] Loaded joint reference seed from $joint_seed_file")
        end
        t0 = time()
        joint_profiles = profile_all_parameters_joint(
            prob;
            indices = idxs,
            window = JOINT_WINDOW,
            step = JOINT_STEP,
            threshold = threshold,
            optimizer = prof_opt,
            parallel_over_parameters = JOINT_PARALLEL_OVER_PARAMS,
            parallelize_grid = JOINT_PARALLELIZE_GRID,
            reference_extra_starts = extra_starts
        )
        elapsed_joint = time() - t0
        sort!(joint_profiles, by=p -> p.index)
        write_profiles(joinpath(OUT_DIR, "joint_ple_results.csv"), joint_profiles)
        summary_df = ple_summary(joint_profiles)
        CSV.write(joinpath(OUT_DIR, "joint_ple_summary.csv"), summary_df)
        println("[$CONFIG_NAME] Joint profiling completed in $(round(elapsed_joint/60, digits=1)) minutes")

        n_identifiable = count(r -> r.identifiable, eachrow(summary_df))
        n_better = count(r -> r.best_found_loss < r.best_loss - 0.1 * max(1.0, abs(r.best_loss)), eachrow(summary_df))
        n_with_failures = count(r -> r.n_failed > 0, eachrow(summary_df))

        mkpath(joinpath(OUT_DIR, "..", "inputs"))
        CSV.write(joinpath(OUT_DIR, "..", "inputs", "$(CONFIG_NAME)_detected_cps.csv"), DataFrame(cp=cps))
        CSV.write(joinpath(OUT_DIR, "..", "inputs", "$(CONFIG_NAME)_refit_params.csv"), DataFrame(parameter=param_labels, value=best_params_original))

        lines = String[]
        push!(lines, "# Profile-likelihood analysis for dual-observable toy-SIRD $CONFIG_NAME")
        push!(lines, "")
        push!(lines, "**Date:** $(Dates.now())")
        push!(lines, "**Config:** $CONFIG_NAME")
        push!(lines, "**Mode:** joint")
        push!(lines, "**Model:** SIRD, n=$n, sigma=$SIGMA_EXAMPLE, seed=$SEED_EXAMPLE, true CPs=$true_cps")
        push!(lines, "**Channels:** I and D observed, scales s_I=$(round(s_I, digits=2)), s_D=$(round(s_D, digits=2))")
        push!(lines, "**Fixed CPs (reference):** $cps")
        push!(lines, "**Loss:** raw pooled RSS over both observed channels (I and D)")
        push!(lines, "**Reference search:** hybrid GA+LBFGS, $REF_MULTISTART starts on $(Threads.nthreads()) threads (pop=$REF_POP, iter=$REF_ITER)")
        push!(lines, "**RSS (re-optimised):** $(round(best_loss, sigdigits=8))")
        push!(lines, "**Threshold (RSS + $(round(CHISQ_95, digits=4))):** $(round(threshold, sigdigits=8))")
        push!(lines, "**Joint CP search window:** +/-$(JOINT_WINDOW), step $(JOINT_STEP)")
        push!(lines, "**Joint optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)")
        push!(lines, "**Joint parameter wall time:** $(round(elapsed_joint/60, digits=1)) minutes")
        push!(lines, "")
        push!(lines, "## Joint adaptive parameter profiles (CPs marginalised)")
        push!(lines, "")
        push!(lines, "| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |")
        push!(lines, "|---|---|---|---|---|---|---|")
        for r in eachrow(summary_df)
            better = r.best_found_loss < r.best_loss - 0.1 * max(1.0, abs(r.best_loss)) ? "WARNING" : ""
            push!(lines, @sprintf("| %s | %.5g | %.5g | %.5g | %s | %d | %s |",
                                  r.parameter, r.best_value, r.ci_lower, r.ci_upper,
                                  r.identifiable ? "yes" : "no", r.n_failed, better))
        end
        push!(lines, "")
        push!(lines, "## Quality diagnostics")
        push!(lines, "- **Identifiable parameters (joint):** $n_identifiable / $(length(joint_profiles))")
        push!(lines, "- **Joint profiles that found a better optimum:** $n_better")
        push!(lines, "- **Joint profiles with optimizer failures:** $n_with_failures")
        open(joinpath(OUT_DIR, "ple_report.md"), "w") do f
            write(f, join(lines, "\n") * "\n")
        end
    end

    println("\n[$CONFIG_NAME] PLE done. Outputs in $OUT_DIR")
end

main()
