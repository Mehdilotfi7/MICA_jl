#!/usr/bin/env julia
# ============================================================
# COVID-19 Germany example: change-point detection with L2 Gaussian NLL
# and BIC penalty.  Per-channel σ is estimated from a no-CP baseline fit.
#
# The goal is to check whether the L2 Gaussian loss detects the same
# change points as the MICA default L1 log loss.
#
# Output root: revision/outputs/TASK_A_L2
# ============================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Random
using JSON
using LinearAlgebra

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_ROOT = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A_L2")
mkpath(OUT_ROOT)

# ---------- model definitions (same as Task A) ----------
function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ * cos(2 * π * ((t - t₀) / 365))
end

function log_transform(data, threshold=1)
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

function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0)
    if tspan[2] < 330
        params[:ν] = 0.0
    end
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-6, reltol=1.0e-6,
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    if SciMLBase.successful_retcode(sol)
        return sol[:, :]
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

# ---------- load data ----------
cases_CP = CSV.read(joinpath(EXAMPLE_DIR, "case_rki_daily.csv"), DataFrame).total
hospital_CP = CSV.read(joinpath(EXAMPLE_DIR, "Hospitalization_rki_daily.csv"), DataFrame).total
death_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "death_rki_daily.csv"), DataFrame).Todesfaelle_neu)
icu_CP = CSV.read(joinpath(EXAMPLE_DIR, "icu_rki_daily.csv"), DataFrame).total
vacc_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "vaccination_rki_daily_allShots.csv"), DataFrame).Total)

data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
max_length = maximum(length, data_CP)
data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
data_CP = [vector[1:400] for vector in data_CP]
data_CP[1] = hma(data_CP[1], 14)
data_CP[4] = hma(data_CP[4], 14)
data_CP[5] = hma(data_CP[5], 14)
data_CP = Matrix(reduce(hcat, data_CP)')

# ---------- MICA settings ----------
parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
            :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9, 1/16, 0.0055,
                      0.2, 0.05, 0.17, 0.144, 0.01, 0.017, 0.173, 0.01]
lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
         0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
upper = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
         0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]
bounds = (lower, upper)
u0 = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tspan = (0.0, 399.0)

ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
model_manager = ModelManager(ode_spec)

n_global = 8
n_segment_specific = 8
min_length = 10
step = 10
data_indices = [5, 6, 7, 9, 11]
n = size(data_CP, 2)

const SEED = 1234

# ---------- flat simulation helper (handles DiffEqArray segments) ----------
function flatten_simulation(sim)
    if sim isa Matrix && !(eltype(sim) <: Number)
        segments = Matrix{Float64}[hcat(seg.u...) for seg in sim]
        return reduce(hcat, segments)
    elseif !(sim isa AbstractMatrix{<:Number})
        return hcat(sim.u...)
    end
    return sim
end

# ---------- L2 Gaussian loss with fixed per-channel σ ----------
function build_l2_loss(sigmas)
    function loss_l2(observed, simulated)
        if any(isnan, simulated)
            return Inf
        end
        total = 0.0
        for (k, r) in enumerate(data_indices)
            σ = sigmas[k]
            resid = log_transform(simulated[r, :]) .- log_transform(observed[k, :])
            total += sum(x -> (x / σ)^2, resid)
        end
        return total
    end
    return loss_l2
end

# ---------- estimate per-channel σ from a parameter set ----------
function estimate_sigmas_from_params(params)
    sim_raw, _ = Mica.simulate_full_model(
        params, Int[], parnames, n_global, n_segment_specific,
        model_manager, data_CP; data_indices=data_indices
    )
    sim_full = flatten_simulation(sim_raw)
    sigmas = Float64[]
    for (k, r) in enumerate(data_indices)
        resid = log_transform(sim_full[r, :]) .- log_transform(data_CP[k, :])
        push!(sigmas, max(std(resid), 1e-6))
    end
    return sigmas
end

# ---------- fit no-CP baseline under L2 loss and estimate σ ----------
# Pure L2 pipeline: start from a σ estimate based on the initial chromosome,
# then iterate L2-weighted baseline fits and σ updates until convergence.
function fit_baseline_l2(sigmas0; max_iter=5, tol=1e-2)
    println("[L2 BIC] Fitting no-CP baseline under L2 Gaussian loss...")
    flush(stdout)
    Random.seed!(SEED)
    ga = GA(populationSize=150, selection=tournament(2), crossover=SBX(0.7, 1),
            mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
    opt = EvolutionaryOptimizer(ga,
            options=Evolutionary.Options(show_trace=false, iterations=1000), seed=SEED)

    sigmas = copy(sigmas0)
    best_params = nothing
    best_loss = Inf
    for iter in 1:max_iter
        loss_l2_iter = build_l2_loss(sigmas)
        baseline_loss, baseline_params = optimize_with_changepoints(
            Mica.objective_function, copy(initial_chromosome), parnames, Int[],
            (copy(bounds[1]), copy(bounds[2])), opt,
            n_global, n_segment_specific,
            model_manager, loss_l2_iter, data_CP
        )
        new_sigmas = estimate_sigmas_from_params(baseline_params)
        println("[L2 BIC] Baseline L2 iter $(iter): loss=$(round(baseline_loss,digits=4)), σ=$(round.(new_sigmas,digits=4))")
        flush(stdout)

        if baseline_loss < best_loss
            best_loss = baseline_loss
            best_params = baseline_params
        end

        if norm(new_sigmas .- sigmas) / max(norm(sigmas), 1e-6) < tol
            sigmas = new_sigmas
            break
        end
        sigmas = new_sigmas
    end
    println("[L2 BIC] Final baseline L2 loss = $(round(best_loss, digits=4))")
    println("[L2 BIC] Final per-channel σ = $(round.(sigmas, digits=4))")
    flush(stdout)
    return sigmas, best_loss, best_params
end

function fit_baseline_and_estimate_sigma()
    # Initialise σ from a simulation at the default initial chromosome (no CPs).
    # This gives a model-based scale estimate for each channel without any L1 fit.
    sigmas0 = estimate_sigmas_from_params(initial_chromosome)
    println("[L2 BIC] Initial σ from default chromosome: $(round.(sigmas0, digits=4))")
    return fit_baseline_l2(sigmas0)
end

# ---------- L2 Gaussian loss with fixed per-channel σ ----------
function build_l2_loss(sigmas)
    function loss_l2(observed, simulated)
        if any(isnan, simulated)
            return Inf
        end
        total = 0.0
        for (k, r) in enumerate(data_indices)
            σ = sigmas[k]
            resid = log_transform(simulated[r, :]) .- log_transform(observed[k, :])
            total += sum(x -> (x / σ)^2, resid)
        end
        return total
    end
    return loss_l2
end

# ---------- run BIC change-point detection with L2 loss ----------
function run_bic_l2(sigmas)
    println("\n[L2 BIC] Running change-point detection with BIC penalty and L2 Gaussian loss...")
    flush(stdout)

    loss_l2 = build_l2_loss(sigmas)

    Random.seed!(SEED)
    ga = GA(populationSize=150, selection=tournament(2), crossover=SBX(0.7, 1),
            mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
    opt = EvolutionaryOptimizer(ga,
            options=Evolutionary.Options(show_trace=false, iterations=1000), seed=SEED)

    t0 = time()
    detected_cp, params = detect_changepoints(
        Mica.objective_function, n, n_global, n_segment_specific,
        model_manager, loss_l2, data_CP,
        copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
        opt,
        min_length, step;
        objective_type=:bic,
        penalty_fn=(p, n) -> 0.0,
        data_indices=data_indices,
        verbose=false, animate=false
    )
    elapsed = time() - t0
    detected_cp = sort(unique(detected_cp))
    n_cps = length(detected_cp)

    # Trim/pad chromosome to expected length
    expected_len = n_global + (n_cps + 1) * n_segment_specific
    if length(params) > expected_len
        params = params[1:expected_len]
    elseif length(params) < expected_len
        params = [params; fill(NaN, expected_len - length(params))]
    end

    raw_loss = Mica.objective_function(params, detected_cp, parnames,
                                       n_global, n_segment_specific,
                                       model_manager, loss_l2, data_CP)
    segment_lengths = diff([0; detected_cp; n])
    final_bic = Mica.compute_objective(raw_loss, n, n_global, n_segment_specific,
                                       n_cps, :bic, (p, n) -> 0.0,
                                       detected_cp, segment_lengths)

    println("[L2 BIC] Detected CPs: $(detected_cp)")
    println("[L2 BIC] #CPs = $(n_cps), raw L2 loss = $(round(raw_loss, digits=4)), BIC = $(round(final_bic, digits=4)), time = $(round(elapsed, digits=1))s")
    flush(stdout)

    # Save outputs
    OUT_DIR = joinpath(OUT_ROOT, "results_bic_l2")
    mkpath(OUT_DIR)

    CSV.write(joinpath(OUT_DIR, "covid_detected_cps_bic_l2.csv"),
              DataFrame(objective="bic_l2", cp=detected_cp))

    param_labs = String[]
    for i in 1:n_global
        push!(param_labs, string(parnames[i]) * "_global")
    end
    for s in 1:(n_cps+1)
        for i in 1:n_segment_specific
            push!(param_labs, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    CSV.write(joinpath(OUT_DIR, "covid_params_bic_l2.csv"),
              DataFrame(parameter=param_labs, value=params[1:length(param_labs)]))

    summary = Dict(
        "objective" => "bic_l2",
        "n_cps" => n_cps,
        "cps" => detected_cp,
        "raw_l2_loss" => raw_loss,
        "bic" => final_bic,
        "time_seconds" => elapsed,
        "seed" => SEED,
        "ga_settings" => Dict(
            "populationSize" => 150,
            "selection" => "tournament(2)",
            "crossover" => "SBX(0.7, 1)",
            "mutationRate" => 0.7,
            "crossoverRate" => 0.7,
            "mutation" => "gaussian(0.0001)",
            "iterations" => 1000,
            "min_length" => min_length,
            "step" => step
        ),
        "channel_sigmas" => sigmas,
        "data_indices" => data_indices
    )
    open(joinpath(OUT_DIR, "summary.json"), "w") do f
        JSON.print(f, summary, 2)
    end

    println("[L2 BIC] Outputs saved to $(OUT_DIR)")
    flush(stdout)
end

# ---------- main ----------
sigmas, baseline_loss, baseline_params = fit_baseline_and_estimate_sigma()
run_bic_l2(sigmas)
