#!/usr/bin/env julia
# ============================================================
# Task A — Run the COVID-19 Germany example under BIC/MDL/AIC
# and a κ-penalty sweep.
#
# Reuses the 11-compartment model, data and loss from
# run_covid_objectives.jl / run_covid_objectives_parallel.jl.
# Output root: revision/outputs/TASK_A
# ============================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Random
using JSON

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_ROOT = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A")
mkpath(OUT_ROOT)

function parameter_labels(parnames, n_global, n_segment_specific, n_cps)
    labels = String[]
    for i in 1:n_global
        push!(labels, string(parnames[i]) * "_global")
    end
    n_seg_total = n_cps + 1
    for s in 1:n_seg_total
        for i in 1:n_segment_specific
            push!(labels, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    return labels
end

# ---------- model definitions (from the original example) ----------
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

function loss_function(observed, simulated)
    if any(isnan, simulated)
        return Inf
    end
    infected = simulated[5, :]
    hospital = simulated[6, :]
    icu      = simulated[7, :]
    death    = simulated[9, :]
    vacc     = simulated[11, :]
    return (
        sum(abs, log_transform(infected)  .- log_transform(observed[1, :])) +
        sum(abs, log_transform(hospital)  .- log_transform(observed[2, :])) +
        sum(abs, log_transform(icu)       .- log_transform(observed[3, :])) +
        sum(abs, log_transform(death)     .- log_transform(observed[4, :])) +
        sum(abs, log_transform(vacc)      .- log_transform(observed[5, :]))
    )
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

ga = GA(populationSize=150, selection=tournament(2), crossover=SBX(0.7, 1),
        mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
opt = EvolutionaryOptimizer(ga,
        options=Evolutionary.Options(show_trace=false, iterations=1000), seed=SEED)

ga_settings = Dict(
    "populationSize" => 150,
    "selection" => "tournament(2)",
    "crossover" => "SBX(0.7, 1)",
    "mutationRate" => 0.7,
    "crossoverRate" => 0.7,
    "mutation" => "gaussian(0.0001)",
    "iterations" => 1000,
    "min_length" => min_length,
    "step" => step
)

# ---------- penalty configurations ----------
function zero_penalty(p, n)
    return 0.0
end

function make_kappa_penalty(κ)
    return (p, n) -> κ * p * log(n)
end

# Legacy κ=40 penalty that mimics the original Mica.jl-main implementation:
# only the segment-specific parameter count is used, so the penalty is a
# constant offset independent of the number of change points.
function legacy_kappa40_penalty(p, n)
    return 40.0 * n_segment_specific * log(n)
end

runs = [
    ("bic",            :bic,     zero_penalty),
    ("mdl",            :mdl,     zero_penalty),
    ("aic",            :aic,     zero_penalty),
    ("penalty_zero",   :penalty, zero_penalty),
    ("legacy_kappa40", :penalty, legacy_kappa40_penalty),
    ("kappa_10",       :penalty, make_kappa_penalty(10)),
    ("kappa_20",       :penalty, make_kappa_penalty(20)),
    ("kappa_40",       :penalty, make_kappa_penalty(40)),
    ("kappa_80",       :penalty, make_kappa_penalty(80)),
    ("kappa_160",      :penalty, make_kappa_penalty(160)),
]

# ---------- run each configuration ----------
all_summaries = Vector{Dict{String,Any}}()

for (label, obj_type, penalty_fn) in runs
    OUT_DIR = joinpath(OUT_ROOT, "results_$(label)")
    mkpath(OUT_DIR)

    println("\n[Task A] Running COVID with objective = $(obj_type), label = $(label)")
flush(stdout)
    result = Dict{String,Any}(
        "penalty_label" => label,
        "objective" => string(obj_type),
        "error" => nothing
    )

    try
        Random.seed!(SEED)
        t0 = time()
        detected_cp, params = detect_changepoints(
            objective_function, n, n_global, n_segment_specific,
            model_manager, loss_function, data_CP,
            copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
            opt,
            min_length, step;
            objective_type=obj_type,
            penalty_fn=penalty_fn,
            data_indices=data_indices,
            verbose=false, animate=false
        )
        elapsed = time() - t0

        # No boundary filter (keep all accepted CPs)
        detected_cp = sort(unique(detected_cp))

        # Trim trailing parameter entries if the chromosome is longer than expected
        n_cps = length(detected_cp)
        expected_len = n_global + (n_cps + 1) * n_segment_specific
        if length(params) > expected_len
            @warn "  $(label): chromosome length $(length(params)) > expected $(expected_len); trimming trailing entries"
            params = params[1:expected_len]
        end

        # Compute final objective value the same way detect_changepoints does
        raw_loss = objective_function(params, detected_cp, parnames,
                                      n_global, n_segment_specific,
                                      model_manager, loss_function, data_CP)
        segment_lengths = diff([0; detected_cp; n])
        final_loss = Mica.compute_objective(raw_loss, n, n_global, n_segment_specific,
                                            n_cps, obj_type, penalty_fn,
                                            detected_cp, segment_lengths)

        # Save CPs
        cp_file = joinpath(OUT_DIR, "covid_detected_cps_origset_$(label).csv")
        CSV.write(cp_file, DataFrame(objective=label, cp=detected_cp))

        # Save parameters
        param_labs = parameter_labels(parnames, n_global, n_segment_specific, n_cps)
        if length(params) < length(param_labs)
            @warn "  $(label): chromosome length $(length(params)) < expected $(length(param_labs)); padding with NaN"
            params = [params; fill(NaN, length(param_labs) - length(params))]
        end
        param_file = joinpath(OUT_DIR, "covid_params_origset_$(label).csv")
        CSV.write(param_file, DataFrame(parameter=param_labs, value=params[1:length(param_labs)]))

        # Save summary
        result["n_cps"] = n_cps
        result["cps"] = detected_cp
        result["time_seconds"] = elapsed
        result["ga_settings"] = ga_settings
        result["seed"] = SEED
        result["loss"] = final_loss
        result["raw_loss"] = raw_loss
        result["data_indices"] = data_indices
        result["min_length"] = min_length
        result["step"] = step

        println("  #CPs = $(n_cps), CPs = $(detected_cp), time = $(round(elapsed,digits=1))s, final_loss = $(final_loss)")
        flush(stdout)

    catch e
        bt = catch_backtrace()
        err_msg = sprint(showerror, e, bt)
        result["error"] = err_msg
        @error "  $(label) failed: $(err_msg)"
    flush(stderr)
    end

    # Always write summary.json, even on failure
    summary_file = joinpath(OUT_DIR, "summary.json")
    open(summary_file, "w") do f
        JSON.print(f, result, 2)
    end
    println("  summary -> $(summary_file)")
    flush(stdout)

    push!(all_summaries, result)
end

# Write top-level sweep summary
open(joinpath(OUT_ROOT, "sweep_summary.json"), "w") do f
    JSON.print(f, all_summaries, 2)
end

println("\n[Task A] Sweep complete. Outputs in $(OUT_ROOT)")
flush(stdout)
