#!/usr/bin/env julia
# ============================================================
# Task A sensitivity — run COVID-19 Germany with a chosen objective
# (BIC/MDL/AIC/zero/kappa) and a chosen loss function.
#
# Usage:
#   julia run_covid_task_a_sensitivity.jl <objective> <loss>
#
#   objective : bic | mdl | aic | zero | kappa_5 | kappa_2 | kappa_1 |
#               kappa_0p5 | kappa_0p2 | kappa_0p1
#   loss      : log | sqrt | boxcox | relative | abs
#
# Output: revision/outputs/TASK_A/results_<objective>_<loss>
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

# ---------- model definitions ----------
function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ * cos(2 * π * ((t - t₀) / 365))
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
        return Matrix(sol)
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

# ---------- data loading ----------
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

# ---------- loss functions ----------
log_transform(data, threshold=1) = [val >= threshold ? log(val) : 0 for val in data]
sqrt_transform(data) = [val > 0 ? sqrt(val) : 0 for val in data]
boxcox_transform(data, λ=0.25) = [val > 0 ? (val^λ - 1) / λ : 0 for val in data]

function make_loss(loss_label::String)
    function loss_function(observed, simulated)
        if any(isnan, simulated)
            return Inf
        end
        infected = simulated[5, :]
        hospital = simulated[6, :]
        icu      = simulated[7, :]
        death    = simulated[9, :]
        vacc     = simulated[11, :]
        sim_channels = [infected, hospital, icu, death, vacc]

        if loss_label == "log"
            return sum(
                sum(abs, log_transform(sim) .- log_transform(obs))
                for (sim, obs) in zip(sim_channels, [observed[i, :] for i in 1:5])
            )
        elseif loss_label == "sqrt"
            return sum(
                sum(abs, sqrt_transform(sim) .- sqrt_transform(obs))
                for (sim, obs) in zip(sim_channels, [observed[i, :] for i in 1:5])
            )
        elseif loss_label == "boxcox"
            return sum(
                sum(abs, boxcox_transform(sim) .- boxcox_transform(obs))
                for (sim, obs) in zip(sim_channels, [observed[i, :] for i in 1:5])
            )
        elseif loss_label == "relative"
            return sum(
                sum(abs, (sim .- obs) ./ max.(obs, 1.0))
                for (sim, obs) in zip(sim_channels, [observed[i, :] for i in 1:5])
            )
        elseif loss_label == "abs"
            return sum(
                sum(abs, sim .- obs)
                for (sim, obs) in zip(sim_channels, [observed[i, :] for i in 1:5])
            )
        else
            error("Unknown loss label: $loss_label")
        end
    end
    return loss_function
end

# ---------- penalty helpers ----------
zero_penalty(p, n) = 0.0
make_kappa_penalty(κ) = (p, n) -> κ * p * log(n)

function parse_kappa(label::String)
    # label like "kappa_0p5" -> 0.5, "kappa_5" -> 5.0
    val_str = replace(label[7:end], "p" => ".")
    return parse(Float64, val_str)
end

function build_run(objective_label::String, loss_label::String)
    if objective_label == "bic"
        return ("$(objective_label)_$(loss_label)", :bic, zero_penalty, loss_label)
    elseif objective_label == "mdl"
        return ("$(objective_label)_$(loss_label)", :mdl, zero_penalty, loss_label)
    elseif objective_label == "aic"
        return ("$(objective_label)_$(loss_label)", :aic, zero_penalty, loss_label)
    elseif objective_label == "zero"
        return ("$(objective_label)_$(loss_label)", :penalty, zero_penalty, loss_label)
    elseif startswith(objective_label, "kappa_")
        κ = parse_kappa(objective_label)
        # kappa runs always use the baseline log loss for a clean sweep
        loss_for_kappa = loss_label == "log" ? "log" : error("Kappa runs must use log loss")
        return (objective_label, :penalty, make_kappa_penalty(κ), loss_for_kappa)
    else
        error("Unknown objective label: $objective_label")
    end
end

# ---------- main ----------
function main()
    if length(ARGS) < 2
        println("Usage: julia run_covid_task_a_sensitivity.jl <objective> <loss>")
        println("  objective: bic | mdl | aic | zero | kappa_5 | kappa_2 | kappa_1 | kappa_0p5 | kappa_0p2 | kappa_0p1")
        println("  loss     : log | sqrt | boxcox | relative | abs")
        exit(1)
    end

    objective_label = ARGS[1]
    loss_label = ARGS[2]

    label, obj_type, penalty_fn, effective_loss = build_run(objective_label, loss_label)
    loss_fn = make_loss(effective_loss)

    OUT_DIR = joinpath(OUT_ROOT, "results_$(label)")
    mkpath(OUT_DIR)

    println("\n[Task A sensitivity] objective = $(objective_label), loss = $(effective_loss), label = $(label)")
    flush(stdout)

    result = Dict{String,Any}(
        "penalty_label" => label,
        "objective" => string(obj_type),
        "objective_label" => objective_label,
        "loss_label" => effective_loss,
        "error" => nothing
    )

    try
        Random.seed!(SEED)
        t0 = time()
        detected_cp, params = detect_changepoints(
            objective_function, n, n_global, n_segment_specific,
            model_manager, loss_fn, data_CP,
            copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
            opt,
            min_length, step;
            objective_type=obj_type,
            penalty_fn=penalty_fn,
            data_indices=data_indices,
            verbose=false, animate=false
        )
        elapsed = time() - t0

        detected_cp = sort(unique(detected_cp))

        n_cps = length(detected_cp)
        expected_len = n_global + (n_cps + 1) * n_segment_specific
        if length(params) > expected_len
            @warn "  $(label): chromosome length $(length(params)) > expected $(expected_len); trimming trailing entries"
            params = params[1:expected_len]
        end

        raw_loss = objective_function(params, detected_cp, parnames,
                                      n_global, n_segment_specific,
                                      model_manager, loss_fn, data_CP)
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

    summary_file = joinpath(OUT_DIR, "summary.json")
    open(summary_file, "w") do f
        JSON.print(f, result, 2)
    end
    println("  summary -> $(summary_file)")
    flush(stdout)
end

main()
