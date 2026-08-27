#!/usr/bin/env julia
# ============================================================
# Task B extended — scaling variants (invmean/invstd/normmean) crossed with
# alternative loss functions (sqrt/relative/abs/boxcox) and penalties
# (BIC/AIC/MDL/kappa_0p1/0.2/0.5/1/2/5/10).
#
# Usage:
#   julia run_covid_task_b_extended.jl <batch_id>
#
# The batch_id (1..12) selects 10 combinations from
# revision/outputs/TASK_B_EXTENDED/task_b_extended_plan.csv.
# ============================================================

ENV["GKSwstype"] = "nul"

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Plots
gr()
using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Random
using JSON
using Dates
using Printf
using LabelledArrays

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_ROOT    = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_B_EXTENDED")
const PLAN_FILE   = joinpath(OUT_ROOT, "task_b_extended_plan.csv")
const SEED = 1234

Mica.reset_animation()

# ---------- model definitions (from the original example) ----------
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
    sol = try
        solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-6, reltol=1.0e-6,
              isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    catch e
        nothing
    end
    if sol !== nothing && SciMLBase.successful_retcode(sol)
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

zero_penalty(p, n) = 0.0

# ---------- transforms ----------
log_transform(data, threshold=1) = [val >= threshold ? log(val) : 0 for val in data]
sqrt_transform(data) = [val > 0 ? sqrt(val) : 0 for val in data]
function boxcox_transform(data, λ=0.25)
    return [val > 0 ? (val^λ - 1) / λ : 0 for val in data]
end

# ---------- scaling / normalization variants ----------
function channel_weights(variant::String)
    if variant == "invmean"
        return Float64[mean(data_CP[k, :]) == 0 ? 1.0 : 1.0 / mean(data_CP[k, :]) for k in 1:5]
    elseif variant == "invstd"
        return Float64[std(data_CP[k, :]) == 0 ? 1.0 : 1.0 / std(data_CP[k, :]) for k in 1:5]
    elseif variant == "normmean"
        return Float64[mean(data_CP[k, :]) == 0 ? 1.0 : 1.0 / mean(data_CP[k, :]) for k in 1:5]
    else
        error("Unknown variant: $variant")
    end
end

# ---------- helpers ----------
function parameter_labels(n_cps)
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

function extract_segment_params(chromosome::Vector{Float64})
    constant = chromosome[1:n_global]
    n_segments = div(length(chromosome) - n_global, n_segment_specific)
    seg_list = Vector{Float64}[
        chromosome[n_global + (s - 1) * n_segment_specific + 1 : n_global + s * n_segment_specific]
        for s in 1:n_segments
    ]
    return constant, seg_list
end

function simulate_with_cps(cps, params)
    constant_pars, segment_pars_list = extract_segment_params(params)
    u0_curr = u0
    sim_segments = Matrix{Float64}[]
    for s in 1:(length(cps) + 1)
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end   = (s > length(cps)) ? n : cps[s]
        all_pars = @LArray [constant_pars; segment_pars_list[s]] parnames
        tspan_seg = (Float64(idx_start), Float64(idx_end))
        raw_seg = example_ode_model(all_pars, tspan_seg, u0_curr)
        sim_seg = hasproperty(raw_seg, :u) ? reduce(hcat, raw_seg.u) : raw_seg
        push!(sim_segments, sim_seg)
        u0_curr = sim_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end

function equal_weight_log_loss(sim)
    rows = data_indices
    total = 0.0
    for (k, r) in enumerate(rows)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(data_CP[k, :]))
    end
    return total
end

# ---------- loss function factory ----------
function make_loss(variant::String, loss_label::String)
    weights = channel_weights(variant)
    norm_factors = Float64[mean(data_CP[k, :]) == 0 ? 1.0 : mean(data_CP[k, :]) for k in 1:5]

    function loss_function(observed, simulated)
        if any(isnan, simulated)
            return Inf
        end
        channels = [simulated[5, :], simulated[6, :], simulated[7, :], simulated[9, :], simulated[11, :]]
        obs_channels = [observed[i, :] for i in 1:5]

        total = 0.0
        for (sim, obs, w, nf) in zip(channels, obs_channels, weights, norm_factors)
            if variant == "normmean"
                sim_norm = sim ./ nf
                obs_norm = obs ./ nf
            else
                sim_norm = sim
                obs_norm = obs
            end

            if loss_label == "log"
                total += w * sum(abs, log_transform(sim_norm) .- log_transform(obs_norm))
            elseif loss_label == "sqrt"
                total += w * sum(abs, sqrt_transform(sim_norm) .- sqrt_transform(obs_norm))
            elseif loss_label == "boxcox"
                total += w * sum(abs, boxcox_transform(sim_norm) .- boxcox_transform(obs_norm))
            elseif loss_label == "relative"
                total += w * sum(abs, (sim_norm .- obs_norm) ./ max.(obs_norm, 1.0))
            elseif loss_label == "abs"
                total += w * sum(abs, sim_norm .- obs_norm)
            else
                error("Unknown loss label: $loss_label")
            end
        end
        return total
    end
    return loss_function
end

# ---------- penalty factory ----------
function parse_kappa(label::String)
    val_str = replace(label[7:end], "p" => ".")
    return parse(Float64, val_str)
end

function build_penalty(penalty_label::String)
    if penalty_label == "bic"
        return (:bic, zero_penalty)
    elseif penalty_label == "mdl"
        return (:mdl, zero_penalty)
    elseif penalty_label == "aic"
        return (:aic, zero_penalty)
    elseif startswith(penalty_label, "kappa_")
        κ = parse_kappa(penalty_label)
        return (:penalty, (p, n) -> κ * p * log(n))
    else
        error("Unknown penalty label: $penalty_label")
    end
end

# ---------- run one combination ----------
function run_combination(variant::String, loss_label::String, penalty_label::String)
    label = "$(variant)_$(loss_label)_$(penalty_label)"
    OUT_DIR = joinpath(OUT_ROOT, "results_$(label)")
    mkpath(OUT_DIR)

    println("\n[Task B extended] $(label)")
    flush(stdout)

    obj_type, penalty_fn = build_penalty(penalty_label)
    loss_fn = make_loss(variant, loss_label)

    result = Dict{String,Any}(
        "label" => label,
        "variant" => variant,
        "loss_label" => loss_label,
        "penalty_label" => penalty_label,
        "objective_type" => string(obj_type),
        "error" => nothing
    )

    t0 = time()
    try
        Random.seed!(SEED)
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
            @warn "  $(label): chromosome length $(length(params)) > expected $(expected_len); trimming"
            params = params[1:expected_len]
        end

        raw_loss = objective_function(params, detected_cp, parnames,
                                      n_global, n_segment_specific,
                                      model_manager, loss_fn, data_CP)
        segment_lengths = diff([0; detected_cp; n])
        final_loss = Mica.compute_objective(raw_loss, n, n_global, n_segment_specific,
                                            n_cps, obj_type, penalty_fn,
                                            detected_cp, segment_lengths)

        # Common comparable metric
        try
            sim = simulate_with_cps(detected_cp, params)
            result["equal_weight_log_loss"] = equal_weight_log_loss(sim)
        catch e
            @warn "  $(label): could not compute equal-weight log-loss: $e"
            result["equal_weight_log_loss"] = NaN
        end

        CSV.write(joinpath(OUT_DIR, "covid_detected_cps_origset_$(label).csv"),
                  DataFrame(label=label, cp=detected_cp))

        param_labs = parameter_labels(n_cps)
        if length(params) < length(param_labs)
            params = [params; fill(NaN, length(param_labs) - length(params))]
        end
        CSV.write(joinpath(OUT_DIR, "covid_params_origset_$(label).csv"),
                  DataFrame(parameter=param_labs, value=params[1:length(param_labs)]))

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
    catch e
        bt = catch_backtrace()
        err_msg = sprint(showerror, e, bt)
        result["error"] = err_msg
        result["time_seconds"] = time() - t0
        @error "  $(label) failed: $(err_msg)"
    end

    summary_file = joinpath(OUT_DIR, "summary.json")
    open(summary_file, "w") do f
        JSON.print(f, result, 2)
    end
    println("  summary -> $(summary_file)")
    flush(stdout)
end

# ---------- main ----------
function main()
    if length(ARGS) < 1
        println("Usage: julia run_covid_task_b_extended.jl <batch_id>")
        exit(1)
    end
    batch_id = parse(Int, ARGS[1])
    if !(1 <= batch_id <= 12)
        error("batch_id must be between 1 and 12")
    end

    if !isfile(PLAN_FILE)
        error("Plan file not found: $(PLAN_FILE)")
    end
    plan = CSV.read(PLAN_FILE, DataFrame)
    subset = plan[plan.batch_id .== batch_id, :]

    println("[Task B extended] Batch $(batch_id): $(nrow(subset)) combinations")
    for r in eachrow(subset)
        run_combination(String(r.variant), String(r.loss), String(r.penalty))
    end

    println("\n[Task B extended] Batch $(batch_id) finished.")
end

main()
