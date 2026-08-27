#!/usr/bin/env julia
# ============================================================
# Run MICA on the 500-day COVID window with zero penalty.
#
# This is an unconstrained baseline: changepoints are accepted as long as
# they improve the raw equal-weight log loss.
#
# Usage:
#   julia -p 20 run_covid_zero_penalty_500.jl [out_suffix]
#
# Output: revision/outputs/TASK_G_500/winners_500_zero_penalty/
# ============================================================

using Distributed, Pkg

mica_project = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
Pkg.activate(mica_project)
proj_file = Base.active_project()
proj_dir  = dirname(proj_file)

# Use 20 workers (adjust via WORKERS env var)
nworkers_target = parse(Int, get(ENV, "WORKERS", "20"))

if nworkers() < nworkers_target
    addprocs(nworkers_target - nworkers(); exeflags="--project=$proj_dir")
end
println("Using $(nworkers()) worker(s).")

# Load Mica on the main process first, then sequentially on each worker to avoid
# concurrent package-callback races when all workers import it at once.
const MICA_HELPER = joinpath(@__DIR__, "..", "..", "..", "load_mica_helper.jl")
using Mica
for w in workers()
    remotecall_fetch(include, w, MICA_HELPER)
end

@everywhere begin
    using CSV, DataFrames, Statistics
    using Evolutionary, OrdinaryDiffEq
    using Smoothers
    using Random
    using JSON

    const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
    const OUT_ROOT    = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_G_500")

    # ---------- model ----------
    function fδ(t::Number, δ::Number, t₀::Number=0.0)
        return 1 + δ * cos(2 * π * ((t - t₀) / 365))
    end

    log_transform(data, threshold=1) = [val >= threshold ? log(val) : 0 for val in data]
    sqrt_transform(data) = [val > 0 ? sqrt(val) : 0 for val in data]
    boxcox_transform(data, λ=0.25) = [val > 0 ? (val^λ - 1) / λ : 0 for val in data]

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

    # ---------- load 500-day data ----------
    cases_CP = CSV.read(joinpath(EXAMPLE_DIR, "case_rki_daily.csv"), DataFrame).total
    hospital_CP = CSV.read(joinpath(EXAMPLE_DIR, "Hospitalization_rki_daily.csv"), DataFrame).total
    death_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "death_rki_daily.csv"), DataFrame).Todesfaelle_neu)
    icu_CP = CSV.read(joinpath(EXAMPLE_DIR, "icu_rki_daily.csv"), DataFrame).total
    vacc_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "vaccination_rki_daily_allShots.csv"), DataFrame).Total)

    data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
    max_length = maximum(length, data_CP)
    data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
    data_CP = [vector[1:500] for vector in data_CP]
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
    tspan = (0.0, 499.0)

    ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
    model_manager = ModelManager(ode_spec)

    n_global = 8
    n_segment_specific = 8
    min_length = 10
    step = 10
    data_indices = [5, 6, 7, 9, 11]
    n = size(data_CP, 2)

    const SEED = 1234

    ga_pop = parse(Int, get(ENV, "GA_POP", "300"))
    ga_iter = parse(Int, get(ENV, "GA_ITER", "2000"))
    ga = GA(populationSize=ga_pop, selection=tournament(2), crossover=SBX(0.7, 1),
            mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
    opt = EvolutionaryOptimizer(ga,
            options=Evolutionary.Options(show_trace=false, iterations=ga_iter), seed=SEED)

    zero_penalty(p, n) = 0.0

    # ---------- loss functions ----------
    function equal_log_loss(sim)
        rows = data_indices
        total = 0.0
        for (k, r) in enumerate(rows)
            total += sum(abs, log_transform(sim[r, :]) .- log_transform(data_CP[k, :]))
        end
        return total
    end

    function make_bic_mdl_loss()
        function loss_function(observed, simulated)
            any(isnan, simulated) && return Inf
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
        return loss_function
    end

    function channel_weights(variant::String)
        if variant == "invmean"
            return Float64[mean(data_CP[k, :]) == 0 ? 1.0 : 1.0 / mean(data_CP[k, :]) for k in 1:5]
        elseif variant == "invstd"
            return Float64[std(data_CP[k, :]) == 0 ? 1.0 : 1.0 / std(data_CP[k, :]) for k in 1:5]
        else
            error("Unknown variant: $variant")
        end
    end

    function make_variant_loss(variant::String, loss_label::String)
        weights = channel_weights(variant)
        norm_factors = Float64[mean(data_CP[k, :]) == 0 ? 1.0 : mean(data_CP[k, :]) for k in 1:5]

        function loss_function(observed, simulated)
            any(isnan, simulated) && return Inf
            channels = [simulated[5, :], simulated[6, :], simulated[7, :], simulated[9, :], simulated[11, :]]
            obs_channels = [observed[i, :] for i in 1:5]

            total = 0.0
            for (sim, obs, w, nf) in zip(channels, obs_channels, weights, norm_factors)
                # invmean/invstd do not additionally normalize the values
                sim_norm = sim
                obs_norm = obs

                if loss_label == "boxcox"
                    total += w * sum(abs, boxcox_transform(sim_norm) .- boxcox_transform(obs_norm))
                elseif loss_label == "sqrt"
                    total += w * sum(abs, sqrt_transform(sim_norm) .- sqrt_transform(obs_norm))
                elseif loss_label == "log"
                    total += w * sum(abs, log_transform(sim_norm) .- log_transform(obs_norm))
                else
                    error("Unknown loss label: $loss_label")
                end
            end
            return total
        end
        return loss_function
    end

    function build_penalty(penalty_label::String)
        if penalty_label == "bic"
            return (:bic, zero_penalty)
        elseif penalty_label == "mdl"
            return (:mdl, zero_penalty)
        elseif penalty_label == "aic"
            return (:aic, zero_penalty)
        elseif penalty_label == "zero_penalty"
            return (:penalty, zero_penalty)
        else
            error("Unknown penalty label: $penalty_label")
        end
    end

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
end

# ---------- run one winner ----------
function run_winner(label::String, variant::Union{String,Nothing}, loss_label::String, penalty_label::String)
    OUT_DIR = joinpath(OUT_ROOT, "winners_500_$(label)")
    mkpath(OUT_DIR)

    println("\n[winners 500] $label")
    flush(stdout)

    obj_type, penalty_fn = build_penalty(penalty_label)
    loss_fn = if variant === nothing
        make_bic_mdl_loss()
    else
        make_variant_loss(variant, loss_label)
    end

    result = Dict{String,Any}(
        "label" => label,
        "variant" => variant,
        "loss_label" => loss_label,
        "penalty_label" => penalty_label,
        "objective_type" => string(obj_type),
        "n" => n,
        "error" => nothing
    )

    t0 = time()
    try
        Random.seed!(SEED)
        detected_cp, params = detect_changepoints(
            Mica.objective_function, n, n_global, n_segment_specific,
            model_manager, loss_fn, data_CP,
            copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
            opt,
            min_length, step;
            objective_type=obj_type,
            penalty_fn=penalty_fn,
            data_indices=data_indices,
            verbose=false, animate=false,
            parallel=true
        )
        elapsed = time() - t0

        detected_cp = sort(unique(detected_cp))
        n_cps = length(detected_cp)
        expected_len = n_global + (n_cps + 1) * n_segment_specific
        if length(params) > expected_len
            @warn "  $(label): chromosome length $(length(params)) > expected $(expected_len); trimming"
            params = params[1:expected_len]
        end

        raw_loss = Mica.objective_function(params, detected_cp, parnames,
                                           n_global, n_segment_specific,
                                           model_manager, loss_fn, data_CP)
        segment_lengths = diff([0; detected_cp; n])
        final_loss = Mica.compute_objective(raw_loss, n, n_global, n_segment_specific,
                                            n_cps, obj_type, penalty_fn,
                                            detected_cp, segment_lengths)

        # Save CPs
        CSV.write(joinpath(OUT_DIR, "covid_detected_cps_$(label).csv"),
                  DataFrame(objective=label, cp=detected_cp))

        # Save params
        param_labs = parameter_labels(n_cps)
        if length(params) < length(param_labs)
            params = [params; fill(NaN, length(param_labs) - length(params))]
        end
        CSV.write(joinpath(OUT_DIR, "covid_params_$(label).csv"),
                  DataFrame(parameter=param_labs, value=params[1:length(param_labs)]))

        result["n_cps"] = n_cps
        result["cps"] = detected_cp
        result["time_seconds"] = elapsed
        result["raw_loss"] = raw_loss
        result["final_loss"] = final_loss
        result["seed"] = SEED
        result["ga_settings"] = Dict(
            "populationSize" => ga_pop,
            "iterations" => ga_iter,
            "min_length" => min_length,
            "step" => step
        )

        println("  #CPs = $(n_cps), CPs = $(detected_cp), time = $(round(elapsed,digits=1))s, final_loss = $(final_loss)")
    catch e
        bt = catch_backtrace()
        err_msg = sprint(showerror, e, bt)
        result["error"] = err_msg
        result["time_seconds"] = time() - t0
        @error "  $(label) failed: $(err_msg)"
    end

    open(joinpath(OUT_DIR, "summary.json"), "w") do f
        JSON.print(f, result, 2)
    end
    flush(stdout)
    return result
end

# ---------- main ----------
function main()
    out_suffix = length(ARGS) >= 1 ? ARGS[1] : "default"
    single_label = length(ARGS) >= 2 ? ARGS[2] : nothing
    mkpath(OUT_ROOT)

    winners = [
        ("zero_penalty", nothing, "log", "zero_penalty"),
    ]

    if single_label !== nothing
        winners = filter(w -> w[1] == single_label, winners)
        if isempty(winners)
            error("Unknown winner label: $single_label")
        end
    end

    println("\n========================================")
    println("Running $(length(winners)) MICA winner(s) on 500-day COVID data")
    println("Output root: $OUT_ROOT")
    println("Workers: $(nworkers())")
    println("========================================")

    results = Dict{String,Any}[]
    for (label, variant, loss, penalty) in winners
        push!(results, run_winner(label, variant, loss, penalty))
    end

    if single_label === nothing
        summary_file = joinpath(OUT_ROOT, "winners_500_summary_$(out_suffix).json")
        open(summary_file, "w") do f
            JSON.print(f, results, 2)
        end
        println("\nAll winners finished. Summary -> $summary_file")
    end
end

main()
