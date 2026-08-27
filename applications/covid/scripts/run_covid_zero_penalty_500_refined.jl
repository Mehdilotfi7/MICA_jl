#!/usr/bin/env julia
# ============================================================
# Refined zero-penalty MICA on the 500-day COVID window.
#
# Changes vs. the original run:
#   - step = 5  (finer CP grid, so wave onsets around day 320 can be captured)
#   - GA mutation scale 1e-3 (instead of 1e-4) to escape local basins
#   - uniformranking selection + larger population/iterations
#   - final refit with Fminbox(LBFGS) starting from the GA solution
#
# Usage:
#   WORKERS=8 julia run_covid_zero_penalty_500_refined.jl
# ============================================================

using Distributed, Pkg

mica_project = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
Pkg.activate(mica_project)
proj_file = Base.active_project()
proj_dir  = dirname(proj_file)

nworkers_target = parse(Int, get(ENV, "WORKERS", "8"))
if nworkers() < nworkers_target
    addprocs(nworkers_target - nworkers(); exeflags="--project=$proj_dir")
end
println("Using $(nworkers()) worker(s).")

const MICA_HELPER = joinpath(@__DIR__, "..", "..", "..", "load_mica_helper.jl")
using Mica
for w in workers()
    remotecall_fetch(include, w, MICA_HELPER)
end

@everywhere begin
    using CSV, DataFrames, Statistics
    using Evolutionary, OrdinaryDiffEq, Optim
    using Smoothers
    using Random
    using JSON

    const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
    const OUT_ROOT    = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_G_500")

    function fδ(t::Number, δ::Number, t₀::Number=0.0)
        return 1 + δ * cos(2 * π * ((t - t₀) / 365))
    end

    log_transform(data, threshold=1) = [val >= threshold ? log(val) : 0 for val in data]

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
    step = 5
    data_indices = [5, 6, 7, 9, 11]
    n = size(data_CP, 2)

    const SEED = 1234

    # More exploratory GA: larger mutation, uniformranking selection
    ga = GA(populationSize=200, selection=uniformranking(20),
            crossover=SBX(0.7, 1), mutationRate=0.7, crossoverRate=0.7,
            mutation=gaussian(0.001))
    opt = EvolutionaryOptimizer(ga,
            options=Evolutionary.Options(show_trace=false, iterations=1500), seed=SEED)

    zero_penalty(p, n) = 0.0

    function bounds_for_cps(lower, upper, n_global, n_segment_specific, n_cps)
        n_seg = n_cps + 1
        lb = copy(lower[1:n_global])
        ub = copy(upper[1:n_global])
        seg_lb = lower[n_global+1:end]
        seg_ub = upper[n_global+1:end]
        for _ in 1:n_seg
            append!(lb, seg_lb)
            append!(ub, seg_ub)
        end
        return lb, ub
    end

    function equal_log_loss(sim)
        rows = data_indices
        total = 0.0
        for (k, r) in enumerate(rows)
            total += sum(abs, log_transform(sim[r, :]) .- log_transform(data_CP[k, :]))
        end
        return total
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

# ---------- run ----------
function main()
    OUT_DIR = joinpath(OUT_ROOT, "winners_500_zero_penalty_refined")
    mkpath(OUT_DIR)

    println("\n[winners 500] zero_penalty_refined")
    println("  step=$step, min_length=$min_length, pop=$(ga.populationSize), iter=$(opt.options.iterations), mutation=0.001")
    flush(stdout)

    t0 = time()
    Random.seed!(SEED)
    detected_cp, params = detect_changepoints(
        Mica.objective_function, n, n_global, n_segment_specific,
        model_manager, equal_log_loss, data_CP,
        copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
        opt,
        min_length, step;
        objective_type=:penalty,
        penalty_fn=zero_penalty,
        data_indices=data_indices,
        verbose=false, animate=false,
        parallel=true
    )
    detected_cp = sort(unique(detected_cp))
    n_cps = length(detected_cp)

    # Final local refinement with Fminbox(LBFGS) from the GA solution
    try
        println("  Final local refinement...")
        lb, ub = bounds_for_cps(bounds[1], bounds[2], n_global, n_segment_specific, n_cps)
        refined_loss, refined_params = Mica.optimize_with_changepoints(
            Mica.objective_function, params, parnames, detected_cp, (lb, ub),
            OptimOptimizer(Fminbox(LBFGS()); options=Optim.Options(show_trace=false, iterations=500), seed=SEED),
            n_global, n_segment_specific, model_manager, equal_log_loss, data_CP)
        current_loss = Mica.objective_function(params, detected_cp, parnames, n_global, n_segment_specific, model_manager, equal_log_loss, data_CP)
        if refined_loss < current_loss
            params = refined_params
            println("  Refinement improved loss.")
        else
            println("  Refinement did not improve; keeping GA params.")
        end
    catch e
        println("  Refinement skipped due to error: $e")
    end

    elapsed = time() - t0
    raw_loss = Mica.objective_function(params, detected_cp, parnames, n_global, n_segment_specific, model_manager, equal_log_loss, data_CP)
    segment_lengths = diff([0; detected_cp; n])
    final_loss = Mica.compute_objective(raw_loss, n, n_global, n_segment_specific,
                                        n_cps, :penalty, zero_penalty,
                                        detected_cp, segment_lengths)

    CSV.write(joinpath(OUT_DIR, "covid_detected_cps_zero_penalty_refined.csv"),
              DataFrame(objective="zero_penalty_refined", cp=detected_cp))

    param_labs = parameter_labels(n_cps)
    if length(params) < length(param_labs)
        params = [params; fill(NaN, length(param_labs) - length(params))]
    end
    CSV.write(joinpath(OUT_DIR, "covid_params_zero_penalty_refined.csv"),
              DataFrame(parameter=param_labs, value=params[1:length(param_labs)]))

    result = Dict(
        "label" => "zero_penalty_refined",
        "n_cps" => n_cps,
        "cps" => detected_cp,
        "time_seconds" => elapsed,
        "raw_loss" => raw_loss,
        "final_loss" => final_loss,
        "seed" => SEED,
        "ga_settings" => Dict(
            "populationSize" => 200,
            "iterations" => 1500,
            "min_length" => min_length,
            "step" => step,
            "mutation" => 0.001,
            "selection" => "uniformranking(20)"
        )
    )
    open(joinpath(OUT_DIR, "summary.json"), "w") do f
        JSON.print(f, result, 2)
    end
    println("  #CPs = $(n_cps), CPs = $(detected_cp), time = $(round(elapsed,digits=1))s, raw_loss = $(raw_loss)")
    println("  Output -> $OUT_DIR")
end

main()
