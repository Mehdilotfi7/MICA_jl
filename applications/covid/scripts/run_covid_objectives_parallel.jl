#!/usr/bin/env julia
# ============================================================
# Parallel zero-penalty COVID run.
# Uses Distributed workers to evaluate candidate change points
# in parallel inside detect_changepoints(...; parallel=true).
#
# Usage:
#   julia run_covid_objectives_parallel.jl [seed] [out_suffix] [populationSize] [global_seed]
#
# Defaults: seed = 1234, out_suffix = "pop150_parallel", populationSize = 150, global_seed = 1234
# Use seed = -1 for no per-optimization seed. In that case global_seed controls reproducibility.
# ============================================================

using Distributed, Pkg

# Activate the Mica.jl project on the main process (workers will use the same project)
mica_project = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
Pkg.activate(mica_project)
proj_file = Base.active_project()

# Use half of the available cores (leave some for the OS / main process)
nworkers_target = max(1, div(Sys.CPU_THREADS, 2))
addprocs(nworkers_target; exeflags="--project=$proj_file")
println("Started $(nworkers()) worker(s).")

@everywhere begin
    using Mica
    using CSV, DataFrames, Statistics
    using Evolutionary, OrdinaryDiffEq
    using Smoothers
    using Random
    using JSON

    const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")

    # ---------- model definitions ----------
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

    # Penalty function used for the zero-penalty (none) run
    zero_penalty(p, n) = 0.0
end

# ---------- run one objective ----------
seed_arg = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1234
seed = seed_arg == -1 ? nothing : seed_arg
out_suffix = length(ARGS) >= 2 ? ARGS[2] : "pop150_parallel"
pop_size = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 150
global_seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1234
OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results_$(out_suffix)")
mkpath(OUT_DIR)

# GA settings (original example, populationSize configurable)
ga = GA(populationSize=pop_size, selection=tournament(2), crossover=SBX(0.7, 1),
        mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
opt = EvolutionaryOptimizer(ga,
        options=Evolutionary.Options(show_trace=false, iterations=1000), seed=seed)

println("\nRunning COVID objective = none (parallel=true, per_opt_seed=$(repr(seed)), global_seed=$global_seed)")
Random.seed!(global_seed)
@everywhere Random.seed!($global_seed)
t0 = time()
detected_cp, params = detect_changepoints(
    objective_function, n, n_global, n_segment_specific,
    model_manager, loss_function, data_CP,
    copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
    opt,
    min_length, step;
    objective_type=:penalty,
    penalty_fn=zero_penalty,
    data_indices=data_indices,
    verbose=false, animate=false,
    parallel=true
)
elapsed = time() - t0

# No boundary filter (matches original example)
detected_cp = sort(unique(detected_cp))

label = "none"
CSV.write(joinpath(OUT_DIR, "covid_detected_cps_origset_$(label).csv"),
          DataFrame(objective=label, cp=detected_cp))

summary_file = joinpath(OUT_DIR, "covid_objectives_origset_summary.json")
summary = Dict{String,Any}(
    "objective" => label,
    "n_cps" => length(detected_cp),
    "cps" => detected_cp,
    "time_seconds" => elapsed,
    "seed" => seed,
    "parallel_workers" => nworkers()
)
open(summary_file, "w") do f
    JSON.print(f, summary, 2)
end
println("  #CPs = $(length(detected_cp)), CPs = $detected_cp, time = $(round(elapsed,digits=1))s")
println("Summary saved to $summary_file")
