#!/usr/bin/env julia
# TASK_RANDOM — Random changepoint baseline for the COVID-19 example.
#
# For each n_cps in 2:8, generate one random set of n_cps changepoints on a
# 10-day grid (deterministic seed per n_cps), refit the SEIRD model with those
# CPs fixed, and save the fitted parameters, the simulation, and the loss.
# The refit is performed with the same GA settings used for the COVID example.
#
# Usage:
#     julia run_random_cp_baseline.jl [out_dir]
#
# Output (in out_dir):
#   - random_cps_<n>.csv            : random CP locations
#   - random_params_<n>.csv         : fitted parameters
#   - random_simulation_<n>.csv      : model simulation (same 5 channels as data)
#   - summary.json                   : losses and CP sets for all n_cps
#
# A MICA reference (BIC winner, 2 CPs at 60,150) is also refit and saved
# for comparison.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Random
using JSON
using Dates
using LabelledArrays

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_DIR = length(ARGS) >= 1 ? ARGS[1] : "outputs/TASK_RANDOM"
mkpath(OUT_DIR)

const BASE_DATE = Date("2020-01-27")
const CP_GRID = collect(20:10:380)   # 10-day grid, avoiding the very edges
const GA_POP = parse(Int, get(ENV, "TASK_RANDOM_GA_POP", "150"))
const GA_ITER = parse(Int, get(ENV, "TASK_RANDOM_GA_ITER", "1000"))
const CP_SEED_BASE = parse(Int, get(ENV, "TASK_RANDOM_CP_SEED_BASE", "10000"))
const GA_SEED = parse(Int, get(ENV, "TASK_RANDOM_GA_SEED", "1234"))

const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                   :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
const n_global = 8
const n_segment_specific = 8
const u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const data_indices = [5, 6, 7, 9, 11]

const lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
               0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
const upper = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
               0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]
const global_init = [0.1633, 1/7, 1/11.4, 1/14, 1/13.4, 1/9, 1/16, 0.0055]
const seg_init    = [0.2, 0.05, 0.17, 0.144, 0.01, 0.017, 0.173, 0.01]

# ---------- model / data (same as the COVID example) ----------
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
                maxiters=20000,
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    if SciMLBase.successful_retcode(sol)
        return sol[:, :]
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

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

const n = size(data_CP, 2)

function loss_abs_log(obs, sim)
    total = 0.0
    for (k, r) in enumerate(data_indices)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(obs[k, :]))
    end
    return total
end

function build_x0(n_cps)
    return [global_init; repeat(seg_init, n_cps + 1)]
end
function build_bounds(n_cps)
    lb = [lower[1:n_global]; repeat(lower[n_global+1:end], n_cps + 1)]
    ub = [upper[1:n_global]; repeat(upper[n_global+1:end], n_cps + 1)]
    return (lb, ub)
end

function simulate(cps, params)
    constant = params[1:n_global]
    n_seg = length(cps) + 1
    seg_params = [params[n_global + (s-1)*n_segment_specific + 1 : n_global + s*n_segment_specific] for s in 1:n_seg]
    u0_curr = u0_full
    sim_segments = Matrix{Float64}[]
    for s in 1:n_seg
        idx_start = (s == 1) ? 1 : cps[s-1] + 1
        idx_end   = (s > length(cps)) ? n : cps[s]
        all_pars = @LArray [constant; seg_params[s]] parnames
        tspan_seg = (Float64(idx_start), Float64(idx_end))
        raw_seg = example_ode_model(all_pars, tspan_seg, u0_curr)
        if raw_seg isa SciMLBase.ODESolution
            sim_seg = reduce(hcat, raw_seg.u)
        elseif raw_seg isa AbstractMatrix
            sim_seg = raw_seg
        else
            sim_seg = Matrix(raw_seg)
        end
        push!(sim_segments, sim_seg)
        u0_curr = sim_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end

function refit_fixed_cps(cps; seed=GA_SEED)
    n_cps = length(cps)
    x0 = build_x0(n_cps)
    bounds = build_bounds(n_cps)
    ode_spec = ODEModelSpec(example_ode_model, x0, u0_full, (0.0, Float64(n)))
    model_manager = ModelManager(ode_spec)

    ga = GA(populationSize=GA_POP, selection=tournament(2), crossover=SBX(0.7, 1),
            mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
    opt = EvolutionaryOptimizer(ga,
                                options=Evolutionary.Options(show_trace=false, iterations=GA_ITER),
                                seed=seed)

    loss, params = optimize_with_changepoints(
        Mica.objective_function, x0, parnames, sort(cps), bounds, opt,
        n_global, n_segment_specific, model_manager, loss_abs_log, data_CP
    )
    return loss, params
end

function generate_random_cps(n_cps; seed=CP_SEED_BASE + n_cps)
    rng = MersenneTwister(seed)
    cps = sort(unique(rand(rng, CP_GRID, n_cps)))
    # Ensure we have exactly n_cps distinct CPs; if duplicates, resample.
    while length(cps) < n_cps
        extra = rand(rng, CP_GRID, n_cps - length(cps))
        cps = sort(unique(vcat(cps, extra)))
    end
    return cps
end

function save_simulation(cps, params, label)
    sim = simulate(cps, params)
    days = 1:n
    df = DataFrame(
        day = days,
        date = [BASE_DATE + Day(d - 1) for d in days],
        infected_obs = data_CP[1, :],
        infected_sim = sim[5, :],
        hospitalized_obs = data_CP[2, :],
        hospitalized_sim = sim[6, :],
        icu_obs = data_CP[3, :],
        icu_sim = sim[7, :],
        death_obs = data_CP[4, :],
        death_sim = sim[9, :],
        vaccinated_obs = data_CP[5, :],
        vaccinated_sim = sim[11, :]
    )
    CSV.write(joinpath(OUT_DIR, "random_simulation_$(label).csv"), df)
    return df
end

function run_for_n_cps(n_cps)
    println("[TASK_RANDOM] n_cps = $n_cps")
    cps = generate_random_cps(n_cps)
    println("  random CPs: $cps")
    loss, params = refit_fixed_cps(cps)
    println("  loss = $(round(loss, digits=4))")

    CSV.write(joinpath(OUT_DIR, "random_cps_$(n_cps).csv"), DataFrame(cp=cps))

    par_labels = String[]
    for i in 1:n_global
        push!(par_labels, string(parnames[i]) * "_global")
    end
    for s in 1:(n_cps+1)
        for i in 1:n_segment_specific
            push!(par_labels, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    CSV.write(joinpath(OUT_DIR, "random_params_$(n_cps).csv"),
              DataFrame(parameter=par_labels, value=params))

    save_simulation(cps, params, n_cps)

    return Dict("n_cps" => n_cps,
                "cps" => cps,
                "seed" => CP_SEED_BASE + n_cps,
                "loss" => loss)
end

function main()
    println("[TASK_RANDOM] Running random CP baseline with $(Threads.nthreads()) threads")
    println("[TASK_RANDOM] Output directory: $OUT_DIR")

    results = Vector{Dict}(undef, 7)
    Threads.@threads for n_cps in 2:8
        results[n_cps - 1] = run_for_n_cps(n_cps)
    end

    # Also save a MICA reference (BIC winner, 2 CPs at 60, 150)
    println("[TASK_RANDOM] Fitting MICA BIC reference (CPs = [60,150])")
    mica_cps = [60, 150]
    mica_loss, mica_params = refit_fixed_cps(mica_cps)
    println("  MICA loss = $(round(mica_loss, digits=4))")
    CSV.write(joinpath(OUT_DIR, "mica_bic_cps.csv"), DataFrame(cp=mica_cps))
    CSV.write(joinpath(OUT_DIR, "mica_bic_params.csv"),
              DataFrame(parameter=[
                  [string(p) * "_global" for p in parnames[1:n_global]];
                  vcat([[string(parnames[n_global + i]) * "_seg$s" for i in 1:n_segment_specific] for s in 1:(length(mica_cps)+1)]...)
              ], value=mica_params))
    save_simulation(mica_cps, mica_params, "mica_bic")

    summary = Dict(
        "random_results" => results,
        "mica_reference" => Dict("n_cps" => 2, "cps" => mica_cps, "loss" => mica_loss),
        "settings" => Dict("cp_seed_base" => CP_SEED_BASE, "ga_seed" => GA_SEED,
                           "cp_grid" => CP_GRID, "ga_population" => GA_POP, "ga_iterations" => GA_ITER)
    )
    open(joinpath(OUT_DIR, "summary.json"), "w") do f
        JSON.print(f, summary, 2)
    end

    println("[TASK_RANDOM] Done. Outputs in $OUT_DIR")
end

main()
