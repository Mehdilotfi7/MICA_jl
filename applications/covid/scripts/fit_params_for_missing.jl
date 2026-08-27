#!/usr/bin/env julia
# Fit parameters for every change-point file that does not yet have a matching
# parameter file, then leave PNG generation to plot_covid_objectives_origset.jl.
#
# The change-point files are kept unchanged; only covid_params_origset_*.csv
# (and, for the global seed sweep, normalized covid_detected_cps_origset_*.csv)
# are created.

using Pkg
mica_project = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
Pkg.activate(mica_project)

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Random, JSON

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")

# ---------- model definitions (must match the detection scripts) ----------
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
const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                   :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
const initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9, 1/16, 0.0055,
                            0.2, 0.05, 0.17, 0.144, 0.01, 0.017, 0.173, 0.01]
const lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
               0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
const upper = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
               0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]
const bounds0 = (lower, upper)
const u0 = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const tspan = (0.0, 399.0)

const ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
const model_manager = ModelManager(ode_spec)

const n_global = 8
const n_segment_specific = 8
const n = size(data_CP, 2)

# ---------- helpers ----------
function make_optimizer(pop_size::Int, seed::Union{Int,Nothing})
    ga = GA(populationSize=pop_size, selection=tournament(2), crossover=SBX(0.7, 1),
            mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
    return EvolutionaryOptimizer(ga,
                                 Evolutionary.Options(show_trace=false, iterations=1000),
                                 seed)
end

function build_chromosome_bounds(n_segments::Int)
    global_init = initial_chromosome[1:n_global]
    seg_init    = initial_chromosome[n_global+1:end]
    lower_global = lower[1:n_global]
    lower_seg    = lower[n_global+1:end]
    upper_global = upper[1:n_global]
    upper_seg    = upper[n_global+1:end]

    chromosome = [global_init; repeat(seg_init, n_segments)]
    lb = [lower_global; repeat(lower_seg, n_segments)]
    ub = [upper_global; repeat(upper_seg, n_segments)]
    return chromosome, (lb, ub)
end

function load_cps_robust(path)
    df = CSV.read(path, DataFrame)
    for col in [:cp, :detected_cp, :change_point]
        if hasproperty(df, col)
            return sort(unique(Int.(df[:, col])))
        end
    end
    error("No recognized cp column in $path")
end

function infer_population(dir)
    name = basename(dir)
    m = match(r"pop(\d+)", name)
    m !== nothing && return parse(Int, m.captures[1])
    # Try reading the first summary JSON that contains a population key
    for fname in readdir(dir)
        if endswith(fname, ".json")
            try
                d = JSON.parsefile(joinpath(dir, fname))
                if haskey(d, "population")
                    return d["population"]
                end
            catch
            end
        end
    end
    return 150
end

function fit_params_for_cps(cps, pop_size, seed)
    n_segments = length(cps) + 1
    chromosome, bounds = build_chromosome_bounds(n_segments)
    opt = make_optimizer(pop_size, seed)
    loss, params = optimize_with_changepoints(
        Mica.objective_function, chromosome, parnames, cps, bounds, opt,
        n_global, n_segment_specific, model_manager, loss_function, data_CP
    )
    return loss, params
end

function process_directory(dir)
    println("\n============================================================")
    println("Processing: $dir")
    println("============================================================")
    pop_size = infer_population(dir)
    fit_seed = 1234  # fixed seed for the parameter-refit step

    # 1) Standard covid_detected_cps_origset_<obj>.csv files
    for fname in readdir(dir)
        m = match(r"^covid_detected_cps_origset_(.*)\.csv$", fname)
        m === nothing && continue
        label = m.captures[1]
        params_file = joinpath(dir, "covid_params_origset_$(label).csv")
        isfile(params_file) && continue
        cps = load_cps_robust(joinpath(dir, fname))
        println("  Fitting params for objective='$label' (#CPs=$(length(cps)), pop=$pop_size, seed=$fit_seed)")
        loss, params = fit_params_for_cps(cps, pop_size, fit_seed)
        CSV.write(params_file, DataFrame(params=params))
        println("    loss=$(round(loss,digits=2)), saved $params_file")
    end

    # 2) Global-seed sweep files: cps_seed<seed>.csv
    for fname in readdir(dir)
        m = match(r"^cps_seed(\d+)\.csv$", fname)
        m === nothing && continue
        seed_val = m.captures[1]
        label = "none_seed$(seed_val)"
        cps_out = joinpath(dir, "covid_detected_cps_origset_$(label).csv")
        params_file = joinpath(dir, "covid_params_origset_$(label).csv")
        isfile(params_file) && continue
        cps = load_cps_robust(joinpath(dir, fname))
        CSV.write(cps_out, DataFrame(objective="none", cp=cps))
        println("  Fitting params for seed=$seed_val (#CPs=$(length(cps)), pop=$pop_size, seed=$fit_seed)")
        loss, params = fit_params_for_cps(cps, pop_size, fit_seed)
        CSV.write(params_file, DataFrame(params=params))
        println("    loss=$(round(loss,digits=2)), saved $params_file")
    end
end

# ---------- main loop over all results* directories ----------
const APP_DIR = joinpath(@__DIR__, "..")
for dir in readdir(APP_DIR, join=true)
    isdir(dir) || continue
    name = basename(dir)
    startswith(name, "results") || continue
    process_directory(dir)
end

println("\nParameter fitting complete. Run generate_all_plots.sh to generate PNGs.")
