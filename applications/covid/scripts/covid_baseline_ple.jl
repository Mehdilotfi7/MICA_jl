#!/usr/bin/env julia
# PLE for the 16-parameter baseline SEIRD model (no changepoints).
# Usage: julia covid_baseline_ple.jl

using Pkg
const _BASELINE_MICA_PROJECT = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
const _BASELINE_EXPECTED_PROJECT = abspath(joinpath(_BASELINE_MICA_PROJECT, "Project.toml"))
if Base.active_project() === nothing || abspath(Base.active_project()) != _BASELINE_EXPECTED_PROJECT
    Pkg.activate(_BASELINE_MICA_PROJECT)
end
# Make BreakpointProfiles available from the sibling package directory
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "codes", "BreakpointProfiles.jl"))
using Mica
using BreakpointProfiles
using CSV, DataFrames, Statistics, Random
using OrdinaryDiffEq, Smoothers
using Optim, Evolutionary
using LabelledArrays
using Printf
using Dates

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_PLE", "baseline_16param")
mkpath(OUT_DIR)

const PLE_NPOINTS = parse(Int, get(ENV, "COVID_PLE_NPOINTS", "20"))
const PLE_ITER = parse(Int, get(ENV, "COVID_PLE_ITER", "100"))
const PLE_POP = parse(Int, get(ENV, "COVID_PLE_POP", "80"))
const CHISQ_95 = 3.8414588206941285

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
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u), maxiters=100_000)
    if SciMLBase.successful_retcode(sol)
        return sol[:, :]
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

function load_covid_data(n_days=400)
    cases_CP = CSV.read(joinpath(EXAMPLE_DIR, "case_rki_daily.csv"), DataFrame).total
    hospital_CP = CSV.read(joinpath(EXAMPLE_DIR, "Hospitalization_rki_daily.csv"), DataFrame).total
    death_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "death_rki_daily.csv"), DataFrame).Todesfaelle_neu)
    icu_CP = CSV.read(joinpath(EXAMPLE_DIR, "icu_rki_daily.csv"), DataFrame).total
    vacc_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "vaccination_rki_daily_allShots.csv"), DataFrame).Total)

    data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
    max_length = maximum(length, data_CP)
    data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
    data_CP = [vector[1:n_days] for vector in data_CP]
    data_CP[1] = hma(data_CP[1], 14)
    data_CP[4] = hma(data_CP[4], 14)
    data_CP[5] = hma(data_CP[5], 14)
    return Matrix(reduce(hcat, data_CP)')
end

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

const data_CP = load_covid_data(400)
const n = size(data_CP, 2)

# Pre-compute log-transformed observed data to avoid repeated allocations in the loss.
const _obs_log = [let row = data_CP[k, :]; [v >= 1 ? log(v) : 0.0 for v in row] end for k in 1:size(data_CP, 1)]

const ode_spec = ODEModelSpec(example_ode_model, zeros(n_global + n_segment_specific), u0_full, (0.0, Float64(n)))
const model_manager = ModelManager(ode_spec)

function loss_abs_log(obs, sim)
    total = 0.0
    for (k, r) in enumerate(data_indices)
        sim_row = @view sim[r, :]
        obs_log_row = _obs_log[k]
        for j in 1:length(sim_row)
            s = sim_row[j]
            s_log = s >= 1 ? log(s) : 0.0
            total += abs(s_log - obs_log_row[j])
        end
    end
    return total
end

function mica_objective(params, cps)
    return Mica.objective_function(
        params, cps, parnames, n_global, n_segment_specific,
        model_manager, loss_abs_log, data_CP
    )
end

# Starting point: try to load a fitted baseline parameter file; otherwise use
# the global MICA values as a reasonable initial guess.
function load_or_guess_baseline_params()
    param_file = joinpath(@__DIR__, "..", "..", "..", "d2d_base", "covid_base_params.csv")
    if isfile(param_file)
        df = CSV.read(param_file, DataFrame)
        return Float64.(df.value)
    end
    # Reasonable default from MICA global + segment1 values
    return Float64[
        0.10486, 0.13677, 0.08849, 0.07760, 0.09164, 0.05263, 0.04251, 0.00713,
        0.04262, 0.89653, 0.48642, 0.49814, 0.00746, 0.07103, 0.19421, 0.00589
    ]
end

const baseline_params = load_or_guess_baseline_params()
const baseline_cps = Int[]

const best_loss = mica_objective(baseline_params, baseline_cps)
println("[baseline] Best loss (MICA objective) = $(round(best_loss, digits=4))"); flush(stdout)

const param_labels = [string(s) for s in parnames]
const prob = ODEChangepointPLEProblem(
    objective = mica_objective,
    data = data_CP,
    loss_fn = loss_abs_log,
    changepoints = baseline_cps,
    best_params = baseline_params,
    best_loss = best_loss,
    lb = lower,
    ub = upper,
    param_names = param_labels,
    n_global = n_global,
    n_segment_specific = n_segment_specific,
    n_obs = n
)

const optimizer = EvolutionaryPLEConfig(population_size=PLE_POP, iterations=PLE_ITER, seed=1234, selection="sus")

function main()
    println("[baseline] PLE for 16 parameters, no changepoints"); flush(stdout)
    println("[baseline] Best loss = $(round(best_loss, digits=4)); threshold = $(round(best_loss + CHISQ_95, digits=4))"); flush(stdout)
    println("[baseline] Using population=$(PLE_POP), iterations=$(PLE_ITER), n_points=$(PLE_NPOINTS) with $(Threads.nthreads()) thread(s)"); flush(stdout)

    n = length(prob.best_params)
    profiles = Vector{ProfileResult}(undef, n)
    Base.Threads.@threads for i in 1:n
        println("[baseline] [thread $(Threads.threadid())] profiling $(param_labels[i])"); flush(stdout)
        profiles[i] = profile_parameter(prob, i;
                                       n_points=PLE_NPOINTS,
                                       method=:adaptive,
                                       optimizer=optimizer)
        println("[baseline] [thread $(Threads.threadid())] $(param_labels[i]) CI = $(profiles[i].ci_lower) - $(profiles[i].ci_upper), identifiable=$(profiles[i].identifiable)"); flush(stdout)
    end

    sort!(profiles, by=p -> p.index)
    println("[baseline] writing profile results..."); flush(stdout)
    write_profiles(joinpath(OUT_DIR, "ple_results.csv"), profiles)
    println("[baseline] writing profile summary..."); flush(stdout)
    summary_df = ple_summary(profiles)
    CSV.write(joinpath(OUT_DIR, "ple_summary.csv"), summary_df)

    println("[baseline] writing Markdown report..."); flush(stdout)
    lines = String[]
    push!(lines, "# Baseline SEIRD profile-likelihood analysis")
    push!(lines, "")
    push!(lines, "**Model:** 16-parameter SEIRD, no changepoints")
    push!(lines, "**Best loss:** $(round(best_loss, digits=4))")
    push!(lines, "**Threshold (Δloss ≤ $(CHISQ_95)):** $(round(best_loss + CHISQ_95, digits=4))")
    push!(lines, "")
    push!(lines, "## Summary")
    push!(lines, "")
    push!(lines, "| parameter | best value | CI lower | CI upper | identifiable |")
    push!(lines, "|---|---|---|---|---|")
    for r in eachrow(summary_df)
        push!(lines, @sprintf("| %s | %.5g | %.5g | %.5g | %s |",
                              r.parameter, r.best_value, r.ci_lower, r.ci_upper, r.identifiable))
    end
    push!(lines, "")
    push!(lines, "## Files")
    push!(lines, "- `ple_results.csv` — full profile curves")
    push!(lines, "- `ple_summary.csv` — approximate 95% confidence intervals")
    open(joinpath(OUT_DIR, "ple_report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end

    println("[baseline] PLE done. Outputs in $OUT_DIR"); flush(stdout)
    exit(0)
end

main()
