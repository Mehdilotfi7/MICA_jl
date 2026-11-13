#!/usr/bin/env julia
# COVID-19 profile-likelihood example using the BIC winner.
#
# This script demonstrates the full PLE workflow with
# BreakpointProfiles.jl for the COVID-19 SEIRD model.  It is
# intentionally lightweight (small GA population/iterations) so that it can be
# run as a tutorial; increase COVID_PLE_POP and COVID_PLE_ITER for production.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "Mica.jl"))

using Mica
using BreakpointProfiles
using CSV, DataFrames, Statistics, Random
using DifferentialEquations, Smoothers
using Optim, Evolutionary
using LabelledArrays
using Printf

const MICA_DIR = joinpath(@__DIR__, "..", "..", "Mica.jl")
const COVID_DIR = joinpath(MICA_DIR, "examples", "Covid-model")
const WINNER_DIR = joinpath(@__DIR__, "..", "..", "publication", "applications", "Covid", "results")

const OUT_DIR = get(ENV, "COVID_PLE_OUT_DIR", joinpath(@__DIR__, "covid_ple_output"))
mkpath(OUT_DIR)

const PLE_NPOINTS = parse(Int, get(ENV, "COVID_PLE_NPOINTS", "10"))
const PLE_ITER = parse(Int, get(ENV, "COVID_PLE_ITER", "50"))
const PLE_POP = parse(Int, get(ENV, "COVID_PLE_POP", "40"))
const CHISQ_95 = 3.8414588206941285

# --- SEIRD model with seasonality and vaccination ----------------------------

fδ(t::Number, δ::Number, t₀::Number=0.0) = 1 + δ * cos(2π * ((t - t₀) / 365))

log_transform(data; threshold=1.0) = [v >= threshold ? log(v) : log(threshold) for v in data]

function CovModel!(du, u, p, t)
    (ᴺS, ᴺE₀, ᴺE₁, ᴺI₀, ᴺI₁, ᴺI₂, ᴺI₃, ᴺR, D, Cases, V) = u
    N = ᴺS + ᴺE₀ + ᴺE₁ + ᴺI₀ + ᴺI₁ + ᴺI₂ + ᴺI₃ + ᴺR + D

    ᴺε₀  = p.ᴺε₀; ᴺε₁ = p.ᴺε₁; ᴺγ₀ = p.ᴺγ₀; ᴺγ₁ = p.ᴺγ₁
    ᴺγ₂  = p.ᴺγ₂; ᴺγ₃ = p.ᴺγ₃; ᴺp₁ = p.ᴺp₁; ᴺp₁₂ = p.ᴺp₁₂
    ᴺp₂₃ = p.ᴺp₂₃; ᴺp₁D = p.ᴺp₁D; ᴺp₂D = p.ᴺp₂D; ᴺp₃D = p.ᴺp₃D
    δ    = p.δ; ᴺβ = p.ᴺβ; ω = p.ω
    ν    = t < 330 ? 0.0 : p.ν

    ᴺβᴺSI = ᴺβ * fδ(t, δ) * ᴺS * (ᴺE₁ + ᴺI₀ + ᴺI₁)

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

function example_ode_model(params, tspan::Tuple{Float64,Float64}, u0)
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-6, reltol=1.0e-6,
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    return SciMLBase.successful_retcode(sol) ? sol[:, :] : fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
end

function load_covid_data(n_days=400)
    cases_CP = CSV.read(joinpath(COVID_DIR, "case_rki_daily.csv"), DataFrame).total
    hospital_CP = CSV.read(joinpath(COVID_DIR, "Hospitalization_rki_daily.csv"), DataFrame).total
    death_CP = cumsum(CSV.read(joinpath(COVID_DIR, "death_rki_daily.csv"), DataFrame).Todesfaelle_neu)
    icu_CP = CSV.read(joinpath(COVID_DIR, "icu_rki_daily.csv"), DataFrame).total
    vacc_CP = cumsum(CSV.read(joinpath(COVID_DIR, "vaccination_rki_daily_allShots.csv"), DataFrame).Total)

    data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
    max_length = maximum(length, data_CP)
    data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
    data_CP = [vector[1:n_days] for vector in data_CP]
    data_CP[1] = hma(data_CP[1], 14)
    data_CP[4] = hma(data_CP[4], 14)
    data_CP[5] = hma(data_CP[5], 14)
    return Matrix(reduce(hcat, data_CP)')
end

# --- Parameter setup ---------------------------------------------------------

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

function loss_abs_log(obs, sim)
    total = 0.0
    for (k, r) in enumerate(data_indices)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(obs[k, :]))
    end
    return total
end

function parameter_labels(n_cps)
    labs = String[]
    for i in 1:n_global
        push!(labs, string(parnames[i]) * "_global")
    end
    for s in 1:(n_cps+1)
        for i in 1:n_segment_specific
            push!(labs, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    return labs
end

# --- Load BIC winner ---------------------------------------------------------

const cps_best = sort(unique(Int.(CSV.read(joinpath(WINNER_DIR, "covid_detected_cps_bic.csv"), DataFrame).cp)))
const params_best = Float64.(CSV.read(joinpath(WINNER_DIR, "covid_params_bic.csv"), DataFrame).value)
const param_labels = parameter_labels(length(cps_best))[1:length(params_best)]

const n_segments = length(cps_best) + 1
const n_params = n_global + n_segments * n_segment_specific
const lb_full = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
const ub_full = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]

const data_CP = load_covid_data(400)
const n_days = size(data_CP, 2)

const ode_spec = ODEModelSpec(example_ode_model, params_best, u0_full, (0.0, Float64(n_days)))
const model_manager = ModelManager(ode_spec)

function mica_objective(params, cps)
    return Mica.objective_function(
        params, cps, parnames, n_global, n_segment_specific,
        model_manager, loss_abs_log, data_CP
    )
end

const best_loss = mica_objective(params_best, cps_best)
println("BIC winner CPs: $cps_best")
println("Best loss (MICA objective) = $(round(best_loss, digits=4))")

const prob = ODEChangepointPLEProblem(
    objective = mica_objective,
    data = data_CP,
    loss_fn = loss_abs_log,
    changepoints = cps_best,
    best_params = params_best,
    best_loss = best_loss,
    lb = lb_full,
    ub = ub_full,
    param_names = param_labels,
    n_global = n_global,
    n_segment_specific = n_segment_specific,
    n_obs = n_days
)

const optimizer = EvolutionaryPLEConfig(
    population_size=PLE_POP,
    iterations=PLE_ITER,
    seed=1234
)

# --- Selection of parameters to profile --------------------------------------

function selected_indices(selection::String)
    sel = strip(lowercase(selection))
    if sel == "all"
        return collect(1:n_params)
    elseif sel == "global"
        return collect(1:n_global)
    else
        wanted = strip.(split(selection, ","))
        idx = Int[]
        for w in wanted
            pos = findfirst(==(w), param_labels)
            pos === nothing && error("Unknown parameter label: $w")
            push!(idx, pos)
        end
        return idx
    end
end

# --- Run PLE -----------------------------------------------------------------

function main()
    selection = get(ENV, "COVID_PLE_SELECTION", "global")
    idxs = selected_indices(selection)
    println("Profiling $(length(idxs)) parameter(s) out of $n_params (selection = '$selection')")
    println("PLE grid: $PLE_NPOINTS points, GA population $PLE_POP × $PLE_ITER iterations")

    profiles = Vector{ProfileResult}(undef, length(idxs))
    Base.Threads.@threads for k in 1:length(idxs)
        idx = idxs[k]
        println("  [thread $(Threads.threadid())] profiling $(param_labels[idx])")
        profiles[k] = profile_parameter(prob, idx;
                                        n_points=PLE_NPOINTS,
                                        method=:adaptive,
                                        optimizer=optimizer)
        println("  [thread $(Threads.threadid())] $(param_labels[idx]) CI = " *
                "$(profiles[k].ci_lower) - $(profiles[k].ci_upper), " *
                "identifiable=$(profiles[k].identifiable)")
    end

    sort!(profiles, by=p -> p.index)

    write_profiles(joinpath(OUT_DIR, "ple_results.csv"), profiles)
    summary_df = ple_summary(profiles)
    CSV.write(joinpath(OUT_DIR, "ple_summary.csv"), summary_df)

    println("\nParameter profile summary:")
    print_summary_table(summary_df)

    # --- Changepoint profile -------------------------------------------------
    println("\nProfiling changepoint location...")
    cp_profiles = profile_all_changepoints(prob; window=7, optimizer=optimizer)
    write_cp_profiles(joinpath(OUT_DIR, "cp_profile_loss.csv"), cp_profiles)
    cp_df = cp_summary(cp_profiles, best_loss + CHISQ_95)
    CSV.write(joinpath(OUT_DIR, "cp_profile_ci.csv"), cp_df)
    println("CP profile summary:")
    println(cp_df)

    # --- Markdown report -----------------------------------------------------
    lines = String[]
    push!(lines, "# Profile-likelihood analysis for COVID-19 BIC winner")
    push!(lines, "")
    push!(lines, "**Parameter selection:** $(selection)")
    push!(lines, "**Original CPs:** $(cps_best)")
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
    push!(lines, "- `cp_profile_loss.csv` — changepoint profile losses")
    push!(lines, "- `cp_profile_ci.csv` — changepoint confidence intervals")

    open(joinpath(OUT_DIR, "ple_report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end

    println("\nDone. Outputs written to $OUT_DIR")
end

main()
