#!/usr/bin/env julia
# Fast CP profile only — re-optimizes parameters while perturbing each CP.
# Reads bootstrap parameter CIs from TASK_D and writes CP profile + combined report.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics, Dates
using OrdinaryDiffEq, Smoothers
using Optim
using LabelledArrays
using Printf

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const TASK_A_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A", "results_penalty_zero")
const TASK_D_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_D")
mkpath(TASK_D_DIR)

const BASE_DATE = Date("2020-01-27")
const CP_WINDOW = 3
const N_PARAMS_LBFGS = 100

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

const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                   :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
const n_global = 8
const n_segment_specific = 8
const u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const data_indices = [5, 6, 7, 9, 11]
const n = size(data_CP, 2)
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

const ode_spec = ODEModelSpec(example_ode_model, zeros(n_global + n_segment_specific), u0_full, (0.0, Float64(n)))
const model_manager = ModelManager(ode_spec)

cps_best = sort(unique(Int.(CSV.read(joinpath(TASK_A_DIR, "covid_detected_cps_origset_penalty_zero.csv"), DataFrame).cp)))
params_best = Float64.(CSV.read(joinpath(TASK_A_DIR, "covid_params_origset_penalty_zero.csv"), DataFrame).value)

function refit(cps, x0)
    n_segments = length(cps) + 1
    n_params = n_global + n_segments * n_segment_specific
    lb = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
    ub = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]
    x0_use = x0[1:n_params]
    opt = OptimOptimizer(Fminbox(LBFGS()),
                         options=Optim.Options(show_trace=false, iterations=N_PARAMS_LBFGS))
    loss, params = optimize_with_changepoints(
        Mica.objective_function, x0_use, parnames, cps, (lb, ub), opt,
        n_global, n_segment_specific, model_manager, loss_abs_log, data_CP
    )
    return loss, params
end

function main()
    best_loss, _ = refit(cps_best, params_best)
    println("[Task D profile] Best loss = $(round(best_loss, digits=2))")

    rows = DataFrame(cp_index=Int[], original_cp=Int[], candidate_cp=Int[],
                     date=Date[], loss=Float64[], delta_loss=Float64[])
    for (j, cp) in enumerate(cps_best)
        others = filter(!=(cp), cps_best)
        for delta in (-CP_WINDOW):CP_WINDOW
            cand = cp + delta
            if cand < 11 || cand > n - 10 || cand in others
                continue
            end
            cps_try = sort([others; cand])
            try
                l, _ = refit(cps_try, params_best)
                push!(rows, (j, cp, cand, BASE_DATE + Day(cand - 1), l, l - best_loss))
            catch e
                @warn "CP $j cand $cand failed: $e"
            end
        end
        println("  CP $j done"); flush(stdout)
    end
    CSV.write(joinpath(TASK_D_DIR, "cp_profile_loss.csv"), rows)

    ci = DataFrame(cp_index=Int[], original_cp=Int[], original_date=Date[],
                   ci_lower=Union{Int,Missing}[], ci_upper=Union{Int,Missing}[],
                   ci_lower_date=Union{Date,Missing}[], ci_upper_date=Union{Date,Missing}[])
    for j in 1:length(cps_best)
        prof = filter(r -> r.cp_index == j, rows)
        subset = filter(r -> r.delta_loss <= 3.84, prof)
        if nrow(subset) > 0
            l, u = extrema(subset.candidate_cp)
            push!(ci, (j, cps_best[j], BASE_DATE + Day(cps_best[j] - 1),
                       l, u, BASE_DATE + Day(l - 1), BASE_DATE + Day(u - 1)))
        else
            push!(ci, (j, cps_best[j], BASE_DATE + Day(cps_best[j] - 1),
                       missing, missing, missing, missing))
        end
    end
    CSV.write(joinpath(TASK_D_DIR, "cp_profile_ci.csv"), ci)
    println("[Task D profile] Saved cp_profile_loss.csv and cp_profile_ci.csv")
end

main()
