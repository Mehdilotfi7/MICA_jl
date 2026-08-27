#!/usr/bin/env julia
# Task F — Post-process detected change points: redundancy test and local relocation.
# Usage: julia covid_postprocess_winner.jl <label> <cps_csv> <params_csv> <out_dir>

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using OrdinaryDiffEq, Smoothers
using Optim
using LabelledArrays
using Printf

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")

# ---------- model / data ----------
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

# ---------- CLI ----------
length(ARGS) >= 4 || error("Usage: julia covid_postprocess_winner.jl <label> <cps_csv> <params_csv> <out_dir>")
const LABEL = ARGS[1]
const CPS_CSV = ARGS[2]
const PARAMS_CSV = ARGS[3]
const OUT_DIR = ARGS[4]
mkpath(OUT_DIR)

# ---------- load selected fit ----------
cps_best = sort(unique(Int.(CSV.read(CPS_CSV, DataFrame).cp)))
params_best = Float64.(CSV.read(PARAMS_CSV, DataFrame).value)

const ode_spec = ODEModelSpec(example_ode_model, params_best, u0_full, (0.0, Float64(n)))
const model_manager = ModelManager(ode_spec)

function refit(cps, x0)
    n_segments = length(cps) + 1
    n_params = n_global + n_segments * n_segment_specific
    lb = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
    ub = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]
    x0_use = x0[1:n_params]

    opt = OptimOptimizer(Fminbox(LBFGS()),
                         options=Optim.Options(show_trace=false, iterations=500))
    loss, params = optimize_with_changepoints(
        Mica.objective_function, x0_use, parnames, cps, (lb, ub), opt,
        n_global, n_segment_specific, model_manager, loss_abs_log, data_CP
    )
    return loss, params
end

function loss_for(cps, x0)
    try
        l, _ = refit(cps, x0)
        return l
    catch e
        @warn "Refit failed for CPs $cps: $e"
        return Inf
    end
end

function main()
    println("[$LABEL] Best-fit CPs: $cps_best")
    best_loss, _ = refit(cps_best, params_best)
    println("[$LABEL] Best loss = $(round(best_loss, digits=2))")

    # 1. Test each CP for redundancy
    redundancy = DataFrame(cp=Int[], loss_without=Float64[], delta_loss=Float64[], relative_increase=Float64[], action=String[])
    for cp in cps_best
        cps_minus = filter(!=(cp), cps_best)
        l_minus = loss_for(cps_minus, params_best)
        delta = l_minus - best_loss
        rel = delta / best_loss
        action = rel < 0.05 ? "redundant (drop)" : "keep"
        push!(redundancy, (cp, l_minus, delta, rel, action))
        println("  CP $cp: without it loss = $(round(l_minus,digits=2)), Δ = $(round(delta,digits=2)), rel = $(round(rel,digits=4)) -> $action")
    end
    CSV.write(joinpath(OUT_DIR, "redundancy_test.csv"), redundancy)

    refined_cps = [r.cp for r in eachrow(redundancy) if r.action == "keep"]
    println("[$LABEL] Refined CP set (drop Δloss < 5%): $refined_cps")

    # 2. Local relocation: for each retained CP, scan ±5 days
    relocation = DataFrame(cp=Int[], best_local_cp=Int[], original_loss=Float64[], local_min_loss=Float64[])
    for cp in refined_cps
        other_cps = filter(!=(cp), refined_cps)
        local_best = cp
        local_best_loss = best_loss
        for cand in max(11, cp - 5):min(n - 10, cp + 5)
            cand in other_cps && continue
            cps_try = sort([other_cps; cand])
            l = loss_for(cps_try, params_best)
            if l < local_best_loss
                local_best_loss = l
                local_best = cand
            end
        end
        push!(relocation, (cp, local_best, best_loss, local_best_loss))
        println("  CP $cp: local best = $local_best, loss = $(round(local_best_loss,digits=2))")
    end
    CSV.write(joinpath(OUT_DIR, "local_relocation.csv"), relocation)

    refined_relocated = sort(unique([r.best_local_cp for r in eachrow(relocation)]))
    println("[$LABEL] Refined + relocated CP set: $refined_relocated")

    CSV.write(joinpath(OUT_DIR, "refined_cps.csv"),
              DataFrame(cp=refined_relocated))

    # Report
    lines = String[]
    push!(lines, "# Task F — Post-processing for winner `$LABEL`")
    push!(lines, "")
    push!(lines, "- **Original CPs:** $(cps_best)")
    push!(lines, "- **Original loss (LBFGS refit):** $(round(best_loss, digits=2))")
    push!(lines, "- **Refined CPs (drop if Δloss < 5%):** $refined_cps")
    push!(lines, "- **After local relocation:** $refined_relocated")
    push!(lines, "")
    push!(lines, "## Redundancy test")
    push!(lines, "")
    push!(lines, "| cp | loss_without | Δloss | relative increase | action |")
    push!(lines, "|---|---|---|---|---|")
    for r in eachrow(redundancy)
        push!(lines, @sprintf("| %d | %.2f | %.2f | %.4f | %s |",
                              r.cp, r.loss_without, r.delta_loss, r.relative_increase, r.action))
    end
    push!(lines, "")
    push!(lines, "## Local relocation")
    push!(lines, "")
    push!(lines, "| cp | best_local_cp | original_loss | local_min_loss |")
    push!(lines, "|---|---|---|---|")
    for r in eachrow(relocation)
        push!(lines, @sprintf("| %d | %d | %.2f | %.2f |",
                              r.cp, r.best_local_cp, r.original_loss, r.local_min_loss))
    end
    push!(lines, "")
    push!(lines, "## Files")
    push!(lines, "- `redundancy_test.csv`")
    push!(lines, "- `local_relocation.csv`")
    push!(lines, "- `refined_cps.csv`")

    open(joinpath(OUT_DIR, "report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end
    println("[$LABEL] Report saved to $(OUT_DIR)/report.md")
end

main()
