#!/usr/bin/env julia
# Data2Dynamic-style profile-likelihood (PLE) for a COVID-19 winner.
# Usage: julia covid_ple_winner.jl <label> <cps_csv> <params_csv> <out_dir> [param_selection]
#
# param_selection: "all" (default), "global", or a comma-separated list of
#                  parameter labels, e.g. "δ_global,ᴺβ_seg2".
#
# Method: for each selected scalar parameter, fix it to a grid of values and
# re-optimise all remaining parameters with LBFGS (Fminbox).  This mirrors the
# PLE strategy used in Data2Dynamics / EbolaToolkit.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics, Random
using OrdinaryDiffEq, Smoothers
using Optim, Evolutionary
using LabelledArrays
using Printf
using Dates
using Base.Threads

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")

const PLE_NPOINTS = parse(Int, get(ENV, "COVID_PLE_NPOINTS", "20"))
const PLE_ITER = parse(Int, get(ENV, "COVID_PLE_ITER", "200"))
const CHISQ_95 = 3.8414588206941285   # chi2(1, 0.95)

# ---------- model / data (same as Tasks A/D/F/G) ----------
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
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u),
                verbose=false)
    if SciMLBase.successful_retcode(sol)
        return sol[:, :]
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

# Lighter ODE settings used only during profiling (many repeated solves).
function ple_ode_model(params, tspan::Tuple{Float64, Float64}, u0)
    if tspan[2] < 330
        params[:ν] = 0.0
    end
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-5, reltol=1.0e-5,
                maxiters=20000,
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u),
                verbose=false)
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
const PLE_NDAYS = parse(Int, get(ENV, "COVID_PLE_NDAYS", "400"))
data_CP = [vector[1:PLE_NDAYS] for vector in data_CP]
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

# ---------- CLI ----------
length(ARGS) >= 4 || error("Usage: julia covid_ple_winner.jl <label> <cps_csv> <params_csv> <out_dir> [all|global|label1,label2,...]")
const LABEL = ARGS[1]
const CPS_CSV = ARGS[2]
const PARAMS_CSV = ARGS[3]
const OUT_DIR = ARGS[4]
mkpath(OUT_DIR)

const cps_best = sort(unique(Int.(CSV.read(CPS_CSV, DataFrame).cp)))
const params_best = Float64.(CSV.read(PARAMS_CSV, DataFrame).value)
const param_labels = parameter_labels(length(cps_best))[1:length(params_best)]

const n_segments = length(cps_best) + 1
const n_params = n_global + n_segments * n_segment_specific
const lb_full = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
const ub_full = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]

const ode_spec = ODEModelSpec(ple_ode_model, params_best, u0_full, (0.0, Float64(n)))
const model_manager = ModelManager(ode_spec)

# ---------- objective with one fixed parameter ----------
function objective_full(chromosome)
    return Mica.objective_function(
        chromosome, cps_best, parnames, n_global, n_segment_specific,
        model_manager, loss_abs_log, data_CP
    )
end

function optimize_free(fixed_idx::Int, fixed_val::Float64, x0::Vector{Float64})
    free = setdiff(1:n_params, [fixed_idx])
    lb = lb_full[free]
    ub = ub_full[free]
    x_init = copy(x0[free])

    function loss_free(x_free)
        θ = copy(x0)
        θ[free] .= x_free
        θ[fixed_idx] = fixed_val
        return objective_full(θ)
    end

    res = Optim.optimize(
        loss_free, lb, ub, x_init,
        Fminbox(LBFGS()),
        Optim.Options(show_trace=false, iterations=PLE_ITER, time_limit=120.0)
    )
    θ_opt = copy(x0)
    θ_opt[free] .= Optim.minimizer(res)
    θ_opt[fixed_idx] = fixed_val
    return Optim.minimum(res), θ_opt
end

# ---------- best-fit reference loss ----------
const REFIT_BEST = parse(Bool, get(ENV, "COVID_PLE_REFIT_BEST", "false"))

function best_fit_loss()
    θ0 = params_best[1:n_params]
    if REFIT_BEST
        opt = OptimOptimizer(Fminbox(LBFGS()),
                             options=Optim.Options(show_trace=false, iterations=PLE_ITER))
        loss, θ = optimize_with_changepoints(
            Mica.objective_function, θ0, parnames, cps_best,
            (lb_full, ub_full), opt,
            n_global, n_segment_specific, model_manager, loss_abs_log, data_CP
        )
        return loss, θ
    else
        return objective_full(θ0), θ0
    end
end

# ---------- grid generation ----------
function ple_grid_for_param(idx::Int, best_val::Float64, n_points::Int)
    lo = lb_full[idx]
    hi = ub_full[idx]
    # Use a log-spaced grid for positive parameters with wide dynamic range,
    # otherwise a linear grid.
    if best_val > 1e-8 && (hi / max(lo, 1e-12) > 10.0)
        gmin = max(lo, best_val / 5.0)
        gmax = min(hi, best_val * 5.0)
        # ensure the best value is inside the grid
        gmin = min(gmin, best_val)
        gmax = max(gmax, best_val)
        return 10.0 .^ range(log10(gmin), log10(gmax), length=n_points)
    else
        half = 0.5 * (hi - lo)
        gmin = max(lo, best_val - half)
        gmax = min(hi, best_val + half)
        gmin = min(gmin, best_val)
        gmax = max(gmax, best_val)
        return collect(range(gmin, gmax, length=n_points))
    end
end

# ---------- profile one parameter ----------
function profile_parameter(idx::Int, best_θ::Vector{Float64}, best_loss::Float64; n_points::Int=PLE_NPOINTS)
    label = param_labels[idx]
    best_val = best_θ[idx]
    grid = ple_grid_for_param(idx, best_val, n_points)

    df = DataFrame(parameter=String[], index=Int[], value=Float64[], loss=Float64[], delta_loss=Float64[])
    x0 = copy(best_θ)

    for (j, v) in enumerate(grid)
        println("    [$label] grid point $j/$(length(grid)): value=$(@sprintf("%.5g", v))"); flush(stdout)
        try
            loss, θ_opt = optimize_free(idx, v, x0)
            x0[setdiff(1:n_params, [idx])] .= θ_opt[setdiff(1:n_params, [idx])]
            println("    [$label]   -> loss=$(@sprintf("%.4f", loss))"); flush(stdout)
            push!(df, (label, idx, v, loss, loss - best_loss))
        catch e
            @warn "PLE failed for $label at $v: $e"
            push!(df, (label, idx, v, Inf, Inf))
        end
    end
    return df
end

# ---------- confidence interval from profile ----------
function ple_ci(df::AbstractDataFrame, ref_loss::Float64, best_val::Float64, idx::Int)
    threshold = ref_loss + CHISQ_95
    valid = filter(r -> isfinite(r.loss) && r.loss <= threshold + 2 * CHISQ_95, df)
    if nrow(valid) == 0
        return (lower=missing, upper=missing, identifiable=false)
    end
    lower = minimum(valid.value)
    upper = maximum(valid.value)
    # Identifiable if the threshold is crossed inside the scanned range on both sides.
    inside = any(row -> row.loss <= threshold, eachrow(valid))
    crossed_low = lower > lb_full[idx] && inside
    crossed_high = upper < ub_full[idx] && inside
    identifiable = crossed_low && crossed_high
    return (lower=lower, upper=upper, identifiable=identifiable)
end

# ---------- parameter selection ----------
function selected_indices(selection::String)
    sel = strip(lowercase(selection))
    if sel == "all"
        return collect(1:n_params)
    elseif sel == "global"
        return collect(1:n_global)
    else
        wanted = split(selection, ",")
        wanted = strip.(wanted)
        idx = Int[]
        for w in wanted
            pos = findfirst(==(w), param_labels)
            if pos === nothing
                error("Unknown parameter label: $w")
            end
            push!(idx, pos)
        end
        return idx
    end
end

# ---------- main ----------
function main()
    selection = length(ARGS) >= 5 ? ARGS[5] : "all"
    idxs = selected_indices(selection)
    println("[$LABEL] PLE for $(length(idxs)) parameter(s) out of $n_params")
    println("[$LABEL] CPs: $cps_best")

    println("[$LABEL] Computing best-fit reference loss..."); flush(stdout)
    best_loss, best_θ = best_fit_loss()
    println("[$LABEL] Best loss = $(round(best_loss, digits=4))"); flush(stdout)

    all_profiles = DataFrame(parameter=String[], index=Int[], value=Float64[], loss=Float64[], delta_loss=Float64[])
    summary = DataFrame(parameter=String[], index=Int[], best_value=Float64[], best_loss=Float64[],
                        ci_lower=Union{Float64,Missing}[], ci_upper=Union{Float64,Missing}[],
                        identifiable=Bool[], threshold=Float64[])

    n_threads = min(Threads.nthreads(), length(idxs))
    if n_threads > 1 && length(idxs) > 1
        results = Vector{DataFrame}(undef, length(idxs))
        Threads.@threads for k in 1:length(idxs)
            idx = idxs[k]
            println("  [thread $(Threads.threadid())] profiling $(param_labels[idx])"); flush(stdout)
            results[k] = profile_parameter(idx, best_θ, best_loss; n_points=PLE_NPOINTS)
        end
        for df in results
            append!(all_profiles, df; cols=:union)
        end
    else
        for idx in idxs
            println("  profiling $(param_labels[idx])"); flush(stdout)
            df = profile_parameter(idx, best_θ, best_loss; n_points=PLE_NPOINTS)
            append!(all_profiles, df; cols=:union)
        end
    end

    CSV.write(joinpath(OUT_DIR, "ple_results.csv"), all_profiles)

    # Reference loss is the best loss observed (best-fit vector or any profile point).
    finite_losses = filter(isfinite, all_profiles.loss)
    ref_loss = isempty(finite_losses) ? best_loss : min(best_loss, minimum(finite_losses))

    gdf = groupby(all_profiles, :index)
    for idx in idxs
        df = gdf[(index=idx,)]
        ci = ple_ci(df, ref_loss, best_θ[idx], idx)
        push!(summary, (parameter=param_labels[idx], index=idx, best_value=best_θ[idx],
                        best_loss=ref_loss, ci_lower=ci.lower, ci_upper=ci.upper,
                        identifiable=ci.identifiable, threshold=ref_loss + CHISQ_95))
    end
    CSV.write(joinpath(OUT_DIR, "ple_summary.csv"), summary)

    write_report(summary, all_profiles, best_loss, selection)
    println("[$LABEL] PLE done. Outputs in $OUT_DIR")
end

function write_report(summary::DataFrame, profiles::DataFrame, best_loss::Float64, selection::String)
    lines = String[]
    push!(lines, "# Profile-likelihood analysis for winner `$(LABEL)`")
    push!(lines, "")
    push!(lines, "**Parameter selection:** $(selection)")
    push!(lines, "**Original CPs:** $(cps_best)")
    push!(lines, "**Best loss (profile reference):** $(round(best_loss, digits=4))")
    push!(lines, "**Threshold (Δloss ≤ 3.84):** $(round(best_loss + CHISQ_95, digits=4))")
    push!(lines, "")
    push!(lines, "## Summary")
    push!(lines, "")
    push!(lines, "| parameter | best value | CI lower | CI upper | identifiable |")
    push!(lines, "|---|---|---|---|---|")
    for r in eachrow(summary)
        l = ismissing(r.ci_lower) ? "—" : @sprintf("%.5g", r.ci_lower)
        u = ismissing(r.ci_upper) ? "—" : @sprintf("%.5g", r.ci_upper)
        id = r.identifiable ? "yes" : "no"
        push!(lines, @sprintf("| %s | %.5g | %s | %s | %s |",
                              r.parameter, r.best_value, l, u, id))
    end
    push!(lines, "")
    push!(lines, "## Files")
    push!(lines, "- `ple_results.csv` — full profile curves")
    push!(lines, "- `ple_summary.csv` — approximate 95% confidence intervals")
    open(joinpath(OUT_DIR, "ple_report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end
end

main()
