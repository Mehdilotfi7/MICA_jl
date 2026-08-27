#!/usr/bin/env julia
# Task D — Uncertainty quantification for a selected COVID-19 winner result.
# Usage: julia covid_uncertainty_winner.jl <label> <cps_csv> <params_csv> <out_dir>
#
# Produces:
#   - parameter_bootstrap_ci.csv / parameter_bootstrap_samples.csv
#   - cp_profile_loss.csv / cp_profile_ci.csv
#   - report.md

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics, Random
using OrdinaryDiffEq, Smoothers
using Optim, Evolutionary
using LabelledArrays
using Printf
using Dates

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")

const BASE_DATE = Date("2020-01-27")
const B_BOOT = parse(Int, get(ENV, "COVID_BOOT_REPS", "30"))
const CP_WINDOW = 5           # ±days for CP profile
const N_PARAMS_LBFGS = 200    # optimizer iterations for fast refits

# ---------- model / data (same as Task A) ----------
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

function extract_segment_params(chromosome)
    constant = chromosome[1:n_global]
    n_segments = div(length(chromosome) - n_global, n_segment_specific)
    seg_list = [chromosome[n_global + (s - 1) * n_segment_specific + 1 : n_global + s * n_segment_specific] for s in 1:n_segments]
    return constant, seg_list
end

function simulate_full(cps, params)
    constant, seg_list = extract_segment_params(params)
    u0_curr = u0_full
    sim_segments = Matrix{Float64}[]
    for s in 1:(length(cps) + 1)
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end   = (s > length(cps)) ? n : cps[s]
        all_pars = @LArray [constant; seg_list[s]] parnames
        tspan_seg = (Float64(idx_start), Float64(idx_end))
        raw_seg = example_ode_model(all_pars, tspan_seg, u0_curr)
        sim_seg = hasproperty(raw_seg, :u) ? reduce(hcat, raw_seg.u) : raw_seg
        push!(sim_segments, sim_seg)
        u0_curr = sim_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end
const lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
               0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
const upper = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
               0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]

# ---------- CLI ----------
length(ARGS) >= 4 || error("Usage: julia covid_uncertainty_winner.jl <label> <cps_csv> <params_csv> <out_dir>")
const LABEL = ARGS[1]
const CPS_CSV = ARGS[2]
const PARAMS_CSV = ARGS[3]
const OUT_DIR = ARGS[4]
mkpath(OUT_DIR)

# ---------- load selected fit ----------
cps_best = sort(unique(Int.(CSV.read(CPS_CSV, DataFrame).cp)))
params_best = Float64.(CSV.read(PARAMS_CSV, DataFrame).value)

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
    return labs[1:length(params_best)]
end

const param_labels = parameter_labels(length(cps_best))

const ode_spec = ODEModelSpec(example_ode_model, params_best, u0_full, (0.0, Float64(n)))
const model_manager = ModelManager(ode_spec)

function loss_abs_log(obs, sim)
    total = 0.0
    for (k, r) in enumerate(data_indices)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(obs[k, :]))
    end
    return total
end

function refit_params(cps, data, x0)
    n_segments = length(cps) + 1
    n_params = n_global + n_segments * n_segment_specific
    lb = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
    ub = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]
    x0_use = copy(x0[1:n_params])

    opt = OptimOptimizer(Fminbox(LBFGS()),
                         options=Optim.Options(show_trace=false, iterations=N_PARAMS_LBFGS))
    loss, params = optimize_with_changepoints(
        Mica.objective_function, x0_use, parnames, cps, (lb, ub), opt,
        n_global, n_segment_specific, model_manager,
        (obs, sim) -> loss_abs_log(obs, sim),
        data
    )
    return loss, params
end

# ---------- 1. Parametric bootstrap for parameters ----------
function bootstrap_parameters()
    println("[$LABEL] Simulating best fit for bootstrap...")
    sim_best = simulate_full(cps_best, params_best)
    log_sim = [log_transform(sim_best[r, :]) for r in data_indices]
    log_obs = [log_transform(data_CP[k, :]) for k in 1:5]
    residuals = [log_obs[k] .- log_sim[k] for k in 1:5]

    boot_matrix = zeros(length(params_best), B_BOOT)
    Random.seed!(42)
    for b in 1:B_BOOT
        boot_log = [log_sim[k] .+ residuals[k][rand(1:n, n)] for k in 1:5]
        boot_data = Matrix(reduce(hcat, [exp.(boot_log[k]) for k in 1:5])')

        try
            loss, p_b = refit_params(cps_best, boot_data, params_best)
            boot_matrix[:, b] = p_b[1:length(params_best)]
        catch e
            @warn "Bootstrap replicate $b failed: $e"
            boot_matrix[:, b] .= NaN
        end
        if b % 5 == 0
            println("  completed bootstrap $b / $B_BOOT"); flush(stdout)
        end
    end

    ci_df = DataFrame(parameter=param_labels)
    ci_df.median = [median(filter(!isnan, boot_matrix[i, :])) for i in 1:size(boot_matrix, 1)]
    ci_df.lower_95 = [quantile(filter(!isnan, boot_matrix[i, :]), 0.025) for i in 1:size(boot_matrix, 1)]
    ci_df.upper_95 = [quantile(filter(!isnan, boot_matrix[i, :]), 0.975) for i in 1:size(boot_matrix, 1)]
    ci_df.best_fit = params_best
    CSV.write(joinpath(OUT_DIR, "parameter_bootstrap_ci.csv"), ci_df)

    boot_df = DataFrame(parameter=param_labels)
    for b in 1:B_BOOT
        boot_df[!, "boot_$b"] = boot_matrix[:, b]
    end
    CSV.write(joinpath(OUT_DIR, "parameter_bootstrap_samples.csv"), boot_df)
    println("[$LABEL] Parameter bootstrap saved.")
    return ci_df
end

# ---------- 2. CP profile / perturbation intervals ----------
function cp_profile_loss()
    println("[$LABEL] Computing CP profile losses...")
    best_loss, _ = refit_params(cps_best, data_CP, params_best)
    println("  best loss = $(round(best_loss, digits=2))")

    all_profiles = DataFrame(cp_index=Int[], original_cp=Int[], candidate_cp=Int[],
                             date=Date[], loss=Float64[], delta_loss=Float64[])

    for (j, cp) in enumerate(cps_best)
        window = max(1 + 10, cp - CP_WINDOW):min(n - 10, cp + CP_WINDOW)
        for cand in window
            cand_cp = copy(cps_best)
            cand_cp[j] = cand
            sort!(cand_cp)
            if length(unique(cand_cp)) < length(cand_cp)
                continue
            end
            try
                loss, _ = refit_params(cand_cp, data_CP, params_best)
                push!(all_profiles, (j, cp, cand, BASE_DATE + Day(cand - 1), loss, loss - best_loss))
            catch e
                @warn "CP $j candidate $cand failed: $e"
            end
        end
        println("  CP $j ($cp) profile done"); flush(stdout)
    end
    CSV.write(joinpath(OUT_DIR, "cp_profile_loss.csv"), all_profiles)

    ci_df = DataFrame(cp_index=Int[], original_cp=Int[], original_date=Date[],
                      ci_lower=Union{Int,Missing}[], ci_upper=Union{Int,Missing}[],
                      ci_lower_date=Union{Date,Missing}[], ci_upper_date=Union{Date,Missing}[])
    for j in 1:length(cps_best)
        prof = filter(r -> r.cp_index == j, all_profiles)
        subset = filter(r -> r.delta_loss <= 3.84, prof)
        if nrow(subset) > 0
            lower_cp = minimum(subset.candidate_cp)
            upper_cp = maximum(subset.candidate_cp)
            push!(ci_df, (j, cps_best[j], BASE_DATE + Day(cps_best[j] - 1),
                          lower_cp, upper_cp,
                          BASE_DATE + Day(lower_cp - 1), BASE_DATE + Day(upper_cp - 1)))
        else
            push!(ci_df, (j, cps_best[j], BASE_DATE + Day(cps_best[j] - 1),
                          missing, missing, missing, missing))
        end
    end
    CSV.write(joinpath(OUT_DIR, "cp_profile_ci.csv"), ci_df)
    println("[$LABEL] CP profile intervals saved.")
    return ci_df
end

# ---------- report ----------
function write_report(ci_params, ci_cps)
    lines = String[]
    push!(lines, "# Task D — Uncertainty quantification for winner `$LABEL`")
    push!(lines, "")
    push!(lines, "**Method:**")
    push!(lines, "- Parameter uncertainty: parametric residual bootstrap with $B_BOOT replicates. Data were simulated from the best fit; residuals in log-space were resampled with replacement and added back. Parameters were refit with fixed change points using an LBFGS optimizer.")
    push!(lines, "- Change-point uncertainty: for each detected change point, the date was perturbed in a ±$CP_WINDOW day window while re-optimizing all other parameters. An approximate 95% interval was taken as candidate dates with Δloss ≤ 3.84 (heuristic; loss is treated as a negative log-likelihood surrogate).")
    push!(lines, "")
    push!(lines, "**Original CPs:** $(cps_best)")
    push!(lines, "")
    push!(lines, "## Change-point profile intervals")
    push!(lines, "")
    push!(lines, "| cp # | best cp | best date | CI lower | CI upper | CI lower date | CI upper date |")
    push!(lines, "|---|---|---|---|---|---|---|")
    for r in eachrow(ci_cps)
        l = ismissing(r.ci_lower) ? "—" : string(r.ci_lower)
        u = ismissing(r.ci_upper) ? "—" : string(r.ci_upper)
        ld = ismissing(r.ci_lower_date) ? "—" : string(r.ci_lower_date)
        ud = ismissing(r.ci_upper_date) ? "—" : string(r.ci_upper_date)
        push!(lines, "| $(r.cp_index) | $(r.original_cp) | $(r.original_date) | $(l) | $(u) | $(ld) | $(ud) |")
    end
    push!(lines, "")
    push!(lines, "## Parameter bootstrap intervals (selected global parameters)")
    push!(lines, "")
    push!(lines, "| parameter | best fit | median | 2.5% | 97.5% |")
    push!(lines, "|---|---|---|---|---|")
    for r in eachrow(ci_params)
        if occursin("_global", r.parameter)
            push!(lines, @sprintf("| %s | %.5f | %.5f | %.5f | %.5f |",
                                  r.parameter, r.best_fit, r.median, r.lower_95, r.upper_95))
        end
    end
    push!(lines, "")
    push!(lines, "## Files")
    push!(lines, "- `parameter_bootstrap_ci.csv` / `parameter_bootstrap_samples.csv`")
    push!(lines, "- `cp_profile_loss.csv` / `cp_profile_ci.csv`")

    open(joinpath(OUT_DIR, "report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end
    println("[$LABEL] Report saved to $(OUT_DIR)/report.md")
end

function main()
    println("[$LABEL] Winner CPs: $cps_best")
    ci_params = bootstrap_parameters()
    ci_cps = cp_profile_loss()
    write_report(ci_params, ci_cps)
    println("[$LABEL] Done. Outputs in $OUT_DIR")
end

main()
