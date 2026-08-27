#!/usr/bin/env julia
# Compute the common equal-weight log-loss for every Task B variant so that the
# scaling comparison table has a comparable metric in every row.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using LabelledArrays
using JSON, Printf

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const TASK_B_DIR  = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_B")

function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ * cos(2 * π * ((t - t₀) / 365))
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
        return Matrix(sol)
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

log_transform(data, threshold=1) = [val >= threshold ? log(val) : 0 for val in data]

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

parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
            :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
n_global = 8
n_segment_specific = 8
u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
data_indices = [5, 6, 7, 9, 11]
n = size(data_CP, 2)

function extract_segment_params(chromosome::Vector{Float64})
    constant = chromosome[1:n_global]
    n_segments = div(length(chromosome) - n_global, n_segment_specific)
    seg_list = Vector{Float64}[
        chromosome[n_global + (s - 1) * n_segment_specific + 1 : n_global + s * n_segment_specific]
        for s in 1:n_segments
    ]
    return constant, seg_list
end

function simulate_with_cps(cps, params)
    constant_pars, segment_pars_list = extract_segment_params(params)
    u0_curr = u0_full
    sim_segments = Matrix{Float64}[]
    for s in 1:(length(cps) + 1)
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end   = (s > length(cps)) ? n : cps[s]
        all_pars = @LArray [constant_pars; segment_pars_list[s]] parnames
        tspan_seg = (Float64(idx_start), Float64(idx_end))
        raw_seg = example_ode_model(all_pars, tspan_seg, u0_curr)
        push!(sim_segments, raw_seg)
        u0_curr = raw_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end

function equal_weight_log_loss(sim)
    rows = data_indices
    total = 0.0
    for (k, r) in enumerate(rows)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(data_CP[k, :]))
    end
    return total
end

# ---------- compute comparable loss for every variant ----------
variants = ["equal", "invmean", "invstd", "normmean"]
for v in variants
    rdir = joinpath(TASK_B_DIR, "results_$(v)")
    cps_file = joinpath(rdir, "covid_detected_cps_origset_$(v).csv")
    params_file = joinpath(rdir, "covid_params_origset_$(v).csv")
    summary_file = joinpath(rdir, "summary.json")

    if !isfile(cps_file) || !isfile(params_file)
        println("Skipping $v: missing files")
        continue
    end

    cps = Int.(CSV.read(cps_file, DataFrame).cp)
    # drop missing (zero-CP cases)
    cps = collect(skipmissing(cps))
    params = Float64.(CSV.read(params_file, DataFrame).value)

    sim = simulate_with_cps(cps, params)
    ll = equal_weight_log_loss(sim)

    # update summary.json
    if isfile(summary_file)
        s = JSON.parsefile(summary_file)
        s["equal_weight_log_loss"] = ll
        # also set loss if it was missing/nothing
        if get(s, "loss", nothing) === nothing
            s["loss"] = ll
        end
        open(summary_file, "w") do f
            JSON.print(f, s, 2)
        end
    end
    println("$v: equal-weight log-loss = $ll")
end

# ---------- update comparison table ----------
comp_csv = joinpath(TASK_B_DIR, "scaling_variant_comparison.csv")
comp_md  = joinpath(TASK_B_DIR, "scaling_variant_comparison.md")

comp = CSV.read(comp_csv, DataFrame)
for r in eachrow(comp)
    v = r.variant
    sfile = joinpath(TASK_B_DIR, "results_$(v)", "summary.json")
    if isfile(sfile)
        s = JSON.parsefile(sfile)
        r.equal_weight_log_loss = get(s, "equal_weight_log_loss", NaN)
    end
end
CSV.write(comp_csv, comp)

md_lines = String[]
push!(md_lines, "# Task B — Scaling-variant comparison")
push!(md_lines, "")
push!(md_lines, "| variant | n_cps | cps | native_loss | equal_weight_log_loss | time_seconds |")
push!(md_lines, "|---|---|---|---|---|---|")
for r in eachrow(comp)
    native_str = isnan(r.final_loss) ? "—" : @sprintf("%.4f", r.final_loss)
    eq_str = isnan(r.equal_weight_log_loss) ? "—" : @sprintf("%.4f", r.equal_weight_log_loss)
    time_str = isnan(r.time_seconds) ? "NA" : @sprintf("%.1f", r.time_seconds)
    push!(md_lines, "| $(r.variant) | $(r.n_cps) | $(r.cps) | $(native_str) | $(eq_str) | $(time_str) |")
end
push!(md_lines, "")
push!(md_lines, "- **equal**: equal weights on log-transformed channels (baseline).")
push!(md_lines, "- **invmean**: weights = 1 / mean(full observed channel).")
push!(md_lines, "- **invstd**: weights = 1 / std(full observed channel).")
push!(md_lines, "- **normmean**: divide each channel by its mean, then use raw (not log) absolute errors.")
push!(md_lines, "")
push!(md_lines, "**Note:** Native losses are computed with each variant's own objective, so they are not directly comparable across rows. The `equal_weight_log_loss` column gives every variant's loss under the common equal-weight log-transformed objective, making cross-variant comparison possible.")
open(comp_md, "w") do f
    write(f, join(md_lines, "\n") * "\n")
end

# ---------- update report.md table ----------
report_file = joinpath(TASK_B_DIR, "report.md")
report = read(report_file, String)

# rebuild the comparison block
new_block = String[]
push!(new_block, "## 3. Change-point comparison across loss variants")
push!(new_block, "")
push!(new_block, "| variant | n_cps | cps | native_loss | equal_weight_log_loss | time_seconds |")
push!(new_block, "|---|---|---|---|---|---|")
for r in eachrow(comp)
    native_str = isnan(r.final_loss) ? "—" : @sprintf("%.4f", r.final_loss)
    eq_str = isnan(r.equal_weight_log_loss) ? "—" : @sprintf("%.4f", r.equal_weight_log_loss)
    time_str = isnan(r.time_seconds) ? "NA" : @sprintf("%.1f", r.time_seconds)
    push!(new_block, "| $(r.variant) | $(r.n_cps) | $(r.cps) | $(native_str) | $(eq_str) | $(time_str) |")
end
push!(new_block, "")
push!(new_block, "*Native losses use each variant's own objective; `equal_weight_log_loss` gives a common comparable metric.*")

idx1 = findfirst("## 3. Change-point comparison across loss variants", report)[1]
idx2 = findfirst("### Discussion", report)[1]
report = report[1:idx1-1] * join(new_block, "\n") * "\n" * report[idx2:end]

write(report_file, report)

println("\nUpdated $(comp_csv), $(comp_md), and $(report_file)")
