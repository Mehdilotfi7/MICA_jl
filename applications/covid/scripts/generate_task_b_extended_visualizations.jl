#!/usr/bin/env julia
# Generate two visualizations for every TASK_A result directory:
#   1) raw-count view
#   2) the scale actually used by the loss function for that setting
#
# For log/sqrt/boxcox this plots the transformed data & simulation.
# For relative/abs it plots the per-channel residual/absolute error.
# Usage: julia generate_task_a_visualizations.jl

ENV["GKSwstype"] = "nul"

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using CSV, DataFrames, Statistics, Dates
using OrdinaryDiffEq
using Smoothers
using Plots
using LabelledArrays
using JSON
gr()

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const RESULT_ROOT = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_B_EXTENDED")

# ---------- model (must match the run scripts) ----------
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

# ---------- data ----------
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

# ---------- settings ----------
parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
            :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
n_global = 8
n_segment_specific = 8
data_indices = [5, 6, 7, 9, 11]
u0 = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

compartment_titles = ["Infected", "Hospitalized", "ICU", "Death", "Vaccination"]
base_date = Date("2020-01-27")
dates = base_date .+ Day.(0:(size(data_CP, 2)-1))

# ---------- transforms (must match run_covid_task_a_sensitivity.jl) ----------
log_transform(data, threshold=1) = [val >= threshold ? log(val) : 0 for val in data]
sqrt_transform(data) = [val > 0 ? sqrt(val) : 0 for val in data]
function boxcox_transform(data, λ=0.25)
    return [val > 0 ? (val^λ - 1) / λ : 0 for val in data]
end

function loss_scale_transform(loss_label::String, data::AbstractVector, sim::AbstractVector)
    if loss_label == "log"
        return log_transform(data), log_transform(sim), "log(count) — loss scale"
    elseif loss_label == "sqrt"
        return sqrt_transform(data), sqrt_transform(sim), "sqrt(count) — loss scale"
    elseif loss_label == "boxcox"
        return boxcox_transform(data), boxcox_transform(sim), "Box–Cox λ=0.25 — loss scale"
    elseif loss_label == "relative"
        # Plot both series on a common relative scale (normalised by the channel max)
        ref = maximum([data; 1.0])
        return data ./ ref, sim ./ ref, "relative scale (value / channel max)"
    elseif loss_label == "abs"
        # Absolute loss operates on raw counts; show the same raw view as the scale plot
        return data, sim, "count — abs loss scale"
    else
        return data, sim, "count"
    end
end

# ---------- helpers ----------
function load_params(csv_path)
    df = CSV.read(csv_path, DataFrame)
    col = hasproperty(df, :value) ? :value : :params
    return Float64.(df[:, col])
end

function load_cps(csv_path)
    df = CSV.read(csv_path, DataFrame)
    col = hasproperty(df, :cp) ? :cp : :detected_cp
    return Int.(df[:, col])
end

function infer_loss_label(label::String)
    # New alternative-loss runs carry the loss in their directory name
    for suffix in ("abs", "relative", "boxcox", "sqrt")
        if endswith(label, "_" * suffix)
            return suffix
        end
    end
    # Everything else (bic/mdl/aic/penalty_zero/legacy_kappa40/kappa_*) uses log loss
    return "log"
end

function extract_segment_params(chromosome::Vector{Float64}, n_global::Int, n_seg::Int)
    constant = chromosome[1:n_global]
    n_segments = div(length(chromosome) - n_global, n_seg)
    seg_list = Vector{Float64}[
        chromosome[n_global + (s - 1) * n_seg + 1 : n_global + s * n_seg]
        for s in 1:n_segments
    ]
    return constant, seg_list
end

function simulate_with_cps(cps, params)
    constant_pars, segment_pars_list = extract_segment_params(params, n_global, n_segment_specific)
    u0_curr = u0
    sim_segments = Matrix{Float64}[]
    for s in 1:(length(cps) + 1)
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end   = (s > length(cps)) ? size(data_CP, 2) : cps[s]
        all_pars = @LArray [constant_pars; segment_pars_list[s]] parnames
        tspan = (Float64(idx_start - 1), Float64(idx_end - 1))
        raw_seg = example_ode_model(all_pars, tspan, u0_curr)
        sim_seg = hasproperty(raw_seg, :u) ? reduce(hcat, raw_seg.u) : raw_seg
        push!(sim_segments, sim_seg)
        u0_curr = sim_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end

function make_covid_panel(cps, params, loss_label::String; title_suffix="")
    sim = simulate_with_cps(cps, params)
    layout = @layout [a b c; d e f]
    plt = plot(layout=layout, size=(1200, 750), dpi=200,
               plot_title="COVID-19 Germany" * (isempty(title_suffix) ? "" : " — $title_suffix"),
               plot_titlefontsize=14)

    for (k, (row, ttl)) in enumerate(zip(data_indices, compartment_titles))
        ax = plt[k]
        d_raw = data_CP[k, :]
        s_raw = sim[row, :]
        d, s, ylbl = loss_scale_transform(loss_label, d_raw, s_raw)

        plot!(ax, dates, d, label="Data", lw=1.2, color=:dodgerblue, alpha=0.8)
        plot!(ax, dates, s, label="Simulation", lw=1.5, color=:orangered)
        for (j, cp) in enumerate(cps)
            vline!(ax, [dates[cp]], color=:green, lw=1.5, linestyle=:dash,
                   label=(j == 1 ? "Change points" : false))
        end
        title!(ax, ttl, fontsize=11)
        ylabel!(ax, ylbl)
        plot!(ax, xrotation=45)
        if k in (4, 5, 6)
            xlabel!(ax, "Date")
        end
        if k == 1
            plot!(ax, legend=:topleft, legendfontsize=7)
        else
            plot!(ax, legend=false)
        end
    end
    # hide unused subplot
    plot!(plt[6], axis=([], false), grid=false, legend=false, frame=:none)
    return plt
end

# ---------- loop over result directories ----------
result_dirs = sort(filter(isdir, readdir(RESULT_ROOT, join=true)))
result_dirs = filter(d -> startswith(basename(d), "results_"), result_dirs)

for d in result_dirs
    label = replace(basename(d), "results_" => "")
    cps_file = joinpath(d, "covid_detected_cps_origset_$(label).csv")
    params_file = joinpath(d, "covid_params_origset_$(label).csv")
    summary_file = joinpath(d, "summary.json")
    if !isfile(cps_file) || !isfile(params_file)
        @warn "Skipping $label (missing CSVs)"
        continue
    end

    # Prefer loss_label from summary.json; fall back to directory-name parsing
    loss_label = infer_loss_label(label)
    if isfile(summary_file)
        try
            s = JSON.parsefile(summary_file)
            if haskey(s, "loss_label") && s["loss_label"] isa String && s["loss_label"] != "?"
                loss_label = s["loss_label"]
            end
        catch
        end
    end

    # Remove stale log-scale figures from the previous version
    for old in filter(f -> startswith(basename(f), "covid_visualization_origset_$(label)_log.png"), readdir(d, join=true))
        rm(old)
    end

    cps = load_cps(cps_file)
    params = load_params(params_file)
    @info "Plotting $label (loss = $loss_label)" cps

    # 1) raw-count view
    plt_raw = make_covid_panel(cps, params, "raw"; title_suffix="$label — raw counts")
    out_raw = joinpath(d, "covid_visualization_origset_$(label)_raw.png")
    savefig(plt_raw, out_raw)
    println("  Saved: $out_raw")

    # 2) loss-scale view
    plt_loss = make_covid_panel(cps, params, loss_label; title_suffix="$label — $loss_label loss scale")
    out_loss = joinpath(d, "covid_visualization_origset_$(label)_$(loss_label).png")
    savefig(plt_loss, out_loss)
    println("  Saved: $out_loss")
end

println("\nAll TASK_A visualizations done.")
