#!/usr/bin/env julia
# Visualize the four COVID objective runs performed with the original GA settings.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Dates
using Plots
using LabelledArrays

# Avoid headless GR/Plots crash when animation is not used
Mica.global_anim[] = Animation()

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
# Optional CLI argument: output-directory suffix (default "pop150")
# If the argument is an absolute path, use it directly; otherwise treat it as a
# suffix and look for ../results_<suffix>.
out_arg = length(ARGS) >= 1 ? ARGS[1] : "pop150"
if isabspath(out_arg)
    const OUT_DIR = out_arg
else
    const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results_$(out_arg)")
end
mkpath(OUT_DIR)

# ---------- model definitions ----------
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
        return Matrix(sol)
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
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

parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
            :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
n_global = 8
n_segment_specific = 8
u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
data_indices = [5, 6, 7, 9, 11]
compartment_titles = ["Infected", "Hospitalized", "ICU", "Death", "Vaccination"]
base_date = Date("2020-01-27")
dates = base_date .+ Day.(0:(size(data_CP, 2)-1))

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
    u0_curr = u0_full
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

safe_log10(v) = [x > 0 ? log10(x) : NaN for x in v]

function make_covid_panel(cps, params; title_suffix="", use_log::Bool=true)
    sim = simulate_with_cps(cps, params)
    layout = @layout [a b c; d e f]
    plt = plot(layout=layout, size=(1200, 750), dpi=200,
               plot_title="COVID-19 Germany" * (isempty(title_suffix) ? "" : " — $title_suffix"),
               plot_titlefontsize=14)
    for (k, (row, ttl)) in enumerate(zip(data_indices, compartment_titles))
        ax = plt[k]
        d = use_log ? safe_log10(data_CP[k, :]) : data_CP[k, :]
        s = use_log ? safe_log10(sim[row, :]) : sim[row, :]
        plot!(ax, dates, d, label="Data", lw=1.2, color=:dodgerblue, alpha=0.8)
        plot!(ax, dates, s, label="Simulation", lw=1.5, color=:orangered)
        for (j, cp) in enumerate(cps)
            vline!(ax, [dates[cp]], color=:green, lw=1.5, linestyle=:dash,
                   label=(j == 1 ? "Change points" : false))
        end
        title!(ax, ttl, fontsize=11)
        ylabel!(ax, use_log ? "log₁₀(count)" : "count")
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
    plot!(plt[6], axis=([], false), grid=false, legend=false, frame=:none)
    return plt
end

function metrics(sim)
    rows = data_indices
    log_loss = sum(
        sum(abs, log_transform(sim[r, :]) .- log_transform(data_CP[k, :]))
        for (k, r) in enumerate(rows)
    )
    raw_mse = mean(
        sum((sim[r, :] .- data_CP[k, :]).^2)
        for (k, r) in enumerate(rows)
    )
    raw_mae = mean(
        sum(abs.(sim[r, :] .- data_CP[k, :]))
        for (k, r) in enumerate(rows)
    )
    return log_loss, raw_mse, raw_mae
end

# ---------- generate figures ----------
# Discover every objective label present in the output directory.
function discover_objectives(out_dir)
    objs = String[]
    for fname in readdir(out_dir)
        m = match(r"^covid_detected_cps_origset_(.*)\.csv$", fname)
        if m !== nothing
            push!(objs, m.captures[1])
        end
    end
    return sort(unique(objs))
end

objectives = discover_objectives(OUT_DIR)
if isempty(objectives)
    println("No covid_detected_cps_origset_*.csv files found in $OUT_DIR")
    exit(0)
end
println("Discovered objectives in $OUT_DIR: $(join(objectives, ", "))")

panels = []
for obj in objectives
    cps_file = joinpath(OUT_DIR, "covid_detected_cps_origset_$(obj).csv")
    params_file = joinpath(OUT_DIR, "covid_params_origset_$(obj).csv")
    if !isfile(cps_file) || !isfile(params_file)
        println("Skipping $obj: missing $cps_file or $params_file")
        continue
    end
    cps = load_cps(cps_file)
    params = load_params(params_file)
    sim = simulate_with_cps(cps, params)
    ll, rmse, rmae = metrics(sim)
    println("$obj: #CPs=$(length(cps)), CPs=$cps, log-loss=$(round(ll,digits=2)), raw-MSE=$(round(rmse,sigdigits=4)), raw-MAE=$(round(rmae,sigdigits=4))")

    for use_log in [true, false]
        suffix = use_log ? "log" : "raw"
        plt = make_covid_panel(cps, params;
                               title_suffix="$(uppercase(obj)) — $suffix scale",
                               use_log=use_log)
        out = joinpath(OUT_DIR, "covid_visualization_origset_$(obj)_$(suffix).png")
        savefig(plt, out)
        println("  Saved: $out")
    end
    push!(panels, (obj, cps, params, sim))
end

# ---------- combined infected-panel comparison ----------
if length(panels) >= 2
    n_panels = length(panels)
    if n_panels == 2
        layout_grid = (1, 2)
        fig_size = (1400, 600)
    elseif n_panels <= 4
        layout_grid = (2, 2)
        fig_size = (1400, 900)
    else
        ncols = ceil(Int, sqrt(n_panels))
        nrows = ceil(Int, n_panels / ncols)
        layout_grid = (nrows, ncols)
        fig_size = (350 * ncols, 300 * nrows)
    end
    fig_log = plot(layout=layout_grid, size=fig_size, dpi=200,
                   plot_title="COVID-19 Germany — Infected (log scale) — original GA settings",
                   plot_titlefontsize=14)
    fig_raw = plot(layout=layout_grid, size=fig_size, dpi=200,
                   plot_title="COVID-19 Germany — Infected (raw scale) — original GA settings",
                   plot_titlefontsize=14)
    for (i, (obj, cps, params, sim)) in enumerate(panels)
        ax_log = fig_log[i]
        ax_raw = fig_raw[i]
        plot!(ax_log, dates, safe_log10(data_CP[1, :]), label="Data", lw=1.0, color=:dodgerblue, alpha=0.8)
        plot!(ax_log, dates, safe_log10(sim[5, :]), label="Simulation", lw=1.5, color=:orangered)
        plot!(ax_raw, dates, data_CP[1, :], label="Data", lw=1.0, color=:dodgerblue, alpha=0.8)
        plot!(ax_raw, dates, sim[5, :], label="Simulation", lw=1.5, color=:orangered)
        for (j, cp) in enumerate(cps)
            vline!(ax_log, [dates[cp]], color=:green, lw=1.5, linestyle=:dash,
                   label=(j == 1 ? "CPs" : false))
            vline!(ax_raw, [dates[cp]], color=:green, lw=1.5, linestyle=:dash,
                   label=(j == 1 ? "CPs" : false))
        end
        title!(ax_log, "$(uppercase(obj)): $(length(cps)) CPs", fontsize=11)
        title!(ax_raw, "$(uppercase(obj)): $(length(cps)) CPs", fontsize=11)
        ylabel!(ax_log, "log₁₀(count)")
        ylabel!(ax_raw, "count")
        xlabel!(ax_log, "Date")
        xlabel!(ax_raw, "Date")
        plot!(ax_log, xrotation=45)
        plot!(ax_raw, xrotation=45)
        if i == 1
            plot!(ax_log, legend=:topleft, legendfontsize=7)
            plot!(ax_raw, legend=:topleft, legendfontsize=7)
        else
            plot!(ax_log, legend=false)
            plot!(ax_raw, legend=false)
        end
    end
    savefig(fig_log, joinpath(OUT_DIR, "covid_visualization_origset_infected_log.png"))
    savefig(fig_raw, joinpath(OUT_DIR, "covid_visualization_origset_infected_raw.png"))
    println("\nSaved combined infected comparisons.")
else
    println("\nSkipping combined infected comparison (need >= 2 objectives, found $(length(panels))).")
end
