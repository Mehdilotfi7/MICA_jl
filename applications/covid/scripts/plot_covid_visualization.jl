#!/usr/bin/env julia
# ============================================================
# Visualize COVID-19 data, MICA simulation, and detected CPs.
#
# Produces figures in ../results/:
#   - covid_visualization_<objective>.png   (BIC / MDL / AIC)
#   - covid_visualization_original_penalty0.png
#   - covid_visualization_comparison.png    (original vs BIC side-by-side)
#
# Based on Mica.jl/examples/Covid-model/_example_covid_ODE.jl
# ============================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Dates
using Plots
using LabelledArrays

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results")
mkpath(OUT_DIR)

# ---------- model definitions (must match run script) ----------
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
parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
            :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9, 1/16, 0.0055,
                      0.2, 0.05, 0.17, 0.144, 0.01, 0.017, 0.173, 0.01]
lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
         0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
upper = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
         0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]
bounds = (lower, upper)
u0 = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tspan = (0.0, 399.0)

ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
model_manager = ModelManager(ode_spec)

n_global = 8
n_segment_specific = 8
data_indices = [5, 6, 7, 9, 11]

# labels / dates
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

function safe_log10(v)
    return [x > 0 ? log10(x) : NaN for x in v]
end

function make_covid_panel(cps, params; title_suffix="", use_log::Bool=true)
    sim = simulate_with_cps(cps, params)
    layout = @layout [a b c; d e f]  # f stays empty
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
    # hide unused subplot
    plot!(plt[6], axis=([], false), grid=false, legend=false, frame=:none)
    return plt
end

# ---------- main visualizations (log + raw) ----------
for obj in ["bic", "mdl", "aic"]
    cps = load_cps(joinpath(OUT_DIR, "covid_detected_cps_$(obj).csv"))
    params = load_params(joinpath(OUT_DIR, "covid_params_$(obj).csv"))
    for use_log in [true, false]
        suffix = use_log ? "log" : "raw"
        plt = make_covid_panel(cps, params;
                               title_suffix="$(uppercase(obj)) — $suffix scale",
                               use_log=use_log)
        out = joinpath(OUT_DIR, "covid_visualization_$(obj)_$(suffix).png")
        savefig(plt, out)
        println("Saved: $out  (#CPs=$(length(cps)), CPs=$cps)")
    end
end

# ---------- original penalty-0 visualization (log + raw) ----------
orig_cps = load_cps(joinpath(EXAMPLE_DIR, "results_detected_cp_penalty0_ts10_pop150.csv"))
orig_params = load_params(joinpath(EXAMPLE_DIR, "results_params_penalty0__ts10_pop150.csv"))
plt_orig_log = make_covid_panel(orig_cps, orig_params;
                                title_suffix="original example (κ=0 penalty) — log scale")
out_orig_log = joinpath(OUT_DIR, "covid_visualization_original_penalty0_log.png")
savefig(plt_orig_log, out_orig_log)
println("Saved: $out_orig_log  (#CPs=$(length(orig_cps)), CPs=$orig_cps)")

plt_orig_raw = make_covid_panel(orig_cps, orig_params; use_log=false,
                                title_suffix="original example (κ=0 penalty) — raw scale")
out_orig_raw = joinpath(OUT_DIR, "covid_visualization_original_penalty0_raw.png")
savefig(plt_orig_raw, out_orig_raw)
println("Saved: $out_orig_raw  (#CPs=$(length(orig_cps)), CPs=$orig_cps)")

# ---------- comparison: original vs BIC (log + raw) ----------
bic_cps = load_cps(joinpath(OUT_DIR, "covid_detected_cps_bic.csv"))
bic_params = load_params(joinpath(OUT_DIR, "covid_params_bic.csv"))

for use_log in [true, false]
    suffix = use_log ? "log" : "raw"
    plt_orig = make_covid_panel(orig_cps, orig_params; use_log=use_log,
                                title_suffix="original κ=0 penalty — $suffix scale")
    plt_bic = make_covid_panel(bic_cps, bic_params; use_log=use_log,
                               title_suffix="new BIC — $suffix scale")
    plt_comp = plot(plt_orig, plt_bic, layout=(2, 1), size=(1200, 1400), dpi=200,
                    plot_title="COVID-19: original κ=0 penalty (top) vs. new BIC (bottom) — $suffix scale",
                    plot_titlefontsize=14)
    out_comp = joinpath(OUT_DIR, "covid_visualization_comparison_$(suffix).png")
    savefig(plt_comp, out_comp)
    println("Saved: $out_comp")
end

println("\nAll COVID visualizations saved to $OUT_DIR")
