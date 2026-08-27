#!/usr/bin/env julia
# ============================================================
# Visualize wind-turbine generator-bearing temperature data,
# MICA simulation, and detected change points.
#
# Produces figures in ../results/:
#   - turbine_visualization_<objective>.png (BIC / MDL / AIC)
# ============================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Dates
using Plots

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Wind_Turbine_model")
const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results")
mkpath(OUT_DIR)

# ---------- model definition (must match run script) ----------
function example_difference_model(θ, T_initial, num_steps, extra_data)
    θ1, θ2, θ3, θ4, θ5, θ6, θ7 = θ.θ1, θ.θ2, θ.θ3, θ.θ4, θ.θ5, θ.θ6, θ.θ7
    wind_speeds, ambient_temperatures = extra_data
    generator_temperatures_sim = zeros(num_steps)
    generator_temperatures_sim[1] = T_initial
    for k in 2:num_steps
        u1, u2 = wind_speeds[k], ambient_temperatures[k]
        y_prev = generator_temperatures_sim[k-1]
        generator_temperatures_sim[k] = ((θ1*u1^3 + θ2*u1^2 + θ3*u1 + y_prev - u2) /
                                         (θ4*u1^3 + θ5*u1^2 + θ6*u1 + θ7)) + u2
    end
    return reshape(Float64.(generator_temperatures_sim), 1, :)
end

# ---------- load data ----------
df = CSV.read(joinpath(EXAMPLE_DIR, "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv"), DataFrame)
wind_speeds = df[:, "Wind speed (m/s)"][1:2500]
ambient_temperatures = df[:, "Ambient temperature (converter) (°C)"][1:2500]
generator_temperatures = df[:, "Generator bearing front temperature (°C)"][1:2500]

times = try
    DateTime.(df[:, "Date and time"][1:2500], dateformat"yyyy-mm-dd HH:MM:SS")
catch
    1:2500
end

initial_chromosome = [1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5]
parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7)
lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
upper = [100.0, 100.0, 100.0, 466.0, 466.0, 466.0, 1466.0]
bounds = (lower, upper)
u0 = generator_temperatures[1]
num_steps = 2500

de_spec = DifferenceModelSpec(example_difference_model, initial_chromosome, u0, num_steps,
                              (wind_speeds, ambient_temperatures))
model_manager = ModelManager(de_spec)

n_global = 3
n_segment_specific = 4
min_length = 10
data_indices = [1]

data_M = reshape(Float64.(generator_temperatures), 1, :)

# ---------- helpers ----------
function load_params(csv_path)
    df = CSV.read(csv_path, DataFrame)
    return Float64.(df.value)
end

function load_cps(csv_path)
    df = CSV.read(csv_path, DataFrame)
    return Int.(df.cp)
end

function make_turbine_plot(cps, params; title_suffix="")
    sim, _ = simulate_full_model(params, cps, parnames,
                                 n_global, n_segment_specific,
                                 model_manager, data_M;
                                 plot_results=false,
                                 data_indices=data_indices)
    plt = plot(size=(1200, 500), dpi=200,
               plot_title="Wind-turbine generator front bearing" *
                          (isempty(title_suffix) ? "" : " — $title_suffix"),
               plot_titlefontsize=13)
    plot!(plt, times, data_M[1, :], label="Data", lw=0.8, color=:dodgerblue, alpha=0.7)
    plot!(plt, times, sim[1, :], label="Simulation", lw=1.4, color=:orangered)
    for (j, cp) in enumerate(cps)
        vline!(plt, [times[cp]], color=:green, lw=1.5, linestyle=:dash,
               label=(j == 1 ? "Change points" : false))
    end
    xlabel!(plt, "Date")
    ylabel!(plt, "Temperature (°C)")
    plot!(plt, xrotation=45)
    plot!(plt, legend=:topleft, legendfontsize=8)
    return plt
end

# ---------- main visualizations ----------
for obj in ["bic", "mdl", "aic"]
    cps = load_cps(joinpath(OUT_DIR, "turbine_detected_cps_$(obj).csv"))
    params = load_params(joinpath(OUT_DIR, "turbine_params_$(obj).csv"))
    plt = make_turbine_plot(cps, params; title_suffix=uppercase(obj))
    out = joinpath(OUT_DIR, "turbine_visualization_$(obj).png")
    savefig(plt, out)
    println("Saved: $out  (#CPs=$(length(cps)), CPs=$cps)")
end

println("\nAll wind-turbine visualizations saved to $OUT_DIR")
