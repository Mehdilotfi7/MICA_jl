#!/usr/bin/env julia
# ============================================================
# PLOS-style visualization of wind-turbine generator-bearing
# temperature data, MICA simulation, and BIC/MDL/AIC change points.
#
# Produces:
#   ../results/fig6_turbine_bic_mdl_aic.png
#   ../results/fig6_turbine_bic_mdl_aic.eps
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

# ---------- PLOS-compliant defaults ----------
font_face = Plots.font("Arial")
default(
    fontfamily              = font_face.family,
    titlefont               = font_face,
    guidefont               = font_face,
    tickfont                = font_face,
    legendfont              = font_face,
    titlefontsize           = 12,
    guidefontsize           = 12,
    tickfontsize            = 10,
    legendfontsize          = 10,
    linewidth               = 1.5,
    markersize              = 5,
    markerstrokewidth       = 0,
    framestyle              = :box,
    grid                    = true,
    gridalpha               = 0.2,
    gridlinewidth           = 0.5,
    dpi                     = 300,
    size                    = (2250, 1200),
    margin                  = 5Plots.mm,
    label                   = "",
)

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

function make_turbine_plos_plot(cps, params)
    sim, _ = simulate_full_model(params, cps, parnames,
                                 n_global, n_segment_specific,
                                 model_manager, data_M;
                                 plot_results=false,
                                 data_indices=data_indices)
    plt = plot()
    plot!(plt, times, data_M[1, :],
          label="Observed temperature",
          lw=1.0, color=:dodgerblue, alpha=0.75)
    plot!(plt, times, sim[1, :],
          label="MICA simulation",
          lw=2.0, color=:orangered)
    for (j, cp) in enumerate(cps)
        vline!(plt, [times[cp]],
               color=:darkgreen, lw=2.0, linestyle=:dash,
               label=(j == 1 ? "Change points" : false))
    end
    xlabel!(plt, "Date")
    ylabel!(plt, "Generator front-bearing temperature (°C)")
    plot!(plt, xrotation=45)
    plot!(plt, legend=:topleft)
    return plt
end

# ---------- main figure (BIC; MDL/AIC are identical) ----------
cps = load_cps(joinpath(OUT_DIR, "turbine_detected_cps_bic.csv"))
params = load_params(joinpath(OUT_DIR, "turbine_params_bic.csv"))
plt = make_turbine_plos_plot(cps, params)

out_png = joinpath(OUT_DIR, "fig6_turbine_bic_mdl_aic.png")
out_pdf = joinpath(OUT_DIR, "fig6_turbine_bic_mdl_aic.pdf")
savefig(plt, out_png)
savefig(plt, out_pdf)
println("Saved: $out_png")
println("Saved: $out_pdf")
println("(#CPs=$(length(cps)), CPs=$cps)")
