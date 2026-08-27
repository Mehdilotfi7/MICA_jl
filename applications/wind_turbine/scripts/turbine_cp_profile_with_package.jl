#!/usr/bin/env julia
# Changepoint profile for the wind-turbine application using
# BreakpointProfiles.jl.
#
# Usage: julia turbine_cp_profile_with_package.jl <label> <cps_csv> <params_csv> <out_dir> [window]

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))
using Mica
using BreakpointProfiles
using CSV, DataFrames, Statistics, Random
using Evolutionary, OrdinaryDiffEq
using Printf, Dates

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Wind_Turbine_model")

const PLE_ITER = parse(Int, get(ENV, "TURBINE_PLE_ITER", "100"))
const PLE_POP = parse(Int, get(ENV, "TURBINE_PLE_POP", "80"))
const CP_WINDOW = parse(Int, get(ENV, "TURBINE_CP_WINDOW", "7"))
const CHISQ_95 = 3.8414588206941285

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

function loss_function(observed, simulated)
    return sum((observed .- simulated).^2)
end

function load_turbine_data()
    df = CSV.read(joinpath(EXAMPLE_DIR, "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv"), DataFrame)
    wind_speeds = df[:, "Wind speed (m/s)"][1:2500]
    ambient_temperatures = df[:, "Ambient temperature (converter) (°C)"][1:2500]
    generator_temperatures = df[:, "Generator bearing front temperature (°C)"][1:2500]
    return (wind_speeds, ambient_temperatures, generator_temperatures)
end

const parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7)
const n_global = 3
const n_segment_specific = 4
const initial_chromosome = [1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5]
const lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
const upper = [100.0, 100.0, 100.0, 466.0, 466.0, 466.0, 1466.0]

const wind_speeds, ambient_temperatures, generator_temperatures = load_turbine_data()
const u0 = generator_temperatures[1]
const num_steps = 2500
const data_M = reshape(Float64.(generator_temperatures), 1, :)
const n = length(data_M)

const de_spec = DifferenceModelSpec(example_difference_model, initial_chromosome, u0, num_steps,
                                    (wind_speeds, ambient_temperatures))
const model_manager = ModelManager(de_spec)

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

length(ARGS) >= 4 || error("Usage: julia turbine_cp_profile_with_package.jl <label> <cps_csv> <params_csv> <out_dir> [window]")
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

function mica_objective(params, cps)
    return Mica.objective_function(
        params, cps, parnames, n_global, n_segment_specific,
        model_manager, loss_function, data_M
    )
end

const best_loss = mica_objective(params_best, cps_best)
println("[$LABEL] Best loss (MICA objective) = $(round(best_loss, digits=4))"); flush(stdout)

const prob = ODEChangepointPLEProblem(
    objective = mica_objective,
    data = data_M,
    loss_fn = loss_function,
    changepoints = cps_best,
    best_params = params_best,
    best_loss = best_loss,
    lb = lb_full,
    ub = ub_full,
    param_names = param_labels,
    n_global = n_global,
    n_segment_specific = n_segment_specific,
    n_obs = n
)

const optimizer = EvolutionaryPLEConfig(population_size=PLE_POP, iterations=PLE_ITER, seed=1234)

function main()
    window = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : CP_WINDOW
    println("[$LABEL] CP profile with window = ±$window days"); flush(stdout)
    println("[$LABEL] CPs: $cps_best"); flush(stdout)

    cp_profiles = profile_all_changepoints(prob; window=window, optimizer=optimizer)
    threshold = best_loss + CHISQ_95
    cp_df = cp_summary(cp_profiles, threshold)

    write_cp_profiles(joinpath(OUT_DIR, "cp_profile_loss.csv"), cp_profiles)
    CSV.write(joinpath(OUT_DIR, "cp_profile_ci.csv"), cp_df)

    base_date = Date("2021-01-01")
    lines = String[]
    push!(lines, "# Changepoint profile intervals for wind-turbine `$(LABEL)`")
    push!(lines, "")
    push!(lines, "**Window:** ±$window steps")
    push!(lines, "**Original CPs:** $(cps_best)")
    push!(lines, "**Best loss:** $(round(best_loss, digits=4))")
    push!(lines, "**Threshold (Δloss ≤ $(CHISQ_95)):** $(round(threshold, digits=4))")
    push!(lines, "")
    push!(lines, "## Summary")
    push!(lines, "")
    push!(lines, "| cp # | original | CI lower | CI upper | identifiable |")
    push!(lines, "|---|---|---|---|---|")
    for r in eachrow(cp_df)
        push!(lines, @sprintf("| %d | %d | %d | %d | %s |",
                              r.cp_index, r.original_cp, r.ci_lower, r.ci_upper, r.identifiable))
    end
    push!(lines, "")
    push!(lines, "## Files")
    push!(lines, "- `cp_profile_loss.csv` — full profile losses")
    push!(lines, "- `cp_profile_ci.csv` — approximate 95% confidence intervals")
    open(joinpath(OUT_DIR, "cp_profile_report.md"), "w") do f
        write(f, join(lines, "\n") * "\n")
    end

    println("[$LABEL] CP profile done. Outputs in $OUT_DIR")
end

main()
