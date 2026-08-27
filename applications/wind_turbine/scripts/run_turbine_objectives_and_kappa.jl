# ============================================================
# Wind-turbine application: BIC/MDL/AIC + κ sensitivity
#
# Based on Mica.jl/examples/Wind_Turbine_model/_example_DE.jl
# Output: results/turbine_objectives_summary.json
#         results/turbine_detected_cps_<obj>.csv
#         results/turbine_kappa_sensitivity.json / .csv
# ============================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Random
using JSON

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Wind_Turbine_model")
const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results")

function parameter_labels(parnames, n_global, n_segment_specific, n_cps)
    labels = String[]
    for i in 1:n_global
        push!(labels, string(parnames[i]) * "_global")
    end
    n_seg_total = n_cps + 1
    for s in 1:n_seg_total
        for i in 1:n_segment_specific
            push!(labels, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    return labels
end

# ---------- model definition ----------
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

# ---------- load data ----------
df = CSV.read(joinpath(EXAMPLE_DIR, "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv"), DataFrame)
wind_speeds = df[:, "Wind speed (m/s)"][1:2500]
ambient_temperatures = df[:, "Ambient temperature (converter) (°C)"][1:2500]
generator_temperatures_front = df[:, "Generator bearing front temperature (°C)"][1:2500]
generator_temperatures_rear = df[:, "Generator bearing rear temperature (°C)"][1:2500]
# Match original example: front data only
generator_temperatures = generator_temperatures_front

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
stepp = 10

ga = GA(populationSize=50, selection=tournament(2), crossover=SBX(0.7, 1),
        mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))

data_M = reshape(Float64.(generator_temperatures), 1, :)
n = length(data_M)

# ---------- BIC / MDL / AIC ----------
summary = []
for obj in [:bic, :mdl, :aic]
    println("\nRunning turbine with objective = $obj")
    Random.seed!(1234)
    t0 = time()
    detected_cp, params = detect_changepoints(
        objective_function, n, n_global, n_segment_specific,
        model_manager, loss_function, data_M,
        copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
        EvolutionaryOptimizer(ga, options=Evolutionary.Options(show_trace=false, iterations=100)),
        min_length, stepp;
        objective_type=obj,
        penalty_fn=(p, n_obs) -> 0.0,
        verbose=false, animate=false
    )
    elapsed = time() - t0
    detected_cp = filter(c -> c > min_length && c < n - min_length, detected_cp)
    detected_cp = sort(unique(detected_cp))

    CSV.write(joinpath(OUT_DIR, "turbine_detected_cps_$(obj).csv"),
              DataFrame(objective=string(obj), cp=detected_cp))
    param_labs = parameter_labels(parnames, n_global, n_segment_specific, length(detected_cp))
    CSV.write(joinpath(OUT_DIR, "turbine_params_$(obj).csv"),
              DataFrame(parameter=param_labs, value=params[1:length(param_labs)]))
    push!(summary, Dict(
        "objective" => string(obj),
        "n_cps" => length(detected_cp),
        "cps" => detected_cp,
        "time_seconds" => elapsed
    ))
    println("  #CPs = $(length(detected_cp)), CPs = $detected_cp")
end

open(joinpath(OUT_DIR, "turbine_objectives_summary.json"), "w") do f
    JSON.print(f, summary, 2)
end
println("\nTurbine objectives summary saved to $(joinpath(OUT_DIR, "turbine_objectives_summary.json"))")

# ---------- κ sensitivity ----------
kappa_values = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
kappa_results = []
for κ in kappa_values
    println("\nRunning turbine with κ = $κ")
    pen_fn = (p, n) -> κ * p * log(n)
    Random.seed!(1234)
    detected_cp, params = detect_changepoints(
        objective_function, n, n_global, n_segment_specific,
        model_manager, loss_function, data_M,
        copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
        EvolutionaryOptimizer(ga, options=Evolutionary.Options(show_trace=false, iterations=100)),
        min_length, stepp;
        objective_type=:penalty,
        penalty_fn=pen_fn,
        verbose=false, animate=false
    )
    detected_cp = filter(c -> c > min_length && c < n - min_length, detected_cp)
    detected_cp = sort(unique(detected_cp))
    push!(kappa_results, Dict("kappa" => κ, "n_cps" => length(detected_cp), "cps" => detected_cp))
    println("  #CPs = $(length(detected_cp))")
end

open(joinpath(OUT_DIR, "turbine_kappa_sensitivity.json"), "w") do f
    JSON.print(f, kappa_results, 2)
end
CSV.write(joinpath(OUT_DIR, "turbine_kappa_sensitivity.csv"),
          DataFrame(kappa=[r["kappa"] for r in kappa_results],
                    n_cps=[r["n_cps"] for r in kappa_results]))
println("Turbine κ sensitivity saved to $(joinpath(OUT_DIR, "turbine_kappa_sensitivity.json"))")
