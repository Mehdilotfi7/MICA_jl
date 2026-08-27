#!/usr/bin/env julia
# ============================================================
# Run the wind-turbine application with one TCPD-style penalty.
#
# Usage:
#   julia run_turbine_tcpd_penalties.jl <obj_idx>
#
# obj_idx is 1..19 and selects one of the implemented objective types.
# Output root: publication/applications/wind_turbine/results/tcpd_pen
# ============================================================

using Pkg
const _TURBINE_MICA_PROJECT = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
const _TURBINE_EXPECTED_PROJECT = abspath(joinpath(_TURBINE_MICA_PROJECT, "Project.toml"))
if Base.active_project() === nothing || abspath(Base.active_project()) != _TURBINE_EXPECTED_PROJECT
    Pkg.activate(_TURBINE_MICA_PROJECT)
end

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Random
using JSON

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Wind_Turbine_model")
const OUT_ROOT = joinpath(@__DIR__, "..", "..", "..", "results", "tcpd_pen")
mkpath(OUT_ROOT)

# ---------- objective type list ----------
const OBJ_TYPES = [
    :bic, :mdl, :aic,
    :tcpd_bic, :tcpd_mbic, :tcpd_aic, :tcpd_hannan_quinn, :tcpd_sic, :tcpd_none,
    :tcpd_bic0, :tcpd_aic0, :tcpd_hannan_quinn0,
    :wbs_bic, :wbs_mbic, :wbs_ssic,
    :rfpop_l1, :rfpop_l2, :rfpop_huber, :rfpop_outlier
]

const OBJ_LABELS = string.(OBJ_TYPES)

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

const SEED = 1234
opt = EvolutionaryOptimizer(ga,
        options=Evolutionary.Options(show_trace=false), seed=SEED)

ga_settings = Dict(
    "populationSize" => 50,
    "selection" => "tournament(2)",
    "crossover" => "SBX(0.7, 1)",
    "mutationRate" => 0.7,
    "crossoverRate" => 0.7,
    "mutation" => "gaussian(0.0001)",
    "iterations" => 100,
    "min_length" => min_length,
    "step" => stepp
)

data_M = reshape(Float64.(generator_temperatures), 1, :)
n = length(data_M)

# ---------- parse objective index ----------
obj_idx = parse(Int, ARGS[1])
if obj_idx < 1 || obj_idx > length(OBJ_TYPES)
    error("obj_idx must be between 1 and $(length(OBJ_TYPES)); got $(obj_idx)")
end
obj_type = OBJ_TYPES[obj_idx]
label = OBJ_LABELS[obj_idx]

println("[Wind turbine TCPD penalties] obj_idx=$(obj_idx), label=$(label), objective=$(obj_type)")
flush(stdout)

OUT_DIR = joinpath(OUT_ROOT, "turbine_$(label)")
mkpath(OUT_DIR)

result = Dict{String,Any}(
    "penalty_label" => label,
    "objective" => string(obj_type),
    "error" => nothing
)

try
    Random.seed!(SEED)
    t0 = time()
    detected_cp, params = detect_changepoints(
        objective_function, n, n_global, n_segment_specific,
        model_manager, loss_function, data_M,
        copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
        opt,
        min_length, stepp;
        objective_type=obj_type,
        penalty_fn=(p, n_obs) -> 0.0,
        verbose=false, animate=false
    )
    elapsed = time() - t0

    detected_cp = filter(c -> c > min_length && c < n - min_length, detected_cp)
    detected_cp = sort(unique(detected_cp))

    n_cps = length(detected_cp)

    cp_file = joinpath(OUT_DIR, "turbine_detected_cps_$(label).csv")
    CSV.write(cp_file, DataFrame(objective=label, cp=detected_cp))

    param_labs = parameter_labels(parnames, n_global, n_segment_specific, n_cps)
    if length(params) < length(param_labs)
        @warn "  $(label): chromosome length $(length(params)) < expected $(length(param_labs)); padding with NaN"
        params = [params; fill(NaN, length(param_labs) - length(params))]
    end
    param_file = joinpath(OUT_DIR, "turbine_params_$(label).csv")
    CSV.write(param_file, DataFrame(parameter=param_labs, value=params[1:length(param_labs)]))

    result["n_cps"] = n_cps
    result["cps"] = detected_cp
    result["time_seconds"] = elapsed
    result["ga_settings"] = ga_settings
    result["seed"] = SEED
    result["min_length"] = min_length
    result["step"] = stepp

    println("  #CPs = $(n_cps), CPs = $(detected_cp), time = $(round(elapsed,digits=1))s")
    flush(stdout)

catch e
    bt = catch_backtrace()
    err_msg = sprint(showerror, e, bt)
    result["error"] = err_msg
    @error "  $(label) failed: $(err_msg)"
    flush(stderr)
end

summary_file = joinpath(OUT_DIR, "summary.json")
open(summary_file, "w") do f
    JSON.print(f, result, 2)
end
println("  summary -> $(summary_file)")
flush(stdout)

println("\n[Wind turbine TCPD penalties] Done for $(label)")
