#!/usr/bin/env julia
# Task B alternative — channel-mean-normalized raw loss.
# Each data stream is divided by its mean before computing absolute error,
# so all five channels contribute on a comparable scale.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using Evolutionary, OrdinaryDiffEq
using Smoothers
using Random, JSON
using LabelledArrays

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_B", "results_normmean")
mkpath(OUT_DIR)

const SEED = 1234

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

const channel_means = [mean(data_CP[k, :]) for k in 1:5]

function loss_function(observed, simulated)
    if any(isnan, simulated)
        return Inf
    end
    total = 0.0
    for k in 1:5
        sim_k = simulated[[5, 6, 7, 9, 11][k], :]
        obs_k = observed[k, :]
        total += sum(abs, sim_k ./ channel_means[k] .- obs_k ./ channel_means[k])
    end
    return total
end

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
min_length = 10
step = 10
data_indices = [5, 6, 7, 9, 11]
n = size(data_CP, 2)

ga = GA(populationSize=150, selection=tournament(2), crossover=SBX(0.7, 1),
        mutationRate=0.7, crossoverRate=0.7, mutation=gaussian(0.0001))
opt = EvolutionaryOptimizer(ga,
        options=Evolutionary.Options(show_trace=false, iterations=1000), seed=SEED)

zero_penalty(p, n) = 0.0

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
    u0_curr = u0
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

println("\nRunning COVID with channel-mean-normalized raw loss (normmean)")
Random.seed!(SEED)
t0 = time()
detected_cp, params = detect_changepoints(
    objective_function, n, n_global, n_segment_specific,
    model_manager, loss_function, data_CP,
    copy(initial_chromosome), parnames, (copy(bounds[1]), copy(bounds[2])),
    opt,
    min_length, step;
    objective_type=:penalty,
    penalty_fn=zero_penalty,
    data_indices=data_indices,
    verbose=false, animate=false
)
elapsed = time() - t0
detected_cp = sort(unique(detected_cp))

n_cps = length(detected_cp)
expected_len = n_global + (n_cps + 1) * n_segment_specific
if length(params) > expected_len
    params = params[1:expected_len]
end

CSV.write(joinpath(OUT_DIR, "covid_detected_cps_origset_normmean.csv"),
          DataFrame(variant="normmean", cp=detected_cp))
CSV.write(joinpath(OUT_DIR, "covid_params_origset_normmean.csv"),
          DataFrame(parameter=parameter_labels(n_cps), value=params))

# Compute losses for the summary
sim = simulate_with_cps(detected_cp, params)
normmean_loss_val = loss_function(data_CP, sim)
eq_log_loss = equal_weight_log_loss(sim)

summary = Dict{String,Any}(
    "variant" => "normmean",
    "objective" => "penalty",
    "penalty_label" => "zero",
    "n_cps" => n_cps,
    "cps" => detected_cp,
    "time_seconds" => elapsed,
    "seed" => SEED,
    "loss" => normmean_loss_val,
    "normmean_loss" => normmean_loss_val,
    "equal_weight_log_loss" => eq_log_loss,
    "data_indices" => data_indices,
    "min_length" => min_length,
    "step" => step
)
open(joinpath(OUT_DIR, "summary.json"), "w") do f
    JSON.print(f, summary, 2)
end
println("  #CPs = $(n_cps), CPs = $detected_cp, time = $(round(elapsed,digits=1))s")
