using LabelledArrays
using DifferentialEquations
using Statistics
using Evolutionary
using Random
using CSV
using DataFrames
using Dates
using Mica

function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p.β, p.γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0::Vector{Float64})
    prob = ODEProblem(sirmodel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-6, reltol=1.0e-6)
    return sol[:, :]
end

function loss_function(observed, simulated)
    simulated = simulated[2:2, :]
    return sqrt(sum((observed .- simulated) .^ 2))
end

function generate_toy_dataset(beta_values, change_points, γ, u0, tspan, noise_level, noise)
    data_CP = Float64[]
    all_times = Float64[]
    u0_current = copy(u0)
    for i in 1:length(change_points)+1
        if i == 1
            tspan_segment = (0.0, change_points[i])
        elseif i == length(change_points)+1
            tspan_segment = (change_points[i-1]+1.0, tspan[2])
        else
            tspan_segment = (change_points[i-1]+1.0, change_points[i])
        end
        params = @LArray [beta_values[i], γ] (:β, :γ)
        prob = ODEProblem(sirmodel!, u0_current, tspan_segment, params)
        sol = solve(prob, saveat=1.0)
        append!(data_CP, sol[2, :] .+ noise_level .* noise(length(sol.t)))
        append!(all_times, sol.t)
        u0_current = sol.u[end]
    end
    return all_times, abs.(data_CP)
end

function calculate_precision(detected_cps, true_cps, tolerance=0)
    TP = 0
    FP = 0
    for detected_cp in detected_cps
        if any(abs(detected_cp - true_cp) <= tolerance for true_cp in true_cps)
            TP += 1
        else
            FP += 1
        end
    end
    FN = length(true_cps) - TP
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1_score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0
    return precision, recall, f1_score
end

function run_once(seed, beta_values, change_points, γ, u0, tspan, data_length, noise_level, noise_type, penalty)
    Random.seed!(seed)
    noise = noise_type == "Gaussian" ? randn : rand
    times, data = generate_toy_dataset(beta_values, change_points, γ, u0, (0.0, data_length), noise_level, noise)
    data_CP = reshape(Float64.(data), 1, :)
    n = length(data_CP)
    tspan_local = (0.0, data_length)
    initial_chromosome = [0.69, 0.0002]
    parnames = (:γ, :β)
    bounds = ([0.1, 0.0], [0.9, 0.1])
    ode_spec = Mica.ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan_local)
    model_manager = Mica.ModelManager(ode_spec)
    n_global = 1
    n_segment_specific = 1
    min_length = 10
    step = 10
    ga = GA(populationSize=150, selection=uniformranking(20), crossover=MILX(0.01, 0.17, 0.5),
            mutationRate=0.3, crossoverRate=0.6, mutation=gaussian(0.0001))
    opt = Mica.EvolutionaryOptimizer(ga)
    detected_cps, pars_cps = Mica.detect_changepoints(
        Mica.objective_function, n, n_global, n_segment_specific,
        model_manager, loss_function, data_CP,
        initial_chromosome, parnames, bounds, opt,
        min_length, step; penalty_fn=penalty
    )
    precision, recall, f1 = calculate_precision(detected_cps, change_points)
    return (seed=seed, precision=precision, recall=recall, f1=f1, detected_cps=detected_cps, pars=pars_cps)
end

function main()
    beta_values = [0.00009, 0.00014, 0.00025, 0.0005]
    change_points = [50.0, 100.0]
    γ = 0.7
    N = 10_000
    u0 = [N-1.0, 1.0, 0.0]
    tspan = (0.0, 250.0)
    data_length = 160
    noise_levels = [0.0, 1.0, 10.0, 20.0]
    noise_types = ["Gaussian"]
    penalties = [BIC_0, BIC_1, BIC_10, BIC_20, BIC_30, BIC_100]
    penalty_names = ["BIC_0", "BIC_1", "BIC_10", "BIC_20", "BIC_30", "BIC_100"]
    n_seeds = 20
    results = DataFrame(seed=Int[], noise_level=Float64[], noise_type=String[], penalty=String[],
                        precision=Float64[], recall=Float64[], f1=Float64[], detected_cps=String[])
    total = length(noise_levels) * length(noise_types) * length(penalties) * n_seeds
    counter = 0
    for noise_level in noise_levels, noise_type in noise_types, (penalty, penalty_name) in zip(penalties, penalty_names)
        for seed in 1:n_seeds
            counter += 1
            println("[$counter/$total] noise=$noise_level type=$noise_type penalty=$penalty_name seed=$seed")
            try
                r = run_once(seed, beta_values, change_points, γ, u0, tspan, data_length, noise_level, noise_type, penalty)
                push!(results, (r.seed, noise_level, noise_type, penalty_name, r.precision, r.recall, r.f1, string(r.detected_cps)))
            catch e
                println("  FAILED: $e")
                push!(results, (seed, noise_level, noise_type, penalty_name, 0.0, 0.0, 0.0, "[]"))
            end
        end
    end
    outdir = "benchmark/results_seed_robustness"
    mkpath(outdir)
    outfile = joinpath(outdir, "synthetic_seed_robustness_$(Dates.format(now(), "yyyyymmdd_HHMMSS")).csv")
    CSV.write(outfile, results)
    println("Results saved to $outfile")
end

BIC_0(p, n) = 0.0 * p * log(n)
BIC_1(p, n) = 1.0 * p * log(n)
BIC_10(p, n) = 10.0 * p * log(n)
BIC_20(p, n) = 20.0 * p * log(n)
BIC_30(p, n) = 30.0 * p * log(n)
BIC_100(p, n) = 100.0 * p * log(n)

main()
