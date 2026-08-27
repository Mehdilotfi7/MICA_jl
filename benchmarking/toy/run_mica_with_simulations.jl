#!/usr/bin/env julia
# ============================================================
# Run MICA.jl on toy datasets with all TCPD-style penalties and
# save the fitted simulations for downstream visualization.
#
# Usage: julia run_mica_with_simulations.jl <dataset_index>
#   dataset_index: 1..9 (matches toy_datasets.json order)
#
# Input:  MICA/benchmarking/Toy/data/toy_datasets.json
# Output: MICA/benchmarking/Toy/simulations/<model>/
#         ds<idx>_<objective>.png  (one figure per objective)
#         ds<idx>_<objective>.json (fitted CPs, params, simulation)
# ============================================================

using Pkg
const _TOY_MICA_PROJECT = joinpath(@__DIR__, "..", "..", "codes", "Mica.jl")
const _TOY_EXPECTED_PROJECT = abspath(joinpath(_TOY_MICA_PROJECT, "Project.toml"))
if Base.active_project() === nothing || abspath(Base.active_project()) != _TOY_EXPECTED_PROJECT
    Pkg.activate(_TOY_MICA_PROJECT)
end

using Mica
using JSON
using Random
using Statistics
using OrdinaryDiffEq
using Evolutionary
using Optim
using Printf

const TOL = 5

# ---- Loss functions -------------------------------------------------

loss_ode(observed::AbstractMatrix{Float64}, simulated::AbstractMatrix{Float64}) =
    sqrt(sum((observed .- simulated[2:2, :]).^2))

loss_rss(observed::AbstractMatrix{Float64}, simulated::AbstractMatrix{Float64}) =
    sqrt(sum((observed .- simulated).^2))

# ---- Metrics --------------------------------------------------------

function f1_score(detected::Vector{Int}, true_cps::Vector{Float64}; tolerance::Int=TOL)
    true_int = Int.(round.(true_cps))
    tp = 0
    used = falses(length(true_int))
    for d in detected
        for (i, t) in enumerate(true_int)
            if abs(d - t) <= tolerance && !used[i]
                tp += 1
                used[i] = true
                break
            end
        end
    end
    precision = length(detected) > 0 ? tp / length(detected) : (length(true_int) == 0 ? 1.0 : 0.0)
    recall    = length(true_int) > 0 ? tp / length(true_int) : 1.0
    f1        = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    return precision, recall, f1
end

function covering_score(detected::Vector{Int}, true_cps::Vector{Float64}, n::Int)
    length(true_cps) == 0 && return length(detected) == 0 ? 1.0 : 0.0
    detected = sort(detected)
    seg_boundaries = [0; detected; n]
    true_seg = sort(unique([0; Int.(round.(true_cps)); n]))
    score = 0.0
    for i in 1:(length(seg_boundaries)-1)
        a, b = seg_boundaries[i], seg_boundaries[i+1]
        best_overlap = 0.0
        for j in 1:(length(true_seg)-1)
            ta, tb = true_seg[j], true_seg[j+1]
            best_overlap = max(best_overlap, max(0.0, min(b, tb) - max(a, ta)))
        end
        score += best_overlap
    end
    return score / n
end

# ---- Model builders -------------------------------------------------

function build_ode_setup(y::Vector{Float64}, n::Int, γ_true::Float64)
    function sir_model!(du, u, p, t)
        S, I, R = u
        du[1] = -p.β * S * I
        du[2] =  p.β * S * I - p.γ * I
        du[3] =  p.γ * I
    end

    function sir_solver(params, tspan::Tuple{Float64,Float64}, u0::Vector{Float64})
        prob = ODEProblem(sir_model!, u0, tspan, params)
        sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
        return hcat(sol.u...)
    end

    n_global = 1
    n_seg    = 1
    chrom    = [γ_true, 0.0002]
    lower    = [0.1, 0.0]
    upper    = [0.9, 0.0005]
    parnames = (:γ, :β)

    spec = ODEModelSpec(
        sir_solver,
        Dict(:γ => γ_true, :β => 0.0002),
        [9999.0, 1.0, 0.0],
        (0.0, Float64(n))
    )
    manager = ModelManager(spec)

    ga = GA(
        populationSize=60,
        selection=uniformranking(15),
        crossover=MILX(0.01, 0.17, 0.5),
        mutationRate=0.3,
        crossoverRate=0.6,
        mutation=gaussian(0.00005)
    )
    opt = EvolutionaryOptimizer(ga, options=Evolutionary.Options(show_trace=false, iterations=40))

    return manager, n_global, n_seg, chrom, (lower, upper), parnames, opt, loss_ode
end

function build_lr_setup(y::Vector{Float64}, n::Int)
    manager, n_global, n_seg, chrom, bounds, parnames, _, loss_fn =
        default_model_setup(y, n, "linear_slope_only"; continuity=true)

    ga = GA(
        populationSize=100,
        selection=uniformranking(25),
        crossover=MILX(0.01, 0.17, 0.5),
        mutationRate=0.3,
        crossoverRate=0.6,
        mutation=gaussian(0.00005)
    )
    optimizer = EvolutionaryOptimizer(ga, options=Evolutionary.Options(show_trace=false, iterations=100))

    return manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn
end

function build_ar_setup(y::Vector{Float64}, n::Int)
    manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn =
        default_model_setup(y, n, "ar1_nodrift")
    return manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn
end

function build_model_setup(model::String, y::Vector{Float64}, n::Int, γ_true::Union{Float64,Nothing})
    if model == "ODE"
        return build_ode_setup(y, n, γ_true)
    elseif model == "LR"
        return build_lr_setup(y, n)
    elseif model == "AR"
        return build_ar_setup(y, n)
    else
        error("Unknown model: $model")
    end
end

# ---- Objective list -------------------------------------------------

const TCPD_OBJECTIVES = [
    :bic, :mdl, :aic,
    :tcpd_bic, :tcpd_mbic, :tcpd_aic, :tcpd_hannan_quinn, :tcpd_sic, :tcpd_none,
    :tcpd_bic0, :tcpd_aic0, :tcpd_hannan_quinn0,
    :wbs_bic, :wbs_mbic, :wbs_ssic,
    :rfpop_l1, :rfpop_l2, :rfpop_huber, :rfpop_outlier
]

# ---- Single run -----------------------------------------------------

function run_instance(model::String, y::Vector{Float64}, n::Int, γ_true::Union{Float64,Nothing},
                      objective_type::Symbol, min_length::Int, step::Int)
    manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn =
        build_model_setup(model, y, n, γ_true)

    data = reshape(y, 1, :)

    cps, params = detect_changepoints(
        objective_function,
        n, n_global, n_seg,
        manager,
        loss_fn,
        data,
        copy(chrom),
        parnames,
        (copy(bounds[1]), copy(bounds[2])),
        optimizer,
        min_length, step;
        objective_type=objective_type,
        penalty_fn=(p, n) -> 0.0,
        verbose=false,
        animate=false
    )

    cps = filter(c -> c > min_length && c < n - min_length, cps)
    cps = sort(unique(cps))

    obj_val = objective_function(params, cps, parnames, n_global, n_seg, manager, loss_fn, data)

    # Generate fitted simulation (extract the observed channel)
    sim_mat, _ = simulate_full_model(params, cps, parnames, n_global, n_seg, manager, data)
    # ODE SIR model returns 3 states per time point; the observed/infected channel is row 2.
    # LR and AR models return a single channel per time point (row 1).
    sim_row = model == "ODE" ? 2 : 1
    sim = vec(sim_mat[sim_row:sim_row, :])

    return cps, params, obj_val, sim
end

# ---- Main per-dataset logic -----------------------------------------

function run_dataset(idx::Int, datasets::Vector{Dict{String,Any}},
                     objective::Symbol, min_length::Int, step::Int)
    ds = datasets[idx]
    model = ds["model"]
    y = Float64.(ds["y"])
    n = length(y)
    true_cps = Float64.(ds["true_cps"])
    γ_true = get(ds, "gamma", nothing)

    println("\n[$idx/$(length(datasets))] $model | seed=$(ds["seed"]) | noise=$(ds["noise_level"]) | n=$n | obj=$objective | min_length=$min_length | step=$step")

    out_dir = joinpath(@__DIR__, "simulations", model)
    mkpath(out_dir)

    try
        cps, params, obj_val, sim = run_instance(model, y, n, γ_true, objective, min_length, step)
        p, r, f = f1_score(cps, true_cps; tolerance=TOL)
        cov = covering_score(cps, true_cps, n)

        rec = Dict(
            "model" => model,
            "seed" => ds["seed"],
            "noise_level" => ds["noise_level"],
            "n" => n,
            "true_cps" => true_cps,
            "objective" => string(objective),
            "min_length" => min_length,
            "step" => step,
            "cps" => cps,
            "params" => params,
            "f1" => f,
            "precision" => p,
            "recall" => r,
            "covering" => cov,
            "obj_value" => obj_val,
            "n_cps" => length(cps),
            "simulation" => sim
        )
        println("  $objective: F1=$(round(f, digits=3)) CPs=$cps obj=$(round(obj_val, digits=2))")

        # Write per-configuration JSON
        obj_file = joinpath(out_dir, @sprintf("ds%02d_%s_ml%d_s%d.json", idx-1, objective, min_length, step))
        open(obj_file, "w") do f
            JSON.print(f, rec, 2)
        end
        return rec
    catch e
        println("  $objective FAILED: $e")
        rec = Dict(
            "model" => model,
            "seed" => ds["seed"],
            "noise_level" => ds["noise_level"],
            "n" => n,
            "true_cps" => true_cps,
            "objective" => string(objective),
            "min_length" => min_length,
            "step" => step,
            "cps" => Int[],
            "params" => Float64[],
            "f1" => 0.0,
            "precision" => 0.0,
            "recall" => 0.0,
            "covering" => 0.0,
            "obj_value" => nothing,
            "n_cps" => 0,
            "error" => string(e)
        )
        return rec
    end
end

# ---- Main entry point -----------------------------------------------

function main()
    if length(ARGS) < 4
        println("Usage: julia run_mica_with_simulations.jl <dataset_index> <objective> <min_length> <step>")
        println("  dataset_index: 1..9 (matches toy_datasets.json order)")
        println("  objective: one of the TCPD objective symbols")
        println("  min_length: minimum segment length")
        println("  step: changepoint grid step")
        return
    end

    idx = parse(Int, ARGS[1])
    objective = Symbol(ARGS[2])
    min_length = parse(Int, ARGS[3])
    step = parse(Int, ARGS[4])

    datasets = convert(Vector{Dict{String,Any}}, JSON.parsefile(joinpath(@__DIR__, "data", "toy_datasets.json")))
    idx = clamp(idx, 1, length(datasets))

    run_dataset(idx, datasets, objective, min_length, step)
end

main()
