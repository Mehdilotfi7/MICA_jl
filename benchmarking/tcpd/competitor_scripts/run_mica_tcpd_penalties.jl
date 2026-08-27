# ============================================================
# Benchmark MICA.jl on the toy datasets using TCPD-style penalties.
#
# Mirrors the structure of run_mica.jl but replaces the original :bic/:mdl/:aic
# objectives with the exact TCPD reference-package penalties implemented in
# Mica.jl/src/penalty.jl.
#
# Usage: julia run_mica_tcpd_penalties.jl <dataset_index>
#   dataset_index: 1..9 (matches toy_datasets.json order)
#
# Input:  MICA/benchmarking/data/toy_datasets.json
# Output: MICA/benchmarking/results/benchmark_toydatasets_mica_tcpd_part<idx>.json
# ============================================================

using Pkg
const _TOY_MICA_PROJECT = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
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
    manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn =
        default_model_setup(y, n, "linear")
    return manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_rss
end

function build_ar_setup(y::Vector{Float64}, n::Int)
    manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn =
        default_model_setup(y, n, "ar1")
    return manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_rss
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

# ---- Objective list (matches COVID/turbine TCPD penalty runs) ----------

const TCPD_OBJECTIVES = [
    :bic, :mdl, :aic,
    :tcpd_bic, :tcpd_mbic, :tcpd_aic, :tcpd_hannan_quinn, :tcpd_sic, :tcpd_none,
    :tcpd_bic0, :tcpd_aic0, :tcpd_hannan_quinn0,
    :wbs_bic, :wbs_mbic, :wbs_ssic,
    :rfpop_l1, :rfpop_l2, :rfpop_huber, :rfpop_outlier
]

# ---- Single run -----------------------------------------------------

function run_instance(model::String, y::Vector{Float64}, n::Int, γ_true::Union{Float64,Nothing},
                      objective_type::Symbol)
    manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn =
        build_model_setup(model, y, n, γ_true)

    data = reshape(y, 1, :)
    min_length = 10
    step       = 10

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

    return cps, params, obj_val
end

# ---- Main per-instance logic ----------------------------------------

function run_dataset(idx::Int, datasets::Vector{Dict{String,Any}})
    ds = datasets[idx]
    model = ds["model"]
    y = Float64.(ds["y"])
    n = length(y)
    true_cps = Float64.(ds["true_cps"])
    γ_true = get(ds, "gamma", nothing)

    println("\n[$idx/$(length(datasets))] $model | seed=$(ds["seed"]) | noise=$(ds["noise_level"]) | n=$n")

    records = Dict{String,Any}[]
    for obj in TCPD_OBJECTIVES
        try
            cps, params, obj_val = run_instance(model, y, n, γ_true, obj)
            p, r, f = f1_score(cps, true_cps; tolerance=TOL)
            cov = covering_score(cps, true_cps, n)

            rec = Dict(
                "model" => model,
                "seed" => ds["seed"],
                "noise_level" => ds["noise_level"],
                "n" => n,
                "true_cps" => true_cps,
                "objective" => string(obj),
                "cps" => cps,
                "f1" => f,
                "precision" => p,
                "recall" => r,
                "covering" => cov,
                "obj_value" => obj_val,
                "n_cps" => length(cps)
            )
            push!(records, rec)
            println("  $obj: F1=$(round(f, digits=3)) CPs=$cps obj=$(round(obj_val, digits=2))")
        catch e
            println("  $obj FAILED: $e")
            push!(records, Dict(
                "model" => model,
                "seed" => ds["seed"],
                "noise_level" => ds["noise_level"],
                "n" => n,
                "true_cps" => true_cps,
                "objective" => string(obj),
                "cps" => Int[],
                "f1" => 0.0,
                "precision" => 0.0,
                "recall" => 0.0,
                "covering" => 0.0,
                "obj_value" => nothing,
                "n_cps" => 0,
                "error" => string(e)
            ))
        end
    end

    # Practical = lowest objective value; Oracle = highest F1
    valid = filter(r -> !haskey(r, "error"), records)
    flat_records = Dict{String,Any}[]

    if !isempty(valid)
        best_prac = valid[argmin([r["obj_value"] for r in valid])]
        best_oracle = valid[argmax([r["f1"] for r in valid])]

        push!(flat_records, Dict(
            "model" => model,
            "seed" => ds["seed"],
            "noise_level" => ds["noise_level"],
            "n" => n,
            "true_cps" => true_cps,
            "method" => "MICA-P-TCPD",
            "config" => "default",
            "objective" => best_prac["objective"],
            "f1" => best_prac["f1"],
            "precision" => best_prac["precision"],
            "recall" => best_prac["recall"],
            "covering" => best_prac["covering"],
            "cps" => best_prac["cps"],
            "obj_value" => best_prac["obj_value"]
        ))
        push!(flat_records, Dict(
            "model" => model,
            "seed" => ds["seed"],
            "noise_level" => ds["noise_level"],
            "n" => n,
            "true_cps" => true_cps,
            "method" => "MICA-O-TCPD",
            "config" => "oracle",
            "objective" => best_oracle["objective"],
            "f1" => best_oracle["f1"],
            "precision" => best_oracle["precision"],
            "recall" => best_oracle["recall"],
            "covering" => best_oracle["covering"],
            "cps" => best_oracle["cps"],
            "obj_value" => best_oracle["obj_value"]
        ))

        println("  Practical: $(best_prac["objective"]) F1=$(round(best_prac["f1"], digits=3))")
        println("  Oracle:    $(best_oracle["objective"]) F1=$(round(best_oracle["f1"], digits=3))")
    else
        push!(flat_records, Dict(
            "model" => model, "seed" => ds["seed"], "noise_level" => ds["noise_level"],
            "n" => n, "true_cps" => true_cps, "method" => "MICA-P-TCPD", "config" => "default",
            "objective" => nothing, "f1" => 0.0, "precision" => 0.0, "recall" => 0.0,
            "covering" => 0.0, "cps" => Int[], "obj_value" => nothing
        ))
        push!(flat_records, Dict(
            "model" => model, "seed" => ds["seed"], "noise_level" => ds["noise_level"],
            "n" => n, "true_cps" => true_cps, "method" => "MICA-O-TCPD", "config" => "oracle",
            "objective" => nothing, "f1" => 0.0, "precision" => 0.0, "recall" => 0.0,
            "covering" => 0.0, "cps" => Int[], "obj_value" => nothing
        ))
    end

    return records, flat_records
end

# ---- Single-objective run --------------------------------------------

function run_single_objective(idx::Int, obj_idx::Int, datasets::Vector{Dict{String,Any}})
    obj_idx = clamp(obj_idx, 1, length(TCPD_OBJECTIVES))
    obj = TCPD_OBJECTIVES[obj_idx]

    ds = datasets[idx]
    model = ds["model"]
    y = Float64.(ds["y"])
    n = length(y)
    true_cps = Float64.(ds["true_cps"])
    γ_true = get(ds, "gamma", nothing)

    println("\n[$idx/$(length(datasets))] $model | seed=$(ds["seed"]) | noise=$(ds["noise_level"]) | n=$n | objective=$obj")

    rec = try
        cps, params, obj_val = run_instance(model, y, n, γ_true, obj)
        p, r, f = f1_score(cps, true_cps; tolerance=TOL)
        cov = covering_score(cps, true_cps, n)
        println("  $obj: F1=$(round(f, digits=3)) CPs=$cps obj=$(round(obj_val, digits=2))")
        Dict(
            "model" => model,
            "seed" => ds["seed"],
            "noise_level" => ds["noise_level"],
            "n" => n,
            "true_cps" => true_cps,
            "objective" => string(obj),
            "cps" => cps,
            "f1" => f,
            "precision" => p,
            "recall" => r,
            "covering" => cov,
            "obj_value" => obj_val,
            "n_cps" => length(cps)
        )
    catch e
        println("  $obj FAILED: $e")
        Dict(
            "model" => model,
            "seed" => ds["seed"],
            "noise_level" => ds["noise_level"],
            "n" => n,
            "true_cps" => true_cps,
            "objective" => string(obj),
            "cps" => Int[],
            "f1" => 0.0,
            "precision" => 0.0,
            "recall" => 0.0,
            "covering" => 0.0,
            "obj_value" => nothing,
            "n_cps" => 0,
            "error" => string(e)
        )
    end

    return rec
end

# ---- Main entry point ------------------------------------------------

function main()
    idx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
    obj_idx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0

    datasets = convert(Vector{Dict{String,Any}}, JSON.parsefile(joinpath(@__DIR__, "toy_datasets.json")))
    idx = clamp(idx, 1, length(datasets))

    out_dir = joinpath(@__DIR__, "..", "..", "..", "results")
    mkpath(out_dir)

    if obj_idx == 0
        # Run all 19 objectives sequentially (for testing / merging later)
        records, flat_records = run_dataset(idx, datasets)
        part_file = joinpath(out_dir, "benchmark_toydatasets_mica_tcpd_part$idx.json")
        open(part_file, "w") do f
            JSON.print(f, Dict(
                "dataset_index" => idx,
                "records" => records,
                "flat_records" => flat_records
            ), 2)
        end
        println("\nPart $idx written to $part_file")
    else
        rec = run_single_objective(idx, obj_idx, datasets)
        obj_file = joinpath(out_dir, "benchmark_toydatasets_mica_tcpd_part$(idx)_obj$(obj_idx).json")
        open(obj_file, "w") do f
            JSON.print(f, rec, 2)
        end
        println("\nObjective $obj_idx for dataset $idx written to $obj_file")
    end
end

main()
