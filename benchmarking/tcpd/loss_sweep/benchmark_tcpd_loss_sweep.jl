# ============================================================
# TCPD Loss-Function Sweep Benchmark
# Usage: julia benchmark_tcpd_loss_sweep.jl <start_idx> <end_idx> <suffix>
#
# Tests multiple loss functions (rss, l1, huber, gaussian_nll) for all
# model families on the TCPD datasets, using BIC-based model selection.
# ============================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))
using Mica, JSON, Statistics

# ---- Parse args ----
args = length(ARGS) >= 3 ? ARGS : ["1", "31", "loss_sweep"]
start_idx = parse(Int, args[1])
end_idx = parse(Int, args[2])
suffix = args[3]

# ---- Data loading ----
DATASET_DIR = joinpath(@__DIR__, "..", "dataset", "datasets")
ANNOTATIONS_FILE = joinpath(@__DIR__, "..", "dataset", "annotations.json")
annotations = JSON.parsefile(ANNOTATIONS_FILE)

function load_series(series_name)
    data = JSON.parsefile(joinpath(DATASET_DIR, series_name, "$(series_name).json"))
    y_raw = data["series"][1]["raw"]
    y = Float64.(filter(x -> x !== nothing, y_raw))
    for i in 2:length(y)
        if !isfinite(y[i]); y[i] = y[i-1]; end
    end
    cp_sets = Vector{Vector{Int}}()
    if haskey(annotations, series_name)
        for (_, cp_list) in annotations[series_name]
            if cp_list !== nothing && length(cp_list) > 0
                cp_list_clean = filter(x -> x !== nothing, cp_list)
                if length(cp_list_clean) > 0
                    push!(cp_sets, Int.(cp_list_clean))
                end
            end
        end
    end
    return y, cp_sets
end

# ---- Metrics ----
function compute_f1(detected, cp_sets; margin=5)
    if length(cp_sets) == 0
        prec = length(detected) == 0 ? 1.0 : 0.0
        rec = 1.0
        f1 = length(detected) == 0 ? 1.0 : 0.0
        return (prec, rec, f1)
    end
    if length(detected) == 0
        return (0.0, 0.0, 0.0)
    end
    all_cp = sort(unique(vcat(cp_sets...)))
    tp_p = 0; used = Set{Int}()
    for d in detected
        for cp in all_cp
            if abs(d - cp) <= margin && cp ∉ used
                tp_p += 1; push!(used, cp); break
            end
        end
    end
    precision = tp_p / length(detected)
    recalls = Float64[]
    for cps in cp_sets
        tp_r = 0; used_d = Set{Int}()
        for cp in cps
            for d in detected
                if abs(d - cp) <= margin && d ∉ used_d
                    tp_r += 1; push!(used_d, d); break
                end
            end
        end
        push!(recalls, tp_r / max(length(cps), 1))
    end
    recall = mean(recalls)
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    return (precision, recall, f1)
end

function compute_covering(detected, cp_sets, n)
    if length(cp_sets) == 0
        return length(detected) == 0 ? 1.0 : 0.0
    end
    detected = sort(detected)
    seg_boundaries = [0; detected; n]
    scores = Float64[]
    for cps in cp_sets
        true_seg = sort(unique([0; cps; n]))
        seg_score = 0.0
        for i in 1:(length(seg_boundaries)-1)
            a, b = seg_boundaries[i], seg_boundaries[i+1]
            best_overlap = 0.0
            for j in 1:(length(true_seg)-1)
                ta, tb = true_seg[j], true_seg[j+1]
                overlap = max(0.0, min(b, tb) - max(a, ta))
                best_overlap = max(best_overlap, overlap)
            end
            seg_score += best_overlap
        end
        push!(scores, seg_score / n)
    end
    return mean(scores)
end

# ---- Run single config ----
function run_config(y, n, manager, n_global, n_seg, chrom, bounds, parnames,
                    objective_type, penalty_fn, min_seg, step, optimizer, loss_function)
    data = reshape(y, 1, :)
    t0 = time()
    cps, best_params = detect_changepoints(
        Mica.objective_function, n, n_global, n_seg,
        manager, loss_function,
        data, copy(chrom), parnames,
        (copy(bounds[1]), copy(bounds[2])), optimizer,
        min_seg, step;
        penalty_fn=penalty_fn, objective_type=objective_type,
        verbose=false, animate=false
    )
    elapsed = time() - t0
    cps = filter(c -> c > min_seg && c < n - min_seg, cps)
    cps = sort(unique(cps))
    return cps, best_params, elapsed
end

# ============================================================
# MAIN BENCHMARK
# ============================================================

models = ["mean", "linear", "quadratic", "cubic", "exponential", "log_linear", "power", "mean_drift",
          "ar1", "ar2", "ar3",
          "hyperbolic", "asymptotic_regression", "michaelis_menten", "weibull_growth",
          "hill_function", "log_logistic", "double_exponential", "rational",
          "debt_dynamics", "accelerator", "compound_growth",
          "poisson", "negbin", "ingarch",
          "ets_aaa", "ets_mmm"]

losses = Dict(
    "rss" => Mica.rss_loss,
    "l1" => Mica.l1_loss,
    "huber" => Mica.huber_loss,
)

# Use a single practical objective (BIC) and dynamic min_seg/step per dataset
obj_types = [:bic]
function dynamic_min_seg_step(n)
    min_seg = max(5, n ÷ 40)
    step = max(1, n ÷ 100)
    return min_seg, step
end

# ---- Load all datasets ----
dataset_names = String[]
for d in readdir(DATASET_DIR)
    json_path = joinpath(DATASET_DIR, d, "$(d).json")
    if isdir(joinpath(DATASET_DIR, d)) && isfile(json_path) && haskey(annotations, d)
        cp_sets = Vector{Vector{Int}}()
        for (_, cp_list) in annotations[d]
            if cp_list !== nothing && length(cp_list) > 0
                cp_clean = filter(x -> x !== nothing, cp_list)
                if length(cp_clean) > 0
                    push!(cp_sets, Int.(cp_clean))
                end
            end
        end
        if length(cp_sets) > 0
            push!(dataset_names, d)
        end
    end
end
sort!(dataset_names)
start_idx = clamp(start_idx, 1, length(dataset_names))
end_idx = clamp(end_idx, 1, length(dataset_names))
dataset_names = dataset_names[start_idx:end_idx]

println("TCPD Loss-Function Sweep [suffix=$(suffix)]")
println("Datasets: $(dataset_names)")
println("Models: $(length(models))")
println("Loss functions: $(keys(losses))")
println("Objectives: $(obj_types)")
flush(stdout)

all_results = Dict{String,Any}[]
run_count = 0

total_start = time()
for (d_idx, d_name) in enumerate(dataset_names)
    y, cp_sets = load_series(d_name)
    n = length(y)
    println("\n[$(d_idx)/$(length(dataset_names))] $(d_name) (n=$(n))")
    flush(stdout)

    for model_name in models
        for (loss_name, loss_fn) in losses
            # Build model manager with this loss
            manager, n_global, n_seg, chrom, bounds, parnames, opt = default_model_setup(
                y, n, model_name; loss_function=loss_fn
            )

            for obj_type in obj_types
                min_seg, step = dynamic_min_seg_step(n)
                try
                    pen_fn = (p, n) -> 0.0
                    cps, params, elapsed = run_config(
                        y, n, manager, n_global, n_seg, chrom, bounds, parnames,
                        obj_type, pen_fn, min_seg, step, opt, loss_fn
                    )
                    prec, rec, f1 = compute_f1(cps, cp_sets)
                    covering = compute_covering(cps, cp_sets, n)
                    push!(all_results, Dict(
                        "dataset" => d_name,
                        "model" => model_name,
                        "loss_function" => loss_name,
                        "objective" => string(obj_type),
                        "min_seg" => min_seg,
                        "step" => step,
                        "n" => n,
                        "cps" => cps,
                        "n_cps" => length(cps),
                        "precision" => prec,
                        "recall" => rec,
                        "f1" => f1,
                        "covering" => covering,
                        "time" => elapsed
                    ))
                    global run_count += 1
                catch e
                    println("  FAILED: $(model_name) / $(loss_name) / $(obj_type) / ms$(min_seg)_s$(step): $(e)")
                    flush(stdout)
                end
            end
        end
    end
    println("  completed | $(run_count) total runs so far")
    flush(stdout)
end

elapsed_total = time() - total_start
println("\nLoss sweep complete for suffix=$(suffix)")
println("Total runs: $(run_count)")
println("Total time: $(round(elapsed_total, digits=1))s")

out_file = "benchmark_tcpd_loss_sweep_$(suffix).json"
open(out_file, "w") do f
    JSON.print(f, all_results, 2)
end
println("Results saved to $(out_file)")
