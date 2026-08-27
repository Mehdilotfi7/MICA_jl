#!/usr/bin/env julia
# Fast smoke test for exact TCPD practical defaults.

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))
using Mica, JSON, Statistics

const RSS_LOSS = (obs, sim) -> sum((obs .- sim).^2)

const DATASET_DIR = "benchmarking/tcpd/dataset/datasets"
const ANNOTATIONS_FILE = "benchmarking/tcpd/dataset/annotations.json"
const annotations = JSON.parsefile(ANNOTATIONS_FILE)

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

function compute_f1(detected, cp_sets; margin=5)
    if length(cp_sets) == 0
        return length(detected) == 0 ? 1.0 : 0.0
    end
    if length(detected) == 0
        return 0.0
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
    return (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
end

function run_config(y, n, manager, n_global, n_seg, chrom, bounds, parnames,
                    objective_type; min_seg=1, step=1)
    data = reshape(y, 1, :)
    pen_fn = (p, n) -> 0.0
    optimizer = suggest_optimizer("mean", n_global + n_seg)
    cps, best_params = detect_changepoints(
        objective_function, n, n_global, n_seg,
        manager, RSS_LOSS,
        data, copy(chrom), parnames,
        (copy(bounds[1]), copy(bounds[2])), optimizer,
        min_seg, step;
        penalty_fn=pen_fn, objective_type=objective_type,
        verbose=false, animate=false
    )
    cps = filter(c -> c > min_seg && c < n - min_seg, cps)
    return sort(unique(cps))
end

const practical_defaults = [
    ("changepoint_mbic", :tcpd_mbic, 1, 1),
    ("changepoint_bic", :tcpd_bic, 1, 1),
    ("wbs_ssic", :wbs_ssic, 1, 1),
    ("wbs_bic", :wbs_bic, 1, 1),
    ("rfpop_outlier", :rfpop_outlier, 1, 1),
]

const models = ["mean", "linear", "ar1"]

y_raw, cp_sets = load_series("apple")
n = length(y_raw)
y = (y_raw .- mean(y_raw)) ./ std(y_raw)

println("Fast smoke test on apple (n=$n)")
for model_name in models
    manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn =
        default_model_setup(y, n, model_name; continuity=true)
    for (label, obj_type, min_seg, step) in practical_defaults
        try
            cps = run_config(y, n, manager, n_global, n_seg, chrom, bounds, parnames,
                             obj_type; min_seg=min_seg, step=step)
            f1 = compute_f1(cps, cp_sets)
            println("  $model_name | $label | n_cps=$(length(cps)) | F1=$(round(f1,digits=3)) | CPs=$cps")
        catch e
            println("  $model_name | $label | FAILED: $e")
        end
    end
    println()
end
