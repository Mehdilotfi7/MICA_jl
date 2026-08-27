# ============================================================
# Benchmark additional baseline methods on TCPD datasets
# Methods: BS-RSS, PELT-RSS, Bayesian-CPD, Kernel-CPD, HMM-Regime, Maulik-ML
# ============================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using JSON
using Statistics
using Random
using LinearAlgebra

# Load baseline implementations
include(joinpath(@__DIR__, "..", "..", "..", "methods",  "generic_baselines.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "methods",  "new_baselines.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "methods",  "published_baselines.jl"))

# ---- Load helpers from comprehensive benchmark (metrics only) ----
ENV["MICA_RUN_BENCHMARK"] = "false"
include(joinpath(@__DIR__, "..", "..", "..", "Main_comprehensive_continuity_NelderMead", "benchmark_tcpd_comprehensive.jl"))

# ---- Baseline definitions: default function and oracle sweep grid generator ----
# Each entry is (default_fn, sweep_fn) where sweep_fn(n) returns a vector of NamedTuples.
function make_baselines()
    baselines = Dict{String, Tuple{Function, Function}}()

    baselines["BS-RSS"] = (
        (y, n) -> binary_segmentation_rss(y; max_cp=5, min_seg_len=10, pen=0.0),
        n -> [
            (max_cp=3, min_seg_len=10, pen=0.0),
            (max_cp=5, min_seg_len=10, pen=0.0),
            (max_cp=8, min_seg_len=10, pen=0.0),
            (max_cp=5, min_seg_len=5,  pen=0.0),
            (max_cp=5, min_seg_len=15, pen=0.0),
            (max_cp=5, min_seg_len=10, pen=log(n)),
            (max_cp=5, min_seg_len=10, pen=2.0*log(n)),
        ]
    )

    baselines["PELT-RSS"] = (
        (y, n) -> pelt_rss(y; min_seg_len=10, pen=2.0*log(n)),
        n -> [
            (min_seg_len=10, pen=1.0*log(n)),
            (min_seg_len=10, pen=2.0*log(n)),
            (min_seg_len=10, pen=3.0*log(n)),
            (min_seg_len=5,  pen=2.0*log(n)),
            (min_seg_len=15, pen=2.0*log(n)),
            (min_seg_len=10, pen=5.0),
        ]
    )

    baselines["Bayesian-CPD"] = (
        (y, n) -> bayesian_cpd(y; max_cp=5, min_seg_len=10, pen_mult=2.0),
        n -> [
            (max_cp=3, min_seg_len=10, pen_mult=1.0),
            (max_cp=5, min_seg_len=10, pen_mult=1.0),
            (max_cp=5, min_seg_len=10, pen_mult=2.0),
            (max_cp=5, min_seg_len=10, pen_mult=3.0),
            (max_cp=8, min_seg_len=10, pen_mult=2.0),
            (max_cp=5, min_seg_len=5,  pen_mult=2.0),
        ]
    )

    baselines["Kernel-CPD"] = (
        (y, n) -> kernel_cpd(y; window_size=20, min_seg_len=10, threshold=0.1),
        n -> [
            (window_size=10, min_seg_len=10, threshold=0.05),
            (window_size=10, min_seg_len=10, threshold=0.10),
            (window_size=10, min_seg_len=10, threshold=0.20),
            (window_size=20, min_seg_len=10, threshold=0.05),
            (window_size=20, min_seg_len=10, threshold=0.10),
            (window_size=20, min_seg_len=10, threshold=0.20),
            (window_size=30, min_seg_len=10, threshold=0.10),
            (window_size=20, min_seg_len=5,  threshold=0.10),
        ]
    )

    baselines["HMM-Regime"] = (
        (y, n) -> hmm_regime_switching(y; K=3, min_seg_len=10),
        n -> [
            (K=2, min_seg_len=10),
            (K=3, min_seg_len=10),
            (K=4, min_seg_len=10),
            (K=5, min_seg_len=10),
            (K=3, min_seg_len=5),
            (K=3, min_seg_len=15),
        ]
    )

    baselines["Maulik-ML"] = (
        (y, n) -> maulik_ml(y; window_size=20, threshold=0.1, min_seg_len=10),
        n -> [
            (window_size=10, threshold=0.05, min_seg_len=10),
            (window_size=10, threshold=0.10, min_seg_len=10),
            (window_size=10, threshold=0.20, min_seg_len=10),
            (window_size=20, threshold=0.05, min_seg_len=10),
            (window_size=20, threshold=0.10, min_seg_len=10),
            (window_size=20, threshold=0.20, min_seg_len=10),
            (window_size=30, threshold=0.10, min_seg_len=10),
            (window_size=20, threshold=0.10, min_seg_len=5),
        ]
    )

    return baselines
end

const baselines = make_baselines()

# Use a single tolerance for F1/Covering (same as TCPD paper)
const METRIC_MARGIN = 5

# ---- Load all datasets ----
annotations = JSON.parsefile(ANNOTATIONS_FILE)
dataset_names = String[]
for d in readdir(DATASET_DIR)
    json_path = joinpath(DATASET_DIR, d, "$(d).json")
    if isdir(joinpath(DATASET_DIR, d)) && isfile(json_path) && haskey(annotations, d)
        push!(dataset_names, d)
    end
end
sort!(dataset_names)

println("Additional baselines benchmark")
println("Datasets: $(length(dataset_names))")
println("Methods: $(length(baselines))")

all_results = Dict{String,Any}[]
all_oracle  = Dict{String,Any}[]
all_default = Dict{String,Any}[]

for (idx, ds_name) in enumerate(dataset_names)
    y_raw, cp_sets = load_series(ds_name)
    n = length(y_raw)
    println("\n[$(idx)/$(length(dataset_names))] $ds_name (n=$n, CPs=$(sum(length.(cp_sets))))")

    for (method_name, (default_fn, sweep_fn)) in baselines
        # ---- Default / practical run ----
        try
            cps_default = default_fn(y_raw, n)
            cps_default = filter(c -> c > 0 && c < n, sort(unique(cps_default)))
            prec_d, rec_d, f1_d = compute_f1(cps_default, cp_sets)
            cover_d = compute_covering(cps_default, cp_sets, n)
            push!(all_default, Dict(
                "dataset" => ds_name, "model" => method_name,
                "f1" => f1_d, "precision" => prec_d, "recall" => rec_d,
                "covering" => cover_d, "n_cps" => length(cps_default), "cps" => cps_default
            ))
        catch e
            println("  $method_name default failed: $e")
            push!(all_default, Dict(
                "dataset" => ds_name, "model" => method_name,
                "f1" => 0.0, "precision" => 0.0, "recall" => 0.0,
                "covering" => 0.0, "n_cps" => 0, "cps" => Int[],
                "error" => string(e)
            ))
        end

        # ---- Oracle sweep ----
        best_f1 = -1.0
        best_record = nothing
        for cfg in sweep_fn(n)
            try
                cps = if method_name == "BS-RSS"
                    binary_segmentation_rss(y_raw; cfg...)
                elseif method_name == "PELT-RSS"
                    pelt_rss(y_raw; cfg...)
                elseif method_name == "Bayesian-CPD"
                    bayesian_cpd(y_raw; cfg...)
                elseif method_name == "Kernel-CPD"
                    kernel_cpd(y_raw; cfg...)
                elseif method_name == "HMM-Regime"
                    hmm_regime_switching(y_raw; cfg...)
                elseif method_name == "Maulik-ML"
                    maulik_ml(y_raw; cfg...)
                else
                    Int[]
                end
                cps = filter(c -> c > 0 && c < n, sort(unique(cps)))
                prec, rec, f1 = compute_f1(cps, cp_sets)
                cover = compute_covering(cps, cp_sets, n)
                if f1 > best_f1
                    best_f1 = f1
                    best_record = Dict(
                        "dataset" => ds_name, "model" => method_name,
                        "config" => string(cfg), "f1" => f1,
                        "precision" => prec, "recall" => rec,
                        "covering" => cover, "n_cps" => length(cps), "cps" => cps
                    )
                end
            catch e
                # ignore failed config
            end
        end

        if best_record !== nothing
            push!(all_oracle, best_record)
        else
            push!(all_oracle, Dict(
                "dataset" => ds_name, "model" => method_name,
                "config" => "all_failed", "f1" => 0.0,
                "precision" => 0.0, "recall" => 0.0,
                "covering" => 0.0, "n_cps" => 0, "cps" => Int[]
            ))
        end

        flush(stdout)
    end

    # Save after each dataset
    open("benchmark_tcpd_additional_baselines_default.json", "w") do f
        JSON.print(f, all_default, 2)
    end
    open("benchmark_tcpd_additional_baselines_oracle.json", "w") do f
        JSON.print(f, all_oracle, 2)
    end
end

println("\nBENCHMARK COMPLETE")
println("Default entries: $(length(all_default))")
println("Oracle entries: $(length(all_oracle))")
