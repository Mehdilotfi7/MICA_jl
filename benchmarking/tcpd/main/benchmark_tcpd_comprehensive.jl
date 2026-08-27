# ============================================================
# TCPD COMPREHENSIVE Benchmark — Numerical Optimizers
# Usage: julia benchmark_tcpd_comprehensive.jl <start_idx> <end_idx> <suffix>
# ============================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))
using Mica, JSON, Statistics, Optim, Evolutionary

const RSS_LOSS = (obs, sim) -> sum((obs .- sim).^2)

# ---- Parse args ----
args = length(ARGS) >= 3 ? ARGS : ["1", "31", "all"]
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

# ---- Model builders with numerical optimizers ----
#
# Uses Mica.default_model_setup, which provides literature-based bounds,
# initial guesses, optimizers and a safe loss wrapper (see
# MODEL_PARAMETER_BOUNDS_RESEARCH.md).
function build_model(y, n, model_name; kwargs...)
    return default_model_setup(y, n, model_name; continuity=true, kwargs...)
end

# ---- Post-hoc BIC ----
function posthoc_bic(y, cps, model_name, n)
    n_cps = length(cps)
    dof_per_seg = if model_name == "mean" 1
    elseif model_name in ["compound_growth"] 1
    elseif model_name in ["linear", "log_linear", "hyperbolic", "ar1", "debt_dynamics", "accelerator"] 2
    elseif model_name in ["quadratic", "exponential", "power", "mean_drift", "ar2", "michaelis_menten", "poisson", "negbin", "saturating_exponential"] 3
    elseif model_name in ["cubic", "ar3", "asymptotic_regression", "hill_function", "log_logistic", "ingarch", "logistic", "gompertz"] 4
    elseif model_name in ["weibull_growth", "double_exponential", "rational", "ets_aaa", "ets_mmm"] 5
    else 2 end
    p_total = (n_cps + 1) * dof_per_seg + n_cps
    boundaries = [0; cps; n]
    total_rss = 0.0
    for i in 1:(length(boundaries)-1)
        a, b = boundaries[i]+1, boundaries[i+1]
        seg = y[a:b]
        m = length(seg)
        m == 0 && continue
        if model_name == "mean"
            total_rss += sum((seg .- mean(seg)).^2)
        elseif model_name in ["ar1", "ar2", "ar3"]
            total_rss += ar_rss(seg, model_name)
        else
            t = Float64.(1:m)
            if model_name == "linear"
                X = hcat(t, ones(m)); pred = X * (X \ seg)
            elseif model_name == "quadratic"
                X = hcat(t.^2, t, ones(m)); coeffs = X \ seg
                pred = coeffs[1] .* t.^2 .+ coeffs[2] .* t .+ coeffs[3]
            elseif model_name == "cubic"
                X = hcat(t.^3, t.^2, t, ones(m)); coeffs = X \ seg
                pred = coeffs[1] .* t.^3 .+ coeffs[2] .* t.^2 .+ coeffs[3] .* t .+ coeffs[4]
            elseif model_name == "exponential"
                off = minimum(seg) <= 0 ? abs(minimum(seg)) + 1.0 : 0.0
                X = hcat(t, ones(m)); coeffs = X \ log.(seg .+ off)
                pred = exp(coeffs[2]) .* exp.(coeffs[1] .* t) .- off
            elseif model_name == "log_linear"
                X = hcat(log.(t), ones(m)); pred = X * (X \ seg)
            elseif model_name == "power"
                off = minimum(seg) <= 0 ? abs(minimum(seg)) + 1.0 : 0.0
                X = hcat(log.(t), ones(m)); coeffs = X \ log.(seg .+ off)
                pred = exp(coeffs[2]) .* (t .^ coeffs[1]) .- off
            elseif model_name == "mean_drift"
                X = hcat(t, ones(m)); ab = X \ seg
                mu = mean(seg .- (ab[1] .* t .+ ab[2]))
                pred = mu .+ ab[1] .* t .+ ab[2]
            else
                # Fallback for models without explicit OLS post-hoc fit
                pred = fill(mean(seg), m)
            end
            total_rss += sum((seg .- pred).^2)
        end
    end
    sigma2 = max(total_rss / n, 1e-10)
    return n * log(sigma2) + p_total * log(n)
end

function ar_rss(seg::Vector{Float64}, model_name::String)
    m = length(seg)
    if model_name == "ar1"
        m < 3 && return sum((seg .- mean(seg)).^2)
        y_t = seg[2:end]; y_lag = seg[1:end-1]
        A = hcat(y_lag, ones(m-1)); coeffs = A \ y_t
        phi, c = coeffs
        pred = vcat([seg[1]], c .+ phi .* y_lag)
        return sum((seg .- pred).^2)
    elseif model_name == "ar2"
        m < 4 && return ar_rss(seg, "ar1")
        y_t = seg[3:end]; y_lag1 = seg[2:end-1]; y_lag2 = seg[1:end-2]
        A = hcat(y_lag1, y_lag2, ones(m-2)); coeffs = A \ y_t
        phi1, phi2, c = coeffs
        pred = vcat(seg[1:2], c .+ phi1 .* y_lag1 .+ phi2 .* y_lag2)
        return sum((seg .- pred).^2)
    elseif model_name == "ar3"
        m < 5 && return ar_rss(seg, "ar2")
        y_t = seg[4:end]; y_lag1 = seg[3:end-1]; y_lag2 = seg[2:end-2]; y_lag3 = seg[1:end-3]
        A = hcat(y_lag1, y_lag2, y_lag3, ones(m-3)); coeffs = A \ y_t
        phi1, phi2, phi3, c = coeffs
        pred = vcat(seg[1:3], c .+ phi1 .* y_lag1 .+ phi2 .* y_lag2 .+ phi3 .* y_lag3)
        return sum((seg .- pred).^2)
    end
    return sum((seg .- mean(seg)).^2)
end

# ---- Run single config ----
function run_config(y, n, manager, n_global, n_seg, chrom, bounds, parnames,
                    objective_type, penalty_fn, min_seg, step, optimizer,
                    loss_function=RSS_LOSS)
    data = reshape(y, 1, :)
    opt = optimizer
    t0 = time()
    cps, best_params = detect_changepoints(
        objective_function, n, n_global, n_seg,
        manager, loss_function,
        data, copy(chrom), parnames,
        (copy(bounds[1]), copy(bounds[2])), opt,
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
obj_types = [:bic, :mdl, :aic]
penalty_kappas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# ---- Load all datasets ----
dataset_names = String[]
for d in readdir(DATASET_DIR)
    json_path = joinpath(DATASET_DIR, d, "$(d).json")
    if isdir(joinpath(DATASET_DIR, d)) && isfile(json_path) && haskey(annotations, d)
        push!(dataset_names, d)
    end
end
sort!(dataset_names)

# Only run the benchmark main loop when this file is executed directly.
# Set ENV["MICA_RUN_BENCHMARK"] = "false" before include() to load helpers only.
_run_benchmark = get(ENV, "MICA_RUN_BENCHMARK", "true") == "true"

if _run_benchmark

# Select subset based on args
start_idx = clamp(start_idx, 1, length(dataset_names))
end_idx = clamp(end_idx, 1, length(dataset_names))
my_datasets = dataset_names[start_idx:end_idx]

println("TCPD Comprehensive Benchmark [suffix=$suffix]")
println("Datasets: $(my_datasets)")
println("Models: $(length(models))")
println("Objectives: $(length(obj_types)) + penalty")
flush(stdout)

all_results = Dict{String,Any}[]
all_oracle = Dict{String,Any}[]
all_practical = Dict{String,Any}[]

for (idx, ds_name) in enumerate(my_datasets)
    y_raw, cp_sets = load_series(ds_name)
    local n = length(y_raw)
    local y = (y_raw .- mean(y_raw)) ./ std(y_raw)
    sigma2 = var(y)

    min_seg_values = sort(unique([
        1, 2, 3,
        max(3, n ÷ 50), max(3, n ÷ 30),
        max(5, n ÷ 20), max(5, n ÷ 10)
    ]))
    step_values = sort(unique([
        1,
        max(1, n ÷ 100), max(1, n ÷ 50), max(1, n ÷ 20)
    ]))

    t_ds = time()
    for model_name in models
        manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn = build_model(y, n, model_name)

        for min_seg in min_seg_values
            for step in step_values
                config_label = "ms$(min_seg)_s$(step)"

                # --- BIC / MDL / AIC ---
                for obj_type in obj_types
                    try
                        pen_fn = (p, n) -> 0.0
                        cps, params, elapsed = run_config(y, n, manager, n_global, n_seg, chrom, bounds, parnames,
                                                          obj_type, pen_fn, min_seg, step, optimizer, loss_fn)
                        prec, rec, f1 = compute_f1(cps, cp_sets)
                        cover = compute_covering(cps, cp_sets, n)
                        bic = posthoc_bic(y, cps, model_name, n)
                        push!(all_results, Dict(
                            "dataset" => ds_name, "model" => model_name,
                            "config" => config_label, "obj" => string(obj_type),
                            "kappa" => nothing, "f1" => f1, "precision" => prec,
                            "recall" => rec, "covering" => cover, "bic" => bic,
                            "n_cps" => length(cps), "cps" => cps, "time" => elapsed
                        ))
                    catch
                    end
                end

                # --- Penalty with κ sweep ---
                kappa_results = []
                for kappa in penalty_kappas
                    scaled_k = kappa * sigma2
                    pen_fn = (p, n) -> scaled_k * p * log(n)
                    try
                        cps, params, elapsed = run_config(y, n, manager, n_global, n_seg, chrom, bounds, parnames,
                                                          :penalty, pen_fn, min_seg, step, optimizer, loss_fn)
                        prec, rec, f1 = compute_f1(cps, cp_sets)
                        cover = compute_covering(cps, cp_sets, n)
                        bic = posthoc_bic(y, cps, model_name, n)
                        push!(kappa_results, (
                            kappa=kappa, cps=cps, f1=f1, precision=prec, recall=rec,
                            covering=cover, bic=bic, n_cps=length(cps), time=elapsed
                        ))
                    catch
                    end
                end

                if !isempty(kappa_results)
                    best_oracle = kappa_results[argmax([r.f1 for r in kappa_results])]
                    best_bic = kappa_results[argmin([r.bic for r in kappa_results])]

                    push!(all_oracle, Dict(
                        "dataset" => ds_name, "model" => model_name,
                        "config" => config_label, "obj" => "penalty_oracle",
                        "kappa" => best_oracle.kappa, "f1" => best_oracle.f1,
                        "precision" => best_oracle.precision, "recall" => best_oracle.recall,
                        "covering" => best_oracle.covering, "bic" => best_oracle.bic,
                        "n_cps" => best_oracle.n_cps, "cps" => best_oracle.cps,
                        "time" => best_oracle.time
                    ))
                    push!(all_practical, Dict(
                        "dataset" => ds_name, "model" => model_name,
                        "config" => config_label, "obj" => "penalty_bic",
                        "kappa" => best_bic.kappa, "f1" => best_bic.f1,
                        "precision" => best_bic.precision, "recall" => best_bic.recall,
                        "covering" => best_bic.covering, "bic" => best_bic.bic,
                        "n_cps" => best_bic.n_cps, "cps" => best_bic.cps,
                        "time" => best_bic.time
                    ))
                    for r in kappa_results
                        push!(all_results, Dict(
                            "dataset" => ds_name, "model" => model_name,
                            "config" => config_label, "obj" => "penalty",
                            "kappa" => r.kappa, "f1" => r.f1, "precision" => r.precision,
                            "recall" => r.recall, "covering" => r.covering, "bic" => r.bic,
                            "n_cps" => r.n_cps, "cps" => r.cps, "time" => r.time
                        ))
                    end
                end
            end
        end
    end

    elapsed_ds = time() - t_ds
    println("[$idx/$(length(my_datasets))] $ds_name done | $(length(all_results)) runs | $(round(elapsed_ds,digits=1))s")
    flush(stdout)

    # Save after each dataset
    open("benchmark_tcpd_comprehensive_numerical_$(suffix).json", "w") do f
        JSON.print(f, all_results, 2)
    end
    open("benchmark_tcpd_comprehensive_numerical_$(suffix)_oracle.json", "w") do f
        JSON.print(f, all_oracle, 2)
    end
    open("benchmark_tcpd_comprehensive_numerical_$(suffix)_practical.json", "w") do f
        JSON.print(f, all_practical, 2)
    end
end

println("\nBENCHMARK COMPLETE for suffix=$suffix")
println("Total runs: $(length(all_results))")
println("Oracle configs: $(length(all_oracle))")
println("Practical configs: $(length(all_practical))")

end  # if _run_benchmark
