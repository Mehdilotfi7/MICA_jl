#!/usr/bin/env julia
# Visualise the best oracle/practical MICA configuration for each TCPD dataset.
# Unlike the Python version, this script re-fits the selected configuration with
# Mica.jl itself and plots the actual package simulation, so the curves are not
# Python refits or segment-mean fallbacks.
#
# Usage: julia visualize_tcpd_best_all42.jl [suffix]
# Default suffix is "global_time_all42".

using Pkg; Pkg.activate("codes/Mica.jl")
using Mica, JSON, Statistics, Plots, Printf

const SUFFIX = length(ARGS) >= 1 ? ARGS[1] : "global_time_all_v2"

gr()

const DATASET_DIR = joinpath(@__DIR__, "..", "dataset", "datasets")
const ANNOTATIONS_FILE = joinpath(@__DIR__, "..", "dataset", "annotations.json")
const FIGURE_DIR = "tcpd_figures_best_all42"
const CACHE_FILE = "tcpd_mica_sse_cache_$(SUFFIX).json"

mkpath(FIGURE_DIR)

const annotations = JSON.parsefile(ANNOTATIONS_FILE)

# ---------- helpers ----------

function load_series(series_name::String)
    data = JSON.parsefile(joinpath(DATASET_DIR, series_name, "$(series_name).json"))
    y_raw = data["series"][1]["raw"]
    y = Float64.(filter(x -> x !== nothing, y_raw))
    for i in 2:length(y)
        if !isfinite(y[i])
            y[i] = y[i-1]
        end
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

function parse_config(config_label::String)
    m = match(r"ms(\d+)_s(\d+)", config_label)
    @assert m !== nothing "Cannot parse config label: $config_label"
    return parse(Int, m.captures[1]), parse(Int, m.captures[2])
end

function fit_and_simulate(y_std::Vector{Float64}, model_name::String, cps::Vector{Int})
    n = length(y_std)
    data = reshape(y_std, 1, :)
    manager, n_global, n_seg, chrom, bounds, parnames, optimizer, loss_fn =
        default_model_setup(y_std, n, model_name; continuity=true)

    cps_sorted = sort(cps)
    num_segments = length(cps_sorted) + 1

    # Expand chromosome/bounds to hold one set of segment parameters per segment
    for _ in 1:(num_segments - 1)
        update_bounds!(chrom, bounds, n_global, n_seg, extract_parameters)
    end

    if optimizer isa AnalyticalOptimizer
        boundaries = [0; cps_sorted; n]
        best_params = Float64[]
        total_loss = 0.0
        for i in 1:(length(boundaries) - 1)
            idx_start = boundaries[i] + 1
            idx_end = boundaries[i + 1]
            seg_data = data[:, idx_start:idx_end]
            loss, seg_params = fit_segment_analytical(manager.base_model, seg_data)
            total_loss += loss
            append!(best_params, seg_params)
        end
    else
        total_loss, best_params = optimize_with_changepoints(
            objective_function, chrom, parnames, cps_sorted, bounds, optimizer,
            n_global, n_seg, manager, loss_fn, data
        )
    end

    yhat, _ = simulate_full_model(best_params, sort(cps), parnames,
                                  n_global, n_seg, manager, data)
    return best_params, vec(yhat), total_loss
end

function row_cache_key(row)
    k = row["kappa"]
    return "$(row["dataset"])|$(row["model"])|$(row["config"])|$(row["obj"])|$(k === nothing ? "null" : k)"
end

function compute_mica_sse!(row, cache::Dict{String,Float64})
    key = row_cache_key(row)
    if haskey(cache, key) && cache[key] < 1e17
        return cache[key]
    end

    y_raw, _ = load_series(row["dataset"])
    y_std = (y_raw .- mean(y_raw)) ./ std(y_raw)
    cps = collect(Int, row["cps"])

    try
        _, yhat, _ = fit_and_simulate(y_std, row["model"], cps)
        sse = sum((y_std .- yhat).^2)
        cache[key] = sse
        return sse
    catch e
        @warn "Mica.jl fit failed for $key" exception = (e, catch_backtrace())
        cache[key] = Inf
        return Inf
    end
end

# ---------- load results and cache ----------

const cache = if isfile(CACHE_FILE) && filesize(CACHE_FILE) > 0
    Dict{String,Float64}(k => Float64(v) for (k, v) in JSON.parsefile(CACHE_FILE))
else
    Dict{String,Float64}()
end

const oracle = JSON.parsefile("benchmark_tcpd_comprehensive_numerical_$(SUFFIX)_oracle.json")
const practical = JSON.parsefile("benchmark_tcpd_comprehensive_numerical_$(SUFFIX)_practical.json")

println("Computing / loading Mica.jl SSE for $(length(oracle)) oracle rows...")
for (i, row) in enumerate(oracle)
    compute_mica_sse!(row, cache)
    i % 20 == 0 && println("  oracle: $i / $(length(oracle))")
end

println("Computing / loading Mica.jl SSE for $(length(practical)) practical rows...")
for (i, row) in enumerate(practical)
    compute_mica_sse!(row, cache)
    i % 20 == 0 && println("  practical: $i / $(length(practical))")
end

open(CACHE_FILE, "w") do f
    JSON.print(f, Dict(k => isfinite(v) ? v : 1e18 for (k, v) in cache), 2)
end
println("SSE cache saved to $CACHE_FILE")

# ---------- selection ----------

function pick_oracle(rows)
    by_ds = Dict{String,Vector{Dict{String,Any}}}()
    for r in rows
        push!(get!(by_ds, r["dataset"], Dict{String,Any}[]), r)
    end
    best = Dict{String,Dict{String,Any}}()
    for (ds, rs) in by_ds
        max_f1 = maximum(r["f1"] for r in rs)
        tied = filter(r -> r["f1"] == max_f1, rs)
        if length(tied) == 1
            best[ds] = tied[1]
        else
            _, idx = findmin(r -> compute_mica_sse!(r, cache), tied)
            best[ds] = tied[idx]
        end
    end
    return best
end

function pick_practical(rows)
    by_ds = Dict{String,Vector{Dict{String,Any}}}()
    for r in rows
        push!(get!(by_ds, r["dataset"], Dict{String,Any}[]), r)
    end
    best = Dict{String,Dict{String,Any}}()
    for (ds, rs) in by_ds
        # Practical selection: choose the configuration with the smallest
        # post-hoc BIC. The true labels are not used for selection; the reported
        # F1 is then the F1 of that configuration evaluated against the true
        # changepoints.
        bic_vals = [r["bic"] for r in rs]
        _, idx = findmin(bic_vals)
        best[ds] = rs[idx]
    end
    return best
end

const best_oracle = pick_oracle(oracle)
const best_practical = pick_practical(practical)

println("Selected $(length(best_oracle)) oracle and $(length(best_practical)) practical configs.")

# ---------- plotting ----------

const SEG_COLORS = [
    :blue, :orange, :green, :red, :purple,
    :brown, :pink, :gray, :olive, :cyan,
    :magenta, :lime, :teal, :yellow, :navy
]

function consensus_color(count::Int, n_ann::Int)
    # Sequential green palette: pale = single annotator, dark = all annotators
    greens = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1b7837", "#006837", "#004529"]
    idx = clamp(count, 1, length(greens))
    return greens[idx]
end

function generate_figure(y_raw::Vector{Float64}, cp_sets::Vector{Vector{Int}},
                         row::Dict{String,Any}, mode::String)
    y_std = (y_raw .- mean(y_raw)) ./ std(y_raw)
    n = length(y_std)
    cps = sort(collect(Int, row["cps"]))
    model_name = row["model"]
    config_label = row["config"]
    f1 = row["f1"]
    covering = get(row, "covering", 0.0)

    _, yhat, _ = fit_and_simulate(y_std, model_name, cps)
    t = 1:n

    boundaries = [0; cps; n]

    # Consensus counts per true CP: how many annotators placed a CP within margin=5
    margin = 5
    n_ann = length(cp_sets)
    true_cp_counts = Dict{Int,Int}()
    all_true = Int[]
    if !isempty(cp_sets)
        all_true = sort(unique(vcat(cp_sets...)))
        for cp in all_true
            cnt = sum(any(abs(a - cp) <= margin for a in cps_ann) for cps_ann in cp_sets)
            true_cp_counts[cp] = cnt
        end
    end

    # Main time-series panel
    p1 = plot(t, y_std, color = :gray, alpha = 0.5, linewidth = 0.8, label = "Data",
              legend = :best, legendfontsize = 7, xaxis = false)
    scatter!(p1, t, y_std, color = :gray, alpha = 0.3, markersize = 2, label = "")

    for i in 1:(length(boundaries) - 1)
        start = boundaries[i] + 1
        ending = boundaries[i + 1]
        plot_end = i < length(boundaries) - 1 ? min(ending + 1, n) : ending
        seg_t = t[start:plot_end]
        seg_yhat = yhat[start:plot_end]
        color = SEG_COLORS[(i - 1) % length(SEG_COLORS) + 1]
        plot!(p1, seg_t, seg_yhat, color = color, linewidth = 2.2,
              label = length(boundaries) > 2 ? "Seg $i" : "Fit")
    end

    for cp in cps
        vline!(p1, [cp], color = :red, linestyle = :dash, linewidth = 1.5,
               alpha = 0.8, label = "")
    end

    if !isempty(cp_sets)
        for cp in all_true
            cnt = get(true_cp_counts, cp, 1)
            col = consensus_color(cnt, n_ann)
            vline!(p1, [cp], color = col, linestyle = :dot, linewidth = 2.0,
                   alpha = 0.9, label = "")
        end
        true_cp_str = join(string.(all_true), ", ")
        n_true = length(all_true)
    else
        true_cp_str = "none"
        n_true = 0
    end
    detected_cp_str = isempty(cps) ? "none" : join(string.(cps), ", ")

    if mode == "oracle"
        title_str = "$(row["dataset"])  |  $(model_name)  |  $(config_label)  |  $(mode)\n" *
                    "F1=$(round(f1, digits=3))\n" *
                    "Detected ($(length(cps))): $(detected_cp_str)\n" *
                    "True ($(n_true)): $(true_cp_str)"
    else
        title_str = "$(row["dataset"])  |  $(model_name)  |  $(config_label)  |  $(mode)\n" *
                    "F1=$(round(f1, digits=3))\n" *
                    "Detected ($(length(cps))): $(detected_cp_str)\n" *
                    "True ($(n_true)): $(true_cp_str)"
    end
    title_str *= "\nTrue CP color = annotator agreement (dark = high)"
    title!(p1, title_str, fontsize = 9, loc = :left)
    ylabel!(p1, "Standardized value")
    plot!(p1, gridalpha = 0.2)

    # Bottom annotation raster panel
    if !isempty(cp_sets)
        ann_labels = ["A$i" for i in 1:n_ann]
        p2 = plot(legend = false, xlabel = "Time", ylabel = "Annotators",
                  yticks = (1:n_ann, ann_labels), ylims = (0.5, n_ann + 0.5),
                  gridalpha = 0.2)
        for (i, cps_ann) in enumerate(cp_sets)
            scatter!(p2, cps_ann, fill(i, length(cps_ann)),
                     markersize = 4, color = :black, label = "")
        end
        for cp in cps
            vline!(p2, [cp], color = :red, linestyle = :dash, linewidth = 1.5,
                   alpha = 0.8)
        end
    else
        p2 = plot(legend = false, framestyle = :none, xlabel = "Time")
    end

    plt = plot(p1, p2, layout = (2, 1), link = :x, size = (1000, 650),
               margin = 5Plots.mm)

    fname = "$(row["dataset"])_$(model_name)_$(config_label)_$(mode).png"
    fpath = joinpath(FIGURE_DIR, fname)
    savefig(plt, fpath)
    return fpath
end

# ---------- generate all figures ----------

# Only generate oracle figures in this folder.
const all_best = [(best_oracle, "oracle")]
n_figs = 0

for (best_dict, mode) in all_best
    for (ds, row) in best_dict
        y_raw, cp_sets = load_series(ds)
        try
            fpath = generate_figure(y_raw, cp_sets, row, mode)
            global n_figs += 1
            n_figs % 10 == 0 && println("  generated $n_figs figures")
        catch e
            @warn "Failed to generate figure for $ds $mode" exception = (e, catch_backtrace())
        end
    end
end

println("Done. $n_figs figures saved to $FIGURE_DIR/")

# ---------- summary table ----------

function write_summary_csv(path::String = "tcpd_best_config_summary_all42.csv")
    datasets = sort(collect(keys(best_oracle)))
    open(path, "w") do f
        println(f, "dataset,model_O,config_O,f1_O,precision_O,recall_O,covering_O,cps_O,kappa_O,n_cps_O,obj_O,mode_O," *
                   "f1_P,precision_P,recall_P,covering_P,cps_P,kappa_P,n_cps_P,obj_P,model_P,config_P,mode_P,sse_O,sse_P")
        for ds in datasets
            o = best_oracle[ds]
            p = best_practical[ds]
            sse_o = compute_mica_sse!(o, cache)
            sse_p = compute_mica_sse!(p, cache)
            cps_o = "[" * join(o["cps"], ", ") * "]"
            cps_p = "[" * join(p["cps"], ", ") * "]"
            println(f,
                "$ds,$(o["model"]),$(o["config"]),$(o["f1"]),$(o["precision"]),$(o["recall"]),$(o["covering"]),\"$cps_o\",$(o["kappa"]),$(o["n_cps"]),$(o["obj"]),MICA-O," *
                "$(p["f1"]),$(p["precision"]),$(p["recall"]),$(p["covering"]),\"$cps_p\",$(p["kappa"]),$(p["n_cps"]),$(p["obj"]),$(p["model"]),$(p["config"]),MICA-P,$sse_o,$sse_p"
            )
        end
    end
    println("Wrote summary CSV to $path")
end

function tex_escape(s::String)
    return "\\texttt{" * replace(s, "_" => "\\_") * "}"
end

function write_summary_tex(path::String = "tcpd_best_config_summary.tex")
    datasets = sort(collect(keys(best_oracle)))
    lines = String[
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Best MICA configuration per TCPD dataset.}",
        "\\label{tab:tcpd-best-configs}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Dataset & \\multicolumn{3}{c}{MICA-O (oracle)} & \\multicolumn{3}{c}{MICA-P (practical)} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        " & Model & Config & F1 & Model & Config & F1 \\\\",
        "\\midrule",
    ]
    for ds in datasets
        o = best_oracle[ds]
        p = best_practical[ds]
        ds_tex = tex_escape(ds)
        m_o = tex_escape(o["model"])
        c_o = tex_escape(o["config"])
        f_o = @sprintf("%.3f", o["f1"])
        m_p = tex_escape(p["model"])
        c_p = tex_escape(p["config"])
        f_p = @sprintf("%.3f", p["f1"])
        push!(lines, "$ds_tex & $m_o & $c_o & $f_o & $m_p & $c_p & $f_p\\\\")
    end
    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}%")
    push!(lines, "\\end{table}")
    push!(lines, "")
    write(path, join(lines, "\n"))
    println("Wrote summary TeX to $path")
end

write_summary_csv()
write_summary_tex()
