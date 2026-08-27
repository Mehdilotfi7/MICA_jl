#!/usr/bin/env julia
# Visualise every configuration tested for every TCPD dataset.
# Output: tcpd_figures_comprehensive/<dataset>/<dataset>_<model>_<config>_<obj>_<kappa>_F<f1>.png
#
# Usage: julia visualize_tcpd_comprehensive.jl [suffix]
# Default suffix is "global_time_all_v2".

const MICA_PROJECT = abspath(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))
using Pkg; Pkg.activate(MICA_PROJECT)
using Mica, JSON, Statistics, Plots, Printf

include(abspath(joinpath(@__DIR__, "mica_plot_theme.jl")))

const SUFFIX = length(ARGS) >= 1 ? ARGS[1] : "global_time_all_v2"

const DATASET_DIR = joinpath(@__DIR__, "..", "dataset", "datasets")
const ANNOTATIONS_FILE = joinpath(@__DIR__, "..", "dataset", "annotations.json")
const OUT_DIR = "tcpd_figures_comprehensive"
const RESULTS_FILE = "benchmark_tcpd_comprehensive_numerical_$(SUFFIX).json"
const ORACLE_FILE = "benchmark_tcpd_comprehensive_numerical_$(SUFFIX)_oracle.json"
const CACHE_FILE = "tcpd_mica_sse_cache_$(SUFFIX).json"

mkpath(OUT_DIR)

const annotations = JSON.parsefile(ANNOTATIONS_FILE)

# ---------- helpers (same as visualize_tcpd_best_all42.jl) ----------

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

const cache = if isfile(CACHE_FILE) && filesize(CACHE_FILE) > 0
    Dict{String,Float64}(k => Float64(v) for (k, v) in JSON.parsefile(CACHE_FILE))
else
    Dict{String,Float64}()
end

function compute_mica_sse!(row, cache::Dict{String,Float64})
    key = "$(row["dataset"])|$(row["model"])|$(row["config"])|$(row["obj"])|$(row["kappa"] === nothing ? "null" : row["kappa"])"
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

# ---------- plotting ----------

const SEG_COLORS = [
    :blue, :orange, :green, :red, :purple,
    :brown, :pink, :gray, :olive, :cyan,
    :magenta, :lime, :teal, :yellow, :navy
]

function consensus_color(count::Int, n_ann::Int)
    greens = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1b7837", "#006837", "#004529"]
    idx = clamp(count, 1, length(greens))
    return greens[idx]
end

function generate_figure(y_raw::Vector{Float64}, cp_sets::Vector{Vector{Int}},
                         row::Dict{String,Any})
    y_std = (y_raw .- mean(y_raw)) ./ std(y_raw)
    n = length(y_std)
    cps = sort(collect(Int, row["cps"]))
    model_name = row["model"]
    config_label = row["config"]
    f1 = row["f1"]
    obj = row["obj"]
    kappa = row["kappa"]

    _, yhat, _ = fit_and_simulate(y_std, model_name, cps)
    t = 1:n
    boundaries = [0; cps; n]
    n_ann = length(cp_sets)
    all_true = !isempty(cp_sets) ? sort(unique(vcat(cp_sets...))) : Int[]
    n_true = length(all_true)

    seg_colors = mica_color_cycle()

    # Top panel: data + fit + detected change points only
    p1 = plot(t, y_std, color = RGB(0.4, 0.4, 0.4), alpha = 0.55, linewidth = 0.8,
              label = "Data", legend = :best, legendfontsize = 6, xaxis = false)
    scatter!(p1, t, y_std, color = RGB(0.4, 0.4, 0.4), alpha = 0.3, markersize = 1.5, label = "")

    for i in 1:(length(boundaries) - 1)
        start = boundaries[i] + 1
        ending = boundaries[i + 1]
        plot_end = i < length(boundaries) - 1 ? min(ending + 1, n) : ending
        seg_t = t[start:plot_end]
        seg_yhat = yhat[start:plot_end]
        color = seg_colors[(i - 1) % length(seg_colors) + 1]
        plot!(p1, seg_t, seg_yhat, color = color, linewidth = 1.5,
              label = length(boundaries) > 2 ? "Seg $i" : "Fit")
    end

    plot!(p1, [], [], color = MICA_RED, linestyle = :dash, linewidth = 1.2,
          alpha = 0.8, label = "Detected CP")
    for cp in cps
        vline!(p1, [cp], color = MICA_RED, linestyle = :dash, linewidth = 1.2,
               alpha = 0.8, label = "")
    end

    kappa_str = kappa === nothing ? "na" : @sprintf("%.3f", kappa)
    title_str = "$(row["dataset"])  |  $(model_name)  |  $(config_label)  |  $(obj)\n" *
                "F1=$(round(f1, digits=3))  |  κ=$(kappa_str)  |  " *
                "Detected=$(length(cps))  |  True=$(n_true)"
    title!(p1, title_str, fontsize = 7, loc = :left)
    ylabel!(p1, "Std. value", fontsize = 8)
    plot!(p1, gridalpha = 0.2, top_margin = 6Plots.mm)

    # Bottom panel: annotator ground truth + detected change points + legend
    if !isempty(cp_sets)
        ann_labels = ["A$i" for i in 1:n_ann]
        p2 = plot(xlabel = "Time", ylabel = "Annot.",
                  yticks = (1:n_ann, ann_labels), ylims = (0.5, n_ann + 0.5),
                  gridalpha = 0.2, xtickfontsize = 6, ytickfontsize = 6,
                  legend = :best, legendfontsize = 6)
        plot!(p2, [], [], color = MICA_RED, linestyle = :dash, linewidth = 1.2,
              alpha = 0.8, label = "Detected CP")
        scatter!(p2, [], [], color = MICA_BLACK, markersize = 2, label = "Annotator CP")
        for (i, cps_ann) in enumerate(cp_sets)
            scatter!(p2, cps_ann, fill(i, length(cps_ann)),
                     markersize = 2, color = MICA_BLACK, label = "")
        end
        for cp in cps
            vline!(p2, [cp], color = MICA_RED, linestyle = :dash, linewidth = 1.2,
                   alpha = 0.8, label = "")
        end
    else
        p2 = plot(legend = false, framestyle = :none, xlabel = "Time")
    end

    plt = plot(p1, p2, layout = (2, 1), link = :x, size = (850, 520),
               margin = 4Plots.mm)

    ds_dir = joinpath(OUT_DIR, row["dataset"])
    mkpath(ds_dir)
    kappa_str = kappa === nothing ? "na" : @sprintf("%.3f", kappa)
    fname = "$(row["dataset"])_$(model_name)_$(config_label)_$(obj)_$(kappa_str)_F$(@sprintf("%.3f", f1)).png"
    fpath = joinpath(ds_dir, fname)
    savefig(plt, fpath)
    return fpath
end

# ---------- main ----------

const all_results = JSON.parsefile(ORACLE_FILE)
println("Visualising $(length(all_results)) configurations...")

n_done = 0
n_fail = 0
for raw_row in all_results
    row = Dict{String,Any}(raw_row)
    try
        y_raw, cp_sets = load_series(row["dataset"])
        generate_figure(y_raw, cp_sets, row)
        global n_done += 1
    catch e
        global n_fail += 1
        @warn "Failed for $(row["dataset"]) $(row["model"]) $(row["config"])" exception = (e, catch_backtrace())
    end
    if (n_done + n_fail) % 100 == 0
        println("  $(n_done + n_fail) / $(length(all_results)) done, $n_fail failed")
    end
end

println("\nDone. $n_done figures saved to $OUT_DIR/ ($n_fail failures)")
