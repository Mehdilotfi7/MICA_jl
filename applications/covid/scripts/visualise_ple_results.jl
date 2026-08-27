#!/usr/bin/env julia
# Visualise PLE results from a results folder.
# Usage: julia visualise_ple_results.jl <results_folder>
# Reads  <results_folder>/ple_curves.csv
#        <results_folder>/ple_summary.csv
# Saves  <results_folder>/ple_profiles.pdf

# Must be set before loading Plots to avoid needing a display.
ENV["GKSwstype"] = "100"

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using CSV, DataFrames, Statistics, Printf
using Plots
gr()

const CHISQ_95 = 3.8414588206941285

function main()
    if length(ARGS) < 1
        println("Usage: julia visualise_ple_results.jl <results_folder>")
        exit(1)
    end

    results_dir = ARGS[1]
    curves_file = joinpath(results_dir, "ple_results_curves.csv")
    summary_file = joinpath(results_dir, "ple_summary.csv")
    out_pdf = joinpath(results_dir, "ple_profiles.pdf")

    if !isdir(results_dir)
        println(stderr, "Results folder does not exist: $results_dir")
        exit(1)
    end
    if !isfile(curves_file)
        println(stderr, "Missing PLE curves file: $curves_file")
        exit(1)
    end
    if !isfile(summary_file)
        println(stderr, "Missing PLE summary file: $summary_file")
        exit(1)
    end

    curves = CSV.read(curves_file, DataFrame)
    summary = CSV.read(summary_file, DataFrame)

    # Validate expected columns
    required_curves = ("parameter", "value", "loss")
    required_sum = ("parameter", "best_value", "threshold")
    for col in required_curves
        col in names(curves) || error("Column '$col' missing from $curves_file")
    end
    for col in required_sum
        col in names(summary) || error("Column '$col' missing from $summary_file")
    end

    params = summary[:, :parameter]
    n = length(params)
    if n == 0
        println("No parameters found in summary; nothing to plot.")
        exit(0)
    end

    # Layout: 4 columns for readability, min 2 cols for tiny panels.
    ncols = n <= 4 ? 2 : 4
    nrows = ceil(Int, n / ncols)

    # Use a larger figure for many parameters.
    fig_width = 300 * ncols
    fig_height = 220 * nrows

    plots = Plots.Plot[]

    for (i, param) in enumerate(params)
        s = summary[summary.parameter .== param, :]
        if nrow(s) == 0
            @warn "Summary row missing for parameter $param"
            continue
        end
        s = s[1, :]

        df = curves[curves.parameter .== param, :]
        if nrow(df) == 0
            @warn "No profile points for parameter $param"
            continue
        end
        df = sort(df, :value)

        best_val = s.best_value
        threshold = s.threshold

        ci_lower = s.ci_lower
        ci_upper = s.ci_upper
        has_ci = !(isnan(ci_lower) || isnan(ci_upper) || ismissing(ci_lower) || ismissing(ci_upper))

        p = plot(df.value, df.loss, marker=:circle, markersize=3, lw=1.2,
                 color=:steelblue, label=nothing, title=string(param),
                 xlabel="parameter value", ylabel="loss",
                 titlefont=font(9), legend=false, grid=true)

        hline!([threshold], color=:crimson, linestyle=:dash, lw=1, label="95% threshold")
        vline!([best_val], color=:darkgreen, linestyle=:dot, lw=1.2, label="best fit")

        if has_ci
            lo, hi = extrema([ci_lower, ci_upper])
            vspan!([lo, hi], color=:gold, alpha=0.25, label="approx. CI")
        end

        # Log scale if the value range spans more than one order of magnitude and all positive.
        vals = df.value
        if all(x -> x > 0, vals) && (maximum(vals) / minimum(vals) > 10)
            plot!(xscale=:log10)
        end

        push!(plots, p)
    end

    if isempty(plots)
        println("No plots generated; check input files.")
        exit(0)
    end

    # Build a single figure from the panel vector.
    fig = plot(plots..., size=(fig_width, fig_height), layout=(nrows, ncols),
               plot_title="Profile-likelihood curves (threshold = χ² 95% = $CHISQ_95)",
               plot_titlefontsize=11, margin=5Plots.mm)

    # Turn off unused subplots.
    for j in (length(plots)+1):(nrows*ncols)
        plot!(sp=j, axis=false, border=:none, grid=false, legend=false)
    end

    savefig(fig, out_pdf)
    println("Saved PLE profiles to: $out_pdf")
end

main()
