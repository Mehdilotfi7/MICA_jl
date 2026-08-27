#!/usr/bin/env julia
# Visualise changepoint profile results from a results folder.
# Usage: julia visualise_cp_profiles.jl <results_folder>
# Reads  <results_folder>/cp_profile_loss.csv
#        <results_folder>/cp_profile_ci.csv
# Saves  <results_folder>/cp_profiles.pdf

# Must be set before loading Plots to avoid needing a display.
ENV["GKSwstype"] = "100"

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using CSV, DataFrames, Statistics, Dates, Printf
using Plots
gr()

const CHISQ_95 = 3.8414588206941285

function main()
    if length(ARGS) < 1
        println("Usage: julia visualise_cp_profiles.jl <results_folder>")
        exit(1)
    end

    results_dir = ARGS[1]
    loss_file = joinpath(results_dir, "cp_profile_loss.csv")
    ci_file = joinpath(results_dir, "cp_profile_ci.csv")
    out_pdf = joinpath(results_dir, "cp_profiles.pdf")

    if !isdir(results_dir)
        println(stderr, "Results folder does not exist: $results_dir")
        exit(1)
    end
    if !isfile(loss_file)
        println(stderr, "Missing CP profile loss file: $loss_file")
        exit(1)
    end
    if !isfile(ci_file)
        println(stderr, "Missing CP profile CI file: $ci_file")
        exit(1)
    end

    loss_df = CSV.read(loss_file, DataFrame)
    ci_df = CSV.read(ci_file, DataFrame)

    # Validate expected columns
    required_loss = ("cp_index", "original_cp", "candidate_cp", "loss")
    required_ci = ("cp_index", "original_cp", "ci_lower", "ci_upper")
    for col in required_loss
        col in names(loss_df) || error("Column '$col' missing from $loss_file")
    end
    for col in required_ci
        col in names(ci_df) || error("Column '$col' missing from $ci_file")
    end

    cp_indices = sort(unique(loss_df[:, :cp_index]))
    n = length(cp_indices)
    if n == 0
        println("No changepoints found in profile; nothing to plot.")
        exit(0)
    end

    ncols = n <= 4 ? 2 : 4
    nrows = ceil(Int, n / ncols)

    fig_width = 300 * ncols
    fig_height = 220 * nrows

    plots = Plots.Plot[]

    for cp_idx in cp_indices
        sub = loss_df[loss_df.cp_index .== cp_idx, :]
        if nrow(sub) == 0
            @warn "No loss points for CP index $cp_idx"
            continue
        end
        sub = sort(sub, :candidate_cp)

        original_cp = sub[1, :original_cp]
        best_loss = minimum(sub[:, :loss])
        threshold = best_loss + CHISQ_95

        ci_row = ci_df[ci_df.cp_index .== cp_idx, :]
        if nrow(ci_row) > 0
            ci_row = ci_row[1, :]
            ci_lower = ci_row.ci_lower
            ci_upper = ci_row.ci_upper
            has_ci = !(ismissing(ci_lower) || ismissing(ci_upper) || isnan(ci_lower) || isnan(ci_upper))
            if has_ci
                ci_lower, ci_upper = extrema([ci_lower, ci_upper])
            end
        else
            has_ci = false
        end

        date_label = ""
        if "date" in names(sub) && !ismissing(sub[1, :date])
            try
                date_label = " ($(sub[1, :date]))"
            catch
            end
        end

        p = plot(sub.candidate_cp, sub.loss, marker=:circle, markersize=3, lw=1.2,
                 color=:steelblue, label=nothing,
                 title="CP $cp_idx: day $original_cp$date_label",
                 xlabel="candidate day", ylabel="loss",
                 titlefont=font(9), legend=false, grid=true)

        hline!([threshold], color=:crimson, linestyle=:dash, lw=1, label="95% threshold")
        vline!([original_cp], color=:darkgreen, linestyle=:dot, lw=1.2, label="best CP")

        if has_ci
            vspan!([ci_lower, ci_upper], color=:gold, alpha=0.25, label="approx. CI")
        end

        push!(plots, p)
    end

    if isempty(plots)
        println("No CP plots generated; check input files.")
        exit(0)
    end

    fig = plot(plots..., size=(fig_width, fig_height), layout=(nrows, ncols),
               plot_title="Changepoint profile likelihoods (threshold = χ² 95% = $CHISQ_95)",
               plot_titlefontsize=11, margin=5Plots.mm)

    for j in (length(plots)+1):(nrows*ncols)
        plot!(sp=j, axis=false, border=:none, grid=false, legend=false)
    end

    savefig(fig, out_pdf)
    println("Saved CP profiles to: $out_pdf")
end

main()
