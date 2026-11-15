#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Dates
using JSON
using Plots
gr()

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS = joinpath(ROOT, "results", "ga_stability")
const OUTPUT = joinpath(RESULTS, "fig_covid_ga_stability_rugplot")
const START_DATE = Date(2020, 1, 27)

default(
    fontfamily = "Arial",
    guidefont = ("Arial", 16),
    tickfont = ("Arial", 14),
    legendfont = ("Arial", 16),
    size = (2250, 2400),
    margin = 5Plots.mm,
    framestyle = :box,
    grid = false,
    foreground_color_legend = nothing,
    background_color_legend = :white,
)

function load_runs(path, group)
    entries = JSON.parsefile(path)
    return [(group=group, seed=entry["seed"], cps=Int.(entry["cps"])) for entry in entries]
end

runs = vcat(
    load_runs(joinpath(RESULTS, "covid_multiseed_summary.json"), "GA standard"),
    load_runs(joinpath(RESULTS, "covid_multiseed_set2_summary.json"), "GA standard"),
    load_runs(joinpath(RESULTS, "covid_multiseed_highbudget_summary.json"), "GA high budget"),
)
const EXCLUDED_STANDARD_SEEDS = Set([9999, 999, 777, 222])
filter!(run -> run.group != "GA standard" || !(run.seed in EXCLUDED_STANDARD_SEEDS), runs)

function run_panel(group, color, show_xlabel)
    selected = filter(run -> run.group == group, runs)
    sort!(selected, by=run -> (length(run.cps) == 1, run.cps[1], length(run.cps) == 2 ? run.cps[2] : -1, run.seed))
    panel = plot(
        xlim = (0, 400),
        ylim = (0, length(selected) + 1),
        yticks = (length(selected):-1:1, ["Seed $(run.seed)" for run in selected]),
        ylabel = "$group\n(n = $(length(selected)))",
        xlabel = show_xlabel ? "Date" : "",
        xticks = (0:50:400, [Dates.format(START_DATE + Day(i), "yyyy-mm") for i in 0:50:400]),
        xrotation = 30,
        leftmargin = 20Plots.mm,
        legend = group == "GA standard" ? :topright : false,
        linewidth = 2.0,
    )
    for cp in (40, 60, 80, 120, 150, 260, 290, 320)
        vline!(panel, [cp], color=:gray, linestyle=:dot, alpha=0.65, linewidth=2.8, label="")
    end
    vline!(panel, [60], color=:black, linestyle=:dash, linewidth=2.4, label="")
    vline!(panel, [150], color=:black, linestyle=:dashdot, linewidth=2.4, label="")
    if group == "GA standard"
        scatter!(panel, [NaN], [NaN], color=:white, markerstrokecolor=:grey40, markerstrokewidth=2.5,
            markersize=11, label="Detected CP in a two-CP run")
        scatter!(panel, [NaN], [NaN], shape=:diamond, color=:white, markerstrokecolor=:grey40,
            markerstrokewidth=2.5, markersize=11, label="Single detected CP")
        plot!(panel, [NaN], [NaN], color=:black, linestyle=:dash, linewidth=2.4,
            label="Primary BIC CP1 (index 60)")
        plot!(panel, [NaN], [NaN], color=:black, linestyle=:dashdot, linewidth=2.4,
            label="Primary BIC CP2 (index 150)")
        plot!(panel, [NaN], [NaN], color=:gray, linestyle=:dot, linewidth=2.0,
            label="Legacy 8-CP solution")
    end
    for (y, run) in zip(length(selected):-1:1, selected)
        if length(run.cps) == 2
            plot!(panel, run.cps, [y, y], color=color, alpha=0.65, linewidth=2.4, label="")
            scatter!(panel, run.cps, [y, y], color=:white, markerstrokecolor=color,
                markerstrokewidth=2.5, markersize=11, label="")
        else
            scatter!(panel, run.cps, [y], shape=:diamond, color=:white, markerstrokecolor=color,
                markerstrokewidth=2.5, markersize=11, label="")
        end
    end
    return panel
end

standard = run_panel("GA standard", "#2171B5", false)
high = run_panel("GA high budget", "#238B45", true)
figure = plot(
    standard, high,
    layout = grid(2, 1, heights=(0.8, 0.2)),
    link = :x,
    bottom_margin = 8Plots.mm,
)

savefig(figure, OUTPUT * ".png")
savefig(figure, OUTPUT * ".pdf")
println("Saved Julia/Plots.jl PLOS-style figure: ", OUTPUT, ".{png,pdf}")
