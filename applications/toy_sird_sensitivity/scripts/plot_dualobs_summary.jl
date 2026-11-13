#!/usr/bin/env julia
# Summary figure for the dual-observable toy-SIRD experiment (Julia/Plots.jl).
# Six panels: F1, number of detected CPs, beta/gamma/delta MAE, sqrt(RSS)
# fit error — per configuration (M1-M4) and noise level.
using JSON, Statistics, Printf
using Plots
gr()

const ROOT = joinpath(@__DIR__, "..", "..")
const DATA = joinpath(ROOT, "results", "data")
const FIGS = joinpath(ROOT, "results", "figures")
mkpath(FIGS)

const SIGMAS = [0.5, 1.0, 2.0]
const SEEDS = [1, 2, 3]
const CONFIGS = ["M1", "M2", "M3", "M4"]
const COLORS = Dict(0.5 => :steelblue, 1.0 => :darkorange, 2.0 => :seagreen)

load(cfg) = [r for r in JSON.parsefile(joinpath(DATA, "sird_dualobs_$(cfg).json")) if !haskey(r, "error")]

function rss_fit(rec)
    yI = Float64.(rec["yI"]); yD = Float64.(rec["yD"])
    sI = Float64.(rec["simulation"]); sD = Float64.(rec["simulation_D"])
    return sqrt(sum((yI .- sI).^2) + sum((yD .- sD).^2))
end

function getvals(recs, sig, key)
    vals = Float64[]
    for s in SEEDS
        r = [x for x in recs if x["sigma"] == sig && x["seed"] == s]
        isempty(r) && continue
        v = key == "rss" ? rss_fit(r[1]) : r[1][key]
        v === nothing && continue
        push!(vals, Float64(v))
    end
    return vals
end

# manual box-and-whisker (3 seeds per cell): IQR box, median line, whiskers
function draw_box!(p, x, vals, w, color)
    q1, q2, q3 = quantile(vals, [0.25, 0.5, 0.75])
    lo, hi = minimum(vals), maximum(vals)
    # box
    plot!(p, [x - w/2, x + w/2, x + w/2, x - w/2], [q1, q1, q3, q3],
          seriestype=:shape, fillcolor=color, fillalpha=0.6, linecolor=:black, lw=0.8, label="")
    # median
    plot!(p, [x - w/2, x + w/2], [q2, q2], color=:black, lw=1.2, label="")
    # whiskers
    plot!(p, [x, x], [lo, q1], color=:black, lw=0.8, label="")
    plot!(p, [x, x], [q3, hi], color=:black, lw=0.8, label="")
    plot!(p, [x - w/4, x + w/4], [lo, lo], color=:black, lw=0.8, label="")
    plot!(p, [x - w/4, x + w/4], [hi, hi], color=:black, lw=0.8, label="")
end

panels = [
    ("f1", "F1 score", (0, 1.05)),
    ("n_cps", "detected change points", (-0.2, 5.2)),
    ("beta_mae", "MAE of β", nothing),
    ("gamma_mae", "MAE of γ", nothing),
    ("delta_mae", "MAE of δ", nothing),
    ("rss", "fit error √RSS", nothing),
]

data = Dict(cfg => load(cfg) for cfg in CONFIGS)

plots = []
for (pi, (key, label, ylim)) in enumerate(panels)
    p = plot(framestyle=:box, grid=true, gridalpha=0.3,
             legend=(pi == 1 ? :topright : false),
             ylabel=label, xticks=(1:4, CONFIGS), tickfontsize=7,
             ylabelfontsize=8)
    ylim !== nothing && ylims!(p, ylim...)
    width = 0.25
    for (si, sig) in enumerate(SIGMAS)
        positions = (1:4) .+ (si - 2) * width
        for (ci, cfg) in enumerate(CONFIGS)
            vals = getvals(data[cfg], sig, key)
            isempty(vals) && continue
            draw_box!(p, positions[ci], vals, width * 0.9, COLORS[sig])
        end
        # legend entries on the first panel only
        pi == 1 && plot!(p, [NaN], [NaN], seriestype=:shape, color=COLORS[sig],
                         fillalpha=0.6, label="σ=$sig")
    end
    push!(plots, p)
end

fig = plot(plots..., layout=(2, 3), size=(1300, 750),
           plot_title="Toy-SIRD experiment | pooled-RSS + MDL, min_length 5, step 1, GA 200/400",
           plot_titlefontsize=10)
savefig(fig, joinpath(FIGS, "summary_dualobs.png"))
savefig(fig, joinpath(FIGS, "summary_dualobs.pdf"))
println("saved summary_dualobs")
