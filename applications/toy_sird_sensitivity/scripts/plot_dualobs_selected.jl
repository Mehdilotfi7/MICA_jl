#!/usr/bin/env julia
# Selected-setting figures for the dual-observable toy-SIRD experiment:
# sigma = 1.0, seed = 2.  Two figures, each with M1-M4 as panels:
#   (a) example fits (both compartments: I and D)
#   (b) estimated piecewise-constant parameter trajectories vs truth
using JSON, Printf
using Plots
gr()

const ROOT = joinpath(@__DIR__, "..", "..")
const DATA = joinpath(ROOT, "results", "data")
const FIGS = joinpath(ROOT, "results", "figures")
mkpath(FIGS)

const SIGMA = 1.0
const SEED = 2
const CONFIGS = ["M1", "M2", "M3", "M4"]
const CONFIG_DESC = Dict(
    "M1" => "β SP; γ, δ shared",
    "M2" => "β, γ SP; δ shared",
    "M3" => "β, γ, δ SP",
    "M4" => "γ SP; β, δ shared (misspecified)",
)
const TRUE_CPS = [20, 40]
const N = 80
const TRUE_VALS = Dict("β" => [0.4, 0.1, 0.3], "γ" => [0.1, 0.1, 0.1], "δ" => [0.02, 0.02, 0.02])
const LAYOUT = Dict(
    "M1" => (2, 1, ["γ", "δ", "β"]),
    "M2" => (1, 2, ["δ", "β", "γ"]),
    "M3" => (0, 3, ["β", "γ", "δ"]),
    "M4" => (2, 1, ["β", "δ", "γ"]),
)
const PCOLORS = Dict("β" => :steelblue, "γ" => :seagreen, "δ" => :darkorange)

function step_xy(cps, vals, n=N)
    bounds = vcat([0], cps, [n])
    xs, ys = Float64[], Float64[]
    for i in eachindex(vals)
        push!(xs, bounds[i], bounds[i + 1])
        push!(ys, vals[i], vals[i])
    end
    return xs, ys
end

function get_rec(cfg)
    recs = [r for r in JSON.parsefile(joinpath(DATA, "sird_dualobs_$(cfg).json")) if !haskey(r, "error")]
    return [x for x in recs if x["sigma"] == SIGMA && x["seed"] == SEED][1]
end

# ---------------- (a) example fits ----------------
plots = []
for (k, cfg) in enumerate(CONFIGS)
    r = get_rec(cfg)
    t = 0:(r["n"] - 1)
    p = plot(framestyle=:box, grid=true, gridalpha=0.3,
             legend=(k == 1 ? :topright : false),
             title="$(cfg): $(CONFIG_DESC[cfg])  (F1=$(round(r["f1"], digits=2)))",
             titlefontsize=9, tickfontsize=7)
    plot!(p, t, Float64.(r["yI"]), seriestype=:scatter, color=:grey60, ms=2.2, markerstrokewidth=0, label=(k == 1 ? "I data" : ""))
    plot!(p, t, Float64.(r["simulation"]), color=:steelblue, lw=1.3, label=(k == 1 ? "I fit" : ""))
    plot!(p, t, Float64.(r["yD"]), seriestype=:scatter, color=:grey85, ms=2.2, markerstrokewidth=0, label=(k == 1 ? "D data" : ""))
    plot!(p, t, Float64.(r["simulation_D"]), color=:darkorange, lw=1.3, label=(k == 1 ? "D fit" : ""))
    k == 1 && plot!(p, [NaN], [NaN], color=:black, ls=:dash, lw=0.9, label="true CP")
    k == 1 && plot!(p, [NaN], [NaN], color=:red, ls=:dot, lw=1.0, label="detected CP")
    for c in r["true_cps"]
        vline!(p, [c], color=:black, ls=:dash, lw=0.9, label="")
    end
    for c in r["detected_cps"]
        vline!(p, [c], color=:red, ls=:dot, lw=1.0, label="")
    end
    push!(plots, p)
end
fig = plot(plots..., layout=(2, 2), size=(1100, 700),
           plot_title="Example MICA fits for the four toy-SIRD configurations (σ=1.0, seed 2)",
           plot_titlefontsize=11)
savefig(fig, joinpath(FIGS, "fits_selected_dualobs.png"))
savefig(fig, joinpath(FIGS, "fits_selected_dualobs.pdf"))
println("saved fits_selected_dualobs")

# ---------------- (b) estimated parameters ----------------
plots = []
for (k, cfg) in enumerate(CONFIGS)
    r = get_rec(cfg)
    n_global, n_seg, parnames = LAYOUT[cfg]
    cps = Int.(r["detected_cps"])
    n_segments = length(cps) + 1
    params = Float64.(r["params"])

    p = plot(framestyle=:box, grid=true, gridalpha=0.3,
             legend=(k == 1 ? :topright : false),
             title="$(cfg): $(CONFIG_DESC[cfg])", titlefontsize=9,
             tickfontsize=7, ylim=(-0.03, 0.55))
    for pname in ["β", "γ", "δ"]
        gidx = findfirst(==(pname), parnames[1:n_global])
        if gidx !== nothing
            gval = params[gidx]
            plot!(p, [0, N], [gval, gval], color=PCOLORS[pname], lw=1.6,
                  label=(k == 1 ? "est. $pname (NSP)" : ""))
        else
            bidx = findfirst(==(pname), parnames[n_global+1:n_global+n_seg])
            vals = [params[n_global + (i - 1) * n_seg + bidx] for i in 1:n_segments]
            xs, ys = step_xy(cps, vals)
            plot!(p, xs, ys, color=PCOLORS[pname], lw=1.6,
                  label=(k == 1 ? "est. $pname (SP)" : ""))
        end
        xs, ys = step_xy(TRUE_CPS, TRUE_VALS[pname])
        plot!(p, xs, ys, color=:black, ls=:dash, lw=0.9,
              label=(k == 1 && pname == "β" ? "true" : ""))
    end
    k == 1 && plot!(p, [NaN], [NaN], color=:red, ls=:dot, lw=0.8, label="detected CP")
    for c in cps
        vline!(p, [c], color=:red, ls=:dot, lw=0.8, label="")
    end
    push!(plots, p)
end
fig = plot(plots..., layout=(2, 2), size=(1100, 700),
           plot_title="Estimated per-segment parameter values for the four toy-SIRD configurations (σ=1.0, seed 2)",
           plot_titlefontsize=11)
savefig(fig, joinpath(FIGS, "params_selected_dualobs.png"))
savefig(fig, joinpath(FIGS, "params_selected_dualobs.pdf"))
println("saved params_selected_dualobs")
