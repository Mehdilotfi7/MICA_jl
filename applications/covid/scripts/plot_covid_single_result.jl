#!/usr/bin/env julia
# Plot visualizations for a single-objective result directory.
# Usage: julia plot_covid_single_result.jl <results_dir> [objective_name]
# Defaults: results_dir="../results_pop150_seed1_verify", objective_name="none"

using Pkg
mica_project = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl")
Pkg.activate(mica_project)

using Mica
using CSV, DataFrames, Dates, Plots
import Mica: simulate!, Model, OdeSettings

OUT_DIR = length(ARGS) >= 1 ? ARGS[1] : "../results_pop150_seed1_verify"
OBJ = length(ARGS) >= 2 ? ARGS[2] : "none"
OUT_DIR = abspath(OUT_DIR)

# Determine project/data paths
app_dir = abspath(joinpath(@__DIR__, ".."))
data_dir = joinpath(app_dir, "data")

# Load data
function load_covid_data()
    df = CSV.read(joinpath(data_dir, "covid_timeseries.csv"), DataFrame)
    infected = df.infected[2:end]
    recovered = df.recovered[2:end]
    deceased = df.deceased[2:end]
    population = df.population[1]
    data_CP = Matrix(transpose(hcat(infected, recovered, deceased)))
    dates = Date.(df.date[2:end], "yyyy-mm-dd")
    data_CP, dates, population
end

const data_CP, dates, population = load_covid_data()
const initial_states = [83300000.0 - sum(data_CP[:, 1]); data_CP[:, 1]]
const tspan = (1.0, Float64(size(data_CP, 2)))
const settings = OdeSettings(
    solver=Mica.RK4(),
    dt=1.0,
    abstol=1e-6,
    reltol=1e-3,
)

function load_cps(path)
    df = CSV.read(path, DataFrame)
    sort(unique(df.change_point))
end

function load_params(path)
    df = CSV.read(path, DataFrame)
    names(df)
    # Find the first column that is not 'change_point'
    cols = names(df)
    col = findfirst(c -> c != "change_point", cols)
    df[:, col]
end

function simulate_with_cps(cps, params)
    model = Model(Mica.seird, initial_states, tspan, settings)
    simulate!(model, params, change_points=cps)
    model.states
end

function safe_log10(x)
    y = float.(x)
    y[y .<= 0] .= NaN
    log10.(y)
end

function make_covid_panel(cps, params; title_suffix="", use_log=true)
    sim = simulate_with_cps(cps, params)
    n_compartments = size(data_CP, 1)
    compartment_labels = ["Infected", "Recovered", "Deceased"]
    pal = palette(:tab10)
    plts = []
    for i in 1:n_compartments
        ydata = use_log ? safe_log10(data_CP[i, :]) : data_CP[i, :]
        ysim = use_log ? safe_log10(sim[5 + (i - 1), :]) : sim[5 + (i - 1), :]
        p = plot(dates, ydata, label="Data", lw=1.0, color=pal[i], alpha=0.8,
                 title="$(compartment_labels[i])", titlefontsize=10)
        plot!(dates, ysim, label="Simulation", lw=1.5, color=:black)
        for (j, cp) in enumerate(cps)
            vline!([dates[cp]], color=:green, lw=1.5, linestyle=:dash,
                   label=(j == 1 ? "CPs" : false))
        end
        xlabel!("Date")
        use_log ? ylabel!("log₁₀(count)") : ylabel!("count")
        plot!(xrotation=45)
        if i == 1
            plot!(legend=:topleft, legendfontsize=7)
        else
            plot!(legend=false)
        end
        push!(plts, p)
    end
    # cumulative infected
    cum_inf_data = cumsum(data_CP[1, :])
    cum_inf_sim = cumsum(sim[5, :])
    ydata = use_log ? safe_log10(cum_inf_data) : cum_inf_data
    ysim = use_log ? safe_log10(cum_inf_sim) : cum_inf_sim
    p = plot(dates, ydata, label="Data", lw=1.0, color=:purple, alpha=0.8,
             title="Cumulative Infected", titlefontsize=10)
    plot!(dates, ysim, label="Simulation", lw=1.5, color=:black)
    for (j, cp) in enumerate(cps)
        vline!([dates[cp]], color=:green, lw=1.5, linestyle=:dash,
               label=(j == 1 ? "CPs" : false))
    end
    xlabel!("Date")
    use_log ? ylabel!("log₁₀(count)") : ylabel!("count")
    plot!(xrotation=45)
    plot!(legend=false)
    push!(plts, p)

    # R_eff
    function compute_R_eff(model, params)
        n_segments = length(cps)
        beta = params[1:4:(3 + 4 * n_segments)]
        gamma = params[4]
        N = population
        r_eff = zeros(size(model.states, 2))
        seg_indices = [1; sort(cps); size(model.states, 2) + 1]
        for s in 1:n_segments
            for t in seg_indices[s]:(seg_indices[s + 1] - 1)
                S_t = model.states[1, t]
                r_eff[t] = beta[s] * S_t / (gamma * N)
            end
        end
        r_eff
    end
    model = Model(Mica.seird, initial_states, tspan, settings)
    simulate!(model, params, change_points=cps)
    r_eff = compute_R_eff(model, params)
    p = plot(dates, r_eff, label="R_eff", lw=1.5, color=:darkred,
             title="Effective reproduction number", titlefontsize=10, ylim=(0, 5))
    hline!([1.0], color=:gray, linestyle=:dash, label="R_eff = 1")
    for (j, cp) in enumerate(cps)
        vline!([dates[cp]], color=:green, lw=1.5, linestyle=:dash,
               label=(j == 1 ? "CPs" : false))
    end
    xlabel!("Date")
    ylabel!("R_eff")
    plot!(xrotation=45)
    plot!(legend=:topright, legendfontsize=7)
    push!(plts, p)

    suptitle = "COVID-19 Germany — $(uppercase(OBJ)) — $title_suffix"
    plot(plts..., layout=(3, 2), size=(1100, 1200), dpi=200,
         plot_title=suptitle, plot_titlefontsize=13,
         left_margin=8Plots.mm, bottom_margin=10Plots.mm)
end

cps_file = joinpath(OUT_DIR, "covid_detected_cps_origset_$(OBJ).csv")
params_file = joinpath(OUT_DIR, "covid_params_origset_$(OBJ).csv")

if !isfile(cps_file) || !isfile(params_file)
    error("Missing files:\n  $cps_file\n  $params_file")
end

cps = load_cps(cps_file)
params = load_params(params_file)

println("$OBJ: #CPs=$(length(cps)), CPs=$cps")

for use_log in [true, false]
    suffix = use_log ? "log" : "raw"
    plt = make_covid_panel(cps, params;
                           title_suffix="$suffix scale",
                           use_log=use_log)
    out = joinpath(OUT_DIR, "covid_visualization_origset_$(OBJ)_$(suffix).png")
    savefig(plt, out)
    println("  Saved: $out")
end
