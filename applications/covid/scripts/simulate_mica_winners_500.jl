#!/usr/bin/env julia
# Post-process MICA 500-day winner outputs: simulate trajectories from the
# detected changepoints and parameters saved by run_covid_winners_500.jl.
# Usage: julia simulate_mica_winners_500.jl [suffix]
#   suffix defaults to "default" (matches run_covid_winners_500.jl output).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using CSV, DataFrames, Statistics
using OrdinaryDiffEq, Smoothers
using LabelledArrays
using Printf

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const OUT_ROOT    = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_G_500")

const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                  :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
const n_global = 8
const n_segment_specific = 8
const u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const data_indices = [5, 6, 7, 9, 11]

function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ * cos(2 * π * ((t - t₀) / 365))
end

function log_transform(data, threshold=1)
    return [val >= threshold ? log(val) : 0 for val in data]
end

function CovModel!(du, u, p, t)
    (ᴺS, ᴺE₀, ᴺE₁, ᴺI₀, ᴺI₁, ᴺI₂, ᴺI₃, ᴺR, D, Cases, V) = u
    N = ᴺS + ᴺE₀ + ᴺE₁ + ᴺI₀ + ᴺI₁ + ᴺI₂ + ᴺI₃ + ᴺR + D
    ᴺε₀  = p.ᴺε₀; ᴺε₁ = p.ᴺε₁; ᴺγ₀ = p.ᴺγ₀; ᴺγ₁ = p.ᴺγ₁
    ᴺγ₂  = p.ᴺγ₂; ᴺγ₃ = p.ᴺγ₃; ᴺp₁ = p.ᴺp₁; ᴺp₁₂ = p.ᴺp₁₂
    ᴺp₂₃ = p.ᴺp₂₃; ᴺp₁D = p.ᴺp₁D; ᴺp₂D = p.ᴺp₂D; ᴺp₃D = p.ᴺp₃D
    δ    = p.δ; δₜ = fδ(t, δ); ᴺβ = p.ᴺβ; ω = p.ω
    ν    = t < 330 ? 0.0 : p.ν
    ᴺβᴺSI = ᴺβ * δₜ * ᴺS * (ᴺE₁ + ᴺI₀ + ᴺI₁)
    du[1]  = -(ᴺβᴺSI) / N + ω * ᴺR - ν * ᴺS
    du[2]  =  (ᴺβᴺSI / N) - (ᴺε₀ * ᴺE₀)
    du[3]  =  (ᴺε₀ * ᴺE₀) - (ᴺε₁ * ᴺE₁)
    du[4]  =  ((1 - ᴺp₁) * ᴺε₁ * ᴺE₁) - (ᴺγ₀ * ᴺI₀)
    du[5]  =  (ᴺp₁ * ᴺε₁ * ᴺE₁) - (ᴺγ₁ * ᴺI₁)
    du[6]  =  (ᴺp₁₂ * ᴺγ₁ * ᴺI₁) - (ᴺγ₂ * ᴺI₂)
    du[7]  =  (ᴺp₂₃ * ᴺγ₂ * ᴺI₂) - (ᴺγ₃ * ᴺI₃)
    du[8]  =  ᴺγ₀ * ᴺI₀ +
              (1 - ᴺp₁₂ - ᴺp₁D) * ᴺγ₁ * ᴺI₁ +
              (1 - ᴺp₂₃ - ᴺp₂D) * ᴺγ₂ * ᴺI₂ +
              (1 - ᴺp₃D) * ᴺγ₃ * ᴺI₃ - ω * ᴺR + ν * ᴺS
    du[9]  =  (ᴺp₁D * ᴺγ₁ * ᴺI₁) + (ᴺp₂D * ᴺγ₂ * ᴺI₂) + (ᴺp₃D * ᴺγ₃ * ᴺI₃)
    du[10] =  ᴺp₁ * ᴺε₁ * ᴺE₁
    du[11] =  ν * ᴺS
end

function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0)
    if tspan[2] < 330
        params[:ν] = 0.0
    end
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-6, reltol=1.0e-6,
                maxiters=20000,
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    if SciMLBase.successful_retcode(sol)
        return Matrix(sol)
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

# ---------- load 500-day observed data ----------
cases_CP = CSV.read(joinpath(EXAMPLE_DIR, "case_rki_daily.csv"), DataFrame).total
hospital_CP = CSV.read(joinpath(EXAMPLE_DIR, "Hospitalization_rki_daily.csv"), DataFrame).total
death_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "death_rki_daily.csv"), DataFrame).Todesfaelle_neu)
icu_CP = CSV.read(joinpath(EXAMPLE_DIR, "icu_rki_daily.csv"), DataFrame).total
vacc_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "vaccination_rki_daily_allShots.csv"), DataFrame).Total)

data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
max_length = maximum(length, data_CP)
data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
data_CP = [vector[1:500] for vector in data_CP]
data_CP[1] = hma(data_CP[1], 14)
data_CP[4] = hma(data_CP[4], 14)
data_CP[5] = hma(data_CP[5], 14)
data_CP = Matrix(reduce(hcat, data_CP)')
const n = size(data_CP, 2)

function loss_equal_log(sim)
    total = 0.0
    for (k, r) in enumerate(data_indices)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(data_CP[k, :]))
    end
    return total
end

function extract_segment_params(chromosome)
    constant = chromosome[1:n_global]
    n_segments = div(length(chromosome) - n_global, n_segment_specific)
    seg_list = [chromosome[n_global + (s - 1) * n_segment_specific + 1 : n_global + s * n_segment_specific] for s in 1:n_segments]
    return constant, seg_list
end

function simulate_full(cps, params)
    constant, seg_list = extract_segment_params(params)
    n_seg = length(cps) + 1
    if length(seg_list) < n_seg
        error("Parameter list has $(length(seg_list)) segments but CPs imply $n_seg segments")
    end
    u0_curr = u0_full
    sim_segments = Matrix{Float64}[]
    for s in 1:n_seg
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end   = (s > length(cps)) ? n : cps[s]
        all_pars = @LArray [constant; seg_list[s]] parnames
        tspan_seg = (Float64(idx_start), Float64(idx_end))
        raw_seg = example_ode_model(all_pars, tspan_seg, u0_curr)
        push!(sim_segments, raw_seg)
        u0_curr = raw_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end

function process_winner(label::String, winner_dir::String)
    cps_file = joinpath(winner_dir, "covid_detected_cps_$(label).csv")
    params_file = joinpath(winner_dir, "covid_params_$(label).csv")

    if !isfile(cps_file) || !isfile(params_file)
        @warn "Skipping $label: missing CP or parameter CSV" label=label
        return nothing
    end

    cps = sort(unique(Int.(CSV.read(cps_file, DataFrame).cp)))
    params_df = CSV.read(params_file, DataFrame)
    chromosome = Float64.(params_df.value)
    # Drop any NaN padding that may have been added by the detection script
    chromosome = filter(x -> !isnan(x), chromosome)

    expected_len = n_global + (length(cps) + 1) * n_segment_specific
    if length(chromosome) != expected_len
        @warn "Skipping $label: chromosome length " * string(length(chromosome)) * " != expected " * string(expected_len)
        return nothing
    end

    sims_dir = joinpath(winner_dir, "simulations")
    mkpath(sims_dir)

    sim = simulate_full(cps, chromosome)
    loss = loss_equal_log(sim)

    sim_df = DataFrame(
        day = 1:size(sim, 2),
        infected = sim[data_indices[1], :],
        hospitalized = sim[data_indices[2], :],
        icu = sim[data_indices[3], :],
        death = sim[data_indices[4], :],
        vaccinated = sim[data_indices[5], :]
    )
    CSV.write(joinpath(sims_dir, "$(label).csv"), sim_df)

    return Dict(
        "label" => label,
        "n_cps" => length(cps),
        "cps" => cps,
        "refit_loss" => loss
    )
end

function main()
    suffix = length(ARGS) >= 1 ? ARGS[1] : ""
    search_root = OUT_ROOT
    winner_dirs = String[]
    for entry in readdir(search_root)
        # run_covid_winners_500.jl writes one directory per winner named winners_500_<label>
        if isdir(joinpath(search_root, entry)) && startswith(entry, "winners_500_")
            push!(winner_dirs, joinpath(search_root, entry))
        end
    end

    if isempty(winner_dirs)
        error("No winner output directories found in " * search_root)
    end

    results = Dict{String,Any}[]
    for winner_dir in winner_dirs
        cps_files = filter(f -> startswith(f, "covid_detected_cps_") && endswith(f, ".csv"), readdir(winner_dir))
        isempty(cps_files) && continue
        label = replace(first(cps_files), r"^covid_detected_cps_|\.csv$" => "")
        println("[simulate 500] Processing $label ...")
        res = process_winner(label, winner_dir)
        if res !== nothing
            push!(results, res)
            println("  -> " * string(res["n_cps"]) * " CPs, loss = " * string(res["refit_loss"]))
        end
    end

    summary_file = joinpath(OUT_ROOT, "winners_500_refit_summary$(suffix).csv")
    if !isempty(results)
        df = DataFrame(
            label = [r["label"] for r in results],
            n_cps = [r["n_cps"] for r in results],
            cps = [join(r["cps"], ";") for r in results],
            refit_loss = [r["refit_loss"] for r in results]
        )
        CSV.write(summary_file, df)
        println("Saved summary -> $summary_file")
    end
end

main()
