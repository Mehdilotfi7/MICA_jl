#!/usr/bin/env julia
# Generate a synthetic COVID-19 data set from the best-fit SEIRD model.
# Ground-truth CPs = MICA zero-penalty CPs; parameters = best-fit parameters.
# Add log-normal observation noise.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics, Random
using OrdinaryDiffEq, Smoothers
using LabelledArrays

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const TASK_A_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A", "results_penalty_zero")
const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_H_synthetic")
mkpath(OUT_DIR)

const SEED = 2024
const NOISE_SIGMA = 0.10

function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ * cos(2 * π * ((t - t₀) / 365))
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
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    if SciMLBase.successful_retcode(sol)
        return sol[:, :]
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                   :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
const n_global = 8
const n_segment_specific = 8
const u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const data_indices = [5, 6, 7, 9, 11]
const n = 400

function extract_segment_params(chromosome)
    constant = chromosome[1:n_global]
    n_segments = div(length(chromosome) - n_global, n_segment_specific)
    seg_list = [chromosome[n_global + (s - 1) * n_segment_specific + 1 : n_global + s * n_segment_specific] for s in 1:n_segments]
    return constant, seg_list
end

function simulate_with_cps(cps, params)
    constant, seg_list = extract_segment_params(params)
    u0_curr = u0_full
    sim_segments = Matrix{Float64}[]
    for s in 1:(length(cps) + 1)
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end   = (s > length(cps)) ? n : cps[s]
        all_pars = @LArray [constant; seg_list[s]] parnames
        tspan_seg = (Float64(idx_start), Float64(idx_end))
        raw_seg = example_ode_model(all_pars, tspan_seg, u0_curr)
        sim_seg = hasproperty(raw_seg, :u) ? reduce(hcat, raw_seg.u) : raw_seg
        push!(sim_segments, sim_seg)
        u0_curr = sim_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end

function main()
    cps_true = sort(unique(Int.(CSV.read(joinpath(TASK_A_DIR, "covid_detected_cps_origset_penalty_zero.csv"), DataFrame).cp)))
    params_true = Float64.(CSV.read(joinpath(TASK_A_DIR, "covid_params_origset_penalty_zero.csv"), DataFrame).value)

    println("[synthetic] True CPs: $cps_true")
    sim_true = simulate_with_cps(cps_true, params_true)

    # Apply same 14-day moving-average smoothing as real data
    function ma14(x)
        return conv(x, ones(14) ./ 14)[7:end-6]
    end
    # smooth cumulative death and vaccination
    # (simplified: use Smoothers.hma if available)
    sim_smoothed = copy(sim_true)
    # We keep the clean model output; the alternative-CPD scripts will apply the same smoothing.

    # Add log-normal noise to the 5 observed channels
    rng = MersenneTwister(SEED)
    clean_observed = Matrix{Float64}(undef, 5, n)
    observed = Matrix{Float64}(undef, 5, n)
    for (k, r) in enumerate(data_indices)
        y = sim_true[r, :]
        clean_observed[k, :] = y
        observed[k, :] = max.(y .* exp.(NOISE_SIGMA .* randn(rng, length(y))), 0.0)
    end

    # Save
    CSV.write(joinpath(OUT_DIR, "synthetic_clean.csv"),
              DataFrame(clean_observed', ["infected", "hospitalized", "icu", "death", "vaccination"]))
    CSV.write(joinpath(OUT_DIR, "synthetic_observed.csv"),
              DataFrame(observed', ["infected", "hospitalized", "icu", "death", "vaccination"]))
    CSV.write(joinpath(OUT_DIR, "true_cps.csv"), DataFrame(cp=cps_true))

    println("[synthetic] Saved clean/observed data and true CPs to $OUT_DIR")
end

main()
