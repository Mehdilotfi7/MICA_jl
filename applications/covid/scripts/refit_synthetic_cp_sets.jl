#!/usr/bin/env julia
# Refit the SEIRD model for selected synthetic CPD CP sets and compare fitting quality.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using OrdinaryDiffEq
using Optim
using LabelledArrays
using Printf
using Base.Threads
using Logging
Logging.disable_logging(Logging.Warn)

const SYNTH_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_H_synthetic")
const CP_SETS_DIR = joinpath(SYNTH_DIR, "cp_sets")
const OUT_DIR = SYNTH_DIR
mkpath(OUT_DIR)

const n_global = 8
const n_segment_specific = 8
const u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const data_indices = [5, 6, 7, 9, 11]
const n = 400
const lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
               0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
const upper = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
               0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]
const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                   :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)

# ---------- model ----------
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
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    if SciMLBase.successful_retcode(sol)
        return sol[:, :]
    else
        return fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
    end
end

function loss_abs_log(obs, sim)
    total = 0.0
    for (k, r) in enumerate(data_indices)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(obs[k, :]))
    end
    return total
end

# ---------- load data ----------
obs_df = CSV.read(joinpath(SYNTH_DIR, "synthetic_observed.csv"), DataFrame)
obs_data = collect(Matrix(obs_df[:, ["infected", "hospitalized", "icu", "death", "vaccination"]])')  # 5 x 400

# initial parameter guess: use best-fit from real data as a starting point
const x0_global = Float64.(CSV.read(joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A", "results_penalty_zero", "covid_params_origset_penalty_zero.csv"), DataFrame).value)

function refit_loss(cps, data)
    n_segments = length(cps) + 1
    n_params = n_global + n_segments * n_segment_specific
    lb = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
    ub = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]
    x0_use = x0_global[1:min(length(x0_global), n_params)]
    if length(x0_use) < n_params
        x0_use = [x0_use; repeat(x0_use[end-n_segment_specific+1:end], div(n_params - length(x0_use), n_segment_specific))]
    end

    ode_spec = ODEModelSpec(example_ode_model, x0_use, u0_full, (0.0, Float64(n)))
    mm = ModelManager(ode_spec)

    opt = OptimOptimizer(Fminbox(LBFGS()),
                         options=Optim.Options(show_trace=false, iterations=150))
    loss, _ = optimize_with_changepoints(
        Mica.objective_function, x0_use, parnames, cps, (lb, ub), opt,
        n_global, n_segment_specific, mm, loss_abs_log, data
    )
    return loss
end

function main()
    cp_files = sort(filter(f -> endswith(f, ".csv"), readdir(CP_SETS_DIR)))
    subset_file = joinpath(SYNTH_DIR, "synthetic_refit_subset.csv")
    use_subset = isfile(subset_file)
    subset_methods = use_subset ? Set(CSV.read(subset_file, DataFrame).method) : Set{String}()

    methods = String[]
    cp_sets = Vector{Int}[]
    for f in cp_files
        method = replace(f, ".csv" => "")
        use_subset && !(method in subset_methods) && continue
        df = CSV.read(joinpath(CP_SETS_DIR, f), DataFrame)
        cps = sort(unique(Int.(df.cp)))
        cps = filter(c -> 11 <= c <= n - 10, cps)
        push!(methods, method)
        push!(cp_sets, cps)
    end

    # baseline: true CPs
    true_cps = sort(unique(Int.(CSV.read(joinpath(SYNTH_DIR, "true_cps.csv"), DataFrame).cp)))
    push!(methods, "TRUE_CPS")
    push!(cp_sets, true_cps)

    n_jobs = length(methods)
    println("[synthetic refit] Refitting $n_jobs CP sets on $(Threads.nthreads()) threads...")

    results = DataFrame(method=String[], n_cps=Int[], cps=String[], refit_loss=Float64[], elapsed=Float64[], error=String[])
    lock_results = ReentrantLock()
    n_threads = min(Threads.nthreads(), 8)
    chunk_size = ceil(Int, n_jobs / n_threads)

    @sync for tid in 1:n_threads
        Threads.@spawn begin
            start_idx = (tid - 1) * chunk_size + 1
            end_idx = min(tid * chunk_size, n_jobs)
            for i in start_idx:end_idx
                method = methods[i]
                cps = cp_sets[i]
                println("  [$tid] Refitting $method (n_cps=$(length(cps)))"); flush(stdout)
                t0 = time()
                local loss, err
                try
                    loss = refit_loss(cps, obs_data)
                    err = ""
                catch e
                    loss = Inf
                    err = sprint(showerror, e)
                    @warn "  Refit failed for $method: $err"
                end
                elapsed = time() - t0
                lock(lock_results) do
                    push!(results, (method, length(cps), join(cps, ";"), loss, elapsed, err))
                    CSV.write(joinpath(OUT_DIR, "synthetic_refit_summary.csv"), results)
                end
            end
        end
    end

    sort!(results, :refit_loss)
    CSV.write(joinpath(OUT_DIR, "synthetic_refit_summary.csv"), results)
    println("[synthetic refit] Done.")
    println(results)
end

main()
