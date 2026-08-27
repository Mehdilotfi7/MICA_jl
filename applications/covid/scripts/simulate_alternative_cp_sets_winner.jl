#!/usr/bin/env julia
# Refit the SEIRD model for every alternative CP set and save fitted parameters + simulated trajectories.
# Usage: julia simulate_alternative_cp_sets_winner.jl <label> <params_csv> <cps_csv> <cp_sets_dir> <out_dir>

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics
using OrdinaryDiffEq, Smoothers
using Optim
using LabelledArrays
using Printf
using Base.Threads
using Logging
Logging.disable_logging(Logging.Warn)

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")

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

cases_CP = CSV.read(joinpath(EXAMPLE_DIR, "case_rki_daily.csv"), DataFrame).total
hospital_CP = CSV.read(joinpath(EXAMPLE_DIR, "Hospitalization_rki_daily.csv"), DataFrame).total
death_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "death_rki_daily.csv"), DataFrame).Todesfaelle_neu)
icu_CP = CSV.read(joinpath(EXAMPLE_DIR, "icu_rki_daily.csv"), DataFrame).total
vacc_CP = cumsum(CSV.read(joinpath(EXAMPLE_DIR, "vaccination_rki_daily_allShots.csv"), DataFrame).Total)

data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
max_length = maximum(length, data_CP)
data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
data_CP = [vector[1:400] for vector in data_CP]
data_CP[1] = hma(data_CP[1], 14)
data_CP[4] = hma(data_CP[4], 14)
data_CP[5] = hma(data_CP[5], 14)
data_CP = Matrix(reduce(hcat, data_CP)')

const parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                   :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
const n_global = 8
const n_segment_specific = 8
const u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
const data_indices = [5, 6, 7, 9, 11]
const n = size(data_CP, 2)
const lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
               0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 10e-5]
const upper = [0.3, 1/3, 1/11.2, 1/5, 1/10.9, 1/5, 1/8, 0.012,
               0.8, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1]

function loss_abs_log(obs, sim)
    total = 0.0
    for (k, r) in enumerate(data_indices)
        total += sum(abs, log_transform(sim[r, :]) .- log_transform(obs[k, :]))
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
    u0_curr = u0_full
    sim_segments = Matrix{Float64}[]
    for s in 1:(length(cps) + 1)
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end   = (s > length(cps)) ? n : cps[s]
        all_pars = @LArray [constant; seg_list[s]] parnames
        tspan_seg = (Float64(idx_start), Float64(idx_end))
        raw_seg = example_ode_model(all_pars, tspan_seg, u0_curr)
        sim_seg = raw_seg
        push!(sim_segments, sim_seg)
        u0_curr = sim_seg[:, end]
    end
    return reduce(hcat, sim_segments)
end

function parameter_labels(n_cps)
    labs = String[]
    for i in 1:n_global
        push!(labs, string(parnames[i]) * "_global")
    end
    for s in 1:(n_cps+1)
        for i in 1:n_segment_specific
            push!(labs, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    return labs
end

# ---------- CLI ----------
length(ARGS) >= 5 || error("Usage: julia simulate_alternative_cp_sets_winner.jl <label> <params_csv> <cps_csv> <cp_sets_dir> <out_dir>")
const LABEL = ARGS[1]
const PARAMS_CSV = ARGS[2]
const CPS_CSV = ARGS[3]
const CP_SETS_DIR = ARGS[4]
const OUT_DIR = ARGS[5]
mkpath(OUT_DIR)

const params_best = Float64.(CSV.read(PARAMS_CSV, DataFrame).value)

function refit(cps, x0)
    n_segments = length(cps) + 1
    n_params = n_global + n_segments * n_segment_specific
    lb = [lower[1:n_global]; repeat(lower[n_global+1:end], n_segments)]
    ub = [upper[1:n_global]; repeat(upper[n_global+1:end], n_segments)]
    x0_use = x0[1:min(length(x0), n_params)]
    if length(x0_use) < n_params
        x0_use = [x0_use; repeat(x0_use[end-n_segment_specific+1:end], div(n_params - length(x0_use), n_segment_specific))]
    end

    ode_spec = ODEModelSpec(example_ode_model, x0_use, u0_full, (0.0, Float64(n)))
    mm = ModelManager(ode_spec)

    opt = OptimOptimizer(Fminbox(LBFGS()),
                         options=Optim.Options(show_trace=false, iterations=150, time_limit=300.0))
    loss, params = optimize_with_changepoints(
        Mica.objective_function, x0_use, parnames, cps, (lb, ub), opt,
        n_global, n_segment_specific, mm, loss_abs_log, data_CP
    )
    return loss, params
end

function save_simulation(method, cps, params, loss, elapsed, err)
    sims_dir = joinpath(OUT_DIR, "simulations")
    params_dir = joinpath(OUT_DIR, "fitted_params")
    mkpath(sims_dir)
    mkpath(params_dir)

    # Save fitted parameters
    labs = parameter_labels(length(cps))
    pvals = params[1:length(labs)]
    CSV.write(joinpath(params_dir, "$(method).csv"),
              DataFrame(parameter=labs, value=pvals))

    # Save simulated observed channels
    sim = simulate_full(cps, params)
    sim_df = DataFrame(
        day = 1:size(sim, 2),
        infected = sim[data_indices[1], :],
        hospitalized = sim[data_indices[2], :],
        icu = sim[data_indices[3], :],
        death = sim[data_indices[4], :],
        vaccinated = sim[data_indices[5], :]
    )
    CSV.write(joinpath(sims_dir, "$(method).csv"), sim_df)

    return (method=method, n_cps=length(cps), cps=join(cps, ";"),
            refit_loss=loss, elapsed=elapsed, error=err)
end

function main()
    cp_files = sort(filter(f -> endswith(f, ".csv"), readdir(CP_SETS_DIR)))
    methods = String[]
    cp_sets = Vector{Int}[]

    for f in cp_files
        method = replace(f, ".csv" => "")
        df = CSV.read(joinpath(CP_SETS_DIR, f), DataFrame)
        cps = sort(unique(Int.(df.cp)))
        cps = filter(c -> 11 <= c <= n - 10, cps)
        push!(methods, method)
        push!(cp_sets, cps)
    end

    # also include MICA winner CP set and ZERO baseline
    mica_cps = sort(unique(Int.(CSV.read(CPS_CSV, DataFrame).cp)))
    push!(methods, "MICA_$(LABEL)")
    push!(cp_sets, mica_cps)
    push!(methods, "ZERO")
    push!(cp_sets, Int[])

    n_jobs = length(methods)
    println("[simulate $LABEL] Refitting + saving simulations for $n_jobs CP sets...")

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
                println("  [$LABEL][$tid] Simulating $method (n_cps=$(length(cps)))"); flush(stdout)
                t0 = time()
                local loss, params, err
                try
                    loss, params = refit(cps, params_best)
                    err = ""
                catch e
                    loss = Inf
                    params = fill(NaN, length(params_best))
                    err = sprint(showerror, e)
                    @warn "  Refit failed for $method: $err"
                end
                elapsed = time() - t0
                local row
                try
                    row = save_simulation(method, cps, params, loss, elapsed, err)
                catch e
                    @warn "  Saving simulation failed for $method: $e"
                    row = (method=method, n_cps=length(cps), cps=join(cps, ";"),
                           refit_loss=loss, elapsed=elapsed, error=err * "; save failed: " * sprint(showerror, e))
                end
                lock(lock_results) do
                    push!(results, row)
                    CSV.write(joinpath(OUT_DIR, "refit_summary.csv"), results)
                end
            end
        end
    end

    println("[simulate $LABEL] Done. Saved $(nrow(results)) simulations to $OUT_DIR/simulations/")
end

main()
