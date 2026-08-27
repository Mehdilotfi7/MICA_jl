#!/usr/bin/env julia
# Diagnostic: compare original κ=0 fit vs. new BIC fit.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using CSV, DataFrames, Statistics
using OrdinaryDiffEq, Smoothers
using LabelledArrays

EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results")

load_params(csv_path) = Float64.(CSV.read(csv_path, DataFrame)[:, hasproperty(CSV.read(csv_path, DataFrame), :value) ? :value : :params])
load_cps(csv_path) = Int.(CSV.read(csv_path, DataFrame)[:, hasproperty(CSV.read(csv_path, DataFrame), :cp) ? :cp : :detected_cp])

# ---------- model ----------
fdelta(t, delta, t0=0.0) = 1 + delta * cos(2π * ((t - t0) / 365))
log_transform(data, threshold=1) = [val >= threshold ? log(val) : 0 for val in data]

function CovModel!(du, u, p, t)
    S, E0, E1, I0, I1, I2, I3, R, D, Cases, V = u
    N = S + E0 + E1 + I0 + I1 + I2 + I3 + R + D
    eps0 = p.ᴺε₀; eps1 = p.ᴺε₁
    gamma0 = p.ᴺγ₀; gamma1 = p.ᴺγ₁; gamma2 = p.ᴺγ₂; gamma3 = p.ᴺγ₃
    pp1 = p.ᴺp₁; pp12 = p.ᴺp₁₂; pp23 = p.ᴺp₂₃
    pp1D = p.ᴺp₁D; pp2D = p.ᴺp₂D; pp3D = p.ᴺp₃D
    delta = p.δ; beta = p.ᴺβ; omega = p.ω
    nu = t < 330 ? 0.0 : p.ν
    beta_eff = beta * fdelta(t, delta) * S * (E1 + I0 + I1)
    du[1]  = -(beta_eff) / N + omega * R - nu * S
    du[2]  =  (beta_eff / N) - (eps0 * E0)
    du[3]  =  (eps0 * E0) - (eps1 * E1)
    du[4]  =  ((1 - pp1) * eps1 * E1) - (gamma0 * I0)
    du[5]  =  (pp1 * eps1 * E1) - (gamma1 * I1)
    du[6]  =  (pp12 * gamma1 * I1) - (gamma2 * I2)
    du[7]  =  (pp23 * gamma2 * I2) - (gamma3 * I3)
    du[8]  =  gamma0 * I0 +
              (1 - pp12 - pp1D) * gamma1 * I1 +
              (1 - pp23 - pp2D) * gamma2 * I2 +
              (1 - pp3D) * gamma3 * I3 - omega * R + nu * S
    du[9]  =  (pp1D * gamma1 * I1) + (pp2D * gamma2 * I2) + (pp3D * gamma3 * I3)
    du[10] =  pp1 * eps1 * E1
    du[11] =  nu * S
end

function example_ode_model(params, tspan, u0)
    if tspan[2] < 330
        params[:ν] = 0.0
    end
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1.0e-6, reltol=1.0e-6,
                isoutofdomain=(u, p, t) -> any(x -> x < 0, u))
    return SciMLBase.successful_retcode(sol) ? sol[:, :] : fill(NaN, length(u0), Int(tspan[2] - tspan[1]) + 1)
end

# ---------- load data ----------
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

parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω, :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
n_global = 8
n_seg = 8
u0_full = [83129285 - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

cps_orig = load_cps(joinpath(EXAMPLE_DIR, "results_detected_cp_penalty0_ts10_pop150.csv"))
params_orig = load_params(joinpath(EXAMPLE_DIR, "results_params_penalty0__ts10_pop150.csv"))
cps_bic = load_cps(joinpath(OUT_DIR, "covid_detected_cps_bic.csv"))
params_bic = load_params(joinpath(OUT_DIR, "covid_params_bic.csv"))

function simulate(cps, params)
    constant = params[1:n_global]
    n_segments = div(length(params) - n_global, n_seg)
    segs = [params[n_global + (s - 1) * n_seg + 1 : n_global + s * n_seg] for s in 1:n_segments]
    u0 = u0_full
    out = Matrix{Float64}[]
    for s in 1:(length(cps) + 1)
        idx_start = (s == 1) ? 1 : cps[s - 1] + 1
        idx_end = (s > length(cps)) ? size(data_CP, 2) : cps[s]
        pp = @LArray [constant; segs[s]] parnames
        raw = example_ode_model(pp, (Float64(idx_start - 1), Float64(idx_end - 1)), u0)
        sim_seg = hasproperty(raw, :u) ? reduce(hcat, raw.u) : raw
        push!(out, sim_seg)
        u0 = sim_seg[:, end]
    end
    return reduce(hcat, out)
end

function metrics(sim)
    rows = [5, 6, 7, 9, 11]
    log_loss = sum(
        sum(abs, log_transform(sim[r, :]) .- log_transform(data_CP[k, :]))
        for (k, r) in enumerate(rows)
    )
    raw_mse = mean(
        sum((sim[r, :] .- data_CP[k, :]).^2)
        for (k, r) in enumerate(rows)
    )
    raw_mae = mean(
        sum(abs.(sim[r, :] .- data_CP[k, :]))
        for (k, r) in enumerate(rows)
    )
    return log_loss, raw_mse, raw_mae
end

println("="^70)
for (label, cps, params) in [
    ("original κ=0", cps_orig, params_orig),
    ("new BIC", cps_bic, params_bic)
]
    sim = simulate(cps, params)
    ll, rmse, rmae = metrics(sim)
    println(label)
    println("  CPs               : $cps ($(length(cps)) CPs)")
    println("  log-loss (SAD)    : $(round(ll, digits=2))")
    println("  raw MSE (channels): $(round(rmse, digits=2))")
    println("  raw MAE (channels): $(round(rmae, digits=2))")
    println()
end
