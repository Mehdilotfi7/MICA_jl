#!/usr/bin/env julia
# Practical identifiability analysis for the COVID-19 11-compartment model.
# Computes the sensitivity of model outputs (log scale) to parameter changes
# around the best-fit parameter set from Task A (zero penalty).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using Mica
using CSV, DataFrames, Statistics, LinearAlgebra
using OrdinaryDiffEq, Smoothers
using LabelledArrays
using Printf

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl", "examples", "Covid-model")
const TASK_A_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A", "results_penalty_zero")
const OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_E")
mkpath(OUT_DIR)

# ---------- model definitions ----------
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

# ---------- data ----------
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

# ---------- load best fit ----------
cp_file = joinpath(TASK_A_DIR, "covid_detected_cps_origset_penalty_zero.csv")
param_file = joinpath(TASK_A_DIR, "covid_params_origset_penalty_zero.csv")
if !isfile(cp_file) || !isfile(param_file)
    error("Task A zero-penalty outputs not found.")
end
cps = sort(unique(Int.(CSV.read(cp_file, DataFrame).cp)))
params_best = Float64.(CSV.read(param_file, DataFrame).value)

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
    n = size(data_CP, 2)
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

# Use log-transformed, observed data-matched outputs for sensitivity
function output_vector(sim)
    # flatten the selected rows into a single vector
    vcat([log_transform(sim[r, :]) for r in data_indices]...)
end

function main()
    println("[Task E] Simulating best-fit model...")
    sim_best = simulate_full(cps, params_best)
    y_best = output_vector(sim_best)
    n_params = length(params_best)

    println("[Task E] Computing finite-difference sensitivity matrix ($n_params params)...")
    # Relative perturbation
    eps_rel = 1e-4
    J = zeros(length(y_best), n_params)
    for i in 1:n_params
        p_plus = copy(params_best)
        h = eps_rel * max(abs(p_plus[i]), 1e-8)
        p_plus[i] += h
        sim_plus = simulate_full(cps, p_plus)
        if any(isnan, sim_plus)
            @warn "NaN simulation for parameter $i"
            continue
        end
        y_plus = output_vector(sim_plus)
        J[:, i] = (y_plus .- y_best) ./ h
    end

    # Fisher information approximation
    FIM = J' * J
    ev = eigen(FIM)
    eigvals_sorted = sort(real.(ev.values); rev=true)
    eigvecs_sorted = ev.vectors[:, sortperm(real.(ev.values); rev=true)]
    condition_number = maximum(eigvals_sorted) / max(minimum(eigvals_sorted), 1e-300)

    # Parameter labels
    param_labels = String[]
    for i in 1:n_global
        push!(param_labels, string(parnames[i]) * "_global")
    end
    n_seg_total = length(cps) + 1
    for s in 1:n_seg_total
        for i in 1:n_segment_specific
            push!(param_labels, string(parnames[n_global + i]) * "_seg$(s)")
        end
    end
    if length(param_labels) < n_params
        param_labels = [param_labels; ["param_$i" for i in (length(param_labels)+1):n_params]]
    elseif length(param_labels) > n_params
        param_labels = param_labels[1:n_params]
    end

    # Sensitivity norm per parameter
    sens_norm = sqrt.(sum(J .^ 2, dims=1)[:])
    sens_df = DataFrame(parameter=param_labels, sensitivity_norm=sens_norm)
    sort!(sens_df, :sensitivity_norm)
    CSV.write(joinpath(OUT_DIR, "parameter_sensitivity_norms.csv"), sens_df)

    # Eigenvalue spectrum
    eig_df = DataFrame(eigenvalue_index=1:length(eigvals_sorted), eigenvalue=eigvals_sorted)
    CSV.write(joinpath(OUT_DIR, "fim_eigenvalue_spectrum.csv"), eig_df)

    # Eigenvectors for smallest 5 modes (non-identifiable directions)
    n_modes = min(5, n_params)
    mode_df = DataFrame(parameter=param_labels)
    for k in 1:n_modes
        mode_df[!, "mode_$(k)_ev_$(round(eigvals_sorted[end-k+1], sigdigits=3))"] = eigvecs_sorted[:, end-k+1]
    end
    CSV.write(joinpath(OUT_DIR, "fim_smallest_modes.csv"), mode_df)

    # Report
    report_lines = String[]
    push!(report_lines, "# Task E — Identifiability analysis for the COVID-19 model")
    push!(report_lines, "")
    push!(report_lines, "- **Best-fit source:** Task A `results_penalty_zero`")
    push!(report_lines, "- **Number of change points:** $(length(cps))")
    push!(report_lines, "- **Number of fitted parameters:** $(n_params)")
    push!(report_lines, "- **Relative perturbation for finite differences:** $(eps_rel)")
    push!(report_lines, "- **Condition number of J'J (FIM):** $(@sprintf("%.4e", condition_number))")
    push!(report_lines, "")
    push!(report_lines, "## Interpretation")
    push!(report_lines, "")
    push!(report_lines, "A very large condition number indicates that some parameter directions are poorly identified: many parameter combinations produce nearly identical model outputs. This is expected for an 11-compartment SEIRD model with 15+ parameters, many of which are segment-specific.")
    push!(report_lines, "")
    push!(report_lines, "## Smallest eigenvalues of the FIM")
    push!(report_lines, "")
    push!(report_lines, "| rank_from_bottom | eigenvalue |")
    push!(report_lines, "|---|---|")
    for k in 1:min(10, length(eigvals_sorted))
        push!(report_lines, "| $(k) | $(@sprintf("%.6e", eigvals_sorted[end-k+1])) |")
    end
    push!(report_lines, "")
    push!(report_lines, "## Least sensitive parameters")
    push!(report_lines, "")
    push!(report_lines, "| parameter | sensitivity_norm |")
    push!(report_lines, "|---|---|")
    for r in eachrow(sens_df[1:min(10, nrow(sens_df)), :])
        push!(report_lines, "| $(r.parameter) | $(@sprintf("%.6e", r.sensitivity_norm)) |")
    end
    push!(report_lines, "")
    push!(report_lines, "## Files created")
    push!(report_lines, "- `parameter_sensitivity_norms.csv`")
    push!(report_lines, "- `fim_eigenvalue_spectrum.csv`")
    push!(report_lines, "- `fim_smallest_modes.csv`")

    open(joinpath(OUT_DIR, "report.md"), "w") do f
        write(f, join(report_lines, "\n") * "\n")
    end

    println("[Task E] Condition number: $(condition_number)")
    println("[Task E] Output saved to $(OUT_DIR)")
end

main()
