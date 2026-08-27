#!/usr/bin/env julia
# Structural identifiability analysis for the baseline (no-changepoint)
# COVID-19 SEIRD model using StructuralIdentifiability.jl.
#
# This uses a rational simplification of the model:
#   - no seasonality (delta = 0)
#   - vaccination is a constant rate nu over the whole horizon
# The observations are the same five channels used in MICA:
#   infected (cumulative Cases), hospitalized (I1), icu (I2), death (D), vaccinated (V).

using Pkg
Pkg.activate(@__DIR__)

using StructuralIdentifiability
using Printf

@time ode = @ODEmodel(
    S'(t) = -(beta * S(t) * (E1(t) + I0(t) + I1(t))) / (S(t) + E0(t) + E1(t) + I0(t) + I1(t) + I2(t) + I3(t) + R(t) + D(t)) + omega * R(t) - nu * S(t),
    E0'(t) = (beta * S(t) * (E1(t) + I0(t) + I1(t))) / (S(t) + E0(t) + E1(t) + I0(t) + I1(t) + I2(t) + I3(t) + R(t) + D(t)) - eps0 * E0(t),
    E1'(t) = eps0 * E0(t) - eps1 * E1(t),
    I0'(t) = (1 - p1) * eps1 * E1(t) - gamma0 * I0(t),
    I1'(t) = p1 * eps1 * E1(t) - gamma1 * I1(t),
    I2'(t) = p12 * gamma1 * I1(t) - gamma2 * I2(t),
    I3'(t) = p23 * gamma2 * I2(t) - gamma3 * I3(t),
    R'(t) = gamma0 * I0(t) + (1 - p12 - p1D) * gamma1 * I1(t) + (1 - p23 - p2D) * gamma2 * I2(t) + (1 - p3D) * gamma3 * I3(t) - omega * R(t) + nu * S(t),
    D'(t) = p1D * gamma1 * I1(t) + p2D * gamma2 * I2(t) + p3D * gamma3 * I3(t),
    Cases'(t) = p1 * eps1 * E1(t),
    V'(t) = nu * S(t),
    y1(t) = Cases(t),
    y2(t) = I1(t),
    y3(t) = I2(t),
    y4(t) = D(t),
    y5(t) = V(t)
)

println("\nRunning structural identifiability assessment...")
@time results = assess_identifiability(ode)

println("\n=== Structural identifiability results ===")
for (p, status) in sort(collect(results), by=first)
    println(@sprintf("%-10s -> %s", p, status))
end

# Write a simple report
open("si_report.md", "w") do f
    println(f, "# Structural identifiability of baseline COVID-19 model")
    println(f, "")
    println(f, "Tool: StructuralIdentifiability.jl")
    println(f, "Model: rational SEIRD without seasonality, constant vaccination rate.")
    println(f, "Observables: Cases (infected), I1 (hospitalized), I2 (icu), D (deaths), V (vaccinated).")
    println(f, "")
    println(f, "| parameter | identifiability |")
    println(f, "|---|---|")
    for (p, status) in sort(collect(results), by=first)
        println(f, "| $p | $status |")
    end
end

println("\nReport saved to si_report.md")
