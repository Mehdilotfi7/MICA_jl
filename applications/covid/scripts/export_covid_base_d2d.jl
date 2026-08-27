#!/usr/bin/env julia
# Export the baseline COVID best-fit parameters to a D2D MATLAB initializer.
# Usage: julia export_covid_base_d2d.jl <params_csv> <output_m>

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using CSV, DataFrames, Dates

const PARAM_MAP = Dict(
    "δ"     => "delta",
    "ᴺε₀"   => "eps0",
    "ᴺε₁"   => "eps1",
    "ᴺγ₀"   => "gamma0",
    "ᴺγ₁"   => "gamma1",
    "ᴺγ₂"   => "gamma2",
    "ᴺγ₃"   => "gamma3",
    "ω"     => "omega",
    "ᴺp₁"   => "p1",
    "ᴺβ"    => "beta",
    "ᴺp₁₂"  => "p12",
    "ᴺp₂₃"  => "p23",
    "ᴺp₁D"  => "p1D",
    "ᴺp₂D"  => "p2D",
    "ᴺp₃D"  => "p3D",
    "ν"     => "nu",
)

const LOWER = Dict(
    "δ" => 0.1,    "ᴺε₀" => 1/10,   "ᴺε₁" => 1/11.7, "ᴺγ₀" => 1/24,
    "ᴺγ₁" => 1/15.8,"ᴺγ₂" => 1/19,   "ᴺγ₃" => 1/27,   "ω" => 0.003,
    "ᴺp₁" => 0.0,  "ᴺβ" => 0.0,     "ᴺp₁₂" => 0.001, "ᴺp₂₃" => 0.001,
    "ᴺp₁D" => 0.001,"ᴺp₂D" => 0.001,"ᴺp₃D" => 0.001, "ν" => 10e-5,
)
const UPPER = Dict(
    "δ" => 0.3,    "ᴺε₀" => 1/3,    "ᴺε₁" => 1/11.2, "ᴺγ₀" => 1/5,
    "ᴺγ₁" => 1/10.9,"ᴺγ₂" => 1/5,   "ᴺγ₃" => 1/8,    "ω" => 0.012,
    "ᴺp₁" => 0.8,  "ᴺβ" => 8.0,    "ᴺp₁₂" => 0.5,   "ᴺp₂₃" => 0.5,
    "ᴺp₁D" => 0.5, "ᴺp₂D" => 0.5,  "ᴺp₃D" => 0.5,    "ν" => 0.1,
)

length(ARGS) >= 2 || error("Usage: julia export_covid_base_d2d.jl <params_csv> <output_m>")
params_csv = ARGS[1]
output_m = ARGS[2]

df = CSV.read(params_csv, DataFrame)
lookup = Dict{String,Float64}()
for row in eachrow(df)
    lookup[string(row.parameter)] = Float64(row.value)
end

pnames_d2d = String[]
values_log10 = Float64[]
lb_log10 = Float64[]
ub_log10 = Float64[]
fix_flags = Int[]

# Use the same order as the model .def file
for orig in ["δ", "ᴺε₀", "ᴺε₁", "ᴺγ₀", "ᴺγ₁", "ᴺγ₂", "ᴺγ₃", "ω",
             "ᴺp₁", "ᴺβ", "ᴺp₁₂", "ᴺp₂₃", "ᴺp₁D", "ᴺp₂D", "ᴺp₃D", "ν"]
    d2d = PARAM_MAP[orig]
    v = lookup[orig]
    lo = max(LOWER[orig], 1e-12)
    hi = UPPER[orig]
    push!(pnames_d2d, d2d)
    push!(values_log10, log10(v))
    push!(lb_log10, log10(lo))
    push!(ub_log10, log10(hi))
    push!(fix_flags, 1)  # all fitted
end

open(output_m, "w") do io
    println(io, "% Auto-generated D2D initial parameter file")
    println(io, "% Source: $(params_csv)")
    println(io, "% Generated: $(Dates.now())")
    println(io, "")

    println(io, "param_names = {")
    for p in pnames_d2d
        println(io, "    '$(p)';")
    end
    println(io, "};")
    println(io, "")

    for (name, vec) in [("values", values_log10), ("lb_p", lb_log10), ("ub_p", ub_log10), ("fix_flags", fix_flags)]
        println(io, "$(name) = [ ...")
        for v in vec
            println(io, "    $(v)")
        end
        println(io, "};")
        println(io, "")
    end
end

println("Wrote $(length(pnames_d2d)) parameters to $(output_m)")
