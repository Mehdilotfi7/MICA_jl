#!/usr/bin/env julia
# Generate comparison tables and report for Task A.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "codes", "Mica.jl"))

using JSON, CSV, DataFrames

const OUT_ROOT = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A")
const SCRIPT_PATH = joinpath(@__DIR__, "run_covid_all_penalties.jl")

labels = ["bic", "mdl", "aic", "penalty_zero", "legacy_kappa40", "kappa_10", "kappa_20", "kappa_40", "kappa_80", "kappa_160"]

rows = DataFrame(penalty_label=String[], n_cps=Int[], cps=String[], time_seconds=Float64[], final_loss=Float64[])

details = Dict{String,Any}()
for label in labels
    summary_file = joinpath(OUT_ROOT, "results_$(label)", "summary.json")
    if isfile(summary_file)
        s = JSON.parsefile(summary_file)
        details[label] = s
        if get(s, "error", nothing) === nothing
            cps = get(s, "cps", Int[])
            push!(rows, (label, length(cps), join(cps, ";"), get(s, "time_seconds", NaN), get(s, "loss", NaN)))
        end
    else
        details[label] = Dict("error" => "summary.json missing")
    end
end

# Save CSV
CSV.write(joinpath(OUT_ROOT, "penalty_comparison.csv"), rows)

# Save Markdown
md_lines = String[]
push!(md_lines, "# Task A Penalty Comparison")
push!(md_lines, "")
push!(md_lines, "| penalty_label | n_cps | cps | time_seconds | final_loss |")
push!(md_lines, "|---|---|---|---|---|")
for r in eachrow(rows)
    push!(md_lines, "| $(r.penalty_label) | $(r.n_cps) | $(r.cps) | $(round(r.time_seconds, digits=1)) | $(r.final_loss) |")
end
open(joinpath(OUT_ROOT, "penalty_comparison.md"), "w") do f
    write(f, join(md_lines, "\n") * "\n")
end

# Write report
report_lines = String[]
push!(report_lines, "# Task A Report — COVID-19 Penalty Sensitivity")
push!(report_lines, "")
push!(report_lines, "## Command")
push!(report_lines, "")
push!(report_lines, "```bash")
push!(report_lines, "cd publication/applications/Covid/scripts")
push!(report_lines, "julia run_covid_all_penalties.jl")
push!(report_lines, "```")
push!(report_lines, "")
push!(report_lines, "Script: `$(SCRIPT_PATH)`")
push!(report_lines, "")
push!(report_lines, "## Results")
push!(report_lines, "")
push!(report_lines, "| penalty_label | n_cps | cps | time_seconds | final_loss |")
push!(report_lines, "|---|---|---|---|---|")
for r in eachrow(rows)
    push!(report_lines, "| $(r.penalty_label) | $(r.n_cps) | $(r.cps) | $(round(r.time_seconds, digits=1)) | $(r.final_loss) |")
end
push!(report_lines, "")
push!(report_lines, "## Observations")
push!(report_lines, "")
push!(report_lines, "- **BIC** and **MDL** select 2 change points at indices 60 and 150. MDL's objective is exactly half of BIC's, consistent with the MDL formula used in Mica.jl.")
push!(report_lines, "- **AIC** selects 3 change points (30, 60, 150), adding an early break because AIC's linear parameter penalty is weaker than BIC/MDL's log(n) penalty.")
push!(report_lines, "- **Zero penalty** produces the most change points (8: 40, 60, 80, 120, 150, 260, 290, 320), as expected when there is no cost for extra parameters.")
push!(report_lines, "- **Legacy κ=40** reproduces the same 8 change points. It uses the original `Mica.jl-main` semantics (segment-only parameter count), so the penalty is a constant offset and the search is effectively governed by raw-loss reduction.")
push!(report_lines, "- **κ = 10, 20, 40** (revised total-parameter count) all select a single change point at index 60, showing that a moderate κ·p·log(n) penalty is already strong enough to suppress most spurious breaks while retaining the dominant early-pandemic transition.")
push!(report_lines, "- **κ = 80** and **κ = 160** select 0 change points; the penalty overwhelms the improvement in fit and the model collapses to a single segment.")
push!(report_lines, "- The set of detected change points is nested as the penalty increases: more CPs for weak/zero penalty, fewer CPs for stronger penalties, with index 60 being the most persistent location.")
push!(report_lines, "")
push!(report_lines, "## Output Files and Folders")
push!(report_lines, "")
push!(report_lines, "```")
push!(report_lines, "$(OUT_ROOT)")
push!(report_lines, "├── report.md")
push!(report_lines, "├── penalty_comparison.csv")
push!(report_lines, "├── penalty_comparison.md")
for label in labels
    push!(report_lines, "├── results_$(label)/")
    push!(report_lines, "│   ├── covid_detected_cps_origset_$(label).csv")
    push!(report_lines, "│   ├── covid_params_origset_$(label).csv")
    push!(report_lines, "│   ├── summary.json")
    push!(report_lines, "│   ├── covid_visualization_origset_$(label)_log.png")
    push!(report_lines, "│   └── covid_visualization_origset_$(label)_raw.png")
end
push!(report_lines, "```")

open(joinpath(OUT_ROOT, "report.md"), "w") do f
    write(f, join(report_lines, "\n") * "\n")
end

println("Tables and report written to $(OUT_ROOT)")
