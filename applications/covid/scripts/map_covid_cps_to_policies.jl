#!/usr/bin/env julia
# Map detected COVID-19 change-points to German policy interventions.
# Uses the policy timeline in revision/outputs/TASK_C/german_policy_timeline.csv
# and CP files produced by Task A.

using CSV, DataFrames, Dates

const BASE_DATE = Date("2020-01-27")
const TASK_A_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_A")
const TASK_C_DIR = joinpath(@__DIR__, "..", "..", "..", "outputs",  "TASK_C")
mkpath(TASK_C_DIR)

const POLICY_FILE = joinpath(TASK_C_DIR, "german_policy_timeline.csv")

function cp_date(cp_idx::Int)
    return BASE_DATE + Day(cp_idx - 1)
end

function load_policy_timeline(path)
    df = CSV.read(path, DataFrame)
    df.date = Date.(df.date, "yyyy-mm-dd")
    return df
end

function nearest_policy(cp_d::Date, policies::DataFrame; max_window::Int=14)
    best_i = 0
    best_diff = max_window + 1
    for i in 1:nrow(policies)
        diff = abs(Day(cp_d - policies.date[i]).value)
        if diff < best_diff
            best_diff = diff
            best_i = i
        end
    end
    if best_i == 0
        return missing, missing
    end
    return policies.date[best_i], policies.event[best_i]
end

function map_cps_for_result(result_name::String, cps::Vector{Int}, policies::DataFrame)
    rows = DataFrame(
        result = String[],
        cp_index = Int[],
        date = Date[],
        days_from_start = Int[],
        nearest_policy_date = Union{Date,Missing}[],
        days_apart = Int[],
        policy_event = Union{String,Missing}[]
    )
    for cp in cps
        d = cp_date(cp)
        pd, ev = nearest_policy(d, policies)
        days_apart = ismissing(pd) ? missing : abs(Day(d - pd).value)
        push!(rows, (result_name, cp, d, cp - 1, pd, days_apart, ev))
    end
    return rows
end

function main()
    policies = load_policy_timeline(POLICY_FILE)

    # Result sets to map. Each is (label, folder, objective_label).
    result_specs = [
        ("Task A — zero penalty", "results_penalty_zero", "penalty_zero"),
        ("Task A — BIC", "results_bic", "bic"),
        ("Task A — MDL", "results_mdl", "mdl"),
        ("Task A — AIC", "results_aic", "aic"),
    ]

    all_rows = DataFrame()
    for (label, folder, obj_label) in result_specs
        cp_file = joinpath(TASK_A_DIR, folder, "covid_detected_cps_origset_$(obj_label).csv")
        if !isfile(cp_file)
            @warn "CP file not found: $cp_file"
            continue
        end
        cps = sort(unique(Int.(CSV.read(cp_file, DataFrame).cp)))
        rows = map_cps_for_result(label, cps, policies)
        append!(all_rows, rows; cols=:union)

        out_file = joinpath(TASK_C_DIR, "cp_policy_map_$(replace(label, " " => "_")).csv")
        CSV.write(out_file, rows)
        println("Saved: $out_file")
    end

    # Combined table
    combined_csv = joinpath(TASK_C_DIR, "cp_policy_map_all.csv")
    CSV.write(combined_csv, all_rows)
    println("Saved: $combined_csv")

    # Markdown report
    md_lines = String[]
    push!(md_lines, "# Task C — COVID-19 change-point to policy-intervention mapping")
    push!(md_lines, "")
    push!(md_lines, "Base date: **$BASE_DATE** (index 1).")
    push!(md_lines, "")
    push!(md_lines, "Mapping window: ±14 days around each detected change-point.")
    push!(md_lines, "")
    push!(md_lines, "## Combined mapping")
    push!(md_lines, "")
    push!(md_lines, "| result | cp_index | date | nearest_policy_date | days_apart | policy_event |")
    push!(md_lines, "|---|---|---|---|---|---|")
    for r in eachrow(all_rows)
        pd = ismissing(r.nearest_policy_date) ? "—" : string(r.nearest_policy_date)
        ev = ismissing(r.policy_event) ? "—" : r.policy_event
        da = ismissing(r.days_apart) ? "—" : string(r.days_apart)
        push!(md_lines, "| $(r.result) | $(r.cp_index) | $(r.date) | $(pd) | $(da) | $(ev) |")
    end
    push!(md_lines, "")
    push!(md_lines, "## Policy timeline")
    push!(md_lines, "")
    push!(md_lines, "| date | event |")
    push!(md_lines, "|---|---|")
    for r in eachrow(policies)
        push!(md_lines, "| $(r.date) | $(r.event) |")
    end

    md_file = joinpath(TASK_C_DIR, "report.md")
    open(md_file, "w") do f
        write(f, join(md_lines, "\n") * "\n")
    end
    println("Saved: $md_file")
end

main()
