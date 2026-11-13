#!/usr/bin/env julia
# ============================================================
# Generate PLE figures for the toy-SIRD complexity experiment
# using the BreakpointProfiles.jl plotting pipeline (Plots.jl/GR),
# in the style of the first-revision COVID/wind-turbine figures.
#
#   julia plot_sird_ple.jl
#
# Per-panel display rules:
#   - dense profiles (>=6 points): PCHIP-smoothed curve (first-revision style)
#   - sparse profiles (<6 points): linear polyline through the profiled
#     points (the CI is computed by linear interpolation between exactly
#     these points, so only a linear display crosses the threshold exactly
#     at the CI edges)
#   - identifiable parameters: x window = CI +/- 0.75 * CI width and y
#     capped just above the threshold, so the gold CI band spans ~40% of
#     the panel and both threshold crossings are visible
#   - non-identifiable parameters: full profiled range (flatness across
#     the bounds is the message)
#
# Produces, for each config (M1-M4) and mode (conditional/joint):
#   - parameter_profile_curves.pdf / .png
#   - changepoint_profile_curves.pdf / .png   (conditional only)
# ============================================================

using Pkg
const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(joinpath(ROOT, "codes", "Mica.jl"))
push!(LOAD_PATH, joinpath(ROOT, "codes", "BreakpointProfiles_v2.jl"))
Pkg.instantiate()

using Mica
using BreakpointProfiles
using CSV, DataFrames
using Plots
gr()

const PLE_BASE = joinpath(@__DIR__, "..", "results")

# -------------------------------------------------------------
# PCHIP (Fritsch-Carlson monotone cubic), no extra dependency
# -------------------------------------------------------------
function pchip_eval(x::Vector{Float64}, y::Vector{Float64}, xs::Vector{Float64})
    n = length(x)
    n == 1 && return fill(y[1], length(xs))
    if n == 2
        return [y[1] + (y[2] - y[1]) / (x[2] - x[1]) * (xi - x[1]) for xi in xs]
    end
    h = diff(x)
    d = diff(y) ./ h
    m = zeros(n)
    m[1] = d[1]
    m[n] = d[n - 1]
    for k in 2:(n - 1)
        if d[k - 1] * d[k] <= 0
            m[k] = 0.0
        else
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        end
    end
    # End slopes (one-sided, with monotonicity clamp)
    m[1] = ((2 * h[1] + h[2]) * d[1] - h[1] * d[2]) / (h[1] + h[2])
    if sign(m[1]) != sign(d[1])
        m[1] = 0.0
    elseif sign(d[1]) != sign(d[2]) && abs(m[1]) > abs(3 * d[1])
        m[1] = 3 * d[1]
    end
    m[n] = ((2 * h[n - 1] + h[n - 2]) * d[n - 1] - h[n - 1] * d[n - 2]) / (h[n - 1] + h[n - 2])
    if sign(m[n]) != sign(d[n - 1])
        m[n] = 0.0
    elseif sign(d[n - 1]) != sign(d[n - 2]) && abs(m[n]) > abs(3 * d[n - 1])
        m[n] = 3 * d[n - 1]
    end
    out = Vector{Float64}(undef, length(xs))
    for (j, xi) in enumerate(xs)
        k = clamp(searchsortedlast(x, xi), 1, n - 1)
        t = (xi - x[k]) / h[k]
        h00 = 2t^3 - 3t^2 + 1
        h10 = t^3 - 2t^2 + t
        h01 = -2t^3 + 3t^2
        h11 = t^3 - t^2
        out[j] = h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]
    end
    return out
end

# Quadratic interpolation through exactly 3 points (unused for display
# decisions; kept for completeness).
function quad_eval(x::Vector{Float64}, y::Vector{Float64}, xs::Vector{Float64})
    A = hcat(x .^ 2, x, ones(3))
    a, b, c = A \ y
    return a .* xs .^ 2 .+ b .* xs .+ c
end

# -------------------------------------------------------------
# Loading
# -------------------------------------------------------------
function load_profiles(summary_path::String, curves_path::String)
    summary_df = CSV.read(summary_path, DataFrame)
    curves_df  = CSV.read(curves_path, DataFrame)
    profiles = ProfileResult[]
    for r in eachrow(summary_df)
        idx = r.index
        sub = filter(row -> row.index == idx, curves_df)
        vals = collect(sub.value)
        los  = collect(sub.loss)
        prof = ProfileResult(
            r.parameter, idx, r.best_value, r.best_loss,
            r.ci_lower, r.ci_upper, r.identifiable, r.threshold,
            vals, los, r.n_failed, r.best_found_loss, Float64[]
        )
        push!(profiles, prof)
    end
    return profiles
end

function load_cp_profiles(cp_path::String)
    df = CSV.read(cp_path, DataFrame)
    groups = groupby(df, :cp_index)
    cps = CPProfileResult[]
    for g in groups
        cands = collect(g.candidate_cp)
        perm  = sortperm(cands)
        cands = cands[perm]
        los   = collect(g.loss)[perm]
        orig  = g.original_cp[1]
        win   = maximum(cands) - orig
        push!(cps, CPProfileResult(g.cp_index[1], orig, win, cands, los, minimum(los)))
    end
    return cps
end

# -------------------------------------------------------------
# Plotting (package style + per-panel scaling)
# -------------------------------------------------------------
function plot_profiles_scaled(profiles::Vector{ProfileResult}; title::String,
                              cp_profiles::Vector{CPProfileResult}=CPProfileResult[],
                              cp_threshold=nothing)
    n_params = length(profiles)
    n = n_params + length(cp_profiles)
    ncols = min(4, n)
    nrows = ceil(Int, n / ncols)
    fig = Plots.plot(layout=(nrows, ncols), size=(500 * ncols, 380 * nrows),
                     framestyle=:box, grid=true, gridalpha=0.3,
                     margin=6Plots.mm, topmargin=4Plots.mm)

    # Shared y-axis across all panels of the figure: the threshold is the same
    # for every parameter of a run, so all panels should show it at the same
    # level.  Walls shooting far above the threshold are clipped (standard for
    # PLE displays).
    threshold = profiles[1].threshold
    # shared y-axis spans the parameter panels only; CP panels have their
    # own zoom (their loss range is much wider)
    global_min = minimum(minimum(p.losses) for p in profiles)
    gap = abs(threshold - global_min)
    ymin = global_min - 0.05 * gap
    ymax = threshold + 0.5 * gap

    n_ident = 0
    n_cp_ident = 0
    for cp in cp_profiles
        perm_cp = sortperm(cp.candidate_cps)
        _, _, id = cp_ci_interpolated(Float64.(cp.candidate_cps[perm_cp]), cp.losses[perm_cp],
                                      Float64(cp_threshold === nothing ? profiles[1].threshold : cp_threshold))
        id && (n_cp_ident += 1)
    end
    for (i, p) in enumerate(profiles)
        p.identifiable && (n_ident += 1)
        perm = sortperm(p.values)
        vals = p.values[perm]
        los  = p.losses[perm]

        use_log = all(vals .> 0) && (maximum(vals) / minimum(vals)) > 10.0
        identifiable = p.identifiable && p.ci_lower < p.ci_upper

        # Per-panel x window
        if identifiable
            lo, hi = minmax(p.ci_lower, p.ci_upper)
            w = max(hi - lo, 1e-12)
            if use_log
                llo, lhi = log10(lo), log10(hi)
                lw = max(lhi - llo, 1e-12)
                xw_lo, xw_hi = 10^(llo - 0.75 * lw), 10^(lhi + 0.75 * lw)
            else
                xw_lo, xw_hi = lo - 0.75 * w, hi + 0.75 * w
            end
        else
            xw_lo, xw_hi = minimum(vals), maximum(vals)
        end

        # Smoothed display curve, drawn over the full x-window so the limbs
        # continue past the threshold up to the top edge of the panel.
        # Sparse identifiable profiles (<6 points) use the local quadratic
        # (Wilks) approximation through (ci_lower, threshold), (best, min
        # loss), (ci_upper, threshold): smooth and crossing the threshold
        # exactly at the gold band edges.  Dense profiles get PCHIP.
        # Outside the profiled range the end-segment slope is extrapolated;
        # tails are clipped at the panel's top edge.  Markers show only the
        # real profiled points.
        lo = identifiable ? min(p.ci_lower, p.ci_upper) : NaN
        hi = identifiable ? max(p.ci_lower, p.ci_upper) : NaN
        m_val = vals[argmin(los)]
        # quadratic display anchor needs three strictly distinct points
        quad_ok = identifiable && lo < m_val < hi && hi - lo > 1e-10
        local_interp = if length(vals) >= 6 || !identifiable
            xi, yi = collect(vals), collect(los)
            x -> pchip_eval(xi, yi, [x])[1]
        elseif quad_ok
            xi = [lo, m_val, hi]
            yi = [p.threshold, minimum(los), p.threshold]
            x -> quad_eval(xi, yi, [x])[1]
        else
            xi2, yi2 = collect(vals), collect(los)
            x -> pchip_eval(xi2, yi2, [x])[1]
        end
        sparse_quad = length(vals) < 6 && quad_ok
        slope_l = (los[2] - los[1]) / (vals[2] - vals[1])
        slope_r = (los[end] - los[end-1]) / (vals[end] - vals[end-1])
        display_f = if sparse_quad
            # quadratic panels: extend the parabola itself over the whole
            # window (a slope-based tail would kink at the junction)
            local_interp
        else
            function (x)
                if x < vals[1]
                    return los[1] + slope_l * (x - vals[1])
                elseif x > vals[end]
                    return los[end] + slope_r * (x - vals[end])
                else
                    return local_interp(x)
                end
            end
        end
        xs = collect(range(xw_lo, xw_hi, length=400))
        ys = display_f.(xs)
        # Clip tails that rise above the panel top: keep the curve up to the
        # first point crossing the top edge on each side, so limbs end at the
        # panel boundary without fake horizontal segments along the top.
        ycap = ymax + 1e-9
        first_ok_l = findfirst(y -> y <= ycap, ys)
        first_ok_l === nothing && (first_ok_l = 1)
        last_ok_r = findlast(y -> y <= ycap, ys)
        last_ok_r === nothing && (last_ok_r = length(ys))
        # include one extra point beyond the threshold-cap crossing on each
        # side (clamped to the top edge) so limbs reach the panel boundary
        il = max(first_ok_l - 1, 1)
        ir = min(last_ok_r + 1, length(ys))
        keep = il:ir
        xs_c = xs[keep]
        ys_c = clamp.(ys[keep], minimum(los), ycap)
        Plots.plot!(fig[i], xs_c, ys_c, color=:steelblue, lw=1.2, label="")
        Plots.scatter!(fig[i], vals, los, markersize=3, color=:steelblue, label="")

        Plots.hline!(fig[i], [p.threshold], color=:crimson, ls=:dash, lw=0.9, label="")
        # Mark the profile's own minimum (green dotted)
        min_val = vals[argmin(los)]
        Plots.vline!(fig[i], [min_val], color=:darkgreen, ls=:dot, lw=1.0, label="")
        if p.ci_lower < p.ci_upper
            lo, hi = minmax(p.ci_lower, p.ci_upper)
            Plots.vspan!(fig[i], [lo, hi], color=:gold, alpha=0.25, label="")
        end

        Plots.plot!(fig[i], xlims=(xw_lo, xw_hi))
        # sensible tick marks: three round values spanning the window, so
        # narrow-CI panels don't get crowded float labels
        if !use_log
            lo_r, hi_r = xw_lo, xw_hi
            raw = range(lo_r, hi_r, length=3)
            step_raw = (hi_r - lo_r) / 2
            digits_r = max(0, -floor(Int, log10(max(step_raw, 1e-300))) + 2)
            ticks = round.(raw, digits=digits_r)
            Plots.plot!(fig[i], xticks=(collect(ticks), [string(t) for t in ticks]))
        end

        # Shared y-axis
        Plots.plot!(fig[i], ylims=(ymin, ymax))

        ci_txt = (p.ci_lower < p.ci_upper) ?
            "  CI=[$(round(p.ci_lower, sigdigits=4)), $(round(p.ci_upper, sigdigits=4))]" :
            "  CI=open"
        id_txt = p.identifiable ? "" : "  (non-identifiable)"
        Plots.plot!(fig[i], xlabel="parameter value", ylabel="loss",
                    title="$(p.parameter)$(ci_txt)$(id_txt)", titlefontsize=10,
                    xscale=use_log ? :log10 : :identity,
                    tickfontsize=8, labelfontsize=8, legend=false)
    end

    for (k, cp) in enumerate(cp_profiles)
        draw_cp_panel!(fig, n_params + k, cp, cp_threshold === nothing ? threshold : cp_threshold)
    end

    for j in (n + 1):(nrows * ncols)
        Plots.plot!(fig[j], axis=([], false), grid=false)
    end

    ref_loss = profiles[1].best_loss
    id_txt = "identifiable=$n_ident/$n_params params"
    if !isempty(cp_profiles)
        id_txt *= " + $n_cp_ident/$(length(cp_profiles)) CPs"
    end
    Plots.plot!(fig, plot_title="$title | ref loss=$(round(ref_loss, digits=2)) | " *
                                "threshold=$(round(threshold, digits=2)) | " *
                                id_txt,
                plot_titlefontsize=11)
    return fig
end


# Custom changepoint-profile figure, styled like the parameter panels:
# zoomed y-axis around the threshold region, red dashed threshold, gold CI
# band, green dotted line at the reference CP, one panel per CP.
# Interpolated CP confidence interval (same convention as ple_ci_interpolated
# for parameters): linear interpolation of the threshold crossings on the
# profiled day-grid; one-sided cases (no crossing within the window on a side)
# are reported as non-identifiable with the bound at the window edge.
function cp_ci_interpolated(cands, los, threshold)
    n = length(cands)
    imin = argmin(los)
    win_lo, win_hi = cands[1], cands[end]
    ci_lo, ci_hi = win_lo, win_hi
    crossed_l = false
    for j in imin:-1:2
        if los[j] <= threshold && los[j-1] > threshold
            frac = (threshold - los[j]) / (los[j-1] - los[j])
            ci_lo = cands[j] - frac * (cands[j] - cands[j-1])
            crossed_l = true
            break
        end
    end
    crossed_r = false
    for j in imin:(n-1)
        if los[j] <= threshold && los[j+1] > threshold
            frac = (threshold - los[j]) / (los[j+1] - los[j])
            ci_hi = cands[j] + frac * (cands[j+1] - cands[j])
            crossed_r = true
            break
        end
    end
    identifiable = crossed_l && crossed_r && ci_lo > win_lo && ci_hi < win_hi
    return ci_lo, ci_hi, identifiable
end

# Draw one changepoint-profile panel into fig[i] (used standalone and merged
# into joint parameter figures).  PCHIP-smoothed curve through the discrete
# day candidates, padded x-range, explicit CI edge lines.
function draw_cp_panel!(fig, i, cp, threshold)
    perm = sortperm(cp.candidate_cps)
    cands = Float64.(cp.candidate_cps[perm])
    los = cp.losses[perm]
    lo, hi, ident = cp_ci_interpolated(cands, los, Float64(threshold))
    # smoothed curve through the discrete day points
    if length(cands) >= 3
        xs = collect(range(cands[1], cands[end], length=200))
        ys = pchip_eval(cands, los, xs)
    else
        xs, ys = cands, los
    end
    Plots.plot!(fig[i], xs, ys, color=:steelblue, lw=1.2, label="")
    Plots.scatter!(fig[i], cands, los, markersize=3, color=:steelblue, label="")
    Plots.hline!(fig[i], [threshold], color=:crimson, ls=:dash, lw=0.9, label="")
    Plots.vline!(fig[i], [cp.original_cp], color=:darkgreen, ls=:dot, lw=1.0, label="")
    if !ismissing(lo)
        Plots.vspan!(fig[i], [lo, hi], color=:gold, alpha=0.25, label="")
        # explicit CI edge lines so bounds at the window edge stay visible
        Plots.vline!(fig[i], [lo], color=:darkorange, ls=:dashdot, lw=1.0, label="")
        Plots.vline!(fig[i], [hi], color=:darkorange, ls=:dashdot, lw=1.0, label="")
    end
    min_loss = minimum(los)
    gap = abs(threshold - min_loss)
    ymax = threshold + 1.2 * gap
    ymin = min_loss - 0.05 * gap
    Plots.plot!(fig[i], ylims=(ymin, ymax))
    Plots.plot!(fig[i], xlims=(cands[1] - 0.5, cands[end] + 0.5))
    ci_txt = ismissing(lo) ? "CI = [-, -]" : "CI = [$(round(lo, digits=1)), $(round(hi, digits=1))]"
    Plots.plot!(fig[i], xlabel="candidate changepoint", ylabel="loss",
                title="CP $(cp.cp_index) (original=$(cp.original_cp))  $ci_txt, identifiable=$ident",
                titlefontsize=9, tickfontsize=8, labelfontsize=8, legend=false)
end

function plot_cp_profiles_scaled(cp_profiles::Vector{CPProfileResult}, threshold; title::String="")
    n = length(cp_profiles)
    fig = Plots.plot(layout=(1, n), size=(460 * n, 380),
                     framestyle=:box, grid=true, gridalpha=0.3,
                     margin=6Plots.mm, topmargin=4Plots.mm)
    for (i, cp) in enumerate(cp_profiles)
        draw_cp_panel!(fig, i, cp, threshold)
    end
    Plots.plot!(fig, plot_title=title, plot_titlefontsize=11)
    return fig
end

function make_figures(config::String, mode::String, cpset::String)
    outdir = joinpath(PLE_BASE, cpset == "true" ? "ple" : "ple_detected", "$(config)_$(mode)")
    isdir(outdir) || return
    figdir = joinpath(outdir, "figures")
    mkpath(figdir)

    if mode == "conditional"
        profs = load_profiles(
            joinpath(outdir, "ple_results.csv"),
            joinpath(outdir, "ple_results_curves.csv")
        )
        # conditional mode fixes the change points: no CP panels
        cp_profs = CPProfileResult[]
        threshold = profs[1].threshold
        cp_threshold = threshold
    else
        profs = load_profiles(
            joinpath(outdir, "joint_ple_results.csv"),
            joinpath(outdir, "joint_ple_results_curves.csv")
        )
        # CP profiles live in the conditional run of the same anchor set;
        # they are merged into the joint figure as trailing subplots.
        cond_dir = joinpath(outdir, "..", "$(config)_conditional")
        cp_file = joinpath(cond_dir, "cp_profile_loss.csv")
        cp_profs = isfile(cp_file) ? load_cp_profiles(cp_file) : CPProfileResult[]
        threshold = profs[1].threshold
        # CP panels keep the conditional run's own threshold (they were
        # computed against the conditional reference)
        cond_summary = joinpath(cond_dir, "ple_summary.csv")
        cp_threshold = isfile(cond_summary) ? CSV.read(cond_summary, DataFrame).threshold[1] : threshold
    end

    cps = get_cps(joinpath(outdir, "ple_report.md"))
    title = "$(config) $(mode) PLE | toy-SIRD | CPs=$(cps)"
    fig_p = plot_profiles_scaled(profs; title=title, cp_profiles=cp_profs, cp_threshold=cp_threshold)
    savefig(fig_p, joinpath(figdir, "parameter_profile_curves.pdf"))
    savefig(fig_p, joinpath(figdir, "parameter_profile_curves.png"))
    println("  saved: parameter_profile_curves.png")

    # CP profiles are shown only as trailing subplots of the joint figures;
    # conditional mode fixes the change points, so no CP panels there.
    
end

function get_cps(report_path::String)
    isfile(report_path) || return "[]"
    for line in eachline(report_path)
        m = match(r"Fixed CPs:\*\*\s*(\[[^\]]*\])", line)
        m !== nothing && return m.captures[1]
    end
    return "[]"
end

function main()
    for cpset in ["true", "detected"]
        for config in ["M1", "M2", "M3", "M4"]
            for mode in ["conditional", "joint"]
                d = joinpath(PLE_BASE, cpset == "true" ? "ple" : "ple_detected", "$(config)_$(mode)")
                isdir(d) || continue
                # skip unfinished jobs (no results CSV yet)
                summary_csv = joinpath(d, mode == "conditional" ? "ple_summary.csv" : "joint_ple_summary.csv")
                isfile(summary_csv) || continue
                println("[$cpset $config $mode]")
                try
                    make_figures(config, mode, cpset)
                catch e
                    println("  ERROR: $e")
                    rethrow(e)
                end
            end
        end
    end
    println("\nAll dual-observable PLE figures written to $PLE_BASE")
end

main()
