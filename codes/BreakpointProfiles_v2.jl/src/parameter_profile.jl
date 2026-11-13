const CHISQ_95 = 3.8414588206941285

"""
    profile_parameter(prob::ODEChangepointPLEProblem, idx::Int;
                      options=ProfileOptions(),
                      optimizer=default_multistart_bobyqa_config(),
                      threshold=nothing)

Profile parameter `idx` by fixing it to a sequence of values and re-optimising all
other parameters (conditional PLE: changepoints stay fixed).

The profile is constructed with D2D-style adaptive outward stepping:
* start at the MLE `θ*_i`;
* step left and right, choosing each step so that the *initial* Δloss at the trial
  point is a small fraction of the threshold margin;
* re-optimise the free parameters at each accepted step;
* stop when the optimised Δloss exceeds the threshold margin or a parameter bound
  is hit.

`options` is a `ProfileOptions` object controlling the D2D stepping, smoothing,
and better-optimum handling.  `threshold` is the **absolute** loss level that
defines the confidence interval boundary.  For Gaussian (L2) likelihoods use
`best_loss + chi2_threshold(1, 0.95)`; for other losses the user must supply
an appropriate threshold.
"""
function profile_parameter(
    prob::ODEChangepointPLEProblem, idx::Int;
    options::ProfileOptions=ProfileOptions(),
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    threshold::Union{Real,Nothing}=nothing,
    # Backward-compatible keywords (pre-ProfileOptions API)
    method::Union{Symbol,Nothing}=nothing,
    n_points::Union{Int,Nothing}=nothing
)
    best_params = prob.best_params
    isempty(best_params) && error("ODEChangepointPLEProblem.best_params must be set")

    # Step 0: Verify the reference loss.
    verified_loss = evaluate_loss(prob, best_params)
    best_loss = if isfinite(prob.best_loss)
        if abs(verified_loss - prob.best_loss) > 0.01 * max(1.0, abs(prob.best_loss))
            @warn "Reference loss mismatch: prob.best_loss=$(prob.best_loss), evaluated=$(verified_loss). Using evaluated value."
        end
        min(prob.best_loss, verified_loss)
    else
        verified_loss
    end

    best_val = best_params[idx]
    user_threshold = threshold !== nothing ? Float64(threshold) : best_loss + CHISQ_95

    n = length(best_params)
    free = setdiff(1:n, idx)
    lb = prob.lb[free]
    ub = prob.ub[free]

    # Optional polishing: wrap non-hybrid optimizers in a two-stage hybrid.
    opt = options.polish && !(optimizer isa HybridOptimizerConfig) ?
          HybridOptimizerConfig(global_optimizer=optimizer) : optimizer

    if method === :fixed
        np = n_points !== nothing ? n_points : 20
        result = _profile_fixed(prob, idx, best_params, best_loss, user_threshold,
                                free, lb, ub, np, opt)
    else
        if method !== nothing && method !== :adaptive
            @warn "Unknown profile method '$method'; using adaptive D2D stepping."
        end
        if n_points !== nothing
            @warn "n_points is ignored for adaptive profiling; use ProfileOptions(samplesize=...) instead."
        end
        result = _profile_adaptive_d2d(prob, idx, best_params, best_loss,
                                       user_threshold, free, lb, ub, options, opt)
    end

    if options.allow_better_optimum
        _profile_update_reference!(result, prob)
    end

    return result
end

"""
    _profile_adaptive_d2d(prob, idx, best_params, best_loss, threshold, free, lb, ub, options, optimizer)

D2D-style adaptive profiling (Raue et al. 2009, `PLE/ple.m`):
1. Start at `θ*_i`.
2. For each direction (lower/upper) choose an adaptive step so that the trial loss
   increase is about `options.rel_step_increase * threshold_margin`.
3. Re-optimise all free parameters at the accepted trial point.
4. Record the optimised loss and update the warm-start for the next step.
5. Stop when `loss - best_loss > options.stop_margin_factor * threshold_margin` or
   the profile parameter hits a bound.
6. Sort, smooth local jumps, and extract CIs by linear interpolation.
"""
function _profile_adaptive_d2d(prob, idx, best_params, best_loss, threshold,
                               free, lb, ub, options, optimizer)
    best_val = best_params[idx]
    lower_bound = prob.lb[idx]
    upper_bound = prob.ub[idx]
    range_width = upper_bound - lower_bound
    threshold_margin = threshold - best_loss
    if threshold_margin <= 0
        threshold_margin = max(1e-6 * max(1.0, abs(best_loss)), 1e-12)
        @warn "Non-positive threshold margin for parameter $idx; using floor $threshold_margin. The profile will likely be very narrow."
    end

    min_step = max(range_width * options.min_step_factor, 1e-12)
    max_step = range_width * options.max_step_factor

    # Initial step: ~2% of parameter range or ~5% of parameter value, like D2D.
    step0 = if best_val > 0
        min(range_width / 50.0, best_val * 0.05)
    else
        range_width / 50.0
    end
    step0 = clamp(step0, min_step, max_step)

    values = Float64[best_val]
    losses = Float64[best_loss]
    params_trace = Vector{Float64}[copy(best_params)]  # warm-start trajectory for smoothing
    n_failed = 0
    best_found = best_loss

    # Lower direction
    l_failed, l_best = _ple_profile_direction!(
        prob, idx, best_params, best_loss, threshold, threshold_margin,
        lower_bound, upper_bound, free, lb, ub, -1.0, step0, min_step, max_step,
        options, optimizer, values, losses, params_trace
    )
    n_failed += l_failed
    best_found = min(best_found, l_best)

    # Upper direction
    u_failed, u_best = _ple_profile_direction!(
        prob, idx, best_params, best_loss, threshold, threshold_margin,
        lower_bound, upper_bound, free, lb, ub, +1.0, step0, min_step, max_step,
        options, optimizer, values, losses, params_trace
    )
    n_failed += u_failed
    best_found = min(best_found, u_best)

    # Sort by parameter value
    perm = sortperm(values)
    values = values[perm]
    losses = losses[perm]
    params_trace = params_trace[perm]

    # D2D pleSmooth: remove local jumps by re-optimising from neighbours.
    if options.smooth_jumps && length(values) >= 3
        _profile_smooth_jumps!(
            prob, idx, best_params, best_loss, threshold_margin, free, lb, ub,
            options, optimizer, values, losses, params_trace
        )
        perm = sortperm(values)
        values = values[perm]
        losses = losses[perm]
        params_trace = params_trace[perm]
    end

    if best_found < best_loss - 0.1
        @warn "Parameter $(prob.param_names[idx]): profiling found loss $best_found < reference $best_loss. The reference optimum may not be the true MLE."
    end

    ci_lower, ci_upper, identifiable = ple_ci_interpolated(
        values, losses, best_val, best_loss,
        lower_bound, upper_bound, threshold
    )

    # Full parameter vector at the minimum loss encountered during profiling.
    _, imin = findmin(losses)
    best_found_params = params_trace[imin]

    label = prob.param_names[idx]
    return ProfileResult(
        label, idx, best_val, best_loss, ci_lower, ci_upper, identifiable, threshold,
        values, losses, n_failed, best_found, best_found_params
    )
end

"""
    _ple_profile_direction!(prob, idx, best_params, best_loss, threshold, threshold_margin,
                            lower_bound, upper_bound, free, lb, ub, direction, step0,
                            min_step, max_step, options, optimizer, values, losses, params_trace)

Profile in one direction (`direction = -1` lower, `+1` upper).  Appends accepted
points to `values`, `losses`, and `params_trace`.  Returns `(n_failed, best_found)`.
"""
function _ple_profile_direction!(
    prob, idx, best_params, best_loss, threshold, threshold_margin,
    lower_bound, upper_bound, free, lb, ub, direction::Float64, step0::Float64,
    min_step::Float64, max_step::Float64, options::ProfileOptions,
    optimizer::AbstractPLEOptimizer,
    values::Vector{Float64}, losses::Vector{Float64},
    params_trace::Vector{Vector{Float64}}
)
    step = step0
    current_val = best_params[idx]
    x_warm = copy(best_params)
    n_failed = 0
    best_found = best_loss
    points_this_dir = 0

    while points_this_dir < options.samplesize
        step = clamp(step, min_step, max_step)
        candidate = current_val + direction * step

        # Clamp to bounds and make sure we make progress.
        if direction < 0
            candidate = max(candidate, lower_bound)
        else
            candidate = min(candidate, upper_bound)
        end

        if (direction < 0 && candidate >= current_val) || (direction > 0 && candidate <= current_val)
            break
        end

        # D2D-style trial step: fix idx, evaluate loss at warm-start without
        # re-optimising.  Reduce step until initial Δloss is acceptable.
        trial_step_ok, used_step = _ple_trial_step(
            prob, idx, candidate, x_warm, free, lb, ub, best_loss, threshold_margin,
            step, direction, min_step, max_step, options
        )

        if !trial_step_ok
            # Cannot find an acceptable step above min_step.
            break
        end
        candidate = current_val + direction * used_step
        if direction < 0
            candidate = max(candidate, lower_bound)
        else
            candidate = min(candidate, upper_bound)
        end

        # Now re-optimise all free parameters with idx fixed to candidate.
        loss_free = _build_loss_free(prob, idx, candidate, free)
        x_init = x_warm[free]
        loss, x_opt = try
            optimize_ple(loss_free, x_init, lb, ub, optimizer)
        catch e
            @warn "Optimization failed for parameter $idx at value $candidate: $e"
            Inf, x_init
        end

        push!(values, candidate)
        push!(losses, loss)

        if !isfinite(loss)
            n_failed += 1
            break
        end

        # Update warm-start for the next step.
        x_warm[free] .= x_opt
        x_warm[idx] = candidate
        push!(params_trace, copy(x_warm))

        if loss < best_found
            best_found = loss
        end

        Δ = loss - best_loss
        if Δ > options.stop_margin_factor * threshold_margin
            break
        end
        if (direction < 0 && candidate <= lower_bound) || (direction > 0 && candidate >= upper_bound)
            break
        end

        # Adapt step size for next iteration based on *optimised* Δloss.
        if Δ < options.rel_step_increase * threshold_margin
            step = used_step * options.step_factor
        elseif Δ > 0.8 * threshold_margin
            step = used_step / options.step_factor
        else
            step = used_step
        end

        current_val = candidate
        points_this_dir += 1
    end

    return n_failed, best_found
end

"""
    _ple_trial_step(prob, idx, candidate, x_warm, free, lb, ub, best_loss,
                    threshold_margin, step, direction, min_step, max_step, options)

D2D-style trial-step selection.  Fix `idx` to `candidate`, evaluate the loss at the
warm-start `x_warm[free]` *without* re-optimising, and reduce `step` until the initial
Δloss is below `rel_step_increase * threshold_margin`.  Returns `(ok, used_step)`.
"""
function _ple_trial_step(
    prob, idx, candidate, x_warm, free, lb, ub, best_loss, threshold_margin,
    step::Float64, direction::Float64, min_step::Float64, max_step::Float64,
    options::ProfileOptions
)
    loss_free = _build_loss_free(prob, idx, candidate, free)
    current_loss = loss_free(x_warm[free])
    target = options.rel_step_increase * max(threshold_margin, 1e-12)

    used_step = step
    for _ in 1:options.max_trial_redos
        # Recompute candidate from current_val = x_warm[idx]
        cand = x_warm[idx] + direction * used_step
        if direction < 0
            cand = max(cand, prob.lb[idx])
        else
            cand = min(cand, prob.ub[idx])
        end

        # Build trial full vector
        θ_trial = copy(x_warm)
        θ_trial[idx] = cand
        lf = _build_loss_free(prob, idx, cand, free)
        trial_loss = lf(θ_trial[free])

        if isfinite(trial_loss) && (trial_loss - current_loss) <= target
            return true, used_step
        end

        used_step /= options.step_factor
        if used_step < min_step
            return false, used_step
        end
    end
    return false, used_step
end

"""
    _build_loss_free(prob, idx, fixed_val, free)

Build a loss function over the free parameters with parameter `idx` fixed to
`fixed_val`.  Returns `f(x_free) -> loss`.
"""
function _build_loss_free(prob, idx, fixed_val, free)
    n = length(prob.best_params)
    function loss_free(x_free)
        θ = Vector{Float64}(undef, n)
        θ[idx] = fixed_val
        k = 1
        for j in 1:n
            if j == idx
                continue
            end
            θ[j] = x_free[k]
            k += 1
        end
        return evaluate_loss(prob, θ)
    end
    return loss_free
end

"""
    _profile_smooth_jumps!(prob, idx, best_params, best_loss, threshold_margin,
                           free, lb, ub, options, optimizer, values, losses, params_trace)

D2D `pleSmooth` equivalent.  Detect local downward jumps in the profile and
re-optimise from the neighbouring point to see whether the jump is a real
feature or an optimization artifact.  Updates `values`, `losses`, and
`params_trace` in place.
"""
function _profile_smooth_jumps!(
    prob, idx, best_params, best_loss, threshold_margin, free, lb, ub,
    options::ProfileOptions, optimizer::AbstractPLEOptimizer,
    values::Vector{Float64}, losses::Vector{Float64}, params_trace::Vector{Vector{Float64}}
)
    n = length(values)
    n == 0 && return
    _, globminindex = findmin(losses)

    candidates = Int[]
    directions = Int[]

    # Lower side of the minimum
    if globminindex != n
        for j in globminindex:-1:2
            crumin = losses[j]
            minchi2, minindex = findmin(losses[1:j])
            if minindex == j - 1 && 1 <= minindex <= n && isfinite(minchi2) && (crumin - minchi2) > options.jump_tol
                push!(candidates, minindex)
                push!(directions, -1)
            end
        end
        # Biggest diff
        d = diff(losses[1:globminindex])
        if length(d) > 2
            dsort = sort(d, rev=true)
            _, inddmax = findmax(d)
            # Candidate is the left point of the largest diff (index in losses)
            cand = inddmax
            if dsort[1] > dsort[2] + 5 * (dsort[2] - dsort[3]) && 1 <= cand <= n
                push!(candidates, cand)
                push!(directions, -1)
            end
        end
    end

    # Upper side of the minimum
    if globminindex != 1
        for j in globminindex:(n - 1)
            crumin = losses[j]
            minchi2, minindex = findmin(losses[j:n])
            cand = minindex + j - 1
            if cand != j && 1 <= cand <= n && isfinite(minchi2) && (crumin - minchi2) > options.jump_tol
                push!(candidates, cand)
                push!(directions, +1)
            end
        end
        d = diff(losses[n:-1:globminindex])
        if length(d) > 2
            dsort = sort(d, rev=true)
            _, inddmax = findmax(d)
            cand = n - inddmax + 1
            if dsort[1] > dsort[2] + 5 * (dsort[2] - dsort[3]) && 1 <= cand <= n
                push!(candidates, cand)
                push!(directions, +1)
            end
        end
    end

    for (ci, dr) in zip(candidates, directions)
        (ci < 1 || ci > n) && continue
        p0 = copy(params_trace[ci])
        while true
            ni = ci + dr
            (ni < 1 || ni > n) && break
            target_val = values[ni]
            p0[idx] = target_val
            loss_free = _build_loss_free(prob, idx, target_val, free)
            try
                loss, x_opt = optimize_ple(loss_free, p0[free], lb, ub, optimizer)
                θ = copy(p0)
                θ[free] .= x_opt
                if loss < losses[ni]
                    losses[ni] = loss
                    params_trace[ni] = θ
                else
                    break
                end
                p0 = θ
                ci = ni
            catch e
                break
            end
        end
    end
end

"""
    _profile_update_reference!(result, prob)

If profiling found a better optimum than the reference, log a warning with the
parameter values at the improved minimum.  The problem struct is immutable, so
the caller must create a new `ODEChangepointPLEProblem` with the improved
reference if they wish to re-profile using it.
"""
function _profile_update_reference!(result::ProfileResult, prob::ODEChangepointPLEProblem)
    if result.best_found_loss < result.best_loss - 1e-3
        _, imin = findmin(result.losses)
        @info "Profiling found a better optimum for parameter $(result.parameter): $(result.best_loss) -> $(result.best_found_loss). Consider rebuilding the problem with the improved reference before profiling other parameters."
    end
end

# _profile_fixed and ple_ci_interpolated (kept largely unchanged)

"""
    _profile_fixed(prob, idx, best_params, best_loss, threshold, free, lb, ub, n_points, optimizer)

Fixed-grid profiling (fallback).  Used when `method=:fixed` is requested.
"""
function _profile_fixed(prob, idx, best_params, best_loss, threshold, free, lb, ub, n_points, optimizer)
    best_val = best_params[idx]
    grid = _fixed_grid(prob.lb[idx], prob.ub[idx], best_val, n_points)

    x0 = copy(best_params)
    values = Float64[]
    losses = Float64[]
    params_trace = Vector{Float64}[]
    n_failed = 0
    best_found = best_loss

    for v in grid
        loss_free = _build_loss_free(prob, idx, v, free)
        x_init = x0[free]
        loss, x_opt = try
            optimize_ple(loss_free, x_init, lb, ub, optimizer)
        catch e
            @warn "Optimization failed for parameter $idx at value $v: $e"
            Inf, x_init
        end

        if isfinite(loss)
            x0[free] .= x_opt
            if loss < best_found
                best_found = loss
            end
        else
            n_failed += 1
        end

        push!(values, v)
        push!(losses, loss)
        push!(params_trace, copy(x0))
    end

    ci_lower, ci_upper, identifiable = ple_ci_interpolated(
        values, losses, best_val, best_loss,
        prob.lb[idx], prob.ub[idx], threshold
    )

    _, imin = findmin(losses)
    best_found_params = params_trace[imin]

    label = prob.param_names[idx]
    return ProfileResult(
        label, idx, best_val, best_loss, ci_lower, ci_upper, identifiable, threshold,
        values, losses, n_failed, best_found, best_found_params
    )
end

function _fixed_grid(lower, upper, best_val, n_points)
    if best_val > 0 && upper > lower && (upper / max(lower, 1e-12) > 10.0)
        gmin = max(lower, best_val / 5.0)
        gmax = min(upper, best_val * 5.0)
        gmin = min(gmin, best_val)
        gmax = max(gmax, best_val)
        grid = 10.0 .^ range(log10(gmin), log10(gmax), length=n_points)
    else
        half = 0.5 * (upper - lower)
        gmin = max(lower, best_val - half)
        gmax = min(upper, best_val + half)
        gmin = min(gmin, best_val)
        gmax = max(gmax, best_val)
        grid = collect(range(gmin, gmax, length=n_points))
    end
    push!(grid, best_val)
    sort!(grid)
    unique!(grid)
    return grid
end

"""
    ple_ci_interpolated(values, losses, best_val, best_loss, lower_bound, upper_bound, threshold)

Return `(ci_lower, ci_upper, identifiable)` using linear interpolation to find where
the profile crosses `threshold`.  Identifiability is true only if the profile crosses
the threshold on both sides before hitting a parameter bound.
"""
function ple_ci_interpolated(
    values::Vector{Float64},
    losses::Vector{Float64},
    best_val::Float64,
    best_loss::Float64,
    lower_bound::Float64,
    upper_bound::Float64,
    threshold::Float64
)
    perm = sortperm(values)
    svals = values[perm]
    sloss = losses[perm]
    n = length(svals)

    if n < 2
        return (best_val, best_val, false)
    end

    best_pos = argmin(abs.(svals .- best_val))

    # Lower crossing
    ci_lower = lower_bound
    crossed_lower = false
    for i in best_pos:-1:2
        if !isfinite(sloss[i - 1]) || !isfinite(sloss[i])
            continue
        end
        if sloss[i] <= threshold && sloss[i - 1] > threshold
            frac = (threshold - sloss[i]) / (sloss[i - 1] - sloss[i])
            ci_lower = svals[i] - frac * (svals[i] - svals[i - 1])
            crossed_lower = true
            break
        end
    end
    if !crossed_lower
        for i in best_pos:-1:1
            if isfinite(sloss[i]) && sloss[i] <= threshold
                ci_lower = svals[i]
            else
                break
            end
        end
    end

    # Upper crossing
    ci_upper = upper_bound
    crossed_upper = false
    for i in best_pos:(n - 1)
        if !isfinite(sloss[i]) || !isfinite(sloss[i + 1])
            continue
        end
        if sloss[i] <= threshold && sloss[i + 1] > threshold
            frac = (threshold - sloss[i]) / (sloss[i + 1] - sloss[i])
            ci_upper = svals[i] + frac * (svals[i + 1] - svals[i])
            crossed_upper = true
            break
        end
    end
    if !crossed_upper
        for i in best_pos:n
            if isfinite(sloss[i]) && sloss[i] <= threshold
                ci_upper = svals[i]
            else
                break
            end
        end
    end

    identifiable = crossed_lower && crossed_upper &&
                   ci_lower > lower_bound && ci_upper < upper_bound

    return (ci_lower, ci_upper, identifiable)
end

# Old discrete CI kept for backward compatibility.
"""
    ple_ci(values, losses, best_val, best_loss, lower_bound, upper_bound, threshold)

Return `(ci_lower, ci_upper, identifiable)` using a discrete contiguous-interval
approach.  Prefer `ple_ci_interpolated` for new code.
"""
function ple_ci(
    values::Vector{Float64},
    losses::Vector{Float64},
    best_val::Float64,
    best_loss::Float64,
    lower_bound::Float64,
    upper_bound::Float64,
    threshold::Float64
)
    perm = sortperm(values)
    svals = values[perm]
    sloss = losses[perm]

    valid = isfinite.(sloss) .& (sloss .<= threshold)
    n = length(svals)

    best_pos = findfirst(==(best_val), svals)
    if best_pos === nothing
        best_pos = clamp(searchsortedfirst(svals, best_val), 1, n)
    end

    left = best_pos
    while left > 1 && valid[left - 1]
        left -= 1
    end

    right = best_pos
    while right < n && valid[right + 1]
        right += 1
    end

    ci_lower = svals[left]
    ci_upper = svals[right]
    identifiable = (ci_lower > lower_bound) && (ci_upper < upper_bound)
    return (ci_lower, ci_upper, identifiable)
end

"""
    profile_all_parameters(prob::ODEChangepointPLEProblem;
                           options=ProfileOptions(),
                           optimizer=default_multistart_bobyqa_config(),
                           indices=nothing)

Profile every parameter (or a subset specified by `indices`) in `prob.best_params`
in parallel using `Base.Threads.@threads`.  Returns a `Vector{ProfileResult}`
ordered by parameter index.
"""
function profile_all_parameters(
    prob::ODEChangepointPLEProblem;
    options::ProfileOptions=ProfileOptions(),
    optimizer::AbstractPLEOptimizer=default_multistart_bobyqa_config(),
    indices::Union{Nothing,Vector{Int}}=nothing,
    threshold::Union{Real,Vector{Float64},Nothing}=nothing,
    method::Union{Symbol,Nothing}=nothing,
    n_points::Union{Int,Nothing}=nothing
)
    idxs = indices === nothing ? collect(1:length(prob.best_params)) : indices
    profiles = Vector{ProfileResult}(undef, length(idxs))
    # If the inner optimizer already uses threads, run parameters serially so each
    # parameter profile can fully utilise the available cores without oversubscription.
    if uses_internal_threads(optimizer)
        for k in 1:length(idxs)
            i = idxs[k]
            thr = threshold isa Vector{Float64} ? threshold[k] : threshold
            profiles[k] = profile_parameter(prob, i;
                                          options=options,
                                          optimizer=optimizer,
                                          threshold=thr,
                                          method=method,
                                          n_points=n_points)
        end
    else
        Base.Threads.@threads for k in 1:length(idxs)
            i = idxs[k]
            thr = threshold isa Vector{Float64} ? threshold[k] : threshold
            profiles[k] = profile_parameter(prob, i;
                                          options=options,
                                          optimizer=optimizer,
                                          threshold=thr,
                                          method=method,
                                          n_points=n_points)
        end
    end
    sort!(profiles, by=p -> p.index)
    return profiles
end

"""
    write_profiles(path, profiles)

Write a summary of the profile results to a CSV file with columns:
`parameter, index, best_value, best_loss, ci_lower, ci_upper, identifiable, threshold, n_failed, best_found_loss`.

Also writes a companion file `<path>_curves.csv` containing the full profile curves with
columns `parameter, index, value, loss, delta_loss`.
"""
function write_profiles(path, profiles::Vector{ProfileResult})
    df = DataFrame(
        parameter = String[],
        index = Int[],
        best_value = Float64[],
        best_loss = Float64[],
        ci_lower = Float64[],
        ci_upper = Float64[],
        identifiable = Bool[],
        threshold = Float64[],
        n_failed = Int[],
        best_found_loss = Float64[]
    )
    for p in profiles
        push!(df, (
            p.parameter, p.index, p.best_value, p.best_loss,
            p.ci_lower, p.ci_upper, p.identifiable, p.threshold,
            p.n_failed, p.best_found_loss
        ))
    end
    CSV.write(path, df)

    curves = to_profile_dataframe(profiles)
    curves_path = _curves_path(path)
    CSV.write(curves_path, curves)

    return df
end

function _curves_path(path)
    dir = dirname(path)
    base, ext = splitext(basename(path))
    return joinpath(dir, base * "_curves" * ext)
end


# ============================================================================

