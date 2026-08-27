"""    evaluate_loss(prob::ODEChangepointPLEProblem, params, cps)

Evaluate the loss for `params` and a *custom* changepoint vector `cps`.  Uses the
same priority as the single-argument version: objective > simulator > generic ODE
solver."""
function evaluate_loss(prob::ODEChangepointPLEProblem, params::Vector{Float64}, cps::Vector{Int})
    if prob.objective !== nothing
        return prob.objective(params, cps)
    elseif prob.simulator !== nothing
        sim = prob.simulator(params, cps)
        return prob.loss_fn(sim, prob.data)
    elseif prob.ode_function !== nothing
        sim = solve_segments(prob, params, cps)
        if any(isnan, sim)
            return Inf
        end
        return prob.loss_fn(sim, prob.data)
    else
        error("ODEChangepointPLEProblem has no objective, simulator, or ode_function")
    end
end

function solve_segments(prob, params, cps::Vector{Int})
    t0, tf = prob.tspan
    cps_sorted = filter(cp -> t0 < cp < tf, sort(Float64.(cps)))

    u0 = copy(prob.u0)
    t_start = t0
    parts = Matrix{Float64}[]

    for cp in cps_sorted
        sol = _solve_segment(prob, u0, (t_start, cp), params)
        any(isnan, sol) && return fill(NaN, length(prob.u0), Int(ceil(tf - t0)) + 1)
        if size(sol, 2) > 1
            push!(parts, sol[:, 1:end-1])
        end
        u0 = sol[:, end]
        t_start = cp
    end

    sol = _solve_segment(prob, u0, (t_start, tf), params)
    any(isnan, sol) && return fill(NaN, length(prob.u0), Int(ceil(tf - t0)) + 1)
    push!(parts, sol)

    return reduce(hcat, parts)
end

"""    _build_loss_free_cp(prob, idx, fixed_val, cps, free)

Build a loss function that fixes parameter `idx` to `fixed_val` and uses the
changepoint vector `cps`.  The free parameters are the same order as `free`."""
function _build_loss_free_cp(prob, idx, fixed_val, cps, free)
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
        return evaluate_loss(prob, θ, cps)
    end
    return loss_free
end

"""    _joint_optimize_params(prob, fixed_idx, fixed_val, cps, x_init, optimizer)

Optimise all parameters except `fixed_idx` for a *fixed* changepoint vector `cps`.
Returns `(loss, params_full, success)`."""
function _joint_optimize_params(prob, fixed_idx, fixed_val, cps, x_init, optimizer)
    n = length(prob.best_params)
    free = setdiff(1:n, fixed_idx)
    lb = prob.lb[free]
    ub = prob.ub[free]
    loss_free = _build_loss_free_cp(prob, fixed_idx, fixed_val, cps, free)
    try
        loss, x_opt = optimize_ple(loss_free, x_init, lb, ub, optimizer)
        θ = Vector{Float64}(undef, n)
        θ[fixed_idx] = fixed_val
        k = 1
        for j in 1:n
            if j == fixed_idx
                continue
            end
            θ[j] = x_opt[k]
            k += 1
        end
        return loss, θ, isfinite(loss)
    catch e
        @warn "Joint param optimization failed for cps=$cps: $e"
        return Inf, copy(prob.best_params), false
    end
end

"""    _joint_cp_grid(cps0, window, n_obs)

Generate a discrete search grid around the initial changepoints `cps0`.  Each CP is
varied by `±window` days, subject to a minimum segment length of `min_seg` and the
data boundary `[min_seg, n_obs - min_seg]`.  Returns a vector of valid CP vectors.

For a small number of CPs (≤3) this exhaustive grid is feasible; for more CPs it
would explode, so the function raises an error when the grid would exceed
`max_combinations` (default 5000)."""
function _joint_cp_grid(cps0::Vector{Int}, window::Int, n_obs::Int; step::Int=1, min_seg::Int=10, max_combinations::Int=5000)
    isempty(cps0) && return [Int[]]
    step = max(1, step)
    lower_t = max(min_seg, 1)
    upper_t = max(lower_t, n_obs - min_seg)

    ranges = Vector{Int}[]
    for cp in cps0
        r = Int[]
        for δ in (-window):step:window
            cand = cp + δ
            if cand < lower_t || cand > upper_t
                continue
            end
            push!(r, cand)
        end
        push!(ranges, sort(unique(r)))
    end

    # Cartesian product of all CP ranges
    combs = Iterators.product(ranges...)
    grid = Vector{Int}[]
    for c in combs
        cps = sort(collect(Int, c))
        # Require minimum separation and no collisions
        if all(cps[i+1] - cps[i] >= min_seg for i in 1:(length(cps)-1))
            push!(grid, cps)
        end
    end

    length(grid) > max_combinations && error("CP grid too large ($(length(grid)) > $max_combinations); reduce window or number of CPs")
    return grid
end

"""    _joint_optimize_cp_grid(prob, fixed_idx, fixed_val, cps0, window, params0, optimizer)

Exhaustively search a discrete grid of CP locations around `cps0` (±`window`),
optimising the free parameters for each CP set, and return the best overall
`(loss, params_full, cps)`.  This is the inner joint-optimization step used when
profiling a parameter while treating changepoint locations as nuisance variables."""
function _joint_optimize_cp_grid(prob, fixed_idx, fixed_val, cps0, window, params0, optimizer; step::Int=1, parallelize_grid::Bool=true)
    grid = _joint_cp_grid(cps0, window, prob.n_obs; step=step)

    free = setdiff(1:length(params0), fixed_idx)
    x_init = params0[free]

    if parallelize_grid && !uses_internal_threads(optimizer) && length(grid) > 1
        n_grid = length(grid)
        losses = Vector{Float64}(undef, n_grid)
        params_list = Vector{Vector{Float64}}(undef, n_grid)
        cps_list = Vector{Vector{Int}}(undef, n_grid)
        oks = Vector{Bool}(undef, n_grid)

        Base.Threads.@threads for i in 1:n_grid
            cps = grid[i]
            loss, θ, ok = _joint_optimize_params(prob, fixed_idx, fixed_val, cps, x_init, optimizer)
            losses[i] = ok ? loss : Inf
            params_list[i] = θ
            cps_list[i] = cps
            oks[i] = ok
        end

        best_idx = argmin(losses)
        best_loss = losses[best_idx]
        best_params = params_list[best_idx]
        best_cps = cps_list[best_idx]
    else
        best_loss = Inf
        best_params = copy(params0)
        best_cps = copy(cps0)

        for cps in grid
            loss, θ, ok = _joint_optimize_params(prob, fixed_idx, fixed_val, cps, x_init, optimizer)
            if ok && loss < best_loss
                best_loss = loss
                best_params = θ
                best_cps = cps
                x_init = θ[free]  # warm-start next CP evaluation
            end
        end
    end

    return best_loss, best_params, best_cps
end

"""    profile_parameter_joint(prob::ODEChangepointPLEProblem, idx;
                            n_points=20, window=7, optimizer=default_de_config(),
                            threshold=nothing)

Adaptive profile-likelihood for parameter `idx` where changepoint locations are
*treated as nuisance variables* and jointly optimised at each profile step.

Algorithm (D2D-style adaptive stepping + joint CP search):
1. Start at the conditional best-fit `(best_params, changepoints)`.
2. Step the profiled parameter `idx` outward in both directions.
3. At each step, fix `idx` to the candidate value and jointly optimise:
   (a) all other parameters, and
   (b) all changepoint locations within `±window` of the current best CP set.
   The joint optimum is found by an exhaustive CP grid search (feasible for the
   BIC winner's 2 CPs) with a parameter re-optimisation for each CP set.
4. Adapt the step size based on the joint optimised Δloss relative to the
   threshold margin, as in `_profile_adaptive`.
5. Stop when Δloss exceeds `2 × threshold_margin` or the parameter bound is hit.

Returns a `ProfileResult` whose `best_loss`, `ci_lower`, `ci_upper` and profile
curve now reflect the uncertainty **after marginalising over changepoint
locations**.  This is the appropriate PLE when the detected CPs are not fixed
constants but also estimated quantities.
"""
function profile_parameter_joint(
    prob::ODEChangepointPLEProblem, idx::Int;
    n_points::Int=20,
    window::Int=7,
    step::Int=1,
    optimizer::AbstractPLEOptimizer=default_de_config(),
    threshold::Union{Real,Nothing}=nothing,
    parallelize_grid::Bool=true
)
    best_params = prob.best_params
    isempty(best_params) && error("ODEChangepointPLEProblem.best_params must be set")
    cps0 = copy(prob.changepoints)

    verified_loss = evaluate_loss(prob, best_params, cps0)
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

    lower_bound = prob.lb[idx]
    upper_bound = prob.ub[idx]
    range_width = upper_bound - lower_bound
    threshold_margin = user_threshold - best_loss
    if threshold_margin <= 0
        threshold_margin = max(1e-6 * max(1.0, abs(best_loss)), 1e-12)
        @warn "Non-positive threshold margin for parameter $idx; using floor $threshold_margin. The profile will likely be very narrow."
    end

    min_step = max(range_width / 1_000_000.0, 1e-12)
    max_step = range_width / 2.0
    step0 = if best_val > 0
        min(range_width / 50.0, best_val * 0.05)
    else
        range_width / 50.0
    end
    step0 = clamp(step0, min_step, max_step)

    values = Float64[best_val]
    losses = Float64[best_loss]
    params_trace = Vector{Float64}[copy(best_params)]
    n_failed = 0
    best_found = best_loss
    best_found_params = copy(best_params)

    # Warm-start state: current best (params, cps, loss)
    current_params = copy(best_params)
    current_cps = copy(cps0)

    function _profile_direction(step_sign)
        step_size = step0
        current_val = best_val
        x_warm = copy(best_params)
        cps_warm = copy(cps0)
        points_this_dir = 0

        while points_this_dir < n_points
            step_size = clamp(step_size, min_step, max_step)
            candidate = current_val + step_sign * step_size
            if step_sign < 0
                candidate = max(candidate, lower_bound)
            else
                candidate = min(candidate, upper_bound)
            end
            if (step_sign < 0 && candidate >= current_val) || (step_sign > 0 && candidate <= current_val)
                break
            end

            loss, θ, cps = _joint_optimize_cp_grid(prob, idx, candidate, cps_warm, window, x_warm, optimizer; step=step, parallelize_grid=parallelize_grid)

            push!(values, candidate)
            push!(losses, loss)
            push!(params_trace, copy(θ))
            points_this_dir += 1

            if !isfinite(loss)
                n_failed += 1
                break
            end

            if loss < best_found
                best_found = loss
                best_found_params = copy(θ)
            end

            # Update warm-starts for next step
            x_warm = θ
            cps_warm = cps
            current_params = θ
            current_cps = cps

            Δ = loss - best_loss
            if Δ > 2.0 * threshold_margin || (step_sign < 0 && candidate <= lower_bound) || (step_sign > 0 && candidate >= upper_bound)
                break
            end

            if Δ < 0.3 * threshold_margin
                step_size *= 1.5
            elseif Δ > 0.8 * threshold_margin
                step_size *= 0.5
            end

            current_val = candidate
        end
    end

    _profile_direction(-1.0)
    _profile_direction(1.0)

    perm = sortperm(values)
    values = values[perm]
    losses = losses[perm]
    params_trace = params_trace[perm]

    if best_found < best_loss - 0.1
        @warn "Parameter $(prob.param_names[idx]): joint profiling found loss $best_found < reference $best_loss. The reference optimum may not be the true joint MLE."
    end

    ci_lower, ci_upper, identifiable = ple_ci_interpolated(
        values, losses, best_val, best_loss,
        lower_bound, upper_bound, user_threshold
    )

    _, imin = findmin(losses)
    best_found_params_from_trace = params_trace[imin]

    label = prob.param_names[idx]
    return ProfileResult(
        label, idx, best_val, best_loss, ci_lower, ci_upper, identifiable, user_threshold,
        values, losses, n_failed, best_found, best_found_params_from_trace
    )
end

"""    profile_all_parameters_joint(prob::ODEChangepointPLEProblem;
                                   n_points=20, window=7, optimizer=default_de_config(),
                                   indices=nothing, threshold=nothing)

Run `profile_parameter_joint` for every parameter (or a subset specified by
`indices`) in parallel using `Base.Threads.@threads`."""
function profile_all_parameters_joint(
    prob::ODEChangepointPLEProblem;
    n_points::Int=20,
    window::Int=7,
    step::Int=1,
    optimizer::AbstractPLEOptimizer=default_de_config(),
    indices::Union{Nothing,Vector{Int}}=nothing,
    threshold::Union{Real,Vector{Float64},Nothing}=nothing,
    parallel_over_parameters::Bool=true,
    parallelize_grid::Bool=true
)
    idxs = indices === nothing ? collect(1:length(prob.best_params)) : indices
    profiles = Vector{ProfileResult}(undef, length(idxs))
    # Avoid nested threading: if we parallelise over parameters, the inner CP-grid
    # loop must be serial.  Conversely, when parameters are processed serially the
    # CP grid can be threaded.
    if parallel_over_parameters && !uses_internal_threads(optimizer)
        Base.Threads.@threads for k in 1:length(idxs)
            i = idxs[k]
            thr = threshold isa Vector{Float64} ? threshold[k] : threshold
            profiles[k] = profile_parameter_joint(prob, i;
                                                  n_points=n_points,
                                                  window=window,
                                                  step=step,
                                                  optimizer=optimizer,
                                                  threshold=thr,
                                                  parallelize_grid=false)
        end
    else
        for k in 1:length(idxs)
            i = idxs[k]
            thr = threshold isa Vector{Float64} ? threshold[k] : threshold
            profiles[k] = profile_parameter_joint(prob, i;
                                                  n_points=n_points,
                                                  window=window,
                                                  step=step,
                                                  optimizer=optimizer,
                                                  threshold=thr,
                                                  parallelize_grid=parallelize_grid)
        end
    end
    sort!(profiles, by=p -> p.index)
    return profiles
end

"""    JointPLEConfig

Configuration bundle for joint adaptive PLE.  Used to pass settings from the
driver to the joint profiling functions."""
struct JointPLEConfig
    n_points::Int
    window::Int
    step::Int
    optimizer::AbstractPLEOptimizer
    parallel_over_parameters::Bool
    parallelize_grid::Bool
end

function JointPLEConfig(;
    n_points::Int=20,
    window::Int=7,
    step::Int=1,
    optimizer::AbstractPLEOptimizer=default_de_config(),
    parallel_over_parameters::Bool=true,
    parallelize_grid::Bool=true
)
    return JointPLEConfig(n_points, window, step, optimizer, parallel_over_parameters, parallelize_grid)
end

