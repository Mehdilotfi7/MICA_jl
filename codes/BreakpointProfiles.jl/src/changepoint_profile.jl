"""
    profile_changepoint(prob::ODEChangepointPLEProblem, cp_idx::Int;
                        window=7, optimizer=default_evolutionary_config())

Profile the location of the `cp_idx`-th changepoint by moving it within
`original_cp ± window`, re-optimising all parameters for each candidate location,
and recording the loss.

Returns a `CPProfileResult`.  The candidate set skips locations that are too
close to the boundaries or that collide with other changepoints.
"""
function profile_changepoint(
    prob::ODEChangepointPLEProblem, cp_idx::Int;
    window::Int=7,
    optimizer::AbstractPLEOptimizer=default_evolutionary_config()
)
    cps = prob.changepoints
    1 <= cp_idx <= length(cps) || error("cp_idx $cp_idx out of range")
    original_cp = cps[cp_idx]

    min_seg = 10
    lower_t = max(min_seg, 1)
    upper_t = max(lower_t, prob.n_obs - min_seg)

    candidate_cps = Int[]
    losses = Float64[]
    best_loss = evaluate_loss(prob, prob.best_params)

    for delta in (-window):window
        cand = original_cp + delta
        if cand < lower_t || cand > upper_t
            continue
        end
        other_cps = filter(!=(original_cp), cps)
        if cand in other_cps
            continue
        end
        new_cps = sort([other_cps; cand])

        new_prob = ODEChangepointPLEProblem(
            objective = prob.objective,
            simulator = prob.simulator,
            ode_function = prob.ode_function,
            u0 = prob.u0,
            tspan = prob.tspan,
            data = prob.data,
            loss_fn = prob.loss_fn,
            changepoints = new_cps,
            best_params = prob.best_params,
            best_loss = prob.best_loss,
            lb = prob.lb,
            ub = prob.ub,
            param_names = prob.param_names,
            n_global = prob.n_global,
            n_segment_specific = prob.n_segment_specific,
            n_obs = prob.n_obs
        )

        loss, _ = try
            solve_problem(new_prob, optimizer)
        catch e
            @warn "CP profile failed for cp_idx=$cp_idx candidate=$cand: $e"
            Inf, prob.best_params
        end

        push!(candidate_cps, cand)
        push!(losses, loss)
    end

    return CPProfileResult(cp_idx, original_cp, window, candidate_cps, losses, best_loss)
end

"""
    profile_all_changepoints(prob::ODEChangepointPLEProblem;
                               window=7, optimizer=default_evolutionary_config())

Profile every changepoint in `prob.changepoints`.  Returns a
`Vector{CPProfileResult}`.
"""
function profile_all_changepoints(
    prob::ODEChangepointPLEProblem;
    window::Int=7,
    optimizer::AbstractPLEOptimizer=default_evolutionary_config()
)
    return [profile_changepoint(prob, j; window=window, optimizer=optimizer)
            for j in 1:length(prob.changepoints)]
end

"""
    cp_ci(cp_prof::CPProfileResult, threshold)

Return `(ci_lower, ci_upper, identifiable)` for a changepoint profile.

The interval is the largest contiguous set of candidate CPs containing the
original CP with `loss <= threshold`.  `identifiable` is true if the loss rises
above the threshold on both sides within the profiling window.
"""
function cp_ci(cp_prof::CPProfileResult, threshold::Float64)
    original = cp_prof.original_cp
    cands = cp_prof.candidate_cps
    losses = cp_prof.losses

    valid = isfinite.(losses) .& (losses .<= threshold)
    n = length(cands)
    pos = findfirst(==(original), cands)
    if pos === nothing
        return (missing, missing, false)
    end

    left = pos
    while left > 1 && valid[left - 1]
        left -= 1
    end

    right = pos
    while right < n && valid[right + 1]
        right += 1
    end

    ci_lower = cands[left]
    ci_upper = cands[right]
    identifiable = (left > 1 && valid[left] && !valid[left - 1]) || (left == 1 && valid[left])
    identifiable = identifiable && ((right < n && valid[right] && !valid[right + 1]) || (right == n && valid[right]))
    return (ci_lower, ci_upper, identifiable)
end

"""
    write_cp_profiles(path, cp_profiles)

Write a long-format CSV of changepoint profile results with columns:
`cp_index, original_cp, candidate_cp, loss, delta_loss`.
"""
function write_cp_profiles(path, cp_profiles::Vector{CPProfileResult})
    df = DataFrame(
        cp_index = Int[],
        original_cp = Int[],
        candidate_cp = Int[],
        loss = Float64[],
        delta_loss = Float64[]
    )
    for cp in cp_profiles
        for (cand, loss) in zip(cp.candidate_cps, cp.losses)
            push!(df, (cp.cp_index, cp.original_cp, cand, loss, loss - cp.best_loss))
        end
    end
    CSV.write(path, df)
    return df
end
