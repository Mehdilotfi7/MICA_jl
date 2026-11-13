"""    AbstractLikelihood

Abstract supertype for built-in likelihood / loss callables."""
abstract type AbstractLikelihood end

"""
    log_transform(data; threshold=1)

Return `log(max(data, threshold))`.  Useful for epidemiological data with zeros."""
function log_transform(data; threshold=1.0)
    return [v >= threshold ? log(v) : log(threshold) for v in data]
end

"""
    GaussianLogLikelihood

Negative Gaussian log-likelihood on log-transformed observations, scaled to the
`-2 log L` scale used by profile-likelihood methods (e.g. Data2Dynamics).

`sigma` is the standard deviation of the observation noise. Minimising this
objective is equivalent to maximising a Gaussian likelihood, and the standard
χ² threshold of 3.8415 is asymptotically valid for 95% profile-likelihood
confidence intervals (Wilks' theorem).
"""
struct GaussianLogLikelihood <: AbstractLikelihood
    sigma::Float64
end
GaussianLogLikelihood() = GaussianLogLikelihood(1.0)

function (ll::GaussianLogLikelihood)(sim, data)
    n = length(data)
    r = log_transform(sim) .- log_transform(data)
    return sum(x -> x^2 / ll.sigma^2, r) + 2 * n * log(ll.sigma * sqrt(2π))
end

"""
    LaplaceLogLikelihood

Negative Laplace log-likelihood on log-transformed observations.

This corresponds to an L1 loss on log-transformed data, similar to the MICA
default. Because the Laplace likelihood is not Gaussian, Wilks' theorem does
not apply and the standard χ² threshold of 3.84 is **not** rigorously valid
for 95% confidence intervals. Users of L1 losses must supply a
likelihood-appropriate threshold.
"""
struct LaplaceLogLikelihood <: AbstractLikelihood
    sigma::Float64
end
LaplaceLogLikelihood() = LaplaceLogLikelihood(1.0)

function (ll::LaplaceLogLikelihood)(sim, data)
    n = length(data)
    r = log_transform(sim) .- log_transform(data)
    return sum(x -> abs(x) / ll.sigma, r) + n * log(2 * ll.sigma)
end

"""
    CustomLoss{F}

Wrap an arbitrary user-provided loss function `f(sim, data)`."""
struct CustomLoss{F} <: AbstractLikelihood
    f::F
end

function (c::CustomLoss)(sim, data)
    return c.f(sim, data)
end

"""
    GaussianLogNLL

Gaussian negative log-likelihood on log-transformed multi-channel data, scaled
to the `-2 log L` scale used by profile-likelihood methods (e.g. Data2Dynamics).

Each observed channel `k` is assumed to follow:

    log(obs_k(t)) ~ N(log(sim_k(t)), σ_k²)

so the contribution is:

    Σ_k Σ_t ((log(sim_k(t)) - log(obs_k(t))) / σ_k)²

Constant terms `2n * log(σ_k √(2π))` are omitted because they do not depend on
the model parameters and thus do not affect the profile shape.

This formulation makes the χ² threshold of 3.8415 valid for approximate 95%
profile-likelihood confidence intervals (Wilks' theorem).
"""
struct GaussianLogNLL <: AbstractLikelihood
    sigma_per_channel::Vector{Float64}
end

"""
    GaussianLogNLL(σ::Real)

Convenience: single σ applied to all channels.
"""
GaussianLogNLL(σ::Real) = GaussianLogNLL(Float64[σ])

function (ll::GaussianLogNLL)(sim, data)
    nch = size(data, 1)
    total = 0.0
    for k in 1:nch
        σ = ll.sigma_per_channel[min(k, length(ll.sigma_per_channel))]
        inv_σ2 = 1.0 / (σ * σ)
        for t in 1:size(data, 2)
            s = max(sim[k, t], 1.0)
            d = max(data[k, t], 1.0)
            r = log(s) - log(d)
            total += r * r * inv_σ2
        end
    end
    return total
end

"""
    estimate_channel_sigma(sim, data; data_indices=nothing)

Estimate per-channel σ from the residuals between `sim` and `data` on log scale.

If `data_indices` is provided (e.g. `[5, 6, 7, 9, 11]` for the COVID model),
it maps data rows to simulation rows.  Otherwise a 1:1 mapping is assumed
(data and sim have the same number of rows).

Returns a `Vector{Float64}` of length `size(data, 1)`.
"""
function estimate_channel_sigma(sim, data; data_indices=nothing)
    nch = size(data, 1)
    sigmas = Float64[]
    for k in 1:nch
        r = if data_indices !== nothing
            log_transform(sim[data_indices[k], :]) .- log_transform(data[k, :])
        else
            log_transform(sim[k, :]) .- log_transform(data[k, :])
        end
        push!(sigmas, max(std(r), 1e-6))
    end
    return sigmas
end

"""
    chi2_threshold(df=1, alpha=0.95)

Return the χ² threshold for the given degrees of freedom and confidence level.

For `df=1` and `alpha=0.95` this is the usual 3.8414588 used for approximate
95% profile-likelihood confidence intervals."""
function chi2_threshold(df::Int=1, alpha::Float64=0.95)
    if df == 1 && alpha == 0.95
        return 3.8414588206941285
    end
    error("Non-default chi2_threshold requires Distributions.jl; df=$df, alpha=$alpha")
end

"""
    evaluate_loss(prob::ODEChangepointPLEProblem, params)

Evaluate the loss for `params` using the problem's objective, simulator, or
generic ODE solver, in that order of priority."""
function evaluate_loss(prob::ODEChangepointPLEProblem, params::Vector{Float64})
    if prob.objective !== nothing
        return prob.objective(params, prob.changepoints)
    elseif prob.simulator !== nothing
        sim = prob.simulator(params, prob.changepoints)
        return prob.loss_fn(sim, prob.data)
    elseif prob.ode_function !== nothing
        sim = solve_segments(prob, params)
        if any(isnan, sim)
            return Inf
        end
        return prob.loss_fn(sim, prob.data)
    else
        error("ODEChangepointPLEProblem has no objective, simulator, or ode_function")
    end
end

function solve_segments(prob, params)
    t0, tf = prob.tspan
    cps = filter(cp -> t0 < cp < tf, sort(Float64.(prob.changepoints)))

    u0 = copy(prob.u0)
    t_start = t0
    parts = Matrix{Float64}[]

    for cp in cps
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

function _solve_segment(prob, u0, tspan, params)
    ode_prob = ODEProblem(prob.ode_function, u0, tspan, params)
    sol = solve(
        ode_prob, Tsit5(),
        saveat = 1.0,
        abstol = 1e-6,
        reltol = 1e-6,
        isoutofdomain = (u, p, t) -> any(x -> x < 0, u)
    )
    mat = hcat(sol.u...)
    expected = Int(ceil(tspan[2] - tspan[1])) + 1
    if any(isnan, mat) || size(mat, 2) != expected
        return fill(NaN, length(u0), expected)
    end
    return mat
end
