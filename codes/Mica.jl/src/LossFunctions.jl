# =============================================================================
# LossFunctions.jl
# =============================================================================
#
# Built-in loss / objective functions for changepoint detection plus a safe
# wrapper that guarantees a finite scalar return value.
#
# Every loss function has signature:
#     loss(obs::AbstractArray{Float64}, sim::AbstractArray{Float64}) -> Float64
# where obs and sim are typically matrices of shape (1, T).
# =============================================================================

# -----------------------------------------------------------------------------
# Safe wrapper
# -----------------------------------------------------------------------------

"""
    safe_loss(loss_fn::Function, obs, sim; bad::Real=1e12, warn::Bool=false)

Call `loss_fn(obs, sim)` and guarantee a finite scalar result.

If the inner call throws, returns `Inf`/`NaN`, or produces a non-finite value,
the wrapper returns `bad` (a large finite number) instead. This is the standard
robustness trick used by nonlinear fitting packages to keep gradient-based
optimisers from crashing when a parameter combination hits an invalid region.

# Arguments
- `loss_fn`: user-provided loss function `(obs, sim) -> Real`.
- `obs`, `sim`: observed and simulated data (typically `Matrix{Float64}`).
- `bad`: finite penalty to return on failure. For likelihood losses use a large
  positive number; for least-squares losses the same value works because the
  optimizer only compares magnitudes.
- `warn`: if `true`, print a warning when the safe value is used.

# Examples
```julia
rss = safe_loss(rss_loss, obs, sim)
ll = safe_loss(poisson_nll, obs, sim; bad=1e8)
```
"""
function safe_loss(loss_fn::Function, obs, sim; bad::Real=1e12, warn::Bool=false)
    l = try
        loss_fn(obs, sim)
    catch e
        warn && @warn "safe_loss: inner loss threw an exception, returning bad=$bad" exception=e
        return convert(Float64, bad)
    end
    if !isfinite(l)
        warn && @warn "safe_loss: inner loss returned non-finite value $l, returning bad=$bad"
        return convert(Float64, bad)
    end
    # Clamp to avoid extremely large but finite values destabilising the optimizer.
    return clamp(convert(Float64, l), -bad, bad)
end

"""
    safe_loss_factory(loss_fn::Function; bad::Real=1e12, warn::Bool=false)

Return a closure `(obs, sim) -> safe_loss(loss_fn, obs, sim; bad=bad, warn=warn)`.
Useful for passing a safely-wrapped loss to `detect_changepoints` or
`optimize_with_changepoints`.
"""
function safe_loss_factory(loss_fn::Function; bad::Real=1e12, warn::Bool=false)
    return (obs, sim) -> safe_loss(loss_fn, obs, sim; bad=bad, warn=warn)
end

# -----------------------------------------------------------------------------
# Least-squares / Lp losses
# -----------------------------------------------------------------------------

"""
    rss_loss(obs, sim)

Residual sum of squares: `sum((obs - sim).^2)`.
"""
function rss_loss(obs, sim)
    r = obs .- sim
    return sum(r .* r)
end

"""
    mse_loss(obs, sim)

Mean squared error: `mean((obs - sim).^2)`.
"""
function mse_loss(obs, sim)
    return rss_loss(obs, sim) / length(obs)
end

"""
    rmse_loss(obs, sim)

Root mean squared error.
"""
function rmse_loss(obs, sim)
    return sqrt(mse_loss(obs, sim))
end

"""
    l1_loss(obs, sim)

L1 loss (mean absolute error): `sum(abs.(obs - sim))`.
"""
function l1_loss(obs, sim)
    return sum(abs.(obs .- sim))
end

"""
    mae_loss(obs, sim)

Mean absolute error: `mean(abs.(obs - sim))`.
"""
function mae_loss(obs, sim)
    return l1_loss(obs, sim) / length(obs)
end

"""
    huber_loss(obs, sim; delta::Real=1.345)

Huber pseudo-loss: quadratic near zero, linear in the tails. `delta` is the
threshold (default 1.345 gives ~95 % efficiency for Gaussian errors).
"""
function huber_loss(obs, sim; delta::Real=1.345)
    r = abs.(obs .- sim)
    delta2 = delta * delta
    loss = sum(ifelse.(r .<= delta,
                       0.5 .* r .* r,
                       delta .* r .- 0.5 .* delta2))
    return loss
end

# -----------------------------------------------------------------------------
# Likelihood-based losses (all returned as *negative* log-likelihoods to be
# minimised)
# -----------------------------------------------------------------------------

"""
    gaussian_nll(obs, sim)

Gaussian negative log-likelihood assuming unit variance. For use with models
where residuals are treated as iid Gaussian; the constant term is omitted
because it does not affect the optimum.
"""
function gaussian_nll(obs, sim)
    r = obs .- sim
    return 0.5 * sum(r .* r)
end

"""
    poisson_nll(obs, sim)

Poisson negative log-likelihood. `sim` is treated as the Poisson rate λ.
"""
function poisson_nll(obs, sim)
    λ = clamp.(sim, 1e-12, 1e12)
    y = clamp.(obs, 0.0, 1e12)
    return sum(λ .- y .* log.(λ) .+ lgamma.(y .+ 1.0))
end

"""
    negbin_nll(obs, sim; dispersion=1.0)

Negative-binomial negative log-likelihood. `sim` is the mean μ and
`dispersion` is the size/θ parameter (variance = μ + μ²/θ).  The
`dispersion` argument may be omitted if the last element of `sim` is the
overdispersion parameter; this function assumes a fixed scalar for simplicity.

For a version that estimates dispersion jointly, see the package documentation.
"""
function negbin_nll(obs, sim; dispersion::Real=1.0)
    μ = clamp.(sim, 1e-12, 1e12)
    y = clamp.(obs, 0.0, 1e12)
    θ = max(convert(Float64, dispersion), 1e-12)
    # log-likelihood up to constants: y*log(μ) - (y+θ)*log(μ+θ)
    ll = sum(y .* log.(μ) .- (y .+ θ) .* log.(μ .+ θ))
    return -ll
end

# -----------------------------------------------------------------------------
# Convenience aliases
# -----------------------------------------------------------------------------

const DEFAULT_LOSS = rss_loss
const L2_LOSS = rss_loss
