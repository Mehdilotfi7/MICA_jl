# =============================================================================
# ModelContinuity.jl
# =============================================================================
#
# Level-continuity enforcement for model-informed changepoint detection.
#
# Stateful models (ODE, Difference, AR, ARIMA) already propagate their final
# state as the initial condition of the next segment via
# `update_initial_condition`.  Stateless models (Regression, Count, Volatility,
# ETS) fit each segment independently and can therefore jump at segment
# boundaries.  This module provides a model-specific adjustment so that the
# first simulated point of segment i+1 equals the last simulated point of
# segment i, while leaving all other (shape/rate/curvature) parameters free to
# change.
#
# This is *level* continuity only: the value at the boundary matches.  Slopes,
# curvature, AR coefficients, volatility parameters, etc. remain segment-
# specific and are exactly what the changepoint detection is allowed to detect.
# =============================================================================

"""
    needs_level_continuity(manager::ModelManager)

Return `true` if the model manager represents a stateless model family that
benefits from explicit level-continuity enforcement.  Stateful models return
`false` because their initial conditions are already propagated by
`update_initial_condition`.
"""
needs_level_continuity(manager::ModelManager) = manager.continuity && needs_level_continuity(manager.base_model)
needs_level_continuity(::ODEModelSpec) = false
needs_level_continuity(::DifferenceModelSpec) = false
needs_level_continuity(::AutoRegressiveModelSpec) = false
needs_level_continuity(::ARIMAModelSpec) = false
needs_level_continuity(::RegressionModelSpec) = true
needs_level_continuity(::CountModelSpec) = true
needs_level_continuity(::VolatilityModelSpec) = true
needs_level_continuity(::ETSModelSpec) = true


"""
    get_last_level(sim_data)

Extract the scalar value to be propagated across a segment boundary.  For the
univariate series used in the TCPD benchmark this is the last entry of the
first (and only) row.
"""
get_last_level(sim_data::AbstractMatrix) = sim_data[1, end]
get_last_level(sim_data::AbstractVector) = sim_data[end]


"""
    continuity_target_level(manager::ModelManager, sim_data)

Return the target level for the next segment.  For multivariate models the row
that corresponds to the observed/compared series is used (row 1 for volatility
models; for ODEs the propagation is handled by `update_initial_condition` and
this function is not called).
"""
continuity_target_level(manager::ModelManager{RegressionModelSpec}, sim_data) = get_last_level(sim_data)
continuity_target_level(manager::ModelManager{CountModelSpec}, sim_data) = get_last_level(sim_data)
continuity_target_level(manager::ModelManager{ETSModelSpec}, sim_data) = get_last_level(sim_data)
continuity_target_level(manager::ModelManager{VolatilityModelSpec}, sim_data) = sim_data[1, end]   # mean series


"""
    adjust_segment_params_for_continuity(manager::ModelManager, params, target_level)

Return a copy of `params` adjusted so that the model's first simulated value
equals `target_level`.  The adjustment is model-specific and preserves the
shape/curvature of the segment.  If no adjustment is implemented for the model,
`params` is returned unchanged.
"""
function adjust_segment_params_for_continuity(manager::ModelManager, params, target_level; t_start::Int=1)
    needs_level_continuity(manager) || return params
    return adjust_segment_params_for_continuity(manager.base_model, params, target_level; t_start=t_start)
end


# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------

function adjust_segment_params_for_continuity(spec::RegressionModelSpec, params, target_level; t_start::Int=1)
    return adjust_segment_params_for_continuity(spec.model_function, params, target_level; t_start=t_start)
end


# Mean model: y = μ
function adjust_segment_params_for_continuity(::typeof(mean_model), params, target_level; t_start::Int=1)
    p = copy(params)
    p[1] = target_level
    return p
end


# Linear model: y = a*t + b  => at t_start, y = a*t_start + b.  Keep a, set b = target - a*t_start.
function adjust_segment_params_for_continuity(::typeof(linear_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a = p[1]
    p[2] = target_level - a * t_start
    return p
end


# Slope-only linear model: y = y0 + slope * (t - t_start)  => at t_start, y = y0.
function adjust_segment_params_for_continuity(::typeof(linear_slope_only_model), params, target_level; t_start::Int=1)
    p = copy(params)
    p[2] = target_level
    return p
end


# Quadratic model: y = a*t^2 + b*t + c  => at t_start, y = a*t_start^2 + b*t_start + c.
function adjust_segment_params_for_continuity(::typeof(quadratic_model), params, target_level; t_start::Int=1)
    p = copy(params)
    p[3] = target_level - p[1] * t_start^2 - p[2] * t_start
    return p
end


# Cubic model: y = a*t^3 + b*t^2 + c*t + d  => at t_start, y = a*t_start^3 + b*t_start^2 + c*t_start + d.
function adjust_segment_params_for_continuity(::typeof(cubic_model), params, target_level; t_start::Int=1)
    p = copy(params)
    p[4] = target_level - p[1] * t_start^3 - p[2] * t_start^2 - p[3] * t_start
    return p
end


# Exponential model: y = c + a * exp(b*t)  => at t_start, y = c + a*exp(b*t_start).
# Keep a and b, solve for c.
function adjust_segment_params_for_continuity(::typeof(exponential_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b = p[1], p[2]
    p[3] = target_level - a * exp(b * t_start)
    return p
end


# Power model: y = c + a * t^b  => at t_start, y = c + a*t_start^b.  Keep a and b, solve for c.
function adjust_segment_params_for_continuity(::typeof(power_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b = p[1], p[2]
    p[3] = target_level - a * t_start^b
    return p
end


# Log-linear model: y = a*log(t) + b  => at t_start, y = a*log(t_start) + b.
function adjust_segment_params_for_continuity(::typeof(log_linear_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a = p[1]
    p[2] = target_level - a * log(t_start)
    return p
end


# Mean with linear drift: y = μ + a*t + b  => at t_start, y = μ + a*t_start + b.  Keep μ and a, set b.
function adjust_segment_params_for_continuity(::typeof(mean_drift_model), params, target_level; t_start::Int=1)
    p = copy(params)
    μ, a = p[1], p[2]
    p[3] = target_level - μ - a * t_start
    return p
end


# Hyperbolic model: y = a / t + b  => at t_start, y = a/t_start + b.  Keep a, set b.
# Note: b is the existing free level parameter for this model.
function adjust_segment_params_for_continuity(::typeof(hyperbolic_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a = p[1]
    p[2] = target_level - a / t_start
    return p
end


# Asymptotic regression: y = c + a - b*exp(-d*t)  => at t_start, y = c + a - b*exp(-d*t_start).
# Keep a, b and d, solve for c.
function adjust_segment_params_for_continuity(::typeof(asymptotic_regression_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b, d = p[1], p[2], p[3]
    p[4] = target_level - a + b * exp(-d * t_start)
    return p
end


# Michaelis-Menten: y = c + Vmax * t / (Km + t)  => at t_start, y = c + Vmax*t_start/(Km + t_start).
# Keep Vmax and Km, solve for c.
function adjust_segment_params_for_continuity(::typeof(michaelis_menten_model), params, target_level; t_start::Int=1)
    p = copy(params)
    Vmax, Km = p[1], p[2]
    p[3] = target_level - Vmax * t_start / (Km + t_start)
    return p
end


# Weibull growth: y = c + a - b*exp(-d*t^e)  => at t_start, y = c + a - b*exp(-d*t_start^e).
# Keep a, b, d, e, solve for c.
function adjust_segment_params_for_continuity(::typeof(weibull_growth_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b, d, e = p[1], p[2], p[3], p[4]
    p[5] = target_level - a + b * exp(-d * t_start^e)
    return p
end


# Hill function: y = c + a * t^n / (k^n + t^n)  => at t_start, y = c + a*t_start^n/(k^n + t_start^n).
# Keep a, k and n, solve for c.
function adjust_segment_params_for_continuity(::typeof(hill_function_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, k, n = p[1], max(p[2], 1e-6), p[3]
    p[4] = target_level - a * t_start^n / (k^n + t_start^n)
    return p
end


# Log-logistic: y = c + a / (1 + (t/b)^e)  => at t_start, y = c + a / (1 + (t_start/b)^e).
# Keep a, b and e, solve for c.
function adjust_segment_params_for_continuity(::typeof(log_logistic_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b, e = p[1], max(p[2], 1e-6), p[3]
    denom = 1.0 + (t_start / b)^e
    p[4] = target_level - a / denom
    return p
end


# Double exponential: y = c + a*exp(b*t) + d*exp(e*t)  => at t_start, y = c + a*exp(b*t_start) + d*exp(e*t_start).
# Keep a, b, d, e, solve for c.
function adjust_segment_params_for_continuity(::typeof(double_exponential_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b, d, e = p[1], p[2], p[3], p[4]
    p[5] = target_level - a * exp(b * t_start) - d * exp(e * t_start)
    return p
end


# Rational: y = c + (a*t + b) / (d*t + e)  => at t_start, y = c + (a*t_start + b) / (d*t_start + e).
# Keep a, b, d, e, solve for c.
function adjust_segment_params_for_continuity(::typeof(rational_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b, d, e = p[1], p[2], p[3], p[4]
    p[5] = target_level - (a * t_start + b) / (d * t_start + e)
    return p
end


# ---------------------------------------------------------------------------
# Logistic / Gompertz (additional growth models that gain an additive level)
# ---------------------------------------------------------------------------

# Logistic model: y = c + K / (1 + exp(-r*(t - t0)))  => at t_start,
# y = c + K / (1 + exp(-r*(t_start - t0))).  Keep K, r, t0, solve for c.
function adjust_segment_params_for_continuity(::typeof(logistic_model), params, target_level; t_start::Int=1)
    p = copy(params)
    K, r, t0 = p[1], p[2], p[3]
    p[4] = target_level - K / (1.0 + exp(-r * (t_start - t0)))
    return p
end


# Gompertz model: y = c + a * exp(-b * exp(-d*t))  => at t_start,
# y = c + a * exp(-b * exp(-d*t_start)).  Keep a, b, d, solve for c.
function adjust_segment_params_for_continuity(::typeof(gompertz_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b, d = p[1], p[2], p[3]
    p[4] = target_level - a * exp(-b * exp(-d * t_start))
    return p
end


# Saturating exponential: y = c + K * (1 - exp(-r*t))  => at t_start, y = c + K*(1 - exp(-r*t_start)).
# Keep K and r, solve for c.
function adjust_segment_params_for_continuity(::typeof(saturating_exponential_model), params, target_level; t_start::Int=1)
    p = copy(params)
    K, r = p[1], p[2]
    p[3] = target_level - K * (1.0 - exp(-r * t_start))
    return p
end


# ---------------------------------------------------------------------------
# Count models
# ---------------------------------------------------------------------------

# Poisson / Negative binomial: y = c + exp(a + b*t)  => at t=t_start, y = c + exp(a + b*t_start).
# Keep a and b, solve for c.
function adjust_segment_params_for_continuity(::typeof(poisson_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b = p[1], p[2]
    p[3] = target_level - exp(a + b * t_start)
    return p
end

function adjust_segment_params_for_continuity(::typeof(negbin_model), params, target_level; t_start::Int=1)
    p = copy(params)
    a, b = p[1], p[2]
    p[3] = target_level - exp(a + b * t_start)
    return p
end


# INGARCH: y = c + λ_t, λ_1 = ω / (1 - α - β)  => at the segment start, y = c + ω/(1 - α - β).
# Keep α and β, solve for c.  The local λ[1] is the intensity at the global boundary time.
function adjust_segment_params_for_continuity(::typeof(ingarch_model), params, target_level; t_start::Int=1)
    p = copy(params)
    ω, α, β = p[1], p[2], p[3]
    denom = max(1.0 - α - β, 1e-12)
    p[4] = target_level - ω / denom
    return p
end

function adjust_segment_params_for_continuity(spec::CountModelSpec, params, target_level; t_start::Int=1)
    return adjust_segment_params_for_continuity(spec.model_function, params, target_level; t_start=t_start)
end


# ---------------------------------------------------------------------------
# ETS models
# ---------------------------------------------------------------------------

# ETS(A,A,A): y_t = l0 + b0*t + γ*t^2 + α*sin(2πt/12) + β*cos(2πt/12)
# At t=t_start: keep b0, α, β, γ, solve for l0.
function adjust_segment_params_for_continuity(::typeof(ets_aaa_model), params, target_level; t_start::Int=1)
    p = copy(params)
    b0, α, β, γ = p[2], p[3], p[4], p[5]
    seasonal = α * sin(2π * t_start / 12.0) + β * cos(2π * t_start / 12.0)
    p[1] = target_level - b0 * t_start - γ * t_start^2 - seasonal
    return p
end


# ETS(M,M,M): y_t = l0*exp(b0*t/100) + γ*t + α*sin(2πt/12) + β*cos(2πt/12)
# At t=t_start: keep b0, α, β, γ, solve for l0.
function adjust_segment_params_for_continuity(::typeof(ets_mmm_model), params, target_level; t_start::Int=1)
    p = copy(params)
    b0, α, β, γ = p[2], p[3], p[4], p[5]
    seasonal = α * sin(2π * t_start / 12.0) + β * cos(2π * t_start / 12.0)
    eb = exp(b0 * t_start / 100.0)
    if abs(eb) > 1e-12
        p[1] = (target_level - γ * t_start - seasonal) / eb
    else
        p[1] = target_level - γ * t_start - seasonal
    end
    return p
end

function adjust_segment_params_for_continuity(spec::ETSModelSpec, params, target_level; t_start::Int=1)
    return adjust_segment_params_for_continuity(spec.model_function, params, target_level; t_start=t_start)
end


# ---------------------------------------------------------------------------
# Volatility models (not used in the current TCPD benchmark, but included for
# completeness).  The propagation is applied to the *mean* series (row 1).
# ---------------------------------------------------------------------------

function adjust_segment_params_for_continuity(::typeof(garch_model), params, target_level; t_start::Int=1)
    p = copy(params)
    p[1] = target_level   # μ
    return p
end

function adjust_segment_params_for_continuity(::typeof(egarch_model), params, target_level; t_start::Int=1)
    p = copy(params)
    p[1] = target_level   # μ
    return p
end

function adjust_segment_params_for_continuity(::typeof(tgarch_model), params, target_level; t_start::Int=1)
    p = copy(params)
    p[1] = target_level   # μ
    return p
end

function adjust_segment_params_for_continuity(spec::VolatilityModelSpec, params, target_level; t_start::Int=1)
    return adjust_segment_params_for_continuity(spec.model_function, params, target_level; t_start=t_start)
end


# ---------------------------------------------------------------------------
# Fallback for unrecognised model functions
# ---------------------------------------------------------------------------

function adjust_segment_params_for_continuity(model_function::Function, params, target_level; t_start::Int=1)
    @warn "Level continuity not implemented for model function $model_function; leaving parameters unchanged."
    return params
end
