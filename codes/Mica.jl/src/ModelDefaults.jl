# =============================================================================
# ModelDefaults.jl
# =============================================================================
#
# Literature-based default bounds, initial guesses, optimizers and loss
# functions for the univariate segment models used in the TCPD benchmark.
#
# The defaults follow the recommendations collected in
# `MODEL_PARAMETER_BOUNDS_RESEARCH.md` (sources: Bates & Watts, 1988;
# Hyndman & Athanasopoulos, 2018; Cryer & Chan, 2008; Plotly enzyme-kinetics
# guide; OriginLab HillCOEFF; quickpsy; statsmodels ETS; fGarch; etc.).
# =============================================================================

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

"""
    _nudge_inward(x, lo, hi; rel=0.01)

Move `x` slightly inside the interval `[lo, hi]` so that optimisers that warn
on boundary-starting points (e.g. `Fminbox`) are happy. If the interval is
degenerate, return `lo`.
"""
function _nudge_inward(x, lo, hi; rel::Real=0.01)
    if !isfinite(lo) || !isfinite(hi) || hi <= lo
        return x
    end
    margin = rel * (hi - lo)
    return clamp(x, lo + margin, hi - margin)
end

"""
    _half_max_t(y, t)

Return the `t` value whose corresponding `y` is closest to `ymax/2`.
Useful for Michaelis-Menten, Hill and log-logistic initialisation.
"""
function _half_max_t(y, t)
    ymax = maximum(y)
    idx = argmin(abs.(y .- ymax / 2))
    return t[idx]
end

"""
    _ols_initial(y, n, model_name)

Return an OLS-based initial parameter vector for models that have an analytical
fit implemented in `ModelSimulation.jl`. Falls back to heuristic values when
OLS fails (e.g. too few observations).
"""
function _ols_initial(y, n, model_name)
    # Build a fake segment_data matrix and call the analytical fit.
    seg = reshape(y, 1, :)
    try
        if model_name == "mean"
            _, p = fit_segment_analytical(mean_model, seg)
        elseif model_name == "linear"
            _, p = fit_segment_analytical(linear_model, seg)
        elseif model_name == "quadratic"
            _, p = fit_segment_analytical(quadratic_model, seg)
        elseif model_name == "cubic"
            _, p = fit_segment_analytical(cubic_model, seg)
        elseif model_name == "exponential"
            _, p = fit_segment_analytical(exponential_model, seg)
        elseif model_name == "power"
            _, p = fit_segment_analytical(power_model, seg)
        elseif model_name == "log_linear"
            _, p = fit_segment_analytical(log_linear_model, seg)
        elseif model_name == "mean_drift"
            _, p = fit_segment_analytical(mean_drift_model, seg)
        elseif model_name == "ar1"
            _, p = fit_segment_analytical(ar1_model, seg)
        elseif model_name == "ar2"
            _, p = fit_segment_analytical(ar2_model, seg)
        elseif model_name == "ar3"
            _, p = fit_segment_analytical(ar3_model, seg)
        elseif model_name == "hyperbolic"
            _, p = fit_segment_analytical(hyperbolic_model, seg)
        elseif model_name == "asymptotic_regression"
            _, p = fit_segment_analytical(asymptotic_regression_model, seg)
        elseif model_name == "michaelis_menten"
            _, p = fit_segment_analytical(michaelis_menten_model, seg)
        else
            error("no analytical fit for $model_name")
        end
        return Float64.(p)
    catch
        return Float64[]
    end
end

"""
    _has_analytical_fit(model_name)

Return `true` if `fit_segment_analytical` is implemented for `model_name`.
"""
function _has_analytical_fit(model_name::String)
    return model_name in (
        "mean", "linear", "linear_slope_only", "quadratic", "cubic",
        "exponential", "power", "log_linear", "mean_drift",
        "ar1", "ar1_nodrift", "ar2", "ar3",
        "hyperbolic", "asymptotic_regression", "michaelis_menten"
    )
end

# -----------------------------------------------------------------------------
# Default optimizer selection
# -----------------------------------------------------------------------------

"""
    default_optimizer(model_name; use_analytical=true)

Return the recommended default optimizer for `model_name`.

- If `use_analytical` is true and the model has a closed-form least-squares fit,
  return `AnalyticalOptimizer()` (fastest and exact for RSS).
- Otherwise return `OptimOptimizer(NelderMead())`, which is derivative-free and
  avoids the NaN-gradient problems seen with `Fminbox(LBFGS())` on nonlinear
  models.
"""
function default_optimizer(model_name::String; use_analytical::Bool=true)
    if use_analytical && _has_analytical_fit(model_name)
        return AnalyticalOptimizer()
    end
    return OptimOptimizer(NelderMead(),
                          options=Optim.Options(show_trace=false,
                                                iterations=200,
                                                f_reltol=1e-8))
end

# -----------------------------------------------------------------------------
# Default loss selection
# -----------------------------------------------------------------------------

"""
    default_loss(model_name)

Return the recommended default loss function for `model_name`.

Currently all models default to `rss_loss`. Count models can optionally use
`poisson_nll` / `negbin_nll` if the user overrides.
"""
function default_loss(model_name::String)
    return rss_loss
end

# -----------------------------------------------------------------------------
# Per-model bounds and initial guesses
# -----------------------------------------------------------------------------

function _mean_defaults(y, n)
    mu = mean(y)
    sig = max(std(y), 1.0)
    lo = y_min = minimum(y)
    hi = y_max = maximum(y)
    chrom = [mu]
    lower = [y_min - 3sig]
    upper = [y_max + 3sig]
    parnames = (:μ,)
    return chrom, lower, upper, parnames, 0, 1
end

function _linear_defaults(y, n)
    t = Float64.(1:n)
    cov_ty = cov(t, y)
    var_t = var(t)
    b_init = var_t > 1e-10 ? cov_ty / var_t : 0.0
    a_init = mean(y) - b_init * mean(t)
    y_range = maximum(y) - minimum(y)
    slope_bound = max(1000.0, abs(y_range) * 20 / n)
    sig = max(std(y), 1.0)
    chrom = [b_init, a_init]
    lower = [-slope_bound, minimum(y) - 3sig]
    upper = [slope_bound, maximum(y) + 3sig]
    parnames = (:b, :a)
    return chrom, lower, upper, parnames, 0, 2
end

function _linear_slope_only_defaults(y, n)
    t = Float64.(1:n)
    cov_ty = cov(t, y)
    var_t = var(t)
    slope_init = var_t > 1e-10 ? cov_ty / var_t : 0.0
    y_range = maximum(y) - minimum(y)
    slope_bound = max(1000.0, abs(y_range) * 20 / n)
    sig = max(std(y), 1.0)
    chrom = [slope_init, y[1]]
    lower = [-slope_bound, minimum(y) - 3sig]
    upper = [slope_bound, maximum(y) + 3sig]
    parnames = (:slope, :y0)
    return chrom, lower, upper, parnames, 0, 2
end

function _polynomial_defaults(y, n, degree::Int)
    # degree=2 -> quadratic [a,b,c]; degree=3 -> cubic [a,b,c,d]
    t = Float64.(1:n)
    sig = max(std(y), 1.0)
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    big = max(10.0, abs(y_range)) * 10
    if degree == 2
        parnames = (:a, :b, :c)
        # OLS initial guess
        A = hcat(t.^2, t, ones(n))
        coeffs = A \ y
        chrom = Float64.(coeffs)
        lower = [-big, -big, y_min - 3sig]
        upper = [big, big, y_max + 3sig]
        n_seg = 3
    else
        parnames = (:a, :b, :c, :d)
        A = hcat(t.^3, t.^2, t, ones(n))
        coeffs = A \ y
        chrom = Float64.(coeffs)
        lower = [-big, -big, -big, y_min - 3sig]
        upper = [big, big, big, y_max + 3sig]
        n_seg = 4
    end
    return chrom, lower, upper, parnames, 0, n_seg
end

function _exponential_defaults(y, n)
    t = Float64.(1:n)
    y_min = minimum(y)
    off = y_min <= 0 ? abs(y_min) + 1.0 : 0.0
    A = hcat(t, ones(n))
    coeffs = A \ log.(y .+ off)
    b_init, log_a_init = coeffs
    a_init = exp(log_a_init)
    y_max = maximum(y)
    big = max(abs(y_min), abs(y_max)) * 10
    c_init = -off  # shift matching the analytical-fit offset
    sig = max(std(y), 1.0)
    chrom = [a_init, b_init, c_init]
    lower = [0.01, -5.0, y_min - 3sig]
    upper = [big, 5.0, y_max + 3sig]
    parnames = (:a, :b, :c)
    return chrom, lower, upper, parnames, 0, 3
end

function _power_defaults(y, n)
    t = Float64.(1:n)
    y_min = minimum(y)
    off = y_min <= 0 ? abs(y_min) + 1.0 : 0.0
    A = hcat(log.(t), ones(n))
    coeffs = A \ log.(y .+ off)
    b_init, log_a_init = coeffs
    a_init = exp(log_a_init)
    y_max = maximum(y)
    big = max(abs(y_min), abs(y_max)) * 10
    c_init = -off
    sig = max(std(y), 1.0)
    chrom = [a_init, b_init, c_init]
    lower = [0.01, -5.0, y_min - 3sig]
    upper = [big, 5.0, y_max + 3sig]
    parnames = (:a, :b, :c)
    return chrom, lower, upper, parnames, 0, 3
end

function _log_linear_defaults(y, n)
    t = Float64.(1:n)
    A = hcat(log.(t), ones(n))
    coeffs = A \ y
    a_init, b_init = coeffs
    y_min, y_max = minimum(y), maximum(y)
    sig = max(std(y), 1.0)
    big = max(abs(y_min), abs(y_max)) * 10
    chrom = [a_init, b_init]
    lower = [-big, y_min - 3sig]
    upper = [big, y_max + 3sig]
    parnames = (:a, :b)
    return chrom, lower, upper, parnames, 0, 2
end

function _mean_drift_defaults(y, n)
    t = Float64.(1:n)
    A = hcat(t, ones(n))
    ab = A \ y
    a_init, b_init = ab
    mu_init = mean(y .- (a_init .* t .+ b_init))
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    slope_bound = max(1000.0, abs(y_range) * 20 / n)
    chrom = [mu_init, a_init, b_init]
    lower = [-3sig, -slope_bound, y_min - 3sig]
    upper = [3sig, slope_bound, y_max + 3sig]
    parnames = (:μ, :a, :b)
    return chrom, lower, upper, parnames, 0, 3
end

function _ar_defaults(y, n, order::Int)
    c_init = mean(y)
    y_min, y_max = minimum(y), maximum(y)
    chrom = [zeros(order); c_init]
    lower = fill(-0.99, order + 1)
    upper = fill(0.99, order + 1)
    lower[end] = y_min - 5.0
    upper[end] = y_max + 5.0
    if order == 1
        parnames = (:phi1, :c)
    elseif order == 2
        parnames = (:phi1, :phi2, :c)
    else
        parnames = (:phi1, :phi2, :phi3, :c)
    end
    return chrom, lower, upper, parnames, 0, order + 1
end

function _ar_nodrift_defaults(y, n)
    chrom = [0.0]
    lower = [-0.99]
    upper = [0.99]
    parnames = (:phi1,)
    return chrom, lower, upper, parnames, 0, 1
end

function _hyperbolic_defaults(y, n)
    mu = mean(y)
    y_range = maximum(y) - minimum(y)
    R = max(abs(y_range), 1.0) * 5
    sig = max(std(y), 1.0)
    chrom = [0.0, mu]
    lower = [-R, minimum(y) - 3sig]
    upper = [R, maximum(y) + 3sig]
    parnames = (:a, :b)
    return chrom, lower, upper, parnames, 0, 2
end

function _asymptotic_regression_defaults(y, n)
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    c_init = y_min - 1.0
    chrom = [y_max, y_min, 0.1, c_init]
    lower = [y_min - 3sig, y_min - 3sig, 1e-4, y_min - 3sig]
    upper = [y_max + 0.5y_range, y_max + 3sig, 5.0, y_max + 3sig]
    parnames = (:a, :b, :d, :c)
    return chrom, lower, upper, parnames, 0, 4
end

function _michaelis_menten_defaults(y, n)
    t = Float64.(1:n)
    y_min, y_max = minimum(y), maximum(y)
    Vmax_init = max(y_max, 1.0)
    Km_init = clamp(_half_max_t(y, t), 0.5, n)
    c_init = y_min - 1.0
    sig = max(std(y), 1.0)
    chrom = [Vmax_init, Km_init, c_init]
    lower = [0.01, 0.01, y_min - 3sig]
    upper = [1.5y_max, Float64(n), y_max + 3sig]
    parnames = (:Vmax, :Km, :c)
    return chrom, lower, upper, parnames, 0, 3
end

function _weibull_growth_defaults(y, n)
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    chrom = [y_max, y_range, 1.0 / max(n, 1), 1.0, y_min - 1.0]
    lower = [y_min, 0.01, 1e-4, 0.1, y_min - 3sig]
    upper = [y_max + 0.5y_range, 2y_range, 5.0, 5.0, y_max + 3sig]
    parnames = (:a, :b, :d, :e, :c)
    return chrom, lower, upper, parnames, 0, 5
end

function _hill_function_defaults(y, n)
    t = Float64.(1:n)
    y_min, y_max = minimum(y), maximum(y)
    a_init = y_max
    k_init = clamp(_half_max_t(y, t), 0.1, Float64(n))
    c_init = y_min - 1.0
    sig = max(std(y), 1.0)
    chrom = [a_init, k_init, 1.5, c_init]
    lower = [0.01, 0.1, 0.1, y_min - 3sig]
    upper = [1.5y_max, Float64(n), 5.0, y_max + 3sig]
    parnames = (:a, :k, :n, :c)
    return chrom, lower, upper, parnames, 0, 4
end

function _log_logistic_defaults(y, n)
    t = Float64.(1:n)
    y_min, y_max = minimum(y), maximum(y)
    a_init = y_max
    b_init = clamp(_half_max_t(y, t), 0.1, Float64(n))
    c_init = y_min - 1.0
    sig = max(std(y), 1.0)
    chrom = [a_init, b_init, 2.0, c_init]
    lower = [0.01, 0.1, 0.1, y_min - 3sig]
    upper = [1.5y_max, Float64(n), 5.0, y_max + 3sig]
    parnames = (:a, :b, :e, :c)
    return chrom, lower, upper, parnames, 0, 4
end

function _double_exponential_defaults(y, n)
    mu = mean(y)
    y_min, y_max = minimum(y), maximum(y)
    R = max(abs(y_min), abs(y_max)) * 10
    sig = max(std(y), 1.0)
    chrom = [mu, -0.1, mu, -1.0, 0.0]
    lower = [-R, -5.0, -R, -5.0, y_min - 3sig]
    upper = [R, 5.0, R, 5.0, y_max + 3sig]
    parnames = (:a, :b, :d, :e, :c)
    return chrom, lower, upper, parnames, 0, 5
end

function _rational_defaults(y, n)
    mu = mean(y)
    y_min, y_max = minimum(y), maximum(y)
    R = max(abs(y_min), abs(y_max)) * 10
    sig = max(std(y), 1.0)
    chrom = [0.0, mu, 1.0 / max(n, 1), 1.0, 0.0]
    # Keep denominator positive: d,e >= small positive. This is restrictive but
    # avoids poles inside the segment.
    lower = [-R, -R, 1e-4, 1e-4, y_min - 3sig]
    upper = [R, R, 10.0, 10.0, y_max + 3sig]
    parnames = (:a, :b, :d, :e, :c)
    return chrom, lower, upper, parnames, 0, 5
end

function _logistic_defaults(y, n)
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    chrom = [y_range, 0.1, Float64(n) / 2.0, y_min]
    lower = [0.01, 0.01, 0.0, y_min - 3sig]
    upper = [3.0 * y_range, 5.0, Float64(n), y_max + 3sig]
    parnames = (:K, :r, :t0, :c)
    return chrom, lower, upper, parnames, 0, 4
end

function _gompertz_defaults(y, n)
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    chrom = [y_range, 1.0, 1.0 / max(n, 1), y_min]
    lower = [0.01, 0.01, 1e-4, y_min - 3sig]
    upper = [3.0 * y_range, 10.0, 5.0, y_max + 3sig]
    parnames = (:a, :b, :d, :c)
    return chrom, lower, upper, parnames, 0, 4
end

function _saturating_exponential_defaults(y, n)
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    chrom = [y_range, 0.1, y_min]
    lower = [0.01, 1e-4, y_min - 3sig]
    upper = [3.0 * y_range, 5.0, y_max + 3sig]
    parnames = (:K, :r, :c)
    return chrom, lower, upper, parnames, 0, 3
end

function _debt_dynamics_defaults(y, n)
    chrom = [0.05, mean(y) * 0.05]
    y_min, y_max = minimum(y), maximum(y)
    R = max(abs(y_min), abs(y_max)) * 5
    lower = [-0.5, -R]
    upper = [0.5, R]
    parnames = (:r, :s)
    return chrom, lower, upper, parnames, 0, 2
end

function _accelerator_defaults(y, n)
    y_min, y_max = minimum(y), maximum(y)
    sig = max(std(y), 1.0)
    chrom = [mean(y), 0.5]
    lower = [y_min - 3sig, -2.0]
    upper = [y_max + 3sig, 2.0]
    parnames = (:c, :v)
    return chrom, lower, upper, parnames, 0, 2
end

function _compound_growth_defaults(y, n)
    r_init = std(diff(y)) / max(abs(mean(y)), 1.0)
    chrom = [r_init]
    lower = [-0.5]
    upper = [0.5]
    parnames = (:r,)
    return chrom, lower, upper, parnames, 0, 1
end

function _poisson_defaults(y, n)
    a_init = log(max(mean(y), 1.0))
    b_init = 0.0
    y_min, y_max = minimum(y), maximum(y)
    sig = max(std(y), 1.0)
    c_init = mean(y)
    chrom = [a_init, b_init, c_init]
    lower = [-5.0, -0.5, y_min - 3sig]
    upper = [5.0, 0.5, y_max + 3sig]
    parnames = (:a, :b, :c)
    return chrom, lower, upper, parnames, 0, 3
end

function _negbin_defaults(y, n)
    a_init = log(max(mean(y), 1.0))
    b_init = 0.0
    y_min, y_max = minimum(y), maximum(y)
    sig = max(std(y), 1.0)
    c_init = mean(y)
    chrom = [a_init, b_init, c_init]
    lower = [-5.0, -0.5, y_min - 3sig]
    upper = [5.0, 0.5, y_max + 3sig]
    parnames = (:a, :b, :c)
    return chrom, lower, upper, parnames, 0, 3
end

function _ingarch_defaults(y, n)
    omega_init = max(mean(y) * 0.1, 0.1)
    y_max = maximum(y)
    y_min = minimum(y)
    R = max(y_max, 1.0) * 5
    sig = max(std(y), 1.0)
    chrom = [omega_init, 0.2, 0.2, 0.0]
    lower = [1e-4, 0.0, 0.0, y_min - 3sig]
    upper = [R, 0.99, 0.99, y_max + 3sig]
    parnames = (:omega, :alpha, :beta, :c)
    return chrom, lower, upper, parnames, 0, 4
end

function _ets_defaults(y, n, mult::Bool)
    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    l0_init = mult ? max(y[1], 0.1) : y[1]
    b0_init = length(y) > 1 ? clamp(diff(y)[1] / max(abs(y[1]), 1.0), -2.0, 2.0) : 0.0
    chrom = [l0_init, b0_init, 0.1, 0.01, 0.01]
    if mult
        lower = [0.01, -2.0, 1e-4, 1e-4, 1e-4]
        upper = [y_max + 3sig, 2.0, 0.9999, 0.9999, 0.9999]
    else
        lower = [y_min - 3sig, -2 * max(abs(y_range), 1.0), 1e-4, 1e-4, 1e-4]
        upper = [y_max + 3sig, 2 * max(abs(y_range), 1.0), 0.9999, 0.9999, 0.9999]
    end
    parnames = (:l0, :b0, :alpha, :beta, :gamma)
    return chrom, lower, upper, parnames, 0, 5
end

# -----------------------------------------------------------------------------
# High-level builder
# -----------------------------------------------------------------------------

"""
    default_model_setup(y, n, model_name;
                        optimizer=nothing,
                        loss_function=nothing,
                        bounds=nothing,
                        initial_guess=nothing,
                        use_analytical=true,
                        continuity::Bool=false,
                        safe_bad::Real=1e12)

Build all components needed to run `detect_changepoints` for a named model,
using literature-based defaults for bounds, initial guesses, optimizer and loss.

# Arguments
- `y`: observed series (used for data-driven bounds / initial guesses).
- `n`: length of `y`.
- `model_name`: one of the 30 supported models (mean, linear, quadratic, cubic,
  exponential, logistic, gompertz, saturating_exponential, power, log_linear,
  mean_drift, ar1, ar2, ar3, hyperbolic, asymptotic_regression, michaelis_menten,
  weibull_growth, hill_function, log_logistic, double_exponential, rational,
  debt_dynamics, accelerator, compound_growth, poisson, negbin, ingarch,
  ets_aaa, ets_mmm).
- `optimizer`: optional override. If `nothing`, chosen automatically.
- `loss_function`: optional override. Wrapped with `safe_loss` internally.
- `bounds`: optional `(lower, upper)` tuple override.
- `initial_guess`: optional chromosome override.
- `use_analytical`: if `true` (default), use `AnalyticalOptimizer()` for models
  with a closed-form RSS fit when no custom loss is supplied.
- `continuity`: if `true`, enforce level continuity across changepoints for stateless
  models (Regression, Count, ETS, Volatility). The first segment is adjusted so
  that its first simulated value equals `y[1]`; later segments are adjusted so
  that each segment starts at the last simulated value of the previous segment.
  Analytical fits are automatically disabled when continuity is enabled because
  they fit each segment independently.
- `safe_bad`: finite penalty returned by `safe_loss` on failure.

# Returns
```
(manager, n_global, n_segment_specific, chromosome, bounds, parnames, optimizer, loss_function)
```
"""
function default_model_setup(y::Vector{Float64}, n::Int, model_name::String;
                             optimizer::Union{AbstractOptimizerConfig,Nothing}=nothing,
                             loss_function::Union{Function,Nothing}=nothing,
                             bounds::Union{Tuple{Vector{Float64},Vector{Float64}},Nothing}=nothing,
                             initial_guess::Union{Vector{Float64},Nothing}=nothing,
                             use_analytical::Bool=true,
                             continuity::Bool=false,
                             safe_bad::Real=1e12)

    y_min, y_max = minimum(y), maximum(y)
    y_range = y_max - y_min
    sig = max(std(y), 1.0)
    t = Float64.(1:n)

    if model_name == "mean"
        chrom, lower, upper, parnames, n_global, n_seg = _mean_defaults(y, n)
        ms = RegressionModelSpec(mean_model, [chrom[1]], n)
    elseif model_name == "linear"
        chrom, lower, upper, parnames, n_global, n_seg = _linear_defaults(y, n)
        ms = RegressionModelSpec(linear_model, chrom, n)
    elseif model_name == "linear_slope_only"
        chrom, lower, upper, parnames, n_global, n_seg = _linear_slope_only_defaults(y, n)
        ms = RegressionModelSpec(linear_slope_only_model, chrom, n)
    elseif model_name == "quadratic"
        chrom, lower, upper, parnames, n_global, n_seg = _polynomial_defaults(y, n, 2)
        ms = RegressionModelSpec(quadratic_model, chrom, n)
    elseif model_name == "cubic"
        chrom, lower, upper, parnames, n_global, n_seg = _polynomial_defaults(y, n, 3)
        ms = RegressionModelSpec(cubic_model, chrom, n)
    elseif model_name == "exponential"
        chrom, lower, upper, parnames, n_global, n_seg = _exponential_defaults(y, n)
        ms = RegressionModelSpec(exponential_model, chrom, n)
    elseif model_name == "log_linear"
        chrom, lower, upper, parnames, n_global, n_seg = _log_linear_defaults(y, n)
        ms = RegressionModelSpec(log_linear_model, chrom, n)
    elseif model_name == "power"
        chrom, lower, upper, parnames, n_global, n_seg = _power_defaults(y, n)
        ms = RegressionModelSpec(power_model, chrom, n)
    elseif model_name == "logistic"
        chrom, lower, upper, parnames, n_global, n_seg = _logistic_defaults(y, n)
        ms = RegressionModelSpec(logistic_model, chrom, n)
    elseif model_name == "gompertz"
        chrom, lower, upper, parnames, n_global, n_seg = _gompertz_defaults(y, n)
        ms = RegressionModelSpec(gompertz_model, chrom, n)
    elseif model_name == "saturating_exponential"
        chrom, lower, upper, parnames, n_global, n_seg = _saturating_exponential_defaults(y, n)
        ms = RegressionModelSpec(saturating_exponential_model, chrom, n)
    elseif model_name == "mean_drift"
        chrom, lower, upper, parnames, n_global, n_seg = _mean_drift_defaults(y, n)
        ms = RegressionModelSpec(mean_drift_model, chrom, n)
    elseif model_name == "ar1"
        chrom, lower, upper, parnames, n_global, n_seg = _ar_defaults(y, n, 1)
        ms = AutoRegressiveModelSpec(ar1_model, chrom, n, 1, [y[1]])
    elseif model_name == "ar1_nodrift"
        chrom, lower, upper, parnames, n_global, n_seg = _ar_nodrift_defaults(y, n)
        ms = AutoRegressiveModelSpec(ar1_nodrift_model, chrom, n, 1, [y[1]])
    elseif model_name == "ar2"
        chrom, lower, upper, parnames, n_global, n_seg = _ar_defaults(y, n, 2)
        ms = AutoRegressiveModelSpec(ar2_model, chrom, n, 2, [y[1], y[2]])
    elseif model_name == "ar3"
        chrom, lower, upper, parnames, n_global, n_seg = _ar_defaults(y, n, 3)
        ms = AutoRegressiveModelSpec(ar3_model, chrom, n, 3, [y[1], y[2], y[3]])
    elseif model_name == "hyperbolic"
        chrom, lower, upper, parnames, n_global, n_seg = _hyperbolic_defaults(y, n)
        ms = RegressionModelSpec(hyperbolic_model, chrom, n)
    elseif model_name == "asymptotic_regression"
        chrom, lower, upper, parnames, n_global, n_seg = _asymptotic_regression_defaults(y, n)
        ms = RegressionModelSpec(asymptotic_regression_model, chrom, n)
    elseif model_name == "michaelis_menten"
        chrom, lower, upper, parnames, n_global, n_seg = _michaelis_menten_defaults(y, n)
        ms = RegressionModelSpec(michaelis_menten_model, chrom, n)
    elseif model_name == "weibull_growth"
        chrom, lower, upper, parnames, n_global, n_seg = _weibull_growth_defaults(y, n)
        ms = RegressionModelSpec(weibull_growth_model, chrom, n)
    elseif model_name == "hill_function"
        chrom, lower, upper, parnames, n_global, n_seg = _hill_function_defaults(y, n)
        ms = RegressionModelSpec(hill_function_model, chrom, n)
    elseif model_name == "log_logistic"
        chrom, lower, upper, parnames, n_global, n_seg = _log_logistic_defaults(y, n)
        ms = RegressionModelSpec(log_logistic_model, chrom, n)
    elseif model_name == "double_exponential"
        chrom, lower, upper, parnames, n_global, n_seg = _double_exponential_defaults(y, n)
        ms = RegressionModelSpec(double_exponential_model, chrom, n)
    elseif model_name == "rational"
        chrom, lower, upper, parnames, n_global, n_seg = _rational_defaults(y, n)
        ms = RegressionModelSpec(rational_model, chrom, n)
    elseif model_name == "debt_dynamics"
        chrom, lower, upper, parnames, n_global, n_seg = _debt_dynamics_defaults(y, n)
        ms = DifferenceModelSpec(debt_dynamics_model, Dict(:r=>chrom[1], :s=>chrom[2]), y[1], n, (Float64[], Float64[]))
    elseif model_name == "accelerator"
        chrom, lower, upper, parnames, n_global, n_seg = _accelerator_defaults(y, n)
        ms = DifferenceModelSpec(accelerator_model, Dict(:c=>chrom[1], :v=>chrom[2]), y[1], n, (Float64[], Float64[]))
    elseif model_name == "compound_growth"
        chrom, lower, upper, parnames, n_global, n_seg = _compound_growth_defaults(y, n)
        ms = DifferenceModelSpec(compound_growth_model, Dict(:r=>chrom[1]), y[1], n, (Float64[], Float64[]))
    elseif model_name == "poisson"
        chrom, lower, upper, parnames, n_global, n_seg = _poisson_defaults(y, n)
        ms = CountModelSpec(poisson_model, chrom, n, :poisson)
    elseif model_name == "negbin"
        chrom, lower, upper, parnames, n_global, n_seg = _negbin_defaults(y, n)
        ms = CountModelSpec(negbin_model, chrom, n, :negbin)
    elseif model_name == "ingarch"
        chrom, lower, upper, parnames, n_global, n_seg = _ingarch_defaults(y, n)
        ms = CountModelSpec(ingarch_model, chrom, n, :poisson)
    elseif model_name == "ets_aaa"
        chrom, lower, upper, parnames, n_global, n_seg = _ets_defaults(y, n, false)
        ms = ETSModelSpec(ets_aaa_model, chrom, n, 12)
    elseif model_name == "ets_mmm"
        chrom, lower, upper, parnames, n_global, n_seg = _ets_defaults(y, n, true)
        ms = ETSModelSpec(ets_mmm_model, chrom, n, 12)
    else
        error("Unknown model: $model_name")
    end

    # Apply user overrides
    if bounds !== nothing
        lower, upper = bounds
    end
    if initial_guess !== nothing
        chrom = initial_guess
    end

    # Nudge initial guess strictly inside bounds
    chrom = [_nudge_inward(chrom[i], lower[i], upper[i]) for i in 1:length(chrom)]

    # Choose optimizer
    custom_loss = loss_function !== nothing
    # Level continuity requires numerical optimisation across segments; analytical
    # per-segment fits do not propagate the boundary level.
    use_analytical_eff = use_analytical && !continuity
    if optimizer === nothing
        optimizer = default_optimizer(model_name; use_analytical=use_analytical_eff && !custom_loss)
    end

    # Choose loss and wrap it safely
    if loss_function === nothing
        loss_function = default_loss(model_name)
    end
    safe_loss_fn = safe_loss_factory(loss_function; bad=safe_bad, warn=false)

    manager = ModelManager(ms; continuity=continuity)
    return manager, n_global, n_seg, chrom, (lower, upper), parnames, optimizer, safe_loss_fn
end
