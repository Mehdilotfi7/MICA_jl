# module ModelSimulation

# =============================================================================
# Model Specification Types
# =============================================================================

"""
Abstract base type for all model specifications.
"""
abstract type AbstractModelSpec end


"""
Specification for an ODE (Ordinary Differential Equation) model.

# Fields
- `model_function::Function`: A function that defines the ODE dynamics.
- `params::Dict{Symbol, Any}`: Parameters needed for the model.
- `initial_conditions::Vector{Float64}`: Initial conditions for the ODE system.
- `tspan::Tuple{Float64, Float64}`: Time span over which to simulate the model.
"""
struct ODEModelSpec <: AbstractModelSpec
    model_function::Function
    params
    initial_conditions::Vector{Float64}
    tspan::Tuple{Float64, Float64}
end

"""
Specification for a discrete Difference Equation model.

# Fields
- `model_function::Function`: A function that defines the difference dynamics.
- `params::Dict{Symbol, Any}`: Parameters needed for the model.
- `initial_conditions::Float64`: Initial state of the system.
- `num_steps::Int`: Number of time steps for the simulation.
- `extra_data::Tuple{Vector{Float64}, Vector{Float64}}`: Additional inputs (e.g., external variables).
"""
struct DifferenceModelSpec <: AbstractModelSpec
    model_function::Function
    params
    initial_conditions::Float64
    num_steps::Int
    extra_data::Tuple{Vector{Float64}, Vector{Float64}}
end

"""
Specification for a simple Regression model (e.g., linear).

# Fields
- `model_function::Function`: A function that defines the regression output.
- `params::Dict{Symbol, Any}`: Parameters needed for the model.
- `time_steps::Int`: Number of time steps or observations.
"""
struct RegressionModelSpec <: AbstractModelSpec
    model_function::Function
    params
    time_steps::Int
    t_start::Int
end

RegressionModelSpec(model_function::Function, params, time_steps::Int) =
    RegressionModelSpec(model_function, params, time_steps, 1)

"""
    ARIMAModelSpec <: AbstractModelSpec

A specification type for defining ARIMA model behavior in the Mica changepoint detection framework.

# Fields
- `model_function::Function`: A function that simulates the ARIMA model given parameters, time span, and optional inputs. Typically returns a time series array of shape `(1, T)` where `T` is the number of time steps.
- `initial_conditions::Vector{Float64}`: First `d` observations used to seed the integrated series (defaults to zeros). For segment continuity this should be set to the last `d` values of the previous segment.

# Usage
Used as input to `ModelManager` for segment-based simulation and parameter estimation via Mica's genetic algorithm-based changepoint detection pipeline.

# Example
```julia
spec = ARIMAModelSpec(
    simulate_model,     # ARIMA simulation function
    [0.1, 0.2, -0.1, 0.05],  # μ, AR, MA coefficients
    200,                # simulate over 200 time steps
    1,                  # AR order p
    1,                  # differencing d
    1,                  # MA order q
    [0.0]               # initial observation(s) for integration
)
"""

struct ARIMAModelSpec <: AbstractModelSpec
    model_function::Function
    params::Vector
    time_steps::Int
    p::Int
    d::Int
    q::Int
    initial_conditions::Vector{Float64}
end

ARIMAModelSpec(model_function::Function, params::Vector, time_steps::Int, p::Int, d::Int, q::Int) =
    ARIMAModelSpec(model_function, params, time_steps, p, d, q, zeros(d))

"""
    AutoRegressiveModelSpec <: AbstractModelSpec

Specification for autoregressive models (AR1, AR2, AR3, etc.) with initial condition propagation across segments.

# Fields
- `model_function::Function`: AR simulation function with signature `(params, time_steps, y0) -> Matrix{Float64}`
- `params`: Model parameters
- `time_steps::Int`: Number of time steps
- `order::Int`: AR order (1, 2, 3, ...)
- `initial_conditions::Vector{Float64}`: Last `order` observations to seed the simulation
"""
struct AutoRegressiveModelSpec <: AbstractModelSpec
    model_function::Function
    params
    time_steps::Int
    order::Int
    initial_conditions::Vector{Float64}
end

# =============================================================================
# Model Simulation Functions
# =============================================================================

"""
Simulates an ODEModelSpec by solving the ODE system.

# Arguments
- `model::ODEModelSpec`: An ODE model specification.

# Returns
- Simulated results over time.
"""
function simulate_model(model::ODEModelSpec)
    return model.model_function(model.params, model.tspan, model.initial_conditions)
end

"""
Simulates a DifferenceModelSpec by iterating the discrete equation.

# Arguments
- `model::DifferenceModelSpec`: A Difference model specification.

# Returns
- Simulated results over discrete time steps.
"""
function simulate_model(model::DifferenceModelSpec)
    return model.model_function(model.params, model.initial_conditions, model.num_steps, model.extra_data)
end

"""
Simulates a RegressionModelSpec by evaluating the regression model.

# Arguments
- `model::RegressionModelSpec`: A Regression model specification.

# Returns
- Simulated outputs.
"""
function simulate_model(model::RegressionModelSpec)
    return model.model_function(model.params, model.time_steps; t_start=model.t_start)
end

"""
Simulates an AutoRegressiveModelSpec with propagated initial conditions.

# Arguments
- `model::AutoRegressiveModelSpec`: An autoregressive model specification.

# Returns
- Simulated values of shape `(1, time_steps)`.
"""
function simulate_model(model::AutoRegressiveModelSpec)
    return model.model_function(model.params, model.time_steps; y0=model.initial_conditions)
end

"""
    simulate_model(model::ARIMAModelSpec)

Simulate an ARIMA(p,d,q) model.

**Domain:** Macro / Finance — e.g., `nile`, `gdp_*`, `businv` (Box-Jenkins time series).

First applies d-th order differencing, then simulates the ARMA(p,q) component,
then integrates back.

# Arguments
- `model::ARIMAModelSpec`: ARIMA specification with fields `p`, `d`, `q`, `params`.

# Returns
- `Matrix{Float64}`: Simulated series of shape `(1, time_steps)`.
"""
function simulate_model(model::ARIMAModelSpec)
    μ = model.params[1]
    ar_coeffs = model.params[2:1+model.p]
    ma_coeffs = model.params[2+model.p:end]
    time_steps = model.time_steps
    d = model.d

    # Ensure initial condition has length d; pad with zeros if necessary
    y0 = length(model.initial_conditions) >= d ? model.initial_conditions[1:d] : zeros(d)

    # Simulate ARMA(p,q) in differences
    y = zeros(time_steps + d)
    ε = randn(time_steps + d)  # white noise innovations

    # Seed the first d integrated values so that the final series starts at y0
    y[1:d] .= y0

    for t in max(d + 1, model.p + 1):time_steps + d
        ar_part = sum(ar_coeffs[i] * y[t - i] for i in 1:model.p; init=0.0)
        ma_part = sum(ma_coeffs[i] * ε[t - i] for i in 1:model.q; init=0.0)
        y[t] = μ + ar_part + ma_part + ε[t]
    end

    # Integrate d times
    for _ in 1:d
        y = cumsum(y)
    end

    return reshape(y[1:time_steps], 1, :)
end

# =============================================================================
# Built-in Regression Model Functions
# =============================================================================

"""
    mean_model(params, time_steps::Int; t_start::Int=1)

Constant mean model: `y = μ`.

# Parameters
- `params[1]` or `params[:μ]`: mean value

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function mean_model(params, time_steps::Int; t_start::Int=1)
    μ = params[1]
    simulated_values = fill(μ, time_steps)
    return reshape(simulated_values, 1, :)
end


"""
    linear_model(params, time_steps::Int; t_start::Int=1)

Linear trend model: `y = a * t + b`.

# Parameters
- `params[1]` or `params[:a]`: slope
- `params[2]` or `params[:b]`: intercept

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function linear_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = a .* time .+ b
    return reshape(simulated_values, 1, :)
end


"""
    linear_slope_only_model(params, time_steps::Int; t_start::Int=1)

Slope-only continuous piecewise-linear model: `y = y0 + slope * (t - t_start)`.

This mirrors the LR toy-dataset generator: each segment has its own slope,
but the level `y0` is propagated from the previous segment (or the first
observation) so the fitted signal is continuous at changepoints.

# Parameters
- `params[1]` or `params[:slope]`: segment slope
- `params[2]` or `params[:y0]`: level at `t_start` (set by continuity enforcement)

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function linear_slope_only_model(params, time_steps::Int; t_start::Int=1)
    slope = params[1]
    y0 = params[2]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = y0 .+ slope .* (time .- t_start)
    return reshape(simulated_values, 1, :)
end


"""
    quadratic_model(params, time_steps::Int; t_start::Int=1)

Quadratic model: `y = a * t² + b * t + c`.

# Parameters
- `params[1]` or `params[:a]`: quadratic coefficient
- `params[2]` or `params[:b]`: linear coefficient
- `params[3]` or `params[:c]`: intercept

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function quadratic_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    c = params[3]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = a .* time.^2 .+ b .* time .+ c
    return reshape(simulated_values, 1, :)
end


"""
    cubic_model(params, time_steps::Int; t_start::Int=1)

Cubic model: `y = a * t³ + b * t² + c * t + d`.

# Parameters
- `params[1:4]`: coefficients [a, b, c, d]

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function cubic_model(params, time_steps::Int; t_start::Int=1)
    a, b, c, d = params[1], params[2], params[3], params[4]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = a .* time.^3 .+ b .* time.^2 .+ c .* time .+ d
    return reshape(simulated_values, 1, :)
end


"""
    exponential_model(params, time_steps::Int; t_start::Int=1)

Exponential growth/decay model with additive level: `y = c + a * exp(b * t)`.

# Parameters
- `params[1]` or `params[:a]`: amplitude
- `params[2]` or `params[:b]`: growth rate
- `params[3]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function exponential_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    c = params[3]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = c .+ a .* exp.(b .* time)
    return reshape(simulated_values, 1, :)
end


"""
    logistic_model(params, time_steps::Int; t_start::Int=1)

Logistic growth model with additive level: `y = c + K / (1 + exp(-r * (t - t0)))`.

# Parameters
- `params[1]` or `params[:K]`: carrying capacity
- `params[2]` or `params[:r]`: growth rate
- `params[3]` or `params[:t0]`: inflection point
- `params[4]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function logistic_model(params, time_steps::Int; t_start::Int=1)
    K = params[1]
    r = params[2]
    t0 = params[3]
    c = params[4]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = c .+ K ./ (1.0 .+ exp.(-r .* (time .- t0)))
    return reshape(simulated_values, 1, :)
end


"""
    gompertz_model(params, time_steps::Int; t_start::Int=1)

Gompertz growth model with additive level: `y = c + a * exp(-b * exp(-d * t))`.

# Parameters
- `params[1]` or `params[:a]`: amplitude
- `params[2]` or `params[:b]`: displacement
- `params[3]` or `params[:d]`: growth rate
- `params[4]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function gompertz_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    d = params[3]
    c = params[4]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = c .+ a .* exp.(-b .* exp.(-d .* time))
    return reshape(simulated_values, 1, :)
end


"""
    saturating_exponential_model(params, time_steps::Int; t_start::Int=1)

Saturating exponential model: `y = K * (1 - exp(-r * t)) + c`.

# Parameters
- `params[1]` or `params[:K]`: saturation level
- `params[2]` or `params[:r]`: rate parameter
- `params[3]` or `params[:c]`: offset

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function saturating_exponential_model(params, time_steps::Int; t_start::Int=1)
    K = params[1]
    r = params[2]
    c = params[3]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = K .* (1.0 .- exp.(-r .* time)) .+ c
    return reshape(simulated_values, 1, :)
end


"""
    power_model(params, time_steps::Int; t_start::Int=1)

Power law model with additive level: `y = c + a * t^b`.

# Parameters
- `params[1]` or `params[:a]`: amplitude
- `params[2]` or `params[:b]`: exponent
- `params[3]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function power_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    c = params[3]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = c .+ a .* (time .^ b)
    return reshape(simulated_values, 1, :)
end


"""
    log_linear_model(params, time_steps::Int; t_start::Int=1)

Log-linear model: `y = a * log(t) + b`.

# Parameters
- `params[1]` or `params[:a]`: coefficient
- `params[2]` or `params[:b]`: intercept

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function log_linear_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = a .* log.(time) .+ b
    return reshape(simulated_values, 1, :)
end


"""
    mean_drift_model(params, time_steps::Int; t_start::Int=1)

Mean with linear drift: `y = μ + a * t + b`.

# Parameters
- `params[1]` or `params[:μ]`: mean offset
- `params[2]` or `params[:a]`: slope
- `params[3]` or `params[:b]`: intercept

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function mean_drift_model(params, time_steps::Int; t_start::Int=1)
    μ = params[1]
    a = params[2]
    b = params[3]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = μ .+ a .* time .+ b
    return reshape(simulated_values, 1, :)
end


"""
    hyperbolic_model(params, time_steps::Int; t_start::Int=1)

Hyperbolic model: `y = a / t + b`.

**Domain:** Sports — e.g., `run_log` (critical speed / hyperbolic fatigue curves).

# Parameters
- `params[1]` or `params[:a]`: amplitude
- `params[2]` or `params[:b]`: offset

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function hyperbolic_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = a ./ time .+ b
    return reshape(simulated_values, 1, :)
end

"""
    asymptotic_regression_model(params, time_steps::Int; t_start::Int=1)

Asymptotic regression (monomolecular) model with additive level:
`y = c + a - b * exp(-d * t)`.

**Domain:** Environment — e.g., `ozone`, `centralia` (exponential decay to asymptote).

# Parameters
- `params[1]` or `params[:a]`: asymptote
- `params[2]` or `params[:b]`: span from intercept to asymptote
- `params[3]` or `params[:d]`: rate parameter
- `params[4]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function asymptotic_regression_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    d = params[3]
    c = params[4]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = c .+ a .- b .* exp.(-d .* time)
    return reshape(simulated_values, 1, :)
end

"""
    michaelis_menten_model(params, time_steps::Int; t_start::Int=1)

Michaelis-Menten saturation model with additive level:
`y = c + Vmax * t / (Km + t)`.

**Domain:** Biology / Macro — e.g., `rail_lines` (infrastructure saturation), enzyme kinetics.

# Parameters
- `params[1]` or `params[:Vmax]`: maximum value (asymptote)
- `params[2]` or `params[:Km]`: half-saturation constant
- `params[3]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function michaelis_menten_model(params, time_steps::Int; t_start::Int=1)
    Vmax = params[1]
    Km = params[2]
    c = params[3]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = c .+ Vmax .* time ./ (Km .+ time)
    return reshape(simulated_values, 1, :)
end

"""
    weibull_growth_model(params, time_steps::Int; t_start::Int=1)

Weibull growth model with additive level: `y = c + a - b * exp(-d * t^e)`.

**Domain:** Demographic — e.g., `us_population`, `children_per_woman` (flexible sigmoid growth).

# Parameters
- `params[1]` or `params[:a]`: asymptote
- `params[2]` or `params[:b]`: span
- `params[3]` or `params[:d]`: rate
- `params[4]` or `params[:e]`: shape parameter
- `params[5]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function weibull_growth_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    d = params[3]
    e = params[4]
    c = params[5]
    time = t_start:(t_start + time_steps - 1)
    simulated_values = c .+ a .- b .* exp.(-d .* time.^e)
    return reshape(simulated_values, 1, :)
end

"""
    hill_function_model(params, time_steps::Int; t_start::Int=1)

Hill function (dose-response / cooperative binding) with additive level:
`y = c + a * t^n / (k^n + t^n)`.

**Domain:** Biology / Health — e.g., `measles` (threshold-like epidemic response).

# Parameters
- `params[1]` or `params[:a]`: maximum response
- `params[2]` or `params[:k]`: half-maximum constant (EC50)
- `params[3]` or `params[:n]`: Hill coefficient (cooperativity)
- `params[4]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function hill_function_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    k = max(params[2], 1e-6)
    n = params[3]
    c = params[4]
    time = Float64.(t_start:(t_start + time_steps - 1))
    # Compute safely in log-space to avoid overflow: a / (1 + (k/t)^n)
    simulated_values = zeros(Float64, time_steps)
    for i in 1:time_steps
        t = time[i]
        if t == 0.0
            simulated_values[i] = c
        else
            ratio = k / t
            # Clamp exponent to avoid overflow
            log_ratio_n = clamp(n * log(ratio), -700.0, 700.0)
            denom = 1.0 + exp(log_ratio_n)
            simulated_values[i] = c + a / denom
        end
    end
    return reshape(simulated_values, 1, :)
end

"""
    log_logistic_model(params, time_steps::Int; t_start::Int=1)

Log-logistic model with additive level: `y = c + a / (1 + (t / b)^e)`.

**Domain:** Demographic / Survival — e.g., `us_population` (growth with heavier tails than logistic).

# Parameters
- `params[1]` or `params[:a]`: maximum value
- `params[2]` or `params[:b]`: scale parameter (median)
- `params[3]` or `params[:e]`: shape parameter
- `params[4]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function log_logistic_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = max(params[2], 1e-6)
    e = params[3]
    c = params[4]
    time = Float64.(t_start:(t_start + time_steps - 1))
    # Compute safely: a / (1 + exp(e * log(time/b)))
    simulated_values = zeros(Float64, time_steps)
    for i in 1:time_steps
        t = time[i]
        log_tb = log(t / b)
        exponent = clamp(e * log_tb, -700.0, 700.0)
        simulated_values[i] = c + a / (1.0 + exp(exponent))
    end
    return reshape(simulated_values, 1, :)
end

"""
    double_exponential_model(params, time_steps::Int; t_start::Int=1)

Double exponential model (two-phase decay/growth) with additive level:
`y = c + a * exp(b * t) + d * exp(e * t)`.

**Domain:** Finance / Environment — e.g., `bitcoin`, `ozone` (two distinct rate processes).

# Parameters
- `params[1]` or `params[:a]`: first amplitude
- `params[2]` or `params[:b]`: first rate
- `params[3]` or `params[:d]`: second amplitude
- `params[4]` or `params[:e]`: second rate
- `params[5]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function double_exponential_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    d = params[3]
    e = params[4]
    c = params[5]
    time = t_start:(t_start + time_steps - 1)
    # Clamp exponentials to avoid Inf/NaN overflow
    safe_exp(x) = exp(clamp(x, -700.0, 700.0))
    simulated_values = c .+ a .* safe_exp.(b .* time) .+ d .* safe_exp.(e .* time)
    return reshape(simulated_values, 1, :)
end

"""
    rational_model(params, time_steps::Int; t_start::Int=1)

Rational function model with additive level: `y = c + (a * t + b) / (d * t + e)`.

**Domain:** General flexible curve fitting — e.g., `centralia`, `uk_coal_employ` (decline with horizontal asymptote).

# Parameters
- `params[1]` or `params[:a]`: numerator slope
- `params[2]` or `params[:b]`: numerator intercept
- `params[3]` or `params[:d]`: denominator slope
- `params[4]` or `params[:e]`: denominator intercept
- `params[5]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function rational_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    d = params[3]
    e = params[4]
    c = params[5]
    time = t_start:(t_start + time_steps - 1)
    # Ensure denominator stays safely positive to avoid division by zero
    denom = max.(d .* time .+ e, 1e-6)
    simulated_values = c .+ (a .* time .+ b) ./ denom
    return reshape(simulated_values, 1, :)
end


# =============================================================================
# Autoregressive Model Functions
# =============================================================================

"""
    ar1_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)

First-order autoregressive model: `y[t] = c + φ * y[t-1]`.

# Parameters
- `params[1]` or `params[:φ]`: AR coefficient
- `params[2]` or `params[:c]`: constant/drift

# Optional
- `y0::Vector{Float64}`: initial value as 1-element vector (defaults to c / (1 - φ) if |φ| < 1, else 0)

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function ar1_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)
    φ = params[1]
    c = params[2]
    y = zeros(time_steps)
    if y0 === nothing || isempty(y0)
        y[1] = abs(φ) < 1 ? c / (1 - φ) : 0.0
    else
        y[1] = y0[end]
    end
    for t in 2:time_steps
        y[t] = c + φ * y[t-1]
    end
    return reshape(y, 1, :)
end


"""
    ar1_nodrift_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)

Zero-drift first-order autoregressive model: `y[t] = φ * y[t-1]`.

This mirrors the AR toy-dataset generator, where only the AR coefficient
changes at changepoints and there is no constant/drift term.

# Parameters
- `params[1]` or `params[:φ]`: AR coefficient

# Optional
- `y0::Vector{Float64}`: initial value as a 1-element vector (defaults to 0)

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function ar1_nodrift_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)
    φ = params[1]
    y = zeros(time_steps)
    if y0 === nothing || isempty(y0)
        y[1] = 0.0
    else
        y[1] = y0[end]
    end
    for t in 2:time_steps
        y[t] = φ * y[t-1]
    end
    return reshape(y, 1, :)
end


"""
    ar2_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)

Second-order autoregressive model: `y[t] = c + φ1 * y[t-1] + φ2 * y[t-2]`.

# Parameters
- `params[1]` or `params[:φ1]`: first AR coefficient
- `params[2]` or `params[:φ2]`: second AR coefficient
- `params[3]` or `params[:c]`: constant/drift

# Optional
- `y0::Vector{Float64}`: initial values as a vector of last 2 observations

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function ar2_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)
    φ1 = params[1]
    φ2 = params[2]
    c = params[3]
    y = zeros(time_steps)
    if y0 === nothing || length(y0) < 2
        denom = 1 - φ1 - φ2
        y[1] = abs(denom) > 1e-10 && abs(φ2) < 1 ? c / denom : 0.0
        y[2] = y[1]
    else
        y[1] = y0[end-1]
        y[2] = y0[end]
    end
    for t in 3:time_steps
        y[t] = c + φ1 * y[t-1] + φ2 * y[t-2]
    end
    return reshape(y, 1, :)
end


"""
    ar3_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)

Third-order autoregressive model: `y[t] = c + φ1*y[t-1] + φ2*y[t-2] + φ3*y[t-3]`.

# Parameters
- `params[1:3]`: AR coefficients [φ1, φ2, φ3]
- `params[4]` or `params[:c]`: constant/drift

# Optional
- `y0::Vector{Float64}`: initial values as a vector of last 3 observations

# Returns
- `Matrix{Float64}`: Simulated values of shape `(1, time_steps)`.
"""
function ar3_model(params, time_steps::Int; y0::Union{Vector{Float64},Nothing}=nothing)
    φ1 = params[1]
    φ2 = params[2]
    φ3 = params[3]
    c = params[4]
    y = zeros(time_steps)
    if y0 === nothing || length(y0) < 3
        y[1] = 0.0
        y[2] = 0.0
        y[3] = 0.0
    else
        y[1] = y0[end-2]
        y[2] = y0[end-1]
        y[3] = y0[end]
    end
    for t in 4:time_steps
        y[t] = c + φ1 * y[t-1] + φ2 * y[t-2] + φ3 * y[t-3]
    end
    return reshape(y, 1, :)
end


# =============================================================================
# Example Model Functions (legacy)
# =============================================================================

"""
Example: Simple exponential decay ODE model.

Defines the dynamics `du/dt = -p * u`.

# Arguments
- `params::Dict`: Model parameters, expects key `:p`.
- `tspan::Tuple`: (start_time, end_time).
- `u0::Vector{Float64}`: Initial condition vector.

# Returns
- `Matrix`: state variable evolution.
"""
function exponential_ode_model(p, tspan, u0)
    function ode!(du, u, p, t)
        du[1] = -p[1] * u[1]
    end

    prob = ODEProblem(ode!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)

    return Matrix(sol)
end

"""
    sir_ode_model(params, tspan, u0)

SIR epidemic ODE model.

**Domain:** Health — e.g., `measles` (compartmental epidemic dynamics).

# Arguments
- `params::Vector{Float64}`: `[β, γ]` — infection and recovery rates.
- `tspan::Tuple{Float64,Float64}`: simulation time span.
- `u0::Vector{Float64}`: `[S0, I0, R0]` — initial susceptible, infected, recovered.

# Returns
- `Matrix{Float64}`: `(3, n_timepoints)` — [S; I; R] trajectories.
"""
function sir_ode_model(params, tspan, u0)
    β, γ = params[1], params[2]
    function sir!(du, u, p, t)
        S, I, R = u
        N = S + I + R
        du[1] = -p[1] * S * I / N
        du[2] = p[1] * S * I / N - p[2] * I
        du[3] = p[2] * I
    end
    prob = ODEProblem(sir!, u0, tspan, [β, γ])
    sol = solve(prob, Tsit5(), saveat=1.0)
    return Matrix(sol)
end

"""
    seir_ode_model(params, tspan, u0)

SEIR epidemic ODE model with exposed compartment.

**Domain:** Health — e.g., `measles` (extended compartmental epidemic with latent period).

# Arguments
- `params::Vector{Float64}`: `[β, σ, γ]` — infection, incubation, and recovery rates.
- `tspan::Tuple{Float64,Float64}`: simulation time span.
- `u0::Vector{Float64}`: `[S0, E0, I0, R0]` — initial compartments.

# Returns
- `Matrix{Float64}`: `(4, n_timepoints)` — [S; E; I; R] trajectories.
"""
function seir_ode_model(params, tspan, u0)
    β, σ, γ = params[1], params[2], params[3]
    function seir!(du, u, p, t)
        S, E, I, R = u
        N = S + E + I + R
        du[1] = -p[1] * S * I / N
        du[2] = p[1] * S * I / N - p[2] * E
        du[3] = p[2] * E - p[3] * I
        du[4] = p[3] * I
    end
    prob = ODEProblem(seir!, u0, tspan, [β, σ, γ])
    sol = solve(prob, Tsit5(), saveat=1.0)
    return Matrix(sol)
end

"""
    logistic_ode_model(params, tspan, u0)

Logistic growth ODE: `dy/dt = r * y * (1 - y / K)`.

**Domain:** Demographic — e.g., `us_population`, `children_per_woman` (Verhulst growth).

# Arguments
- `params::Vector{Float64}`: `[r, K]` — growth rate and carrying capacity.
- `tspan::Tuple{Float64,Float64}`: simulation time span.
- `u0::Vector{Float64}`: `[y0]` — initial population.

# Returns
- `Matrix{Float64}`: `(1, n_timepoints)` — population trajectory.
"""
function logistic_ode_model(params, tspan, u0)
    r, K = params[1], params[2]
    function logistic!(du, u, p, t)
        du[1] = p[1] * u[1] * (1.0 - u[1] / p[2])
    end
    prob = ODEProblem(logistic!, u0, tspan, [r, K])
    sol = solve(prob, Tsit5(), saveat=1.0)
    return Matrix(sol)
end

"""
Example: Discrete difference equation model.

Simulates a difference equation influenced by external variables.

# Arguments
- `params::Dict`: Model parameters.
- `initial_conditions::Float64`: Initial state.
- `num_steps::Int`: Number of steps.
- `extra_data::Tuple`: (wind_speeds, ambient_temperatures).

# Returns
- `DataFrame`: Time and state variable evolution.
"""
function example_difference_model(params, initial_conditions, num_steps, extra_data)
    wind_speeds, ambient_temperatures = extra_data

    state_values = zeros(num_steps)
    state_values[1] = initial_conditions

    for k in 2:num_steps
        u1 = wind_speeds[k]
        u2 = ambient_temperatures[k]
        y_prev = state_values[k - 1]

        state_values[k] = (params[:θ1] * u1^3 + params[:θ2] * u1^2 + params[:θ3] * u1 + y_prev - u2) /
                          (params[:θ4] * u1^3 + params[:θ5] * u1^2 + params[:θ6] * u1 + params[:θ7]) + u2
    end

    time = 1:num_steps
    return reshape(state_values, 1, :)
end

"""
    debt_dynamics_model(params, initial_conditions, num_steps, extra_data)

Debt dynamics difference equation: `d_t = (1 + r) * d_{t-1} - s`.

**Domain:** Macro — e.g., `debt_ireland` (Domar debt dynamics).

# Arguments
- `params::Dict`: expects keys `:r` (interest rate), `:s` (surplus / primary balance).
- `initial_conditions::Float64`: initial debt level `d_0`.
- `num_steps::Int`: number of time steps.
- `extra_data::Tuple`: ignored (pass `([], [])` if not needed).

# Returns
- `Matrix{Float64}`: Debt evolution of shape `(1, num_steps)`.
"""
function debt_dynamics_model(params, initial_conditions, num_steps, extra_data)
    r = params[:r]
    s = params[:s]
    d = zeros(num_steps)
    d[1] = initial_conditions
    for t in 2:num_steps
        d[t] = (1.0 + r) * d[t-1] - s
    end
    return reshape(d, 1, :)
end

"""
    accelerator_model(params, initial_conditions, num_steps, extra_data)

Accelerator model: `y_t = c + v * (y_{t-1} - y_{t-2})`.

**Domain:** Macro — e.g., `businv` (inventory accelerator, investment cycles).

# Arguments
- `params::Dict`: expects keys `:c` (autonomous component), `:v` (accelerator coefficient).
- `initial_conditions::Float64`: initial output level.
- `num_steps::Int`: number of time steps.
- `extra_data::Tuple`: ignored.

# Returns
- `Matrix{Float64}`: Output evolution of shape `(1, num_steps)`.
"""
function accelerator_model(params, initial_conditions, num_steps, extra_data)
    c = params[:c]
    v = params[:v]
    y = zeros(num_steps)
    y[1] = initial_conditions
    y[2] = initial_conditions
    for t in 3:num_steps
        y[t] = c + v * (y[t-1] - y[t-2])
    end
    return reshape(y, 1, :)
end

"""
    compound_growth_model(params, initial_conditions, num_steps, extra_data)

Compound growth model: `y_t = y_{t-1} * (1 + r)`.

**Domain:** Finance — e.g., `bank` (compound interest, geometric Brownian motion drift).

# Arguments
- `params::Dict`: expects key `:r` (growth rate).
- `initial_conditions::Float64`: initial value `y_0`.
- `num_steps::Int`: number of time steps.
- `extra_data::Tuple`: ignored.

# Returns
- `Matrix{Float64}`: Value evolution of shape `(1, num_steps)`.
"""
function compound_growth_model(params, initial_conditions, num_steps, extra_data)
    r = params[:r]
    y = zeros(num_steps)
    y[1] = initial_conditions
    for t in 2:num_steps
        y[t] = y[t-1] * (1.0 + r)
    end
    return reshape(y, 1, :)
end

"""
    example_regression_model(params, time_steps::Int; t_start::Int=1)

Example: Simple linear regression model.

Simulates a linear trend `y = a * t + b`.

# Arguments
- `params::Dict`: Model parameters, expects `:a` and `:b`.
- `time_steps::Int`: Number of time steps.
- `t_start::Int`: Starting time index (default 1).

# Returns
- `DataFrame`: Time and simulated values.
"""
function example_regression_model(params, time_steps::Int; t_start::Int=1)
    simulated_values = [params[1] * t + params[2] for t in t_start:(t_start + time_steps - 1)]
    return reshape(simulated_values, 1, :)
end

#end # module

# =============================================================================
# Analytical least-squares fitting for regression models
# =============================================================================

"""
    fit_segment_analytical(model_spec::RegressionModelSpec, segment_data::Matrix{Float64})

Fit a regression model to a single segment using exact least squares.
Returns `(loss, params_vector)`.
"""
function fit_segment_analytical(model_spec::RegressionModelSpec, segment_data::Matrix{Float64})
    return fit_segment_analytical(model_spec.model_function, segment_data)
end

function fit_segment_analytical(model_spec::AutoRegressiveModelSpec, segment_data::Matrix{Float64})
    return fit_segment_analytical(model_spec.model_function, segment_data)
end

function fit_segment_analytical(::typeof(mean_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    mu = mean(y)
    pred = fill(mu, length(y))
    loss = sum((y .- pred).^2)
    return loss, [mu]
end

function fit_segment_analytical(::typeof(linear_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    A = hcat(t, ones(n))
    coeffs = A \ y
    a, b = coeffs
    pred = a .* t .+ b
    loss = sum((y .- pred).^2)
    return loss, [a, b]
end

function fit_segment_analytical(::typeof(linear_slope_only_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    if n < 2
        return sum((y .- y[1]).^2), [0.0, y[1]]
    end
    t = Float64.(1:n)
    y0 = y[1]
    # Fit a line through the first point: minimize RSS of y = y0 + slope * (t - 1)
    dt = t .- 1.0
    slope = sum(dt .* (y .- y0)) / sum(dt.^2)
    pred = y0 .+ slope .* dt
    loss = sum((y .- pred).^2)
    return loss, [slope, y0]
end

function fit_segment_analytical(::typeof(quadratic_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    A = hcat(t.^2, t, ones(n))
    coeffs = A \ y
    a, b, c = coeffs
    pred = a .* t.^2 .+ b .* t .+ c
    loss = sum((y .- pred).^2)
    return loss, [a, b, c]
end

function fit_segment_analytical(::typeof(cubic_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    A = hcat(t.^3, t.^2, t, ones(n))
    coeffs = A \ y
    a, b, c, d = coeffs
    pred = a .* t.^3 .+ b .* t.^2 .+ c .* t .+ d
    loss = sum((y .- pred).^2)
    return loss, [a, b, c, d]
end

function fit_segment_analytical(::typeof(exponential_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    y_min = minimum(y)
    offset = y_min <= 0 ? abs(y_min) + 1.0 : 0.0
    log_y = log.(y .+ offset)
    A = hcat(t, ones(n))
    coeffs = A \ log_y
    b, log_a = coeffs
    a = exp(log_a)
    pred = a .* exp.(b .* t) .- offset
    loss = sum((y .- pred).^2)
    # New parameterization: y = c + a*exp(b*t); offset = -c
    return loss, [a, b, -offset]
end

function fit_segment_analytical(::typeof(power_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    y_min = minimum(y)
    offset = y_min <= 0 ? abs(y_min) + 1.0 : 0.0
    log_y = log.(y .+ offset)
    log_t = log.(t)
    A = hcat(log_t, ones(n))
    coeffs = A \ log_y
    b, log_a = coeffs
    a = exp(log_a)
    pred = a .* (t .^ b) .- offset
    loss = sum((y .- pred).^2)
    # New parameterization: y = c + a*t^b; offset = -c
    return loss, [a, b, -offset]
end

function fit_segment_analytical(::typeof(log_linear_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    log_t = log.(t)
    A = hcat(log_t, ones(n))
    coeffs = A \ y
    a, b = coeffs
    pred = a .* log_t .+ b
    loss = sum((y .- pred).^2)
    return loss, [a, b]
end

function fit_segment_analytical(::typeof(mean_drift_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    A = hcat(t, ones(n))
    ab = A \ y
    a, b = ab
    detrended = y .- (a .* t .+ b)
    mu = mean(detrended)
    pred = mu .+ a .* t .+ b
    loss = sum((y .- pred).^2)
    return loss, [mu, a, b]
end

function fit_segment_analytical(::typeof(ar1_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    if n < 3
        mu = mean(y)
        return sum((y .- mu).^2), [0.0, mu]
    end
    y_t = y[2:end]
    y_lag = y[1:end-1]
    A = hcat(y_lag, ones(n-1))
    coeffs = A \ y_t
    phi, c = coeffs
    pred = vcat([y[1]], c .+ phi .* y_lag)
    loss = sum((y .- pred).^2)
    return loss, [phi, c]
end

function fit_segment_analytical(::typeof(ar1_nodrift_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    if n < 2
        return sum(y.^2), [0.0]
    end
    y_t = y[2:end]
    y_lag = y[1:end-1]
    # Conditional least squares without drift: phi = sum(y_t * y_lag) / sum(y_lag^2)
    phi = sum(y_t .* y_lag) / sum(y_lag.^2)
    pred = vcat([y[1]], phi .* y_lag)
    loss = sum((y .- pred).^2)
    return loss, [phi]
end

function fit_segment_analytical(::typeof(ar2_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    if n < 4
        return fit_segment_analytical(ar1_model, segment_data)
    end
    y_t = y[3:end]
    y_lag1 = y[2:end-1]
    y_lag2 = y[1:end-2]
    A = hcat(y_lag1, y_lag2, ones(n-2))
    coeffs = A \ y_t
    phi1, phi2, c = coeffs
    pred = vcat(y[1:2], c .+ phi1 .* y_lag1 .+ phi2 .* y_lag2)
    loss = sum((y .- pred).^2)
    return loss, [phi1, phi2, c]
end

function fit_segment_analytical(::typeof(ar3_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    if n < 5
        return fit_segment_analytical(ar2_model, segment_data)
    end
    y_t = y[4:end]
    y_lag1 = y[3:end-1]
    y_lag2 = y[2:end-2]
    y_lag3 = y[1:end-3]
    A = hcat(y_lag1, y_lag2, y_lag3, ones(n-3))
    coeffs = A \ y_t
    phi1, phi2, phi3, c = coeffs
    pred = vcat(y[1:3], c .+ phi1 .* y_lag1 .+ phi2 .* y_lag2 .+ phi3 .* y_lag3)
    loss = sum((y .- pred).^2)
    return loss, [phi1, phi2, phi3, c]
end

function fit_segment_analytical(::typeof(hyperbolic_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    # Linear in 1/t: y = a*(1/t) + b
    X = hcat(1.0 ./ t, ones(n))
    coeffs = X \ y
    a, b = coeffs
    pred = a ./ t .+ b
    loss = sum((y .- pred).^2)
    return loss, [a, b]
end

function fit_segment_analytical(::typeof(asymptotic_regression_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    # New parameterization: y = c + a - b*exp(-d*t).  Fix a heuristic shift
    # c = min(y) - 1 and fit the old shape to y_shifted = y - c.
    c_shift = minimum(y) - 1.0
    y_shifted = y .- c_shift
    best_loss = Inf
    best_params = [mean(y_shifted), 1.0, 0.1, c_shift]
    for d in 10.0.^(-3:0.1:1)
        X = hcat(ones(n), -exp.(-d .* t))
        coeffs = X \ y_shifted
        a, b = coeffs
        if a > maximum(y_shifted) || b < 0; continue; end
        pred = c_shift .+ a .- b .* exp.(-d .* t)
        loss = sum((y .- pred).^2)
        if loss < best_loss
            best_loss = loss
            best_params = [a, b, d, c_shift]
        end
    end
    return best_loss, best_params
end

function fit_segment_analytical(::typeof(michaelis_menten_model), segment_data::Matrix{Float64})
    y = vec(segment_data)
    n = length(y)
    t = Float64.(1:n)
    # New parameterization: y = c + Vmax*t/(Km+t).  Fix a heuristic shift
    # c = min(y) - 1 and fit the old shape to y_shifted = y - c.
    c_shift = minimum(y) - 1.0
    y_shifted = y .- c_shift
    valid = y_shifted .> 0
    if sum(valid) < 3
        return sum((y .- mean(y)).^2), [mean(y), mean(t), c_shift]
    end
    y_inv = 1.0 ./ y_shifted[valid]
    t_inv = 1.0 ./ t[valid]
    X = hcat(t_inv, ones(length(t_inv)))
    coeffs = X \ y_inv
    Km_over_Vmax, one_over_Vmax = coeffs
    Vmax = 1.0 / one_over_Vmax
    Km = Km_over_Vmax * Vmax
    pred = c_shift .+ Vmax .* t ./ (Km .+ t)
    loss = sum((y .- pred).^2)
    return loss, [Vmax, Km, c_shift]
end

function fit_segment_analytical(::typeof(weibull_growth_model), segment_data::Matrix{Float64})
    error("Analytical fitting not implemented for model function weibull_growth_model")
end

function fit_segment_analytical(::typeof(hill_function_model), segment_data::Matrix{Float64})
    error("Analytical fitting not implemented for model function hill_function_model")
end

function fit_segment_analytical(::typeof(log_logistic_model), segment_data::Matrix{Float64})
    error("Analytical fitting not implemented for model function log_logistic_model")
end

function fit_segment_analytical(::typeof(double_exponential_model), segment_data::Matrix{Float64})
    error("Analytical fitting not implemented for model function double_exponential_model")
end

function fit_segment_analytical(::typeof(rational_model), segment_data::Matrix{Float64})
    error("Analytical fitting not implemented for model function rational_model")
end

# Fallback for unsupported models
function fit_segment_analytical(model_fn::Function, segment_data::Matrix{Float64})
    error("Analytical fitting not implemented for model function $model_fn")
end

#end # module

# =============================================================================
# Volatility Model Specification (GARCH family)
# =============================================================================

"""
    VolatilityModelSpec <: AbstractModelSpec

Specification for volatility models (GARCH, EGARCH, TGARCH).

**Domain:** Finance — `bitcoin`, `apple`, `brent_spot`, `ratner_stock`, `usd_isk`.

The `model_function` returns a 2×T matrix where row 1 is the mean series
and row 2 is the conditional variance series.

# Fields
- `model_function::Function`: volatility model simulator
- `params`: model parameters
- `time_steps::Int`: number of time steps
"""
struct VolatilityModelSpec <: AbstractModelSpec
    model_function::Function
    params
    time_steps::Int
    t_start::Int
end

VolatilityModelSpec(model_function::Function, params, time_steps::Int) =
    VolatilityModelSpec(model_function, params, time_steps, 1)

"""
    garch_model(params, time_steps::Int)

GARCH(1,1) variance model: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`.

Mean is constant: `y_t = μ + ε_t` where `ε_t ~ N(0, σ²_t)`.

# Parameters
- `params[1]` or `params[:μ]`: mean
- `params[2]` or `params[:ω]`: variance intercept
- `params[3]` or `params[:α]`: ARCH coefficient
- `params[4]` or `params[:β]`: GARCH coefficient

# Returns
- `Matrix{Float64}`: `(2, time_steps)` — [mean_series; variance_series].
"""
function garch_model(params, time_steps::Int; t_start::Int=1)
    μ = params[1]
    ω = max(params[2], 1e-6)
    α = params[3]
    β = params[4]
    σ2 = zeros(time_steps)
    σ2[1] = ω / max(1.0 - α - β, 1e-6)
    for t in 2:time_steps
        σ2[t] = ω + α * σ2[t-1] + β * σ2[t-1]
    end
    return vcat(fill(μ, time_steps)', σ2')
end

"""
    egarch_model(params, time_steps::Int)

EGARCH(1,1) log-variance model.

# Parameters
- `params[1]` or `params[:μ]`: mean
- `params[2]` or `params[:ω]`: log-variance intercept
- `params[3]` or `params[:α]`: ARCH effect
- `params[4]` or `params[:β]`: GARCH effect
- `params[5]` or `params[:γ]`: leverage effect

# Returns
- `Matrix{Float64}`: `(2, time_steps)` — [mean_series; variance_series].
"""
function egarch_model(params, time_steps::Int; t_start::Int=1)
    μ = params[1]
    ω = params[2]
    α = params[3]
    β = params[4]
    γ = params[5]
    log_σ2 = zeros(time_steps)
    log_σ2[1] = ω
    for t in 2:time_steps
        log_σ2[t] = ω + β * log_σ2[t-1] + γ * log_σ2[t-1]
    end
    σ2 = exp.(log_σ2)
    return vcat(fill(μ, time_steps)', σ2')
end

"""
    tgarch_model(params, time_steps::Int)

Threshold GARCH(1,1) model.

# Parameters
- `params[1]` or `params[:μ]`: mean
- `params[2]` or `params[:ω]`: variance intercept
- `params[3]` or `params[:α]`: positive shock coefficient
- `params[4]` or `params[:β]`: GARCH coefficient
- `params[5]` or `params[:γ]`: negative shock coefficient

# Returns
- `Matrix{Float64}`: `(2, time_steps)` — [mean_series; variance_series].
"""
function tgarch_model(params, time_steps::Int; t_start::Int=1)
    μ = params[1]
    ω = max(params[2], 1e-6)
    α = params[3]
    β = params[4]
    γ = params[5]
    σ2 = zeros(time_steps)
    σ2[1] = ω / max(1.0 - α - β - γ, 1e-6)
    for t in 2:time_steps
        σ2[t] = ω + (α + γ) * σ2[t-1] + β * σ2[t-1]
    end
    return vcat(fill(μ, time_steps)', σ2')
end

# =============================================================================
# Count Model Specification (Poisson / Negative Binomial / INGARCH)
# =============================================================================

"""
    CountModelSpec <: AbstractModelSpec

Specification for count-data models.

**Domain:** Health / IoT — `robocalls`, `measles`, `occupancy`.

# Fields
- `model_function::Function`: count model simulator
- `params`: model parameters
- `time_steps::Int`: number of time steps
- `distribution::Symbol`: `:poisson`, `:negbin`
"""
struct CountModelSpec <: AbstractModelSpec
    model_function::Function
    params
    time_steps::Int
    distribution::Symbol
    t_start::Int
end

CountModelSpec(model_function::Function, params, time_steps::Int, distribution::Symbol) =
    CountModelSpec(model_function, params, time_steps, distribution, 1)

"""
    poisson_model(params, time_steps::Int)

Poisson rate model with additive level: `y_t = c + exp(a + b * t)`.

# Parameters
- `params[1]` or `params[:a]`: log-rate intercept
- `params[2]` or `params[:b]`: log-rate trend
- `params[3]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: `(1, time_steps)` — expected counts `y_t`.
"""
function poisson_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    c = params[3]
    time = t_start:(t_start + time_steps - 1)
    λ = c .+ exp.(a .+ b .* time)
    return reshape(λ, 1, :)
end

"""
    negbin_model(params, time_steps::Int)

Negative binomial mean model with additive level: `y_t = c + exp(a + b * t)`.

# Parameters
- `params[1]` or `params[:a]`: log-mean intercept
- `params[2]` or `params[:b]`: log-mean trend
- `params[3]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: `(1, time_steps)` — expected mean `y_t`.
"""
function negbin_model(params, time_steps::Int; t_start::Int=1)
    a = params[1]
    b = params[2]
    c = params[3]
    time = t_start:(t_start + time_steps - 1)
    μ = c .+ exp.(a .+ b .* time)
    return reshape(μ, 1, :)
end

"""
    ingarch_model(params, time_steps::Int)

INGARCH(1,1) count model with additive level: `y_t = c + λ_t`, where
`λ_t = ω + α * λ_{t-1} + β * λ_{t-1}`.

# Parameters
- `params[1]` or `params[:ω]`: intercept
- `params[2]` or `params[:α]`: observation feedback
- `params[3]` or `params[:β]`: autoregressive feedback
- `params[4]` or `params[:c]`: additive shift / level

# Returns
- `Matrix{Float64}`: `(1, time_steps)` — expected intensity `y_t`.
"""
function ingarch_model(params, time_steps::Int; t_start::Int=1)
    ω = max(params[1], 1e-6)
    α = params[2]
    β = params[3]
    c = params[4]
    λ = zeros(time_steps)
    λ[1] = ω / max(1.0 - α - β, 1e-6)
    for t in 2:time_steps
        λ[t] = ω + α * λ[t-1] + β * λ[t-1]
    end
    return reshape(c .+ λ, 1, :)
end

# =============================================================================
# ETS Model Specification (Exponential Smoothing)
# =============================================================================

"""
    ETSModelSpec <: AbstractModelSpec

Specification for exponential smoothing state-space models.

**Domain:** Transport / Environment — `jfk_passengers`, `lga_passengers`, `iceland_tourism`, `co2_canada`.

# Fields
- `model_function::Function`: ETS simulator
- `params`: model parameters
- `time_steps::Int`: number of time steps
- `seasonal_period::Int`: seasonal period (1 for non-seasonal)
"""
struct ETSModelSpec <: AbstractModelSpec
    model_function::Function
    params
    time_steps::Int
    seasonal_period::Int
    t_start::Int
end

ETSModelSpec(model_function::Function, params, time_steps::Int, seasonal_period::Int) =
    ETSModelSpec(model_function, params, time_steps, seasonal_period, 1)

"""
    ets_aaa_model(params, time_steps::Int)

ETS(A,A,A) — additive error, additive trend, additive seasonality (Holt-Winters).

# Parameters
- `params[1]` or `params[:l0]`: initial level
- `params[2]` or `params[:b0]`: initial trend
- `params[3]` or `params[:α]`: level smoothing
- `params[4]` or `params[:β]`: trend smoothing
- `params[5]` or `params[:γ]`: seasonal smoothing

# Returns
- `Matrix{Float64}`: `(1, time_steps)` — forecast series.
"""
function ets_aaa_model(params, time_steps::Int; t_start::Int=1)
    l0 = params[1]
    b0 = params[2]
    α = params[3]
    β = params[4]
    γ = params[5]

    # Deterministic trend + seasonal + quadratic term
    # All 5 parameters directly affect the output
    time = t_start:(t_start + time_steps - 1)
    y = zeros(time_steps)
    for (i, t) in enumerate(time)
        seasonal = α * sin(2π * t / 12.0) + β * cos(2π * t / 12.0)
        y[i] = l0 + b0 * t + γ * t^2 + seasonal
    end
    return reshape(y, 1, :)
end

"""
    ets_mmm_model(params, time_steps::Int)

ETS(M,M,M) — multiplicative error, multiplicative trend, multiplicative seasonality.

# Parameters
- `params[1]` or `params[:l0]`: initial level
- `params[2]` or `params[:b0]`: initial trend
- `params[3]` or `params[:α]`: level smoothing
- `params[4]` or `params[:β]`: trend smoothing
- `params[5]` or `params[:γ]`: seasonal smoothing

# Returns
- `Matrix{Float64}`: `(1, time_steps)` — forecast series.
"""
function ets_mmm_model(params, time_steps::Int; t_start::Int=1)
    l0 = params[1]
    b0 = params[2]
    α = params[3]
    β = params[4]
    γ = params[5]

    # Exponential trend + seasonal + linear correction
    # All 5 parameters directly affect the output
    time = t_start:(t_start + time_steps - 1)
    y = zeros(time_steps)
    for (i, t) in enumerate(time)
        seasonal = α * sin(2π * t / 12.0) + β * cos(2π * t / 12.0)
        y[i] = l0 * exp(b0 * t / 100.0) + γ * t + seasonal
    end
    return reshape(y, 1, :)
end

# =============================================================================
# simulate_model dispatches for new spec types
# =============================================================================

function simulate_model(model::CountModelSpec)
    return model.model_function(model.params, model.time_steps; t_start=model.t_start)
end

function simulate_model(model::VolatilityModelSpec)
    return model.model_function(model.params, model.time_steps; t_start=model.t_start)
end

function simulate_model(model::ETSModelSpec)
    return model.model_function(model.params, model.time_steps; t_start=model.t_start)
end
