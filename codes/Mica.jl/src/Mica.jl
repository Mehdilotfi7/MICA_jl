module Mica

# ========== Dependencies ==========
using Evolutionary
using OrdinaryDiffEq
using LabelledArrays
using Random
using Plots
using Dates
using Statistics
using CSV
using SciMLBase
using Smoothers
using DataFrames
using Measurements
using Printf
using Distributed
using Optim
using Metaheuristics


# ========== Internal Modules ==========
include("ModelSimulation.jl")
include("ModelHandling.jl")
include("ModelContinuity.jl")
include("ObjectiveFunction.jl")
include("LossFunctions.jl")
include("ChangePointDetection.jl")
include("penalty.jl")
include("ModelDefaults.jl")
include("Visualization.jl")

# ========== Exported API ==========

# -- Model Simulation --
export AbstractModelSpec
export ODEModelSpec, DifferenceModelSpec, RegressionModelSpec, ARIMAModelSpec, AutoRegressiveModelSpec
export VolatilityModelSpec, CountModelSpec, ETSModelSpec
export simulate_model, exponential_ode_model
export example_difference_model, example_regression_model

# Regression models
export mean_model, linear_model, linear_slope_only_model, quadratic_model, cubic_model
export exponential_model, logistic_model, gompertz_model, saturating_exponential_model
export power_model, log_linear_model, mean_drift_model
export hyperbolic_model, asymptotic_regression_model, michaelis_menten_model
export weibull_growth_model, hill_function_model, log_logistic_model
export double_exponential_model, rational_model

# Autoregressive models
export ar1_model, ar1_nodrift_model, ar2_model, ar3_model

# ODE models
export sir_ode_model, seir_ode_model, logistic_ode_model

# Difference models
export debt_dynamics_model, accelerator_model, compound_growth_model

# Volatility models
export garch_model, egarch_model, tgarch_model

# Count models
export poisson_model, negbin_model, ingarch_model

# ETS models
export ets_aaa_model, ets_mmm_model

# -- Model Handling --
export ModelManager, get_initial_condition, update_initial_condition, segment_model, get_model_type
export get_continuity, needs_level_continuity, continuity_target_level
export adjust_segment_params_for_continuity

# -- Objective Function --
export extract_parameters
export objective_function, wrapped_obj_function

# -- Loss Functions --
export safe_loss, safe_loss_factory
export rss_loss, mse_loss, rmse_loss, l1_loss, mae_loss, huber_loss
export gaussian_nll, poisson_nll, negbin_nll

# -- Model Defaults --
export default_model_setup, default_optimizer, default_loss

# -- Change Point Detection --
export AbstractOptimizerConfig
export EvolutionaryOptimizer, OptimOptimizer, MetaheuristicsOptimizer, AnalyticalOptimizer
export suggest_optimizer, optimize_params
export optimize_with_changepoints, update_bounds!, evaluate_segment, detect_changepoints

# -- Penalties --
export call_penalty_fn, method_argnames, default_penalty, BIC_penalty
export compute_information_criterion
export compute_tcpd_changepoint_penalty, compute_tcpd_wbs_penalty, compute_tcpd_rfpop_penalty
export fit_segment_analytical

# -- Visualization --
export simulate_full_model, plot_parameter_changes

end # module Mica
