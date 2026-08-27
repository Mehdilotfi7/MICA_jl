# module _ModelHandling


# =============================================================================
# ModelManager: Unified Model Interface for Change Point Detection
# =============================================================================

"""
`ModelManager{T<:AbstractModelSpec}`

A wrapper around any model specification (ODE, Difference, or Regression) that provides
a unified interface to:

- Access the initial condition
- Segment model-specific data for a given interval
- Generate per-segment models with updated parameters
- Identify the model type for dispatching logic
- Optionally enforce level continuity across changepoints for stateless models.

This allows the objective function and other algorithms to remain model-agnostic.
"""
struct ModelManager{T<:AbstractModelSpec}
    base_model::T
    continuity::Bool
end

ModelManager(model::AbstractModelSpec; continuity::Bool=false) = ModelManager(model, continuity)
get_continuity(manager::ModelManager) = manager.continuity

# =============================================================================
# Initial Condition Accessor
# =============================================================================

"""
`get_initial_condition(manager::ModelManager) -> Any`

Returns the initial condition used by the model. For:

- ODE models: returns the vector of initial states.
- Difference models: returns the scalar initial value.
- Regression models: returns `nothing` (no initial condition concept).
"""
get_initial_condition(manager::ModelManager{ODEModelSpec}) = manager.base_model.initial_conditions
get_initial_condition(manager::ModelManager{DifferenceModelSpec}) = manager.base_model.initial_conditions
get_initial_condition(manager::ModelManager{RegressionModelSpec}) = nothing
get_initial_condition(manager::ModelManager{AutoRegressiveModelSpec}) = manager.base_model.initial_conditions
get_initial_condition(manager::ModelManager{ARIMAModelSpec}) = manager.base_model.initial_conditions
get_initial_condition(manager::ModelManager{VolatilityModelSpec}) = nothing
get_initial_condition(manager::ModelManager{CountModelSpec}) = nothing
get_initial_condition(manager::ModelManager{ETSModelSpec}) = nothing

# =============================================================================
# Initial Condition Updating
# =============================================================================

"""
`update_initial_condition(manager::ModelManager, sim_data::DataFrame)`

Returns the updated initial condition for the next segment based on
the output of the last segment's simulation.
"""
function update_initial_condition(manager::ModelManager{ODEModelSpec}, sim_data)
    return sim_data[:,end]
end

function update_initial_condition(manager::ModelManager{DifferenceModelSpec}, sim_data)
    #@show sim_data
    return sim_data[end]
end

function update_initial_condition(manager::ModelManager{RegressionModelSpec}, sim_data)
    return nothing
end

function update_initial_condition(manager::ModelManager{AutoRegressiveModelSpec}, sim_data)
    order = manager.base_model.order
    n = length(sim_data)
    if n >= order
        return vec(sim_data)[(n - order + 1):n]
    else
        # Pad with zeros if segment is shorter than order
        return [zeros(order - n); vec(sim_data)]
    end
end

function update_initial_condition(manager::ModelManager{ARIMAModelSpec}, sim_data)
    d = manager.base_model.d
    n = length(sim_data)
    if n >= d
        return vec(sim_data)[(n - d + 1):n]
    else
        return [zeros(d - n); vec(sim_data)]
    end
end

function update_initial_condition(manager::ModelManager{VolatilityModelSpec}, sim_data)
    return nothing
end

function update_initial_condition(manager::ModelManager{CountModelSpec}, sim_data)
    return nothing
end

function update_initial_condition(manager::ModelManager{ETSModelSpec}, sim_data)
    return nothing
end


# =============================================================================
# Model Segmentation
# =============================================================================

"""
`segment_model(manager, seg_pars, parnames, idx_start, idx_end, u0) -> AbstractModelSpec`

Builds a new per-segment model specification using:

- The segment-specific parameters `seg_pars` and their names `parnames`
- The index range `[idx_start:idx_end]` defining the segment
- The initial condition `u0` passed from the last segment

Dispatches based on model type to slice and prepare data correctly.
"""
function segment_model(
    manager::ModelManager{ODEModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    u0
    )

    model = manager.base_model
    
    return ODEModelSpec(model.model_function, pars, u0, (idx_start, idx_end))
end

function segment_model(
    manager::ModelManager{DifferenceModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    u0
)
    model = manager.base_model
    
    segmented_extra = map(x -> length(x) >= idx_end ? x[idx_start:idx_end] : x, model.extra_data)
    num_steps = idx_end - idx_start + 1
    return DifferenceModelSpec(model.model_function, pars, u0, num_steps, segmented_extra)
end

function segment_model(
    manager::ModelManager{RegressionModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    _  # no initial condition
)
    model = manager.base_model
    time_steps = idx_end - idx_start + 1
    return RegressionModelSpec(model.model_function, pars, time_steps, idx_start)
end

function segment_model(
    manager::ModelManager{AutoRegressiveModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    u0::Vector{Float64}
)
    model = manager.base_model
    time_steps = idx_end - idx_start + 1
    return AutoRegressiveModelSpec(model.model_function, pars, time_steps, model.order, u0)
end

function segment_model(
    manager::ModelManager{ARIMAModelSpec},
    pars,
    idx_start::Int,
    idx_end::Int,
    u0::Vector{Float64}
)
    model = manager.base_model
    time_steps = idx_end - idx_start + 1
    return ARIMAModelSpec(model.model_function, pars, time_steps, model.p, model.d, model.q, u0)
end

function segment_model(
    manager::ModelManager{VolatilityModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    _
)
    model = manager.base_model
    time_steps = idx_end - idx_start + 1
    return VolatilityModelSpec(model.model_function, pars, time_steps, idx_start)
end

function segment_model(
    manager::ModelManager{CountModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    _
)
    model = manager.base_model
    time_steps = idx_end - idx_start + 1
    return CountModelSpec(model.model_function, pars, time_steps, model.distribution, idx_start)
end

function segment_model(
    manager::ModelManager{ETSModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    _
)
    model = manager.base_model
    time_steps = idx_end - idx_start + 1
    return ETSModelSpec(model.model_function, pars, time_steps, model.seasonal_period, idx_start)
end

# =============================================================================
# Optional Model Type Name
# =============================================================================

"""
`get_model_type(manager::ModelManager) -> String`

Returns a string identifier for the model type: "ODE", "Difference", or "Regression".
Useful for logging or conditional logic.
"""
get_model_type(manager::ModelManager{ODEModelSpec}) = "ODE"
get_model_type(manager::ModelManager{DifferenceModelSpec}) = "Difference"
get_model_type(manager::ModelManager{RegressionModelSpec}) = "Regression"
get_model_type(manager::ModelManager{AutoRegressiveModelSpec}) = "AutoRegressive"
get_model_type(manager::ModelManager{ARIMAModelSpec}) = "ARIMA"
get_model_type(manager::ModelManager{VolatilityModelSpec}) = "Volatility"
get_model_type(manager::ModelManager{CountModelSpec}) = "Count"
get_model_type(manager::ModelManager{ETSModelSpec}) = "ETS"

#end # module
