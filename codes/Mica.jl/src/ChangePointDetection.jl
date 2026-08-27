# module ChangePointDetection

"""
# ChangePointDetection Module

This module implements a framework for detecting change points in time-series or similar data
using optimization techniques. It relies on a generic `ModelManager` interface and uses
Evolutionary.jl for optimization.

### Main Functions:
- `detect_changepoints`: Detect change points in data.
- `optimize_with_changepoints`: Optimize the model parameters with fixed change points.
- `evaluate_segment`: Evaluate all possible change points in a given segment.
- `update_bounds!`: Dynamically update bounds and chromosome.
"""

# =============================================================================
# Optimizer Configuration Interface
# =============================================================================

abstract type AbstractOptimizerConfig end

struct EvolutionaryOptimizer <: AbstractOptimizerConfig
    ga::Any
    options::Any
    seed::Union{Int,Nothing}
end

EvolutionaryOptimizer(ga; options=Evolutionary.Options(show_trace=false), seed=nothing) =
    EvolutionaryOptimizer(ga, options, seed)

struct OptimOptimizer <: AbstractOptimizerConfig
    method::Any
    options::Any
    seed::Union{Int,Nothing}
end

OptimOptimizer(method=Fminbox(LBFGS()); options=Optim.Options(show_trace=false), seed=nothing) =
    OptimOptimizer(method, options, seed)

struct MetaheuristicsOptimizer <: AbstractOptimizerConfig
    algorithm::Any
    options::Any
    seed::Union{Int,Nothing}
end

MetaheuristicsOptimizer(algorithm=ECA(); options=Metaheuristics.Options(), seed=nothing) =
    MetaheuristicsOptimizer(algorithm, options, seed)

"""
    AnalyticalOptimizer

Optimizer that uses exact least-squares fitting for regression/AR models.
No iterative optimization — computes globally optimal parameters in one step.
"""
struct AnalyticalOptimizer <: AbstractOptimizerConfig end

function optimize_params(config::EvolutionaryOptimizer, f, x0, lower, upper)
    if config.seed !== nothing
        Random.seed!(config.seed)
    end
    result = Evolutionary.optimize(f, BoxConstraints(lower, upper), x0, config.ga, config.options)
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end

function optimize_params(config::OptimOptimizer, f, x0, lower, upper)
    if config.seed !== nothing
        Random.seed!(config.seed)
    end
    if config.method isa Optim.Fminbox
        result = Optim.optimize(f, lower, upper, x0, config.method, config.options)
    else
        result = Optim.optimize(f, x0, config.method, config.options)
    end
    return Optim.minimum(result), Optim.minimizer(result)
end

function optimize_params(config::MetaheuristicsOptimizer, f, x0, lower, upper)
    if config.seed !== nothing
        Random.seed!(config.seed)
    end
    bounds = [lower upper]
    
    # Use deepcopy to ensure a fresh, uninitialized algorithm for every run,
    # avoiding dimension-mismatch errors when segment counts change.
    fresh_alg = deepcopy(config.algorithm)
    
    # Seed the population with x0 so the optimizer starts warm (like Evolutionary.jl).
    # Metaheuristics.create_child wraps (position, objective_value) into an Individual.
    # Setting it in status.population + status.best_sol causes gen_initial_state to
    # keep this individual and fill the rest of the population randomly via
    # _complete_population!.
    x0_sol = Metaheuristics.create_child(copy(x0), f(x0))
    fresh_alg.status = Metaheuristics.State(x0_sol, [x0_sol])
    
    # Enable threaded parallel population evaluation if parallel_evaluation is set
    # in the algorithm options.  Wrap scalar objective into a batch function that
    # evaluates each row of the population matrix on a separate thread.
    use_parallel = fresh_alg.options.parallel_evaluation
    if use_parallel
        function f_batch(X::Matrix{Float64})
            n_rows = size(X, 1)
            fitness = Vector{Float64}(undef, n_rows)
            Threads.@threads for i in 1:n_rows
                fitness[i] = f(X[i, :])
            end
            return fitness
        end
        result = Metaheuristics.optimize(f_batch, bounds, fresh_alg)
    else
        result = Metaheuristics.optimize(f, bounds, fresh_alg)
    end
    
    return Metaheuristics.minimum(result), Metaheuristics.minimizer(result)
end

function suggest_optimizer(model_type::String, n_params::Int; complex::Bool=false)
    if complex || n_params > 20
        ga = GA(populationSize=150, selection=uniformranking(20),
                crossover=MILX(0.01, 0.17, 0.5), mutationRate=0.3,
                crossoverRate=0.6, mutation=gaussian(0.0001))
        return EvolutionaryOptimizer(ga)
    elseif n_params > 6
        ga = GA(populationSize=80, selection=uniformranking(10),
                crossover=MILX(0.01, 0.17, 0.5), mutationRate=0.3,
                crossoverRate=0.6, mutation=gaussian(0.0001))
        return EvolutionaryOptimizer(ga)
    else
        return OptimOptimizer(Fminbox(LBFGS()),
                              options=Optim.Options(show_trace=false, iterations=500))
    end
end

# =============================================================================
# Animation state (lazy initialization)
# =============================================================================
const global_anim = Ref{Union{Animation,Nothing}}(nothing)

function reset_animation()
    global_anim[] = Animation()
end

function get_animation()
    if global_anim[] === nothing
        global_anim[] = Animation()
    end
    return global_anim[]
end

# ----------------------------------------------------------------------
# Function: optimize_with_changepoints
# ----------------------------------------------------------------------
"""
    optimize_with_changepoints(
        objective_function, chromosome, CP, bounds, ga,
        n_global, n_segment_specific, parnames,
        model_manager, loss_function, data; options=...
    )

Optimize model parameters for a fixed set of change points using Evolutionary.jl.

Returns the minimum loss and the best parameter set.
"""
function optimize_with_changepoints(
    objective_function, chromosome, parnames, CP, bounds, optimizer::AbstractOptimizerConfig,
    n_global, n_segment_specific,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64}
)
    wrapped_obj(chrom) = objective_function(
        chrom, CP, parnames, n_global, n_segment_specific,
        model_manager, loss_function, data
    )
    return optimize_params(optimizer, wrapped_obj, chromosome, bounds[1], bounds[2])
end

# ----------------------------------------------------------------------
# Function: update_bounds!
# ----------------------------------------------------------------------
"""
    update_bounds!(chromosome, bounds, n_global, n_segment_specific, extract_parameters)

Update bounds and chromosome by appending segment-specific parameters.
"""
function update_bounds!(chromosome, bounds, n_global, n_segment_specific, extract_parameters)
    _, seg_specific = extract_parameters(chromosome, n_global, n_segment_specific)
    _, seg_lower = extract_parameters(bounds[1], n_global, n_segment_specific)
    _, seg_upper = extract_parameters(bounds[2], n_global, n_segment_specific)

    # i should add the estimation of previous segment pars as initial guesses for next segment
    append!(chromosome, seg_specific[1])
    append!(bounds[1], seg_lower[1])
    append!(bounds[2], seg_upper[1])
end

function update_bounds!(initial_chromosome, previous_chromosome, bounds, n_global, n_segment_specific, extract_parameters)
    _, seg_specific = extract_parameters(previous_chromosome, n_global, n_segment_specific)
    _, seg_lower = extract_parameters(bounds[1], n_global, n_segment_specific)
    _, seg_upper = extract_parameters(bounds[2], n_global, n_segment_specific)

    # i should add the estimation of previous segment pars as initial guesses for next segment
    append!(initial_chromosome, seg_specific[end])
    append!(bounds[1], seg_lower[1])
    append!(bounds[2], seg_upper[1])
end

# ----------------------------------------------------------------------
# Function: compute_objective
# ----------------------------------------------------------------------
"""
    compute_objective(loss, n, n_global, n_segment_specific, num_cps,
                      objective_type, penalty_fn, CP, segment_lengths,
                      sigma2_null=1.0; scale_penalty=true, include_cp_locations=true)

Combine the raw model-loss `loss` with a complexity penalty chosen by `objective_type`.

Built-in MICA criteria (`:bic`, `:mdl`, `:aic`) use

```julia
p_total = n_global + (num_cps + 1) * n_segment_specific + num_cps
```

Exact TCPD reference-package criteria are also available:
- `:tcpd_bic`, `:tcpd_mbic`, `:tcpd_aic`, `:tcpd_hannan_quinn`, `:tcpd_sic`, `:tcpd_none`
- `:tcpd_bic0`, `:tcpd_aic0`, `:tcpd_hannan_quinn0` (postfix-0 variants)
- `:wbs_bic`, `:wbs_mbic`, `:wbs_ssic`
- `:rfpop_l1`, `:rfpop_l2`, `:rfpop_huber`, `:rfpop_outlier`, `:rfpop_manual`

When `scale_penalty=true`, the complexity penalty is scaled by `sigma2_null`
(the residual variance of a constant-mean null model). RFPOP-style raw penalties
are never scaled.

Returns the penalized objective value.
"""
function compute_objective(loss::Real, n::Int, n_global::Int, n_segment_specific::Int,
                              num_cps::Int, objective_type::Symbol, penalty_fn::Function,
                              CP::Vector{Int}, segment_lengths::Vector{Int},
                              sigma2_null::Real=1.0; scale_penalty::Bool=true,
                              include_cp_locations::Bool=true)
    num_segments = num_cps + 1
    p_total = n_global + num_segments * n_segment_specific + (include_cp_locations ? num_cps : 0)

    if objective_type == :bic
        return compute_bic(loss, n, p_total; sigma2_null=sigma2_null, scale_penalty=scale_penalty)
    elseif objective_type == :mdl
        return compute_mdl(loss, n, p_total; sigma2_null=sigma2_null, scale_penalty=scale_penalty)
    elseif objective_type == :aic
        return compute_aic(loss, n, p_total; sigma2_null=sigma2_null, scale_penalty=scale_penalty)
    elseif objective_type in (:tcpd_bic, :tcpd_mbic, :tcpd_aic, :tcpd_hannan_quinn,
                              :tcpd_sic, :tcpd_none,
                              :tcpd_bic0, :tcpd_aic0, :tcpd_hannan_quinn0)
        # Exact changepoint / changepoint.np penalties from the TCPD benchmark.
        # diffparam = n_segment_specific; penalty scales with the parameters that
        # actually change at a CP, not with the full p_total count.
        type_map = Dict(
            :tcpd_bic => :bic,
            :tcpd_mbic => :mbic,
            :tcpd_aic => :aic,
            :tcpd_hannan_quinn => :hannan_quinn,
            :tcpd_sic => :sic,
            :tcpd_none => :none,
            :tcpd_bic0 => :bic0,
            :tcpd_aic0 => :aic0,
            :tcpd_hannan_quinn0 => :hannan_quinn0
        )
        return compute_tcpd_changepoint_penalty(loss, n, num_cps, n_segment_specific;
            sigma2_null=sigma2_null, scale_penalty=scale_penalty,
            type=type_map[objective_type])
    elseif objective_type in (:wbs_bic, :wbs_mbic, :wbs_ssic)
        # Exact WBS penalties from the TCPD benchmark.
        type_map = Dict(
            :wbs_bic => :bic,
            :wbs_mbic => :mbic,
            :wbs_ssic => :ssic
        )
        return compute_tcpd_wbs_penalty(loss, n, num_cps, segment_lengths;
            sigma2_null=sigma2_null, scale_penalty=scale_penalty,
            type=type_map[objective_type])
    elseif objective_type == :rfpop_l1
        return compute_tcpd_rfpop_penalty(loss, n, num_cps; lambda=log(n))
    elseif objective_type == :rfpop_l2
        return compute_tcpd_rfpop_penalty(loss, n, num_cps; lambda=log(n))
    elseif objective_type == :rfpop_huber
        return compute_tcpd_rfpop_penalty(loss, n, num_cps; lambda=1.4 * log(n))
    elseif objective_type == :rfpop_outlier
        return compute_tcpd_rfpop_penalty(loss, n, num_cps; lambda=2.0 * log(n))
    elseif objective_type == :rfpop_manual
        # Raw lambda is supplied by the caller through penalty_fn or a closure.
        # For now fall back to the generic penalty branch with no variance scaling.
        pen = call_penalty_fn(penalty_fn;
            p=n_segment_specific, n=n, CP=CP,
            segment_lengths=segment_lengths,
            num_segments=num_segments)
        return loss + pen
    else
        # :penalty (default, backward compatible)
        pen = call_penalty_fn(penalty_fn;
            p=p_total, n=n, CP=CP,
            segment_lengths=segment_lengths,
            num_segments=num_segments)
        return scale_penalty ? loss + sigma2_null * pen : loss + pen
    end
end

function evaluate_candidate(
    j::Int, a::Int, CP::Vector{Int},
    bounds::Tuple{Vector{Float64},Vector{Float64}},
    chromosome::Vector{Float64}, parnames, optimizer::AbstractOptimizerConfig,
    n_global::Int, n_segment_specific::Int, n::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64},
    penalty_fn::Function,
    data_indices::Union{Vector{Int},Nothing},
    animate::Bool,
    objective_type::Symbol = :penalty,
    sigma2_null::Real=1.0;
    scale_penalty::Bool=true,
    include_cp_locations::Bool=true
)
    new_cp = sort([CP; j])
    
    # Level continuity requires propagation across segments; analytical per-segment
    # fits cannot satisfy it, so fall back to the provided numerical optimizer.
    effective_optimizer = (optimizer isa AnalyticalOptimizer && needs_level_continuity(model_manager)) ?
        suggest_optimizer(get_model_type(model_manager), n_global + n_segment_specific) : optimizer
    
    if effective_optimizer isa AnalyticalOptimizer
        # Exact least-squares fitting — no iterative optimization
        total_loss = 0.0
        best_params = Float64[]
        num_segments = length(new_cp) + 1
        for i in 1:num_segments
            idx_start = (i == 1) ? 1 : new_cp[i - 1] + 1
            idx_end   = (i > length(new_cp)) ? size(data, 2) : new_cp[i]
            seg_data = data[:, idx_start:idx_end]
            loss, seg_params = fit_segment_analytical(model_manager.base_model, seg_data)
            total_loss += loss
            append!(best_params, seg_params)
        end
        loss = total_loss
        best = best_params
    else
        loss, best = optimize_with_changepoints(
            objective_function, chromosome, parnames, new_cp, bounds, effective_optimizer,
            n_global, n_segment_specific,
            model_manager, loss_function, data
        )
    end
    
    if animate
        sim, plt = simulate_full_model(best, new_cp, parnames,
            n_global, n_segment_specific,
            model_manager, data;
            plot_results=true,
            show_change_points=true,
            show_data=true,
            data_indices=data_indices)
        frame(get_animation(), plt)
    end
    num_cps = length(new_cp)
    segment_lengths = diff([0; new_cp; n])
    obj = compute_objective(loss, n, n_global, n_segment_specific, num_cps,
                            objective_type, penalty_fn, new_cp, segment_lengths,
                            sigma2_null; scale_penalty=scale_penalty,
                            include_cp_locations=include_cp_locations)
    return (obj, best)
end

function evaluate_segment(
    objective_function, a::Int, b::Int, CP::Vector{Int}, bounds,
    chromosome::Vector{Float64}, parnames, optimizer::AbstractOptimizerConfig,
    min_length::Int, step::Int,
    n_global::Int, n_segment_specific::Int, n::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64},
    penalty_fn::Function,
    data_indices::Union{Vector{Int},Nothing},
    objective_type::Symbol = :penalty,
    sigma2_null::Real=1.0;
    parallel::Bool=false,
    verbose::Bool=false,
    animate::Bool=false,
    scale_penalty::Bool=true,
    include_cp_locations::Bool=true
)
    candidates = collect((a + min_length):step:(b - min_length))
    if isempty(candidates)
        return Float64[], Vector{Vector{Float64}}()
    end
    if verbose
        println("  Evaluating segment [$a, $b] with $(length(candidates)) candidates")
    end
    if parallel && length(candidates) > 1 && nworkers() > 0
        tasks = [
            (j, a, CP, bounds, chromosome, parnames, optimizer,
             n_global, n_segment_specific, n, model_manager,
             loss_function, data, penalty_fn, data_indices, animate, objective_type, sigma2_null)
            for j in candidates
        ]
        results = pmap(t -> evaluate_candidate(t...; scale_penalty=scale_penalty, include_cp_locations=include_cp_locations), tasks)
        x = [r[1] for r in results]
        y = [r[2] for r in results]
    else
        x = Float64[]
        y = Vector{Vector{Float64}}()
        for j in candidates
            result = evaluate_candidate(
                j, a, CP, bounds, chromosome, parnames, optimizer,
                n_global, n_segment_specific, n, model_manager,
                loss_function, data, penalty_fn, data_indices, animate, objective_type,
                sigma2_null; scale_penalty=scale_penalty,
                include_cp_locations=include_cp_locations
            )
            push!(x, result[1])
            push!(y, result[2])
        end
    end
    return x, y
end

# ----------------------------------------------------------------------
# Null variance helper
# ----------------------------------------------------------------------
"""
    compute_null_variance(data::Matrix{Float64})

Compute the residual variance of a constant-mean model on the whole dataset.
This is used to scale the penalty so that it is comparable to the RSS loss
regardless of the scale of the data.
"""
function compute_null_variance(data::Matrix{Float64})
    μ = mean(data)
    rss_null = sum((data .- μ).^2)
    return max(rss_null / length(data), 1e-12)
end

# ----------------------------------------------------------------------
# Main Function: detect_changepoints
# ----------------------------------------------------------------------
"""
    detect_changepoints(
        objective_function, n, n_global, n_segment_specific, parnames,
        model_manager, loss_function, data,
        initial_chromosome, bounds, ga, min_length, step;
        penalty_fn=default_penalty, objective_type=:penalty,
        scale_penalty=true, ...
    )

Detect optimal change points using a greedy search strategy and regularized loss.

When `scale_penalty=true` (default), the penalty is scaled by the null-model
variance `sigma2_null` so that the objective is comparable across datasets
with different scales.

Returns:
- Vector of change point indices (CP)
- Best parameter vector found
"""
function detect_changepoints(
    objective_function,
    n::Int, n_global::Int, n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64}, 
    initial_chromosome::Vector{Float64},
    parnames,
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    optimizer::AbstractOptimizerConfig,
    min_length::Int, step::Int;
    penalty_fn::Function = default_penalty,
    objective_type::Symbol = :penalty,
    scale_penalty::Bool = true,
    include_cp_locations::Bool = true,
    data_indices::Union{Vector{Int},Nothing} = nothing,
    parallel::Bool = false,
    verbose::Bool = false,
    animate::Bool = false
)
    tau = [(0, n)]
    CP = Int[]
    verbose && println("[detect_changepoints] n=$n, min_length=$min_length, step=$step, objective=$objective_type, scale_penalty=$scale_penalty, include_cp_locations=$include_cp_locations")

    # Null-model variance for scaling the penalty to the data scale.
    sigma2_null = compute_null_variance(data)
    verbose && println("  Null variance sigma2_null=$sigma2_null")

    # Level continuity requires propagation across segments; analytical per-segment
    # fits cannot satisfy it, so fall back to a numerical optimizer when needed.
    effective_optimizer = (optimizer isa AnalyticalOptimizer && needs_level_continuity(model_manager)) ?
        suggest_optimizer(get_model_type(model_manager), n_global + n_segment_specific) : optimizer

    if effective_optimizer isa AnalyticalOptimizer
        loss_val, best_params = fit_segment_analytical(model_manager.base_model, data)
    else
        loss_val, best_params = optimize_with_changepoints(
            objective_function, initial_chromosome, parnames, CP, bounds, effective_optimizer,
            n_global, n_segment_specific,
            model_manager, loss_function, data
        )
    end
    
    # Convert initial loss to the chosen objective type
    segment_lengths = [n]
    loss_val = compute_objective(loss_val, n, n_global, n_segment_specific, 0,
                                 objective_type, penalty_fn, CP, segment_lengths,
                                 sigma2_null; scale_penalty=scale_penalty,
                                 include_cp_locations=include_cp_locations)

    verbose && println("  Initial objective (0 CPs): $loss_val")
    
    if animate
        sim, plt = simulate_full_model(best_params, CP, parnames,
            n_global, n_segment_specific,
            model_manager, data;
            plot_results=true,
            show_change_points=true,
            show_data=true,
            data_indices=data_indices)
        mkpath("plots")
        frame(get_animation(), plt)
    end

    update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, extract_parameters)

    while !isempty(tau)
        a, b = pop!(tau)
        x, y = evaluate_segment(
            objective_function, a, b, CP, bounds, initial_chromosome, parnames, optimizer, min_length, step,
            n_global, n_segment_specific, n,
            model_manager, loss_function, data,
            penalty_fn, data_indices, objective_type, sigma2_null;
            parallel=parallel, verbose=verbose, animate=animate, scale_penalty=scale_penalty,
            include_cp_locations=include_cp_locations
        )
        if !isempty(x)
            minval, idx = findmin(x)
            if minval < loss_val
                chpt = a + min_length + (idx - 1) * step
                push!(CP, chpt)
                CP = sort(CP)
                loss_val = minval
                best_params = y[idx]
                verbose && println("  Accepted CP at $chpt (obj=$loss_val)")

                if animate
                    sim, plt = simulate_full_model(best_params, CP, parnames,
                        n_global, n_segment_specific,
                        model_manager, data;
                        plot_results=true,
                        show_change_points=true,
                        show_data=true,
                        data_indices=data_indices)
                    frame(get_animation(), plt)
                end

                update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, extract_parameters)
                if chpt != a + min_length
                    push!(tau, (a, chpt))
                end
                if chpt != b - min_length
                    push!(tau, (chpt, b))
                end
            end
        end
    end

    verbose && println("[detect_changepoints] Done. $(length(CP)) CPs: $CP")
    return CP, best_params
end

# end
