"""
    EvolutionaryPLEConfig

Genetic-algorithm configuration for the inner PLE optimisation.

Fields are string-encoded so that the exact GA used for the MICA COVID-19
change-point analysis can be reproduced:

```julia
EvolutionaryPLEConfig(
    population_size=150,
    iterations=1000,
    selection="tournament(2)",
    crossover="SBX(0.7, 1)",
    mutation="gaussian(0.0001)",
    mutation_rate=0.7,
    crossover_rate=0.7,
    parallelization=:thread
)
```
"""
struct EvolutionaryPLEConfig <: AbstractPLEOptimizer
    population_size::Int
    iterations::Int
    seed::Union{Int,Nothing}
    selection::String
    crossover::String
    mutation::String
    mutation_rate::Float64
    crossover_rate::Float64
    parallelization::Symbol  # :serial or :thread
end

function EvolutionaryPLEConfig(;
    population_size::Int=80,
    iterations::Int=100,
    seed::Union{Int,Nothing}=nothing,
    selection::String="uniformranking(10)",
    crossover::String="MILX(0.01, 0.17, 0.5)",
    mutation::String="gaussian(0.0001)",
    mutation_rate::Float64=0.3,
    crossover_rate::Float64=0.6,
    parallelization::Symbol=:serial
)
    return EvolutionaryPLEConfig(
        population_size, iterations, seed, selection, crossover, mutation,
        mutation_rate, crossover_rate, parallelization
    )
end

"""
    OptimPLEConfig

Gradient-based optimisation configuration for the inner PLE optimisation.

Default is `Fminbox(LBFGS())`.  This can be fast on smooth models but may fail
on stiff epidemiological models."""
struct OptimPLEConfig{M,O} <: AbstractPLEOptimizer
    method::M
    options::O
end

function OptimPLEConfig(;
    method=Fminbox(LBFGS()),
    options=Optim.Options(show_trace=false, iterations=200)
)
    return OptimPLEConfig(method, options)
end

"""
    NLoptPLEConfig

Direct-search configuration via NLopt for the inner PLE optimisation.

Default is `LN_BOBYQA`.  NLopt is useful if both GA and gradient methods fail."""
struct NLoptPLEConfig <: AbstractPLEOptimizer
    algorithm::Symbol
    xtol_rel::Float64
    maxeval::Int
end

function NLoptPLEConfig(;
    algorithm::Symbol=:LN_BOBYQA,
    xtol_rel::Float64=1.0e-6,
    maxeval::Int=10_000
)
    return NLoptPLEConfig(algorithm, xtol_rel, maxeval)
end

"""    default_evolutionary_config

Convenience constructor for the default GA-based PLE optimiser."""
default_evolutionary_config(; kwargs...) = EvolutionaryPLEConfig(; kwargs...)

"""
    MetaheuristicsDEConfig

Differential-Evolution (DE) configuration using Metaheuristics.jl for the inner
PLE optimisation.

DE is generally more robust than a GA on high-dimensional, rugged landscapes,
which is why it is preferred for the COVID-19 MICA PLE problem.

Fields:
- `population`: population size N (default 150)
- `F`: differential mutation factor (default 0.9)
- `CR`: crossover probability (default 0.9)
- `max_evals`: maximum function evaluations (default 150_000, ~1000 DE generations with N=150)
- `parallel`: run population evaluations in parallel with `Threads.@threads` (default `true`)
"""
struct MetaheuristicsDEConfig <: AbstractPLEOptimizer
    population::Int
    F::Float64
    CR::Float64
    max_evals::Int
    parallel::Bool
    warm_start_perturbation::Float64  # relative perturbation around x0 for warm-start population
end

function MetaheuristicsDEConfig(;
    population::Int=150,
    F::Float64=0.9,
    CR::Float64=0.9,
    max_evals::Int=150_000,
    parallel::Bool=true,
    warm_start_perturbation::Float64=0.5
)
    return MetaheuristicsDEConfig(population, F, CR, max_evals, parallel, warm_start_perturbation)
end

"""    default_de_config

Convenience constructor for the default DE-based PLE optimiser."""
default_de_config(; kwargs...) = MetaheuristicsDEConfig(; kwargs...)

"""    default_optim_config

Convenience constructor for the default LBFGS-based PLE optimiser."""
default_optim_config(; kwargs...) = OptimPLEConfig(; kwargs...)

"""
    optimize_ple(f, x0, lb, ub, optimizer::AbstractPLEOptimizer)

Minimise the scalar function `f` inside the box `[lb, ub]` using the selected
backend.  Returns `(minimum, minimizer)`."""
function optimize_ple(f, x0, lb, ub, optimizer::EvolutionaryPLEConfig)
    if optimizer.seed !== nothing
        Random.seed!(optimizer.seed)
    end
    bounds = Evolutionary.BoxConstraints(lb, ub)

    # Parse selection operator
    sel_str = strip(optimizer.selection)
    sel_op = if startswith(sel_str, "uniformranking(")
        k = parse(Int, match(r"uniformranking\((\d+)\)", sel_str).captures[1])
        Evolutionary.uniformranking(k)
    elseif startswith(sel_str, "tournament(")
        k = parse(Int, match(r"tournament\((\d+)\)", sel_str).captures[1])
        Evolutionary.tournament(k)
    elseif sel_str == "sus"
        Evolutionary.sus
    elseif sel_str == "roulette"
        Evolutionary.roulette
    else
        error("Unknown selection method: $sel_str. Use uniformranking(k), tournament(k), sus, or roulette.")
    end

    # Parse crossover operator
    cx_str = strip(optimizer.crossover)
    cx_op = if startswith(cx_str, "MILX(")
        m = match(r"MILX\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\)", cx_str)
        isnothing(m) && error("Invalid MILX crossover: $cx_str")
        Evolutionary.MILX(parse(Float64, m.captures[1]), parse(Float64, m.captures[2]), parse(Float64, m.captures[3]))
    elseif startswith(cx_str, "SBX(")
        m = match(r"SBX\(([\d\.]+),\s*(\d+)\)", cx_str)
        isnothing(m) && error("Invalid SBX crossover: $cx_str. Use SBX(eta, n) where n is an integer, e.g. SBX(0.7, 1).")
        Evolutionary.SBX(parse(Float64, m.captures[1]), parse(Int, m.captures[2]))
    else
        error("Unknown crossover operator: $cx_str. Use MILX(a,b,c) or SBX(eta,n).")
    end

    # Parse mutation operator
    mut_str = strip(optimizer.mutation)
    mut_op = if startswith(mut_str, "gaussian(")
        m = match(r"gaussian\(([\d\.]+)\)", mut_str)
        isnothing(m) && error("Invalid gaussian mutation: $mut_str")
        Evolutionary.gaussian(parse(Float64, m.captures[1]))
    else
        error("Unknown mutation operator: $mut_str. Use gaussian(sigma).")
    end

    ga = Evolutionary.GA(
        populationSize=optimizer.population_size,
        selection=sel_op,
        crossover=cx_op,
        mutationRate=optimizer.mutation_rate,
        crossoverRate=optimizer.crossover_rate,
        mutation=mut_op
    )
    result = Evolutionary.optimize(
        f, bounds, x0, ga,
        Evolutionary.Options(show_trace=false, iterations=optimizer.iterations,
                               parallelization=optimizer.parallelization)
    )
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end

function optimize_ple(f, x0, lb, ub, optimizer::MetaheuristicsDEConfig)
    # Metaheuristics.jl expects bounds as an n×2 matrix [lb ub].
    # When parallel_evaluation=true, Metaheuristics batches evaluations by passing a
    # matrix where each row is a candidate solution and expects a vector of fitness
    # values in return. We wrap the scalar objective in a threaded batch evaluator.
    bounds = hcat(lb, ub)
    options = Metaheuristics.Options(f_calls_limit=optimizer.max_evals, parallel_evaluation=optimizer.parallel)

    # Warm-start DE with x0: one individual is the provided initial point, the rest
    # are small random perturbations inside the bounds. This is critical for the MICA
    # PLE problem because the loss landscape is rugged and wide uniform random
    # sampling misses the good basin around the MICA fit.
    initial_state = make_de_initial_state(f, x0, lb, ub, optimizer.population;
                                          perturbation=optimizer.warm_start_perturbation)

    de = Metaheuristics.DE(
        N=optimizer.population,
        F=optimizer.F,
        CR=optimizer.CR,
        options=options,
        initial_state=initial_state
    )
    f_batch = optimizer.parallel ? make_batched_objective(f) : f
    result = Metaheuristics.optimize(f_batch, bounds, de)
    return Metaheuristics.minimum(result), Metaheuristics.minimizer(result)
end

"""    make_de_initial_state(f, x0, lb, ub, N)

Create a Metaheuristics.State for DE warm-started at `x0`. The first individual is
`x0` itself; the remaining `N-1` individuals are random perturbations around `x0`
that stay within `[lb, ub]`."""
function make_de_initial_state(f, x0, lb, ub, N; perturbation=0.5)
    best_sol = Metaheuristics.create_child(copy(x0), f(x0))
    population = [best_sol]
    n = length(x0)
    range = ub .- lb
    for i in 2:N
        x_pert = copy(x0)
        for j in 1:n
            δ = perturbation * range[j] * (2.0 * rand() - 1.0)
            x_pert[j] = clamp(x_pert[j] + δ, lb[j], ub[j])
        end
        push!(population, Metaheuristics.create_child(x_pert, f(x_pert)))
    end
    return Metaheuristics.State(best_sol, population; f_calls=N, iteration=1)
end

"""    make_batched_objective(f)

Wrap a scalar objective `f(x::AbstractVector)` so that it can also evaluate a batch
of candidates `X::AbstractMatrix` where each row is a solution. The batched
evaluation is threaded."""
function make_batched_objective(f)
    function f_batch(x)
        if ndims(x) == 1
            return f(x)
        end
        n = size(x, 1)
        results = Vector{Float64}(undef, n)
        Threads.@threads for i in 1:n
            results[i] = f(x[i, :])
        end
        return results
    end
    return f_batch
end

"""    multi_start_reference_search(f, x0, lb, ub, optimizer, n_starts; perturbation=0.5)

Run the PLE optimizer `n_starts` times from perturbed copies of `x0` and return the
best (loss, minimizer) found.  The first start uses `x0` itself; subsequent starts are
`x0` plus independent uniform perturbations of relative size `perturbation` inside the
box `[lb, ub]`.

This is useful as a pre-processing step for ODE+PLE problems where a single local
re-optimization from the MICA fit can miss better basins.  The best reference found is
then used as the MICA-loss baseline and starting point for the actual PLE."""
function multi_start_reference_search(f, x0, lb, ub, optimizer::AbstractPLEOptimizer, n_starts::Int; perturbation::Float64=0.5)
    n_starts >= 1 || error("n_starts must be >= 1")
    n = length(x0)
    range = ub .- lb

    best_loss = Inf
    best_x = copy(x0)

    for s in 1:n_starts
        x_init = if s == 1
            copy(x0)
        else
            x_pert = copy(x0)
            for j in 1:n
                δ = perturbation * range[j] * (2.0 * rand() - 1.0)
                x_pert[j] = clamp(x_pert[j] + δ, lb[j], ub[j])
            end
            x_pert
        end

        try
            loss, x_opt = optimize_ple(f, x_init, lb, ub, optimizer)
            if isfinite(loss) && loss < best_loss
                best_loss = loss
                best_x = copy(x_opt)
            end
            @info "Multi-start $s/$n_starts: loss=$(isfinite(loss) ? round(loss, digits=4) : "Inf") (best=$(round(best_loss, digits=4)))"
        catch e
            @warn "Multi-start $s/$n_starts failed: $e"
        end
    end

    if !isfinite(best_loss)
        @warn "All multi-start attempts failed or returned Inf; falling back to the provided warm-start point x0."
        best_loss = try
            f(x0)
        catch e
            @warn "Failed to evaluate f(x0): $e"
            Inf
        end
        best_x = copy(x0)
    end

    return best_loss, best_x
end

function optimize_ple(f, x0, lb, ub, optimizer::OptimPLEConfig)
    result = Optim.optimize(f, lb, ub, x0, optimizer.method, optimizer.options)
    return Optim.minimum(result), Optim.minimizer(result)
end

function optimize_ple(f, x0, lb, ub, optimizer::NLoptPLEConfig)
    opt = NLopt.Opt(optimizer.algorithm, length(x0))
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.xtol_rel = optimizer.xtol_rel
    opt.maxeval = optimizer.maxeval
    opt.min_objective = (x, grad) -> f(x)
    minf, minx, ret = NLopt.optimize(opt, x0)
    return minf, minx
end

"""
    MultiStartBOBYQAConfig

Multi-start BOBYQA (Bound Optimization BY Quadratic Approximation) for the
inner PLE optimisation.

Runs BOBYQA from `n_starts` initial points: the warm-start `x0` plus
`n_starts - 1` randomly perturbed copies.  Returns the best result across all
starts.  This mitigates the risk of BOBYQA getting trapped in a local minimum.

BOBYQA is derivative-free and builds a quadratic interpolation model, making
it well suited to stiff ODE objectives where gradients are unreliable or the
loss landscape has discontinuities (solver failures → Inf).

Recommended settings for PLE of 30-70 dimensional ODE models:
  - `n_starts = 5`
  - `maxeval = 2000`
  - `xtol_rel = 1e-6`
  - `perturbation = 0.1` (±10% of bound range)
"""
struct MultiStartBOBYQAConfig <: AbstractPLEOptimizer
    n_starts::Int
    maxeval::Int
    xtol_rel::Float64
    perturbation::Float64
end

function MultiStartBOBYQAConfig(;
    n_starts::Int=5,
    maxeval::Int=2000,
    xtol_rel::Float64=1.0e-6,
    perturbation::Float64=0.1
)
    return MultiStartBOBYQAConfig(n_starts, maxeval, xtol_rel, perturbation)
end

"""    default_multistart_bobyqa_config

Convenience constructor for the default multi-start BOBYQA PLE optimiser."""
default_multistart_bobyqa_config(; kwargs...) = MultiStartBOBYQAConfig(; kwargs...)

function optimize_ple(f, x0, lb, ub, optimizer::MultiStartBOBYQAConfig)
    n = length(x0)
    best_loss = Inf
    best_x = copy(x0)

    # Generate starting points: x0 + (n_starts-1) perturbed copies
    starts = Vector{Vector{Float64}}(undef, optimizer.n_starts)
    starts[1] = copy(x0)
    for s in 2:optimizer.n_starts
        x_pert = copy(x0)
        for j in 1:n
            range_j = ub[j] - lb[j]
            δ = optimizer.perturbation * range_j * (2.0 * rand() - 1.0)
            x_pert[j] = clamp(x_pert[j] + δ, lb[j], ub[j])
        end
        starts[s] = x_pert
    end

    for x_init in starts
        try
            opt = NLopt.Opt(:LN_BOBYQA, n)
            opt.lower_bounds = lb
            opt.upper_bounds = ub
            opt.xtol_rel = optimizer.xtol_rel
            opt.maxeval = optimizer.maxeval
            opt.min_objective = (x, grad) -> f(x)
            minf, minx, ret = NLopt.optimize(opt, x_init)
            if isfinite(minf) && minf < best_loss
                best_loss = minf
                best_x = copy(minx)
            end
        catch e
            # Silently skip failed starts; at least one should succeed
            continue
        end
    end

    return best_loss, best_x
end

"""
    HybridOptimizerConfig

Two-stage optimizer for rugged ODE+PLE landscapes.

Stage 1: run a global/metaheuristic optimizer (`global_optimizer`) to locate a good
basin.  Stage 2: polish the best point with a fast local optimizer
(`local_optimizer`, e.g. `OptimPLEConfig(Fminbox(LBFGS()))` or `NLoptPLEConfig(:LN_BOBYQA)`).

This is the recommended optimizer for the COVID-19 conditional PLE because the
epidemiological loss is multimodal (global stage) but locally smooth enough for
gradient/direct-search polishing.
"""
struct HybridOptimizerConfig{G,L} <: AbstractPLEOptimizer
    global_optimizer::G
    local_optimizer::L
end

function HybridOptimizerConfig(;
    global_optimizer::AbstractPLEOptimizer=MetaheuristicsDEConfig(),
    local_optimizer::AbstractPLEOptimizer=OptimPLEConfig(Fminbox(LBFGS()),
                                                           Optim.Options(show_trace=false,
                                                                         iterations=200))
)
    return HybridOptimizerConfig(global_optimizer, local_optimizer)
end

function optimize_ple(f, x0, lb, ub, optimizer::HybridOptimizerConfig)
    # Stage 1: global/metaheuristic search
    loss_g, x_g = optimize_ple(f, x0, lb, ub, optimizer.global_optimizer)

    # Stage 2: local polish from the best global point
    x_start = isfinite(loss_g) ? x_g : copy(x0)
    # Clamp strictly inside bounds to avoid Fminbox ArgumentError if GA drifted
    x_start = clamp.(x_start, lb .+ 1e-10, ub .- 1e-10)
    loss_l, x_l = optimize_ple(f, x_start, lb, ub, optimizer.local_optimizer)

    # Return the better of the two
    if isfinite(loss_l) && loss_l < loss_g
        return loss_l, x_l
    else
        return loss_g, x_g
    end
end

"""
    uses_internal_threads(optimizer)

Return `true` if `optimizer` internally parallelises its fitness evaluations
with `Threads.@threads` (e.g. `MetaheuristicsDEConfig(parallel=true)` or
`EvolutionaryPLEConfig(parallelization=:thread)`).  This is used to avoid
nested `Threads.@threads` loops when the outer loop is already threaded.
"""
uses_internal_threads(opt::MetaheuristicsDEConfig) = opt.parallel
uses_internal_threads(opt::EvolutionaryPLEConfig) = opt.parallelization == :thread
uses_internal_threads(opt::HybridOptimizerConfig) = uses_internal_threads(opt.global_optimizer)
uses_internal_threads(opt::AbstractPLEOptimizer) = false

"""
    solve_problem(prob::ODEChangepointPLEProblem, optimizer)

Re-optimise the parameters of `prob` for the current changepoint configuration.
Returns `(loss, params)`."""
function solve_problem(prob::ODEChangepointPLEProblem, optimizer::AbstractPLEOptimizer)
    n_segments = length(prob.changepoints) + 1
    n_params = prob.n_global + prob.n_segment_specific * n_segments
    lb = prob.lb[1:n_params]
    ub = prob.ub[1:n_params]
    x0 = prob.best_params[1:n_params]
    f = x -> evaluate_loss(prob, x)
    return optimize_ple(f, x0, lb, ub, optimizer)
end
