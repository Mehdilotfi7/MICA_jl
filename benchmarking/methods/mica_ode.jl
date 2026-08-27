# ============================================================
# 03_mica_wrapper.jl
# Standalone MICA runner — reimplements core MICA without plotting
# ============================================================
using DifferentialEquations
using Evolutionary
using LabelledArrays
using Random
using Statistics

# SIR model for MICA (uses LabelledArrays / named parameter access)
function sirmodel_mica!(du, u, p, t)
    S, I, R = u
    β, γ = p.β, p.γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0::Vector{Float64})
    prob = ODEProblem(sirmodel_mica!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
    return sol[:, :]
end

function loss_function_mica(observed, simulated)
    simulated = simulated[2:2, :]
    return sqrt(sum((observed .- simulated).^2))
end

# ============================================================
# Standalone MICA core (no plotting, no animations)
# ============================================================

function extract_parameters(chromosome::Vector{T}, n_global::Int, n_segment_specific::Int) where T
    global_parameters = chromosome[1:n_global]
    segment_parameters = [chromosome[i:i+n_segment_specific-1] for i in n_global+1:n_segment_specific:length(chromosome)]
    return global_parameters, segment_parameters
end

function objective_function_mica(
    chromosome, 
    change_points, 
    n_global::Int, 
    n_segment_specific::Int, 
    data::Matrix{Float64}
)
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    total_loss = 0.0
    num_segments = length(change_points) + 1
    u0 = [9999.0, 1.0, 0.0]
    
    for i in 1:num_segments
        idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
        idx_end   = (i > length(change_points)) ? size(data, 2) : change_points[i]
        segment_data = data[:, idx_start:idx_end]
        
        seg_pars = segment_pars_list[i]
        all_pars = @LArray [constant_pars; seg_pars] (:γ, :β)
        
        tspan = (0.0, Float64(idx_end - idx_start))
        prob = ODEProblem(sirmodel_mica!, u0, tspan, all_pars)
        sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
        sim_data = sol[:, :]
        
        total_loss += loss_function_mica(segment_data, sim_data)
        
        if length(sol.u) > 0
            u0 = sol.u[end]
        end
    end
    
    return total_loss
end

function optimize_with_changepoints_mica(
    chromosome, change_points, bounds, n_global, n_segment_specific, data; 
    populationSize=150, seed=1234
)
    wrapped_obj(chrom) = objective_function_mica(chrom, change_points, n_global, n_segment_specific, data)
    
    ga = GA(
        populationSize=populationSize,
        selection=uniformranking(20),
        crossover=MILX(0.01, 0.17, 0.5),
        mutationRate=0.3,
        crossoverRate=0.6,
        mutation=gaussian(0.0001)
    )
    
    Random.seed!(seed)
    result = Evolutionary.optimize(wrapped_obj, BoxConstraints(bounds...), chromosome, ga, 
                                   Evolutionary.Options(show_trace=false))
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end

function detect_changepoints_mica(
    data::Matrix{Float64},
    initial_chromosome::Vector{Float64},
    bounds::Tuple{Vector{Float64}, Vector{Float64}};
    n_global::Int=1,
    n_segment_specific::Int=1,
    min_length::Int=10,
    step::Int=10,
    penalty_fn::Function=(p, n) -> 10.0 * p * log(n),
    populationSize::Int=150,
    seed::Int=1234
)
    n = size(data, 2)
    tau = [(0, n)]
    CP = Int[]
    
    # Initial optimization: no change points
    loss_val, best_params = optimize_with_changepoints_mica(
        initial_chromosome, CP, bounds, n_global, n_segment_specific, data;
        populationSize=populationSize, seed=seed
    )
    # Apply penalty to initial model (0 CPs → 1 segment)
    loss_val += penalty_fn(n_global + n_segment_specific, n)
    
    # Extend chromosome for first segment
    # FIX: use original initial guess for new segment params, not fitted values
    chromosome = copy(best_params)
    append!(chromosome, initial_chromosome[n_global+1:end])
    append!(bounds[1], bounds[1][n_global+1:end])
    append!(bounds[2], bounds[2][n_global+1:end])
    
    while !isempty(tau)
        a, b = pop!(tau)
        
        x = Float64[]
        y = Vector{Vector{Float64}}()
        
        for j in (a + min_length):step:(b - min_length)
            new_cp = sort([CP; j])
            
            # Build chromosome with enough segment parameters
            test_chrom = copy(chromosome)
            test_bounds = (copy(bounds[1]), copy(bounds[2]))
            target_len = n_global + (length(new_cp)+1) * n_segment_specific
            while length(test_chrom) < target_len
                # Append first segment's params as initial guess for new segment
                append!(test_chrom, chromosome[n_global+1:n_global+n_segment_specific])
                append!(test_bounds[1], bounds[1][n_global+1:n_global+n_segment_specific])
                append!(test_bounds[2], bounds[2][n_global+1:n_global+n_segment_specific])
            end
            test_chrom = test_chrom[1:target_len]
            test_bounds = (
                test_bounds[1][1:target_len],
                test_bounds[2][1:target_len]
            )
            
            loss, best = optimize_with_changepoints_mica(
                test_chrom, new_cp, test_bounds, n_global, n_segment_specific, data;
                populationSize=populationSize, seed=seed
            )
            
            num_segments = length(new_cp) + 1
            total_p = n_global + num_segments * n_segment_specific
            pen = penalty_fn(total_p, n)
            
            push!(x, loss + pen)
            push!(y, best)
        end
        
        if !isempty(x)
            minval, idx = findmin(x)
            if minval < loss_val
                # FIX: correct mapping from array index to candidate position
                j_values = collect((a + min_length):step:(b - min_length))
                chpt = j_values[idx]
                push!(CP, chpt)
                CP = sort(CP)
                loss_val = minval
                best_params = y[idx]
                
                # Extend chromosome for next segment
                # FIX: use original initial guess for new segment params
                append!(chromosome, initial_chromosome[n_global+1:end])
                append!(bounds[1], bounds[1][n_global+1:n_global+n_segment_specific])
                append!(bounds[2], bounds[2][n_global+1:n_global+n_segment_specific])
                
                if chpt != a + min_length
                    push!(tau, (a, chpt))
                end
                if chpt != b - min_length
                    push!(tau, (chpt, b))
                end
            end
        end
    end
    
    return CP, best_params
end

# ============================================================
# Public API
# ============================================================
function run_mica(data_I::Vector{Float64}, γ_true::Float64;
    populationSize=100,
    min_length=10,
    step=15,
    penalty_fn=nothing,
    seed=1234,
    num_runs=1
)
    n = length(data_I)
    data_CP = reshape(Float64.(data_I), 1, :)
    
    initial_chromosome = [γ_true, 0.0002]
    bounds = ([0.1, 0.0], [0.9, 0.1])
    
    if penalty_fn === nothing
        penalty_fn = (p, n) -> 10.0 * p * log(n)
    end
    
    best_cp = Int[]
    best_params = nothing
    best_score = Inf
    all_results = []
    
    for r in 1:num_runs
        run_seed = seed + (r - 1) * 1000
        try
            detected_cp, params = detect_changepoints_mica(
                data_CP, copy(initial_chromosome), (copy(bounds[1]), copy(bounds[2]));
                n_global=1, n_segment_specific=1,
                min_length=min_length, step=step,
                penalty_fn=penalty_fn,
                populationSize=populationSize, seed=run_seed
            )
            
            # Compute penalized score
            score = objective_function_mica(params, detected_cp, 1, 1, data_CP)
            num_seg = length(detected_cp) + 1
            score += penalty_fn(1, n) * num_seg
            
            push!(all_results, (detected_cp, params, score))
            
            if score < best_score
                best_score = score
                best_cp = detected_cp
                best_params = params
            end
        catch e
            @warn "MICA run $r failed with seed $run_seed: $e"
        end
    end
    
    return best_cp, best_params, all_results
end

function run_mica_bic(data_I::Vector{Float64}, γ_true::Float64, kappa::Float64; kwargs...)
    pen_fn = (p, n) -> kappa * p * log(n)
    return run_mica(data_I, γ_true; penalty_fn=pen_fn, kwargs...)
end
