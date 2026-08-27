# ============================================================
# 18_mica_unified.jl
# Unified MICA implementation using GA for ALL models (SIR, LR, AR)
# Same greedy recursive CP detection + GA parameter optimization
# Parameter structure: 1 global + 1 segment-specific per segment
# ============================================================
using Evolutionary
using Random
using Statistics
using DifferentialEquations
using LabelledArrays

# ============================================================
# Model-specific objective functions
# ============================================================

# --- SIR objective (original) ---
function sirmodel_mica!(du, u, p, t)
    S, I, R = u
    β, γ = p.β, p.γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

function objective_sir(chromosome, change_points, n_global::Int, n_segment_specific::Int, data::Matrix{Float64})
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
        
        # Loss: RMSE on I(t) only
        sim_I = sim_data[2:2, :]
        total_loss += sqrt(sum((segment_data .- sim_I).^2))
        
        if length(sol.u) > 0
            u0 = sol.u[end]
        end
    end
    
    return total_loss
end

# --- LR objective: GA optimizes β_global and β_seg per segment ---
function objective_lr(chromosome, change_points, n_global::Int, n_segment_specific::Int, data::Matrix{Float64})
    # data is 2×n: [x; y]
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    total_rss = 0.0
    num_segments = length(change_points) + 1
    n = size(data, 2)
    
    β_global = constant_pars[1]
    
    for i in 1:num_segments
        idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
        idx_end   = (i > length(change_points)) ? n : change_points[i]
        
        x_seg = data[1, idx_start:idx_end]
        y_seg = data[2, idx_start:idx_end]
        
        β_seg = segment_pars_list[i][1]
        y_pred = β_seg .* x_seg .+ β_global
        total_rss += sum((y_seg .- y_pred).^2)
    end
    
    return total_rss
end

# --- AR objective: GA optimizes σ and φ per segment ---
function objective_ar(chromosome, change_points, n_global::Int, n_segment_specific::Int, data::Matrix{Float64})
    # data is 1×n: [x]
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    total_rss = 0.0
    num_segments = length(change_points) + 1
    n = size(data, 2)
    
    # σ is global but not used in RSS (only counts for penalty)
    # φ is segment-specific
    
    for i in 1:num_segments
        idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
        idx_end   = (i > length(change_points)) ? n : change_points[i]
        
        seg = data[1, idx_start:idx_end]
        
        if length(seg) < 2
            total_rss += sum(seg.^2)
            continue
        end
        
        φ = segment_pars_list[i][1]
        φ = clamp(φ, -0.99, 0.99)
        
        y = seg[2:end]
        x_lag = seg[1:end-1]
        residuals = y .- φ .* x_lag
        total_rss += sum(residuals.^2)
    end
    
    return total_rss
end

# ============================================================
# Helper: extract parameters from chromosome
# ============================================================
function extract_parameters(chromosome::Vector{T}, n_global::Int, n_segment_specific::Int) where T
    global_parameters = chromosome[1:n_global]
    segment_parameters = [chromosome[i:i+n_segment_specific-1] for i in n_global+1:n_segment_specific:length(chromosome)]
    return global_parameters, segment_parameters
end

# ============================================================
# GA optimization wrapper (model-agnostic)
# ============================================================
function optimize_with_changepoints(
    chromosome, change_points, bounds, n_global, n_segment_specific, data, objective_fn;
    populationSize=100, seed=1234, mutation_std=0.0001
)
    wrapped_obj(chrom) = objective_fn(chrom, change_points, n_global, n_segment_specific, data)
    
    ga = GA(
        populationSize=populationSize,
        selection=uniformranking(20),
        crossover=MILX(0.01, 0.17, 0.5),
        mutationRate=0.3,
        crossoverRate=0.6,
        mutation=gaussian(mutation_std)
    )
    
    Random.seed!(seed)
    result = Evolutionary.optimize(wrapped_obj, BoxConstraints(bounds...), chromosome, ga, 
                                   Evolutionary.Options(show_trace=false))
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end

# ============================================================
# Greedy recursive CP detection (model-agnostic)
# EXACT same structure as original MICA
# ============================================================
function detect_changepoints_mica(
    data::Matrix{Float64},
    initial_chromosome::Vector{Float64},
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    objective_fn::Function;
    n_global::Int=1,
    n_segment_specific::Int=1,
    min_length::Int=10,
    step::Int=10,
    penalty_fn::Function=(p, n) -> 10.0 * p * log(n),
    populationSize::Int=100,
    seed::Int=1234,
    mutation_std::Float64=0.0001
)
    n = size(data, 2)
    tau = [(0, n)]
    CP = Int[]
    
    # Initial optimization: no change points
    loss_val, best_params = optimize_with_changepoints(
        initial_chromosome, CP, bounds, n_global, n_segment_specific, data, objective_fn;
        populationSize=populationSize, seed=seed, mutation_std=mutation_std
    )
    loss_val += penalty_fn(n_global + n_segment_specific, n)
    
    # Extend chromosome for first segment
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
                append!(test_chrom, chromosome[n_global+1:n_global+n_segment_specific])
                append!(test_bounds[1], bounds[1][n_global+1:n_global+n_segment_specific])
                append!(test_bounds[2], bounds[2][n_global+1:n_global+n_segment_specific])
            end
            test_chrom = test_chrom[1:target_len]
            test_bounds = (
                test_bounds[1][1:target_len],
                test_bounds[2][1:target_len]
            )
            
            loss, best = optimize_with_changepoints(
                test_chrom, new_cp, test_bounds, n_global, n_segment_specific, data, objective_fn;
                populationSize=populationSize, seed=seed, mutation_std=mutation_std
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
                j_values = collect((a + min_length):step:(b - min_length))
                chpt = j_values[idx]
                push!(CP, chpt)
                CP = sort(CP)
                loss_val = minval
                best_params = y[idx]
                
                # Extend chromosome for next segment
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
# Public API: run_mica for each model
# ============================================================

# --- SIR ---
function run_mica_sir(data_I::Vector{Float64}, γ_true::Float64, kappa::Float64;
    populationSize=100, min_length=10, step=10, seed=1234, mutation_std=0.0001
)
    n = length(data_I)
    data = reshape(Float64.(data_I), 1, :)
    
    initial_chromosome = [γ_true, 0.0002]
    bounds = ([0.1, 0.0], [0.9, 0.1])
    
    pen_fn = (p, n) -> kappa * p * log(n)
    
    detected_cp, params = detect_changepoints_mica(
        data, copy(initial_chromosome), (copy(bounds[1]), copy(bounds[2])), objective_sir;
        n_global=1, n_segment_specific=1,
        min_length=min_length, step=step,
        penalty_fn=pen_fn,
        populationSize=populationSize, seed=seed, mutation_std=mutation_std
    )
    
    return detected_cp, params
end

# --- Linear Regression ---
function run_mica_lr(x::Vector{Float64}, y::Vector{Float64}, kappa::Float64;
    populationSize=100, min_length=10, step=10, seed=1234, mutation_std=0.5
)
    n = length(y)
    data = vcat(reshape(x, 1, :), reshape(y, 1, :))
    
    initial_chromosome = [0.5, 1.0]  # β_global, β_seg
    bounds = ([-10.0, -10.0], [10.0, 10.0])
    
    pen_fn = (p, n) -> kappa * p * log(n)
    
    detected_cp, params = detect_changepoints_mica(
        data, copy(initial_chromosome), (copy(bounds[1]), copy(bounds[2])), objective_lr;
        n_global=1, n_segment_specific=1,
        min_length=min_length, step=step,
        penalty_fn=pen_fn,
        populationSize=populationSize, seed=seed, mutation_std=mutation_std
    )
    
    return detected_cp, params
end

# --- AR(1) ---
function run_mica_ar(x::Vector{Float64}, kappa::Float64;
    populationSize=100, min_length=10, step=10, seed=1234, mutation_std=0.1
)
    n = length(x)
    data = reshape(x, 1, :)
    
    initial_chromosome = [1.0, 0.0]  # σ, φ
    bounds = ([0.01, -0.99], [10.0, 0.99])
    
    pen_fn = (p, n) -> kappa * p * log(n)
    
    detected_cp, params = detect_changepoints_mica(
        data, copy(initial_chromosome), (copy(bounds[1]), copy(bounds[2])), objective_ar;
        n_global=1, n_segment_specific=1,
        min_length=min_length, step=step,
        penalty_fn=pen_fn,
        populationSize=populationSize, seed=seed, mutation_std=mutation_std
    )
    
    return detected_cp, params
end
