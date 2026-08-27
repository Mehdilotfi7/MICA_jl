# ============================================================
# 10_mica_linear_regression.jl
# MICA-style GA for piecewise linear regression
# Model: y = beta_seg * x + beta_global + noise
# ============================================================
using Random
using Statistics
using Evolutionary

include("../models/lr_data_generation.jl")

function run_mica_lr(x::Vector{Float64}, y::Vector{Float64}, kappa::Float64;
    populationSize=100, min_length=10, step=10, seed=1234, num_runs=1)
    
    Random.seed!(seed)
    n = length(y)
    
    function fit_linear_regression(x_seg, y_seg)
        # OLS: y = beta_seg * x + beta_global
        n_seg = length(y_seg)
        x_mean = mean(x_seg)
        y_mean = mean(y_seg)
        beta_seg = cov(x_seg, y_seg) / var(x_seg)
        beta_global = y_mean - beta_seg * x_mean
        y_pred = beta_seg .* x_seg .+ beta_global
        rss = sum((y_seg .- y_pred).^2)
        return beta_seg, beta_global, rss
    end
    
    function fitness(chromosome)
        cp_list = sort(unique(chromosome[1:end-1]))
        # Last gene is not used for CPs in this encoding
        num_seg = length(cp_list) + 1
        
        total_rss = 0.0
        for i in 1:num_seg
            idx_start = (i == 1) ? 1 : Int(round(cp_list[i-1])) + 1
            idx_end = (i > length(cp_list)) ? n : Int(round(cp_list[i]))
            if idx_end > n
                idx_end = n
            end
            if idx_start < 1
                idx_start = 1
            end
            if idx_end - idx_start + 1 < 2
                total_rss += sum(y[idx_start:idx_end].^2)
                continue
            end
            _, _, rss = fit_linear_regression(x[idx_start:idx_end], y[idx_start:idx_end])
            total_rss += rss
        end
        
        # BIC penalty
        penalty = kappa * (length(cp_list) + 1) * log(n)
        return total_rss + penalty
    end
    
    # Generate initial population
    max_cp = 5
    pop = Vector{Float64}[]
    for _ in 1:populationSize
        n_cp = rand(0:max_cp)
        cp_genes = sort(rand(1.0:Float64(n), n_cp))
        chrom = vcat(cp_genes, fill(0.0, max_cp - n_cp))
        push!(pop, chrom)
    end
    
    ga = GA(populationSize=populationSize, crossoverRate=0.8, mutationRate=0.1)
    result = Evolutionary.optimize(fitness, pop, ga,
        Evolutionary.Options(iterations=200, show_trace=false))
    
    best_chrom = Evolutionary.minimizer(result)
    detected_cp = sort(unique(Int.(round.(best_chrom[best_chrom .> 0]))))
    detected_cp = filter(cp -> cp >= min_length && cp <= n - min_length, detected_cp)
    
    return sort(detected_cp)
end
