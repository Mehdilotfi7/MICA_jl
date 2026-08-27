# ============================================================
# 11_mica_ar1.jl
# MICA-style GA for piecewise AR(1) time series
# Model: x[t] = phi_seg * x[t-1] + epsilon
# ============================================================
using Random
using Statistics
using Evolutionary

function run_mica_ar1(x::Vector{Float64}, kappa::Float64;
    populationSize=100, min_length=10, step=10, seed=1234, num_runs=1)
    
    Random.seed!(seed)
    n = length(x)
    
    function fit_ar1(seg_data)
        n_seg = length(seg_data)
        if n_seg < 3
            return 0.0, sum(seg_data.^2)
        end
        y = seg_data[2:end]
        x_lag = seg_data[1:end-1]
        phi = cov(x_lag, y) / max(var(x_lag), 1e-10)
        phi = clamp(phi, -0.99, 0.99)
        residuals = y .- phi .* x_lag
        rss = sum(residuals.^2)
        return phi, rss
    end
    
    function fitness(chromosome)
        cp_list = sort(unique(chromosome[1:end-1]))
        cp_list = filter(cp -> cp >= min_length && cp <= n - min_length, Int.(round.(cp_list)))
        cp_list = sort(unique(cp_list))
        num_seg = length(cp_list) + 1
        
        total_rss = 0.0
        for i in 1:num_seg
            idx_start = (i == 1) ? 1 : cp_list[i-1] + 1
            idx_end = (i > length(cp_list)) ? n : cp_list[i]
            if idx_end > n
                idx_end = n
            end
            if idx_start < 1
                idx_start = 1
            end
            if idx_end - idx_start + 1 < 3
                total_rss += sum(x[idx_start:idx_end].^2)
                continue
            end
            _, rss = fit_ar1(x[idx_start:idx_end])
            total_rss += rss
        end
        
        penalty = kappa * (length(cp_list) + 1) * log(n)
        return total_rss + penalty
    end
    
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
