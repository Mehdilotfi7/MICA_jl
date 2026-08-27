# ============================================================
# 12_baselines_linear_regression.jl
# Baseline methods adapted for piecewise linear regression
# ============================================================
using Statistics, Random, Optim

include("../models/lr_data_generation.jl")
include("generic_baselines.jl")

# --- BS-RSS (model-free) ---
function bs_rss_lr(y; max_cp=5, min_seg_len=10)
    return binary_segmentation_rss(y, max_cp=max_cp, min_seg_len=min_seg_len)
end

# --- PELT-RSS (model-free) ---
function pelt_rss_lr(y; min_seg_len=10, pen=2.0)
    n = length(y)
    F = fill(Inf, n)
    cp_set = [Int[] for _ in 1:n]
    
    F[min_seg_len] = segment_cost_rss(y, 1, min_seg_len)
    cp_set[min_seg_len] = Int[]
    
    for t in (min_seg_len+1):n
        best_cost = Inf
        best_cp = Int[]
        for τ in min_seg_len:(t-min_seg_len)
            seg_cost = segment_cost_rss(y, τ+1, t)
            cost = F[τ] + seg_cost + pen * log(n)
            if cost < best_cost
                best_cost = cost
                best_cp = vcat(cp_set[τ], [τ])
            end
        end
        F[t] = best_cost
        cp_set[t] = best_cp
    end
    return cp_set[n]
end

# --- Harrison LR (model-based) ---
function harrison_lr(x, y; min_seg_len=10, n_bkps=0, switch_gap=5, seed=1234)
    Random.seed!(seed)
    n = length(y)
    
    if n_bkps > 0
        cp_raw = binseg_ar1(y, max_cp=n_bkps, min_seg_len=min_seg_len)
    else
        cp_raw = binseg_ar1(y, max_cp=5, min_seg_len=min_seg_len)
    end
    
    if !isempty(cp_raw)
        filtered = [cp_raw[1]]
        for i in 2:length(cp_raw)
            if cp_raw[i] > filtered[end] + switch_gap
                push!(filtered, cp_raw[i])
            end
        end
        if filtered[end] == n
            pop!(filtered)
        end
        cp_raw = filtered
    end
    
    # Per-segment linear regression fit (Harrison-style)
    num_seg = length(cp_raw) + 1
    for seg_i in 1:num_seg
        idx_start = (seg_i == 1) ? 1 : cp_raw[seg_i-1] + 1
        idx_end = (seg_i > length(cp_raw)) ? n : cp_raw[seg_i]
        x_seg = x[idx_start:idx_end]
        y_seg = y[idx_start:idx_end]
        # OLS fit
        if length(y_seg) >= 2
            beta = cov(x_seg, y_seg) / var(x_seg)
            beta0 = mean(y_seg) - beta * mean(x_seg)
        end
    end
    
    return sort(cp_raw)
end

# --- SegSIR-style grid search for LR ---
function segsir_lr(x, y; max_cp=2, min_seg_len=10, grid_step=10, n_starts=2, seed=1234)
    Random.seed!(seed)
    n = length(y)
    
    function fit_lr_segmented(cp_list)
        num_seg = length(cp_list) + 1
        sorted_cp = sort(cp_list)
        total_rss = 0.0
        for i in 1:num_seg
            idx_start = (i == 1) ? 1 : sorted_cp[i-1] + 1
            idx_end = (i > length(sorted_cp)) ? n : sorted_cp[i]
            x_seg = x[idx_start:idx_end]
            y_seg = y[idx_start:idx_end]
            if length(y_seg) >= 2
                beta = cov(x_seg, y_seg) / max(var(x_seg), 1e-10)
                beta0 = mean(y_seg) - beta * mean(x_seg)
                y_pred = beta .* x_seg .+ beta0
                total_rss += sum((y_seg .- y_pred).^2)
            end
        end
        return total_rss
    end
    
    pen = 2.0 * log(n)
    best_cost = Inf
    best_cp = Int[]
    
    # 0 CP
    cost0 = fit_lr_segmented(Int[]) + pen
    best_cost = cost0
    best_cp = Int[]
    
    # 1 CP
    for cp in (1+min_seg_len):grid_step:(n-min_seg_len)
        cost = fit_lr_segmented([cp]) + 2*pen
        if cost < best_cost
            best_cost = cost
            best_cp = [cp]
        end
    end
    
    # 2 CPs
    for cp1 in (1+min_seg_len):grid_step:(n-2*min_seg_len)
        for cp2 in (cp1+min_seg_len):grid_step:(n-min_seg_len)
            cost = fit_lr_segmented([cp1, cp2]) + 3*pen
            if cost < best_cost
                best_cost = cost
                best_cp = [cp1, cp2]
            end
        end
    end
    
    return sort(best_cp)
end
