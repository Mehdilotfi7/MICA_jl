# ============================================================
# 02_baselines.jl
# Baseline CPD methods for comparison with MICA
# ============================================================
using Statistics
using DifferentialEquations
using Optim

include("../models/ode_data_generation.jl")

# ============================================================
# Utility: Precision / Recall / F1
# ============================================================
function calc_metrics(detected_cps, true_cps; tolerance=5)
    TP = 0
    FP = 0
    matched_true = falses(length(true_cps))
    
    for dcp in detected_cps
        found_match = false
        for (i, tcp) in enumerate(true_cps)
            if !matched_true[i] && abs(dcp - tcp) <= tolerance
                TP += 1
                matched_true[i] = true
                found_match = true
                break
            end
        end
        if !found_match
            FP += 1
        end
    end
    
    FN = length(true_cps) - TP
    precision = (TP + FP) > 0 ? TP / (TP + FP) : 0.0
    recall = (TP + FN) > 0 ? TP / (TP + FN) : 0.0
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    
    return precision, recall, f1
end

# ============================================================
# Baseline 1: Binary Segmentation with RSS cost (BS-RSS)
# Pure statistical method: detects changes in mean of I(t)
# ============================================================
function segment_cost_rss(data, start_idx, end_idx)
    seg = data[start_idx:end_idx]
    m = mean(seg)
    return sum((seg .- m).^2)
end

function binary_segmentation_rss(data; max_cp=5, min_seg_len=10, pen=0.0)
    n = length(data)
    cp_list = Int[]
    segments = Tuple{Int,Int}[(1, n)]
    
    # Initial cost: no change points
    total_cost = segment_cost_rss(data, 1, n)
    
    for _ in 1:max_cp
        best_improvement = 0.0
        best_cp = nothing
        best_seg_idx = nothing
        best_new_cost = total_cost
        
        for (si, (a, b)) in enumerate(segments)
            if b - a + 1 < 2 * min_seg_len
                continue
            end
            current_seg_cost = segment_cost_rss(data, a, b)
            
            for j in (a + min_seg_len):(b - min_seg_len)
                left_cost = segment_cost_rss(data, a, j)
                right_cost = segment_cost_rss(data, j+1, b)
                new_cost = left_cost + right_cost
                improvement = current_seg_cost - new_cost
                
                if improvement > best_improvement
                    best_improvement = improvement
                    best_cp = j
                    best_seg_idx = si
                    best_new_cost = total_cost - current_seg_cost + new_cost
                end
            end
        end
        
        if best_cp === nothing || best_improvement <= pen
            break
        end
        
        total_cost = best_new_cost
        a, b = segments[best_seg_idx]
        deleteat!(segments, best_seg_idx)
        push!(segments, (a, best_cp))
        push!(segments, (best_cp+1, b))
        push!(cp_list, best_cp)
        sort!(cp_list)
        sort!(segments)
    end
    
    return sort(cp_list)
end

# ============================================================
# Baseline 2: PELT with RSS cost (PELT-RSS)
# Pruned Exact Linear Time algorithm
# ============================================================
function pelt_rss(data; min_seg_len=10, pen=2.0 * log(length(data)))
    n = length(data)
    
    # Cumulative sum for fast RSS computation
    cumsum_data = cumsum(data)
    cumsum_sq = cumsum(data.^2)
    
    function rss(a, b)
        if a > b
            return 0.0
        end
        len = b - a + 1
        s = cumsum_data[b] - (a > 1 ? cumsum_data[a-1] : 0.0)
        sq = cumsum_sq[b] - (a > 1 ? cumsum_sq[a-1] : 0.0)
        return sq - s^2 / len
    end
    
    # F[i] = optimal cost for data[1:i]
    F = fill(Inf, n)
    cp_set = [Int[] for _ in 1:n]
    
    # Initialize: cost for first min_seg_len points as one segment
    F[min_seg_len] = rss(1, min_seg_len)
    cp_set[min_seg_len] = Int[]
    
    for t in (min_seg_len + 1):n
        best_cost = Inf
        best_cp = Int[]
        
        # Try all possible last change points
        for τ in min_seg_len:(t - min_seg_len)
            cost = F[τ] + rss(τ + 1, t) + pen
            if cost < best_cost
                best_cost = cost
                best_cp = vcat(cp_set[τ], [τ])
            end
        end
        
        # Also consider no new change point (extend last segment)
        # This is implicitly handled by τ = t-1 but we need min_seg_len
        # Actually we handle it via the loop above
        
        F[t] = best_cost
        cp_set[t] = best_cp
    end
    
    return cp_set[n]
end

# ============================================================
# Baseline 3: Segmented SIR with Nelder-Mead (SegSIR-NM)
# Model-based baseline: for a given # of CPs, grid search
# over locations + Nelder-Mead for parameter optimization
# ============================================================
function fit_sir_segmented(data_I, cp_list, γ_fixed::Float64; u0=[9999.0, 1.0, 0.0])
    n = length(data_I)
    num_segments = length(cp_list) + 1
    sorted_cp = sort(cp_list)
    
    # Objective: sum of squared errors across all segments
    function obj(params)
        total_loss = 0.0
        u0_seg = copy(u0)
        
        for i in 1:num_segments
            β = params[i]
            γ = length(params) > num_segments ? params[end] : γ_fixed
            
            # Match Mica.jl segment indexing
            idx_start = (i == 1) ? 1 : sorted_cp[i - 1] + 1
            idx_end   = (i > length(sorted_cp)) ? n : sorted_cp[i]
            
            tspan = (0.0, Float64(idx_end - idx_start))
            
            prob = ODEProblem(sirmodel!, u0_seg, tspan, [β, γ])
            sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
            
            sim_I = sol[2, :]
            seg_data = data_I[idx_start:idx_end]
            
            # Match lengths
            len = min(length(sim_I), length(seg_data))
            total_loss += sum((sim_I[1:len] .- seg_data[1:len]).^2)
            
            if length(sol.u) > 0
                u0_seg = sol.u[end]
            end
        end
        
        return total_loss
    end
    
    initial_β = fill(0.0002, num_segments)
    lower = fill(1e-6, num_segments)
    upper = fill(0.01, num_segments)
    
    result = optimize(obj, lower, upper, initial_β, Fminbox(NelderMead()))
    best_params = Optim.minimizer(result)
    best_loss = Optim.minimum(result)
    
    return best_loss, best_params
end

function segsir_nm(data_I, true_γ::Float64; max_cp=3, min_seg_len=10, grid_step=5, u0=[9999.0, 1.0, 0.0])
    n = length(data_I)
    
    # Try 0, 1, 2, ..., max_cp change points
    best_overall_cost = Inf
    best_cp_list = Int[]
    best_params = nothing
    
    # 0 change points
    loss0, params0 = fit_sir_segmented(data_I, Int[], true_γ; u0=u0)
    pen = 2.0 * log(n)  # BIC-like penalty per parameter
    cost0 = loss0 + pen * 1  # 1 segment-specific parameter (β)
    
    best_overall_cost = cost0
    best_cp_list = Int[]
    best_params = params0
    
    # 1 change point
    for cp in (1+min_seg_len):(grid_step):(n-min_seg_len)
        loss, params = fit_sir_segmented(data_I, [cp], true_γ; u0=u0)
        cost = loss + pen * 2  # 2 segment-specific βs
        if cost < best_overall_cost
            best_overall_cost = cost
            best_cp_list = [cp]
            best_params = params
        end
    end
    
    # 2 change points (coarse grid)
    for cp1 in (1+min_seg_len):(grid_step*2):(n-2*min_seg_len)
        for cp2 in (cp1+min_seg_len):(grid_step*2):(n-min_seg_len)
            loss, params = fit_sir_segmented(data_I, [cp1, cp2], true_γ; u0=u0)
            cost = loss + pen * 3
            if cost < best_overall_cost
                best_overall_cost = cost
                best_cp_list = [cp1, cp2]
                best_params = params
            end
        end
    end
    
    return sort(best_cp_list), best_params, best_overall_cost
end

# ============================================================
# Test baselines
# ============================================================
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(1234)
    times, data_I, true_cp, true_beta, γ = generate_toy_dataset(
        beta_values=[0.00009, 0.00014, 0.00025],
        change_points=[50.0, 100.0],
        noise_level=10.0,
        noise_type="Uniform",
        tspan=(0.0, 160.0)
    )
    
    println("\n=== BS-RSS ===")
    cp_bs = binary_segmentation_rss(data_I, max_cp=5, min_seg_len=10, pen=0.0)
    println("Detected CPs: $cp_bs")
    p, r, f1 = calc_metrics(cp_bs, true_cp)
    println("Precision=$p, Recall=$r, F1=$f1")
    
    println("\n=== PELT-RSS ===")
    cp_pelt = pelt_rss(data_I, min_seg_len=10, pen=2.0*log(length(data_I)))
    println("Detected CPs: $cp_pelt")
    p, r, f1 = calc_metrics(cp_pelt, true_cp)
    println("Precision=$p, Recall=$r, F1=$f1")
    
    println("\n=== SegSIR-NM ===")
    cp_nm, params_nm, cost_nm = segsir_nm(data_I, γ, max_cp=2, min_seg_len=10, grid_step=10)
    println("Detected CPs: $cp_nm")
    p, r, f1 = calc_metrics(cp_nm, true_cp)
    println("Precision=$p, Recall=$r, F1=$f1")
end
