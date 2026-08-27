# ============================================================
# 07_published_baselines.jl
# Published model-based CPD methods adapted for SIR benchmarking
# Harrison implementation matches their repo EXACTLY:
#   - One-step-ahead Euler objective (not full ODE integration)
#   - Actual observed data as initial condition at each step
#   - Nelder-Mead only (no DE)
#   - Previous segment's params as next segment's initial guess
# ============================================================
using Statistics
using DifferentialEquations
using Optim
using Random

include("../models/ode_data_generation.jl")

# ============================================================
# Helper: AR(1) cost for binary segmentation
# ============================================================
function ar1_cost(data)
    n = length(data)
    if n < 3
        return sum(data.^2)
    end
    y = data[2:end]
    x_lag = data[1:end-1]
    φ = cov(x_lag, y) / max(var(x_lag), 1e-10)
    μ = mean(y) - φ * mean(x_lag)
    residuals = y .- φ .* x_lag .- μ
    return sum(residuals.^2)
end

function segment_cost_ar1(data, a, b)
    seg = data[a:b]
    return ar1_cost(seg)
end

# Binary segmentation with AR(1) cost (like ruptures.Binseg)
function binseg_ar1(data; max_cp=5, min_seg_len=10, epsilon=0.0)
    n = length(data)
    cp_list = Int[]
    segments = Tuple{Int,Int}[(1, n)]
    total_cost = segment_cost_ar1(data, 1, n)
    
    for _ in 1:max_cp
        best_improvement = 0.0
        best_cp = nothing
        best_seg_idx = nothing
        
        for (si, (a, b)) in enumerate(segments)
            if b - a + 1 < 2 * min_seg_len
                continue
            end
            current_cost = segment_cost_ar1(data, a, b)
            
            for j in (a + min_seg_len):(b - min_seg_len)
                left_cost = segment_cost_ar1(data, a, j)
                right_cost = segment_cost_ar1(data, j+1, b)
                improvement = current_cost - left_cost - right_cost
                if improvement > best_improvement
                    best_improvement = improvement
                    best_cp = j
                    best_seg_idx = si
                end
            end
        end
        
        if best_cp === nothing || best_improvement <= epsilon
            break
        end
        
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
# Helper: coarse log-spaced grid search for NM initial guess
# ============================================================
function grid_search_init(obj, lower, upper; n_grid=20)
    best_x = lower[1]
    best_val = obj([best_x])
    for x in 10 .^ range(log10(lower[1]), log10(upper[1]), length=n_grid)
        val = obj([x])
        if val < best_val
            best_val = val
            best_x = x
        end
    end
    return [best_x]
end

# ============================================================
# Method 1a: Harrison et al. — SINGLE state observation (I only)
# EXACT match to their repo:
#   - One-step-ahead Euler objective
#   - Observed data[i] as initial condition for predicting data[i+1]
# ============================================================

function harrison_nm_single(data_I::Vector{Float64}, γ_fixed::Float64;
    min_seg_len::Int=10, n_bkps::Int=0, epsilon::Float64=0.0,
    switch_gap::Int=5, seed::Int=1234)
    
    Random.seed!(seed)
    n = length(data_I)
    
    # Step 1: Binary segmentation with AR(1) cost on I(t)
    if n_bkps > 0
        cp_raw = binseg_ar1(data_I, max_cp=n_bkps, min_seg_len=min_seg_len)
    else
        cp_raw = binseg_ar1(data_I, max_cp=5, min_seg_len=min_seg_len, epsilon=epsilon)
    end
    
    # Remove close switches
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
    
    # Step 2: Per-segment parameter fitting with Nelder-Mead
    # Harrison's objective: one-step-ahead Euler prediction
    # At each step i: x_pred[i+1] = x_obs[i] + f(x_obs[i], params) * dt
    num_seg = length(cp_raw) + 1
    lower = fill(1e-6, 1)
    upper = fill(0.01, 1)
    
    β_estimates = Float64[]
    
    for seg_i in 1:num_seg
        idx_start = (seg_i == 1) ? 1 : cp_raw[seg_i-1] + 1
        idx_end = (seg_i > length(cp_raw)) ? n : cp_raw[seg_i]
        seg_data = data_I[idx_start:idx_end]
        seg_len = length(seg_data)
        
        if seg_len < 2
            push!(β_estimates, 0.0002)
            continue
        end
        
        # Harrison's one-step-ahead Euler objective
        # Uses ACTUAL OBSERVED data[i] as initial condition for step i+1
        function seg_obj(β_vec)
            β = β_vec[1]
            total_err = 0.0
            for i in 1:(seg_len-1)
                S_i = 10000.0 - seg_data[i]  # approximate S from I (for single-state)
                # Euler step: I_pred[i+1] = I[i] + (β*S*I - γ*I) * dt
                # dt = 1.0 (saveat=1.0)
                dI = β * S_i * seg_data[i] - γ_fixed * seg_data[i]
                I_pred = seg_data[i] + dI
                total_err += (I_pred - seg_data[i+1])^2
            end
            return total_err
        end
        
        # Initial guess: grid search for segment 1, previous result for others
        if seg_i == 1
            β_init = grid_search_init(seg_obj, lower, upper; n_grid=20)
        else
            β_init = [β_estimates[end]]
        end
        β_init = clamp.(β_init, lower .+ 1e-8, upper .- 1e-8)
        
        result = optimize(seg_obj, lower, upper, β_init, Fminbox(NelderMead()))
        β_opt = Optim.minimizer(result)[1]
        push!(β_estimates, β_opt)
    end
    
    return sort(cp_raw), β_estimates
end

# ============================================================
# Method 1b: Harrison et al. — ALL states observation (S, I, R)
# EXACT match to their repo:
#   - One-step-ahead Euler objective
#   - Observed state[:, i] as initial condition for predicting step i+1
# ============================================================

function harrison_nm_all(data_all::Matrix{Float64}, γ_fixed::Float64;
    min_seg_len::Int=10, n_bkps::Int=0, epsilon::Float64=0.0,
    switch_gap::Int=5, seed::Int=1234)
    
    Random.seed!(seed)
    n = size(data_all, 2)
    num_states = size(data_all, 1)
    
    # Step 1: Binary segmentation with AR(1) cost on EACH state
    all_cps = Int[]
    for j in 1:num_states
        state_data = data_all[j, :]
        if n_bkps > 0
            cps = binseg_ar1(state_data, max_cp=n_bkps, min_seg_len=min_seg_len)
        else
            cps = binseg_ar1(state_data, max_cp=5, min_seg_len=min_seg_len, epsilon=epsilon)
        end
        
        # Remove close switches within this state
        if !isempty(cps)
            filtered = [cps[1]]
            for i in 2:length(cps)
                if cps[i] > filtered[end] + switch_gap
                    push!(filtered, cps[i])
                end
            end
            if !isempty(filtered) && filtered[end] == n
                pop!(filtered)
            end
            append!(all_cps, filtered)
        end
    end
    
    # Merge switches across all states
    all_cps = sort(unique(all_cps))
    
    # Remove close switches across all states
    if !isempty(all_cps)
        result = [all_cps[1]]
        for i in 2:length(all_cps)
            if all_cps[i] > result[end] + switch_gap
                push!(result, all_cps[i])
            end
        end
        if result[end] == n
            pop!(result)
        end
        all_cps = result
    end
    
    # Note: Harrison's implementation does NOT cap total merged CPs to n_bkps.
    # Each state independently returns up to n_bkps CPs, and the merge can produce more.
    
    # Step 2: Per-segment parameter fitting with Nelder-Mead
    # Harrison's objective: one-step-ahead Euler prediction
    # At each step i: x_pred[i+1] = x_obs[i] + f(x_obs[i], params) * dt
    num_seg = length(all_cps) + 1
    lower = fill(1e-6, 1)
    upper = fill(0.01, 1)
    
    β_estimates = Float64[]
    
    for seg_i in 1:num_seg
        idx_start = (seg_i == 1) ? 1 : all_cps[seg_i-1] + 1
        idx_end = (seg_i > length(all_cps)) ? n : all_cps[seg_i]
        seg_data = data_all[:, idx_start:idx_end]
        seg_len = size(seg_data, 2)
        
        if seg_len < 2
            push!(β_estimates, 0.0002)
            continue
        end
        
        # Harrison's one-step-ahead Euler objective
        # Uses ACTUAL OBSERVED state[:, i] as initial condition for step i+1
        function seg_obj(β_vec)
            β = β_vec[1]
            total_err = 0.0
            for i in 1:(seg_len-1)
                S_i = seg_data[1, i]
                I_i = seg_data[2, i]
                R_i = seg_data[3, i]
                # Euler step
                dS = -β * S_i * I_i
                dI = β * S_i * I_i - γ_fixed * I_i
                dR = γ_fixed * I_i
                S_pred = S_i + dS
                I_pred = I_i + dI
                R_pred = R_i + dR
                total_err += (S_pred - seg_data[1, i+1])^2
                total_err += (I_pred - seg_data[2, i+1])^2
                total_err += (R_pred - seg_data[3, i+1])^2
            end
            return total_err
        end
        
        # Initial guess
        if seg_i == 1
            β_init = grid_search_init(seg_obj, lower, upper; n_grid=20)
        else
            β_init = [β_estimates[end]]
        end
        β_init = clamp.(β_init, lower .+ 1e-8, upper .- 1e-8)
        
        result = optimize(seg_obj, lower, upper, β_init, Fminbox(NelderMead()))
        β_opt = Optim.minimizer(result)[1]
        push!(β_estimates, β_opt)
    end
    
    return sort(all_cps), β_estimates
end

# ============================================================
# Method 2: Xu et al. (IISE Transactions, 2023)
# ============================================================

function xu_multipelt(data_I::Vector{Float64}, γ_fixed::Float64;
    min_seg_len::Int=10, max_cp::Int=5, pen_cp::Float64=2.0, pen_mode::Float64=1.0)
    
    n = length(data_I)
    
    function param_cost(a, b)
        seg = data_I[a:b]
        len = b - a + 1
        if len < min_seg_len
            return Inf
        end
        if len < 3
            return sum(seg.^2)
        end
        y = seg[2:end]
        x_lag = seg[1:end-1]
        φ = cov(x_lag, y) / max(var(x_lag), 1e-10)
        μ = mean(y) - φ * mean(x_lag)
        residuals = y .- φ .* x_lag .- μ
        return sum(residuals.^2)
    end
    
    F = fill(Inf, n)
    cp_set = [Int[] for _ in 1:n]
    
    F[min_seg_len] = param_cost(1, min_seg_len)
    cp_set[min_seg_len] = Int[]
    
    for t in (min_seg_len+1):n
        best_cost = Inf
        best_cp = Int[]
        
        for τ in min_seg_len:(t-min_seg_len)
            seg_cost = param_cost(τ+1, t)
            num_modes = length(cp_set[τ]) + 1
            cost = F[τ] + seg_cost + pen_cp * log(n) + pen_mode * log(num_modes)
            if cost < best_cost
                best_cost = cost
                best_cp = vcat(cp_set[τ], [τ])
            end
        end
        
        F[t] = best_cost
        cp_set[t] = best_cp
    end
    
    detected = cp_set[n]
    
    refined = Int[]
    for cp in detected
        window = max(cp-5, min_seg_len+1):min(cp+5, n-min_seg_len)
        best_shift = cp
        best_local_cost = Inf
        for new_cp in window
            test_cps = sort([setdiff(detected, [cp]); [new_cp]])
            c = 0.0
            for i in 1:(length(test_cps)+1)
                a = (i == 1) ? 1 : test_cps[i-1] + 1
                b = (i > length(test_cps)) ? n : test_cps[i]
                c += param_cost(a, b)
            end
            c += pen_cp * log(n) * (length(test_cps) + 1)
            if c < best_local_cost
                best_local_cost = c
                best_shift = new_cp
            end
        end
        push!(refined, best_shift)
    end
    
    return sort(unique(refined))
end

# ============================================================
# Method 3: Maulik et al. (arXiv:2305.10423)
# ============================================================

function maulik_ml(data_I::Vector{Float64}; window_size::Int=20, threshold::Float64=0.1, min_seg_len::Int=10)
    n = length(data_I)
    scores = zeros(n)
    
    for t in (window_size+1):(n-window_size)
        left = data_I[t-window_size:t-1]
        right = data_I[t:t+window_size-1]
        
        yL = left[2:end]
        xL = left[1:end-1]
        φL = cov(xL, yL) / max(var(xL), 1e-10)
        μL = mean(yL) - φL * mean(xL)
        
        yR = right[2:end]
        xR = right[1:end-1]
        φR = cov(xR, yR) / max(var(xR), 1e-10)
        μR = mean(yR) - φR * mean(xR)
        
        param_diff = abs(φL - φR) + abs(μL - μR) / max(std(data_I), 1e-6)
        
        pred_R = φL .* xR .+ μL
        recon_error = sqrt(mean((pred_R .- yR).^2))
        
        scores[t] = param_diff + recon_error / max(std(data_I), 1e-6)
    end
    
    cp_list = Int[]
    for t in (window_size+2):(n-window_size-1)
        if scores[t] > threshold && scores[t] > scores[t-1] && scores[t] > scores[t+1]
            push!(cp_list, t)
        end
    end
    
    if !isempty(cp_list)
        merged = [cp_list[1]]
        for cp in cp_list[2:end]
            if cp - merged[end] >= min_seg_len
                push!(merged, cp)
            end
        end
        cp_list = merged
    end
    
    return cp_list
end

# ============================================================
# Method 4: Improved SegSIR-NM
# ============================================================

function segsir_nm_improved(data_I::Vector{Float64}, γ_fixed::Float64;
    max_cp::Int=3, min_seg_len::Int=10, grid_step::Int=10, n_starts::Int=5, seed::Int=1234)
    
    Random.seed!(seed)
    n = length(data_I)
    
    function fit_sir_segmented(cp_list, γ)
        n_data = length(data_I)
        num_segments = length(cp_list) + 1
        sorted_cp = sort(cp_list)
        
        function obj(params)
            total_loss = 0.0
            u0_seg = [9999.0, 1.0, 0.0]
            for i in 1:num_segments
                β = params[i]
                idx_start = (i == 1) ? 1 : sorted_cp[i-1] + 1
                idx_end = (i > length(sorted_cp)) ? n_data : sorted_cp[i]
                tspan = (0.0, Float64(idx_end - idx_start))
                prob = ODEProblem(sirmodel!, u0_seg, tspan, [β, γ])
                sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
                sim_I = sol[2, :]
                seg_data = data_I[idx_start:idx_end]
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
        
        best_loss = Inf
        best_params = initial_β
        
        for s in 1:n_starts
            init = lower .+ (upper .- lower) .* rand(length(lower))
            init = clamp.(init, lower .+ 1e-8, upper .- 1e-8)
            try
                result = optimize(obj, lower, upper, init, Fminbox(NelderMead()))
                if Optim.minimum(result) < best_loss
                    best_loss = Optim.minimum(result)
                    best_params = Optim.minimizer(result)
                end
            catch
                # Skip failed starts
            end
        end
        
        return best_loss, best_params
    end
    
    pen = 2.0 * log(n)
    best_overall_cost = Inf
    best_cp_list = Int[]
    best_params = nothing
    
    loss0, params0 = fit_sir_segmented(Int[], γ_fixed)
    cost0 = loss0 + pen * 1
    best_overall_cost = cost0
    best_cp_list = Int[]
    best_params = params0
    
    for cp in (1+min_seg_len):grid_step:(n-min_seg_len)
        loss, params = fit_sir_segmented([cp], γ_fixed)
        cost = loss + pen * 2
        if cost < best_overall_cost
            best_overall_cost = cost
            best_cp_list = [cp]
            best_params = params
        end
    end
    
    for cp1 in (1+min_seg_len):grid_step:(n-2*min_seg_len)
        for cp2 in (cp1+min_seg_len):grid_step:(n-min_seg_len)
            loss, params = fit_sir_segmented([cp1, cp2], γ_fixed)
            cost = loss + pen * 3
            if cost < best_overall_cost
                best_overall_cost = cost
                best_cp_list = [cp1, cp2]
                best_params = params
            end
        end
    end
    
    if max_cp >= 3
        coarse_step = grid_step * 2
        for cp1 in (1+min_seg_len):coarse_step:(n-3*min_seg_len)
            for cp2 in (cp1+min_seg_len):coarse_step:(n-2*min_seg_len)
                for cp3 in (cp2+min_seg_len):coarse_step:(n-min_seg_len)
                    loss, params = fit_sir_segmented([cp1, cp2, cp3], γ_fixed)
                    cost = loss + pen * 4
                    if cost < best_overall_cost
                        best_overall_cost = cost
                        best_cp_list = [cp1, cp2, cp3]
                        best_params = params
                    end
                end
            end
        end
    end
    
    return sort(best_cp_list), best_params, best_overall_cost
end
