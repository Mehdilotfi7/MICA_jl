# ============================================================
# 06_new_baselines.jl
# Additional CPD baselines: Bayesian, Kernel MMD, HMM Regime Switching
# ============================================================
using Statistics
using Random
using LinearAlgebra

# ============================================================
# Baseline 4: Bayesian CPD (BIC-based binary segmentation)
# Uses BIC = n*log(RSS/n) + k*log(n) for model selection
# ============================================================

function bic_segment(data::Vector{Float64})
    n = length(data)
    if n <= 1
        return Inf
    end
    rss = sum((data .- mean(data)).^2)
    rss = max(rss, 1e-10)
    return n * log(rss / n) + log(n)
end

function bayesian_cpd(data::Vector{Float64}; max_cp=5, min_seg_len=10, pen_mult=2.0)
    n = length(data)
    cp_list = Int[]
    segments = Tuple{Int,Int}[(1, n)]
    total_cost = bic_segment(data)
    
    for _ in 1:max_cp
        best_improvement = 0.0
        best_cp = nothing
        best_seg_idx = nothing
        best_new_cost = total_cost
        
        for (si, (a, b)) in enumerate(segments)
            if b - a + 1 < 2 * min_seg_len
                continue
            end
            current_cost = bic_segment(data[a:b])
            
            for j in (a + min_seg_len):(b - min_seg_len)
                left_cost = bic_segment(data[a:j])
                right_cost = bic_segment(data[j+1:b])
                pen = pen_mult * log(n)
                new_cost = left_cost + right_cost + pen
                improvement = current_cost - new_cost
                
                if improvement > best_improvement
                    best_improvement = improvement
                    best_cp = j
                    best_seg_idx = si
                    best_new_cost = new_cost
                end
            end
        end
        
        if best_cp === nothing || best_improvement <= 0.0
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
# Baseline 5: Kernel-based CPD (Maximum Mean Discrepancy)
# Sliding window MMD with Gaussian kernel
# ============================================================

function gaussian_kernel(x, y; sigma=1.0)
    return exp(-sum((x .- y).^2) / (2.0 * sigma^2))
end

function mmd_statistic(X::Matrix{Float64}, Y::Matrix{Float64}; sigma=1.0)
    nx = size(X, 2)
    ny = size(Y, 2)
    
    # k(X,X)
    kxx = 0.0
    for i in 1:nx
        for j in 1:nx
            kxx += gaussian_kernel(X[:,i], X[:,j]; sigma=sigma)
        end
    end
    
    # k(Y,Y)
    kyy = 0.0
    for i in 1:ny
        for j in 1:ny
            kyy += gaussian_kernel(Y[:,i], Y[:,j]; sigma=sigma)
        end
    end
    
    # k(X,Y)
    kxy = 0.0
    for i in 1:nx
        for j in 1:ny
            kxy += gaussian_kernel(X[:,i], Y[:,j]; sigma=sigma)
        end
    end
    
    mmd2 = kxx/(nx*nx) + kyy/(ny*ny) - 2.0*kxy/(nx*ny)
    return sqrt(max(mmd2, 0.0))
end

function kernel_cpd(data::Vector{Float64}; window_size=20, min_seg_len=10, threshold=0.1)
    n = length(data)
    mmd_scores = zeros(n)
    
    # Median heuristic for bandwidth
    all_pairs = Float64[]
    for i in 1:min(100, n)
        for j in (i+1):min(100, n)
            push!(all_pairs, (data[i] - data[j])^2)
        end
    end
    sigma = length(all_pairs) > 0 ? sqrt(median(all_pairs)) : 1.0
    sigma = max(sigma, 1e-6)
    
    for t in (window_size + 1):(n - window_size)
        X = reshape(data[t-window_size:t-1], 1, window_size)
        Y = reshape(data[t:t+window_size-1], 1, window_size)
        mmd_scores[t] = mmd_statistic(X, Y; sigma=sigma)
    end
    
    # Find peaks above threshold
    cp_list = Int[]
    for t in (window_size + 2):(n - window_size - 1)
        if mmd_scores[t] > threshold &&
           mmd_scores[t] > mmd_scores[t-1] &&
           mmd_scores[t] > mmd_scores[t+1]
            push!(cp_list, t)
        end
    end
    
    # Merge close CPs (within min_seg_len)
    if !isempty(cp_list)
        merged = [cp_list[1]]
        for cp in cp_list[2:end]
            if cp - merged[end] >= min_seg_len
                push!(merged, cp)
            elseif mmd_scores[cp] > mmd_scores[merged[end]]
                merged[end] = cp
            end
        end
        cp_list = merged
    end
    
    return cp_list
end

# ============================================================
# Baseline 6: HMM Regime Switching (Gaussian HMM)
# Baum-Welch EM + Viterbi decoding
# ============================================================

function pdf_normal(x, mu, sigma)
    return exp(-0.5 * ((x - mu)/sigma)^2) / (sqrt(2π) * max(sigma, 1e-10))
end

function forward_hmm(obs, A, means, stds, pi_init)
    K = length(means)
    T = length(obs)
    alpha = zeros(K, T)
    c = zeros(T)
    
    for k in 1:K
        alpha[k,1] = pi_init[k] * pdf_normal(obs[1], means[k], stds[k])
    end
    c[1] = sum(alpha[:,1])
    alpha[:,1] ./= c[1]
    
    for t in 2:T
        for k in 1:K
            alpha[k,t] = pdf_normal(obs[t], means[k], stds[k]) * sum(A[j,k] * alpha[j,t-1] for j in 1:K)
        end
        c[t] = sum(alpha[:,t])
        alpha[:,t] ./= c[t]
    end
    
    return alpha, c
end

function backward_hmm(obs, A, means, stds, c)
    K = size(A, 1)
    T = length(obs)
    beta = zeros(K, T)
    beta[:,T] .= 1.0
    
    for t in (T-1):-1:1
        for k in 1:K
            beta[k,t] = sum(A[k,j] * pdf_normal(obs[t+1], means[j], stds[j]) * beta[j,t+1] / c[t+1] for j in 1:K)
        end
    end
    
    return beta
end

function baum_welch_hmm(obs, K; max_iter=50, tol=1e-4)
    T = length(obs)
    
    seg_len = div(T, K)
    means = Float64[]
    stds = Float64[]
    for k in 1:K
        start_idx = min((k-1)*seg_len + 1, T)
        end_idx = min(k*seg_len, T)
        seg = obs[start_idx:end_idx]
        push!(means, mean(seg))
        push!(stds, max(std(seg), 1e-3))
    end
    
    A = ones(K, K) / K
    for k in 1:K
        A[k,k] = 0.7
        A[k,:] ./= sum(A[k,:])
    end
    pi_init = fill(1.0/K, K)
    
    prev_loglik = -Inf
    
    for iter in 1:max_iter
        alpha, c = forward_hmm(obs, A, means, stds, pi_init)
        beta = backward_hmm(obs, A, means, stds, c)
        
        loglik = sum(log.(c .+ 1e-300))
        if abs(loglik - prev_loglik) < tol
            break
        end
        prev_loglik = loglik
        
        gamma = alpha .* beta
        for t in 1:T
            s = sum(gamma[:,t])
            if s > 0
                gamma[:,t] ./= s
            end
        end
        
        for k in 1:K
            nk = sum(gamma[k,:])
            if nk > 0
                means[k] = sum(gamma[k,t] * obs[t] for t in 1:T) / nk
                var_k = sum(gamma[k,t] * (obs[t] - means[k])^2 for t in 1:T) / nk
                stds[k] = sqrt(max(var_k, 1e-6))
            end
        end
        
        for i in 1:K
            for j in 1:K
                num = 0.0
                den = 0.0
                for t in 1:(T-1)
                    xi = alpha[i,t] * A[i,j] * pdf_normal(obs[t+1], means[j], stds[j]) * beta[j,t+1] / c[t+1]
                    num += xi
                    den += alpha[i,t] * beta[i,t] / c[t]
                end
                A[i,j] = num / max(den, 1e-10)
            end
            A[i,:] ./= sum(A[i,:])
        end
        
        pi_init = gamma[:,1]
        pi_init ./= sum(pi_init)
    end
    
    return A, means, stds, pi_init
end

function viterbi_hmm(obs, A, means, stds, pi_init)
    K = length(means)
    T = length(obs)
    
    log_delta = zeros(K, T)
    psi = zeros(Int, K, T)
    
    for k in 1:K
        log_delta[k,1] = log(max(pi_init[k], 1e-300)) + log(max(pdf_normal(obs[1], means[k], stds[k]), 1e-300))
    end
    
    for t in 2:T
        for k in 1:K
            best_val = -Inf
            best_j = 1
            for j in 1:K
                val = log_delta[j,t-1] + log(max(A[j,k], 1e-300))
                if val > best_val
                    best_val = val
                    best_j = j
                end
            end
            log_delta[k,t] = best_val + log(max(pdf_normal(obs[t], means[k], stds[k]), 1e-300))
            psi[k,t] = best_j
        end
    end
    
    states = zeros(Int, T)
    states[T] = argmax(log_delta[:,T])
    for t in (T-1):-1:1
        states[t] = psi[states[t+1], t+1]
    end
    
    return states
end

function hmm_regime_switching(data::Vector{Float64}; K=3, min_seg_len=10)
    K = min(K, 5)
    K = max(K, 2)
    
    A, means, stds, pi_init = baum_welch_hmm(data, K)
    states = viterbi_hmm(data, A, means, stds, pi_init)
    
    cp_list = Int[]
    for t in 2:length(states)
        if states[t] != states[t-1]
            push!(cp_list, t - 1)
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
    
    return sort(cp_list)
end
