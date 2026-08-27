

"""
    call_penalty_fn(f::Function; kwargs...) -> Real

Safely call a user-provided penalty function `f` with only the arguments it accepts, using the *last defined method* of `fn`.

This allows users to define custom penalty functions that take any subset of the following:

- `p`: number of segment-specific parameters
- `n`: total data length
- `CP`: vector of change points
- `segment_lengths`: vector of segment lengths (computed as `diff([0; CP; n])`)
- `num_segments`: number of segments (`length(CP) + 1`)

# Important Note

Due to Julia's multiple dispatch behavior, defining multiple methods for the same function name
accumulates methods rather than replacing them. This means:

- `call_penalty_fn` uses the *last method* returned by `methods(fn)`.
- To test different penalty functions, **use different function names** 
  (e.g., `my_penalty1`, `my_penalty2`, `my_penalty3`) to avoid ambiguity.

# Examples

```julia
my_penalty(p, n) = 2 * p * log(n)

call_penalty_fn(my_penalty;
    p=3, n=250, CP=[50, 100], segment_lengths=[50, 50, 150], num_segments=3
)
# => 33.21
More complex example:
function imbalance_penalty(p, n, CP)
    seg_lengths = diff([0; CP; n])
    imbalance = std(seg_lengths)
    return 3.3 * p * length(CP) * log(n) + 0.12 * imbalance
end

call_penalty_fn(imbalance_penalty;
    p=3, n=250, CP=[60, 130], segment_lengths=[60, 70, 120], num_segments=3
)
If the user omits some arguments, only those required by their function are passed.
"""
function call_penalty_fn(fn::Function; kwargs...)
    ms = collect(methods(fn))
    argnames = method_argnames(last(ms))[2:end]
    args = Vector{Any}()
    for arg in argnames
        if haskey(kwargs, arg)
            push!(args, kwargs[arg])
        else
            error("Missing argument `$(arg)` needed by penalty function.")
        end
    end

    return fn(args...)  # Call with positional arguments
end

function method_argnames(m::Method)
    argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), m.slot_syms)
    isempty(argnames) && return argnames
    return argnames[1:m.nargs]
end

#------------------------

# Proper Information Criteria (for objective_type = :bic, :mdl, :aic)

#------------------------

"""
    compute_bic(loss, n, p_total; sigma2_null=1.0, scale_penalty=true)

Compute the Bayesian Information Criterion for least-squares regression.

Standard form:
    BIC = n * log(RSS / n) + p_total * log(n)

When `scale_penalty=true`, the complexity penalty is scaled by the null-model
variance `sigma2_null = RSS_null / n` (the residual variance of a constant-mean
model on the whole series). This makes the penalty term comparable in scale to
the RSS loss regardless of whether the data are raw or standardised:

    BIC_scaled = n * log(RSS / n) + sigma2_null * p_total * log(n)

With `scale_penalty=false` the ordinary BIC is returned.
"""
function compute_bic(loss::Real, n::Int, p_total::Int; sigma2_null::Real=1.0, scale_penalty::Bool=true)
    sigma2 = max(loss / n, 1e-10)
    penalty = p_total * log(n)
    return n * log(sigma2) + (scale_penalty ? sigma2_null * penalty : penalty)
end

"""
    compute_mdl(loss, n, p_total; sigma2_null=1.0, scale_penalty=true)

Compute the Minimum Description Length criterion (Rissanen's MDL).

Standard form:
    MDL = (n/2) * log(RSS / n) + (p_total/2) * log(n)

Scaled form (when `scale_penalty=true`):
    MDL_scaled = (n/2) * log(RSS / n) + sigma2_null * (p_total/2) * log(n)
"""
function compute_mdl(loss::Real, n::Int, p_total::Int; sigma2_null::Real=1.0, scale_penalty::Bool=true)
    sigma2 = max(loss / n, 1e-10)
    penalty = (p_total / 2) * log(n)
    return (n / 2) * log(sigma2) + (scale_penalty ? sigma2_null * penalty : penalty)
end

"""
    compute_aic(loss, n, p_total; sigma2_null=1.0, scale_penalty=true)

Compute Akaike Information Criterion for least-squares.

Standard form:
    AIC = n * log(RSS / n) + 2 * p_total

Scaled form (when `scale_penalty=true`):
    AIC_scaled = n * log(RSS / n) + sigma2_null * 2 * p_total
"""
function compute_aic(loss::Real, n::Int, p_total::Int; sigma2_null::Real=1.0, scale_penalty::Bool=true)
    sigma2 = max(loss / n, 1e-10)
    penalty = 2 * p_total
    return n * log(sigma2) + (scale_penalty ? sigma2_null * penalty : penalty)
end

#------------------------

# TCPD-style information criteria (objective_type = :tcpd_*)
#
# These criteria use a parameter count that matches the TCPD benchmark:
#   p_count = n_global + num_cps
# i.e. global parameters plus one parameter per change point.  They do NOT
# multiply by n_segment_specific, because TCPD methods (changepoint, WBS,
# RFPOP) penalise one parameter per segment change, not one per
# segment-specific model parameter.

#------------------------

"""
    compute_information_criterion(loss, n, p_count; sigma2_null=1.0, scale_penalty=true, type=:bic)

Compute BIC/MDL/AIC/Hannan-Quinn/SIC with a user-supplied parameter count.

- `type = :bic` or `:sic`:  `n*log(RSS/n) + p_count*log(n)`
- `type = :mdl`:             `(n/2)*log(RSS/n) + (p_count/2)*log(n)`
- `type = :aic`:             `n*log(RSS/n) + 2*p_count`
- `type = :hannan_quinn`:    `n*log(RSS/n) + 2*p_count*log(log(n))`
- `type = :none`:             `n*log(RSS/n)` (no penalty)

When `scale_penalty=true`, the penalty is multiplied by `sigma2_null`
(`RSS_null / n`) so that the objective is comparable across datasets with
different scales.
"""
function compute_information_criterion(loss::Real, n::Int, p_count::Int;
                                       sigma2_null::Real=1.0,
                                       scale_penalty::Bool=true,
                                       type::Symbol=:bic)
    sigma2 = max(loss / n, 1e-10)
    if type == :mdl
        base = (n / 2) * log(sigma2)
        penalty = (p_count / 2) * log(n)
    else
        base = n * log(sigma2)
        if type == :bic || type == :sic
            penalty = p_count * log(n)
        elseif type == :mbic
            penalty = p_count * log(n)  # note: WBS MBIC is handled separately
        elseif type == :aic
            penalty = 2 * p_count
        elseif type == :hannan_quinn
            penalty = 2 * p_count * log(log(n))
        elseif type == :none
            penalty = 0.0
        else
            error("Unknown information criterion type: $type")
        end
    end
    return base + (scale_penalty ? sigma2_null * penalty : penalty)
end

#------------------------

# Exact TCPD reference-package penalties
#
# These mirror the formulas used by the R/Python packages in the TCPD benchmark
# (van den Burg & Williams, 2020).  The key difference from the criteria above
# is that TCPD methods count only the parameters introduced by a change, not
# one copy of every segment-specific parameter for every segment.
#
# diffparam := number of segment-specific parameters that change at a CP
#              (for MICA this is n_segment_specific).
# k         := number of change points.
# l_i       := segment lengths (computed as diff([0; CP; n])).
#
# Sources: TCPDBench, changepoint, changepoint.np, wbs, robseg.

#------------------------

"""
    compute_tcpd_changepoint_penalty(loss, n, num_cps, n_segment_specific;
                                     sigma2_null=1.0, scale_penalty=true, type=:bic)

Exact `changepoint` / `changepoint.np` penalty family.

- `:bic`  / `:sic`:  `(diffparam + 1) * log(n)`
- `:mbic`:            `(diffparam + 2) * log(n)`
- `:aic`:              `2 * (diffparam + 1)`
- `:hannan_quinn`:     `2 * (diffparam + 1) * log(log(n))`
- `:none`:             `0`

The postfix-0 variants (which omit the `+1`) are exposed as `:bic0`, `:aic0`,
`:hannan_quinn0`.  The base likelihood term is the standard least-squares BIC
base: `n * log(RSS / n)`.
"""
function compute_tcpd_changepoint_penalty(loss::Real, n::Int, num_cps::Int,
                                          n_segment_specific::Int;
                                          sigma2_null::Real=1.0,
                                          scale_penalty::Bool=true,
                                          type::Symbol=:bic)
    sigma2 = max(loss / n, 1e-10)
    base = n * log(sigma2)
    diffparam = n_segment_specific
    if type == :bic || type == :sic
        penalty = (diffparam + 1) * log(n)
    elseif type == :mbic
        penalty = (diffparam + 2) * log(n)
    elseif type == :aic
        penalty = 2 * (diffparam + 1)
    elseif type == :hannan_quinn
        penalty = 2 * (diffparam + 1) * log(log(n))
    elseif type == :none
        penalty = 0.0
    elseif type == :bic0 || type == :sic0
        penalty = diffparam * log(n)
    elseif type == :aic0
        penalty = 2 * diffparam
    elseif type == :hannan_quinn0
        penalty = 2 * diffparam * log(log(n))
    else
        error("Unknown changepoint penalty type: $type")
    end
    return base + (scale_penalty ? sigma2_null * penalty : penalty)
end

"""
    compute_tcpd_wbs_penalty(loss, n, num_cps, segment_lengths;
                           sigma2_null=1.0, scale_penalty=true, type=:bic, alpha=1.01)

Exact WBS penalty family.  WBS uses the objective `n/2 * log(sigma2) + penalty`.

- `:bic`:  `k * log(n)`
- `:mbic`: `1.5 * k * log(n) + 0.5 * sum(log(l_i / n))`
- `:ssic`: `k * (log(n))^alpha`  (default alpha = 1.01)
"""
function compute_tcpd_wbs_penalty(loss::Real, n::Int, num_cps::Int,
                                  segment_lengths::Vector{Int};
                                  sigma2_null::Real=1.0,
                                  scale_penalty::Bool=true,
                                  type::Symbol=:bic,
                                  alpha::Real=1.01)
    sigma2 = max(loss / n, 1e-10)
    base = (n / 2) * log(sigma2)
    k = num_cps
    if type == :bic || type == :sic
        penalty = k * log(n)
    elseif type == :mbic
        penalty = 1.5 * k * log(n) + 0.5 * sum(log.(segment_lengths ./ n))
    elseif type == :ssic
        penalty = k * (log(n))^alpha
    else
        error("Unknown WBS penalty type: $type")
    end
    return base + (scale_penalty ? sigma2_null * penalty : penalty)
end

#------------------------

# RFPOP / raw-per-CP penalties (robseg package)
#
# RFPOP minimizes  Σ cost_t(y_t, θ_t) + λ · (#changes in θ).
# The raw penalty is added directly to the total (un-normalised) cost `loss`
# returned by the chosen loss function.  TCPD default λ values depend on the
# loss.  MICA uses the supplied loss_function; the formulas below add the TCPD
# default λ to whatever cost is provided.

#------------------------

"""
    compute_tcpd_rfpop_penalty(loss, n, num_cps; lambda)

RFPOP-style raw per-change penalty.  Returns `loss + lambda * num_cps`.

TCPD default λ values (from `robseg`):
- L1:     `log(n)`
- L2:     `log(n)`
- Huber:  `1.4 * log(n)`
- Outlier: `2 * log(n)`

Because MICA's `loss_function` is supplied by the caller, this function is
loss-agnostic: it adds `lambda * k` to the cost that was actually computed.
"""
function compute_tcpd_rfpop_penalty(loss::Real, n::Int, num_cps::Int;
                                    lambda::Real=2.0 * log(n))
    return loss + lambda * num_cps
end

#------------------------

# Example penalty functions (for objective_type = :penalty)

#------------------------

"""
default_penalty(p, n)

A basic penalty proportional to BIC.
"""
default_penalty(p, n) = 2 * p * log(n)

"""
BIC_penalty(p, n)

Bayesian Information Criterion-style penalty.
"""
BIC_penalty(p, n) = 100.0 * p * log(n)



