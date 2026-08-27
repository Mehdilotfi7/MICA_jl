# Domain-Aware Parameter Bounds and Initial Guesses for Mica.jl Models

**Date:** 2026-06-15  
**Scope:** Literature review and best-practice synthesis for the 27 model families used in the TCPD benchmark (`benchmark_tcpd_comprehensive.jl`).  
**Goal:** Replace the current “±big” / “10× data range” generic bounds with statistically principled, domain-aware defaults that improve optimizer convergence, reduce NaN/Inf in gradients, and give users a transparent starting point for their own bounds.

---

## 1. Executive Summary

The Mica.jl TCPD benchmark currently fits 27 univariate segment models inside a greedy changepoint search. Each candidate segment is short (often < 100 observations), and the same model is re-fit millions of times. In this regime the choice of parameter bounds and initial guesses is not a minor tuning detail—it determines whether the optimizer escapes invalid regions, whether the gradient is finite, and whether the search finishes in hours rather than days.

This document collects standard practice from the nonlinear-regression, time-series, and dose-response literature for every model family in the benchmark. The recommendations are conservative enough to be used as automatic defaults, but tight enough to keep the optimizer in the valid parameter region.

**Key cross-cutting principles**

| Principle | Rationale |
|-----------|-----------|
| **Box-bound every parameter** | Unbounded parameters invite overflow/underflow and NaN gradients (Bates & Watts, 1988; Nocedal & Wright). |
| **Keep the initial guess strictly inside bounds** | `Fminbox` and many metaheuristics fail or warn when `x0` sits on a bound. Nudge inward by ~1 % of the range (Gao, 2023; statsmodels ETS). |
| **Prefer interpretable bounds over data-scaled bounds** | e.g. `Vmax ∈ [0, 1.5·ymax]` and `Km ∈ [tmin, tmax]` for Michaelis–Menten are far more stable than `±big`. |
| **Use log- or logit-link parameterisations internally for positive / probability parameters** | σ, λ, α, β, γ, ω, shape/scale parameters are usually fit on the log scale (R `nlminb`; statsmodels; SciPy tutorials). |
| **Return a finite penalty instead of Inf/NaN** | A `safe_loss` wrapper is the standard robustness trick when the model can evaluate to invalid values (calibr8 docs; reliability package; Würtz, 2004). |
| **Use derivative-free optimisers for ill-conditioned or non-smooth losses** | Nelder–Mead and Particle Swarm are widely recommended when gradients are unreliable (Gao, 2023; our own tests confirm this). |

---

## 2. Model-by-Model Recommendations

### 2.1 Mean / linear / polynomial regression

**Models:** `mean`, `linear`, `quadratic`, `cubic`, `mean_drift`, `log_linear`  
**Standard formulation:**

```
mean:     y_t = μ
linear:   y_t = b·t + a
quadratic:y_t = a·t² + b·t + c
cubic:    y_t = a·t³ + b·t² + c·t + d
mean_drift: y_t = μ + a·t + b
log_linear: y_t = a·log(t) + b
```

**Bounds and initial guesses**

| Model | Parameters | Initial guess | Lower bound | Upper bound | Source |
|-------|------------|---------------|-------------|-------------|--------|
| `mean` | μ | `mean(y)` | `y_min - 3σ` | `y_max + 3σ` | OLS closed form |
| `linear` | b, a | `cov(t,y)/var(t)`, `mean(y)-b·mean(t)` | `[-s, y_min-3σ]` | `[s, y_max+3σ]` | OLS closed form; `s = 20·|y_range|/n` |
| `quadratic`/`cubic` | polynomial coeffs | OLS on `t^k` | `[-B, ..., y_min-3σ]` | `[B, ..., y_max+3σ]` | OLS; `B = 5·max(|y_range|)/n^k` scaled per power |
| `log_linear` | a, b | OLS on `log(t)` | `[-A, y_min-3σ]` | `[A, y_max+3σ]` | OLS; `A ≈ 5·|y_range|` |
| `mean_drift` | μ, a, b | OLS + residual mean | `[-R, -s, y_min-3σ]` | `[R, s, y_max+3σ]` | OLS; `R = 3σ`, `s = 20·|y_range|/n` |

**Notes from literature**
- Polynomial coefficients are famously ill-conditioned for high degree; the benchmark only goes to cubic, which is safe if time is mean-centred or scaled to `[0,1]` (Draper & Smith, 1998).
- GraphPad Prism recommends avoiding raw values > 10⁵ or < 10⁻⁵; scaling `t` to `[1,n]` or `[0,1]` is good practice.
- For the `mean_drift` model the intercept `μ` is not separately identifiable from `b`; consider collapsing to `linear` or constraining `μ ∈ [-σ, σ]`.

**Recommended improvement for Mica.jl:**  
Use OLS-derived closed-form initial guesses and replace the generic `±big` with bounds scaled by the range of the *predictor-transformed* design matrix. Use `big = max(10, 5·|y_range| / n^k)` for the `k`-th polynomial coefficient rather than `10·max(|y|)`.

---

### 2.2 Exponential and power models

**Models:** `exponential`, `power`  
**Standard formulation:**

```
exponential: y_t = a·exp(b·t)
power:       y_t = a·t^b
```

**Bounds and initial guesses**

| Model | Parameters | Initial guess | Lower bound | Upper bound | Source |
|-------|------------|---------------|-------------|-------------|--------|
| `exponential` | a, b | `a = exp(c)` from linearised `log(y+off) ~ b·t + c`; `b = slope` | `[0.01, -5]` | `[A, 5]` | Linearisation + positivity of `a` (SciPy `curve_fit` tutorials; lmfit docs) |
| `power` | a, b | `a = exp(c)` from `log(y+off) ~ b·log(t) + c`; `b = slope` | `[0.01, -5]` | `[A, 5]` | Linearisation + positivity of `a` |

**Notes from literature**
- The amplitude `a` must be positive for the log-linearisation to be valid. If data can be negative, add an offset `off = |min(y)| + 1` before linearisation and fit `y_t + off = a·exp(b·t)`.
- `b` is usually in `[-2, 2]` for real-world growth/decay; wider bounds are rarely needed (Bates & Watts, 1988).
- For very short segments, linearisation gives a much better `x0` than midpoint-of-bounds.

**Recommended improvement for Mica.jl:**  
Keep the linearised initial guess but tighten bounds to `[0.01, 10·ymax]` for `a` and `[-5, 5]` for `b` (or `[-2, 2]` as a conservative default). Use `safe_exp` already present in `ModelSimulation.jl`.

---

### 2.3 Michaelis–Menten

**Model:** `michaelis_menten`  
**Standard formulation:**

```
y_t = Vmax · t / (Km + t)
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| Vmax | `1.05·ymax` or `ymax + 0.1·|y_range|` | `0.0` (or `0.01`) | `1.5·ymax` | Plotly enzyme-kinetics guide; OriginLab NLSF defaults |
| Km | substrate concentration at `y ≈ ymax/2` | `0.0` (or `0.01`) | `tmax` (or `5·tmax`) | Plotly; nlstools; Cornell QBIO notes |

**Notes from literature**
- Both parameters are non-negative by physical meaning. Plotly and nlstools use `(0, +∞)`; in practice an upper bound of `tmax` or a small multiple of it prevents the optimizer from wandering into numerically flat regions.
- A good `Km` guess is the `t` value closest to `ymax/2` after sorting by `t` (Plotly, 2025).
- Lineweaver–Burk linearisation can give a starting value, but direct nonlinear least squares from the physical guess is preferred (Cornell QBIO).

**Recommended improvement for Mica.jl:**  
`Vmax ∈ [0.01, 1.5·ymax]`, `Km ∈ [0.01, tmax]` (or `2·tmax` to allow extrapolation). Initial guess: `Vmax = ymax`, `Km = t[argmin(|y - ymax/2|)]`.

---

### 2.4 Hill function

**Model:** `hill_function`  
**Standard formulation:**

```
y_t = a · t^n / (k^n + t^n)
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| a (Vmax) | `ymax` | `0.0` | `1.5·ymax` | OriginLab HillCOEFF; CDD dose-response notes |
| k (EC50/IC50) | `t` at `y ≈ ymax/2` | `0.0` | `tmax` | OriginLab; CDD Support |
| n (Hill slope) | `1.8` or `2.0` | `0.1` | `10.0` | OriginLab; CDD Support; note singularities at origin for non-integer `n` |

**Notes from literature**
- CDD Support (2026) recommends allowing the Hill slope to float but constraining it to `≥0` for inhibition assays or `=1` for the standard slope model.
- Recent work (arXiv 2512.14325) warns that Hill functions with non-integer `n` are only `C^⌊n⌋` at zero, causing gradient/Hessian singularities. For robust automatic fitting, restrict `n` to `[0.5, 5]` or even `[0.1, 5]` and use a derivative-free optimiser.
- OriginLab’s built-in Hill initialiser sets `n` to the average of local estimates around the half-max point.

**Recommended improvement for Mica.jl:**  
`a ∈ [0.01, 1.5·ymax]`, `k ∈ [0.01, tmax]`, `n ∈ [0.1, 5.0]`. Initial guess: `a = ymax`, `k = t[argmin(|y - ymax/2|)]`, `n = 1.5`. Consider clamping `t` away from zero to avoid the `0^n` singularity.

---

### 2.5 Log-logistic

**Model:** `log_logistic`  
**Standard formulation:**

```
y_t = a / (1 + (t/b)^c)
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| a (upper asymptote) | `ymax` | `0.0` | `1.5·ymax` | Psychometric/4PL curve conventions (quickpsy) |
| b (inflection / ED50) | `t` at `ymax/2` | `0.0` | `tmax` | quickpsy; CDD dose-response |
| c (shape / slope) | `2.0` | `0.1` | `10.0` | quickpsy defaults |

**Notes from literature**
- The log-logistic / 4-parameter logistic is the standard dose-response model. `a` and `b` are non-negative; `c` controls steepness and is typically in `[0.5, 5]`.
- quickpsy (Linares et al.) uses probit-linear initialisation for psychometric functions, but for time-series-like data the half-max initialisation is adequate.

**Recommended improvement for Mica.jl:**  
`a ∈ [0.01, 1.5·ymax]`, `b ∈ [0.01, tmax]`, `c ∈ [0.1, 5.0]`. Initial guess: `a = ymax`, `b = t[argmin(|y - ymax/2|)]`, `c = 2.0`.

---

### 2.6 Weibull growth

**Model:** `weibull_growth`  
**Standard formulation:**

```
y_t = a - b · exp(-c · t^d)
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| a (asymptote) | `ymax` or `mean(y)` | `y_min` | `ymax + 0.5·|y_range|` | Reliability / growth-curve fitting |
| b (range) | `ymax - y_min` | `0.0` | `2·|y_range|` | Interpreted as the total growth range |
| c (scale) | `0.1` or `1/tmax` | `0.0` (or `1e-4`) | `5.0` | SciPy Weibull MLE tutorial |
| d (shape) | `1.0` | `0.1` | `5.0` | SciPy reliability docs; typical Weibull shape bounds `[1e-10, 10]` |

**Notes from literature**
- Weibull shape/scale parameters must be positive. SciPy’s `weibull_min.fit` tutorial uses lower bounds of `1e-10` and often upper bounds of `10` for shape and `10` for scale (SciPy 1.15 docs).
- The reliability package recommends `bounds=(0, None)` for Weibull parameters and reports that bounds “help a lot with stability”.

**Recommended improvement for Mica.jl:**  
`a ∈ [y_min, ymax + 0.5·|y_range|]`, `b ∈ [0.01, 2·|y_range|]`, `c ∈ [1e-4, 5.0]`, `d ∈ [0.1, 5.0]`. Initial guess: `a = ymax`, `b = ymax - y_min`, `c = 1/tmax`, `d = 1.0`.

---

### 2.7 Double exponential

**Model:** `double_exponential`  
**Standard formulation (in Mica.jl):**

```
y_t = a·exp(b·t) + c·exp(d·t)
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| a, c (amplitudes) | OLS in `exp(b·t)`, `exp(d·t)` with fixed `b,d` | `[-A, A]` | `[A, A]` | Linear-amplitude / nonlinear-rate formulation common in pharmacokinetics |
| b, d (rates) | `-0.1`, `-1.0` (or fit from linearised residuals) | `[-5, -5]` | `[5, 5]` | Bates & Watts; sum-of-exponentials is ill-conditioned so rates need ordering |

**Notes from literature**
- Sum-of-exponentials is a classic ill-conditioned problem. The standard remedy is to enforce `b < d` (or `b > d`) and use a partly linear algorithm (Golub–Pereyra / `nls(..., algorithm="plinear")`) so that `a,c` are profiled out (Bates & Watts, 1988; R `nls` docs).
- If fitted directly, use bounds that keep rates away from zero (e.g. `|b|,|d| ∈ [1e-3, 5]`) and order them to avoid label switching.

**Recommended improvement for Mica.jl:**  
Profile out `a,c` given `(b,d)` (linear least squares) and optimise only two rates. Bounds: `b ∈ [-5, -1e-3]`, `d ∈ [1e-3, 5]` (or vice versa, enforcing `b < d`). Initial guess: `b = -0.1`, `d = -1.0`. This reduces dimensionality from 4 to 2 and removes amplitude ill-conditioning.

---

### 2.8 Rational function

**Model:** `rational`  
**Standard formulation:**

```
y_t = (a·t + b) / (c·t + d)
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| a, b | `0`, `mean(y)` | `[-R, -R]` | `[R, R]` | Linear-rational initialisation |
| c, d | `1`, `1` | `[ε, ε]` or `[-R, ε]` | `[R, R]` | Denominator must stay positive; protect with `max(denom, 1e-6)` already in code |

**Notes from literature**
- Rational functions are notoriously unstable near poles. The denominator `c·t + d` must not cross zero inside `[1, n]`. The safest default is `c ≥ 0`, `d ≥ ε` or to reparameterise as `denom = exp(c)·t + exp(d)` (Bates & Watts; NIST nonlinear regression examples).
- If both signs are allowed, bound `|c|/|d|` away from `1/t` for all `t` in the segment, which is hard to do with simple box bounds.

**Recommended improvement for Mica.jl:**  
Reparameterise to enforce a positive denominator: `c = exp(c_raw)`, `d = exp(d_raw)` with `c_raw, d_raw ∈ [-5, 5]`. Alternatively, keep the current `max(denom, 1e-6)` protection and use bounds `c,d ∈ [1e-4, R]` with `R = 10/|y_range|`. Initial guess: `a = 0`, `b = mean(y)`, `c = 1/n`, `d = 1`.

---

### 2.9 Asymptotic regression / hyperbolic

**Models:** `asymptotic_regression`, `hyperbolic`  
**Standard formulation:**

```
asymptotic_regression: y_t = a + (b - a)·exp(-c·t)
hyperbolic:            y_t = a + b / t
```

**Bounds and initial guesses**

| Model | Parameters | Initial guess | Lower bound | Upper bound | Source |
|-------|------------|---------------|-------------|-------------|--------|
| `asymptotic_regression` | a, b, c | `a=ymax`, `b=y_min`, `c=0.1` | `[y_min-3σ, 0, 1e-4]` | `[ymax+3σ, big, 5.0]` | calibr8 asymmetric-logistic notes; Moffat/4PL conventions |
| `hyperbolic` | a, b | `a=mean(y)`, `b=0` | `[-R, -R]` | `[R, R]` | Simple rational form; keep `b` bounded to avoid pole at `t=0` |

**Recommended improvement for Mica.jl:**  
For `asymptotic_regression`, `a` is the upper asymptote, `b` the lower intercept, and `c` the rate; current bounds are reasonable but `c` should be positive and capped at ~5. For `hyperbolic`, add a small regularisation to keep `b/t` finite near `t=1`.

---

### 2.10 Autoregressive models (AR1–AR3)

**Models:** `ar1`, `ar2`, `ar3`  
**Standard formulation:**

```
AR(1): y_t = c + φ1·y_{t-1} + ε_t
AR(2): y_t = c + φ1·y_{t-1} + φ2·y_{t-2} + ε_t
AR(3): y_t = c + φ1·y_{t-1} + φ2·y_{t-2} + φ3·y_{t-3} + ε_t
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| φ1, φ2, φ3 | `0` or Yule–Walker estimates | `[-0.99, -0.99, -0.99]` | `[0.99, 0.99, 0.99]` | Cryer & Chan (2008); stationarity triangle for AR(2) |
| c | `mean(y)·(1 - Σφ)` | `y_min - 5σ` | `ymax + 5σ` | AR unconditional mean `c/(1-Σφ)` |

**Notes from literature**
- For stationarity, AR coefficients must lie inside the unit-root region. For AR(2) this is the triangle `φ2 > φ1 - 1`, `φ2 > -φ1 - 1`, `φ2 < 1`. Simple box bounds `[-0.99, 0.99]` are a conservative superset (Cryer & Chan).
- Yule–Walker estimates from the sample autocorrelations give much better starting values than zeros, especially for short segments.

**Recommended improvement for Mica.jl:**  
Switch from GA to derivative-free bounded Nelder–Mead (or `Fminbox(LBFGS())` with safe loss). Use Yule–Walker initial guesses. Add explicit stationarity constraints for AR(2)/AR(3) if possible, or at least tighten bounds to the empirical PACF-driven ranges.

---

### 2.11 Difference / growth models

**Models:** `debt_dynamics`, `accelerator`, `compound_growth`  
**Standard formulation:**

```
debt_dynamics:    y_t = (1+r)·y_{t-1} + s
accelerator:      y_t = c + v·(y_{t-1} - y_{t-2})
compound_growth:  y_t = (1+r)·y_{t-1}
```

**Bounds and initial guesses**

| Model | Parameters | Initial guess | Lower bound | Upper bound | Source |
|-------|------------|---------------|-------------|-------------|--------|
| `debt_dynamics` | r, s | `r=0.05`, `s=mean(y)·0.05` | `[-0.5, -R]` | `[0.5, R]` | Economic debt dynamics; `r` is an interest/growth rate |
| `accelerator` | c, v | `c=mean(y)`, `v=0.5` | `[y_min-3σ, -5]` | `[ymax+3σ, 5]` | Macro-economic accelerator; `v` is the accelerator coefficient |
| `compound_growth` | r | `std(diff(y))/|mean(y)|` | `[-0.5]` | `[0.5]` | Per-period growth rate |

**Notes from literature**
- `r` in `debt_dynamics` and `compound_growth` is a per-period rate; `[-0.5, 0.5]` is very wide but safe for quarterly/annual data. For monthly data a tighter `[-0.1, 0.1]` may be better.
- `s` is a flow in the same units as `y`; bound by the data range rather than `±big`.

**Recommended improvement for Mica.jl:**  
`r ∈ [-0.5, 0.5]` (or `[-0.1, 0.1]` if the series is not heavily differenced). `s ∈ [-5σ, 5σ]` instead of `±big`. `v ∈ [-2, 2]` for accelerator.

---

### 2.12 Count models

**Models:** `poisson`, `negbin`, `ingarch`  
**Standard formulation:**

```
poisson:   λ_t = exp(a + b·t)
negbin:    μ_t = exp(a + b·t), overdispersion parameter
ingarch:   λ_t = ω + α·y_{t-1} + β·λ_{t-1}
```

**Bounds and initial guesses**

| Model | Parameters | Initial guess | Lower bound | Upper bound | Source |
|-------|------------|---------------|-------------|-------------|--------|
| `poisson` | a, b | `a=log(max(mean(y),1))`, `b=0` | `[-5, -0.5]` | `[5, 0.5]` | GLM IRLS; log-link keeps λ positive (Brenndoerfer, 2025; SAS/IML) |
| `negbin` | a, b, (dispersion) | same as Poisson + small dispersion | `[-5, -0.5, 1e-4]` | `[5, 0.5, 10]` | NB2 / NB1 dispersion > 0 |
| `ingarch` | ω, α, β | `ω=max(mean(y)·0.1,0.1)`, `α=β=0.2` | `[1e-4, 0, 0]` | `[big, 0.99, 0.99]` | CountTimeSeries package; stationarity requires `α+β < 1` |

**Notes from literature**
- Poisson regression is almost always fit via log-link, so the linear predictor `a + b·t` can be unconstrained while `λ` stays positive. Current bounds `[-5,5]` and `[-0.5,0.5]` are reasonable for standardised data (Brenndoerfer, 2025).
- INGARCH stationarity requires `α + β < 1` (Ferland et al.; CountTimeSeries paper). An explicit linear constraint should be added; if not possible, use bounds `α,β ∈ [0, 0.99]` and a penalty term `safe_loss` that blows up when `α+β ≥ 1`.

**Recommended improvement for Mica.jl:**  
Keep log-link parameterisation. For INGARCH add a stationarity penalty or constraint. Tighten `ω` to `[1e-4, 5·mean(y)]`.

---

### 2.13 ETS (exponential smoothing) models

**Models:** `ets_aaa`, `ets_mmm`  
**Standard formulation:**

```
ETS(A,A,A):  additive errors, additive trend, additive seasonality
ETS(M,M,M):  multiplicative errors, multiplicative trend, multiplicative seasonality
```

**Bounds and initial guesses**

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| l0 (initial level) | `y[1]` | `[y_min-3σ]` or `[0.01]` | `[ymax+3σ]` | statsmodels ETS; Hyndman & Athanasopoulos (2018) |
| b0 (initial trend) | `(y[2]-y[1])` or OLS slope | `[-2·|y_range|, -R]` | `[2·|y_range|, R]` | Minitab; statsmodels |
| α (level smoothing) | `0.1` | `[1e-4]` | `[0.9999]` | forecast::ets default `[1e-4, 0.9999]` |
| β (trend smoothing) | `0.01` | `[1e-4]` | `[α]` or `[0.9999]` | Hyndman: `0 < β < α`; modeltime/forecast allow `[1e-4, 0.9999]` |
| γ (seasonal smoothing) | `0.01` | `[1e-4]` | `[1-α]` or `[0.9999]` | Hyndman: `0 < γ < 1-α` |

**Notes from literature**
- forecast::ets uses `lower=c(rep(1e-4,3), 0.8)` and `upper=c(rep(0.9999,3), 0.98)` for α,β,γ,φ (modeltime docs).
- statsmodels ETS defaults to “traditional (nonlinear) bounds” but allows per-parameter dictionary bounds.
- Bayesian ETS work (arXiv 2309.13950) uses Beta priors on α,β,γ in `(0,1)` and notes that admissible regions can be broader, but traditional bounds are safer for automatic fitting.

**Recommended improvement for Mica.jl:**  
Use `α,β,γ ∈ [1e-4, 0.9999]`. Add a soft constraint or penalty so that `β ≤ α` and `γ ≤ 1-α` for AAA. For MMM, l0 and b0 must be positive; bound them away from zero. Use the OLS/Heuristic initialisation from statsmodels rather than raw first differences.

---

### 2.14 GARCH / EGARCH / TGARCH (if added later)

Although not in the current 27-model benchmark, volatility models are natural extensions. The literature gives very stable defaults:

| Parameter | Initial guess | Lower bound | Upper bound | Source |
|-----------|---------------|-------------|-------------|--------|
| μ | `mean(y)` | `[-10·|mean|, 10·|mean|]` or unconstrained | Gao (2023); Würtz (2004) |
| ω | `0.1·var(y)` | `[1e-6]` | `[100·var(y)]` | fGarch; rugarch |
| α | `0.1` | `[0]` or `[1e-6]` | `[1-1e-6]` | fGarch |
| β | `0.8` | `[0]` or `[1e-6]` | `[1-1e-6]` | fGarch |
| γ (leverage) | `0.0` | `[-0.99]` | `[0.99]` | EGARCH/GJR-GARCH |

**Stationarity / positivity**
- Standard GARCH(1,1): `ω > 0`, `α,β ≥ 0`, `α+β < 1` (strict stationarity) or `≤ 1` (weak stationarity).
- EGARCH: no positivity constraint on ω,α,β,γ because variance is modelled in logs (Nelson, 1991).

---

## 3. Recommended Default Bounds Table for Mica.jl

The table below is designed to be dropped into `build_model` in `benchmark_tcpd_comprehensive.jl` (or into a future `Mica.jl` model registry).  
Notation:
- `ymax, y_min, y_range = ymax - y_min, σ = std(y), tmax = n`
- `R_y = max(abs(y_min), abs(ymax))`
- `A = 1.5·ymax` for positive-amplitude models
- `S = 20·|y_range|/n` for slope-like parameters

| Model | Params | Initial guess | Lower bounds | Upper bounds | Notes |
|-------|--------|---------------|--------------|--------------|-------|
| `mean` | μ | `mean(y)` | `y_min-3σ` | `ymax+3σ` | |
| `linear` | b,a | OLS | `[-S, y_min-3σ]` | `[S, ymax+3σ]` | |
| `quadratic` | a,b,c | OLS on `t²,t,1` | `[-B2, -S, y_min-3σ]` | `[B2, S, ymax+3σ]` | `B2 = 5·|y_range|/n²` |
| `cubic` | a,b,c,d | OLS on `t³,t²,t,1` | scaled per power | scaled per power | consider centring `t` |
| `exponential` | a,b | log-linearisation | `[0.01, -5]` | `[10·ymax, 5]` | use `safe_exp` |
| `power` | a,b | log-log-linearisation | `[0.01, -5]` | `[10·ymax, 5]` | |
| `log_linear` | a,b | OLS on `log(t)` | `[-5·|y_range|, y_min-3σ]` | `[5·|y_range|, ymax+3σ]` | |
| `mean_drift` | μ,a,b | OLS+residual mean | `[-3σ, -S, y_min-3σ]` | `[3σ, S, ymax+3σ]` | `μ` may be redundant |
| `hyperbolic` | a,b | `mean(y), 0` | `[-5·|y_range|, -5·|y_range|]` | `[5·|y_range|, 5·|y_range|]` | protect `t=0` |
| `asymptotic_regression` | a,b,c | `ymax, y_min, 0.1` | `[y_min-3σ, 0, 1e-4]` | `[ymax+3σ, 2·|y_range|, 5]` | |
| `michaelis_menten` | Vmax,Km | `ymax, t[argmin(|y-ymax/2|)]` | `[0.01, 0.01]` | `[1.5·ymax, tmax]` | |
| `weibull_growth` | a,b,c,d | `ymax, y_range, 1/tmax, 1` | `[y_min, 0.01, 1e-4, 0.1]` | `[ymax+0.5·|y_range|, 2·|y_range|, 5, 5]` | |
| `hill_function` | a,k,n | `ymax, t[argmin(|y-ymax/2|)], 1.5` | `[0.01, 0.01, 0.1]` | `[1.5·ymax, tmax, 5]` | clamp `t` away from 0 |
| `log_logistic` | a,b,c | `ymax, t[argmin(|y-ymax/2|)], 2` | `[0.01, 0.01, 0.1]` | `[1.5·ymax, tmax, 5]` | |
| `double_exponential` | a,b,c,d | profile `a,c`; optimise `b,d` | `b ∈ [-5,-1e-3], d ∈ [1e-3,5]` | enforce `b < d` | 2-D optimisation only |
| `rational` | a,b,c,d | `0, mean(y), 1/n, 1` | `[-R, -R, 1e-4, 1e-4]` | `[R, R, R, R]` | positive-denom reparam. preferred |
| `ar1` | φ1,c | Yule–Walker | `[-0.99, y_min-5σ]` | `[0.99, ymax+5σ]` | |
| `ar2` | φ1,φ2,c | Yule–Walker + stationarity triangle | box + triangle | box + triangle | derivative-free optimiser |
| `ar3` | φ1,φ2,φ3,c | Yule–Walker | `[-0.99, -0.99, -0.99, ...]` | `[0.99, ...]` | |
| `debt_dynamics` | r,s | `0.05, mean(y)·0.05` | `[-0.5, -5·|y_range|]` | `[0.5, 5·|y_range|]` | |
| `accelerator` | c,v | `mean(y), 0.5` | `[y_min-3σ, -2]` | `[ymax+3σ, 2]` | |
| `compound_growth` | r | `std(diff(y))/|mean(y)|` | `[-0.5]` | `[0.5]` | |
| `poisson` | a,b | `log(max(mean(y),1)), 0` | `[-5, -0.5]` | `[5, 0.5]` | log-link |
| `negbin` | a,b,disp | Poisson init + disp | `[-5, -0.5, 1e-4]` | `[5, 0.5, 10]` | |
| `ingarch` | ω,α,β | `max(mean·0.1,0.1), 0.2, 0.2` | `[1e-4, 0, 0]` | `[5·mean(y), 0.99, 0.99]` | enforce `α+β < 1` |
| `ets_aaa` | l0,b0,α,β,γ | `y[1], diff(y)[1], 0.1, 0.01, 0.01` | `[y_min-3σ, -2·|y_range|, 1e-4, 1e-4, 1e-4]` | `[ymax+3σ, 2·|y_range|, 0.9999, 0.9999, 0.9999]` | soft `β≤α`, `γ≤1-α` |
| `ets_mmm` | l0,b0,α,β,γ | `max(y[1],0.1), clamp(diff/y,±2), 0.1, 0.01, 0.01` | `[0.01, -2, 1e-4, 1e-4, 1e-4]` | `[ymax+3σ, 2, 0.9999, 0.9999, 0.9999]` | positive level |

---

## 4. Optimiser and Loss-Function Recommendations

### 4.1 Optimiser choice

Our own tests show `NelderMead` and `ParticleSwarm` are the fastest and most reliable on the current benchmark. This matches the wider literature:

- **Nelder–Mead** is derivative-free and ignores bounds, so it does not suffer from NaN gradients at boundaries. It is the recommended fallback in fGarch (`nlminb+nm`) and in many scientific Python workflows.
- **L-BFGS-B / Fminbox** is fast when the loss is smooth and the initial guess is good. Use it *with* a `safe_loss` wrapper and with `x0` nudged inward.
- **GA / ECA** are global metaheuristics. They are much slower per call and should be reserved for models with many local minima (e.g. `double_exponential`). ECA currently fails with `MethodError` and should be fixed or removed from defaults.

**Practical default:**
- Use `NelderMead()` as the universal default for the full benchmark.
- Use `Fminbox(LBFGS())` only for analytically well-behaved models (`mean`, `linear`, `quadratic`, `cubic`, `log_linear`).

### 4.2 Safe loss wrapper

A robust objective should never return `Inf` or `NaN`. Standard pattern (reliability package; calibr8; Gao 2023):

```julia
function safe_loss(loss_fn, obs, sim; bad=1e12)
    l = try
        loss_fn(obs, sim)
    catch
        bad
    end
    if !isfinite(l)
        return bad
    end
    return clamp(l, -bad, bad)
end
```

For likelihood-based losses (Poisson, NB, GARCH), replace the `bad` value with a large finite negative log-likelihood.

### 4.3 Model-aware loss functions

The benchmark currently uses RSS for every model. Literature-backed alternatives:

| Model class | Recommended loss | When to use |
|-------------|------------------|-------------|
| Gaussian / regression | L2 (RSS), L1, Huber | Default; Huber for outliers |
| Count data | Poisson NLL, Negative-Binomial NLL | Overdispersion |
| Volatility | Gaussian NLL on conditional variance | GARCH family |
| Heavy-tailed errors | Student-t NLL | Financial / noisy data |
| Robust generic | L1, Huber, Tukey biweight | Outliers |

**User-defined loss support:**  
Expose the loss function in the public API, e.g.:

```julia
detect_changepoints(..., loss_function=huber_loss)
```

and provide a small library of built-ins:

```julia
rss_loss(obs, sim) = sum((obs .- sim).^2)
l1_loss(obs, sim) = sum(abs.(obs .- sim))
huber_loss(obs, sim; δ=1.345) = ...
poisson_nll(obs, sim) = sum(sim .- obs.*log.(sim .+ eps))
```

---

## 5. Implementation Plan for Mica.jl

1. **Add a `safe_loss` utility** in `Mica.jl/src/ObjectiveFunction.jl` and apply it inside `detect_changepoints` before calling the optimiser.
2. **Create a model registry** (`src/model_defaults.jl`) mapping each model name to:
   - parameter names,
   - default bounds (lower/upper),
   - default initial-guess heuristic,
   - recommended loss function,
   - recommended optimiser.
3. **Update `build_model`** in `benchmark_tcpd_comprehensive.jl` to use the registry, with optional user overrides:
   ```julia
   build_model(y, n, "hill_function"; bounds=bounds, initial_guess=x0, loss=huber_loss)
   ```
4. **Switch the default optimiser** to `OptimOptimizer(NelderMead())` for the full TCPD benchmark; keep `Fminbox(LBFGS())` only for the OLS-analytical models.
5. **Add explicit constraints/penalties** where simple box bounds are insufficient:
   - AR(2)/AR(3) stationarity triangle,
   - INGARCH `α+β < 1`,
   - ETS `β ≤ α`, `γ ≤ 1-α`.
6. **Re-run a small calibration benchmark** on 3–5 datasets to validate that the new defaults reduce runtime and eliminate NaN-gradient failures before launching the full 31-dataset run.

---

## 6. References

1. **Bates, D. M., & Watts, D. G. (1988).** *Nonlinear Regression Analysis and Its Applications*. Wiley. (General nonlinear regression bounds/initial guesses; partially linear algorithms.)
2. **Brenndoerfer, M. (2025).** “Poisson Regression: Complete Guide to Count Data Modeling.” https://mbrenndoerfer.com/writing/poisson-regression-complete-guide-count-data-modeling-mathematical-foundations-python-implementation
3. **CDD Support (2026).** “Hill Equation: Setting up a dose-response curve.” https://support.collaborativedrug.com/hc/en-us/articles/35945212110868
4. **Cornell QBIO (2008).** “Quantitative Comparison of Models.” https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2008/2.4_Quantitative_Comparison_of_Models.pdf
5. **Cryer, J. D., & Chan, K.-S. (2008).** *Time Series Analysis with Applications in R*. Springer. (AR bounds/stationarity.)
6. **Gao, M. (2023).** “GARCH Estimation.” https://mingze-gao.com/posts/garch-estimation/
7. **Gardner, E. S. (2005).** “Exponential smoothing: The state of the art – Part II.” *International Journal of Forecasting*.
8. **Hyndman, R. J., & Athanasopoulos, G. (2018).** *Forecasting: Principles and Practice*, 3rd ed. OTexts. (ETS bounds.)
9. **Hyndman et al. (2008).** *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
10. **Lancaster MATH337 notes.** “PELT, WBS and Penalty choices.” https://www.lancaster.ac.uk/~romano/teaching/2425MATH337/4_algos_and_penalties.html
11. **Linares, D., et al. (quickpsy).** R package for psychometric functions. https://cran.r-project.org/web/packages/quickpsy/ (Log-logistic / Weibull psychometric bounds.)
12. **lmfit-py documentation.** “Built-in Fitting Models.” https://lmfit.github.io/lmfit-py/builtin_models.html (Power-law, exponential, peak-model defaults.)
13. **Maidstone, R. (2016).** PhD thesis, Lancaster. “Choice of Penalty in the Penalised Minimisation Problem.”
14. **MATLAB / MathWorks.** “Fit Custom Distributions.” https://www.mathworks.com/help/stats/fitting-custom-univariate-distributions.html
15. **Minitab.** “Methods and formulas for Double Exponential Smoothing.” https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/how-to/double-exponential-smoothing/methods-and-formulas/methods-and-formulas/
16. **NIST Engineering Statistics Handbook.** “Double Exponential Smoothing.” https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc433.htm
17. **Nocedal, J., & Wright, S. J.** *Numerical Optimization*. Springer. (Box constraints, initial guesses, scaling.)
18. **OriginLab forum (2004).** “Hill coefficient.” https://my.originlab.com/forum/topic.asp?TOPIC_ID=3167
19. **Plotly (2025).** “Michaelis-Menten Fitting in Python for Enzyme Kinetics.” https://plotivy.app/techniques/michaelis-menten-fitting
20. **Rummel, N., et al. (2025).** WENDy for nonlinear-in-parameter ODEs. arXiv:2502.08881.
21. **SciPy documentation (1.15).** “Random Variable Transition Guide — Maximum Likelihood Estimation.” https://docs.scipy.org/doc/scipy-1.15.2/tutorial/stats/rv_infrastructure.html (Weibull bounds.)
22. **statsmodels.** `tsa.exponential_smoothing.ets` documentation. https://www.statsmodels.org/dev/_modules/statsmodels/tsa/exponential_smoothing/ets.html
23. **Würtz, D., et al. (2004).** “ARMA Models with GARCH/APARCH Errors.” https://www.math.pku.edu.cn/teachers/heyb/TimeSeries/lectures/garch.pdf
24. **Zhang, N. R., & Siegmund, D. O. (2007).** “A modified Bayes information criterion with applications to the analysis of comparative genomic hybridization data.” *Biometrics*.

---

## 7. Quick-Reference Checklist

- [ ] Replace `big = max(abs(y_min), abs(y_max)) * 10` with parameter-specific ranges.
- [ ] Nudge every initial guess at least 1 % inside its bounds.
- [ ] Add `safe_loss` so no optimizer sees `Inf`/`NaN`.
- [ ] Default optimiser: `NelderMead()`; keep `Fminbox(LBFGS())` for polynomial models.
- [ ] Use log-link for count models; enforce `α+β < 1` for INGARCH.
- [ ] Use Yule–Walker starts and stationarity-aware bounds for AR models.
- [ ] Constrain ETS smoothing parameters to `(0,1)` with `β ≤ α` and `γ ≤ 1-α`.
- [ ] Reparameterise `rational` to keep denominator positive.
- [ ] Profile amplitudes in `double_exponential` to reduce to a 2-D rate problem.
- [ ] Expose `loss_function` keyword to users.
