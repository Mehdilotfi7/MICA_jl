# Main comprehensive TCPD benchmark — continuity + NelderMead

## What this version tests
MICA on all 42 TCPD datasets with the full 27-model library, using continuous fits at changepoints and Nelder-Mead numerical optimization.

## Settings
- **Script:** `benchmark_tcpd_comprehensive.jl`
- **Optimizer:** Nelder-Mead via `default_model_setup(...; continuity=true)`; analytical fits are disabled because continuity is enabled.
- **Continuity:** `true` (C0 continuity enforced at changepoints).
- **Standardization:** Yes, z-score: `(y .- mean(y)) ./ std(y)`.
- **Penalties / objectives:** Built-in `BIC`, `MDL`, `AIC`. Penalty coefficient κ grid: `{0.1, 0.5, 1, 2, 5, 10, 20, 50, 100}` scaled by σ² of the standardized series.
- **Search grid:** `min_seg ∈ {1, 2, 3, max(3,n÷50), max(3,n÷30), max(5,n÷20), max(5,n÷10)}`; `step ∈ {1, max(1,n÷100), max(1,n÷50), max(1,n÷20)}`.
- **Oracle selection:** Best F1 across all (model, min_seg, step, κ, criterion) combinations per dataset.
- **Practical selection:** Minimum post-hoc BIC among the same combinations.
- **Model set:** 27 MICA models (mean, linear, quadratic, cubic, exponential, log_linear, power, mean_drift, ar1–ar3, hyperbolic, asymptotic_regression, michaelis_menten, weibull_growth, hill_function, log_logistic, double_exponential, rational, debt_dynamics, accelerator, compound_growth, poisson, negbin, ingarch, ets_aaa, ets_mmm).
- **Evaluation metric:** F1 / Covering with margin 5.

## Result files
- `benchmark_tcpd_comprehensive_numerical_all42.json` (early combined run)
- `benchmark_tcpd_comprehensive_numerical_continuity_all42.json` + `_oracle.json` + `_practical.json`
- `benchmark_tcpd_comprehensive_numerical_global_time_all42_oracle.json` + `_practical.json`
- `benchmark_tcpd_comprehensive_numerical_global_time_all_v2_oracle.json` + `_practical.json`
- `benchmark_tcpd_comprehensive_numerical_global_time_all_v3.json` + `_oracle.json` + `_practical.json`

Part files (`_partN.json`) are **not** kept here; they were merged into the combined files above.
