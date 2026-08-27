# MICA Domain-Specific Models Benchmark

**Date:** 2026-06-05  
**Datasets:** TCPD — 32 univariate series with annotations  
**Approach:** Assign per-dataset model families based on domain literature; run MICA binary segmentation unchanged.

---

## 1. Motivation

For each TCPD dataset, we identified the mathematical/statistical models that researchers in the respective field actually use (see user-provided domain analysis). The experiment tests whether restricting MICA's model library to these domain-relevant families improves changepoint detection accuracy.

**Key principle:** MICA's architecture is **not modified**. We only change:
1. The model library (added `gompertz`, `saturating_exponential`)
2. The per-dataset model configuration (`DOMAIN_SPECIFIC_MODELS`)

All parameters remain **segment-specific** (fitted independently per segment) — no continuous constraints or cross-segment parameter sharing.

---

## 2. Domain-Specific Model Assignments

| Domain | Dataset | Primary Model Family | Models Tested |
|--------|---------|---------------------|---------------|
| Financial | `apple`, `bitcoin`, `brent_spot`, `ratner_stock`, `usd_isk` | ARMA-GARCH, ARFIMA | `ar1`, `ar2`, `ar3`, `exponential`, `mean` |
| Financial | `bank` | Compound Poisson / jumps | `mean`, `piecewise_const`, `linear` |
| Financial | `shanghai_license` | M-ARMA / demand | `quadratic`, `cubic`, `exponential`, `saturating_exponential` |
| Macro | `gdp_*` | ARIMA, growth models | `exponential`, `power`, `log_linear`, `quadratic`, `cubic`, `linear` |
| Macro | `debt_ireland` | Debt dynamics | `exponential`, `power`, `quadratic`, `mean` |
| Macro | `uk_coal_employ` | Exponential decay | `exponential`, `power`, `quadratic`, `gompertz` |
| Macro | `unemployment_nl` | ARIMA, Beveridge curve | `ar1`, `ar2`, `quadratic`, `cubic`, `linear` |
| Macro | `construction`, `businv` | ARMA, accelerator | `ar1`, `ar2`, `linear`, `quadratic`, `exponential` |
| Macro | `rail_lines` | Logistic decline | `exponential`, `quadratic`, `gompertz` |
| Demographic | `us_population` | Logistic, Gompertz | `logistic`, `gompertz`, `exponential`, `power`, `cubic` |
| Demographic | `children_per_woman` | Logistic decline | `logistic`, `gompertz`, `exponential`, `quadratic` |
| Demographic | `centralia` | Exponential decay | `exponential`, `power`, `mean`, `linear` |
| Transport | `jfk_passengers`, `lga_passengers` | SARIMA / trend | `quadratic`, `cubic`, `exponential`, `saturating_exponential` |
| Environment | `co2_canada`, `global_co2` | ARIMA, trend breaks | `quadratic`, `cubic`, `exponential`, `power`, `log_linear` |
| Environment | `ozone` | Exponential decay | `exponential`, `quadratic`, `gompertz` |
| Environment | `well_log` | Bayesian CP | `mean`, `piecewise_const`, `linear` |
| Health | `seatbelts` | Interrupted time series | `mean`, `piecewise_const`, `mean_drift` |
| Sports | `homeruns` | Structural break | `mean`, `piecewise_const`, `quadratic` |
| Other | `run_log` | Step changes | `mean`, `piecewise_const`, `linear`, `exponential` |
| QC | `quality_control_*` | Mean shifts | `mean`, `piecewise_const`, `linear` |

---

## 3. Results Summary

### Oracle (best model + κ per dataset)
| Metric | Integrated Baseline | Domain-Specific | Δ |
|--------|---------------------|-----------------|---|
| Mean F1 | **0.9182** | **0.9182** | +0.0000 |
| Mean Cover | 0.9338 | 0.9338 | +0.0000 |
| Perfect F1 | 12/32 | 12/32 | — |

### Practical (BIC/AIC/MDL/AICc meta-selection)
| Metric | Integrated Baseline | Domain-Specific | Δ |
|--------|---------------------|-----------------|---|
| Mean F1 | 0.7100 | **0.7221** | **+0.0121** |
| Mean Cover | 0.9345 | 0.9382 | +0.0037 |
| Perfect F1 | 6/32 | 7/32 | +1 |

**Interpretation:** Oracle performance is identical because the integrated baseline already included the optimal model for each dataset. The domain-specific approach yields a modest practical improvement (+1.2 pp F1) by reducing model-selection noise.

---

## 4. Per-Dataset Breakdown

### Datasets where domain config improved practical F1

| Dataset | Base F1 → Domain F1 | Δ | Base Model → Domain Model | Explanation |
|---------|---------------------|---|---------------------------|-------------|
| `homeruns` | 0.491 → **0.818** | **+0.327** | `ar2` → `quadratic` | Domain config dropped AR models that overfit; quadratic captures regime changes in slugging percentage |
| `jfk_passengers` | 0.600 → **0.857** | **+0.257** | `quadratic_continuous` → `cubic` | Standard cubic trend outperformed continuous quadratic on passenger growth |
| `children_per_woman` | 0.581 → **0.649** | **+0.068** | `cubic` → `quadratic` | Quadratic better captures fertility decline trajectory |
| `usd_isk` | 0.672 → **0.790** | **+0.117** | `ar2` → `ar3` | AR(3) better models exchange-rate persistence |
| `run_log` | 0.754 → **0.870** | **+0.116** | `ar1` → `exponential` | Exponential better fits pace decay in training data |

### Datasets where domain config hurt practical F1

| Dataset | Base F1 → Domain F1 | Δ | Base Model → Domain Model | Explanation |
|---------|---------------------|---|---------------------------|-------------|
| `centralia` | 0.759 → **0.435** | **−0.324** | `log_linear` → `linear` | **Removed `log_linear` from domain config** — it was the best practical model. Log-linear trend was critical for this short declining series. |
| `construction` | 0.636 → **0.462** | **−0.175** | `linear` → `ar2` | Added AR models to domain config; AIC over-selected AR(2) which produced too many CPs. |

---

## 5. Key Findings

### 5.1 Model-family accuracy by domain

| Domain | Best-performing model | Worst-performing model |
|--------|----------------------|------------------------|
| Population / Growth | `exponential`, `cubic` | `logistic`, `gompertz` (bound failures on short/negative data) |
| Financial / AR | `ar1` | `ar3` (often overfits) |
| Quality Control | `mean` | `quadratic` (overfits) |
| Intervention / Steps | `mean`, `piecewise_const` | `linear` (misses abrupt shifts) |
| Transportation | `quadratic`, `cubic` | `exponential` (misses saturation) |

### 5.2 New models added

| Model | Use case | Status |
|-------|----------|--------|
| `gompertz` | Population growth / decline | ⚠️ Fails when `max(y) ≤ 0` (fixed in script) |
| `saturating_exponential` | Capacity-limited growth | ✅ Works on `jfk_passengers`, `lga_passengers`, `shanghai_license` but oracle still prefers polynomial |

### 5.3 Practical selection still struggles

Even with domain-restricted models, the practical (BIC/AIC/MDL/AICc) mean F1 (0.722) lags oracle (0.918) by **19.6 pp**. The gap is structural:
- Information criteria favor parsimony (fewer CPs)
- Many datasets have sparse or inconsistent annotator CPs
- Multi-criterion meta-selection is a partial workaround but not fully honest

### 5.4 Removing models can hurt

The `centralia` case is instructive: the integrated baseline included `log_linear`, which BIC selected for F1=0.759. The domain-specific config (focused on "exponential decay" literature) excluded `log_linear`, forcing BIC to choose `linear` (F1=0.435). **Domain priors are not always correct for changepoint detection.**

---

## 6. Remaining Gaps (Oracle losses)

These datasets still do not achieve perfect F1 even with oracle model selection:

| Dataset | Oracle F1 | Best Model | Issue |
|---------|-----------|------------|-------|
| `global_co2` | 0.750 | `log_linear` | Annotators disagree (some say 0 CPs, some say 2); model finds 5 CPs |
| `centralia` | 0.759 | `exponential` | Very short series (n=15); annotators disagree (0–3 CPs) |
| `construction` | 0.769 | `linear` | Annotators split between 0–2 CPs; model finds 1 CP |
| `homeruns` | 0.818 | `quadratic` | High annotator disagreement (0–7 CPs); model finds 6 CPs |
| `brent_spot` | 0.821 | `exponential` | Oil price volatility; 3–11 CPs across annotators; model finds 10 CPs |
| `quality_control_4` | 0.825 | `mean` | 0–4 CPs across annotators; model finds 4 CPs |
| `co2_canada` | 0.877 | `exponential` | 2–7 CPs across annotators; model finds 7 CPs |
| `businv` | 0.857 | `linear` | 0–3 CPs; model finds 2 CPs |
| `jfk_passengers` | 0.857 | `cubic` | 0–2 CPs; model finds 1 CP |
| `children_per_woman` | 0.857 | `quadratic` | 1–4 CPs; model finds 3 CPs |
| `gdp_argentina` | 0.857 | `log_linear` | 0–3 CPs; model finds 4 CPs |
| `lga_passengers` | 0.886 | `quadratic` | 0–7 CPs; model finds 3 CPs |

**Common theme:** Annotator disagreement is the dominant source of F1 loss, not model misspecification. When annotators disagree on CP count/location, no model can achieve perfect F1.

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `mica_domain_specific.py` | Domain-specific benchmark script |
| `tcpd_mica_domain_oracle.csv` | Oracle results (best model+κ per dataset) |
| `tcpd_mica_domain_practical.csv` | Practical results (IC meta-selection) |
| `tcpd_mica_domain_detailed.csv` | Per-model, per-κ detailed results |

---

## 8. Conclusions

1. **Domain-specific model configs do not improve oracle F1** when the integrated baseline already includes the relevant models. The integrated baseline (F1=0.918) is already near-optimal.

2. **Domain configs improve practical F1 modestly** (+1.2 pp) by reducing model-selection noise — but only when the right models are retained. Over-restriction (e.g., removing `log_linear` from `centralia`) can hurt.

3. **Annotator disagreement is the fundamental ceiling.** The largest F1 gaps (global_co2: 0.75, centralia: 0.76, construction: 0.77) occur where annotators themselves disagree on CP count.

4. **New models (`gompertz`, `saturating_exponential`) did not displace existing winners.** Polynomial trends (`quadratic`, `cubic`) and `exponential` remain the most reliable within MICA's segment-wise fitting framework.
