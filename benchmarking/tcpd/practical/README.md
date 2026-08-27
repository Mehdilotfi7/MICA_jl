# Practical TCPD exact defaults

## What this version tests
MICA run with the exact default penalty settings used by the reference packages in the TCPD benchmark, so each MICA-P row corresponds to a specific competitor method's default criterion.

## Settings
- **Script:** `benchmark_tcpd_practical_tcpd_exact.jl`
- **Optimizer:** Nelder-Mead via `default_model_setup(...; continuity=true)`.
- **Continuity:** `true`.
- **Standardization:** Yes, z-score.
- **Penalties / objectives:** 12 TCPD-method default configurations:
  - `changepoint_mbic`, `changepoint_bic`, `changepoint_aic`, `changepoint_hannan_quinn`
  - `cpnp_mbic`
  - `wbs_ssic`, `wbs_bic`, `wbs_mbic`
  - `rfpop_outlier`, `rfpop_l2`, `rfpop_huber`
  - `ecp_style_min30`
- **Search grid:** `min_seg = 1`, `step = 1` for all except `ecp_style_min30` which uses `min_seg = 30`, `step = 1`.
- **Practical selection:** Each configuration is run once; the result is the MICA-P score for that TCPD-method default.
- **Model set:** 27 MICA models.
- **Evaluation metric:** F1 / Covering with margin 5.

## Result file
- `benchmark_tcpd_practical_tcpd_exact_all.json`

Part files in `practical_tcpd_exact_parts/` were merged into this file and are not kept inside this folder.
