# Profile-likelihood analysis for `covid_400d_cp_l2_neldermead_v3` (L2 loss, conditional PLE)

**Date:** 2026-08-23T15:17:05.351
**Parameter selection:** all
**Mode:** cp
**CP selection:** all
**Original CPs:** [60, 150]
**L1 loss (MICA fit):** 866.7204
**Reference search:** 10 starts with Hybrid Evolutionary GA + LBFGS (GA: population=150, iterations=1000, selection=tournament(2), crossover=SBX(0.7, 1), mutation=gaussian(0.0001), mutation_rate=0.7, crossover_rate=0.7, parallel=thread) (perturbation=0.5)
**L2 loss (re-optimised):** 1595.6618
**Threshold (Δloss ≤ 3.8415):** 1599.5033
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step_increase=0.1, step_factor=1.5, stop_margin=1.2
**D2D options:** polish=true, smooth_jumps=true, allow_better_optimum=false
**CP wall time:** 0.1 minutes

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 60 | 60 | 60 | yes |
| 2 | 150 | 146 | 150 | yes |

## Quality diagnostics

- **Identifiable changepoints:** 2 / 2

## Files
- `cp_profile_loss.csv` — changepoint profile curves
- `cp_profile_ci.csv` — changepoint profile CIs
- `L2_refit_params.csv` — L2 re-optimised parameters
