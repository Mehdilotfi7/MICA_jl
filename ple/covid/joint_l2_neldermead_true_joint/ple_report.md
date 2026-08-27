# Profile-likelihood analysis for `covid_400d_joint_l2_neldermead_true_joint_v3` (L2 loss, conditional PLE)

**Date:** 2026-08-23T19:49:03.545
**Parameter selection:** all
**Mode:** both
**CP selection:** all
**Original CPs:** [60, 150]
**L1 loss (MICA fit):** 866.7204
**Reference search:** 10 starts with Hybrid Evolutionary GA + LBFGS (GA: population=150, iterations=1000, selection=tournament(2), crossover=SBX(0.7, 1), mutation=gaussian(0.0001), mutation_rate=0.7, crossover_rate=0.7, parallel=thread) (perturbation=0.5)
**L2 loss (re-optimised):** 1594.2781
**Threshold (Δloss ≤ 3.8415):** 1598.1196
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step_increase=0.1, step_factor=1.5, stop_margin=1.2
**D2D options:** polish=true, smooth_jumps=true, allow_better_optimum=false
**Parameter wall time:** 267.6 minutes
**CP wall time:** 0.0 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| δ_global | 0.3 | 0.27857 | 0.3 | no | 0 |  |
| ᴺε₀_global | 0.33333 | 0.26925 | 0.33333 | no | 0 |  |
| ᴺε₁_global | 0.089286 | 0.08547 | 0.089286 | no | 0 |  |
| ᴺγ₀_global | 0.2 | 0.13526 | 0.2 | no | 0 |  |
| ᴺγ₁_global | 0.091743 | 0.091213 | 0.091743 | no | 0 |  |
| ᴺγ₂_global | 0.052632 | 0.052632 | 0.054006 | no | 0 |  |
| ᴺγ₃_global | 0.066176 | 0.054721 | 0.078303 | yes | 0 |  |
| ω_global | 0.0083923 | 0.0066784 | 0.010445 | yes | 0 |  |
| ᴺp₁_seg1 | 0.00027613 | 0.00023967 | 0.00031973 | yes | 0 |  |
| ᴺβ_seg1 | 0.8941 | 0.88178 | 1.0372 | yes | 0 |  |
| ᴺp₁₂_seg1 | 0.5 | 0.49404 | 0.5 | no | 0 |  |
| ᴺp₂₃_seg1 | 0.5 | 0.48447 | 0.5 | no | 0 |  |
| ᴺp₁D_seg1 | 0.001 | 0.001 | 0.0018952 | no | 0 |  |
| ᴺp₂D_seg1 | 0.0042487 | 0.001 | 0.018299 | no | 0 |  |
| ᴺp₃D_seg1 | 0.5 | 0.40544 | 0.5 | no | 0 |  |
| ν_seg1 | 0.044449 | 0.0001 | 0.1 | no | 0 |  |
| ᴺp₁_seg2 | 0.00037489 | 0.00032921 | 0.00042609 | yes | 0 |  |
| ᴺβ_seg2 | 0.11403 | 0.09225 | 0.14102 | yes | 0 |  |
| ᴺp₁₂_seg2 | 0.5 | 0.47239 | 0.5 | no | 0 |  |
| ᴺp₂₃_seg2 | 0.31736 | 0.20623 | 0.47355 | yes | 0 |  |
| ᴺp₁D_seg2 | 0.069426 | 0.023478 | 0.12003 | yes | 0 |  |
| ᴺp₂D_seg2 | 0.5 | 0.36508 | 0.5 | no | 0 |  |
| ᴺp₃D_seg2 | 0.2774 | 0.049068 | 0.5 | no | 0 |  |
| ν_seg2 | 0.064136 | 0.0001 | 0.1 | no | 0 |  |
| ᴺp₁_seg3 | 0.0027204 | 0.0024358 | 0.0030868 | yes | 0 |  |
| ᴺβ_seg3 | 0.18184 | 0.16168 | 0.20735 | yes | 0 |  |
| ᴺp₁₂_seg3 | 0.5 | 0.49309 | 0.5 | no | 0 |  |
| ᴺp₂₃_seg3 | 0.5 | 0.45761 | 0.5 | no | 0 |  |
| ᴺp₁D_seg3 | 0.001 | 0.001 | 0.0081448 | no | 0 |  |
| ᴺp₂D_seg3 | 0.228 | 0.19354 | 0.26941 | yes | 0 |  |
| ᴺp₃D_seg3 | 0.5 | 0.42132 | 0.5 | no | 0 |  |
| ν_seg3 | 0.010929 | 0.0063675 | 0.017672 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 60 | 60 | 60 | yes |
| 2 | 150 | 146 | 150 | yes |

## Quality diagnostics

- **Identifiable parameters (conditional):** 12 / 32
- **Conditional profiles that found a better optimum:** 0 (indicates the L2 re-fit may not have converged)
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2

## Files
- `ple_results.csv` — conditional profile summary
- `ple_results_curves.csv` — conditional full profile curves
- `ple_summary.csv` — conditional identifiability summary
- `cp_profile_loss.csv` — changepoint profile curves
- `cp_profile_ci.csv` — changepoint profile CIs
- `L2_refit_params.csv` — L2 re-optimised parameters
