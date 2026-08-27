# Profile-likelihood analysis for `bic_conditional_l2_neldermead` (L2 loss, conditional PLE)

**Date:** 2026-08-04T16:14:00.103
**Parameter selection:** all
**Mode:** parameter
**Original CPs:** [60, 150]
**L1 loss (MICA fit):** 866.7204
**Reference search:** 10 starts with Hybrid Evolutionary GA + LBFGS (GA: population=150, iterations=1000, selection=tournament(2), crossover=SBX(0.7, 1), mutation=gaussian(0.0001), mutation_rate=0.7, crossover_rate=0.7, parallel=thread) (perturbation=0.5)
**L2 loss (re-optimised):** 1597.661
**Threshold (Δloss ≤ 3.8415):** 1601.5024
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step_increase=0.1, step_factor=1.5, stop_margin=1.2
**D2D options:** polish=true, smooth_jumps=true, allow_better_optimum=false
**Parameter wall time:** 2.0 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| δ_global | 0.3 | 0.29474 | 0.3 | no | 0 |  |
| ᴺε₀_global | 0.27554 | 0.27025 | 0.27974 | yes | 0 |  |
| ᴺε₁_global | 0.089286 | 0.089129 | 0.089286 | no | 0 |  |
| ᴺγ₀_global | 0.2 | 0.19678 | 0.2 | no | 0 |  |
| ᴺγ₁_global | 0.091743 | 0.091291 | 0.091743 | no | 0 |  |
| ᴺγ₂_global | 0.052632 | 0.052632 | 0.053651 | no | 0 |  |
| ᴺγ₃_global | 0.066646 | 0.062733 | 0.069379 | yes | 0 |  |
| ω_global | 0.0080393 | 0.0079754 | 0.0080988 | yes | 0 |  |
| ᴺp₁_seg1 | 0.00027619 | 0.00026625 | 0.00028673 | yes | 0 |  |
| ᴺβ_seg1 | 1.021 | 1.0125 | 1.0265 | yes | 0 |  |
| ᴺp₁₂_seg1 | 0.5 | 0.49459 | 0.5 | no | 0 |  |
| ᴺp₂₃_seg1 | 0.5 | 0.48921 | 0.5 | no | 0 |  |
| ᴺp₁D_seg1 | 0.001 | 0.001 | 0.001101 | no | 0 |  |
| ᴺp₂D_seg1 | 0.00327 | 0.0029786 | 0.0035993 | yes | 0 |  |
| ᴺp₃D_seg1 | 0.5 | 0.48389 | 0.5 | no | 0 |  |
| ν_seg1 | 0.052249 | 0.050204 | 0.054294 | yes | 0 |  |
| ᴺp₁_seg2 | 0.00037313 | 0.0003532 | 0.00039793 | yes | 0 |  |
| ᴺβ_seg2 | 0.11524 | 0.11174 | 0.11822 | yes | 0 |  |
| ᴺp₁₂_seg2 | 0.5 | 0.48985 | 0.5 | no | 0 |  |
| ᴺp₂₃_seg2 | 0.32327 | 0.31309 | 0.34055 | yes | 0 |  |
| ᴺp₁D_seg2 | 0.073827 | 0.070059 | 0.07998 | yes | 0 |  |
| ᴺp₂D_seg2 | 0.5 | 0.48446 | 0.5 | no | 0 |  |
| ᴺp₃D_seg2 | 0.25228 | 0.2421 | 0.27948 | yes | 0 |  |
| ν_seg2 | 0.0014989 | 0.0013656 | 0.0016322 | yes | 0 |  |
| ᴺp₁_seg3 | 0.0027583 | 0.0026914 | 0.0028255 | yes | 0 |  |
| ᴺβ_seg3 | 0.18987 | 0.18934 | 0.19056 | yes | 0 |  |
| ᴺp₁₂_seg3 | 0.5 | 0.49425 | 0.5 | no | 0 |  |
| ᴺp₂₃_seg3 | 0.5 | 0.48464 | 0.5 | no | 0 |  |
| ᴺp₁D_seg3 | 0.001 | 0.001 | 0.0010882 | no | 0 |  |
| ᴺp₂D_seg3 | 0.22985 | 0.2197 | 0.24357 | yes | 0 |  |
| ᴺp₃D_seg3 | 0.5 | 0.48272 | 0.5 | no | 0 |  |
| ν_seg3 | 0.011419 | 0.010836 | 0.012003 | yes | 0 |  |

## Quality diagnostics

- **Identifiable parameters (conditional):** 17 / 32
- **Conditional profiles that found a better optimum:** 0 (indicates the L2 re-fit may not have converged)
- **Conditional profiles with optimizer failures:** 0

## Files
- `ple_results.csv` — conditional profile summary
- `ple_results_curves.csv` — conditional full profile curves
- `ple_summary.csv` — conditional identifiability summary
- `L2_refit_params.csv` — L2 re-optimised parameters
