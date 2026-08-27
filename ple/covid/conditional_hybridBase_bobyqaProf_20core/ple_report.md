# Profile-likelihood analysis for `covid_400d_cond_hybridBase_bobyqaProf_20core` (L2 loss, conditional PLE)

**Date:** 2026-08-22T22:47:12.348
**Parameter selection:** all
**Mode:** parameter
**Original CPs:** [60, 150]
**L1 loss (MICA fit):** 866.7204
**Reference search:** single run with Hybrid Evolutionary GA + LBFGS (GA: population=150, iterations=1000, selection=tournament(2), crossover=SBX(0.7, 1), mutation=gaussian(0.0001), mutation_rate=0.7, crossover_rate=0.7, parallel=thread)
**L2 loss (re-optimised):** 1509.4206
**Threshold (Δloss ≤ 3.8415):** 1513.2621
**Profiling optimizer:** NLopt BOBYQA (maxeval=10000)
**D2D adaptive profiling:** samplesize=100, rel_step_increase=0.1, step_factor=1.5, stop_margin=1.2
**D2D options:** polish=true, smooth_jumps=true, allow_better_optimum=true
**Parameter wall time:** 216.3 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| δ_global | 0.36 | 0.34845 | 0.36 | no | 0 |  |
| ᴺε₀_global | 0.4 | 0.1946 | 0.4 | no | 0 |  |
| ᴺε₁_global | 0.10714 | 0.07952 | 0.10714 | no | 0 |  |
| ᴺγ₀_global | 0.24 | 0.16151 | 0.24 | no | 0 |  |
| ᴺγ₁_global | 0.11009 | 0.10935 | 0.11009 | no | 0 |  |
| ᴺγ₂_global | 0.066556 | 0.062929 | 0.070185 | yes | 0 |  |
| ᴺγ₃_global | 0.092583 | 0.07771 | 0.10893 | yes | 0 |  |
| ω_global | 0.013103 | 0.0093876 | 0.0144 | no | 0 |  |
| ᴺp₁_seg1 | 0.075897 | 0.05926 | 0.13104 | yes | 0 |  |
| ᴺβ_seg1 | 0.36275 | 0.35774 | 0.50994 | yes | 0 |  |
| ᴺp₁₂_seg1 | 0.6 | 0.59294 | 0.6 | no | 0 |  |
| ᴺp₂₃_seg1 | 0.6 | 0.58206 | 0.6 | no | 0 |  |
| ᴺp₁D_seg1 | 0.00083333 | 0.00083333 | 0.0099573 | no | 0 |  |
| ᴺp₂D_seg1 | 0.0024909 | 0.00083333 | 0.11136 | no | 0 |  |
| ᴺp₃D_seg1 | 0.6 | 0.00083333 | 0.6 | no | 0 |  |
| ν_seg1 | 0.12 | 8.3333e-05 | 0.12 | no | 0 |  |
| ᴺp₁_seg2 | 0.00097099 | 0.00078853 | 0.00132 | yes | 0 |  |
| ᴺβ_seg2 | 0.061897 | 0.045828 | 0.064417 | yes | 0 |  |
| ᴺp₁₂_seg2 | 0.6 | 0.53461 | 0.6 | no | 0 |  |
| ᴺp₂₃_seg2 | 0.33432 | 0.24464 | 0.48357 | yes | 0 |  |
| ᴺp₁D_seg2 | 0.00083333 | 0.00083333 | 0.023126 | no | 0 |  |
| ᴺp₂D_seg2 | 0.21863 | 0.12519 | 0.31323 | yes | 0 |  |
| ᴺp₃D_seg2 | 0.58454 | 0.30572 | 0.6 | no | 0 |  |
| ν_seg2 | 0.046464 | 8.3333e-05 | 0.12 | no | 0 |  |
| ᴺp₁_seg3 | 0.0025535 | 0.0023351 | 0.0028367 | yes | 0 |  |
| ᴺβ_seg3 | 0.1501 | 0.13009 | 0.15093 | yes | 0 |  |
| ᴺp₁₂_seg3 | 0.6 | 0.57992 | 0.6 | no | 0 |  |
| ᴺp₂₃_seg3 | 0.6 | 0.48163 | 0.6 | no | 0 |  |
| ᴺp₁D_seg3 | 0.00083333 | 0.00083333 | 0.0088559 | no | 0 |  |
| ᴺp₂D_seg3 | 0.003824 | 0.00083333 | 0.075342 | no | 0 |  |
| ᴺp₃D_seg3 | 0.6 | 0.54185 | 0.6 | no | 0 |  |
| ν_seg3 | 0.0089203 | 0.0045255 | 0.016423 | yes | 0 |  |

## Quality diagnostics

- **Identifiable parameters (conditional):** 11 / 32
- **Conditional profiles that found a better optimum:** 0 (indicates the L2 re-fit may not have converged)
- **Conditional profiles with optimizer failures:** 0

## Files
- `ple_results.csv` — conditional profile summary
- `ple_results_curves.csv` — conditional full profile curves
- `ple_summary.csv` — conditional identifiability summary
- `L2_refit_params.csv` — L2 re-optimised parameters
