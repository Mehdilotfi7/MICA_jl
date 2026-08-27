# Profile-likelihood analysis for `bic_joint_l2_hybrid` (L2 loss, joint adaptive PLE)

**Date:** 2026-08-06T13:18:17.868
**Parameter selection:** all
**Mode:** joint
**Joint CP search window:** ±20, step 5
**Joint points per direction:** 10
**Joint optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**Joint reference search:** hybrid (conditional reference)
**Original CPs:** [60, 150]
**L1 loss (MICA fit):** 866.7204
**Reference search:** 10 starts with Hybrid Evolutionary GA + LBFGS (GA: population=150, iterations=1000, selection=tournament(2), crossover=SBX(0.7, 1), mutation=gaussian(0.0001), mutation_rate=0.7, crossover_rate=0.7, parallel=thread) (perturbation=0.5)
**L2 loss (re-optimised):** 1594.3136
**Threshold (Δloss ≤ 3.8415):** 1598.1551
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step_increase=0.1, step_factor=1.5, stop_margin=1.2
**D2D options:** polish=true, smooth_jumps=true, allow_better_optimum=false
**Joint parameter wall time:** 2.1 minutes

## Joint adaptive parameter profiles (CPs marginalised)

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| δ_global | 0.3 | 0.29998 | 0.3 | yes | 0 |  |
| ᴺε₀_global | 0.33333 | 0.33331 | 0.33333 | yes | 0 |  |
| ᴺε₁_global | 0.089286 | 0.089285 | 0.089286 | yes | 0 |  |
| ᴺγ₀_global | 0.2 | 0.19999 | 0.2 | yes | 0 |  |
| ᴺγ₁_global | 0.091743 | 0.091741 | 0.091743 | yes | 0 |  |
| ᴺγ₂_global | 0.052632 | 0.052632 | 0.052644 | yes | 0 |  |
| ᴺγ₃_global | 0.066839 | 0.06683 | 0.066847 | yes | 0 |  |
| ω_global | 0.008382 | 0.008381 | 0.0083827 | yes | 0 |  |
| ᴺp₁_seg1 | 0.00027598 | 0.00027592 | 0.00027604 | yes | 0 |  |
| ᴺβ_seg1 | 0.89411 | 0.89391 | 0.89548 | yes | 0 |  |
| ᴺp₁₂_seg1 | 0.5 | 0.49995 | 0.5 | yes | 0 |  |
| ᴺp₂₃_seg1 | 0.5 | 0.49995 | 0.5 | yes | 0 |  |
| ᴺp₁D_seg1 | 0.001 | 0.001 | 0.0010002 | yes | 0 |  |
| ᴺp₂D_seg1 | 0.0038647 | 0.0038638 | 0.0038656 | yes | 0 |  |
| ᴺp₃D_seg1 | 0.5 | 0.49995 | 0.5 | yes | 0 |  |
| ν_seg1 | 0.037285 | 0.037276 | 0.037294 | yes | 0 |  |
| ᴺp₁_seg2 | 0.00037504 | 0.00037495 | 0.00037513 | yes | 0 |  |
| ᴺβ_seg2 | 0.1141 | 0.067206 | 0.11587 | yes | 0 | ⚠ |
| ᴺp₁₂_seg2 | 0.5 | 0.49995 | 0.5 | yes | 0 |  |
| ᴺp₂₃_seg2 | 0.32868 | 0.32864 | 0.32873 | yes | 0 |  |
| ᴺp₁D_seg2 | 0.069954 | 0.069937 | 0.06997 | yes | 0 |  |
| ᴺp₂D_seg2 | 0.5 | 0.49995 | 0.5 | yes | 0 |  |
| ᴺp₃D_seg2 | 0.26616 | 0.26611 | 0.26621 | yes | 0 |  |
| ν_seg2 | 0.062045 | 0.062036 | 0.062054 | yes | 0 |  |
| ᴺp₁_seg3 | 0.002724 | 0.0027233 | 0.0027246 | yes | 0 |  |
| ᴺβ_seg3 | 0.18197 | 0.18082 | 0.18201 | yes | 0 |  |
| ᴺp₁₂_seg3 | 0.5 | 0.49995 | 0.5 | no | 0 |  |
| ᴺp₂₃_seg3 | 0.5 | 0.49995 | 0.5 | yes | 0 |  |
| ᴺp₁D_seg3 | 0.001 | 0.001 | 0.0010002 | yes | 0 |  |
| ᴺp₂D_seg3 | 0.22712 | 0.22707 | 0.22716 | yes | 0 |  |
| ᴺp₃D_seg3 | 0.5 | 0.49995 | 0.5 | yes | 0 |  |
| ν_seg3 | 0.010926 | 0.010923 | 0.010928 | yes | 0 |  |

## Quality diagnostics

- **Identifiable parameters (joint):** 31 / 32
- **Joint profiles that found a better optimum:** 1 (indicates the reference may not be the joint MLE)
- **Joint profiles with optimizer failures:** 0

## Files
- `joint_ple_results.csv` — joint profile summary
- `joint_ple_results_curves.csv` — joint full profile curves
- `joint_ple_summary.csv` — joint identifiability summary
- `L2_refit_params.csv` — L2 re-optimised parameters
