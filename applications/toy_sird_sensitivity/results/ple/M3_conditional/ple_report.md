# Profile-likelihood analysis for dual-observable toy-SIRD M3

**Date:** 2026-11-03T10:47:33.234
**Config:** M3
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [20, 40]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 335.79534
**Threshold (RSS + 3.8415):** 339.6368
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 605.3 minutes
**CP wall time:** 1.6 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| β_seg1 | 0.39906 | 0.39896 | 0.39914 | yes | 0 |  |
| γ_seg1 | 0.098267 | 0.098123 | 0.098402 | yes | 0 |  |
| δ_seg1 | 0.021272 | 0.021205 | 0.02134 | yes | 0 |  |
| β_seg2 | 0.12399 | 0.10804 | 0.1385 | yes | 0 |  |
| γ_seg2 | 0.10837 | 0.10565 | 0.11015 | yes | 0 |  |
| δ_seg2 | 0.02145 | 0.021348 | 0.021553 | yes | 0 |  |
| β_seg3 | 0.05 | 0.05 | 0.1828 | no | 0 |  |
| γ_seg3 | 0.059409 | 0.058146 | 0.087144 | yes | 0 |  |
| δ_seg3 | 0.019709 | 0.018659 | 0.020766 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 20 | 20 | 20 | yes |
| 2 | 40 | 40 | 45 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 8 / 9
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
