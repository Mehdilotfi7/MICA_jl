# Profile-likelihood analysis for dual-observable toy-SIRD M3

**Date:** 2026-11-07T19:24:54.174
**Config:** M3
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [20, 48]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 457.28241
**Threshold (RSS + 3.8415):** 461.12387
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 111.8 minutes
**CP wall time:** 0.0 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| β_seg1 | 0.39897 | 0.39888 | 0.39904 | yes | 0 |  |
| γ_seg1 | 0.098088 | 0.097973 | 0.098194 | yes | 0 |  |
| δ_seg1 | 0.021384 | 0.021322 | 0.021447 | yes | 0 |  |
| β_seg2 | 0.05 | 0.05 | 0.054531 | no | 0 |  |
| γ_seg2 | 0.087872 | 0.087817 | 0.088934 | yes | 0 |  |
| δ_seg2 | 0.021101 | 0.021026 | 0.021176 | yes | 0 |  |
| β_seg3 | 0.85346 | 0.73069 | 0.96504 | yes | 0 |  |
| γ_seg3 | 0.24304 | 0.2169 | 0.26584 | yes | 0 |  |
| δ_seg3 | 0.019595 | 0.01771 | 0.021511 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 20 | 20 | 20 | yes |
| 2 | 48 | 44 | 48 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 8 / 9
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
