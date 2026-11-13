# Profile-likelihood analysis for dual-observable toy-SIRD M2

**Date:** 2026-11-07T23:00:42.563
**Config:** M2
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [20, 43]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 255.25463
**Threshold (RSS + 3.8415):** 259.09609
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 271.6 minutes
**CP wall time:** 0.5 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| δ_global | 0.021229 | 0.021226 | 0.021232 | yes | 0 |  |
| β_seg1 | 0.39865 | 0.39856 | 0.39873 | yes | 0 |  |
| γ_seg1 | 0.097991 | 0.097875 | 0.098106 | yes | 0 |  |
| β_seg2 | 0.083061 | 0.060773 | 0.10123 | yes | 0 |  |
| γ_seg2 | 0.09792 | 0.092806 | 0.10194 | yes | 0 |  |
| β_seg3 | 0.41396 | 0.27307 | 0.52754 | yes | 0 |  |
| γ_seg3 | 0.1376 | 0.10904 | 0.15766 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 20 | 20 | 20 | yes |
| 2 | 43 | 42 | 43 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 7 / 7
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
