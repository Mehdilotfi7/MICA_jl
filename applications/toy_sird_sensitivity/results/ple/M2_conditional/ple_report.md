# Profile-likelihood analysis for dual-observable toy-SIRD M2

**Date:** 2026-11-03T06:29:36.042
**Config:** M2
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [20, 40]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 344.16743
**Threshold (RSS + 3.8415):** 348.00889
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 347.6 minutes
**CP wall time:** 1.4 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| δ_global | 0.021263 | 0.021259 | 0.021266 | yes | 0 |  |
| β_seg1 | 0.39911 | 0.39902 | 0.3992 | yes | 0 |  |
| γ_seg1 | 0.098293 | 0.098169 | 0.098409 | yes | 0 |  |
| β_seg2 | 0.12556 | 0.10981 | 0.13994 | yes | 0 |  |
| γ_seg2 | 0.10895 | 0.10635 | 0.11063 | yes | 0 |  |
| β_seg3 | 0.05 | 0.05 | 0.17919 | no | 0 |  |
| γ_seg3 | 0.058261 | 0.057204 | 0.085135 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 20 | 20 | 20 | yes |
| 2 | 40 | 40 | 45 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 6 / 7
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
