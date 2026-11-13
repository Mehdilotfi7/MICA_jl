# Profile-likelihood analysis for dual-observable toy-SIRD M1

**Date:** 2026-11-07T18:49:40.100
**Config:** M1
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [20, 42]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 261.61113
**Threshold (RSS + 3.8415):** 265.45259
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 43.7 minutes
**CP wall time:** 0.6 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| γ_global | 0.097938 | 0.097824 | 0.098048 | yes | 0 |  |
| δ_global | 0.021239 | 0.021235 | 0.021242 | yes | 0 |  |
| β_seg1 | 0.39861 | 0.39855 | 0.39869 | yes | 0 |  |
| β_seg2 | 0.082403 | 0.078061 | 0.086822 | yes | 0 |  |
| β_seg3 | 0.2209 | 0.21229 | 0.22995 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 20 | 20 | 20 | yes |
| 2 | 42 | 42 | 43 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 5 / 5
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
