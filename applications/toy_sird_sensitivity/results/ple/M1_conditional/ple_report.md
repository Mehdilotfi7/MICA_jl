# Profile-likelihood analysis for dual-observable toy-SIRD M1

**Date:** 2026-11-03T01:53:47.499
**Config:** M1
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [20, 40]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 382.62566
**Threshold (RSS + 3.8415):** 386.46711
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 37.3 minutes
**CP wall time:** 0.8 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| γ_global | 0.097619 | 0.097503 | 0.09773 | yes | 0 |  |
| δ_global | 0.021263 | 0.02126 | 0.021266 | yes | 0 |  |
| β_seg1 | 0.39822 | 0.39817 | 0.39831 | yes | 0 |  |
| β_seg2 | 0.078382 | 0.074137 | 0.082702 | yes | 0 |  |
| β_seg3 | 0.19799 | 0.19029 | 0.20609 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 20 | 20 | 20 | yes |
| 2 | 40 | 40 | 45 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 5 / 5
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
