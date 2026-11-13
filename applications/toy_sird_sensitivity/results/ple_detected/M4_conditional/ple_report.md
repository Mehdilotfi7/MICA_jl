# Profile-likelihood analysis for dual-observable toy-SIRD M4

**Date:** 2026-11-07T18:53:27.995
**Config:** M4
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [37, 43]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 5766.1542
**Threshold (RSS + 3.8415):** 5769.9957
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 30.5 minutes
**CP wall time:** 1.7 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| β_global | 0.43846 | 0.43845 | 0.43847 | yes | 0 |  |
| δ_global | 0.019958 | 0.019956 | 0.019961 | yes | 0 |  |
| γ_seg1 | 0.12253 | 0.12252 | 0.12253 | yes | 0 |  |
| γ_seg2 | 0.12637 | 0.12429 | 0.12851 | yes | 0 |  |
| γ_seg3 | 0.067918 | 0.066774 | 0.069154 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 37 | 37 | 41 | yes |
| 2 | 43 | 40 | 43 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 5 / 5
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
