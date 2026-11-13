# Profile-likelihood analysis for dual-observable toy-SIRD M4

**Date:** 2026-11-03T01:47:22.484
**Config:** M4
**Mode:** conditional
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs:** [20, 40]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 3123.5958
**Threshold (RSS + 3.8415):** 3127.4373
**Profiling optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**D2D adaptive profiling:** samplesize=100, rel_step=0.1, step_factor=1.5, stop_margin=1.2
**Parameter wall time:** 28.6 minutes
**CP wall time:** 1.9 minutes

## Conditional parameter profiles

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| β_global | 0.41411 | 0.41403 | 0.41417 | yes | 0 |  |
| δ_global | 0.021249 | 0.021246 | 0.021253 | yes | 0 |  |
| γ_seg1 | 0.10828 | 0.1082 | 0.10835 | yes | 0 |  |
| γ_seg2 | 0.14753 | 0.14748 | 0.14758 | yes | 0 |  |
| γ_seg3 | 0.064959 | 0.064499 | 0.065475 | yes | 0 |  |

## Changepoint profile intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 20 | 18 | 20 | yes |
| 2 | 40 | 33 | 40 | yes |

## Quality diagnostics
- **Identifiable parameters (conditional):** 5 / 5
- **Conditional profiles that found a better optimum:** 0
- **Conditional profiles with optimizer failures:** 0
- **Identifiable changepoints:** 2 / 2
