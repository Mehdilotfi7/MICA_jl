# Profile-likelihood analysis for dual-observable toy-SIRD M4

**Date:** 2026-11-03T10:51:03.714
**Config:** M4
**Mode:** joint
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs (reference):** [20, 40]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 3123.5958
**Threshold (RSS + 3.8415):** 3127.4373
**Joint CP search window:** +/-7, step 1
**Joint optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**Joint parameter wall time:** 119.4 minutes

## Joint adaptive parameter profiles (CPs marginalised)

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| β_global | 0.39564 | 0.3955 | 0.39586 | yes | 0 |  |
| δ_global | 0.021404 | 0.021401 | 0.021408 | yes | 0 |  |
| γ_seg1 | 0.095457 | 0.095259 | 0.095681 | yes | 0 |  |
| γ_seg2 | 0.15542 | 0.15533 | 0.15556 | yes | 0 |  |
| γ_seg3 | 0.093133 | 0.092019 | 0.093459 | yes | 0 |  |

## Quality diagnostics
- **Identifiable parameters (joint):** 5 / 5
- **Joint profiles that found a better optimum:** 0
- **Joint profiles with optimizer failures:** 0
