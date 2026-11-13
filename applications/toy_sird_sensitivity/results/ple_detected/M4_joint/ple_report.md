# Profile-likelihood analysis for dual-observable toy-SIRD M4

**Date:** 2026-11-07T23:51:27.279
**Config:** M4
**Mode:** joint
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs (reference):** [37, 43]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 5766.1542
**Threshold (RSS + 3.8415):** 5769.9957
**Joint CP search window:** +/-7, step 1
**Joint optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**Joint parameter wall time:** 328.3 minutes

## Joint adaptive parameter profiles (CPs marginalised)

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| β_global | 0.43822 | 0.43821 | 0.43823 | yes | 0 |  |
| δ_global | 0.019866 | 0.019863 | 0.019869 | yes | 0 |  |
| γ_seg1 | 0.12248 | 0.12248 | 0.12249 | yes | 0 |  |
| γ_seg2 | 0.078109 | 0.053585 | 0.08 | yes | 0 |  |
| γ_seg3 | 0.072393 | 0.011856 | 0.34882 | yes | 0 |  |

## Quality diagnostics
- **Identifiable parameters (joint):** 5 / 5
- **Joint profiles that found a better optimum:** 0
- **Joint profiles with optimizer failures:** 0
