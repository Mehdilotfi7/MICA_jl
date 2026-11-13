# Profile-likelihood analysis for dual-observable toy-SIRD M2

**Date:** 2026-11-08T21:25:53.049
**Config:** M2
**Mode:** joint
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs (reference):** [20, 43]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 255.25463
**Threshold (RSS + 3.8415):** 259.09609
**Joint CP search window:** +/-7, step 1
**Joint optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**Joint parameter wall time:** 1620.0 minutes

## Joint adaptive parameter profiles (CPs marginalised)

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| δ_global | 0.021229 | 0.021226 | 0.021232 | yes | 0 |  |
| β_seg1 | 0.39865 | 0.39856 | 0.39873 | yes | 0 |  |
| γ_seg1 | 0.097991 | 0.097875 | 0.09811 | yes | 0 |  |
| β_seg2 | 0.083061 | 0.060798 | 0.10591 | no | 0 |  |
| γ_seg2 | 0.09792 | 0.092806 | 0.10345 | yes | 0 |  |
| β_seg3 | 0.41395 | 0.24851 | 0.52564 | yes | 0 |  |
| γ_seg3 | 0.1376 | 0.10108 | 0.15766 | yes | 0 |  |

## Quality diagnostics
- **Identifiable parameters (joint):** 6 / 7
- **Joint profiles that found a better optimum:** 0
- **Joint profiles with optimizer failures:** 0
