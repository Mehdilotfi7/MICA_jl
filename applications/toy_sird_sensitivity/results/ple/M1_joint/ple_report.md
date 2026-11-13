# Profile-likelihood analysis for dual-observable toy-SIRD M1

**Date:** 2026-11-03T11:37:12.657
**Config:** M1
**Mode:** joint
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs (reference):** [20, 40]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 382.62566
**Threshold (RSS + 3.8415):** 386.46711
**Joint CP search window:** +/-7, step 1
**Joint optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**Joint parameter wall time:** 164.6 minutes

## Joint adaptive parameter profiles (CPs marginalised)

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| γ_global | 0.097938 | 0.097824 | 0.098051 | yes | 0 |  |
| δ_global | 0.021239 | 0.021235 | 0.021242 | yes | 0 |  |
| β_seg1 | 0.39861 | 0.39855 | 0.39869 | yes | 0 |  |
| β_seg2 | 0.082403 | 0.078046 | 0.087852 | yes | 0 |  |
| β_seg3 | 0.2209 | 0.21229 | 0.23756 | yes | 0 |  |

## Quality diagnostics
- **Identifiable parameters (joint):** 5 / 5
- **Joint profiles that found a better optimum:** 0
- **Joint profiles with optimizer failures:** 0
