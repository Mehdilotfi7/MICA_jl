# Profile-likelihood analysis for dual-observable toy-SIRD M3

**Date:** 2026-11-08T11:22:53.106
**Config:** M3
**Mode:** joint
**Model:** SIRD, n=80, sigma=1.0, seed=2, true CPs=[20, 40]
**Channels:** I and D observed, scales s_I=343.39, s_D=138.54
**Fixed CPs (reference):** [20, 48]
**Loss:** raw pooled RSS over both observed channels (I and D)
**Reference search:** hybrid GA+LBFGS, 10 starts on 8 threads (pop=150, iter=1000)
**RSS (re-optimised):** 457.28241
**Threshold (RSS + 3.8415):** 461.12387
**Joint CP search window:** +/-7, step 1
**Joint optimizer:** Optim Nelder-Mead with bounds (Fminbox, iterations=2000)
**Joint parameter wall time:** 1069.7 minutes

## Joint adaptive parameter profiles (CPs marginalised)

| parameter | best value | CI lower | CI upper | identifiable | n_failed | found better |
|---|---|---|---|---|---|---|
| β_seg1 | 0.39856 | 0.39847 | 0.39865 | yes | 0 |  |
| γ_seg1 | 0.097865 | 0.097728 | 0.098005 | yes | 0 |  |
| δ_seg1 | 0.021323 | 0.021258 | 0.021389 | yes | 0 |  |
| β_seg2 | 0.081083 | 0.058336 | 0.10484 | yes | 0 |  |
| γ_seg2 | 0.0973 | 0.092243 | 0.10296 | yes | 0 |  |
| δ_seg2 | 0.021293 | 0.021204 | 0.021389 | yes | 0 |  |
| β_seg3 | 0.41915 | 0.25756 | 0.52796 | yes | 0 |  |
| γ_seg3 | 0.14052 | 0.10449 | 0.1602 | yes | 0 |  |
| δ_seg3 | 0.019437 | 0.018201 | 0.020685 | yes | 0 |  |

## Quality diagnostics
- **Identifiable parameters (joint):** 9 / 9
- **Joint profiles that found a better optimum:** 0
- **Joint profiles with optimizer failures:** 0
