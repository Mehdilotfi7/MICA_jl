# Corrected profile-likelihood analysis for wind-turbine `bic`

**Correction:** the original `tentative_ple_bic` used a raw delta-loss threshold of 3.8415.
Because the profile is based on the SSE loss, the threshold is rescaled by the estimated residual variance.

**n_obs:** 2500
**n_params:** 23
**Best SSE:** 147213.1471
**Estimated sigma^2:** 59.4320
**Corrected threshold (SSE + 3.8415 * sigma^2):** 147441.4528

**Identifiable parameters:** 22 / 23
**Identifiable changepoints:** 0 / 4

## Parameter intervals

| parameter | best value | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| θ1_global | 1.1677 | 0 | 1.2261 | yes |
| θ2_global | 81.592 | 76.842 | 82.592 | yes |
| θ3_global | 1.0819 | 0 | 1.163 | yes |
| θ4_seg1 | 1.4958 | 0.50952 | 1.5706 | yes |
| θ5_seg1 | 1.4931 | 0 | 1.5678 | yes |
| θ6_seg1 | 1.5594 | 0 | 1.6374 | yes |
| θ7_seg1 | 1.5232 | 0 | 1.811 | yes |
| θ4_seg2 | 0.16838 | 0.14733 | 0.1768 | yes |
| θ5_seg2 | 1.2346 | 1.1729 | 1.2964 | yes |
| θ6_seg2 | 1.5021 | 1.427 | 1.8589 | yes |
| θ7_seg2 | 1.4777 | 1.4038 | 3.8547 | yes |
| θ4_seg3 | 0.11563 | 0.10985 | 0.13876 | yes |
| θ5_seg3 | 1.4845 | 1.4103 | 1.6701 | yes |
| θ6_seg3 | 1.4922 | 1.4175 | 2.476 | yes |
| θ7_seg3 | 1.4883 | 1.4139 | 3.8824 | yes |
| θ4_seg4 | 0.2327 | 0.2327 | 0.38614 | no |
| θ5_seg4 | 1.4967 | 1.4219 | 2.1047 | yes |
| θ6_seg4 | 1.491 | 1.4165 | 3.0403 | yes |
| θ7_seg4 | 1.5769 | 0.53714 | 1.6558 | yes |
| θ4_seg5 | 0.11033 | 0.10482 | 0.15516 | yes |
| θ5_seg5 | 1.8369 | 1.745 | 1.9287 | yes |
| θ6_seg5 | 1.4316 | 1.0916 | 1.5032 | yes |
| θ7_seg5 | 1.4655 | 0.49918 | 1.5388 | yes |

## Changepoint intervals

| cp # | original | CI lower | CI upper | identifiable |
|---|---|---|---|---|
| 1 | 140 | 133.0 | 133.0 | no |
| 2 | 500 | nan | nan | no |
| 3 | 1150 | nan | nan | no |
| 4 | 1860 | nan | nan | no |

