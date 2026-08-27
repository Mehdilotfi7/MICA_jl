# Task E — Identifiability analysis for the COVID-19 model

- **Best-fit source:** Task A `results_penalty_zero`
- **Number of change points:** 8
- **Number of fitted parameters:** 80
- **Relative perturbation for finite differences:** 0.0001
- **Condition number of J'J (FIM):** 1.6944e+307

## Interpretation

A very large condition number indicates that some parameter directions are poorly identified: many parameter combinations produce nearly identical model outputs. This is expected for an 11-compartment SEIRD model with 15+ parameters, many of which are segment-specific.

## Smallest eigenvalues of the FIM

| rank_from_bottom | eigenvalue |
|---|---|
| 1 | -4.246323e-11 |
| 2 | -2.887528e-11 |
| 3 | -3.784792e-12 |
| 4 | -3.182945e-12 |
| 5 | 1.193376e-19 |
| 6 | 4.228609e-15 |
| 7 | 8.011356e-15 |
| 8 | 8.113946e-15 |
| 9 | 6.781197e-10 |
| 10 | 3.171458e-07 |

## Least sensitive parameters

| parameter | sensitivity_norm |
|---|---|
| ν_seg1 | 0.000000e+00 |
| ν_seg2 | 0.000000e+00 |
| ν_seg3 | 0.000000e+00 |
| ν_seg4 | 0.000000e+00 |
| ν_seg5 | 0.000000e+00 |
| ν_seg6 | 0.000000e+00 |
| ν_seg7 | 0.000000e+00 |
| ν_seg8 | 0.000000e+00 |
| ᴺp₃D_seg1 | 2.685877e-01 |
| ᴺp₃D_seg5 | 5.178358e-01 |

## Files created
- `parameter_sensitivity_norms.csv`
- `fim_eigenvalue_spectrum.csv`
- `fim_smallest_modes.csv`
