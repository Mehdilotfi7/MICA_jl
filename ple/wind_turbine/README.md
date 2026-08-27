# Wind turbine — PLE materials

This folder collects scripts and inputs for the wind-turbine profile-likelihood
analysis. A final reliable result is still being established.

## Inputs (`inputs/`)

- `detected_cps.csv` — BIC-selected change points (indices 140, 500, 1150, 1860).
- `refit_params.csv` — Corresponding fitted parameters.

## Scripts (`scripts/`)

- `turbine_ple_with_package.jl` — Original local PLE driver (Evolutionary GA profiling).
- `turbine_ple_hybrid.jl` — Upgraded driver with hybrid reference + BOBYQA profiling.

## Tentative result (`outputs/tentative_ple_bic/`)

The 2026-07-18 `ple_bic/` result is kept as a placeholder. It used the MICA
best parameters as the reference and Evolutionary GA for profiling. Because the
reference was not re-optimised, it is not yet a verified MLE-based interval.

## Next step

Re-run turbine CPD + PLE with the consistent L2/hybrid recipe (same as COVID)
to obtain a reliable MLE reference and valid χ²-based confidence intervals.
