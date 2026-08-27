# D2D profile-likelihood for the baseline COVID-19 model

This folder contains a Data2Dynamics-compatible baseline (no-changepoint) version of the COVID-19 SEIRD model for local MATLAB PLE.

## Files

- `Models/covid_base.def` — D2D model definition (11 states, 16 parameters).
- `Data/covid_data.def` / `Data/covid_data.csv` — observed data, log10(obs+1) transformed.
- `d2d_initial_params.m` — best-fit parameters and bounds exported from Julia/MICA.
- `run_ple.m` — MATLAB script that loads the model, fits, and runs D2D PLE.

## Model notes

- The model is the **single-segment baseline**: no changepoints, constant parameters.
- Seasonality is kept as `1 + delta * cos(2*pi*t/365)`.
- Vaccination is approximated by a smooth step centered at day 330.
- Observables are `log10(observable + 1)` to match the log-transform used in MICA.
- Error-model parameters (`sd_*`) are fixed; only the 16 structural parameters are profiled.

## How to run

1. Open MATLAB in this directory.
2. Make sure the D2D toolbox path in `run_ple.m` matches your local install.
3. Run:

```matlab
run_ple
```

Results are saved in `Output_YYYYMMDD_HHMMSS/`.

## Parameter source

Best-fit values come from `fit_covid_base_model.jl` (Julia/MICA), loss = 1845.75.
