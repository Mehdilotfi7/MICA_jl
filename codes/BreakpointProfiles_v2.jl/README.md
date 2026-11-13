> **Note.** This is an extended copy of `../BreakpointProfiles.jl` with an improved joint-profile pipeline (anchored reference re-optimization, multi-start warm starts, retry logic, and a final polish step in both grid branches). It is kept as a separate directory so that the original package continues to reproduce the earlier COVID-19 and wind-turbine profile-likelihood results unchanged. Used by `applications/toy_sird_sensitivity/`.

# BreakpointProfiles.jl

Profile-likelihood-based uncertainty quantification for ordinary differential equation (ODE) models with fixed changepoints.

This package is designed to be used **after** a changepoint-detection step (for example with [Mica.jl](https://github.com/Mehdilotfi7/Mica.jl)), once the number and locations of changepoints have been determined. `BreakpointProfiles.jl` then quantifies the uncertainty around the fitted parameters and the detected changepoint locations via profile-likelihood methods.

The adaptive profile-likelihood engine and the confidence-interval extraction in this package were inspired by and validated against the Data2Dynamics (D2D) MATLAB profile-likelihood (Raue et al., *Bioinformatics* 2015; Raue et al., *PLOS ONE* 2013). Users familiar with D2D will recognise the D2D-style adaptive outward stepping, warm-start re-optimisation, and jump-smoothing strategy.

## Installation

Install the package directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Mehdilotfi7/BreakpointProfiles.jl")
```

Or clone the repository and develop it locally:

```julia
using Pkg
Pkg.develop(path="path/to/BreakpointProfiles.jl")
```

## Quick start

```julia
using BreakpointProfiles

# Build a problem from a previously fitted changepoint model
# (e.g. obtained with Mica.jl; see examples/ for a full COVID-19 workflow)
prob = ODEChangepointPLEProblem(
    objective = (params, cps) -> my_loss(params, cps),
    data = my_data,
    loss_fn = my_loss,
    changepoints = [50, 100],
    best_params = best_fit,
    best_loss = best_loss,
    lb = lb,
    ub = ub,
    n_global = n_global,
    n_segment_specific = n_segment_specific
)

# Conditional profile-likelihood for a parameter (changepoints fixed)
prof = profile_parameter(prob, 1)

# Joint profile-likelihood for a parameter (changepoints re-optimised)
joint_prof = profile_parameter_joint(prob, 1)

# Profile a changepoint location
cp_prof = profile_changepoint(prob, 1; window=7)

# Summarise
summary = ple_summary([prof])
```

## Features

- **Conditional profile likelihood** for continuous parameters (changepoints fixed).
- **Joint profile likelihood** for continuous parameters (changepoints treated as discrete nuisance variables and re-optimised).
- **Changepoint-location profiling** with re-optimisation of all other parameters.
- Adaptive profile-likelihood grids with bound handling.
- Multiple optimiser backends: GA (Evolutionary.jl), differential evolution (Metaheuristics.jl), LBFGS (Optim.jl), BOBYQA (NLopt.jl), and multi-start/hybrid pipelines.
- Likelihood options: Gaussian (L2), Laplace (L1), log-scale, and custom losses.
- Identifiability assessment and confidence-interval extraction.
- CSV/DataFrame export, Markdown reports, and plotting utilities.

## L2 vs L1 losses and thresholds

- **Gaussian (L2) losses:** Wilks' theorem applies; use the standard χ² threshold
  (`best_loss + 3.8415` for 1 df, 95%).
- **Laplace (L1) losses:** Wilks' theorem does not apply. Pass a user-derived
  threshold to `profile_parameter(...; threshold=...)`.

The COVID driver (`examples/covid_ple_example.jl`) supports both `L1` and `L2`
modes via the `COVID_PLE_LOSS` environment variable.


## Citation

If you use this package in research, please cite the MICA paper.

## License

MIT
