# BreakpointProfiles.jl

Profile-likelihood-based uncertainty quantification for ordinary differential equation (ODE) models with fixed changepoints.

## Installation

The package is not yet registered in the Julia General registry. Install it directly from GitHub:

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

# Define an ODEChangepointPLEProblem (see examples/ for a full COVID-19 example)
prob = ODEChangepointPLEProblem(...)

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

## Documentation

Build the documentation locally with Documenter.jl:

```bash
julia --project=docs docs/make.jl
```

## Citation

If you use this package in research, please cite the MICA paper.

## License

MIT
