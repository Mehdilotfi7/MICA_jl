# BreakpointProfiles.jl

Profile-likelihood-based uncertainty quantification for ordinary differential
equation (ODE) models with fixed changepoints.

## Overview

`BreakpointProfiles.jl` provides a small, reusable Julia package for
profiling the likelihood of parameters and changepoint locations in ODE models
whose changepoints are treated as known.  It was originally written to support
the uncertainty analysis in the MICA project, but is designed to be generic.

The package can work with any model that exposes one of the following:

* an objective function `(params, changepoints) -> loss`;
* a simulator `(params, changepoints) -> simulation_matrix` plus a loss function;
* a plain ODE right-hand side `(du, u, p, t)` plus initial condition and time span.

## Installation

The package is not yet registered.  Install it from a local clone by developing
it:

```julia
using Pkg
Pkg.develop(path="path/to/BreakpointProfiles.jl")
```

## Quick start

```julia
using BreakpointProfiles

# Build a problem (objective, simulator, or ODE-based)
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

# Conditional profile-likelihood for the first parameter
prof = profile_parameter(prob, 1)

# Joint profile-likelihood for the first parameter
joint_prof = profile_parameter_joint(prob, 1)

# Profile a changepoint location
cp_prof = profile_changepoint(prob, 1; window=7)

# Summarise and export
summary_df = ple_summary([prof])
write_profiles("ple_results.csv", [prof])
```

## Key features

* Conditional and joint profile-likelihood for continuous parameters.
* Changepoint-location profiling by re-optimisation of all other parameters.
* Adaptive or fixed profile-likelihood grids with automatic bound handling.
* Multiple optimiser backends: Evolutionary (GA), Metaheuristics (DE), Optim (LBFGS), NLopt (BOBYQA), and multi-start/hybrid pipelines.
* Built-in Gaussian and Laplace negative log-likelihoods plus custom losses.
* Confidence-interval extraction and practical identifiability checks.
* CSV/DataFrame export, Markdown reports, and optional plotting helpers.

## Next steps

* See the [API reference](api.md) for a complete list of types and functions.
* See the [Examples](examples.md) page for a COVID-19 workflow.

## License

This package is released under the MIT license.
