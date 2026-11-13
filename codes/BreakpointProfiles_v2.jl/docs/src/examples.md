# Examples

## Analytic piecewise-linear ODE (test model)

The test suite uses a simple piecewise-linear model with one changepoint:

```math
u(t) = \begin{cases}
u_0 + r_1 t & t < c\\
u_0 + r_1 c + r_2 (t - c) & t \ge c
\end{cases}
```

Profiling ``u_0``, ``r_1``, ``r_2`` and the changepoint ``c`` is done with
`profile_parameter` and `profile_changepoint`, and the results are exported with
`write_profiles` and `write_cp_profiles`.  See `test/test_parameter_profile.jl`
and `test/test_changepoint_profile.jl` for the full code.

## COVID-19 SEIRD model (BIC winner)

The file `examples/covid_ple_example.jl` demonstrates the full PLE workflow for
the BIC winner of the COVID-19 application in the MICA paper.  It loads the
BIC-winning changepoint and parameter set from the `publication/applications/Covid/results`
folder, builds a `ODEChangepointPLEProblem` using the MICA objective, and runs
parameter and changepoint profiling.

Run it with default settings (profiles the eight global parameters):

```bash
julia examples/covid_ple_example.jl
```

Profile all parameters with a denser grid:

```bash
COVID_PLE_SELECTION=all COVID_PLE_NPOINTS=20 COVID_PLE_POP=80 COVID_PLE_ITER=100 \
    julia examples/covid_ple_example.jl
```

The script writes:

* `ple_results.csv` — full profile curves for each parameter,
* `ple_summary.csv` — approximate 95% confidence intervals,
* `cp_profile_loss.csv` — changepoint profile losses,
* `cp_profile_ci.csv` — changepoint confidence intervals,
* `ple_report.md` — a short Markdown report.

The output directory defaults to `examples/covid_ple_output` and can be changed
with the `COVID_PLE_OUT_DIR` environment variable.

## Plotting

If `Plots.jl` is loaded in the calling environment, profile curves can be
visualised with `plot_profiles`:

```julia
using BreakpointProfiles, Plots

# Conditional parameter profiles (changepoints fixed)
profiles = [profile_parameter(prob, i) for i in 1:length(prob)]
plot_profiles(profiles; filename="profiles.png")

# Joint parameter profiles (changepoints re-optimised)
joint_profiles = [profile_parameter_joint(prob, i) for i in 1:length(prob)]
plot_profiles(joint_profiles; filename="joint_profiles.png",
              title="Joint profile-likelihood curves")

# Changepoint-location profiles
threshold = prob.best_loss + chi2_threshold(1, 0.95)
cp_profiles = profile_all_changepoints(prob; window=7)
plot_cp_profiles(cp_profiles, threshold; filename="cp_profiles.png")
```

If `Plots` is not loaded, the plotting functions raise an error prompting the
user to load it first.
