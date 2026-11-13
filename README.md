# MICA: Model-Informed Change-point Analysis

This repository contains the source code, data, scripts, and results for the paper:

> **MICA: Model-Informed Change-point Analysis**  
> PLOS Computational Biology

## Repository overview

MICA performs model-informed changepoint detection by embedding domain-specific ordinary differential equation (ODE) or algebraic models inside an optimization framework. This repository reproduces the paper's applications, benchmarking, and profile-likelihood uncertainty analyses.

The profile-likelihood methods in this repository (implemented in `codes/BreakpointProfiles.jl`) were inspired by and validated against the Data2Dynamics (D2D) MATLAB profile-likelihood (Raue et al., *Bioinformatics* 2015; Raue et al., *PLOS ONE* 2013). A D2D-based COVID-19 baseline is provided in `applications/covid/d2d_base/` for comparison.

## Folder structure

```
MICA_jl/
├── codes/
│   ├── Mica.jl/                # Main MICA Julia package
│   ├── BreakpointProfiles.jl/  # Profile-likelihood / changepoint-profile engine
│   └── BreakpointProfiles_v2.jl/  # Extended copy with improved joint-profile pipeline (used by toy_sird_sensitivity)
├── applications/
│   ├── covid/                  # COVID-19 SEIRD case study
│   │   ├── scripts/            # BIC/MDL/AIC runs, PLE, CPD benchmark, plotting
│   │   ├── results/            # Detected CPs, parameters, visualizations
│   │   ├── tcpd_penalties/     # TCPD-style penalty comparison
│   │   ├── ple/                # Profile-likelihood outputs
│   │   ├── d2d_base/           # Data2Dynamics baseline (MATLAB) model/data
│   │   └── identifiability/    # Structural-identifiability analysis
│   ├── toy_sird_sensitivity/   # Toy-SIRD SP/NSP sensitivity study (two-channel observation)
│   │   ├── scripts/            # Detection runs, PLE (conditional/joint), plotting
│   │   └── results/            # Fit metrics (JSON), PLE profiles (CSV), summary tables
│   └── wind_turbine/           # Wind-turbine power-curve case study
│       ├── scripts/            # Objective/kappa runs, PLE, plotting
│       ├── results/            # Detected CPs, parameters, visualizations
│       └── ple/                # Profile-likelihood outputs
├── benchmarking/
│   ├── toy/                    # Nine synthetic toy datasets + MICA vs. baselines
│   └── tcpd/                   # 42-series Turing Change Point Dataset benchmark
│       ├── main/               # Main MICA TCPD comprehensive results
│       ├── additional_baselines/
│       ├── competitor_scripts/ # R / Python / Julia baseline runners
│       └── dataset/            # TCPD data and annotations
└── ple/
    ├── covid/                  # COVID PLE results + scripts
    └── wind_turbine/           # Wind-turbine PLE results + scripts
```

## Installing the Julia packages

From the repository root, activate the local copies directly:

```julia
using Pkg
Pkg.activate("codes/Mica.jl")        # MICA package
# or
Pkg.activate("codes/BreakpointProfiles.jl")  # PLE engine
```

To instantiate dependencies:

```bash
cd MICA_jl
julia -e 'using Pkg; Pkg.activate("codes/Mica.jl"); Pkg.instantiate()'
julia -e 'using Pkg; Pkg.activate("codes/BreakpointProfiles.jl"); Pkg.instantiate()'
```

All Julia scripts in this repository are written assuming they are run from the repository root, so that relative paths such as `codes/Mica.jl`, `applications/covid/...`, and `benchmarking/toy/...` resolve correctly.

## Reproducing key experiments

### 1. COVID-19 application

Run BIC/MDL/AIC changepoint detection on the 400-day Germany COVID-19 data:

```bash
julia applications/covid/scripts/run_covid_objectives.jl
```

Plot the results:

```bash
julia applications/covid/scripts/plot_covid_visualization.jl
```

Run the profile-likelihood analysis (conditional or joint):

```bash
julia applications/covid/ple/scripts/covid_ple_bobyqa.jl
```

Run the GA multi-seed stability analysis (standard and high budget) and
generate the summary rugplot:

```bash
julia applications/covid/scripts/run_covid_bic_multiseed.jl
julia applications/covid/scripts/run_covid_bic_multiseed_set2.jl
python applications/covid/scripts/summarize_covid_stability_ga_only.py
```

Benchmark MICA against standard CPD methods on the case-number series:

```bash
python applications/covid/scripts/benchmark_covid_cpd_winner.py \
    --label bic --cps-csv applications/covid/results/covid_detected_cps_bic.csv \
    --out-dir outputs/TASK_G/bic
```

### 2. Wind-turbine application

Run objectives and κ sensitivity:

```bash
julia applications/wind_turbine/scripts/run_turbine_objectives_and_kappa.jl
```

Plot the BIC/MDL/AIC comparison:

```bash
julia applications/wind_turbine/scripts/plot_turbine_visualization.jl
```

Run the wind-turbine PLE:

```bash
julia applications/wind_turbine/ple/scripts/turbine_ple_hybrid.jl
```

### 3. Toy benchmarking

Generate / run MICA on the nine toy datasets:

```bash
julia benchmarking/toy/run_mica_with_simulations.jl 1   # dataset index 1..9
```

Aggregate and plot competitor results:

```bash
python benchmarking/toy/aggregate_mica_results.py
python benchmarking/toy/generate_toy_benchmark_figures.py
```

### 4. TCPD benchmarking

Run the main MICA comprehensive benchmark on the 42 TCPD series:

```bash
julia benchmarking/tcpd/main/benchmark_tcpd_comprehensive.jl
```

Generate the summary figure and LaTeX table:

```bash
python benchmarking/tcpd/main/update_tcpd_continuity_publication.py
```

Run Julia baseline competitors:

```bash
julia benchmarking/tcpd/competitor_scripts/run_mica.jl
julia benchmarking/tcpd/competitor_scripts/run_mica_tcpd_penalties.jl
```

R and Python baseline wrappers are in `benchmarking/tcpd/competitor_scripts/`.

### 5. Profile-likelihood / identifiability

PLE figures are generated with Julia/Plots.jl (no Python step). Toy-SIRD
conditional and joint PLE figures (including change-point profiles):

```bash
julia applications/toy_sird_sensitivity/scripts/plot_dualobs_ple.jl
```

COVID-19 PLE figures:

```bash
julia applications/covid/scripts/visualise_ple_results.jl <results_folder>
```

### 6. Toy-SIRD SP/NSP sensitivity study

Run the four-configuration (M1-M4) detection experiment on the two-channel
toy-SIRD benchmark (36 fits; parallel over threads):

```bash
julia applications/toy_sird_sensitivity/scripts/run_sird_complexity_dualobs.jl
```

Run the profile-likelihood analysis (conditional or joint mode) and generate
all figures (summary, example fits, parameter trajectories, PLE curves):

```bash
julia applications/toy_sird_sensitivity/scripts/run_sird_ple_dualobs.jl
julia applications/toy_sird_sensitivity/scripts/plot_dualobs_summary.jl
julia applications/toy_sird_sensitivity/scripts/plot_dualobs_selected.jl
julia applications/toy_sird_sensitivity/scripts/plot_dualobs_ple.jl
```

These scripts use `codes/Mica.jl` and `codes/BreakpointProfiles_v2.jl`.

## Notes on reproducibility

- All scripts assume the working directory is the repository root (`MICA_jl/`).
- Absolute paths from the original project tree were rewritten to repo-root-relative paths.
- Some shell helpers and cluster submission scripts still contain legacy absolute paths; they are not required for local reproduction.
- The D2D (Data2Dynamics) baseline PLE for COVID-19 is in `applications/covid/d2d_base/` and requires MATLAB with the D2D toolbox.

## Citation and code repositories

If you use MICA or BreakpointProfiles.jl, please cite the paper and the relevant software:

- **Mica.jl:** https://github.com/mehdilotfi7/Mica.jl
- **BreakpointProfiles.jl:** https://github.com/mehdilotfi7/BreakpointProfiles.jl

## License

See `LICENSE`.


