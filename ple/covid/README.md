# COVID-19 — reliable identifiability / PLE results

This folder collects the final, reproducible identifiability and profile-likelihood
material for the COVID-19 Germany application used in Supplementary Section S7.

## Inputs (`inputs/`)

- `detected_cps.csv` — BIC-selected change points (indices 60 and 150).
- `refit_params.csv` — Corresponding fitted parameters from `revision/outputs/TASK_A/results_bic/`.

These are the reference point for the profile-likelihood analysis.

## Scripts (`scripts/`)

- `covid_ple_bobyqa.jl` — Main PLE driver (L2 loss, hybrid GA+LBFGS reference, BOBYQA profiling).
- `covid_ple_bobyqa.jl` — Main PLE driver (L2 loss, hybrid GA+LBFGS reference, BOBYQA profiling).

## Profile-likelihood results

Four completed runs are kept here as the reproducible PLE set for COVID-19.

### 1. Conditional PLE, L1 CPD + L2 PLE — `conditional_hybridBase_bobyqaProf_20core/`

Original source: `ple/covid/conditional_hybridBase_bobyqaProf_20core/`

Configuration:
- Change points fixed at indices **60 and 150** (from BIC / L1 CPD).
- L2 Gaussian likelihood loss, χ²(1, 0.95) = 3.8415 threshold.
- Reference re-optimised with hybrid Evolutionary GA + LBFGS (`allow_better=true`).
- Profiling with NLopt BOBYQA (10k evaluations).
- 0/32 profiles found a better optimum.

Key numbers:
- Identifiable parameters: **11 / 32** (2 / 16 global, 9 / 16 segment-specific).
- Reference L2 loss: **1509.42**.

Files:
- `ple_summary.csv` / `ple_report.md` — parameter intervals.
- `ple_results.csv` / `ple_results_curves.csv` — full profiles.
- `figures/ple_profiles.png` — parameter profile figure.

### 2. Joint PLE, L1 CPD + L2 PLE — `joint_hybridBase_bobyqaProf_20core/`

Original source: `ple/covid/joint_hybridBase_bobyqaProf_20core/`

Configuration:
- Change points treated as nuisance parameters (joint adaptive PLE, window ±7 days).
- Same reference and profiling settings as the conditional run above.
- 0/32 profiles found a better optimum.

Key numbers:
- Identifiable parameters: **11 / 32**.
- Change point 1 (index 60): 95% CI **[55, 60]**.
- Change point 2 (index 150): 95% CI **[143, 157]**.
- Reference L2 loss: **1509.42**.

Files:
- `ple_summary.csv` / `ple_report.md` — parameter intervals.
- `ple_results.csv` / `ple_results_curves.csv` — full profiles.
- `cp_profile_ci.csv` / `cp_profile_loss.csv` — changepoint intervals.
- `figures/ple_profiles.png` and `figures/cp_profiles.png`.

### 3. Conditional PLE, L2 CPD + L2 PLE — `conditional_l2_neldermead_10multistart/`

Original source: `ple/covid/conditional_l2_neldermead_10multistart/`

Configuration:
- Change points fixed at indices **60 and 150** (from BIC / L2 CPD).
- L2 loss used for both CPD and PLE.
- Reference re-optimised with 10 multistart Hybrid Evolutionary GA + LBFGS.
- Profiling with Optim Nelder-Mead (Fminbox, 2000 iterations).
- 0/32 profiles found a better optimum.

Key numbers:
- Identifiable parameters: **17 / 32**.
- Reference L2 loss: **1597.66**.

Files:
- `ple_summary.csv` / `ple_report.md`.
- `ple_results.csv` / `ple_results_curves.csv`.
- `figures/ple_profiles.png`.

### 4. True joint PLE, L2 CPD + L2 PLE — `joint_l2_hybrid_true_joint/`

Original source: `revision/outputs/TASK_PLE/winners/bic_joint_l2_hybrid/`

Configuration:
- Change points treated as ordinary parameters (true joint PLE, window ±20 days, step 5).
- L2 loss used for both CPD and PLE.
- Reference re-optimised with 10 multistart Hybrid Evolutionary GA + LBFGS.
- Profiling with Optim Nelder-Mead (Fminbox, 2000 iterations).
- 1/32 profiles found a better optimum (`ᴺβ_seg2`), indicating the reference may not be the joint MLE.

Key numbers:
- Identifiable parameters: **31 / 32**.
- Reference L2 loss: **1594.31**.

Files:
- `joint_ple_summary.csv` / `ple_report.md`.
- `joint_ple_results.csv` / `joint_ple_results_curves.csv`.
- `figures/ple_profiles.png`.

## Structural identifiability (`identifiability/structural/`)

- Tool: `StructuralIdentifiability.jl`.
- Model: rational SEIRD simplification (no seasonality, constant vaccination).
- Result: all 16 unknown parameters are **globally identifiable** from the five observed outputs.

## Practical identifiability / FIM (`identifiability/fim/`)

- Source: `revision/outputs/TASK_E/`.
- Model: 8-change-point zero-penalty fit, 80 parameters.
- Result: condition number of the Gauss-Newton FIM ≈ **1.69 × 10³⁰⁷**; segment-specific vaccination-rate parameters ν are the least identifiable directions.
