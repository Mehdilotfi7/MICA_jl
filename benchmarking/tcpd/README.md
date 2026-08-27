# TCPD Benchmark — MICA vs. published CPD methods

This folder contains the Turing Change Point Detection (TCPD) benchmark used to answer Reviewer 1, Comment 1 (quantitative comparison with alternative changepoint-detection methods).

**Reference paper**
> van den Burg, G. J. J., & Williams, C. (2020). *An evaluation of change point detection algorithms.* arXiv preprint arXiv:2003.06222.

**Data**
- 42 univariate time series from the TCPD benchmark suite.
- Datasets and annotations are in `tcpd_dataset/`.

**Methods compared**

1. **14 methods from the TCPD paper** (taken from the paper's extracted tables, not rerun by us):
   AMOC, BinSeg, BOCPD, BOCPDMS, CPNP, ECP, KCPA (KernelCPD), PELT, PROPHET, RBOCPDMS, RFPOP, SegNeigh, WBS, ZERO.

2. **One additional model-based baseline** (run by us):
   HMM-Regime.

3. **MICA-O** and **MICA-P** (run by us):
   - MICA-O: oracle selection, best F1 over all model / hyperparameter configurations per dataset.
   - MICA-P: practical selection, best configuration by post-hoc BIC (no access to true labels).

**MICA experimental settings**

- Run through `Mica.jl` (`benchmark_tcpd_comprehensive.jl`).
- Each series is standardized to zero mean and unit variance before fitting.
- Candidate models (27): `mean`, `linear`, `quadratic`, `cubic`, `exponential`, `log_linear`, `power`, `mean_drift`, `ar1`, `ar2`, `ar3`, `hyperbolic`, `asymptotic_regression`, `michaelis_menten`, `weibull_growth`, `hill_function`, `log_logistic`, `double_exponential`, `rational`, `debt_dynamics`, `accelerator`, `compound_growth`, `poisson`, `negbin`, `ingarch`, `ets_aaa`, `ets_mmm`.
- Objectives: `BIC`, `MDL`, `AIC` plus a penalty sweep with κ ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0}.
- `min_seg` and `step` are chosen adaptively per dataset (e.g. `max(5, n//20)`, `max(1, n//50)`).
- Evaluation metrics: F1 with margin = 5; TCPD covering metric.
- **Level continuity:** for stateless models the initial level/intercept of segment `i+1` is constrained to match the last simulated value of segment `i`. This removes artificial jumps at changepoint boundaries while keeping slopes, curvature, and other shape parameters free per segment.

**Key files**

| File | Purpose |
|---|---|
| `Extracted_Tables.docx` | Extracted baseline tables from the TCPD paper |
| `benchmark_tcpd_comprehensive.jl` | Julia / `Mica.jl` runner for all MICA configurations |
| `benchmark_tcpd_comprehensive_numerical_all42.json` | All raw MICA results (42 datasets) |
| `benchmark_tcpd_comprehensive_numerical_all42_oracle.json` | Best-F1 (oracle) MICA result per dataset |
| `benchmark_tcpd_comprehensive_numerical_all42_practical.json` | Best-BIC (practical) MICA result per dataset |
| `benchmark_tcpd_additional_baselines_*.json` | HMM-Regime and other extra baselines |
| `update_tcpd_publication_from_paper.py` | Generates the summary bar plot and LaTeX/PDF tables |
| `tcpd_summary_multipanel.png` | Main comparison figure (mean F1, oracle + practical) |
| `tcpd_benchmark_comparison.tex` / `.pdf` | Full per-dataset comparison tables |
| `tcpd_best_config_summary_all42.csv` | Best model / config / CPs for each dataset (MICA-O and MICA-P) |
| `generate_best_figures_all42.py` | Generates best-per-dataset fitted figures |
| `tcpd_figures_best_all42/` | Best-per-dataset fitted figures (data + simulation + detected/true CPs) |
| `tcpd_dataset_overview.png` | Overview of the 42 TCPD datasets |
| `mica_comprehensive_benchmark.py` | Python helper module used by the figure scripts |

**Reproduce the main figure**

```bash
cd MICA/benchmarking/TCPD
python3 update_tcpd_publication_from_paper.py
```

This reads `Extracted_Tables.docx` and the MICA JSON files, then writes `tcpd_summary_multipanel.png` and `tcpd_benchmark_comparison.pdf`.

**Reproduce the best-per-dataset figures**

```bash
cd MICA/benchmarking/TCPD
python3 generate_best_figures_all42.py
```

Output: `tcpd_figures_best_all42/` (84 PNGs, one oracle + one practical per dataset).

**Reproduce the best-config summary table**

```bash
cd MICA/benchmarking/TCPD
python3 - <<'PY'
import json, pandas as pd
oracle = pd.DataFrame(json.load(open("benchmark_tcpd_comprehensive_numerical_all42_oracle.json")))
practical = pd.DataFrame(json.load(open("benchmark_tcpd_comprehensive_numerical_all42_practical.json")))
best_o = oracle.loc[oracle.groupby('dataset')['f1'].idxmax()]
best_p = practical.loc[practical.groupby('dataset')['f1'].idxmax()]
summary = pd.merge(best_o, best_p, on='dataset', suffixes=('_O', '_P'))
summary.to_csv("tcpd_best_config_summary_all42.csv", index=False)
PY
```

**Notes**

- Baseline numbers from the TCPD paper are treated as reported; missing/failed runs are counted as F1 = 0.0 when computing means.
- The three R-only methods from the paper (BOCPDMS, ECP, RBOCPDMS) are included through the extracted tables, not re-implemented by us.
- `benchmark_tcpd_additional_baselines.jl` references helper files that no longer exist in the repository; the corresponding JSON outputs are retained for the figure generation, but the script cannot be rerun from scratch without restoring those helpers.
