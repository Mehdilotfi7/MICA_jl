# Loss-function sweep — BIC, no continuity

## What this version tests
MICA on the TCPD datasets comparing different loss functions (RSS, L1, Huber) under a single BIC objective.

## Settings
- **Script:** `benchmark_tcpd_loss_sweep.jl`
- **Optimizer:** `AnalyticalOptimizer` for models with closed-form fits; `NelderMead` otherwise.
- **Continuity:** `false`.
- **Standardization:** Yes, z-score.
- **Penalties / objectives:** Built-in `:bic` only.
- **Loss functions tested:** `rss`, `l1`, `huber`.
- **Search grid:** Dynamic per dataset: `min_seg = max(5, n ÷ 40)`, `step = max(1, n ÷ 100)`.
- **Practical selection:** Minimum BIC.
- **Model set:** 27 MICA models.
- **Evaluation metric:** F1 / Covering with margin 5.

## Result files
- `benchmark_tcpd_loss_sweep_combined.json`
- Derived CSVs: `tcpd_loss_sweep_best_per_dataset.csv`, `tcpd_loss_sweep_domain_best_loss.csv`, `tcpd_loss_sweep_model_best_loss.csv`, `tcpd_loss_sweep_model_loss_f1.csv`
- Derived figures: `tcpd_loss_sweep_domain.png`, `tcpd_loss_sweep_heatmap.png`
