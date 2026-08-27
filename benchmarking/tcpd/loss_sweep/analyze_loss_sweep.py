#!/usr/bin/env python3
"""
Analyze loss-function sweep results.

Inputs:
- benchmark_tcpd_loss_sweep_combined.json
- tcpd_newresults_per_dataset.csv (for domain labels)

Outputs:
- TCPD_LOSS_SWEEP_SUMMARY.md
- tcpd_loss_sweep_best_per_dataset.csv
- tcpd_loss_sweep_model_loss_f1.csv
- tcpd_loss_sweep_domain_best_loss.csv
- tcpd_loss_sweep_heatmap.png
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load results
results = pd.read_json("benchmark_tcpd_loss_sweep_combined.json")
print(f"Loaded {len(results)} loss-sweep records")

# Load domain labels from main results
per_dataset = pd.read_csv("benchmarking/TCPD/tcpd_newresults_per_dataset.csv")
domain_map = per_dataset.set_index("dataset")["domain"].to_dict()
results["domain"] = results["dataset"].map(domain_map).fillna("Other")

# Best loss per dataset (highest F1 under BIC selection)
best_per_dataset = results.loc[results.groupby("dataset")["f1"].idxmax()].copy()
best_per_dataset = best_per_dataset.sort_values("f1", ascending=False)

# Best loss per dataset-model pair
best_per_dataset_model = results.loc[results.groupby(["dataset", "model"])["f1"].idxmax()].copy()

# Best loss per model (averaged over datasets where it was best)
model_best_loss = []
for model, grp in best_per_dataset_model.groupby("model"):
    loss_counts = grp["loss_function"].value_counts()
    best_loss = loss_counts.index[0]
    model_best_loss.append({
        "model": model,
        "best_loss": best_loss,
        "best_loss_count": int(loss_counts.iloc[0]),
        "mean_f1": grp["f1"].mean(),
        "mean_covering": grp["covering"].mean(),
    })
model_best_loss = pd.DataFrame(model_best_loss).sort_values("mean_f1", ascending=False)

# Mean F1 per model-loss combination
model_loss_f1 = results.groupby(["model", "loss_function"])["f1"].mean().unstack(fill_value=np.nan)
model_loss_covering = results.groupby(["model", "loss_function"])["covering"].mean().unstack(fill_value=np.nan)

# Mean F1 per domain-loss (across all datasets in domain)
domain_loss_f1 = results.groupby(["domain", "loss_function"])["f1"].mean().unstack(fill_value=np.nan)
# Best loss per domain (count how often each loss wins)
best_per_dataset_domain = results.loc[results.groupby(["dataset"])["f1"].idxmax()].copy()
domain_best_loss = []
for domain, grp in best_per_dataset_domain.groupby("domain"):
    loss_counts = grp["loss_function"].value_counts()
    best_loss = loss_counts.index[0]
    domain_all = results[results["domain"] == domain]
    row = {
        "domain": domain,
        "n_datasets": len(grp),
        "best_loss": best_loss,
        "best_loss_count": int(loss_counts.iloc[0]),
        "mean_f1": grp["f1"].mean(),
    }
    for loss in ["rss", "l1", "huber"]:
        loss_data = domain_all[domain_all["loss_function"] == loss]["f1"]
        row[f"{loss}_mean_f1"] = loss_data.mean() if len(loss_data) > 0 else np.nan
    domain_best_loss.append(row)
domain_best_loss = pd.DataFrame(domain_best_loss).sort_values("mean_f1", ascending=False)

# Save CSVs
best_per_dataset.to_csv("tcpd_loss_sweep_best_per_dataset.csv", index=False)
model_loss_f1.to_csv("tcpd_loss_sweep_model_loss_f1.csv")
domain_best_loss.to_csv("tcpd_loss_sweep_domain_best_loss.csv", index=False)
model_best_loss.to_csv("tcpd_loss_sweep_model_best_loss.csv", index=False)

# Heatmap: model x loss function F1
fig, ax = plt.subplots(figsize=(10, 14))
im = ax.imshow(model_loss_f1.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(model_loss_f1.columns)))
ax.set_yticks(np.arange(len(model_loss_f1.index)))
ax.set_xticklabels(model_loss_f1.columns)
ax.set_yticklabels(model_loss_f1.index)
ax.set_title("Mean F1 by Model and Loss Function (BIC selection)")
ax.set_xlabel("Loss Function")
ax.set_ylabel("Model")
for i in range(len(model_loss_f1.index)):
    for j in range(len(model_loss_f1.columns)):
        val = model_loss_f1.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if val < 0.5 else "black")
plt.colorbar(im, ax=ax, label="Mean F1")
plt.tight_layout()
plt.savefig("tcpd_loss_sweep_heatmap.png", dpi=150)
plt.close()

# Bar chart: best loss per domain
fig, ax = plt.subplots(figsize=(10, 6))
domain_best_sorted = domain_best_loss.sort_values("mean_f1", ascending=True)
bars = ax.barh(domain_best_sorted["domain"], domain_best_sorted["mean_f1"])
ax.set_xlabel("Mean F1 (best loss per dataset)")
ax.set_title("Best Loss-Function Performance by Domain")
ax.set_xlim(0, 1)
for i, (domain, row) in enumerate(domain_best_sorted.iterrows()):
    ax.text(row["mean_f1"] + 0.01, i, f"{row['best_loss']}\n{row['mean_f1']:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("tcpd_loss_sweep_domain.png", dpi=150)
plt.close()

# Markdown summary
md = []
md.append("# Loss-Function Sweep Summary\n")
md.append(f"**Total runs:** {len(results)}  \n")
md.append(f"**Datasets:** {results['dataset'].nunique()}  \n")
md.append(f"**Models:** {results['model'].nunique()}  \n")
md.append(f"**Loss functions:** {', '.join(results['loss_function'].unique())}  \n\n")

md.append("## Best Loss Function per Domain\n")
md.append("| Domain | Datasets | Best Loss | Count | Mean F1 | RSS Mean F1 | L1 Mean F1 | Huber Mean F1 |\n")
md.append("|---|---|---|---|---|---|---|---|\n")
for _, row in domain_best_loss.iterrows():
    rss = f"{row['rss_mean_f1']:.3f}" if not np.isnan(row['rss_mean_f1']) else "—"
    l1 = f"{row['l1_mean_f1']:.3f}" if not np.isnan(row['l1_mean_f1']) else "—"
    huber = f"{row['huber_mean_f1']:.3f}" if not np.isnan(row['huber_mean_f1']) else "—"
    md.append(f"| {row['domain']} | {int(row['n_datasets'])} | {row['best_loss']} | {int(row['best_loss_count'])} | "
              f"{row['mean_f1']:.3f} | {rss} | {l1} | {huber} |\n")
md.append("\n")

md.append("## Best Loss Function per Model\n")
md.append("| Model | Best Loss | Times Best | Mean F1 | Mean Covering |\n")
md.append("|---|---|---|---|---|\n")
for _, row in model_best_loss.head(20).iterrows():
    md.append(f"| {row['model']} | {row['best_loss']} | {int(row['best_loss_count'])} | {row['mean_f1']:.3f} | {row['mean_covering']:.3f} |\n")
md.append("\n")

md.append("## Best Loss per Dataset\n")
md.append("| Dataset | Domain | Best Loss | F1 | Model | n_cps |\n")
md.append("|---|---|---|---|---|---|\n")
for _, row in best_per_dataset.head(30).iterrows():
    md.append(f"| {row['dataset']} | {row['domain']} | {row['loss_function']} | {row['f1']:.3f} | {row['model']} | {int(row['n_cps'])} |\n")
md.append("\n")

md.append("## Key Findings\n")
best_overall_loss = results.groupby("loss_function")["f1"].mean().sort_values(ascending=False)
md.append(f"1. **Best overall loss (mean F1):** {best_overall_loss.index[0]} ({best_overall_loss.iloc[0]:.4f})\n")
md.append(f"2. **Loss ranking:** " + " > ".join([f"{k} ({v:.3f})" for k, v in best_overall_loss.items()]) + "\n")
md.append(f"3. **Dataset where loss choice matters most:** {best_per_dataset_model.groupby('dataset')['f1'].max().idxmax()}\n")
md.append("\n")

md.append("## Files Generated\n")
md.append("- `tcpd_loss_sweep_best_per_dataset.csv`\n")
md.append("- `tcpd_loss_sweep_model_loss_f1.csv`\n")
md.append("- `tcpd_loss_sweep_model_best_loss.csv`\n")
md.append("- `tcpd_loss_sweep_domain_best_loss.csv`\n")
md.append("- `tcpd_loss_sweep_heatmap.png`\n")
md.append("- `tcpd_loss_sweep_domain.png`\n")

with open("TCPD_LOSS_SWEEP_SUMMARY.md", "w") as f:
    f.write("".join(md))

print("\n=== Loss sweep analysis complete ===")
print("Best overall loss:")
print(best_overall_loss)
print("\nDomain best loss:")
print(domain_best_loss)
print("\nTop models by mean F1:")
print(model_best_loss.head(10))
