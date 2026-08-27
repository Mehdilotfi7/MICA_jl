#!/usr/bin/env python3
"""Generate summary tables and comparison figures for the toy-dataset benchmark.

Data sources
------------
- Competitor package-based results are reconstructed from the per_dataset JSON
  files on Brain (9 datasets x 15 methods x 2 configs = 270 records).  The
  top-level benchmark_toydatasets_package_based.json on Brain is currently
  incomplete (only 4 records), so the per_dataset directory is used here.
- MICA TCPD-penalty results come from benchmark_toydatasets_mica_tcpd.json
  (9 datasets x 2 configs = 18 records).
"""

import json
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
TABLES_DIR = os.path.join(BASE, "tables")
FIGURES_DIR = os.path.join(BASE, "figures")

for d in (TABLES_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)


def load_competitor_records():
    """Load all 270 competitor records from results/per_dataset/*.json."""
    records = []
    pattern = os.path.join(BASE, "..", "results", "per_dataset", "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            records.extend(json.load(f))
    return records


def load_mica_records():
    """Load MICA TCPD-penalty records."""
    path = os.path.join(DATA_DIR, "benchmark_toydatasets_mica_tcpd.json")
    with open(path) as f:
        return json.load(f)


def load_metadata():
    """Load toy dataset metadata."""
    path = os.path.join(DATA_DIR, "toy_datasets.json")
    with open(path) as f:
        return json.load(f)


def dataset_label(rec):
    """Create a short, unique label for each toy dataset."""
    return f"{rec['model']} n={int(rec['n'])} noise={rec['noise_level']}"


def dataset_key(rec):
    """Unique key for joining records across sources."""
    return (rec["model"], rec["seed"], rec["noise_level"], rec["n"])


def build_dataset_index(meta):
    """Map dataset key -> label."""
    idx = {}
    for m in meta:
        idx[dataset_key(m)] = dataset_label(m)
    return idx


def add_dataset_label(records, label_map):
    for r in records:
        r["dataset"] = label_map.get(dataset_key(r), "unknown")
    return records


def summarise(records, config):
    """Mean F1/precision/recall/covering per method for one config."""
    df = pd.DataFrame(records)
    df = df[df["config"] == config]
    agg = df.groupby("method")[["f1", "precision", "recall", "covering"]].mean()
    agg = agg.reset_index()
    agg = agg.sort_values("f1", ascending=False)
    return agg


def save_summary_tables(competitor, mica):
    """Write per-config summary CSVs and a combined summary."""
    # Competitor summaries
    practical = summarise(competitor, "default")
    oracle = summarise(competitor, "oracle")

    # MICA summaries
    mica_practical = summarise(mica, "default")
    mica_oracle = summarise(mica, "oracle")

    practical_all = pd.concat([practical, mica_practical], ignore_index=True)
    oracle_all = pd.concat([oracle, mica_oracle], ignore_index=True)

    # Rename config to mode
    practical_all = practical_all.copy()
    oracle_all = oracle_all.copy()
    practical_all["mode"] = "practical"
    oracle_all["mode"] = "oracle"

    combined = pd.concat([practical_all, oracle_all], ignore_index=True)
    # Reorder columns
    combined = combined[["method", "mode", "f1", "precision", "recall", "covering"]]
    combined = combined.sort_values(["mode", "f1"], ascending=[True, False])

    # Round for display
    for col in ("f1", "precision", "recall", "covering"):
        combined[col] = combined[col].round(4)
        practical_all[col] = practical_all[col].round(4)
        oracle_all[col] = oracle_all[col].round(4)

    combined.to_csv(os.path.join(TABLES_DIR, "method_summary.csv"), index=False)
    practical_all.to_csv(os.path.join(TABLES_DIR, "method_summary_practical.csv"), index=False)
    oracle_all.to_csv(os.path.join(TABLES_DIR, "method_summary_oracle.csv"), index=False)

    return combined, practical_all, oracle_all


def plot_mean_f1_barplot(practical, oracle, with_mica=True):
    """Grouped bar plot of mean F1 per method (practical vs oracle)."""
    methods = sorted(set(practical["method"]).union(oracle["method"]))
    f1_practical = [practical.set_index("method").loc[m, "f1"] if m in practical["method"].values else 0.0 for m in methods]
    f1_oracle = [oracle.set_index("method").loc[m, "f1"] if m in oracle["method"].values else 0.0 for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - width/2, f1_practical, width, label="practical", color="#4C78A8")
    bars2 = ax.bar(x + width/2, f1_oracle, width, label="oracle", color="#F58518")

    ax.set_ylabel("Mean F1", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    title = "Toy datasets: mean F1 per method"
    if with_mica:
        title += " (MICA included)"
    else:
        title += " (competitors)"
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    suffix = "with_mica" if with_mica else "competitors"
    out_path = os.path.join(FIGURES_DIR, f"mean_f1_barplot_{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined_heatmap(records):
    """Heatmap of F1 and covering across datasets for all methods/configs."""
    df = pd.DataFrame(records)
    # Combine method + config for rows
    df["method_config"] = df["method"] + " (" + df["config"] + ")"

    # Pivot for F1 and covering
    f1_pivot = df.pivot_table(index="method_config", columns="dataset", values="f1", aggfunc="mean")
    cov_pivot = df.pivot_table(index="method_config", columns="dataset", values="covering", aggfunc="mean")

    # Ensure consistent order: sort rows by mean F1 descending
    row_order = f1_pivot.mean(axis=1).sort_values(ascending=False).index
    f1_pivot = f1_pivot.loc[row_order]
    cov_pivot = cov_pivot.loc[row_order]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.25 * len(row_order) + 2)))

    for ax, pivot, title, cmap in (
        (axes[0], f1_pivot, "F1 score", "YlGnBu"),
        (axes[1], cov_pivot, "Covering", "YlOrRd"),
    ):
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(title, fontsize=13)
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    text_color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title, fontsize=10)

    fig.suptitle("Toy datasets: all methods across the 9 datasets", fontsize=15, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "combined_comparison_f1_covering.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_direct_comparison(competitor, mica):
    """Bar plot of mean F1 including MICA-P-TCPD / MICA-O-TCPD."""
    # Summarise separately
    comp_prac = summarise(competitor, "default")
    comp_orac = summarise(competitor, "oracle")
    mica_prac = summarise(mica, "default")
    mica_orac = summarise(mica, "oracle")

    practical = pd.concat([comp_prac, mica_prac], ignore_index=True)
    oracle = pd.concat([comp_orac, mica_orac], ignore_index=True)

    # Sort by oracle F1 descending for consistent method order
    order = oracle.sort_values("f1", ascending=False)["method"].tolist()
    # Make sure all methods appear
    for m in practical["method"]:
        if m not in order:
            order.append(m)

    practical = practical.set_index("method").reindex(order).reset_index()
    oracle = oracle.set_index("method").reindex(order).reset_index()

    methods = practical["method"].tolist()
    f1_practical = practical["f1"].fillna(0).tolist()
    f1_oracle = oracle["f1"].fillna(0).tolist()

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 5))
    # MICA methods highlighted with stronger colors
    colors_prac = ["#54A24B" if "MICA" in m else "#4C78A8" for m in methods]
    colors_orac = ["#E45756" if "MICA" in m else "#F58518" for m in methods]

    ax.bar(x - width/2, f1_practical, width, label="practical", color=colors_prac)
    ax.bar(x + width/2, f1_oracle, width, label="oracle", color=colors_orac)

    ax.set_ylabel("Mean F1", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_title("Toy datasets: direct comparison (MICA-P-TCPD / MICA-O-TCPD vs competitors)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "direct_comparison_with_mica.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    competitor = load_competitor_records()
    mica = load_mica_records()
    meta = load_metadata()
    label_map = build_dataset_index(meta)

    competitor = add_dataset_label(competitor, label_map)
    mica = add_dataset_label(mica, label_map)

    # Sanity checks
    assert len(competitor) == 270, f"Expected 270 competitor records, got {len(competitor)}"
    assert len(mica) == 18, f"Expected 18 MICA records, got {len(mica)}"

    # Summary tables
    combined, practical_all, oracle_all = save_summary_tables(competitor, mica)

    # Competitor-only bar plot
    comp_prac = practical_all[~practical_all["method"].str.startswith("MICA")]
    comp_orac = oracle_all[~oracle_all["method"].str.startswith("MICA")]
    plot_mean_f1_barplot(comp_prac, comp_orac, with_mica=False)

    # MICA-included bar plot
    plot_mean_f1_barplot(practical_all, oracle_all, with_mica=True)

    # Combined heatmap of all records (competitors + MICA)
    all_records = competitor + mica
    plot_combined_heatmap(all_records)

    # Direct comparison figure with MICA
    plot_direct_comparison(competitor, mica)

    # Print concise result summary
    print("=" * 60)
    print("Toy benchmark summary")
    print("=" * 60)
    print("\nBest practical method:")
    print(comp_prac.head(1).to_string(index=False))
    print("\nBest oracle method:")
    print(comp_orac.head(1).to_string(index=False))
    print("\nMICA practical:")
    print(practical_all[practical_all["method"].str.startswith("MICA")].to_string(index=False))
    print("\nMICA oracle:")
    print(oracle_all[oracle_all["method"].str.startswith("MICA")].to_string(index=False))
    print("\nGenerated files:")
    for root, dirs, files in os.walk(BASE):
        for f in files:
            if f.endswith((".csv", ".png", ".json")):
                print("  ", os.path.join(root, f))


if __name__ == "__main__":
    main()
