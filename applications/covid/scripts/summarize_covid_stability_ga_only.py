#!/usr/bin/env python3
"""Aggregate COVID-19 BIC multi-seed GA stability results (GA only, no deterministic optimizers)."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

BASE = Path("/home/mlotfi/mica_revision/second_revision/results")
OUT_FIG = Path("/home/mlotfi/mica_revision/second_revision/results/figures")
OUT_TAB = Path("/home/mlotfi/mica_revision/second_revision/results/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

START_DATE = datetime(2020, 1, 27)

def idx_to_date(idx):
    return START_DATE + timedelta(days=int(idx))

def load_summary(path):
    with open(path) as f:
        return json.load(f)

def flatten_runs(data, group_label):
    rows = []
    for entry in data:
        cps = entry.get("cps", [])
        rows.append({
            "group": group_label,
            "seed": entry.get("seed"),
            "n_cps": entry.get("n_cps", len(cps)),
            "cp1": cps[0] if len(cps) > 0 else None,
            "cp2": cps[1] if len(cps) > 1 else None,
            "cp_list": str(cps),
            "bic": entry.get("bic"),
            "raw_loss": entry.get("raw_loss"),
            "time_seconds": entry.get("time_seconds"),
        })
    return rows

rows = []
rows.extend(flatten_runs(load_summary(BASE / "covid_multiseed" / "multiseed_summary.json"), "GA standard"))
rows.extend(flatten_runs(load_summary(BASE / "covid_multiseed_set2" / "multiseed_summary.json"), "GA standard"))
rows.extend(flatten_runs(load_summary(BASE / "covid_multiseed_highbudget" / "multiseed_summary.json"), "GA high budget"))

df = pd.DataFrame(rows)
df = df.sort_values(["group", "bic"]).reset_index(drop=True)

# Save full table
csv_table = df[["group", "seed", "n_cps", "cp_list", "bic", "raw_loss"]].copy()
csv_table.columns = ["Group", "Seed", "Num_CPs", "Detected CPs (index)", "BIC", "Raw loss"]
csv_table.to_csv(OUT_TAB / "covid_ga_stability_summary.csv", index=False)

with open(OUT_TAB / "covid_ga_stability_summary.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n")
    f.write("\\hline\n")
    f.write("Group & Seed & \\# CPs & Detected CPs (index) & BIC & Raw loss \\\\\n")
    f.write("\\hline\n")
    for _, row in csv_table.iterrows():
        f.write(f"{row['Group']} & {row['Seed']} & {row['Num_CPs']} & {row['Detected CPs (index)']} & {row['BIC']:.2f} & {row['Raw loss']:.2f} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")

# Summary statistics per group
summary_stats = []
for group, gdf in df.groupby("group"):
    cp1s = gdf["cp1"].dropna().astype(int)
    cp2s = gdf["cp2"].dropna()
    summary_stats.append({
        "group": group,
        "n_runs": len(gdf),
        "n_cps_mode": gdf["n_cps"].mode().iloc[0] if not gdf["n_cps"].mode().empty else None,
        "cp1_mean": cp1s.mean(),
        "cp1_std": cp1s.std(),
        "cp1_min": cp1s.min(),
        "cp1_max": cp1s.max(),
        "cp2_n": cp2s.notna().sum(),
        "cp2_mean": cp2s.mean() if cp2s.notna().sum() > 0 else None,
        "cp2_std": cp2s.std() if cp2s.notna().sum() > 1 else None,
        "cp2_min": cp2s.min() if cp2s.notna().sum() > 0 else None,
        "cp2_max": cp2s.max() if cp2s.notna().sum() > 0 else None,
        "bic_min": gdf["bic"].min(),
        "bic_max": gdf["bic"].max(),
        "bic_mean": gdf["bic"].mean(),
    })
stats_df = pd.DataFrame(summary_stats)
stats_df.to_csv(OUT_TAB / "covid_ga_stability_stats.csv", index=False)

# Figure: rug plot of detected CPs per group
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

fig, ax = plt.subplots(figsize=(13, 5.5))

groups = ["GA standard", "GA high budget"]
colors = ["#1f77b4", "#2ca02c"]
legacy_cps = [40, 60, 80, 120, 150, 260, 290, 320]

for y, group in enumerate(groups):
    gdf = df[df["group"] == group]
    color = colors[y]
    for _, row in gdf.iterrows():
        cps = [row["cp1"], row["cp2"]]
        cps = [c for c in cps if pd.notna(c)]
        for cp in cps:
            ax.plot([idx_to_date(cp), idx_to_date(cp)], [y - 0.3, y + 0.3],
                    color=color, alpha=0.75, linewidth=2.0)
    # Place n-label on the left side of the panel, inside the plot area
    ax.text(idx_to_date(35), y, f"n = {len(gdf)}", va="center", ha="left", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

for cp in legacy_cps:
    ax.axvline(idx_to_date(cp), color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

ax.set_yticks(range(len(groups)))
ax.set_yticklabels(groups)
ax.set_xlabel("Date")
pass  # no title inside figure (PLOS style)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30, ha="right")
ax.set_xlim(idx_to_date(0), idx_to_date(400))
ax.grid(axis="x", alpha=0.3)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color="gray", linestyle="--", alpha=0.4, label="Legacy 8-CP solution")]
ax.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
fig.savefig(OUT_FIG / "fig_covid_ga_stability_rugplot.png", dpi=300)
fig.savefig(OUT_FIG / "fig_covid_ga_stability_rugplot.pdf")
plt.close()

# Reset rcParams to defaults so the BIC boxplot is not affected
plt.rcParams.update(plt.rcParamsDefault)

# Figure 2: BIC distribution per group
fig, ax = plt.subplots(figsize=(8, 5))
parts = []
for group in groups:
    gdf = df[df["group"] == group]
    parts.append(gdf["bic"].values)
bp = ax.boxplot(parts, tick_labels=groups, patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel("BIC")
ax.set_title("BIC distribution across GA runs")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
fig.savefig(OUT_FIG / "fig_covid_ga_stability_bic.png", dpi=300)
fig.savefig(OUT_FIG / "fig_covid_ga_stability_bic.pdf")
plt.close()

print(f"Saved table: {OUT_TAB / 'covid_ga_stability_summary.csv'}")
print(f"Saved table: {OUT_TAB / 'covid_ga_stability_summary.tex'}")
print(f"Saved stats: {OUT_TAB / 'covid_ga_stability_stats.csv'}")
print(f"Saved figure: {OUT_FIG / 'fig_covid_ga_stability_rugplot.png'}")
print(f"Saved figure: {OUT_FIG / 'fig_covid_ga_stability_bic.png'}")
print("\nSummary statistics:")
print(stats_df.to_string(index=False))
