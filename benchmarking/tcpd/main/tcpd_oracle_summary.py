#!/usr/bin/env python3
"""Quick oracle-only summary bar chart for global-time MICA results."""
import json
import os
import numpy as np
import pandas as pd
from docx import Document
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE

SUFFIX = os.environ.get("TCPD_SUFFIX", "global_time_all42")
FIG_NAME = f"tcpd_summary_oracle_{SUFFIX}.png"

# Load paper tables
doc = Document(os.path.join(HERE, "..", "Extracted_Tables.docx"))

def extract_table_df(table_idx):
    table = doc.tables[table_idx]
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.set_index("Dataset")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

paper_oracle_f1 = extract_table_df(7)

# Load MICA results
with open(os.path.join(OUT_DIR, f"benchmark_tcpd_comprehensive_numerical_{SUFFIX}_oracle.json")) as f:
    mica_o_raw = pd.DataFrame(json.load(f))

mica_o = mica_o_raw.loc[mica_o_raw.groupby("dataset")["f1"].idxmax()][["dataset", "f1"]].copy()
mica_o["method"] = "MICA-O"

# Additional baselines
with open(os.path.join(HERE, "..", "additional_baselines", "benchmark_tcpd_additional_baselines_oracle.json")) as f:
    add_o_raw = pd.DataFrame(json.load(f))
add_methods = ["HMM-Regime"]
add_o_f1 = add_o_raw.pivot(index="dataset", columns="model", values="f1")[add_methods]

common_datasets = [ds for ds in paper_oracle_f1.index if ds in set(mica_o["dataset"])]
print(f"Datasets in comparison: {len(common_datasets)}")

df = paper_oracle_f1.loc[common_datasets].copy()
for col in add_o_f1.columns:
    df[col] = add_o_f1.loc[common_datasets, col].values
mica_by_ds = mica_o.set_index("dataset").loc[common_datasets]
df["MICA"] = mica_by_ds["f1"].values

methods = list(paper_oracle_f1.columns) + add_methods + ["MICA"]
df = df[methods]
mean_f1 = df.fillna(0).mean().sort_values(ascending=False)

mica_mean_f1 = mica_o[mica_o["dataset"].isin(common_datasets)]["f1"].mean()
print(f"\nMICA-O mean F1 (common): {mica_mean_f1:.4f}")
print("\nOracle F1 ranking:")
print(mean_f1)

method_display = {
    "amoc": "AMOC", "binseg": "BinSeg", "bocpd": "BOCPD", "bocpdms": "BOCPDMS",
    "cpnp": "CPNP", "ecp": "ECP", "kcpa": "KernelCPD", "pelt": "PELT",
    "prophet": "PROPHET", "rbocpdms": "RBOCPDMS", "rfpop": "RFPOP",
    "segneigh": "SegNeigh", "wbs": "WBS", "zero": "ZERO",
    "BS-RSS": "BS-RSS", "PELT-RSS": "PELT-RSS", "Bayesian-CPD": "Bayesian-CPD",
    "Kernel-CPD": "Kernel-CPD", "HMM-Regime": "HMM-Regime", "Maulik-ML": "Maulik-ML",
    "MICA": "MICA-O"
}

series = mean_f1.rename(method_display).sort_values(ascending=True)
colors = ["#e74c3c" if "MICA" in m else "#3498db" for m in series.index]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(series.index, series.values, color=colors)
ax.set_xlabel("Mean F1")
ax.set_title(f"Mean F1 ({len(common_datasets)} TCPD datasets, oracle selection)")
ax.set_xlim(0, 1)
ax.grid(axis="x", alpha=0.3)
for i, (m, v) in enumerate(zip(series.index, series.values)):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, FIG_NAME), dpi=150)
plt.close()
print(f"\nSaved figure: {FIG_NAME}")
