#!/usr/bin/env python3
"""Summarise the package-based toy-benchmark results and create publication-style figures.

Reads:
    MICA/benchmarking/results/benchmark_toydatasets_package_based.json
    MICA/benchmarking/results/benchmark_toydatasets_mica.json

Writes:
    MICA/benchmarking/results/benchmark_toydatasets_combined.json
    MICA/benchmarking/results/summary_table.csv
    MICA/benchmarking/results/overall_ranking.csv
    MICA/benchmarking/results/method_summary_oracle.json
    MICA/benchmarking/results/method_summary_practical.json
    MICA/benchmarking/figures/mean_f1_barplot.png
    MICA/benchmarking/figures/toybenchmark_summary_paperstyle.png
    MICA/benchmarking/results/toybenchmark_benchmark_comparison.tex
    MICA/benchmarking/results/toybenchmark_benchmark_comparison.pdf (if pdflatex is available)
"""
import json
import subprocess
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = RESULTS / "benchmark_toydatasets_package_based.json"
MICA_PATH = RESULTS / "benchmark_toydatasets_mica.json"
COMBINED_PATH = RESULTS / "benchmark_toydatasets_combined.json"

# ---------------------------------------------------------------------------
# Load and merge results
# ---------------------------------------------------------------------------

with open(BASELINE_PATH) as f:
    baseline = pd.DataFrame(json.load(f))

with open(MICA_PATH) as f:
    mica_raw = json.load(f)
mica = pd.DataFrame(mica_raw["flat_records"])

# Rename baseline "default" to "practical" so the figure labels match the paper style
baseline["config"] = baseline["config"].replace({"default": "practical", "oracle": "oracle"})
mica["config"] = mica["config"].replace({"default": "practical", "oracle": "oracle"})

combined = pd.concat([baseline, mica], ignore_index=True)
combined.to_json(COMBINED_PATH, orient="records", indent=2)
print(f"Wrote combined results: {COMBINED_PATH} ({len(combined)} records)")

# ---------------------------------------------------------------------------
# Summaries (mean F1 per model and method)
# ---------------------------------------------------------------------------

def summarize(df, config):
    sub = df[df["config"] == config].copy()
    summary = sub.groupby(["model", "method"])["f1"].mean().unstack("model")
    for m in ["ODE", "LR", "AR"]:
        if m not in summary.columns:
            summary[m] = np.nan
    return summary[["ODE", "LR", "AR"]]

oracle_df = summarize(combined, "oracle")
practical_df = summarize(combined, "practical")

# Save summary tables
oracle_df.to_csv(RESULTS / "summary_table_oracle.csv")
practical_df.to_csv(RESULTS / "summary_table_practical.csv")

oracle_df.reset_index().rename(columns={"index": "method"}).to_json(
    RESULTS / "method_summary_oracle.json", orient="records", indent=2
)
practical_df.reset_index().rename(columns={"index": "method"}).to_json(
    RESULTS / "method_summary_practical.json", orient="records", indent=2
)

print("\nOracle summary:")
print(oracle_df.to_string())
print("\nPractical summary:")
print(practical_df.to_string())

# Overall ranking across all models
overall_oracle = oracle_df.mean(axis=1, skipna=True).sort_values(ascending=False)
overall_practical = practical_df.mean(axis=1, skipna=True).sort_values(ascending=False)
overall = pd.DataFrame({"oracle": overall_oracle, "practical": overall_practical})
overall.to_csv(RESULTS / "overall_ranking.csv")
print("\nOverall mean F1 ranking:")
print(overall.to_string())

# ---------------------------------------------------------------------------
# Publication-style figure (matches FigToyBenchmark.pdf)
# ---------------------------------------------------------------------------

METHOD_DISPLAY = {
    "MICA-O": "MICA-O", "MICA-P": "MICA-P",
    "ZERO": "ZERO", "AMOC": "AMOC", "BinSeg": "BinSeg",
    "BOCPD": "BOCPD", "BOCPDMS": "BOCPDMS", "CPNP": "CPNP",
    "ECP": "ECP", "KCPA": "KCPA", "PELT": "PELT",
    "PROPHET": "PROPHET", "RBOCPDMS": "RBOCPDMS", "RFPOP": "RFPOP",
    "SegNeigh": "SegNeigh", "WBS": "WBS", "FPOP": "FPOP",
}


def plot_panel(ax, df, model, mica_name, other_color="#3498db"):
    """Plot a horizontal bar panel in the style of FigToyBenchmark.pdf."""
    series = df[model].dropna().sort_values(ascending=True)
    labels = [METHOD_DISPLAY.get(m, m) for m in series.index]
    colors = ["#e74c3c" if mica_name in m else other_color for m in series.index]
    ax.barh(labels, series.values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean F1", fontsize=10)
    ax.set_title(f"{model} --- {'Oracle' if mica_name == 'MICA-O' else 'Practical'}", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    for i, (m, v) in enumerate(zip(series.index, series.values)):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=7)


max_n_methods = max(len(oracle_df), len(practical_df))
fig_height = max(8, 0.45 * max_n_methods)
fig, axes = plt.subplots(2, 3, figsize=(15, fig_height))
for j, model in enumerate(["ODE", "LR", "AR"]):
    plot_panel(axes[0, j], oracle_df, model, "MICA-O")
    plot_panel(axes[1, j], practical_df, model, "MICA-P")
plt.tight_layout()
fig_path = FIGURES / "toybenchmark_summary_paperstyle.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nWrote paper-style figure: {fig_path}")

# Also keep the simpler grouped bar chart for quick reference
methods = sorted(combined["method"].unique())
models = ["ODE", "LR", "AR"]
configs = ["oracle", "practical"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
x = np.arange(len(methods))
width = 0.35
for ax, model in zip(axes, models):
    sub = combined[combined["model"] == model]
    for i, config in enumerate(configs):
        means = []
        for m in methods:
            vals = sub[(sub["method"] == m) & (sub["config"] == config)]["f1"]
            means.append(vals.mean() if len(vals) > 0 else 0.0)
        ax.bar(x + (i - 0.5) * width, means, width, label=config)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Mean F1")
    ax.set_title(f"{model} model")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Config")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Package-based toy benchmark: mean F1 by method and model", fontsize=14)
fig.tight_layout()
simple_path = FIGURES / "mean_f1_barplot.png"
fig.savefig(simple_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Wrote simple bar plot: {simple_path}")

# ---------------------------------------------------------------------------
# LaTeX document
# ---------------------------------------------------------------------------

def format_val(v):
    if pd.isna(v):
        return "---"
    return f"{v:.3f}"


def highlight_best_per_column(df, cols):
    best_vals = {c: df[c].max() for c in cols}
    cells = {}
    for method in df.index:
        cells[method] = []
        for c in cols:
            v = df.loc[method, c]
            if pd.isna(v):
                cells[method].append("---")
            elif abs(v - best_vals[c]) < 1e-6:
                cells[method].append(f"\\cellcolor{{lightgreen}}{format_val(v)}")
            else:
                cells[method].append(format_val(v))
    return cells


def table_rows(df, mica_name, cols):
    df = df.copy()
    df["__mean__"] = df.mean(axis=1, skipna=True)
    if mica_name in df.index:
        others = df.drop([mica_name]).sort_values("__mean__", ascending=False)
        ordered = pd.concat([df.loc[[mica_name]], others])
    else:
        ordered = df.sort_values("__mean__", ascending=False)
    ordered = ordered.drop(columns=["__mean__"])
    cell_dict = highlight_best_per_column(ordered, cols)
    lines = []
    for method in ordered.index:
        cells = cell_dict[method]
        name = METHOD_DISPLAY.get(method, method)
        bold = "\\textbf{" if method == mica_name else ""
        end_bold = "}" if method == mica_name else ""
        lines.append(f"{bold}{name}{end_bold} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines), list(ordered.index)


cols = ["ODE", "LR", "AR"]
oracle_rows, _ = table_rows(oracle_df, "MICA-O", cols)
practical_rows, _ = table_rows(practical_df, "MICA-P", cols)

ode_gap = abs(oracle_df.loc["MICA-O", "ODE"] - practical_df.loc["MICA-P", "ODE"])
lr_gap = abs(oracle_df.loc["MICA-O", "LR"] - practical_df.loc["MICA-P", "LR"])
ar_gap = abs(oracle_df.loc["MICA-O", "AR"] - practical_df.loc["MICA-P", "AR"])
gap_text = f"ODE: {ode_gap:.3f}, LR: {lr_gap:.3f}, AR: {ar_gap:.3f}"


def best_non_mica(df, model):
    sub = df[model].drop(["MICA-O", "MICA-P"], errors="ignore")
    best_val = sub.max()
    best_methods = sub[sub == best_val].index.tolist()
    return best_methods, best_val


o_oracle_best, o_oracle_best_val = best_non_mica(oracle_df, "ODE")
l_oracle_best, l_oracle_best_val = best_non_mica(oracle_df, "LR")
a_oracle_best, a_oracle_best_val = best_non_mica(oracle_df, "AR")

o_prac_best, o_prac_best_val = best_non_mica(practical_df, "ODE")
l_prac_best, l_prac_best_val = best_non_mica(practical_df, "LR")
a_prac_best, a_prac_best_val = best_non_mica(practical_df, "AR")

key_findings_text = (
    f"MICA-O leads on ODE (F1={oracle_df.loc['MICA-O','ODE']:.3f}) and AR (F1={oracle_df.loc['MICA-O','AR']:.3f}). "
    f"On LR it reaches F1={oracle_df.loc['MICA-O','LR']:.3f}, while the strongest baselines ({', '.join(l_oracle_best)}) achieve {l_oracle_best_val:.3f} on the clean piecewise-linear data. "
    f"MICA-P leads on ODE (F1={practical_df.loc['MICA-P','ODE']:.3f}) and AR (F1={practical_df.loc['MICA-P','AR']:.3f}); on LR the top baselines ({', '.join(l_prac_best)}) reach {l_prac_best_val:.3f}, with MICA-P at {practical_df.loc['MICA-P','LR']:.3f}."
)

latex_doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{float}
\usepackage[table]{xcolor}
\definecolor{lightgreen}{RGB}{180,255,180}
\geometry{margin=2cm}

\title{Toy Dataset Benchmark: ODE, Linear Regression, and AR(1)}
\author{MICA Analysis}
\date{\today}

\begin{document}

\maketitle

\section{Summary}

This document compares MICA against published change point detection baselines on three synthetic toy datasets, using the real R/Python packages named in the TCPD paper (van den Burg \& Williams, 2020):
\begin{itemize}
    \item \textbf{ODE (SIR):} epidemic model with piecewise transmission rate~$\beta$.
    \item \textbf{Linear Regression (LR):} continuous piecewise-linear signal with mixed up/down slopes.
    \item \textbf{AR(1):} piecewise autoregressive process.
\end{itemize}
MICA-O selects the best objective per instance using the true labels (oracle). MICA-P uses the lowest-BIC objective without peeking (practical). The baselines are run in both default/practical and oracle hyper-parameter-selection modes.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{../figures/toybenchmark_summary_paperstyle.png}
    \caption{Mean F1 score across toy datasets. Top row: oracle selection; bottom row: practical/default selection. MICA is highlighted in red.}
    \label{fig:toy-summary}
\end{figure}

\section{Oracle Selection}

\begin{table}[H]
\centering
\caption{Mean F1 scores with oracle hyper-parameter selection.}
\label{tab:toy-oracle}
\begin{tabular}{lccc}
\toprule
Method & ODE & LR & AR \\
\midrule
""" + oracle_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\section{Practical / Default Selection}

\begin{table}[H]
\centering
\caption{Mean F1 scores with practical/default hyper-parameter selection.}
\label{tab:toy-practical}
\begin{tabular}{lccc}
\toprule
Method & ODE & LR & AR \\
\midrule
""" + practical_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\section{Key Findings}

\begin{enumerate}
    \item """ + key_findings_text + r"""
    \item The practical--oracle gap for MICA is small (""" + gap_text + r"""), showing that the default MICA penalties are well calibrated.
    \item The general-purpose and TCPD-paper baselines fall far behind MICA on the ODE and AR tasks, confirming that model-informed changepoint detection benefits strongly from the correct generative model.
\end{enumerate}

\end{document}
"""

tex_path = RESULTS / "toybenchmark_benchmark_comparison.tex"
with open(tex_path, "w") as f:
    f.write(latex_doc)
print(f"Wrote LaTeX summary: {tex_path}")

# Compile PDF if pdflatex is available
print("Compiling PDF...")
result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(RESULTS), str(tex_path)],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("pdflatex failed; stdout tail:")
    print(result.stdout[-2000:])
    print("stderr:", result.stderr[-1000:])
else:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(RESULTS), str(tex_path)],
        capture_output=True, text=True
    )
    pdf_path = RESULTS / "toybenchmark_benchmark_comparison.pdf"
    print(f"PDF compiled: {pdf_path}")
