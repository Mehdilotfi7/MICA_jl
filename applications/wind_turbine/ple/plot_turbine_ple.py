import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT = Path("ple/wind_turbine")
SRC = Path("applications/wind_turbine/ple/conditional_hybridBase_bobyqaProf_20core")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def read_summary(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "parameter": row["parameter"],
                "index": int(row["index"]),
                "best_value": float(row["best_value"]),
                "best_loss": float(row["best_loss"]),
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
                "identifiable": row["identifiable"].lower() == "true",
                "threshold": float(row["threshold"]),
                "relative_width": float(row["relative_width"]),
                "n_failed": int(row["n_failed"]),
                "best_found_loss": float(row["best_found_loss"]),
            })
    return rows


def read_curves(path):
    curves = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            curves[row["parameter"]].append({
                "value": float(row["value"]),
                "loss": float(row["loss"]),
                "delta_loss": float(row["delta_loss"]),
            })
    return curves


def plot_identifiability(summary, title, outfile):
    params = [s["parameter"] for s in summary]
    colors = ["#2ca02c" if s["identifiable"] else "#d62728" for s in summary]
    rel_widths = [max(s["relative_width"], 1e-12) for s in summary]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(params, rel_widths, color=colors)
    ax.set_xlabel("Relative CI width (log scale)")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Identifiable"),
        Patch(facecolor="#d62728", label="Not identifiable"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    fig.savefig(outfile.with_suffix(".pdf"))
    plt.close()


def plot_profile_curves(summary, curves, title, outfile, n_cols=4):
    params = [s["parameter"] for s in summary]
    n_params = len(params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        sub = sorted(curves[param], key=lambda x: x["value"])
        vals = [x["value"] for x in sub]
        losses = [x["loss"] for x in sub]
        best = summary[idx]
        threshold = best["threshold"]

        ax.plot(vals, losses, color="steelblue", lw=1.5)
        ax.axhline(threshold, color="red", linestyle="--", lw=1)
        ax.axvline(best["best_value"], color="green", linestyle="--", lw=1)
        if best["identifiable"]:
            ax.axvline(best["ci_lower"], color="orange", linestyle=":", lw=1)
            ax.axvline(best["ci_upper"], color="orange", linestyle=":", lw=1)
        ax.set_title(f"{param}\n{'identifiable' if best['identifiable'] else 'not identifiable'}", fontsize=8)
        ax.set_xlabel("Parameter value", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=1.5, label="Profile"),
        Line2D([0], [0], color="red", linestyle="--", lw=1, label="95% threshold"),
        Line2D([0], [0], color="green", linestyle="--", lw=1, label="Best value"),
        Line2D([0], [0], color="orange", linestyle=":", lw=1, label="CI bounds"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title, y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_summary_table(summary, title, outfile):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    header = ["Parameter", "Best value", "CI lower", "CI upper", "Identifiable?"]
    cell_text = []
    for s in summary:
        cell_text.append([
            s["parameter"],
            f"{s['best_value']:.4g}",
            f"{s['ci_lower']:.4g}",
            f"{s['ci_upper']:.4g}",
            "yes" if s["identifiable"] else "no",
        ])
    table = ax.table(cellText=cell_text, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax.set_title(title, fontsize=12, pad=20)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


summary = read_summary(SRC / "ple_summary.csv")
curves = read_curves(SRC / "ple_results_curves.csv")

plot_identifiability(
    summary,
    "Wind-turbine conditional PLE: parameter identifiability\n(L2 CPD + L2 PLE, 4 CPs [140,500,1150,1860])",
    OUT / "turbine_conditional_ple_identifiability.png",
)
plot_profile_curves(
    summary,
    curves,
    "Wind-turbine conditional PLE profile curves (L2 CPD + L2 PLE)",
    OUT / "turbine_conditional_ple_profile_curves.png",
)
plot_summary_table(
    summary,
    "Wind-turbine conditional PLE summary (L2 CPD + L2 PLE)",
    OUT / "turbine_conditional_ple_summary_table.png",
)

print(f"Figures saved to {OUT}")
