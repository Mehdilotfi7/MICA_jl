import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT = Path("ple/covid")
COND = ROOT / "conditional_hybridBase_bobyqaProf_20core"
JOINT = ROOT / "joint_hybridBase_bobyqaProf_20core"
CP_COND_L2 = ROOT / "conditional_l2_neldermead_10multistart_cp_profiles"
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

    fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.25)))
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
        ax.set_xlabel("Value", fontsize=7)
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
    fig, ax = plt.subplots(figsize=(10, max(6, len(summary) * 0.25)))
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
    table.set_fontsize(7)
    table.scale(1.2, 1.5)
    ax.set_title(title, fontsize=12, pad=20)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _parse_report_value(line):
    val = line.split(":")[-1].strip()
    val = val.replace("**", "").strip()
    return float(val)


def read_cp_threshold(cp_dir):
    """Extract best_loss and threshold from ple_report.md if available."""
    report = cp_dir / "ple_report.md"
    if report.exists():
        text = report.read_text()
        best_loss = None
        threshold = None
        for line in text.splitlines():
            if "L2 loss (re-optimised):" in line:
                best_loss = _parse_report_value(line)
            if "Threshold" in line and "3.8415" in line:
                threshold = _parse_report_value(line)
        if best_loss is not None and threshold is not None:
            return best_loss, threshold
    # Fallback: compute from profile data
    cp_loss = []
    with open(cp_dir / "cp_profile_loss.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cp_loss.append(float(row["loss"]))
    best_loss = min(cp_loss)
    return best_loss, best_loss + 3.8414588206941285


def plot_cp_profiles(cp_dir, title, outfile):
    cp_loss = []
    with open(cp_dir / "cp_profile_loss.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cp_loss.append({
                "cp_index": int(row["cp_index"]),
                "original_cp": int(row["original_cp"]),
                "candidate_cp": int(row["candidate_cp"]),
                "loss": float(row["loss"]),
                "delta_loss": float(row["delta_loss"]),
            })

    cp_ci = []
    with open(cp_dir / "cp_profile_ci.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cp_ci.append({
                "cp_index": int(row["cp_index"]),
                "original_cp": int(row["original_cp"]),
                "ci_lower": int(row["ci_lower"]),
                "ci_upper": int(row["ci_upper"]),
                "identifiable": row["identifiable"].lower() == "true",
            })

    by_cp = defaultdict(list)
    for row in cp_loss:
        by_cp[row["cp_index"]].append(row)

    best_loss, threshold = read_cp_threshold(cp_dir)

    n_cps = len(by_cp)
    fig, axes = plt.subplots(1, n_cps, figsize=(6 * n_cps, 5))
    if n_cps == 1:
        axes = [axes]

    for idx, (cp_idx, rows) in enumerate(sorted(by_cp.items())):
        ax = axes[idx]
        sub = sorted(rows, key=lambda x: x["candidate_cp"])
        cps = [x["candidate_cp"] for x in sub]
        losses = [x["loss"] for x in sub]
        ci = [c for c in cp_ci if c["cp_index"] == cp_idx][0]

        ax.plot(cps, losses, color="steelblue", lw=1.5, marker="o", markersize=3)
        ax.axhline(threshold, color="red", linestyle="--", lw=1, label="95% threshold")
        ax.axvline(ci["original_cp"], color="green", linestyle="--", lw=1, label="Detected CP")
        ax.axvline(ci["ci_lower"], color="#ffcc00", linestyle=":", lw=1.5, label="CI bounds")
        ax.axvline(ci["ci_upper"], color="#ffcc00", linestyle=":", lw=1.5)
        ax.set_title(f"CP {cp_idx} (original={ci['original_cp']})\nCI = [{ci['ci_lower']}, {ci['ci_upper']}], identifiable={ci['identifiable']}")
        ax.set_xlabel("Candidate changepoint")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


# Conditional PLE plots
summary_cond = read_summary(COND / "ple_summary.csv")
curves_cond = read_curves(COND / "ple_results_curves.csv")

plot_identifiability(
    summary_cond,
    "COVID conditional PLE: parameter identifiability\n(L1 CPD + L2 PLE, 2 CPs [60,150])",
    OUT / "covid_conditional_ple_identifiability.png",
)
plot_profile_curves(
    summary_cond,
    curves_cond,
    "COVID conditional PLE profile curves (L1 CPD + L2 PLE)",
    OUT / "covid_conditional_ple_profile_curves.png",
)
plot_summary_table(
    summary_cond,
    "COVID conditional PLE summary (L1 CPD + L2 PLE)",
    OUT / "covid_conditional_ple_summary_table.png",
)

# Joint PLE plots (parameter profiles are actually conditional in this "both" mode)
summary_joint = read_summary(JOINT / "ple_summary.csv")
curves_joint = read_curves(JOINT / "ple_results_curves.csv")

plot_identifiability(
    summary_joint,
    "COVID 'joint' PLE: parameter identifiability\n(L1 CPD + L2 PLE, CPs fixed during parameter profiling)",
    OUT / "covid_joint_ple_identifiability.png",
)
plot_profile_curves(
    summary_joint,
    curves_joint,
    "COVID 'joint' PLE parameter profile curves (L1 CPD + L2 PLE)",
    OUT / "covid_joint_ple_profile_curves.png",
)
plot_summary_table(
    summary_joint,
    "COVID 'joint' PLE summary (L1 CPD + L2 PLE)",
    OUT / "covid_joint_ple_summary_table.png",
)

# CP profile plots
plot_cp_profiles(
    JOINT,
    "COVID changepoint profile curves (L1 CPD + L2 PLE)",
    OUT / "covid_cp_profile_curves.png",
)

plot_cp_profiles(
    CP_COND_L2,
    "COVID changepoint profile curves (L2 CPD + L2 PLE, 10-multistart hybrid ref, Nelder-Mead)",
    OUT / "covid_conditional_l2_cp_profile_curves.png",
)

print(f"Figures saved to {OUT}")
