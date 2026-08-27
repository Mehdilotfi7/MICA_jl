#!/usr/bin/env python3
"""Regenerate parameter and changepoint PLE figures with uniform CI coloring."""

import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.interpolate import PchipInterpolator

ROOT = Path("ple/covid/conditional_l2_neldermead_10multistart")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def parse_cps_report(report_path):
    """Read original CPs from ple_report.md."""
    import re
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r"Original CPs:.*?\[([^\]]*)\]", line)
            if m:
                return [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
    return []


def plot_parameter_profiles():
    results = pd.read_csv(ROOT / "ple_results_curves.csv")
    summary = pd.read_csv(ROOT / "ple_summary.csv")
    cps = parse_cps_report(ROOT / "ple_report.md")

    params = summary['parameter'].tolist()
    n = len(params)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        df = results[results['parameter'] == param].sort_values('value')
        s = summary[summary['parameter'] == param].iloc[0]

        best_val = s['best_value']
        threshold = s['threshold']
        ci_lower = s['ci_lower'] if pd.notna(s['ci_lower']) else None
        ci_upper = s['ci_upper'] if pd.notna(s['ci_upper']) else None
        identifiable = bool(s['identifiable'])

        vals = df['value'].values
        losses = df['loss'].values

        # Smooth monotonic interpolation through profile points (PCHIP)
        if len(vals) >= 3:
            try:
                x_new = np.linspace(vals.min(), vals.max(), 300)
                spl = PchipInterpolator(vals, losses)
                y_new = spl(x_new)
                ax.plot(x_new, y_new, '-', color='steelblue', lw=1.0)
            except Exception:
                ax.plot(vals, losses, '-', color='steelblue', lw=1.0)
        else:
            ax.plot(vals, losses, '-', color='steelblue', lw=1.0)
        ax.plot(vals, losses, 'o', color='steelblue', markersize=3)

        ax.axhline(threshold, color='crimson', ls='--', lw=0.8, label='95% threshold')
        ax.axvline(best_val, color='darkgreen', ls=':', lw=1.0, label='best fit')

        # Uniform gold/yellow CI shading regardless of identifiability
        if ci_lower is not None and ci_upper is not None:
            lo, hi = sorted([ci_lower, ci_upper])
            ax.axvspan(lo, hi, color='#ffcc00', alpha=0.25, label='approx. CI')

        vals = df['value'].values
        if np.all(vals > 0) and np.max(vals) / np.min(vals) > 10:
            ax.set_xscale('log')

        title_color = 'black'
        ax.set_title(param, fontsize=9, fontweight='bold', color=title_color)
        ax.set_xlabel('parameter value', fontsize=7)
        ax.set_ylabel('loss', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    n_ident = int(summary['identifiable'].sum())
    ref_loss = summary['best_loss'].iloc[0]
    threshold = summary['threshold'].iloc[0]
    fig.suptitle(
        f"conditional L2 Nelder-Mead | mode=conditional | CPs={cps} | "
        f"ref loss={ref_loss:.2f} | threshold={threshold:.2f} | "
        f"identifiable={n_ident}/{n}",
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    png = FIG_DIR / "ple_profiles.png"
    pdf = png.with_suffix('.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved parameter PLE figure: {png}")


def read_cp_threshold(cp_dir):
    report = cp_dir / "ple_report.md"
    if report.exists():
        text = report.read_text()
        best_loss = None
        threshold = None
        for line in text.splitlines():
            if "L2 loss (re-optimised):" in line:
                val = line.split(":")[-1].replace("**", "").strip()
                best_loss = float(val)
            if "Threshold" in line and "3.8415" in line:
                val = line.split(":")[-1].replace("**", "").strip()
                threshold = float(val)
        if best_loss is not None and threshold is not None:
            return best_loss, threshold
    cp_loss = []
    with open(cp_dir / "cp_profile_loss.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cp_loss.append(float(row["loss"]))
    best_loss = min(cp_loss)
    return best_loss, best_loss + 3.8414588206941285


def plot_cp_profiles():
    cp_dir = ROOT / ".." / "conditional_l2_neldermead_10multistart_cp_profiles"
    if not cp_dir.exists():
        print(f"CP profile directory not found: {cp_dir}")
        return

    cp_loss = []
    with open(cp_dir / "cp_profile_loss.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cp_loss.append({
                "cp_index": int(row["cp_index"]),
                "original_cp": int(row["original_cp"]),
                "candidate_cp": int(row["candidate_cp"]),
                "loss": float(row["loss"]),
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

        ax.plot(cps, losses, color='steelblue', lw=1.5, marker='o', markersize=3)
        ax.axhline(threshold, color='crimson', ls='--', lw=1, label='95% threshold')
        ax.axvline(ci["original_cp"], color='darkgreen', ls='--', lw=1, label='Detected CP')
        ax.axvline(ci["ci_lower"], color='#ffcc00', ls=':', lw=1.5, label='CI bounds')
        ax.axvline(ci["ci_upper"], color='#ffcc00', ls=':', lw=1.5)
        ax.set_title(f"CP {cp_idx} (original={ci['original_cp']})\nCI = [{ci['ci_lower']}, {ci['ci_upper']}], identifiable={ci['identifiable']}")
        ax.set_xlabel("Candidate changepoint")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("COVID changepoint profile curves (L2 CPD + L2 PLE, 10-multistart hybrid ref, Nelder-Mead)", fontsize=14)
    plt.tight_layout()
    png = FIG_DIR / "cp_profile_curves.png"
    pdf = png.with_suffix('.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved CP profile figure: {png}")


if __name__ == "__main__":
    plot_parameter_profiles()
    plot_cp_profiles()
    print("All figures done.")
