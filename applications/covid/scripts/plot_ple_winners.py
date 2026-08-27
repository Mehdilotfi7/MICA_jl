#!/usr/bin/env python3
"""Generate profile-likelihood (PLE) figure for each finished COVID winner."""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = "outputs/TASK_PLE/winners"


def parse_cps(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r"Original CPs:.*?\[([^\]]*)\]", line)
            if m:
                return [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
    return []


def plot_winner(winner_dir):
    label = os.path.basename(winner_dir)
    results = pd.read_csv(os.path.join(winner_dir, "ple_results.csv"))
    summary = pd.read_csv(os.path.join(winner_dir, "ple_summary.csv"))
    report_path = os.path.join(winner_dir, "ple_report.md")
    cps = parse_cps(report_path)

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

        ax.plot(df['value'], df['loss'], 'o-', color='steelblue', markersize=4, lw=1.2)
        ax.axhline(threshold, color='crimson', ls='--', lw=1, label='95% threshold')
        ax.axvline(best_val, color='darkgreen', ls=':', lw=1.2, label='best fit')

        if ci_lower is not None and ci_upper is not None:
            lo, hi = sorted([ci_lower, ci_upper])
            ax.axvspan(lo, hi, color='gold', alpha=0.25, label='approx. CI')

        # log scale for value if it spans more than one order of magnitude
        vals = df['value'].values
        if np.all(vals > 0) and np.max(vals) / np.min(vals) > 10:
            ax.set_xscale('log')

        ax.set_title(param, fontsize=10, fontweight='bold')
        ax.set_xlabel('parameter value', fontsize=8)
        ax.set_ylabel('loss', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8, ncol=1,
               bbox_to_anchor=(0.99, 0.99))

    ref_loss = summary['best_loss'].iloc[0]
    fig.suptitle(
        f"PLE for {label} | CPs: {cps} | ref loss = {ref_loss:.2f} | threshold = {ref_loss + 3.841:.2f}",
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    png = os.path.join(winner_dir, "ple_profiles.png")
    pdf = os.path.join(winner_dir, "ple_profiles.pdf")
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PLE figure for {label}: {png}")


def main():
    winner_dirs = sorted(glob.glob(os.path.join(ROOT, "*")))
    for d in winner_dirs:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "ple_results.csv")):
            plot_winner(d)
    print("All PLE figures done.")


if __name__ == "__main__":
    main()
