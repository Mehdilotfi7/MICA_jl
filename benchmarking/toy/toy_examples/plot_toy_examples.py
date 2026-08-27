#!/usr/bin/env python3
"""Visualize the nine synthetic toy datasets with true changepoints.

Produces a 3x3 multipanel figure similar in style to the TCPD examples figure
(Figure S1 in the supplementary material). Each panel shows the noisy time
series, the underlying clean simulation (noise=0 counterpart), the true
changepoints, and the segment boundaries.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "data", "toy_datasets.json")


def load_datasets():
    with open(DATA_PATH) as f:
        return json.load(f)


def model_label(model):
    return {"ODE": "SIR/ODE", "LR": "piecewise-linear", "AR": "AR(1)"}[model]


def plot_toy_examples(datasets, out_base, ncols=3, nrows=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    # Color cycle for segments
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    cp_color = "#d95f02"
    sim_color = "#1b9e77"

    for idx, (ax, ds) in enumerate(zip(axes, datasets)):
        y = np.array(ds["y"])
        times = np.array(ds.get("times", np.arange(len(y))))
        cps = np.array(ds["true_cps"])
        model = ds["model"]
        noise = ds["noise_level"]
        n = ds["n"]

        # Segment boundaries: start, CPs, end
        boundaries = np.concatenate([[times[0]], cps, [times[-1] + 1]])

        # Plot each segment with a different color
        for seg_i in range(len(boundaries) - 1):
            t0 = boundaries[seg_i]
            t1 = boundaries[seg_i + 1]
            mask = (times >= t0) & (times < t1)
            if seg_i == len(boundaries) - 2:
                mask = (times >= t0)
            ax.plot(times[mask], y[mask], color=colors[seg_i], lw=1.5, zorder=2)

        # True changepoints as dashed vertical lines
        for cp in cps:
            ax.axvline(x=cp, color=cp_color, linestyle="--", lw=1.2, zorder=3)

        # Title
        label = model_label(model)
        cp_str = ", ".join(str(int(c)) for c in cps)
        ax.set_title(f"{label} | noise={noise} | n={n} | CPs=[{cp_str}]", fontsize=10)

        # Axis labels for bottom row
        if idx >= ncols * (nrows - 1):
            ax.set_xlabel("Time", fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("Value", fontsize=9)

        # Clean PLOS-style spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=1.5, label="Segment"),
        Line2D([0], [0], color=cp_color, lw=1.2, linestyle="--", label="True CP"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


def main():
    datasets = load_datasets()
    out_base = os.path.join(BASE, "toy_examples_3x3")
    plot_toy_examples(datasets, out_base)


if __name__ == "__main__":
    main()
