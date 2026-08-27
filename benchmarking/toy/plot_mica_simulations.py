#!/usr/bin/env python3
"""Generate figures of MICA-fitted simulations for all toy-dataset/objective
combinations produced by `run_mica_with_simulations.jl`.

Figures are organized under simulations/<model>/figures/.
"""

import json
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


BASE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(BASE, "simulations")


def model_label(model):
    return {"ODE": "SIR/ODE", "LR": "piecewise-linear", "AR": "AR(1)"}[model]


def plot_single(json_path, out_path):
    with open(json_path) as f:
        rec = json.load(f)

    if "error" in rec:
        return None

    model = rec["model"]
    noise = rec["noise_level"]
    n = rec["n"]
    true_cps = rec["true_cps"]
    detected_cps = rec["cps"]
    objective = rec["objective"]
    f1 = rec["f1"]
    sim = np.array(rec["simulation"])

    # Load raw data
    ds_idx = int(os.path.basename(json_path).split("_")[0].replace("ds", ""))
    toy_path = os.path.join(BASE, "data", "toy_datasets.json")
    with open(toy_path) as f:
        datasets = json.load(f)
    data = np.array(datasets[ds_idx]["y"])
    times = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Data
    ax.plot(times, data, color="#999999", lw=0.8, alpha=0.7, label="Data", zorder=1)

    # MICA simulation, colored by detected segment
    boundaries = [0] + sorted(detected_cps) + [n]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(boundaries) - 1, 1)))
    for seg_i in range(len(boundaries) - 1):
        t0 = boundaries[seg_i]
        t1 = boundaries[seg_i + 1]
        mask = (times >= t0) & (times < t1)
        if seg_i == len(boundaries) - 2:
            mask = (times >= t0)
        ax.plot(times[mask], sim[mask], color=colors[seg_i], lw=1.8, label="MICA fit" if seg_i == 0 else "", zorder=2)

    # True changepoints
    for cp in true_cps:
        ax.axvline(x=cp, color="#1b9e77", linestyle="--", lw=1.5, alpha=0.9, zorder=3)

    # Detected changepoints
    for cp in detected_cps:
        ax.axvline(x=cp, color="#d95f02", linestyle=":", lw=1.5, alpha=0.9, zorder=3)

    title = f"{model_label(model)} | noise={noise} | {objective} | F1={f1:.3f} | D={len(detected_cps)} | T={len(true_cps)}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#999999", lw=0.8, label="Data"),
        Line2D([0], [0], color=colors[0] if len(boundaries) > 1 else "#1f77b4", lw=1.8, label="MICA fit"),
        Line2D([0], [0], color="#1b9e77", lw=1.5, linestyle="--", label="True CP"),
        Line2D([0], [0], color="#d95f02", lw=1.5, linestyle=":", label="Detected CP"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    for model in ("ODE", "LR", "AR"):
        fig_dir = os.path.join(SIM_DIR, model, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        json_paths = sorted(glob.glob(os.path.join(SIM_DIR, model, "ds*.json")))
        for json_path in json_paths:
            base = os.path.basename(json_path).replace(".json", ".png")
            out_path = os.path.join(fig_dir, base)
            plot_single(json_path, out_path)
        print(f"{model}: generated {len(json_paths)} figures in {fig_dir}")


if __name__ == "__main__":
    main()
