#!/usr/bin/env python3
"""
TASK_C — Alternative visualisations for changepoint-to-NPI alignment.

Generates figures in revision/outputs/TASK_C/figures/alternatives/:
  - task_c_alt_event_bands.png          : rug plot with ±14-day NPI event bands
  - task_c_alt_event_distance.png       : distance from each CP to nearest NPI event
  - task_c_alt_alignment_scores.png     : bar chart of % CPs within 7/14/21 days of NPI
  - task_c_alt_density_heatmap.png      : CP density heatmap by month
  - task_c_alt_event_windows.png        : CPs inside/outside NPI ±14-day windows

Usage:
    python plot_task_c_alignment_alternatives.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = Path(__file__).resolve().parents[4]  # repository root
OUT_DIR = BASE_DIR / "applications" / "covid" / "figures" / "task_c" / "alternatives"
TASK_C_DIR = BASE_DIR / "applications" / "covid" / "results" / "task_c"
TASK_RANDOM_DIR = BASE_DIR / "applications" / "covid" / "results" / "task_random"
TASK_G_DIR = BASE_DIR / "applications" / "covid" / "results" / "task_g" / "winners"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TASK_C_DIR.mkdir(parents=True, exist_ok=True)
TASK_RANDOM_DIR.mkdir(parents=True, exist_ok=True)
TASK_G_DIR.mkdir(parents=True, exist_ok=True)

BASE_DATE = pd.Timestamp("2020-01-27")
START_DATE = pd.Timestamp("2020-02-15")
END_DATE = pd.Timestamp("2021-03-15")

WINDOWS = [7, 14, 21]


def idx_to_date(idx):
    return BASE_DATE + pd.Timedelta(days=int(idx) - 1)


def load_cps(col="cp"):
    def loader(path):
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path)
        if col not in df.columns:
            return []
        vals = df[col].dropna().astype(int).tolist()
        return sorted([idx_to_date(v) for v in vals])
    return loader


def load_cps_index(path):
    return load_cps("cp_index")(path)


def load_selected(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        label = f"{row['family']}"
        cps = []
        if pd.notna(row.get("cps", "")) and str(row["cps"]).strip():
            cps = [idx_to_date(int(x)) for x in str(row["cps"]).split(";") if x.strip()]
        out[label] = sorted(cps)
    return out


def load_all_cp_sets():
    """Return dict of label -> sorted cp dates."""
    sets = {}

    # MICA methods
    sets["MICA zero-penalty"] = load_cps_index(os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_zero_penalty.csv"))
    sets["MICA BIC"] = load_cps_index(os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_BIC.csv"))
    sets["MICA MDL"] = load_cps_index(os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_MDL.csv"))
    sets["MICA AIC"] = load_cps_index(os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_AIC.csv"))

    # Random CP baseline
    for n in range(2, 9):
        sets[f"Random {n} CPs"] = load_cps(f"cp")(os.path.join(TASK_RANDOM_DIR, f"random_cps_{n}.csv"))

    # Task G competitors
    for winner in ["bic", "mdl"]:
        selected_path = os.path.join(TASK_G_DIR, winner, f"selected_{winner}.csv")
        for label, cps in load_selected(selected_path).items():
            sets[f"{label} ({winner.upper()})"] = cps

    return sets


def load_policy_events():
    df = pd.read_csv(os.path.join(TASK_C_DIR, "german_policy_timeline.csv"), parse_dates=["date"])
    return df


def nearest_event_distance(cp_date, events):
    """Return minimum absolute days from cp_date to any event date."""
    return (events["date"] - cp_date).abs().dt.days.min()


def alignment_counts(cps, events, windows):
    """Return list of counts of CPs within each window."""
    if not cps:
        return [0] * len(windows)
    counts = []
    for w in windows:
        n = 0
        for cp in cps:
            if nearest_event_distance(cp, events) <= w:
                n += 1
        counts.append(n)
    return counts


def plot_event_bands(all_sets, events, out_path):
    """Rug plot with shaded ±14-day bands around each NPI event."""
    fig, ax = plt.subplots(figsize=(16, 10))
    rows = list(all_sets.items())
    y_positions = np.arange(len(rows))[::-1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))

    # shaded event bands
    for _, ev in events.iterrows():
        start = ev["date"] - pd.Timedelta(days=14)
        end = ev["date"] + pd.Timedelta(days=14)
        ax.axvspan(start, end, color="green", alpha=0.08, zorder=0)

    for y, (label, cps) in zip(y_positions, rows):
        ax.scatter(cps, [y] * len(cps), color=colors[y], s=70, zorder=3, marker="|", linewidths=2)
        ax.text(START_DATE - pd.Timedelta(days=5), y, label, ha="right", va="center", fontsize=8)

    # event labels on top
    for _, ev in events.iterrows():
        if START_DATE <= ev["date"] <= END_DATE:
            ax.axvline(ev["date"], color="black", linestyle=":", linewidth=0.7, alpha=0.5, zorder=1)
            ax.text(ev["date"], len(rows) + 0.5, ev["event"], rotation=60, ha="left", va="bottom", fontsize=6, alpha=0.8)

    ax.set_xlim(START_DATE, END_DATE)
    ax.set_ylim(-1, len(rows) + 1)
    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Task C: Changepoints vs. ±14-day NPI event windows", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    green_patch = mpatches.Patch(color="green", alpha=0.15, label="±14 days around NPI event")
    ax.legend(handles=[green_patch], loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_event_distance(all_sets, events, out_path):
    """For each CP, show distance to nearest NPI event."""
    fig, ax = plt.subplots(figsize=(12, 14))
    rows = list(all_sets.items())
    y_positions = np.arange(len(rows))[::-1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))

    for y, (label, cps) in zip(y_positions, rows):
        if not cps:
            continue
        dists = [nearest_event_distance(cp, events) for cp in cps]
        # jitter y slightly for visibility
        jitter = np.random.uniform(-0.15, 0.15, size=len(dists))
        ax.scatter(dists, [y + j for j in jitter], color=colors[y], s=50, zorder=3, alpha=0.8)
        ax.axhline(y, color="lightgray", linewidth=0.5, zorder=0)
        ax.text(ax.get_xlim()[1] * 1.05, y, label, ha="left", va="center", fontsize=8)

    for w in WINDOWS:
        ax.axvline(w, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(w, -0.8, f"±{w}d", ha="center", fontsize=8, color="red")

    ax.set_xlim(-2, 120)
    ax.set_ylim(-1, len(rows))
    ax.set_yticks([])
    ax.set_xlabel("Days to nearest NPI event", fontsize=12)
    ax.set_title("Task C: Distance from each detected CP to nearest German NPI event", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_alignment_scores(all_sets, events, out_path):
    """Bar chart: % of CPs within 7/14/21 days of an NPI event."""
    fig, ax = plt.subplots(figsize=(12, 14))
    rows = list(all_sets.items())
    y_positions = np.arange(len(rows))
    bar_width = 0.25
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    for i, w in enumerate(WINDOWS):
        scores = []
        for label, cps in rows:
            if not cps:
                scores.append(0)
            else:
                n = sum(1 for cp in cps if nearest_event_distance(cp, events) <= w)
                scores.append(100 * n / len(cps))
        ax.barh(y_positions + i * bar_width, scores, height=bar_width, color=colors[i], label=f"±{w} days", alpha=0.85)

    ax.set_yticks(y_positions + bar_width)
    ax.set_yticklabels([label for label, _ in rows], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of CPs aligned with NPI event", fontsize=12)
    ax.set_title("Task C: Alignment score for each method / CP set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_density_heatmap(all_sets, out_path):
    """Heatmap: methods vs month, number of CPs in each month."""
    months = pd.date_range("2020-03-01", "2021-03-01", freq="MS")
    month_labels = [m.strftime("%b %Y") for m in months]
    rows = list(all_sets.items())
    matrix = np.zeros((len(rows), len(months)))

    for i, (label, cps) in enumerate(rows):
        for cp in cps:
            for j, m in enumerate(months):
                if m <= cp < m + pd.offsets.MonthBegin(1):
                    matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=8)
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Task C: Changepoint density by month and method", fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of CPs", rotation=270, labelpad=15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_event_windows(all_sets, events, out_path):
    """Each CP is green if within ±14 days of an NPI event, red otherwise."""
    fig, ax = plt.subplots(figsize=(16, 10))
    rows = list(all_sets.items())
    y_positions = np.arange(len(rows))[::-1]

    for y, (label, cps) in zip(y_positions, rows):
        for cp in cps:
            dist = nearest_event_distance(cp, events)
            color = "green" if dist <= 14 else "red"
            ax.scatter(cp, y, color=color, s=60, zorder=3, marker="v", alpha=0.8)
        ax.text(START_DATE - pd.Timedelta(days=5), y, label, ha="right", va="center", fontsize=8)

    for _, ev in events.iterrows():
        if START_DATE <= ev["date"] <= END_DATE:
            ax.axvline(ev["date"], color="black", linestyle=":", linewidth=0.7, alpha=0.5, zorder=1)

    ax.set_xlim(START_DATE, END_DATE)
    ax.set_ylim(-1, len(rows))
    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Task C: CPs inside (green) / outside (red) ±14-day NPI window", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    green = plt.Line2D([], [], color="green", marker="v", linestyle="None", markersize=8, label="Within ±14 days")
    red = plt.Line2D([], [], color="red", marker="v", linestyle="None", markersize=8, label="Outside ±14 days")
    ax.legend(handles=[green, red], loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_sets = load_all_cp_sets()
    events = load_policy_events()

    plot_event_bands(all_sets, events, os.path.join(OUT_DIR, "task_c_alt_event_bands.png"))
    plot_event_distance(all_sets, events, os.path.join(OUT_DIR, "task_c_alt_event_distance.png"))
    plot_alignment_scores(all_sets, events, os.path.join(OUT_DIR, "task_c_alt_alignment_scores.png"))
    plot_density_heatmap(all_sets, os.path.join(OUT_DIR, "task_c_alt_density_heatmap.png"))
    plot_event_windows(all_sets, events, os.path.join(OUT_DIR, "task_c_alt_event_windows.png"))

    print("All alternative TASK_C figures saved.")


if __name__ == "__main__":
    main()
