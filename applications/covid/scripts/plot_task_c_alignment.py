#!/usr/bin/env python3
"""
TASK_C — Alignment of detected changepoints with German COVID-19 policy interventions.

Generates four figures in revision/outputs/TASK_C/figures/:
  - task_c_mica_methods.png           : MICA zero-penalty, BIC, MDL, AIC winners
  - task_c_random_cp_baseline.png     : random-CP baseline (2–8 CPs)
  - task_c_competitor_methods.png     : Task G selected competitor algorithms
  - task_c_all_methods.png            : unified figure with all methods above

Usage:
    python plot_task_c_alignment.py
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

BASE_DIR = Path(__file__).resolve().parents[4]  # repository root
OUT_DIR = BASE_DIR / "applications" / "covid" / "figures" / "task_c"
TASK_C_DIR = BASE_DIR / "applications" / "covid" / "results" / "task_c"
TASK_RANDOM_DIR = BASE_DIR / "applications" / "covid" / "results" / "task_random"
TASK_G_DIR = BASE_DIR / "applications" / "covid" / "results" / "task_g" / "winners"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TASK_C_DIR.mkdir(parents=True, exist_ok=True)
TASK_RANDOM_DIR.mkdir(parents=True, exist_ok=True)
TASK_G_DIR.mkdir(parents=True, exist_ok=True)

BASE_DATE = pd.Timestamp("2020-01-27")
START_DATE = pd.Timestamp("2020-02-01")
END_DATE = pd.Timestamp("2021-03-15")


def idx_to_date(idx):
    return BASE_DATE + pd.Timedelta(days=int(idx) - 1)


def date_to_idx(d):
    return (pd.Timestamp(d) - BASE_DATE).days + 1


def load_cps_from_csv(path, col="cp"):
    """Load a column of CP indices from a CSV and return sorted list of dates."""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(int).tolist()
    return sorted([idx_to_date(v) for v in vals])


def load_cps_from_selected_csv(path):
    """Load selected Task G methods: one row per family, semicolon-separated CPs."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        label = f"{row['family']} ({row['method']})"
        cps = []
        if pd.notna(row.get("cps", "")) and str(row["cps"]).strip():
            cps = [idx_to_date(int(x)) for x in str(row["cps"]).split(";") if x.strip()]
        out[label] = sorted(cps)
    return out


def load_policy_events():
    path = os.path.join(TASK_C_DIR, "german_policy_timeline.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def plot_timeline(rows, title, out_path, show_policy=True, figsize=(16, 10), event_fontsize=7):
    """rows: list of (label, [cp_dates])"""
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(rows))[::-1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))

    for y, (label, cps) in zip(y_positions, rows):
        ax.scatter(cps, [y] * len(cps), color=colors[y], s=80, zorder=3, marker="v")
        ax.hlines(y, START_DATE, END_DATE, color="lightgray", linewidth=0.5, zorder=1)
        ax.text(START_DATE - pd.Timedelta(days=5), y, label, ha="right", va="center", fontsize=9)

    if show_policy:
        events = load_policy_events()
        # keep only events within the plotted window
        events = events[(events["date"] >= START_DATE) & (events["date"] <= END_DATE)]
        for _, ev in events.iterrows():
            ax.axvline(ev["date"], color="black", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)
            ax.text(ev["date"] + pd.Timedelta(days=1), len(rows) - 0.5, ev["event"],
                    rotation=45, ha="left", va="bottom", fontsize=event_fontsize, alpha=0.85)

    ax.set_xlim(START_DATE, END_DATE)
    ax.set_ylim(-1, len(rows))
    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- MICA methods ---
    mica_rows = []
    mica_rows.append(("MICA zero-penalty (8 CPs)", load_cps_from_csv(
        os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_zero_penalty.csv"), col="cp_index")))
    mica_rows.append(("MICA BIC (2 CPs)", load_cps_from_csv(
        os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_BIC.csv"), col="cp_index")))
    mica_rows.append(("MICA MDL (2 CPs)", load_cps_from_csv(
        os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_MDL.csv"), col="cp_index")))
    mica_rows.append(("MICA AIC (3 CPs)", load_cps_from_csv(
        os.path.join(TASK_C_DIR, "cp_policy_map_Task_A_-_AIC.csv"), col="cp_index")))

    plot_timeline(mica_rows, "Task C: MICA detected changepoints vs. German NPI events",
                  os.path.join(OUT_DIR, "task_c_mica_methods.png"))

    # --- Random CP baseline ---
    random_rows = []
    for n in range(2, 9):
        cps = load_cps_from_csv(os.path.join(TASK_RANDOM_DIR, f"random_cps_{n}.csv"))
        random_rows.append((f"Random {n} CPs", cps))
    plot_timeline(random_rows, "Task C: Random-changepoint baseline vs. German NPI events",
                  os.path.join(OUT_DIR, "task_c_random_cp_baseline.png"))

    # --- Task G competitor algorithms (selected per family, BIC criterion) ---
    competitor_rows = []
    # Collect selected methods from each winner folder
    for winner in ["bic", "mdl"]:
        selected_path = os.path.join(TASK_G_DIR, winner, f"selected_{winner}.csv")
        competitor_rows.extend(list(load_cps_from_selected_csv(selected_path).items()))
    # Drop duplicates that happen to have the same label (e.g. MICA_bic may appear twice)
    seen = set()
    unique_competitor_rows = []
    for label, cps in competitor_rows:
        if label not in seen:
            seen.add(label)
            unique_competitor_rows.append((label, cps))

    plot_timeline(unique_competitor_rows,
                  "Task C: Task G competitor algorithms (BIC/MDL selected) vs. German NPI events",
                  os.path.join(OUT_DIR, "task_c_competitor_methods.png"),
                  figsize=(16, 14), event_fontsize=6)

    # --- Unified figure ---
    all_rows = []
    all_rows.append(("MICA zero-penalty", mica_rows[0][1]))
    all_rows.append(("MICA BIC", mica_rows[1][1]))
    all_rows.append(("MICA MDL", mica_rows[2][1]))
    all_rows.append(("MICA AIC", mica_rows[3][1]))
    # Add a few representative random configurations
    for label, cps in random_rows:
        if label in ["Random 2 CPs", "Random 6 CPs", "Random 8 CPs"]:
            all_rows.append((label, cps))
    # Add selected competitor algorithms
    for label, cps in unique_competitor_rows:
        all_rows.append((label, cps))

    plot_timeline(all_rows, "Task C: All detected changepoints vs. German NPI events",
                  os.path.join(OUT_DIR, "task_c_all_methods.png"),
                  figsize=(18, 20), event_fontsize=6)

    print("All TASK_C alignment figures saved.")


if __name__ == "__main__":
    main()
