#!/usr/bin/env python3
"""
TASK_RANDOM — Visualisation of random-changepoint baseline fits.

For each n_cps in 2:8, plot observed vs. simulated COVID-19 data for the
five channels, with vertical lines at the random changepoint locations.

Usage:
    python plot_random_cp_baseline.py [out_dir]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

BASE = "outputs/TASK_RANDOM"
OUT_DIR = sys.argv[1] if len(sys.argv) >= 2 else BASE

BASE_DATE = datetime(2020, 1, 27)
CHANNELS = [
    ("infected", "Infected"),
    ("hospitalized", "Hospitalized"),
    ("icu", "ICU"),
    ("death", "Cumulative deaths"),
    ("vaccinated", "Cumulative vaccinations"),
]


def date_from_idx(idx):
    return (BASE_DATE + timedelta(days=int(idx) - 1))


def log1p_transform(x):
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))


def plot_one_case(n_cps, cps, loss, df_sim, out_path, mica_loss=None, title_prefix="Random CP baseline", transform="log1p"):
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    dates = [date_from_idx(d) for d in df_sim["day"].values]
    cps_dates = [date_from_idx(int(c)) for c in cps]

    title = f"{title_prefix}: {n_cps} changepoints (loss = {loss:.2f})"
    if transform == "raw":
        title += " — raw scale"
    if mica_loss is not None:
        title += f" — MICA BIC reference loss = {mica_loss:.2f}"
    fig.suptitle(title, fontsize=14)

    for ax, (key, label) in zip(axes, CHANNELS):
        obs_raw = np.asarray(df_sim[f"{key}_obs"].values, dtype=float)
        sim_raw = np.asarray(df_sim[f"{key}_sim"].values, dtype=float)

        if transform == "log1p":
            obs = log1p_transform(obs_raw)
            sim = log1p_transform(sim_raw)
            ylabel = f"log1p({label})"
        else:
            obs = obs_raw
            sim = sim_raw
            ylabel = label

        ax.plot(dates, obs, "k-", linewidth=1.5, label="Observed", alpha=0.8, zorder=3)
        ax.plot(dates, sim, "C0-", linewidth=1.5, label="Random-CP fit" if title_prefix.startswith("Random") else "MICA BIC fit", alpha=0.8, zorder=2)

        for cp_date in cps_dates:
            ax.axvline(cp_date, color="C3", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[-1].set_xlabel("Date")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "summary.json"), "r") as f:
        summary = json.load(f)

    mica_loss = summary.get("mica_reference", {}).get("loss", None)

    for r in summary["random_results"]:
        n_cps = r["n_cps"]
        cps = r["cps"]
        loss = r["loss"]
        df_sim = pd.read_csv(os.path.join(OUT_DIR, f"random_simulation_{n_cps}.csv"))
        plot_one_case(n_cps, cps, loss, df_sim,
                      os.path.join(OUT_DIR, f"random_cp_{n_cps}.png"),
                      mica_loss=mica_loss, transform="log1p")
        plot_one_case(n_cps, cps, loss, df_sim,
                      os.path.join(OUT_DIR, f"random_cp_{n_cps}_raw.png"),
                      mica_loss=mica_loss, transform="raw")

    # Also plot the MICA BIC reference (if not already done by the Julia script)
    mica_ref = summary.get("mica_reference", {})
    if mica_ref and os.path.exists(os.path.join(OUT_DIR, "random_simulation_mica_bic.csv")):
        df_mica = pd.read_csv(os.path.join(OUT_DIR, "random_simulation_mica_bic.csv"))
        plot_one_case(
            mica_ref["n_cps"],
            mica_ref["cps"],
            mica_ref["loss"],
            df_mica,
            os.path.join(OUT_DIR, "mica_bic_reference.png"),
            mica_loss=None,
            title_prefix="MICA BIC reference",
            transform="log1p",
        )
        plot_one_case(
            mica_ref["n_cps"],
            mica_ref["cps"],
            mica_ref["loss"],
            df_mica,
            os.path.join(OUT_DIR, "mica_bic_reference_raw.png"),
            mica_loss=None,
            title_prefix="MICA BIC reference",
            transform="raw",
        )

    print("All TASK_RANDOM figures saved.")


if __name__ == "__main__":
    main()
