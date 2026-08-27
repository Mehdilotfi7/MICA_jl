#!/usr/bin/env python3
"""Create slide-ready 500-day COVID-19 figures for MICA winners and NPI baselines."""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

BASE = "revision/outputs"
DATA_DIR = "codes/Mica.jl/examples/Covid-model"
CHANNELS = ["infected", "hospitalized", "icu", "death"]
LABELS = ["Infected", "Hospitalized", "ICU", "Death"]
N = 500
N_GLOBAL = 8
N_SEGMENT = 8

COL_OBS = "#1a1a1a"
COL_MICA = "#00a8a8"
COL_FAIL = "#e63946"
COL_CP = "#6a4c93"


def load_observed():
    cases = pd.read_csv(os.path.join(DATA_DIR, "case_rki_daily.csv")).total.values
    hosp = pd.read_csv(os.path.join(DATA_DIR, "Hospitalization_rki_daily.csv")).total.values
    icu = pd.read_csv(os.path.join(DATA_DIR, "icu_rki_daily.csv")).total.values
    death = np.cumsum(pd.read_csv(os.path.join(DATA_DIR, "death_rki_daily.csv")).Todesfaelle_neu.values)
    vacc = np.cumsum(pd.read_csv(os.path.join(DATA_DIR, "vaccination_rki_daily_allShots.csv")).Total.values)
    data = [cases, hosp, icu, death, vacc]
    max_len = max(len(x) for x in data)
    data = [np.pad(x, (max_len - len(x), 0), constant_values=0) for x in data]
    data = [x[:N] for x in data]

    def ma14(x):
        return pd.Series(x).rolling(window=14, center=True, min_periods=1).mean().values

    data[0] = ma14(data[0])
    data[3] = ma14(data[3])
    data[4] = ma14(data[4])
    return np.column_stack(data)


def selection_scores(loss, n_cps):
    p_total = N_GLOBAL + (n_cps + 1) * N_SEGMENT + n_cps
    logn = np.log(N)
    bic = loss + p_total * logn
    mdl = loss + 0.5 * p_total * logn
    aic = loss + 2.0 * p_total
    return bic, mdl, aic


def fancy_style():
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="talk", font_scale=0.85,
                      rc={"axes.edgecolor": "#333333", "axes.linewidth": 1.2,
                          "grid.color": "#eeeeee", "figure.facecolor": "white"})
    else:
        plt.rcdefaults()
        plt.rcParams.update({
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.color": "#eeeeee",
            "figure.facecolor": "white",
            "font.size": 12,
        })


def plot_single(method, csv_path, cps, title, out_path, color=COL_MICA):
    observed = load_observed()
    days = np.arange(1, observed.shape[0] + 1)
    sim = pd.read_csv(csv_path)

    loss = np.inf
    summary_csv = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "refit_summary.csv")
    if os.path.exists(summary_csv):
        try:
            summary = pd.read_csv(summary_csv)
            row = summary[summary["method"] == method]
            if not row.empty:
                loss = float(row["refit_loss"].values[0])
                cps_from_csv = row["cps"].values[0]
                if pd.notna(cps_from_csv) and str(cps_from_csv):
                    cps = [int(x) for x in str(cps_from_csv).split(";") if x]
        except Exception:
            pass

    # Fallback: MICA summary.json
    if not np.isfinite(loss):
        summary_json = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "summary.json")
        if os.path.exists(summary_json):
            try:
                with open(summary_json) as f:
                    info = json.load(f)
                loss = info.get("raw_loss", np.inf)
                if "cps" in info:
                    cps = [int(x) for x in info["cps"]]
            except Exception:
                pass

    bic, mdl, aic = selection_scores(loss, len(cps)) if np.isfinite(loss) else (np.nan, np.nan, np.nan)

    fancy_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    axes = axes.ravel()

    for k, (ch, lab) in enumerate(zip(CHANNELS, LABELS)):
        ax = axes[k]
        ax.fill_between(days, observed[:, k], color=COL_OBS, alpha=0.10)
        ax.plot(days, observed[:, k], color=COL_OBS, linewidth=2.2, label='Observed', zorder=10)
        ax.plot(days, sim[ch].values, color=color, linewidth=2.4, label=method.replace('_', ' '), zorder=4)
        for cp in cps:
            ax.axvline(cp, color=COL_CP, linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_ylabel(lab, fontweight='bold')
        ax.set_xlim(0, N)
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

    score_text = (
        f"CPs: {len(cps)} ({';'.join(map(str, cps))})\n"
        f"Loss = {loss:.1f} | BIC = {bic:.0f} | AIC = {aic:.0f}"
    ) if np.isfinite(loss) else f"CPs: {len(cps)} ({';'.join(map(str, cps))})"

    fig.suptitle(f"{title}\n{score_text}", fontsize=20, fontweight='bold', y=0.98)
    axes[-2].set_xlabel("Day (2020-01-27 → 2021-06-11)", fontweight='bold')
    axes[-1].set_xlabel("Day (2020-01-27 → 2021-06-11)", fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mica_winners():
    task_dir = os.path.join(BASE, "TASK_G_500")
    winner_dirs = [os.path.join(task_dir, d) for d in sorted(os.listdir(task_dir))
                   if d.startswith("winners_500_") and os.path.isdir(os.path.join(task_dir, d))]
    if not winner_dirs:
        print("No MICA winner directories found in TASK_G_500")
        return

    for winner_dir in winner_dirs:
        cps_files = [f for f in os.listdir(winner_dir)
                     if f.startswith("covid_detected_cps_") and f.endswith(".csv")]
        if not cps_files:
            continue
        label = re.sub(r"^covid_detected_cps_|\.csv$", "", cps_files[0])
        csv_path = os.path.join(winner_dir, "simulations", f"{label}.csv")
        if not os.path.exists(csv_path):
            print(f"Simulation CSV not found for {label}; skipping figure")
            continue
        cps = pd.read_csv(os.path.join(winner_dir, cps_files[0])).cp.astype(int).tolist()
        out_path = os.path.join(BASE, "TASK_G_500", "figures",
                                f"presentation_MICA_{label}_500.pdf")
        plot_single(label, csv_path, cps, f"MICA winner: {label}", out_path, color=COL_MICA)


def plot_baseline(method, csv_path, cps, title, outname, color=COL_FAIL):
    if not os.path.exists(csv_path):
        print(f"Baseline CSV not found: {csv_path}")
        return
    out_path = os.path.join(BASE, "TASK_G_500", "figures", outname)
    plot_single(method, csv_path, cps, title, out_path, color=color)


def plot_comparison_summary():
    """Bar chart comparing BIC for the main 500-day models."""
    entries = []

    # MICA winners
    for mica_label, display_name in [
        ("bic", "MICA bic"),
        ("zero_penalty", "MICA zero penalty"),
    ]:
        mica_dir = os.path.join(BASE, "TASK_G_500", f"winners_500_{mica_label}")
        if not os.path.isdir(mica_dir):
            continue
        try:
            with open(os.path.join(mica_dir, "summary.json")) as f:
                info = json.load(f)
            loss = info.get("raw_loss", np.nan)
            n_cps = info.get("n_cps", 0)
            bic, _, aic = selection_scores(loss, n_cps)
            entries.append((display_name, bic, aic, loss, n_cps))
        except Exception:
            pass

    # Baselines from handpicked 500-day refit summary
    baseline_files = {
        "handpicked_major": os.path.join(BASE, "TASK_G_FAILD", "handpicked_test_500", "refit_summary.csv"),
        "handpicked_sparse": os.path.join(BASE, "TASK_G_FAILD", "handpicked_test_500", "refit_summary.csv"),
        "npi_full_500": os.path.join(BASE, "TASK_G_FAILD", "npi_full_500", "refit_summary.csv"),
    }
    for method, csv in baseline_files.items():
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        row = df[df["method"] == method]
        if row.empty:
            continue
        loss = float(row["refit_loss"].values[0])
        n_cps = int(row["n_cps"].values[0])
        bic, _, aic = selection_scores(loss, n_cps)
        entries.append((method.replace("_", " "), bic, aic, loss, n_cps))

    if len(entries) < 2:
        print("Not enough models for comparison summary; skipping")
        return

    entries.sort(key=lambda x: x[1])
    names, bics, aics, losses, ncpss = zip(*entries)

    fancy_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COL_MICA if "MICA" in n else COL_FAIL for n in names]
    bars = ax.barh(names, bics, color=colors, alpha=0.85)
    ax.set_xlabel("BIC (lower is better)", fontweight='bold')
    ax.set_title("500-day COVID-19 model comparison", fontsize=18, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, bic, aic, loss, ncps in zip(bars, bics, aics, losses, ncpss):
        width = bar.get_width()
        ax.text(width + max(bics) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{bic:.0f}  (AIC={aic:.0f}, loss={loss:.0f}, CPs={ncps})",
                va='center', fontsize=10)

    out_path = os.path.join(BASE, "TASK_G_500", "figures", "presentation_500_BIC_comparison.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(os.path.join(BASE, "TASK_G_500", "figures"), exist_ok=True)

    plot_mica_winners()

    # Hand-picked / NPI baselines (already refit on the 500-day window)
    plot_baseline(
        "handpicked_major",
        os.path.join(BASE, "TASK_G_FAILD", "handpicked_test_500", "simulations", "handpicked_major.csv"),
        [50, 85, 276, 325, 350],
        "Hand-picked major NPI schedule",
        "presentation_handpicked_major_500.pdf"
    )
    plot_baseline(
        "handpicked_sparse",
        os.path.join(BASE, "TASK_G_FAILD", "handpicked_test_500", "simulations", "handpicked_sparse.csv"),
        [50, 350],
        "Hand-picked sparse schedule",
        "presentation_handpicked_sparse_500.pdf"
    )
    plot_baseline(
        "npi_full_500",
        os.path.join(BASE, "TASK_G_FAILD", "npi_full_500", "simulations", "npi_full_500.csv"),
        [42, 47, 50, 56, 85, 99, 141, 188, 234, 276, 281, 325, 345, 407, 422, 451, 477],
        "Full reported-NPI schedule (17 CPs)",
        "presentation_NPI_full_500.pdf"
    )

    plot_comparison_summary()


if __name__ == "__main__":
    main()
