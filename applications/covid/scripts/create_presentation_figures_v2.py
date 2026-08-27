#!/usr/bin/env python3
"""Create fancy, slide-ready presentation figures for specific failure cases."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "revision/outputs"
DATA_DIR = "codes/Mica.jl/examples/Covid-model"
CHANNELS = ["infected", "hospitalized", "icu", "death"]
LABELS = ["Infected", "Hospitalized", "ICU", "Death"]
N = 400
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
    data = [x[:400] for x in data]

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
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85,
                  rc={"axes.edgecolor": "#333333", "axes.linewidth": 1.2,
                      "grid.color": "#eeeeee", "figure.facecolor": "white"})


def plot_competitor_failure(method, outname):
    """Single-slide figure for a competitor failure case (invstd_sqrt_aic)."""
    observed = load_observed()
    days = np.arange(1, observed.shape[0] + 1)
    fail = pd.read_csv(os.path.join(BASE, "TASK_G", "winners", "invstd_sqrt_aic",
                                    "simulations", f"{method}.csv"))

    fail_loss = 3459.2363919606773
    fail_bic, fail_mdl, fail_aic = selection_scores(fail_loss, 5)

    # CPs for this method (same for SegNeigh_param5.0 and PELT_param5.0)
    cps = [31, 44, 95, 194, 260]

    fancy_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    axes = axes.ravel()

    for k, (ch, lab) in enumerate(zip(CHANNELS, LABELS)):
        ax = axes[k]
        ax.fill_between(days, observed[:, k], color=COL_OBS, alpha=0.10)
        ax.plot(days, observed[:, k], color=COL_OBS, linewidth=2.2, label='Observed', zorder=10)
        ax.plot(days, fail[ch].values, color=COL_FAIL, linewidth=2.4, label=method.split('_')[0], zorder=4)
        for cp in cps:
            ax.axvline(cp, color=COL_CP, linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_ylabel(lab, fontweight='bold')
        ax.set_xlim(0, 400)
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

    fig.suptitle(
        f"{method.split('_')[0]}\n"
        f"CPs: {';'.join(map(str, cps))}  |  "
        f"AIC = {fail_aic:.0f}",
        fontsize=22, fontweight='bold', y=0.98
    )

    axes[-2].set_xlabel("Day (2020-01-27 → 2021-03-27)", fontweight='bold')
    axes[-1].set_xlabel("Day (2020-01-27 → 2021-03-27)", fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(BASE, "TASK_G_FAILD", outname)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def plot_npi_failure():
    """Single-slide figure for hand-picked reported-NPI dates."""
    observed = load_observed()
    days = np.arange(1, observed.shape[0] + 1)
    npi = pd.read_csv(os.path.join(BASE, "TASK_G_FAILD", "npi_test",
                                   "simulations", "npi_dense.csv"))

    npi_loss = 875.9916719557253
    npi_bic, _, npi_aic = selection_scores(npi_loss, 13)

    npi_cps = [42, 47, 50, 56, 85, 99, 141, 188, 234, 276, 281, 325, 345]
    npi_labels = [
        "mass-event ban", "school closures", "contact ban", "federal contact ban",
        "first easing", "further easing", "border openings", "travel warnings",
        "local restrictions", "lockdown light", "partial lockdown", "hard lockdown",
        "stricter measures"
    ]

    fancy_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    axes = axes.ravel()

    for k, (ch, lab) in enumerate(zip(CHANNELS, LABELS)):
        ax = axes[k]
        ax.fill_between(days, observed[:, k], color=COL_OBS, alpha=0.10)
        ax.plot(days, observed[:, k], color=COL_OBS, linewidth=2.2, label='Observed', zorder=10)
        ax.plot(days, npi[ch].values, color=COL_FAIL, linewidth=2.4, label='Reported NPIs', zorder=4)
        for cp in npi_cps:
            ax.axvline(cp, color=COL_CP, linestyle='--', linewidth=1.3, alpha=0.7)
        ax.set_ylabel(lab, fontweight='bold')
        ax.set_xlim(0, 400)
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

    fig.suptitle(
        "Reported NPI dates\n"
        "13 non-pharmaceutical intervention breakpoints from German news reports",
        fontsize=20, fontweight='bold', y=0.98
    )

    # summary text box
    textstr = (
        "Using every reported NPI date over-segments\n"
        "the epidemic curve and produces spurious spikes.\n\n"
        f"NPIs: 13 CPs, BIC = {npi_bic:.0f}, AIC = {npi_aic:.0f}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=COL_CP, alpha=0.95)
    axes[1].text(0.97, 0.88, textstr, transform=axes[1].transAxes, fontsize=13,
                 verticalalignment='top', horizontalalignment='right', bbox=props,
                 fontweight='bold', color='#333333')

    axes[-2].set_xlabel("Day (2020-01-27 → 2021-03-27)", fontweight='bold')
    axes[-1].set_xlabel("Day (2020-01-27 → 2021-03-27)", fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(BASE, "TASK_G_FAILD", "presentation_handpicked_NPI_failure.pdf")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "TASK_G_FAILD"), exist_ok=True)
    plot_competitor_failure("SegNeigh_param5.0", "presentation_SegNeigh_failure.pdf")
    plot_competitor_failure("PELT_param5.0", "presentation_PELT_failure.pdf")
    plot_npi_failure()
