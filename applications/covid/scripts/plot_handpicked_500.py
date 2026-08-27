#!/usr/bin/env python3
"""500-day figure using *all* reported German NPI breakpoints (17 CPs)."""

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
N = 500
N_GLOBAL = 8
N_SEGMENT = 8

COL_OBS = "#1a1a1a"
COL_NPI = "#e63946"
COL_CP = "#6a4c93"


def load_observed(n=500):
    cases = pd.read_csv(os.path.join(DATA_DIR, "case_rki_daily.csv")).total.values
    hosp = pd.read_csv(os.path.join(DATA_DIR, "Hospitalization_rki_daily.csv")).total.values
    icu = pd.read_csv(os.path.join(DATA_DIR, "icu_rki_daily.csv")).total.values
    death = np.cumsum(pd.read_csv(os.path.join(DATA_DIR, "death_rki_daily.csv")).Todesfaelle_neu.values)
    vacc = np.cumsum(pd.read_csv(os.path.join(DATA_DIR, "vaccination_rki_daily_allShots.csv")).Total.values)
    data = [cases, hosp, icu, death, vacc]
    max_len = max(len(x) for x in data)
    data = [np.pad(x, (max_len - len(x), 0), constant_values=0) for x in data]
    data = [x[:n] for x in data]

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
    aic = loss + 2.0 * p_total
    return bic, aic


def fancy_style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85,
                  rc={"axes.edgecolor": "#333333", "axes.linewidth": 1.2,
                      "grid.color": "#eeeeee", "figure.facecolor": "white"})


def main():
    observed = load_observed(N)
    days = np.arange(1, observed.shape[0] + 1)

    npi = pd.read_csv(os.path.join(BASE, "TASK_G_FAILD", "npi_full_500", "simulations", "npi_full_500.csv"))

    npi_loss = 2676.518124745903
    npi_cps = [42, 47, 50, 56, 85, 99, 141, 188, 234, 276, 281, 325, 345, 407, 422, 451, 477]
    npi_bic, npi_aic = selection_scores(npi_loss, len(npi_cps))

    mica_loss = 1160.929310276116
    mica_bic, _ = selection_scores(mica_loss, 2)

    event_labels = {
        42: "mass-event ban",
        47: "school closures",
        50: "contact ban",
        56: "federal contact ban",
        85: "first easing",
        99: "further easing",
        141: "border openings",
        188: "travel warnings",
        234: "local restrictions",
        276: "lockdown light",
        281: "partial lockdown",
        325: "hard lockdown",
        345: "stricter measures",
        407: "reopening phase 1",
        422: "Easter lockdown / extension",
        451: "federal emergency brake",
        477: "broad May easing",
    }

    fancy_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    axes = axes.ravel()

    for k, (ch, lab) in enumerate(zip(CHANNELS, LABELS)):
        ax = axes[k]
        ax.fill_between(days, observed[:, k], color=COL_OBS, alpha=0.10)
        ax.plot(days, observed[:, k], color=COL_OBS, linewidth=2.2, label='Observed', zorder=10)
        ax.plot(days, npi[ch].values, color=COL_NPI, linewidth=2.4, label='All reported NPIs', zorder=4)
        for cp in npi_cps:
            ax.axvline(cp, color=COL_CP, linestyle='--', linewidth=1.3, alpha=0.7)
        ax.set_ylabel(lab, fontweight='bold')
        ax.set_xlim(0, N)
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

    axes[-2].set_xlabel("Day (2020-01-27 → 2021-06-09)", fontweight='bold')
    axes[-1].set_xlabel("Day (2020-01-27 → 2021-06-09)", fontweight='bold')

    fig.suptitle(
        "All reported German NPI breakpoints on a 500-day horizon\n"
        f"17 CPs  |  BIC = {npi_bic:.0f}  |  AIC = {npi_aic:.0f}",
        fontsize=20, fontweight='bold', y=0.98
    )

    # event legend / text box on hospitalized panel
    textstr = (
        "Even with every major reported NPI date, the 500-day\n"
        "fit degrades: BIC is >2× the MICA winner (BIC = 1161).\n\n"
        "NPI schedule includes the spring 2021 reopening,\n"
        "Easter lockdown, federal emergency brake, and May easing."
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=COL_CP, alpha=0.95)
    axes[1].text(0.97, 0.88, textstr, transform=axes[1].transAxes, fontsize=13,
                 verticalalignment='top', horizontalalignment='right', bbox=props,
                 fontweight='bold', color='#333333')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(BASE, "TASK_G_FAILD", "presentation_handpicked_500day.pdf")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
