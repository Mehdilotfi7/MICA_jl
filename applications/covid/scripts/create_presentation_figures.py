#!/usr/bin/env python3
"""Create fancy presentation figures for failed statistical CPD methods and
hand-picked changepoints. Output goes to TASK_G_FAILD."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "revision/outputs"
DATA_DIR = "codes/Mica.jl/examples/Covid-model"
CHANNELS = ["infected", "hospitalized", "icu", "death", "vaccinated"]
N = 400
N_GLOBAL = 8
N_SEGMENT = 8

# professional palette
COL_OBS = "#1a1a1a"
COL_MICA = "#00a8a8"
COL_FAIL1 = "#e63946"
COL_FAIL2 = "#f4a261"
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


def load_sim(winner, method):
    path = os.path.join(BASE, "TASK_G", "winners", winner, "simulations", f"{method}.csv")
    return pd.read_csv(path)


def load_handpicked(name):
    path = os.path.join(BASE, "TASK_G_FAILD", "handpicked_test", "simulations", f"{name}.csv")
    return pd.read_csv(path)


def fancy_style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85,
                  rc={"axes.edgecolor": "#333333", "axes.linewidth": 1.2,
                      "grid.color": "#eeeeee", "figure.facecolor": "white"})


def add_cp_lines(ax, cps, ymax=None):
    for cp in cps:
        ax.axvline(cp, color=COL_CP, linestyle='--', linewidth=1.5, alpha=0.8)


def plot_statistical_failure():
    observed = load_observed()
    days = np.arange(1, observed.shape[0] + 1)
    fail1 = load_sim("bic", "WBS_param1.0")
    fail2 = load_sim("bic", "HMMRegime_K3")

    f1_loss = 5694.4148963703365
    f1_bic, f1_mdl, f1_aic = selection_scores(f1_loss, 0)
    f2_loss = 5338.813748517137
    f2_bic, f2_mdl, f2_aic = selection_scores(f2_loss, 1)

    fancy_style()
    fig, axes = plt.subplots(5, 2, figsize=(18, 16), sharex=True)
    fig.suptitle(
        "Statistical changepoint detection can fail when plugged into the SEIRD model\n"
        "(BIC-selected benchmark on the COVID-19 winner 'bic')",
        fontsize=20, fontweight='bold', y=0.98
    )

    cols = [
        ("WBS_param1.0", "no CPs detected", fail1, COL_FAIL1, f1_loss, f1_bic, []),
        ("HMMRegime_K3", "one late CP at day 215", fail2, COL_FAIL2, f2_loss, f2_bic, [215]),
    ]

    for j, (name, subtitle, sim, color, loss, bic, cps) in enumerate(cols):
        for k, ch in enumerate(CHANNELS):
            ax = axes[k, j]
            ax.fill_between(days, observed[:, k], color=COL_OBS, alpha=0.12)
            ax.plot(days, observed[:, k], color=COL_OBS, linewidth=2.0, label='Observed', zorder=10)
            ax.plot(days, sim[ch].values, color=color, linewidth=2.2, label=name, zorder=4)
            add_cp_lines(ax, cps)
            ax.set_ylabel(ch.capitalize(), fontweight='bold')
            if k == 0:
                ax.set_title(f"{name}\n{subtitle}\nBIC = {bic:.0f}",
                             fontsize=14, fontweight='bold', pad=10)
            if k == 0 and j == 0:
                ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            ax.set_xlim(0, 400)
            ax.grid(True, alpha=0.4)

    axes[-1, 0].set_xlabel("Day (2020-01-27 → 2021-03-27)", fontweight='bold')
    axes[-1, 1].set_xlabel("Day (2020-01-27 → 2021-03-27)", fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(BASE, "TASK_G_FAILD", "presentation_statistical_CPD_failure.pdf")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def plot_handpicked_failure():
    observed = load_observed()
    days = np.arange(1, observed.shape[0] + 1)
    hand = load_handpicked("handpicked_sparse")

    hand_cps = [50, 350]
    hand_loss = 1520.8263053114251
    hand_bic, hand_mdl, hand_aic = selection_scores(hand_loss, len(hand_cps))

    fancy_style()
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        "Hand-picked policy-date changepoints also fail\n"
        "CPs at day 50 (first lockdown) and day 350 (vaccination rollout)",
        fontsize=20, fontweight='bold', y=0.98
    )

    for k, ch in enumerate(CHANNELS):
        ax = axes[k]
        ax.fill_between(days, observed[:, k], color=COL_OBS, alpha=0.12)
        ax.plot(days, observed[:, k], color=COL_OBS, linewidth=2.2, label='Observed', zorder=10)
        ax.plot(days, hand[ch].values, color=COL_FAIL1, linewidth=2.4,
                label=f'Hand-picked (BIC = {hand_bic:.0f})', zorder=4)
        for cp in hand_cps:
            ax.axvline(cp, color=COL_CP, linestyle='--', linewidth=2.0, alpha=0.9)
        if k == 0:
            ax.annotate("first lockdown", xy=(50, ax.get_ylim()[1]*0.92), xytext=(70, ax.get_ylim()[1]*0.85),
                        arrowprops=dict(arrowstyle='->', color=COL_CP, lw=1.5),
                        fontsize=12, color=COL_CP, fontweight='bold')
            ax.annotate("vaccination rollout", xy=(350, ax.get_ylim()[1]*0.92), xytext=(250, ax.get_ylim()[1]*0.85),
                        arrowprops=dict(arrowstyle='->', color=COL_CP, lw=1.5),
                        fontsize=12, color=COL_CP, fontweight='bold')
        ax.set_ylabel(ch.capitalize(), fontweight='bold')
        ax.set_xlim(0, 400)
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

    axes[-1].set_xlabel("Day (2020-01-27 → 2021-03-27)", fontweight='bold')

    # summary text box
    textstr = (
        "Hand-picked CPs miss the autumn/winter resurgence\n"
        "and produce a delayed artificial spike.\n\n"
        f"Hand-picked: 2 CPs, BIC = {hand_bic:.0f}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=COL_CP, alpha=0.95)
    axes[1].text(0.97, 0.85, textstr, transform=axes[1].transAxes, fontsize=13,
                 verticalalignment='top', horizontalalignment='right', bbox=props,
                 fontweight='bold', color='#333333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(BASE, "TASK_G_FAILD", "presentation_handpicked_CP_failure.pdf")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "TASK_G_FAILD"), exist_ok=True)
    plot_statistical_failure()
    plot_handpicked_failure()
