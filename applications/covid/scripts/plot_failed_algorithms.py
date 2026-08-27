#!/usr/bin/env python3
"""Create a multi-page PDF showing observed data + simulation + CPs for every
failed algorithm identified in TASK_G_FAILD/failed_methods.csv."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = "revision/outputs"
DATA_DIR = "codes/Mica.jl/examples/Covid-model"
CHANNELS = ["infected", "hospitalized", "icu", "death", "vaccinated"]


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


def main():
    observed = load_observed()
    days = np.arange(1, observed.shape[0] + 1)

    fail_path = os.path.join(BASE, "TASK_G_FAILD", "failed_methods.csv")
    failed = pd.read_csv(fail_path)

    out_pdf = os.path.join(BASE, "TASK_G_FAILD", "failed_algorithms_gof.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        for _, row in failed.iterrows():
            winner = row['winner']
            method = row['method']
            family = row['family']
            cps_str = row['cps'] if pd.notna(row['cps']) else ''
            cps = [int(x) for x in cps_str.split(';') if x]

            wdir = os.path.join(BASE, "TASK_G", "winners", winner)
            sim_path = os.path.join(wdir, "simulations", f"{method}.csv")

            fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
            fig.suptitle(
                f"{winner} — {method} ({family})\n"
                f"CPs: {cps_str if cps_str else 'none'}  |  "
                f"loss={row['refit_loss']:.1f}  |  "
                f"MICA loss={row['mica_loss']:.1f}  |  "
                f"ratio={row['loss_ratio']}",
                y=0.995
            )

            if os.path.exists(sim_path):
                sim = pd.read_csv(sim_path)
                sim_ok = True
            else:
                sim_ok = False

            for k, ch in enumerate(CHANNELS):
                ax = axes[k]
                ax.plot(days, observed[:, k], 'k-', linewidth=1.5, alpha=0.8, label='observed', zorder=10)
                if sim_ok:
                    ax.plot(days, sim[ch].values, color='#d62728', linewidth=1.5, alpha=0.8,
                            label=method, zorder=3)
                else:
                    ax.text(0.5, 0.5, 'simulation missing', transform=ax.transAxes,
                            ha='center', va='center', color='#d62728')
                if k == 0:
                    for cp in cps:
                        ax.axvline(cp, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
                ax.set_ylabel(ch)
                if k == 0:
                    ax.legend(loc='upper left', fontsize=8)

            axes[-1].set_xlabel('day')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"Saved {len(failed)} failed-algorithm GOF pages to {out_pdf}")


if __name__ == "__main__":
    main()
