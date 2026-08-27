#!/usr/bin/env python3
"""Plot wind-turbine κ-sensitivity curve."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def main():
    with open(RESULTS / "turbine_kappa_sensitivity.json") as f:
        recs = json.load(f)

    kappas = np.array([r["kappa"] for r in recs])
    n_cps = np.array([r["n_cps"] for r in recs])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kappas, n_cps, marker="o", lw=2, markersize=6, color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel(r"Penalty scaling $\kappa$", fontsize=12)
    ax.set_ylabel("Number of detected change points", fontsize=12)
    ax.set_title(r"Wind-turbine difference-equation model: $\kappa$-sensitivity", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-0.5, max(n_cps.max(), 5) + 0.5)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1k", "10k"])

    # Annotate stable plateau and collapse
    for r in recs:
        if r["kappa"] in [20.0, 200.0, 1000.0]:
            ax.annotate(f"{r['n_cps']} CPs", xy=(r["kappa"], r["n_cps"]),
                        textcoords="offset points", xytext=(10, 5), fontsize=9)

    fig.tight_layout()
    out_path = RESULTS / "turbine_kappa_sensitivity.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
