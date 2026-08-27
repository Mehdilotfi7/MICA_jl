#!/usr/bin/env python3
"""
Aggregate COVID 400-day TCPD-penalty results and generate summary figures.

Reads the 19 covid_400d_* folders under ../tcpd_penalties/ (each with a
summary.json and detected-CP CSV), writes:
  - ../covid_400d_tcpd_penalties_summary.json
  - ../covid_400d_tcpd_penalties_summary.csv
  - figures/covid_400d_tcpd_penalties_cps_rug.{png,pdf}
  - figures/covid_400d_tcpd_penalties_n_cps_bar.{png,pdf}
"""

import json
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep text editable in vector outputs.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10

ROOT = Path(__file__).resolve().parent  # .../MICA/applications/covid/tcpd_penalties
COVID_DIR = ROOT.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def natural_sort_key(label: str):
    """Sort labels such as tcpd_aic0 before tcpd_aic10 (not needed here but robust)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", label)]


def main():
    records = []
    for folder in sorted(ROOT.glob("covid_400d_*")):
        if not folder.is_dir():
            continue
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            print(f"Skipping {folder.name}: no summary.json")
            continue
        with summary_path.open() as f:
            data = json.load(f)

        rec = {
            "penalty_label": data.get("penalty_label", folder.name.replace("covid_400d_", "")),
            "objective": data.get("objective", folder.name.replace("covid_400d_", "")),
            "n_cps": int(data.get("n_cps", len(data.get("cps", [])))),
            "cps": list(data.get("cps", [])),
            "loss": float(data.get("loss", np.nan)),
            "raw_loss": float(data.get("raw_loss", np.nan)),
            "time_seconds": float(data.get("time_seconds", np.nan)),
        }
        records.append(rec)

    records.sort(key=lambda r: natural_sort_key(r["penalty_label"]))

    # ------------------------------------------------------------------
    # 1. Aggregate JSON
    # ------------------------------------------------------------------
    agg_json_path = COVID_DIR / "covid_400d_tcpd_penalties_summary.json"
    with agg_json_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {agg_json_path}")

    # ------------------------------------------------------------------
    # 2. Summary CSV
    # ------------------------------------------------------------------
    csv_path = COVID_DIR / "covid_400d_tcpd_penalties_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["penalty_label", "n_cps", "cps", "loss", "raw_loss"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "penalty_label": rec["penalty_label"],
                "n_cps": rec["n_cps"],
                "cps": ";".join(str(cp) for cp in rec["cps"]),
                "loss": rec["loss"],
                "raw_loss": rec["raw_loss"],
            })
    print(f"Wrote {csv_path}")

    # ------------------------------------------------------------------
    # 3. Rug plot of detected changepoints
    # ------------------------------------------------------------------
    labels = [r["penalty_label"] for r in records]
    y_positions = np.arange(len(labels))
    palette = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(14, 8))
    for y, rec, color in zip(y_positions, records, palette):
        for cp in rec["cps"]:
            ax.axvline(
                x=cp,
                ymin=(y - 0.35) / len(labels),
                ymax=(y + 0.35) / len(labels),
                color=color,
                linewidth=1.2,
                alpha=0.9,
            )
        # Add a small text annotation for CP count at the right end
        ax.text(
            405,
            y,
            f"n={rec['n_cps']}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(-5, 450)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Day (400-day COVID window)", fontsize=12)
    ax.set_title(
        "Detected changepoints per TCPD penalty objective — COVID-19 Germany (400 days)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    rug_path_png = FIG_DIR / "covid_400d_tcpd_penalties_cps_rug.png"
    rug_path_pdf = FIG_DIR / "covid_400d_tcpd_penalties_cps_rug.pdf"
    plt.savefig(rug_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(rug_path_pdf, bbox_inches="tight")
    print(f"Wrote {rug_path_png}")
    print(f"Wrote {rug_path_pdf}")

    # ------------------------------------------------------------------
    # 4. Bar plot of #CPs per penalty
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    n_cps = [r["n_cps"] for r in records]
    bars = ax.barh(
        y_positions,
        n_cps,
        color=palette,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of detected changepoints", fontsize=12)
    ax.set_title(
        "Number of changepoints per TCPD penalty — COVID-19 Germany (400 days)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(0, max(n_cps) + 2 if n_cps else 10)
    for bar, val in zip(bars, n_cps):
        ax.text(
            val + 0.15,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            ha="left",
            fontsize=9,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    bar_path_png = FIG_DIR / "covid_400d_tcpd_penalties_n_cps_bar.png"
    bar_path_pdf = FIG_DIR / "covid_400d_tcpd_penalties_n_cps_bar.pdf"
    plt.savefig(bar_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(bar_path_pdf, bbox_inches="tight")
    print(f"Wrote {bar_path_png}")
    print(f"Wrote {bar_path_pdf}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    for rec in records:
        print(
            f"{rec['penalty_label']:25s}  n_cps={rec['n_cps']:2d}  "
            f"loss={rec['loss']:10.2f}  raw_loss={rec['raw_loss']:10.2f}"
        )


if __name__ == "__main__":
    main()
