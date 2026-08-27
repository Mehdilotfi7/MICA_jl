#!/usr/bin/env python3
"""Recompute wind-turbine PLE intervals using a variance-scaled chi-squared threshold.

The existing tentative_ple_bic profiles were generated with an SSE loss but used a
fixed delta-loss threshold of 3.8415. Because the residual variance is much larger
than 1, the correct threshold is:

    threshold = SSE_best + chi2(1, 0.95) * sigma^2

where sigma^2 = SSE_best / (n_obs - n_params).
"""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator


ROOT = Path("ple/wind_turbine/outputs/tentative_ple_bic")
OUT = ROOT / "corrected"
OUT.mkdir(exist_ok=True)

N_OBS = 2500
N_PARAMS = 23
CHISQ_95 = 3.8414588206941285


def smooth_profile(x, y, n=200):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return x, y
    ux, idx = np.unique(x, return_index=True)
    uy = y[idx]
    if len(ux) < 3:
        return x, y
    xi = np.linspace(ux.min(), ux.max(), n)
    yi = PchipInterpolator(ux, uy)(xi)
    return xi, yi


def compute_ci(values, losses, best_value, threshold):
    """Find contiguous CI around best_value where loss <= threshold."""
    below = losses <= threshold
    if not np.any(below):
        return np.nan, np.nan, False

    # Find the contiguous region containing best_value
    sorted_idx = np.argsort(values)
    sv = values[sorted_idx]
    sb = below[sorted_idx]

    # Locate best_value in sorted array
    pos = np.searchsorted(sv, best_value)
    pos = min(pos, len(sv) - 1)

    # Expand left and right while below threshold
    left = pos
    while left > 0 and sb[left - 1]:
        left -= 1
    right = pos
    while right < len(sv) - 1 and sb[right + 1]:
        right += 1

    ci_lower = float(sv[left])
    ci_upper = float(sv[right])
    identifiable = ci_lower < best_value < ci_upper
    return ci_lower, ci_upper, identifiable


def compute_cp_ci(values, losses, threshold):
    """Find contiguous CI around the grid minimum where loss <= threshold."""
    below = losses <= threshold
    if not np.any(below):
        return np.nan, np.nan, False

    sorted_idx = np.argsort(values)
    sv = values[sorted_idx]
    sb = below[sorted_idx]

    # Centre the CI on the candidate with minimum loss
    min_pos = int(np.argmin(losses[sorted_idx]))

    left = min_pos
    while left > 0 and sb[left - 1]:
        left -= 1
    right = min_pos
    while right < len(sv) - 1 and sb[right + 1]:
        right += 1

    ci_lower = float(sv[left])
    ci_upper = float(sv[right])
    identifiable = ci_lower < ci_upper
    return ci_lower, ci_upper, identifiable


def recompute_parameter_cis():
    curves = pd.read_csv(ROOT / "ple_results_curves.csv")
    summary_old = pd.read_csv(ROOT / "ple_summary.csv")

    best_loss = summary_old["best_loss"].iloc[0]
    sigma2 = best_loss / (N_OBS - N_PARAMS)
    threshold = best_loss + CHISQ_95 * sigma2

    rows = []
    for param in summary_old["parameter"]:
        df = curves[curves["parameter"] == param].sort_values("value")
        s_old = summary_old[summary_old["parameter"] == param].iloc[0]
        best_value = s_old["best_value"]
        ci_lower, ci_upper, identifiable = compute_ci(
            df["value"].to_numpy(), df["loss"].to_numpy(), best_value, threshold
        )
        relative_width = (ci_upper - ci_lower) / max(abs(best_value), 1e-12)
        rows.append({
            "parameter": param,
            "index": int(s_old["index"]),
            "best_value": best_value,
            "best_loss": best_loss,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "identifiable": identifiable,
            "threshold": threshold,
            "relative_width": relative_width,
            "n_failed": int(s_old.get("n_failed", 0)),
            "best_found_loss": float(s_old.get("best_found_loss", best_loss)),
        })

    summary_new = pd.DataFrame(rows)
    summary_new.to_csv(OUT / "ple_summary.csv", index=False)
    return summary_new, curves, threshold, sigma2


def recompute_cp_cis():
    cp_curves = pd.read_csv(ROOT / "cp_profile_loss.csv")
    best_loss = float(cp_curves["loss"].min())
    sigma2 = best_loss / (N_OBS - N_PARAMS)
    threshold = best_loss + CHISQ_95 * sigma2

    rows = []
    for cp_idx in sorted(cp_curves["cp_index"].unique()):
        df = cp_curves[cp_curves["cp_index"] == cp_idx].sort_values("candidate_cp")
        original_cp = df["original_cp"].iloc[0]
        ci_lower, ci_upper, identifiable = compute_cp_ci(
            df["candidate_cp"].to_numpy(), df["loss"].to_numpy(), threshold
        )
        rows.append({
            "cp_index": int(cp_idx),
            "original_cp": int(original_cp),
            "ci_lower": int(np.floor(ci_lower)) if np.isfinite(ci_lower) else np.nan,
            "ci_upper": int(np.ceil(ci_upper)) if np.isfinite(ci_upper) else np.nan,
            "identifiable": identifiable,
        })

    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(OUT / "cp_profile_ci.csv", index=False)
    return ci_df, cp_curves, threshold, sigma2


def plot_parameter_profiles(summary, curves, threshold, sigma2, outfile):
    params = summary["parameter"].tolist()
    n = len(params)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8), constrained_layout=True)
    axes = np.atleast_2d(axes).flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        df = curves[curves["parameter"] == param].sort_values("value")
        s = summary.iloc[i]
        x = df["value"].to_numpy()
        y = df["loss"].to_numpy()
        xi, yi = smooth_profile(x, y)

        ax.plot(xi, yi, "-", lw=1.2, color="steelblue", zorder=1)
        ax.plot(x, y, "o", markersize=3, color="steelblue", alpha=0.5, zorder=2)
        ax.axhline(threshold, color="crimson", ls="--", lw=1.2, zorder=3)
        ax.axvline(s["best_value"], color="green", ls="--", lw=1.2, zorder=3)

        if s["identifiable"]:
            ax.axvspan(s["ci_lower"], s["ci_upper"], alpha=0.18, color="gold", zorder=0)
            ax.axvline(s["ci_lower"], color="orange", ls=":", lw=1.2, zorder=3)
            ax.axvline(s["ci_upper"], color="orange", ls=":", lw=1.2, zorder=3)

        ax.set_title(
            f"{param}\nCI=[{s['ci_lower']:.4g},{s['ci_upper']:.4g}] "
            f"{'Y' if s['identifiable'] else 'N'}",
            fontsize=8,
        )
        ax.set_xlabel("value", fontsize=7)
        ax.set_ylabel("SSE loss", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    n_ident = summary["identifiable"].sum()
    fig.suptitle(
        f"Wind-turbine parameter PLE | SSE | σ²={sigma2:.2f} | threshold={threshold:.2f} | "
        f"identifiable={n_ident}/{n}",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


def plot_cp_profiles(cp_curves, ci_df, threshold, outfile):
    cp_indices = sorted(ci_df["cp_index"].unique())
    fig, axes = plt.subplots(1, len(cp_indices), figsize=(4 * len(cp_indices), 3.5), constrained_layout=True)
    if len(cp_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, cp_indices):
        df = cp_curves[cp_curves["cp_index"] == idx].sort_values("candidate_cp")
        s = ci_df[ci_df["cp_index"] == idx].iloc[0]
        ax.plot(df["candidate_cp"], df["loss"], "-o", markersize=3, color="steelblue", lw=1.2)
        ax.axvline(s["original_cp"], color="green", ls="--", lw=1.2, label=f"best={s['original_cp']}")
        ax.axhline(threshold, color="crimson", ls="--", lw=1.2, label="95% threshold")
        if s["identifiable"]:
            ax.axvspan(s["ci_lower"], s["ci_upper"], alpha=0.18, color="gold", label=f"CI=[{s['ci_lower']},{s['ci_upper']}]")
            ax.axvline(s["ci_lower"], color="orange", ls=":", lw=1.2)
            ax.axvline(s["ci_upper"], color="orange", ls=":", lw=1.2)
        ax.set_title(f"CP {idx}: best={s['original_cp']}, CI=[{s['ci_lower']},{s['ci_upper']}]")
        ax.set_xlabel("candidate changepoint")
        ax.set_ylabel("SSE loss")
        ax.legend(fontsize=7)

    fig.suptitle("Wind-turbine changepoint PLE", fontsize=12)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


def write_report(summary, ci_df, threshold, sigma2, outfile):
    lines = []
    lines.append("# Corrected profile-likelihood analysis for wind-turbine `bic`")
    lines.append("")
    lines.append("**Correction:** the original `tentative_ple_bic` used a raw delta-loss threshold of 3.8415.")
    lines.append("Because the profile is based on the SSE loss, the threshold is rescaled by the estimated residual variance.")
    lines.append("")
    lines.append(f"**n_obs:** {N_OBS}")
    lines.append(f"**n_params:** {N_PARAMS}")
    lines.append(f"**Best SSE:** {summary['best_loss'].iloc[0]:.4f}")
    lines.append(f"**Estimated sigma^2:** {sigma2:.4f}")
    lines.append(f"**Corrected threshold (SSE + {CHISQ_95:.4f} * sigma^2):** {threshold:.4f}")
    lines.append("")
    lines.append(f"**Identifiable parameters:** {summary['identifiable'].sum()} / {len(summary)}")
    lines.append(f"**Identifiable changepoints:** {ci_df['identifiable'].sum()} / {len(ci_df)}")
    lines.append("")
    lines.append("## Parameter intervals")
    lines.append("")
    lines.append("| parameter | best value | CI lower | CI upper | identifiable |")
    lines.append("|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['parameter']} | {r['best_value']:.5g} | {r['ci_lower']:.5g} | {r['ci_upper']:.5g} | {'yes' if r['identifiable'] else 'no'} |")
    lines.append("")
    lines.append("## Changepoint intervals")
    lines.append("")
    lines.append("| cp # | original | CI lower | CI upper | identifiable |")
    lines.append("|---|---|---|---|---|")
    for _, r in ci_df.iterrows():
        lines.append(f"| {int(r['cp_index'])} | {int(r['original_cp'])} | {r['ci_lower']} | {r['ci_upper']} | {'yes' if r['identifiable'] else 'no'} |")
    lines.append("")
    outfile.write_text("\n".join(lines) + "\n")
    print(f"Saved: {outfile}")


def main():
    summary, curves, threshold, sigma2 = recompute_parameter_cis()
    ci_df, cp_curves, cp_threshold, cp_sigma2 = recompute_cp_cis()

    print(f"Best SSE: {summary['best_loss'].iloc[0]:.4f}")
    print(f"Estimated sigma^2: {sigma2:.4f}")
    print(f"Parameter threshold: {threshold:.4f}")
    print(f"CP threshold: {cp_threshold:.4f}")
    print(f"Identifiable parameters: {summary['identifiable'].sum()} / {len(summary)}")
    print(f"Identifiable CPs: {ci_df['identifiable'].sum()} / {len(ci_df)}")

    plot_parameter_profiles(summary, curves, threshold, sigma2, OUT / "ple_profiles_corrected.png")
    plot_cp_profiles(cp_curves, ci_df, cp_threshold, OUT / "cp_profiles_corrected.png")
    write_report(summary, ci_df, threshold, sigma2, OUT / "ple_report_corrected.md")


if __name__ == "__main__":
    main()
