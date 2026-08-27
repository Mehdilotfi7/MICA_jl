#!/usr/bin/env python3
"""
Task G (winner-specific) — Benchmark a MICA winner CP set against standard CPD
methods with hyperparameter grids, then select competitor configurations using
the same SEIRD BIC/MDL/AIC criterion MICA uses.

Usage:
    python benchmark_covid_cpd_winner.py --label <winner_label> \
        --cps-csv <winner_cps.csv> --out-dir <out_dir>

Outputs:
    - cp_sets/*.csv        : one CSV per competitor/hyperparameter CP set
    - candidate_methods.csv: maps each CP-set label to (family, hyperparameter, value)
    - benchmark_summary.csv: symmetric nearest-CP matching vs the winner
    - report.md
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EXAMPLE_DIR = "codes/Mica.jl/examples/Covid-model"
OUT_DIR_ROOT = "outputs/TASK_G"
R_WRAPPER = os.path.join(os.path.dirname(__file__), "cpd_r_wrapper.R")
BASE_DATE = datetime(2020, 1, 27)
CHANNELS = ["infected", "hospitalized", "icu", "death", "vaccination"]
MATCH_WINDOW = 7
N_DAYS = 400
MIN_CP = 11
MAX_CP = N_DAYS - 10
CP_GRID = list(range(20, N_DAYS, 10))   # 20, 30, ..., 390

# Hyperparameter grids for the TCPD competitor set.
# Each candidate CP set is refit with the SEIRD model and scored with BIC/MDL/AIC.
KERNEL_PEN_GRID = [3, 5, 10]
PROPHET_PRIOR_GRID = [0.01, 0.05, 0.1, 0.5]
HMM_K_GRID = [2, 3, 4, 5]

# R-method grids.  For penalty-based methods the value is a scale on a BIC-like
# term; for threshold/sig-level methods it is the actual threshold.
R_PEN_SCALE_GRID = [0.5, 1.0, 2.0, 5.0]
R_THRESH_GRID = [0.3, 0.5, 0.7]
R_SIG_GRID = [0.01, 0.05, 0.1]

# RFPOP grid (penalty scales; L2 loss is used).
RFPOP_PEN_GRID = [0.1, 1.0, 10.0, 100.0]


def load_channel(name):
    mapping = {
        "infected": ("case_rki_daily.csv", "total"),
        "hospitalized": ("Hospitalization_rki_daily.csv", "total"),
        "icu": ("icu_rki_daily.csv", "total"),
        "death": ("death_rki_daily.csv", "Todesfaelle_neu"),
        "vaccination": ("vaccination_rki_daily_allShots.csv", "Total"),
    }
    file, col = mapping[name]
    return pd.read_csv(os.path.join(EXAMPLE_DIR, file))[col].values


def prep_data():
    """Load the same first-400-day window used by the MICA Covid example."""
    infected = load_channel("infected")
    hospitalized = load_channel("hospitalized")
    icu = load_channel("icu")
    death = np.cumsum(load_channel("death"))
    vacc = np.cumsum(load_channel("vaccination"))

    series = [infected, hospitalized, icu, death, vacc]
    max_len = max(len(s) for s in series)
    padded = [np.pad(s, (max_len - len(s), 0), constant_values=0) for s in series]
    # Use the first 400 days (2020-01-27 -> 2021-03-27), exactly like MICA.
    trimmed = [s[:N_DAYS] if len(s) >= N_DAYS else s for s in padded]

    def ma14(x):
        # Centered 14-day average with endpoints using available points only,
        # so cumulative curves do not drop artificially at the right edge.
        return pd.Series(x).rolling(window=14, center=True, min_periods=1).mean().values

    trimmed[0] = ma14(trimmed[0])
    trimmed[3] = ma14(trimmed[3])
    trimmed[4] = ma14(trimmed[4])
    return np.column_stack(trimmed)  # (N_DAYS, 5)


def date_from_idx(idx):
    return (BASE_DATE + timedelta(days=int(idx) - 1)).date().isoformat()


def idx_from_date(ds):
    return (pd.to_datetime(ds).date() - BASE_DATE.date()).days + 1


def feasible_cps(cps):
    return sorted(set(int(c) for c in cps if MIN_CP <= c <= MAX_CP))


def merge_nearby(cps, min_gap=3):
    if not cps:
        return []
    cps = sorted(set(cps))
    clusters = []
    current = [cps[0]]
    for c in cps[1:]:
        if c - current[-1] <= min_gap:
            current.append(c)
        else:
            clusters.append(int(np.median(current)))
            current = [c]
    clusters.append(int(np.median(current)))
    return clusters


# ---------------------------------------------------------------------------
# Python-side standard algorithms (hyperparameter grids)
# ---------------------------------------------------------------------------
def run_kernelcpd(logcases):
    """KernelCPD from ruptures (the TCPD implementation of KCPA)."""
    import ruptures as rpt

    results = {}
    k_algo = rpt.KernelCPD(kernel="linear", min_size=10)
    for pen in KERNEL_PEN_GRID:
        algo = k_algo.fit(logcases.reshape(-1, 1))
        pred = algo.predict(pen=pen)
        cps = sorted(c for c in pred if 1 <= c < N_DAYS)
        results[f"KernelCPD_pen{pen}"] = feasible_cps(cps)
    return results


def run_prophet(cases):
    try:
        from prophet import Prophet
    except Exception as e:
        print(f"  PROPHET import failed: {e}")
        return {}

    dates = [BASE_DATE + timedelta(days=int(i)) for i in range(N_DAYS)]
    y = np.log1p(np.maximum(cases, 0))
    df = pd.DataFrame({"ds": dates, "y": y})
    results = {}
    for prior in PROPHET_PRIOR_GRID:
        try:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                changepoint_prior_scale=prior,
            )
            m.fit(df)
            cps = m.changepoints
            idxs = [idx_from_date(d) for d in cps]
            results[f"PROPHET_prior{prior}"] = feasible_cps(sorted(idxs))
        except Exception as e:
            print(f"  PROPHET prior={prior} failed: {e}")
            results[f"PROPHET_prior{prior}"] = []
    return results


def run_hmm_regime(y):
    """HMM-Regime: fit a Gaussian HMM, declare CPs where the decoded state changes."""
    from hmmlearn.hmm import GaussianHMM

    # Work with log1p-transformed observations, as for the other univariate methods.
    z = np.log1p(np.maximum(y, 0)).reshape(-1, 1)
    results = {}
    for K in HMM_K_GRID:
        label = f"HMMRegime_K{K}"
        try:
            model = GaussianHMM(
                n_components=K,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
                init_params="stmc",
            )
            model.fit(z)
            states = model.predict(z)
            cps = [int(i) for i in range(1, len(states)) if states[i] != states[i - 1]]
            cps = feasible_cps(merge_nearby(cps, min_gap=3))
            results[label] = cps
        except Exception as e:
            print(f"  {label} failed: {e}")
            results[label] = []
    return results


# ---------------------------------------------------------------------------
# R wrappers (hyperparameter grids)
# ---------------------------------------------------------------------------
def run_r_algorithm(algo, logcases, param=None):
    if not os.path.isfile(R_WRAPPER):
        print(f"  {algo}: R wrapper not found at {R_WRAPPER}")
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        in_csv = os.path.join(tmpdir, "data.csv")
        out_csv = os.path.join(tmpdir, "cps.csv")
        pd.DataFrame({"ch0": logcases}).to_csv(in_csv, index=False)
        cmd = [
            "Rscript",
            "--vanilla",
            R_WRAPPER,
            algo,
            in_csv,
            out_csv,
        ]
        if param is not None:
            cmd.append(str(param))
        env = os.environ.copy()
        env.setdefault("R_LIBS_USER", os.environ.get("R_LIBS_USER", ""))
        try:
            subprocess.run(cmd, check=True, env=env, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  {algo} param={param}: R timeout")
            return []
        except subprocess.CalledProcessError as e:
            print(f"  {algo} param={param}: R error: {e.stderr.strip()[:200]}")
            return []

        if not os.path.isfile(out_csv):
            return []
        df = pd.read_csv(out_csv)
        cps = df["cp"].astype(int).tolist()
        return feasible_cps(cps)


def run_r_algorithms(logcases):
    # TCPD competitor set implemented in the R wrapper.
    # (algo, grid of parameter values)
    configs = [
        ("AMOC", [None]),
        ("PELT", R_PEN_SCALE_GRID),
        ("BinSeg", R_PEN_SCALE_GRID),
        ("SegNeigh", R_PEN_SCALE_GRID),
        ("CPNP", R_PEN_SCALE_GRID),
        ("WBS", R_PEN_SCALE_GRID),
        ("ECP", R_SIG_GRID),
        ("FPOP", R_PEN_SCALE_GRID),
        ("RFPOP", RFPOP_PEN_GRID),
        ("BOCPD", R_THRESH_GRID),
        ("BOCPDMS", R_THRESH_GRID),
        ("RBOCPDMS", R_THRESH_GRID),
    ]
    results = {}
    for algo, grid in configs:
        for param in grid:
            label = algo if param is None else f"{algo}_param{param}"
            print(f"  Running {label} (R)...")
            cps = run_r_algorithm(algo, logcases, param)
            results[label] = cps
    return results


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------
def nearest_in_set(cp, candidates, max_window=100000):
    best = None
    bestd = max_window + 1
    for m in candidates:
        d = abs(cp - m)
        if d < bestd:
            bestd = d
            best = m
    return best, bestd


def symmetric_match(mica_cps, algo_cps, window=MATCH_WINDOW):
    mica_matches = 0
    mica_dists = []
    for m in mica_cps:
        _, d = nearest_in_set(m, algo_cps, max_window=window)
        if d <= window:
            mica_matches += 1
        mica_dists.append(d)

    algo_matches = 0
    algo_dists = []
    for c in algo_cps:
        _, d = nearest_in_set(c, mica_cps, max_window=window)
        if d <= window:
            algo_matches += 1
        algo_dists.append(d)

    return {
        "mica_matches_within_7d": mica_matches,
        "mica_total": len(mica_cps),
        "algo_cps_matched_to_mica": algo_matches,
        "avg_distance_mica_to_algo": np.mean(mica_dists) if mica_dists else 0.0,
        "avg_distance_algo_to_mica": np.mean(algo_dists) if algo_dists else 0.0,
    }


def family_of(label):
    """Map a candidate label to its method family."""
    if label == "ZERO":
        return "ZERO"
    if label.startswith("HMMRegime_"):
        return "HMM-Regime"
    # Strip the hyperparameter suffix (last underscore + value).
    # Labels are of the form FAMILY_paramVALUE, e.g. PELT_param2.0, PROPHET_prior0.05, AMOC.
    if "_param" in label:
        return label.rsplit("_param", 1)[0]
    # Python grids use _pen<>, _prior<>, _thresh<>
    for suffix in ["_pen", "_prior", "_thresh"]:
        if suffix in label:
            return label.rsplit(suffix, 1)[0]
    return label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cps-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cp_sets_dir = os.path.join(args.out_dir, "cp_sets")
    os.makedirs(cp_sets_dir, exist_ok=True)

    # Remove stale CP-set files from previous runs so the refit step only sees
    # the current competitor set.
    for stale in glob.glob(os.path.join(cp_sets_dir, "*.csv")):
        os.remove(stale)

    mica_cps = sorted(pd.read_csv(args.cps_csv)["cp"].astype(int).unique().tolist())
    print(f"[{args.label}] Winner CPs: {mica_cps}")

    data = prep_data()
    cases = data[:, 0]
    logcases = np.log1p(np.maximum(cases, 0))

    all_results = {}
    all_results.update(run_r_algorithms(logcases))
    all_results.update(run_kernelcpd(logcases))
    all_results.update(run_prophet(cases))
    all_results.update(run_hmm_regime(cases))
    all_results["ZERO"] = []

    # Write CP sets and candidate metadata.
    # ZERO is handled internally by the Julia simulation script, so skip it here.
    candidate_rows = []
    for method, cps in all_results.items():
        cps = feasible_cps(cps)
        all_results[method] = cps
        if method != "ZERO":
            with open(os.path.join(cp_sets_dir, f"{method}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cp"])
                for c in cps:
                    w.writerow([c])
        candidate_rows.append({
            "label": method,
            "family": family_of(method),
            "hyperparameter": "-",
            "value": "-",
        })

    # Add hyperparameter info where it can be parsed from the label
    for row in candidate_rows:
        label = row["label"]
        if "_param" in label:
            row["hyperparameter"] = "param"
            row["value"] = label.rsplit("_param", 1)[1]
        elif "_pen" in label:
            row["hyperparameter"] = "pen"
            row["value"] = label.rsplit("_pen", 1)[1]
        elif "_prior" in label:
            row["hyperparameter"] = "prior"
            row["value"] = label.rsplit("_prior", 1)[1]
        elif "_thresh" in label:
            row["hyperparameter"] = "thresh"
            row["value"] = label.rsplit("_thresh", 1)[1]

    pd.DataFrame(candidate_rows).to_csv(
        os.path.join(args.out_dir, "candidate_methods.csv"), index=False
    )

    summary_rows = []
    for method, cps in all_results.items():
        match = symmetric_match(mica_cps, cps)
        summary_rows.append({
            "method": method,
            "family": family_of(method),
            "n_cps": len(cps),
            "cps": ";".join(map(str, cps)),
            "dates": ";".join(date_from_idx(c) for c in cps),
            **match,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["family", "method"])
    summary.to_csv(os.path.join(args.out_dir, "benchmark_summary.csv"), index=False)

    lines = []
    lines.append(f"# Task G — Benchmark for winner `{args.label}`")
    lines.append("")
    lines.append(f"Winner CPs: {mica_cps}. ±{MATCH_WINDOW} day matching window.")
    lines.append("")
    lines.append("| method | family | n_cps | MICA matched | algo CPs matched to MICA | avg dist MICA→algo | avg dist algo→MICA |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {r['family']} | {r['n_cps']} | "
            f"{r['mica_matches_within_7d']}/{r['mica_total']} | "
            f"{r['algo_cps_matched_to_mica']}/{r['n_cps']} | "
            f"{r['avg_distance_mica_to_algo']:.1f} | {r['avg_distance_algo_to_mica']:.1f} |"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("- `benchmark_summary.csv`")
    lines.append("- `candidate_methods.csv`")
    lines.append("- `cp_sets/*.csv` (for SEIRD refit)")

    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[{args.label}] Task G benchmark saved to {args.out_dir}")


if __name__ == "__main__":
    main()
