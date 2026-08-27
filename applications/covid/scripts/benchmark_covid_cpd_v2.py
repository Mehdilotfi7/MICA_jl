#!/usr/bin/env python3
"""
Task G (expanded) — Benchmark MICA against several standard CPD methods.

Methods: PELT, Binseg, Window, BottomUp (multivariate and per-channel).
For each detected change-point set we:
  1. Save the CP list (to be refit by the SEIRD model in Julia).
  2. Report symmetric nearest-CP distances to MICA's zero-penalty result.
  3. Summarise match counts using a ±7 day tolerance.
"""

import csv
import os
import numpy as np
import pandas as pd
import ruptures as rpt
from datetime import datetime, timedelta

EXAMPLE_DIR = "codes/Mica.jl/examples/Covid-model"
TASK_A_DIR = "outputs/TASK_A/results_penalty_zero"
OUT_DIR = "outputs/TASK_G"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "cp_sets"), exist_ok=True)

BASE_DATE = datetime(2020, 1, 27)
CHANNELS = ["infected", "hospitalized", "icu", "death", "vaccination"]
MATCH_WINDOW = 7


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
    infected = load_channel("infected")
    hospitalized = load_channel("hospitalized")
    icu = load_channel("icu")
    death = np.cumsum(load_channel("death"))
    vacc = np.cumsum(load_channel("vaccination"))

    series = [infected, hospitalized, icu, death, vacc]
    max_len = max(len(s) for s in series)
    padded = [np.pad(s, (max_len - len(s), 0), constant_values=0) for s in series]
    trimmed = [s[-400:] if len(s) >= 400 else s for s in padded]

    def ma14(x):
        return np.convolve(x, np.ones(14) / 14, mode="same")
    trimmed[0] = ma14(trimmed[0])
    trimmed[3] = ma14(trimmed[3])
    trimmed[4] = ma14(trimmed[4])
    return np.column_stack(trimmed)  # (400, 5)


def date_from_idx(idx):
    return (BASE_DATE + timedelta(days=int(idx) - 1)).date().isoformat()


def run_method(name, data, pen, mode="multivar", width=20):
    """Run a ruptures method and return internal CP indices (0-based end points removed)."""
    logdata = np.log1p(np.maximum(data, 0))
    n = len(data)
    if name == "PELT":
        if mode == "multivar":
            algo = rpt.Pelt(model="l2", min_size=10, jump=10).fit(logdata)
            cps = algo.predict(pen=pen)
        else:
            cps = set()
            for k in range(logdata.shape[1]):
                algo = rpt.Pelt(model="l2", min_size=10, jump=10).fit(logdata[:, k].reshape(-1, 1))
                cps.update(algo.predict(pen=pen))
            cps = sorted(cps)
    elif name == "Binseg":
        if mode == "multivar":
            algo = rpt.Binseg(model="l2", min_size=10, jump=10).fit(logdata)
            cps = algo.predict(pen=pen)
        else:
            cps = set()
            for k in range(logdata.shape[1]):
                algo = rpt.Binseg(model="l2", min_size=10, jump=10).fit(logdata[:, k].reshape(-1, 1))
                cps.update(algo.predict(pen=pen))
            cps = sorted(cps)
    elif name == "Window":
        if mode == "multivar":
            algo = rpt.Window(width=width, model="l2", jump=10).fit(logdata)
            cps = algo.predict(pen=pen)
        else:
            cps = set()
            for k in range(logdata.shape[1]):
                algo = rpt.Window(width=width, model="l2", jump=10).fit(logdata[:, k].reshape(-1, 1))
                cps.update(algo.predict(pen=pen))
            cps = sorted(cps)
    elif name == "BottomUp":
        if mode == "multivar":
            algo = rpt.BottomUp(model="l2", min_size=10, jump=10).fit(logdata)
            cps = algo.predict(pen=pen)
        else:
            cps = set()
            for k in range(logdata.shape[1]):
                algo = rpt.BottomUp(model="l2", min_size=10, jump=10).fit(logdata[:, k].reshape(-1, 1))
                cps.update(algo.predict(pen=pen))
            cps = sorted(cps)
    else:
        raise ValueError(name)

    # ruptures returns terminal index = n; remove it
    if cps and cps[-1] == n:
        cps = cps[:-1]
    return sorted(c for c in cps if 1 <= c < n)


def nearest_in_set(cp, candidates, max_window=100000):
    best = None
    bestd = max_window + 1
    for m in candidates:
        d = abs(cp - m)
        if d < bestd:
            bestd = d
            best = m
    return best, bestd


def main():
    data = prep_data()
    mica_cps = load_mica_cps()
    n_mica = len(mica_cps)

    methods = ["PELT", "Binseg", "Window", "BottomUp"]
    penalties = [1, 5, 10, 20, 50]
    modes = ["multivar", "perchannel"]

    results = {}
    summary_rows = []

    for method in methods:
        for mode in modes:
            for pen in penalties:
                label = f"{method}_{mode}_l2_pen{pen}"
                try:
                    cps = run_method(method, data, pen, mode=mode)
                except Exception as e:
                    print(f"  {label} failed: {e}")
                    cps = []
                results[label] = cps

                # symmetric nearest-CP analysis
                # (a) for each MICA cp, nearest algorithm cp
                mica_matches = 0
                mica_distances = []
                for m in mica_cps:
                    b, d = nearest_in_set(m, cps, max_window=MATCH_WINDOW)
                    if d <= MATCH_WINDOW:
                        mica_matches += 1
                    mica_distances.append(d)

                # (b) for each algorithm cp, nearest MICA cp
                algo_matches = 0
                algo_distances = []
                for c in cps:
                    b, d = nearest_in_set(c, mica_cps, max_window=MATCH_WINDOW)
                    if d <= MATCH_WINDOW:
                        algo_matches += 1
                    algo_distances.append(d)

                summary_rows.append({
                    "method": label,
                    "base_method": method,
                    "mode": mode,
                    "penalty": pen,
                    "n_cps": len(cps),
                    "cps": ";".join(map(str, cps)),
                    "dates": ";".join(date_from_idx(c) for c in cps),
                    "mica_matches_within_7d": mica_matches,
                    "mica_total": n_mica,
                    "algo_cps_matched_to_mica": algo_matches,
                    "avg_distance_mica_to_algo": np.mean(mica_distances) if mica_distances else np.nan,
                    "avg_distance_algo_to_mica": np.mean(algo_distances) if algo_distances else np.nan,
                })

                # save CP set for Julia refit
                with open(os.path.join(OUT_DIR, "cp_sets", f"{label}.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["cp"])
                    for c in cps:
                        w.writerow([c])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "benchmark_summary_v2.csv"), index=False)

    # Detailed symmetric comparison for one selected method (PELT multivar pen=5,
    # which matches the mica_vs_pelt_comparison.csv used in previous plots)
    selected = "PELT_multivar_l2_pen5"
    selected_cps = results.get(selected, [])
    comp_rows = []
    for m in mica_cps:
        b, d = nearest_in_set(m, selected_cps, max_window=MATCH_WINDOW)
        comp_rows.append({
            "mica_cp": m,
            "mica_date": date_from_idx(m),
            "nearest_algo_cp": b if b is not None else "",
            "nearest_algo_date": date_from_idx(b) if b is not None else "",
            "days_apart": d if b is not None else "",
            "within_7": (b is not None and d <= MATCH_WINDOW)
        })
    for c in selected_cps:
        b, d = nearest_in_set(c, mica_cps, max_window=MATCH_WINDOW)
        comp_rows.append({
            "mica_cp": "",
            "mica_date": "",
            "nearest_algo_cp": c,
            "nearest_algo_date": date_from_idx(c),
            "days_apart": d if b is not None else "",
            "within_7": (b is not None and d <= MATCH_WINDOW),
            "direction": "algo_to_mica"
        })
    pd.DataFrame(comp_rows).to_csv(os.path.join(OUT_DIR, "mica_vs_algo_comparison.csv"), index=False)

    # Report
    lines = []
    lines.append("# Task G — Benchmark of MICA against standard CPD methods (expanded)")
    lines.append("")
    lines.append(f"Methods: {', '.join(methods)}. MICA zero-penalty CPs: {mica_cps}.")
    lines.append("")
    lines.append("## Summary (±7 day matching window)")
    lines.append("")
    lines.append("| method | n_cps | MICA matched | algo CPs matched to MICA | avg dist MICA→algo | avg dist algo→MICA |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['method']} | {r['n_cps']} | {r['mica_matches_within_7d']}/{r['mica_total']} | {r['algo_cps_matched_to_mica']}/{r['n_cps']} | {r['avg_distance_mica_to_algo']:.1f} | {r['avg_distance_algo_to_mica']:.1f} |")
    lines.append("")
    lines.append(f"## Symmetric comparison for {selected}")
    lines.append("")
    lines.append("See `mica_vs_algo_comparison.csv`.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `benchmark_summary_v2.csv`")
    lines.append("- `cp_sets/*.csv` (one per method, for SEIRD refit)")
    lines.append("- `mica_vs_algo_comparison.csv`")

    with open(os.path.join(OUT_DIR, "report_v2.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Expanded benchmark saved to", OUT_DIR)


def load_mica_cps():
    path = os.path.join(TASK_A_DIR, "covid_detected_cps_origset_penalty_zero.csv")
    df = pd.read_csv(path)
    return sorted(df["cp"].astype(int).unique().tolist())


if __name__ == "__main__":
    main()
