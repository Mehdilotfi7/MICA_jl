#!/usr/bin/env python3
"""
Task H (synthetic benchmark) — Run standard CPD methods on synthetic COVID data
and compare detected change points to the known true CPs.
"""

import csv
import os
import numpy as np
import pandas as pd
import ruptures as rpt
from datetime import datetime, timedelta

SYNTH_DIR = "outputs/TASK_H_synthetic"
OUT_DIR = SYNTH_DIR
os.makedirs(OUT_DIR, exist_ok=True)

BASE_DATE = datetime(2020, 1, 27)
MATCH_WINDOW = 7


def prep_data():
    df = pd.read_csv(os.path.join(SYNTH_DIR, "synthetic_observed.csv"))
    data = df[["infected", "hospitalized", "icu", "death", "vaccination"]].values
    # Apply 14-day moving average to infected, death, vaccination (cumulative channels)
    def ma14(x):
        return np.convolve(x, np.ones(14) / 14, mode="same")
    data[:, 0] = ma14(data[:, 0])
    data[:, 3] = ma14(data[:, 3])
    data[:, 4] = ma14(data[:, 4])
    return data


def load_true_cps():
    df = pd.read_csv(os.path.join(SYNTH_DIR, "true_cps.csv"))
    return sorted(df["cp"].astype(int).unique().tolist())


def date_from_idx(idx):
    return (BASE_DATE + timedelta(days=int(idx) - 1)).date().isoformat()


def run_method(name, data, pen, mode="multivar", width=20):
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
    true_cps = load_true_cps()
    n_true = len(true_cps)

    methods = ["PELT", "Binseg", "Window", "BottomUp"]
    penalties = [1, 5, 10, 20, 50]
    modes = ["multivar", "perchannel"]

    os.makedirs(os.path.join(OUT_DIR, "cp_sets"), exist_ok=True)
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

                # true -> detected
                true_matched = 0
                true_dists = []
                for t in true_cps:
                    b, d = nearest_in_set(t, cps, max_window=MATCH_WINDOW)
                    if d <= MATCH_WINDOW:
                        true_matched += 1
                    true_dists.append(d)

                # detected -> true
                detected_matched = 0
                detected_dists = []
                for c in cps:
                    b, d = nearest_in_set(c, true_cps, max_window=MATCH_WINDOW)
                    if d <= MATCH_WINDOW:
                        detected_matched += 1
                    detected_dists.append(d)

                summary_rows.append({
                    "method": label,
                    "base_method": method,
                    "mode": mode,
                    "penalty": pen,
                    "n_cps": len(cps),
                    "cps": ";".join(map(str, cps)),
                    "dates": ";".join(date_from_idx(c) for c in cps),
                    "true_matches_within_7d": true_matched,
                    "true_total": n_true,
                    "detected_cps_matched_to_true": detected_matched,
                    "avg_distance_true_to_detected": np.mean(true_dists) if true_dists else np.nan,
                    "avg_distance_detected_to_true": np.mean(detected_dists) if detected_dists else np.nan,
                })

                with open(os.path.join(OUT_DIR, "cp_sets", f"{label}.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["cp"])
                    for c in cps:
                        w.writerow([c])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "synthetic_cpd_summary.csv"), index=False)

    # Detailed symmetric comparison for a representative method
    selected = "PELT_multivar_l2_pen5"
    selected_cps = [r for r in summary_rows if r["method"] == selected]
    selected_cps = selected_cps[0]["cps"].split(";") if selected_cps and selected_cps[0]["cps"] else []
    selected_cps = [int(c) for c in selected_cps if c]
    comp_rows = []
    for t in true_cps:
        b, d = nearest_in_set(t, selected_cps, max_window=MATCH_WINDOW)
        comp_rows.append({
            "true_cp": t,
            "true_date": date_from_idx(t),
            "nearest_detected_cp": b if b is not None else "",
            "nearest_detected_date": date_from_idx(b) if b is not None else "",
            "days_apart": d if b is not None else "",
            "within_7": (b is not None and d <= MATCH_WINDOW)
        })
    for c in selected_cps:
        b, d = nearest_in_set(c, true_cps, max_window=MATCH_WINDOW)
        comp_rows.append({
            "true_cp": "",
            "true_date": "",
            "nearest_detected_cp": c,
            "nearest_detected_date": date_from_idx(c),
            "days_apart": d if b is not None else "",
            "within_7": (b is not None and d <= MATCH_WINDOW),
            "direction": "detected_to_true"
        })
    pd.DataFrame(comp_rows).to_csv(os.path.join(OUT_DIR, "synthetic_true_vs_selected.csv"), index=False)

    # Report
    lines = []
    lines.append("# Synthetic benchmark — standard CPD methods vs. known true CPs")
    lines.append("")
    lines.append(f"True CPs: {true_cps}")
    lines.append("")
    lines.append("## Summary (±7 day matching window)")
    lines.append("")
    lines.append("| method | n_cps | true matched | detected matched to true | avg dist true→detected | avg dist detected→true |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['method']} | {r['n_cps']} | {r['true_matches_within_7d']}/{r['true_total']} | {r['detected_cps_matched_to_true']}/{r['n_cps']} | {r['avg_distance_true_to_detected']:.1f} | {r['avg_distance_detected_to_true']:.1f} |")
    lines.append("")
    lines.append(f"## Symmetric comparison for {selected}")
    lines.append("")
    lines.append("See `synthetic_true_vs_selected.csv`.")

    with open(os.path.join(OUT_DIR, "synthetic_cpd_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Synthetic CPD benchmark saved to", OUT_DIR)


if __name__ == "__main__":
    main()
