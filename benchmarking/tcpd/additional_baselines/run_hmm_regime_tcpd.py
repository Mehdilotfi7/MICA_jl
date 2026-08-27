#!/usr/bin/env python3
"""Run HMM-Regime on all TCPD datasets using hmmlearn (local venv_cpd).

This replaces the from-scratch Julia additional-baseline implementation with the
official Python package, matching the HMM-Regime protocol already used for the
toy-dataset benchmark.

Outputs:
    benchmark_tcpd_hmm_regime_oracle.json   best K in {2,3,4,5} by F1
    benchmark_tcpd_hmm_regime_default.json  fixed K=3
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[4]  # repository root
DATASET_DIR = BASE / "benchmarking" / "tcpd" / "dataset" / "datasets"
ANNOTATIONS_FILE = BASE / "benchmarking" / "tcpd" / "dataset" / "annotations.json"
OUT_DIR = Path(__file__).parent

METRIC_MARGIN = 5
MIN_GAP = 10
K_DEFAULT = 3
K_GRID = [2, 3, 4, 5]


def load_series(series_name):
    """Load a TCPD series and its annotation CP sets."""
    with open(DATASET_DIR / series_name / f"{series_name}.json") as f:
        data = json.load(f)
    y_raw = data["series"][0]["raw"]
    y = np.array([x for x in y_raw if x is not None], dtype=float)
    # Forward-fill non-finite values (matches Julia preprocessing)
    for i in range(1, len(y)):
        if not np.isfinite(y[i]):
            y[i] = y[i - 1]

    cp_sets = []
    annotations = json.loads(ANNOTATIONS_FILE.read_text())
    for cp_list in annotations.get(series_name, {}).values():
        if cp_list is not None and len(cp_list) > 0:
            cp_clean = [int(x) for x in cp_list if x is not None]
            if cp_clean:
                cp_sets.append(cp_clean)
    return y, cp_sets


def compute_f1(detected, cp_sets, margin=METRIC_MARGIN):
    """TCPD-style F1 across multiple annotator CP sets."""
    detected = sorted(set(int(c) for c in detected))
    if not cp_sets:
        if not detected:
            return 1.0, 1.0, 1.0
        return 0.0, 1.0, 0.0
    if not detected:
        return 0.0, 0.0, 0.0

    all_cp = sorted(set(int(c) for cp in cp_sets for c in cp))
    tp_p = 0
    used = set()
    for d in detected:
        for cp in all_cp:
            if abs(d - cp) <= margin and cp not in used:
                tp_p += 1
                used.add(cp)
                break
    precision = tp_p / len(detected)

    recalls = []
    for cps in cp_sets:
        tp_r = 0
        used_d = set()
        for cp in cps:
            for d in detected:
                if abs(d - cp) <= margin and d not in used_d:
                    tp_r += 1
                    used_d.add(d)
                    break
        recalls.append(tp_r / max(len(cps), 1))
    recall = float(np.mean(recalls))

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compute_covering(detected, cp_sets, n):
    """TCPD covering metric averaged over annotator CP sets."""
    if not cp_sets:
        return 1.0 if not detected else 0.0
    detected = sorted(set(int(c) for c in detected))
    seg_boundaries = [0] + detected + [n]
    scores = []
    for cps in cp_sets:
        true_seg = sorted(set([0] + [int(c) for c in cps] + [n]))
        seg_score = 0.0
        for i in range(len(seg_boundaries) - 1):
            a, b = seg_boundaries[i], seg_boundaries[i + 1]
            best_overlap = 0.0
            for j in range(len(true_seg) - 1):
                ta, tb = true_seg[j], true_seg[j + 1]
                overlap = max(0.0, min(b, tb) - max(a, ta))
                best_overlap = max(best_overlap, overlap)
            seg_score += best_overlap
        scores.append(seg_score / n)
    return float(np.mean(scores))


def filter_cps(cps, n, min_gap=MIN_GAP):
    """Keep CPs in [1, n-1] and enforce a minimum gap."""
    cps = sorted(set(int(c) for c in cps if 1 <= c <= n - 1))
    if not cps:
        return []
    filtered = [cps[0]]
    for c in cps[1:]:
        if c - filtered[-1] >= min_gap:
            filtered.append(c)
    return filtered


def run_hmm(y, n, true_cp_sets, mode):
    """Run GaussianHMM and return a benchmark record dict."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:
        return {
            "dataset": None,
            "model": "HMM-Regime",
            "config": f"(K = {K_DEFAULT}, min_seg_len = {MIN_GAP})",
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "covering": 0.0,
            "n_cps": 0,
            "cps": [],
            "error": f"import failed: {exc}",
        }

    z = (y - np.mean(y)) / (np.std(y) + 1e-10)
    z = z.reshape(-1, 1)

    k_grid = [K_DEFAULT] if mode == "default" else K_GRID

    best_f1 = -1.0
    best = None
    for K in k_grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=K,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                    init_params="stmc",
                )
                model.fit(z)
                states = model.predict(z)
        except Exception:
            continue

        cps = [int(i) for i in range(1, len(states)) if states[i] != states[i - 1]]
        cps = filter_cps(cps, n, MIN_GAP)
        prec, rec, f1 = compute_f1(cps, true_cp_sets)
        cov = compute_covering(cps, true_cp_sets, n)
        if f1 > best_f1:
            best_f1 = f1
            best = {
                "config": f"(K = {K}, min_seg_len = {MIN_GAP})",
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "covering": cov,
                "n_cps": len(cps),
                "cps": cps,
                "K": K,
            }

    if best is None:
        return {
            "model": "HMM-Regime",
            "config": "all_failed",
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "covering": 0.0,
            "n_cps": 0,
            "cps": [],
        }
    best["model"] = "HMM-Regime"
    return best


def main():
    annotations = json.loads(ANNOTATIONS_FILE.read_text())
    dataset_names = sorted(
        d
        for d in DATASET_DIR.iterdir()
        if d.is_dir()
        and (d / f"{d.name}.json").is_file()
        and d.name in annotations
    )
    dataset_names = [d.name for d in dataset_names]

    print(f"Datasets: {len(dataset_names)}")
    oracle_records = []
    default_records = []

    for idx, ds_name in enumerate(dataset_names, start=1):
        y, cp_sets = load_series(ds_name)
        n = len(y)
        print(f"[{idx}/{len(dataset_names)}] {ds_name} (n={n}, annotators={len(cp_sets)})")

        for mode, records in [("oracle", oracle_records), ("default", default_records)]:
            rec = run_hmm(y, n, cp_sets, mode)
            rec["dataset"] = ds_name
            records.append(rec)

        # Save incrementally
        with open(OUT_DIR / "benchmark_tcpd_hmm_regime_oracle.json", "w") as f:
            json.dump(oracle_records, f, indent=2)
        with open(OUT_DIR / "benchmark_tcpd_hmm_regime_default.json", "w") as f:
            json.dump(default_records, f, indent=2)

    print("\nBENCHMARK COMPLETE")
    print(f"Oracle entries: {len(oracle_records)}")
    print(f"Default entries: {len(default_records)}")


if __name__ == "__main__":
    main()
