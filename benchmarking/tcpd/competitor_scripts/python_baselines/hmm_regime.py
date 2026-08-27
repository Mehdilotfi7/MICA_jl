"""HMM-Regime changepoint detector for the toy-datasets benchmark.

Fits a Gaussian hidden Markov model to the univariate series and declares
changepoints at indices where the Viterbi-decoded hidden state changes.
"""
import warnings

import numpy as np


def _filter_cps(cps, n, min_gap=10):
    """Keep CPs in [1, n-1] and enforce a minimum gap between consecutive CPs."""
    cps = sorted(set(int(c) for c in cps if 1 <= c <= n - 1))
    if not cps:
        return []
    filtered = [cps[0]]
    for c in cps[1:]:
        if c - filtered[-1] >= min_gap:
            filtered.append(c)
    return filtered


def run_hmm_regime(y, true_cps, n, mode="oracle", min_gap=10):
    """Run HMM-Regime and return a result dict matching the benchmark format.

    Parameters
    ----------
    y : array-like, shape (n,)
        Input univariate time series.
    true_cps : list[int]
        Ground-truth changepoint indices (0-based, same convention as y).
    n : int
        Length of the series.
    mode : {"default", "oracle"}
        * default: use K=3 states.
        * oracle: select the best K from {2, 3, 4, 5} by F1.
    min_gap : int
        Minimum allowed distance between consecutive detected changepoints.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:
        return {
            "method": "HMM-Regime",
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "covering": 0.0,
            "cps": [],
            "error": f"import failed: {exc}",
        }

    y = np.asarray(y, dtype=float)
    z = (y - np.mean(y)) / (np.std(y) + 1e-10)
    z = z.reshape(-1, 1)

    K_grid = [3] if mode == "default" else [2, 3, 4, 5]

    best_f1 = -1.0  # allow storing the first candidate even when F1 == 0
    best_p = 0.0
    best_r = 0.0
    best_cov = 0.0
    best_cps = []
    best_K = None

    for K in K_grid:
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
        except Exception as exc:
            continue

        cps = [int(i) for i in range(1, len(states)) if states[i] != states[i - 1]]
        cps = _filter_cps(cps, n, min_gap=min_gap)

        p, r, f1 = _calc_metrics(cps, true_cps)
        cov = _calc_covering(cps, true_cps, n)
        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            best_cov = cov
            best_cps = cps
            best_K = K

    result = {
        "method": "HMM-Regime",
        "f1": best_f1,
        "precision": best_p,
        "recall": best_r,
        "covering": best_cov,
        "cps": best_cps,
    }
    if best_K is not None:
        result["K"] = best_K
    return result


def _calc_metrics(detected, true_cps, margin=5):
    """One-to-one matching precision/recall/F1."""
    TP = 0
    matched = [False] * len(true_cps)
    for d in detected:
        for i, cp in enumerate(true_cps):
            if not matched[i] and abs(d - cp) <= margin:
                TP += 1
                matched[i] = True
                break
    FP = len(detected) - TP
    FN = len(true_cps) - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _calc_covering(detected, true_cps, n):
    """TCPD covering metric for a single set of true CPs."""
    if len(true_cps) == 0:
        return 1.0 if len(detected) == 0 else 0.0
    detected = sorted(detected)
    seg_boundaries = [0] + detected + [n]
    true_seg = sorted(set([0] + list(true_cps) + [n]))
    seg_score = 0.0
    for i in range(len(seg_boundaries) - 1):
        a, b = seg_boundaries[i], seg_boundaries[i + 1]
        best_overlap = 0.0
        for j in range(len(true_seg) - 1):
            ta, tb = true_seg[j], true_seg[j + 1]
            best_overlap = max(best_overlap, max(0.0, min(b, tb) - max(a, ta)))
        seg_score += best_overlap
    return seg_score / n
