#!/usr/bin/env python3
"""TCPD-paper package-based toy-dataset changepoint-detection benchmark.

Runs the exact competitor set and hyperparameter grids described in the TCPD
paper (van den Burg & Williams, 2020) on the 9 synthetic toy datasets.

R methods are executed through MICA/benchmarking/R/run_baseline.R, which now
accepts a JSON configuration.  Python methods are run natively, with BOCPDMS
and RBOCPDMS dispatched to the exact GitHub repositories cloned under
MICA/benchmarking/external/.

Output:
    MICA/benchmarking/results/benchmark_toydatasets_package_based.json

Usage:
    python run_benchmark.py                  # full benchmark
    python run_benchmark.py --smoke-test     # one ODE instance, selected methods
    python run_benchmark.py --methods PELT,PROPHET --model ODE --limit 2
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

MARGIN = 5
R_CP_INDEX_OFFSET = 1

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "toy_datasets.json"
OUT_PATH = ROOT / "results" / "benchmark_toydatasets_package_based.json"
R_RUNNER = ROOT / "R" / "run_baseline.R"
# Set this to your local R library path, or leave as an environment override.
R_LIBS_USER = os.environ.get("R_LIBS_USER", "")

BOCPDMS_ROOT = ROOT / "external" / "bocpdms"
RBOCPDMS_ROOT = ROOT / "external" / "rbocpdms"

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------
R_TIMEOUT_DEFAULT = 60
R_TIMEOUT_RFPOP = 30
BOCPDMS_ORACLE_TIMEOUT = 60
BOCPDMS_DEFAULT_TIMEOUT = 180
PROPHET_TIMEOUT = 300
HMM_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calc_metrics(detected, true_cps, margin=MARGIN):
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


def calc_covering(detected, true_cps, n):
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


def score_cps(cps, true_cps, n):
    """Return (precision, recall, f1, covering) for a candidate CP set."""
    p, r, f1 = calc_metrics(cps, true_cps)
    cov = calc_covering(cps, true_cps, n)
    return p, r, f1, cov


# ---------------------------------------------------------------------------
# R runner
# ---------------------------------------------------------------------------


def run_r_algorithm(algo, y, config, n, timeout=R_TIMEOUT_DEFAULT):
    """Call the R baseline runner for a single config.

    Returns a list of 0-based CP indices, or None on failure.
    """
    if not R_RUNNER.is_file():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        in_csv = Path(tmpdir) / "input.csv"
        out_csv = Path(tmpdir) / "output.csv"
        pd.DataFrame({"y": y}).to_csv(in_csv, index=False)

        cmd = [
            "Rscript", "--vanilla", str(R_RUNNER),
            algo, str(in_csv), str(out_csv),
            json.dumps(config, separators=(",", ":")),
        ]

        env = os.environ.copy()
        env.setdefault("R_LIBS_USER", R_LIBS_USER)

        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            print(f"    {algo} config={config} R call failed: {stderr[:200]}", flush=True)
            return None

        if not out_csv.is_file():
            return None
        try:
            df = pd.read_csv(out_csv)
        except Exception:
            return None

        col = "cp" if "cp" in df.columns else df.columns[0]
        cps = df[col].dropna().astype(int).tolist()
        cps = [c - R_CP_INDEX_OFFSET for c in cps]
        cps = sorted(c for c in set(cps) if 1 <= c <= n - 1)
        return cps


def run_r_batch(algo, y, configs, n, timeout):
    """Call the R baseline runner once for a whole grid of configs (used for RFPOP).

    Returns a list of (config, cps) tuples, or None on failure.
    """
    if not R_RUNNER.is_file():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        in_csv = Path(tmpdir) / "input.csv"
        config_json = Path(tmpdir) / "config.json"
        out_json = Path(tmpdir) / "output.json"
        pd.DataFrame({"y": y}).to_csv(in_csv, index=False)
        with open(config_json, "w") as f:
            json.dump(configs, f, separators=(",", ":"))

        cmd = [
            "Rscript", "--vanilla", str(R_RUNNER),
            algo, str(in_csv), str(out_json),
            str(config_json),
        ]

        env = os.environ.copy()
        env.setdefault("R_LIBS_USER", R_LIBS_USER)

        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            print(f"    {algo} batch with {len(configs)} configs failed: {stderr[:200]}", flush=True)
            return None

        if not out_json.is_file():
            return None
        try:
            with open(out_json) as f:
                data = json.load(f)
        except Exception:
            return None

        results = []
        for entry in data.get("results", []):
            cfg = entry.get("config", {})
            cps = [int(c - R_CP_INDEX_OFFSET) for c in entry.get("cps", [])]
            cps = sorted(c for c in set(cps) if 1 <= c <= n - 1)
            results.append((cfg, cps))
        return results


def make_r_runner(algo, default_configs, oracle_configs, timeout=R_TIMEOUT_DEFAULT, batch_timeout=None):
    """Return a runner function that grid-searches the R algorithm by F1."""
    def runner(y, true_cps, n, mode="oracle"):
        configs = default_configs if mode == "default" else oracle_configs
        best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "covering": 0.0, "cps": []}

        # RFPOP has 2,424 configs; running them in one R call avoids subprocess overhead.
        if algo == "RFPOP" and len(configs) > 1:
            batch = run_r_batch(algo, y, configs, n, timeout=batch_timeout or 600)
            if batch is None:
                return {"method": algo, **best}
            for cfg, cps in batch:
                p, r, f1, cov = score_cps(cps, true_cps, n)
                if f1 > best["f1"]:
                    best.update(
                        {"f1": f1, "precision": p, "recall": r, "covering": cov, "cps": cps}
                    )
            return {"method": algo, **best}

        # All other R methods run per-config with individual timeouts.
        for cfg in configs:
            cps = run_r_algorithm(algo, y, cfg, n, timeout=timeout)
            if cps is None:
                continue
            p, r, f1, cov = score_cps(cps, true_cps, n)
            if f1 > best["f1"]:
                best.update(
                    {"f1": f1, "precision": p, "recall": r, "covering": cov, "cps": cps}
                )
        return {"method": algo, **best}

    return runner


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------


def changepoint_grid():
    """Oracle grid for AMOC/PELT/BinSeg/SegNeigh (R changepoint package).

    Matches the TCPD paper (van den Burg & Williams, 2020):
      - penalties: None, SIC, BIC, MBIC, AIC, Hannan-Quinn, Asymptotic
      - manual penalty: 101 log-spaced values from 1e-3 to 1e3
      - test statistics: Normal, CUSUM, CSS, Gamma, Exponential, Poisson
      - invalid combinations are skipped at runtime by the R runner.
    """
    penalties = ["None", "SIC", "BIC", "MBIC", "AIC", "Hannan-Quinn", "Asymptotic"]
    manual_scales = np.logspace(-3, 3, 101).tolist()
    cpt_types = {
        "mean": ["Normal", "CUSUM", "Gamma", "Exponential", "Poisson"],
        "var": ["Normal", "CSS"],
        "meanvar": ["Normal", "CSS", "Gamma", "Exponential", "Poisson"],
    }
    configs = []
    for cpt_type, test_stats in cpt_types.items():
        for test_stat in test_stats:
            # Named penalties
            for pen in penalties:
                configs.append(
                    {"cpt_type": cpt_type, "test_stat": test_stat, "penalty": pen}
                )
            # Manual penalty grid
            for scale in manual_scales:
                configs.append(
                    {
                        "cpt_type": cpt_type,
                        "test_stat": test_stat,
                        "penalty": "Manual",
                        "pen_scale": scale,
                    }
                )
    return configs


def changepoint_default_config():
    return [{"cpt_type": "mean", "penalty": "MBIC", "test_stat": "Normal"}]


def segneigh_default_config():
    # SegNeigh does not support MBIC in changepoint >= 2.3.
    return [{"cpt_type": "mean", "penalty": "SIC", "test_stat": "Normal"}]


def binseg_segneigh_grid(n):
    """Oracle grid for BinSeg/SegNeigh including Q choices.

    Matches the TCPD paper: Q = 5 (default) or Q = T/2 + 1 (max).
    """
    base = changepoint_grid()
    Qs = [5, int(n / 2) + 1]
    configs = []
    for cfg in base:
        for Q in Qs:
            new_cfg = cfg.copy()
            new_cfg["Q"] = Q
            configs.append(new_cfg)
    return configs


def cpnp_grid():
    """Oracle grid for CPNP (changepoint.np).

    Matches the TCPD paper:
      - same penalty grid as PELT
      - number of quantiles varied on {10, 20, 30, 40}
    """
    penalties = ["None", "SIC", "BIC", "MBIC", "AIC", "Hannan-Quinn", "Asymptotic"]
    manual_scales = np.logspace(-3, 3, 101).tolist()
    nquantiles_grid = [10, 20, 30, 40]
    configs = []
    for nq in nquantiles_grid:
        for pen in penalties:
            configs.append({"penalty": pen, "nquantiles": nq})
        for scale in manual_scales:
            configs.append({"penalty": "Manual", "pen_scale": scale, "nquantiles": nq})
    return configs


def cpnp_default_config():
    return [{"penalty": "MBIC", "nquantiles": 10}]


def rfpop_grid():
    """Oracle grid for RFPOP (robseg).  Total 2,424 configs."""
    lambdas = np.logspace(-3, 3, 101).tolist()
    lthresholds = np.logspace(-1, 1, 11).tolist()
    configs = []
    for loss in ["L1", "L2"]:
        for lam in lambdas:
            configs.append({"loss": loss, "lambda": lam, "lthreshold": 1.0})
    for loss in ["Huber", "Outlier"]:
        for lam in lambdas:
            for lth in lthresholds:
                configs.append({"loss": loss, "lambda": lam, "lthreshold": lth})
    return configs


def rfpop_default_config():
    return [{"loss": "Outlier", "lambda": "log_n", "lthreshold": 3.0}]


def ecp_grid():
    """Oracle grid for ECP (ecp)."""
    configs = []
    for alg in ["e.divisive", "e.agglo"]:
        for sig in [0.05, 0.01]:
            for min_size in [2, 30]:
                for alpha in [0.5, 1.0, 1.5]:
                    configs.append(
                        {"algorithm": alg, "sig.lvl": sig, "min.size": min_size, "alpha": alpha}
                    )
    return configs


def ecp_default_config():
    return [{"algorithm": "e.divisive", "sig.lvl": 0.05, "min.size": 30, "alpha": 1.0}]


def kcpa_grid():
    """Oracle grid for KCPA (ecp).

    Matches the TCPD paper:
      - cost varied on 101 log-spaced values from 1e-3 to 1e3
      - max CPs = maximum possible OR 5
    """
    costs = np.logspace(-3, 3, 101).tolist()
    configs = []
    for cost in costs:
        configs.append({"cost": cost, "max_cp": "max"})
        configs.append({"cost": cost, "max_cp": 5})
    return configs


def kcpa_default_config():
    return [{"cost": 1.0, "max_cp": "max"}]


def wbs_grid(n):
    """Oracle grid for WBS (wbs)."""
    configs = []
    for penalty in ["SSIC", "BIC", "MBIC"]:
        for integrated in [True, False]:
            for max_cp in [50, n]:
                configs.append(
                    {"penalty": penalty, "integrated": integrated, "max.cp": max_cp}
                )
    return configs


def wbs_default_config():
    return [{"penalty": "SSIC", "integrated": True, "max.cp": 50}]


def bocpd_grid():
    """Oracle grid for BOCPD (R ocp package).

    Matches the TCPD paper:
      - intensity: 10, 50, 100, 200
      - alpha0, beta0, kappa0: 0.01, 0.1, 1, 10, 100
    """
    intensities = [10, 50, 100, 200]
    alphas = [0.01, 0.1, 1, 10, 100]
    betas = [0.01, 0.1, 1, 10, 100]
    kappas = [0.01, 0.1, 1, 10, 100]
    configs = []
    for intensity in intensities:
        for alpha0 in alphas:
            for beta0 in betas:
                for kappa0 in kappas:
                    configs.append(
                        {
                            "intensity": intensity,
                            "alpha0": alpha0,
                            "beta0": beta0,
                            "kappa0": kappa0,
                        }
                    )
    return configs


def bocpd_default_config():
    return [{"intensity": 100, "alpha0": 1, "beta0": 1, "kappa0": 1}]


def bocpdms_oracle_grid():
    """Oracle grid for BOCPDMS/RBOCPDMS Python implementations.

    Matches the TCPD-paper settings for BOCPD:
      - intensity: 10, 50, 100, 200
      - alpha0, beta0, kappa0: 0.01, 0.1, 1, 10, 100
    For RBOCPDMS, alpha_param and alpha_rld are fixed at 0.5 as in the paper.
    """
    intensities = [10, 50, 100, 200]
    alphas = [0.01, 0.1, 1, 10, 100]
    betas = [0.01, 0.1, 1, 10, 100]
    kappas = [0.01, 0.1, 1, 10, 100]
    configs = []
    for intensity in intensities:
        for alpha0 in alphas:
            for beta0 in betas:
                for kappa0 in kappas:
                    configs.append(
                        {
                            "intensity": intensity,
                            "alpha0": alpha0,
                            "beta0": beta0,
                            "kappa0": kappa0,
                            "alpha_param": 0.5,
                            "alpha_rld": 0.5,
                        }
                    )
    return configs


def bocpdms_default_config():
    return [{"intensity": 100, "alpha0": 1, "beta0": 1, "kappa0": 1}]


# ---------------------------------------------------------------------------
# PROPHET
# ---------------------------------------------------------------------------


def run_prophet(y, true_cps, n, mode="oracle"):
    """Prophet trend changepoints."""
    try:
        from prophet import Prophet
        import logging

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
    except Exception as exc:
        return {
            "method": "PROPHET",
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "covering": 0.0,
            "cps": [],
            "error": str(exc),
        }

    if mode == "default":
        scales = [0.05]
        max_cps = [25]
    else:
        scales = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        max_cps = [25, n]

    df = pd.DataFrame(
        {
            "ds": pd.date_range(start="2020-01-01", periods=len(y), freq="D"),
            "y": y,
        }
    )

    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "covering": 0.0, "cps": []}
    for cp_scale in scales:
        for n_cp in max_cps:
            try:
                m = Prophet(
                    changepoint_prior_scale=cp_scale,
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_range=0.95,
                    n_changepoints=min(n_cp, max(1, int(0.95 * len(y)) - 1)),
                )
                m.fit(df)
                cp_dates = m.changepoints
                delta = m.params.get("delta")
                threshold = 0.01
                if delta is not None and len(cp_dates) == delta.shape[1]:
                    delta_median = np.median(delta, axis=0)
                    selected = np.abs(delta_median) > threshold
                    cp_dates = cp_dates[selected]
                start_date = df["ds"].iloc[0]
                cps = []
                for d in cp_dates:
                    idx = (pd.to_datetime(d) - pd.to_datetime(start_date)).days
                    if 1 <= idx <= n - 1:
                        cps.append(int(idx))
                cps = sorted(set(cps))
                p, r, f1, cov = score_cps(cps, true_cps, n)
                if f1 > best["f1"]:
                    best.update(
                        {"f1": f1, "precision": p, "recall": r, "covering": cov, "cps": cps}
                    )
            except Exception:
                continue
    return {"method": "PROPHET", **best}


# ---------------------------------------------------------------------------
# BOCPDMS / RBOCPDMS wrappers (exact GitHub repositories)
# ---------------------------------------------------------------------------


BOCPDMS_SUBPROCESS_TEMPLATE = r'''
import json
import numpy as np
import scipy
import scipy.misc
import scipy.special
import sys
import time

# Compatibility shims for Python 3.8+ / NumPy 2.0+.
# time.clock was removed; scipy.misc.logsumexp moved to scipy.special;
# np.in1d was renamed np.isin in NumPy 2.0;
# np.reshape no longer accepts the 'newshape' keyword argument.
if not hasattr(time, "clock"):
    time.clock = time.perf_counter
if not hasattr(scipy.misc, "logsumexp"):
    scipy.misc.logsumexp = scipy.special.logsumexp
if not hasattr(np, "in1d") and hasattr(np, "isin"):
    np.in1d = np.isin
_orig_reshape = np.reshape

def _reshape_compat(a, shape=None, *args, **kwargs):
    if "newshape" in kwargs:
        shape = kwargs.pop("newshape")
    return _orig_reshape(a, shape, *args, **kwargs)

np.reshape = _reshape_compat

sys.path.insert(0, sys.argv[1])

data_path = sys.argv[2]
config_path = sys.argv[3]
out_path = sys.argv[4]
algorithm = sys.argv[5]

with open(config_path) as f:
    config = json.load(f)

y = np.load(data_path)
n = len(y)

# Standardize like the TCPD wrappers
if np.std(y) > 1e-12:
    ys = (y - np.mean(y)) / np.std(y)
else:
    ys = y - np.mean(y)

try:
    from cp_probability_model import CpModel
    from detector import Detector
    if algorithm == "BOCPDMS":
        from BVAR_NIG import BVARNIG
        ModelClass = BVARNIG
        generalized_bayes_rld = "kullback_leibler"
        alpha_param = None
        alpha_rld = None
    else:  # RBOCPDMS
        from BVAR_NIG_DPD import BVARNIGDPD
        ModelClass = BVARNIGDPD
        generalized_bayes_rld = "power_divergence"
        alpha_param = config.get("alpha_param", 0.5)
        alpha_rld = config.get("alpha_rld", 0.5)

    intensity = float(config.get("intensity", 100))
    alpha0 = float(config.get("alpha0", 1))
    beta0 = float(config.get("beta0", 1))
    kappa0 = float(config.get("kappa0", 1))

    S1, S2 = 1, 1
    T = n

    model = ModelClass(
        prior_a=alpha0,
        prior_b=beta0,
        S1=S1,
        S2=S2,
        prior_mean_scale=0.0,
        prior_var_scale=1.0 / max(kappa0, 1e-12),
        general_nbh_sequence=[[[]]] * (S1 * S2),
        general_nbh_restriction_sequence=[[0]],
        general_nbh_coupling="weak coupling",
        hyperparameter_optimization=None,
    )
    model_universe = np.array([model])
    model_prior = np.array([1.0])
    cp_model = CpModel(intensity)

    detector = Detector(
        data=ys,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=S1,
        S2=S2,
        T=T,
        store_rl=False,
        store_mrl=False,
        trim_type="keep_K",
        threshold=100,
        notifications=10**9,
        save_performance_indicators=False,
        training_period=T,
        generalized_bayes_rld=generalized_bayes_rld,
        alpha_param_learning=None,
        alpha_param=alpha_param,
        alpha_rld=alpha_rld,
        alpha_rld_learning=False,
        loss_der_rld_learning="squared_loss",
        loss_param_learning="squared_loss",
    )
    detector.run()

    # Extract MAP CPs from the second-to-last time point, as in the repo examples.
    raw_cps = detector.CPs[-2]
    if raw_cps is None:
        cps = []
    else:
        cps = []
        for entry in raw_cps:
            if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                loc = int(entry[0])
            else:
                loc = int(entry)
            # Convert from 1-based to 0-based indexing.
            loc0 = loc - 1
            if 1 <= loc0 <= n - 2:
                cps.append(loc0)
        cps = sorted(set(cps))
except Exception as e:
    cps = []
    # Uncomment for debugging: print(str(e), file=sys.stderr)

with open(out_path, "w") as f:
    json.dump({"cps": cps}, f)
'''


def run_bocpdms_subprocess(algorithm, y, config, timeout):
    """Run one BOCPDMS/RBOCPDMS config in a subprocess with a hard timeout."""
    repo_root = str(BOCPDMS_ROOT if algorithm == "BOCPDMS" else RBOCPDMS_ROOT)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "y.npy"
        config_path = Path(tmpdir) / "config.json"
        out_path = Path(tmpdir) / "cps.json"
        np.save(data_path, np.asarray(y, dtype=float))
        with open(config_path, "w") as f:
            json.dump(config, f)

        try:
            subprocess.run(
                [
                    PYTHON,
                    "-c",
                    BOCPDMS_SUBPROCESS_TEMPLATE,
                    repo_root,
                    str(data_path),
                    str(config_path),
                    str(out_path),
                    algorithm,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            print(f"    {algorithm} config={config} failed: {stderr[:200]}")
            return None

        if not out_path.is_file():
            return None
        try:
            with open(out_path) as f:
                result = json.load(f)
            return [int(c) for c in result.get("cps", [])]
        except Exception:
            return None


def make_bocpdms_runner(algorithm):
    """Return a runner for BOCPDMS or RBOCPDMS."""
    def runner(y, true_cps, n, mode="oracle"):
        if mode == "default":
            configs = bocpdms_default_config()
            timeout = BOCPDMS_DEFAULT_TIMEOUT
        else:
            configs = bocpdms_oracle_grid()
            timeout = BOCPDMS_ORACLE_TIMEOUT

        best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "covering": 0.0, "cps": []}
        for cfg in configs:
            cps = run_bocpdms_subprocess(algorithm, y, cfg, timeout=timeout)
            if cps is None:
                continue
            p, r, f1, cov = score_cps(cps, true_cps, n)
            if f1 > best["f1"]:
                best.update(
                    {"f1": f1, "precision": p, "recall": r, "covering": cov, "cps": cps}
                )
        return {"method": algorithm, **best}

    return runner


# ---------------------------------------------------------------------------
# ZERO and HMM-Regime
# ---------------------------------------------------------------------------


def run_zero(y, true_cps, n, mode="oracle"):
    """ZERO baseline: no changepoints."""
    cps = []
    p, r, f1, cov = score_cps(cps, true_cps, n)
    return {"method": "ZERO", "f1": f1, "precision": p, "recall": r, "covering": cov, "cps": cps}


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from hmm_regime import run_hmm_regime


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Python method timed out")


def run_with_timeout(fn, args, timeout):
    """Run fn(*args) with a SIGALRM timeout (Linux)."""
    if timeout is None or timeout <= 0:
        return fn(*args)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        return fn(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def make_timed_hmm_runner():
    """HMM-Regime with a per-call timeout."""
    def runner(y, true_cps, n, mode="oracle"):
        try:
            return run_with_timeout(run_hmm_regime, (y, true_cps, n, mode), HMM_TIMEOUT)
        except TimeoutError:
            print("    HMM-Regime timed out")
            return {
                "method": "HMM-Regime",
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "covering": 0.0,
                "cps": [],
                "error": "timeout",
            }

    return runner


def make_timed_prophet_runner():
    """PROPHET with a per-call timeout."""
    def runner(y, true_cps, n, mode="oracle"):
        try:
            return run_with_timeout(run_prophet, (y, true_cps, n, mode), PROPHET_TIMEOUT)
        except TimeoutError:
            print("    PROPHET timed out")
            return {
                "method": "PROPHET",
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "covering": 0.0,
                "cps": [],
                "error": "timeout",
            }

    return runner


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------


def make_method_table(n):
    """Build the dispatch table for this dataset length (some grids depend on n)."""
    return [
        ("AMOC", make_r_runner("AMOC", changepoint_default_config(), changepoint_grid(), timeout=30)),
        ("PELT", make_r_runner("PELT", changepoint_default_config(), changepoint_grid(), timeout=30)),
        (
            "BinSeg",
            make_r_runner(
                "BinSeg",
                changepoint_default_config(),
                binseg_segneigh_grid(n),
                timeout=60,
            ),
        ),
        (
            "SegNeigh",
            make_r_runner(
                "SegNeigh",
                segneigh_default_config(),
                binseg_segneigh_grid(n),
                timeout=60,
            ),
        ),
        ("CPNP", make_r_runner("CPNP", cpnp_default_config(), cpnp_grid(), timeout=30)),
        (
            "RFPOP",
            make_r_runner(
                "RFPOP",
                rfpop_default_config(),
                rfpop_grid(),
                timeout=R_TIMEOUT_RFPOP,
            ),
        ),
        ("ECP", make_r_runner("ECP", ecp_default_config(), ecp_grid(), timeout=60)),
        ("KCPA", make_r_runner("KCPA", kcpa_default_config(), kcpa_grid(), timeout=30)),
        ("WBS", make_r_runner("WBS", wbs_default_config(), wbs_grid(n), timeout=30)),
        (
            "BOCPD",
            make_r_runner("BOCPD", bocpd_default_config(), bocpd_grid(), timeout=30),
        ),
        ("BOCPDMS", make_bocpdms_runner("BOCPDMS")),
        ("RBOCPDMS", make_bocpdms_runner("RBOCPDMS")),
        ("PROPHET", make_timed_prophet_runner()),
        ("HMM-Regime", make_timed_hmm_runner()),
        ("ZERO", run_zero),
    ]


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_benchmark(datasets, methods=None, progress=True):
    """Run all requested methods on the supplied datasets."""
    results = []
    for ds in datasets:
        model = ds.get("model")
        seed = ds.get("seed")
        noise = ds.get("noise_level")
        y = np.array(ds.get("y"), dtype=float)
        n = len(y)
        true_cps = [int(c) for c in ds.get("true_cps", [])]

        if progress:
            print(f"[{model}] seed={seed} noise={noise} n={n} true_cps={true_cps}")

        method_table = make_method_table(n)
        method_map = {name: fn for name, fn in method_table}
        if methods is not None:
            method_table = [(name, method_map[name]) for name in methods if name in method_map]

        for name, fn in method_table:
            for mode in ("default", "oracle"):
                start = time.time()
                try:
                    res = fn(y, true_cps, n, mode=mode)
                    res.setdefault("method", name)
                    res["config"] = mode
                    res["model"] = model
                    res["seed"] = seed
                    res["noise_level"] = noise
                    res["n"] = n
                    res["true_cps"] = true_cps
                    if "error" not in res:
                        res.pop("error", None)
                    results.append(res)
                    elapsed = time.time() - start
                    if progress:
                        print(
                            f"  {name:12s} {mode:7s} F1={res['f1']:.3f} "
                            f"P={res['precision']:.3f} R={res['recall']:.3f} "
                            f"Cov={res['covering']:.3f} CPs={res['cps']} "
                            f"({elapsed:.1f}s)"
                        )
                except Exception as exc:
                    print(f"  {name} {mode} failed: {exc}")
                    results.append(
                        {
                            "model": model,
                            "seed": seed,
                            "noise_level": noise,
                            "n": n,
                            "true_cps": true_cps,
                            "method": name,
                            "config": mode,
                            "f1": 0.0,
                            "precision": 0.0,
                            "recall": 0.0,
                            "covering": 0.0,
                            "cps": [],
                            "error": str(exc),
                        }
                    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test on one ODE instance with a few methods.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to run (default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter datasets by model name (e.g. ODE).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matching datasets.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Skip the first N matching datasets.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_PATH),
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    if not DATA_PATH.is_file():
        print(f"Dataset file not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(DATA_PATH) as f:
        datasets = json.load(f)

    if args.model:
        datasets = [ds for ds in datasets if ds.get("model") == args.model]
    if args.offset is not None:
        datasets = datasets[args.offset :]
    if args.limit is not None:
        datasets = datasets[: args.limit]

    if args.smoke_test:
        smoke_methods = ["PELT", "WBS", "PROPHET", "HMM-Regime"]
        if args.methods:
            smoke_methods = [m.strip() for m in args.methods.split(",")]
        print("=== Smoke test mode ===")
        print(f"Methods: {smoke_methods}")
        results = run_benchmark(datasets[:1], methods=smoke_methods, progress=True)
    else:
        methods = None
        if args.methods:
            methods = [m.strip() for m in args.methods.split(",")]
        results = run_benchmark(datasets, methods=methods, progress=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path} ({len(results)} records)")

    df = pd.DataFrame(results)
    if not df.empty and {"model", "method", "f1"}.issubset(df.columns):
        summary = df.groupby(["model", "method"])["f1"].mean().unstack("model")
        print("\nMean F1 summary:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
