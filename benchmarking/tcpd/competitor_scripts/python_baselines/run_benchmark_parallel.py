#!/usr/bin/env python3
"""Run the package-based toy benchmark in parallel over (dataset, method) pairs.

This driver spawns workers from a global pool.  Each worker calls
run_benchmark.py for a single dataset and a single method, writing a
per-(dataset,method) JSON file.  The driver then merges all outputs into the
final `benchmark_toydatasets_package_based.json`.

Using a global pool prevents a slow method on one dataset from blocking all
other work, and keeps CPU utilisation high on multi-core machines.

Usage:
    python run_benchmark_parallel.py [out_path]
"""
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUNNER = ROOT / "python" / "run_benchmark.py"
OUT_DIR = ROOT / "results" / "per_dataset"
DEFAULT_OUT = ROOT / "results" / "benchmark_toydatasets_package_based.json"

ALL_METHODS = [
    "AMOC", "PELT", "BinSeg", "SegNeigh", "CPNP", "RFPOP", "ECP", "KCPA",
    "WBS", "BOCPD", "BOCPDMS", "RBOCPDMS", "PROPHET", "HMM-Regime", "ZERO",
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out", nargs="?", default=str(DEFAULT_OUT),
        help="Path to merged output JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--methods", default=None,
        help="Comma-separated list of methods to run (default: all).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=16,
        help="Maximum number of parallel workers (default: %(default)s).",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first N datasets (default: %(default)s).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only N datasets (default: all).",
    )
    return parser.parse_args()


def run_dataset_method(index, method):
    """Run a single method on a single dataset."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ds{index:02d}_{method}.json"
    cmd = [
        PYTHON,
        "-u",
        str(RUNNER),
        "--offset",
        str(index),
        "--limit",
        "1",
        "--methods",
        method,
        "--out",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return index, method, out_path, result.returncode, result.stdout, result.stderr


def main():
    args = parse_args()
    out_path = Path(args.out)
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else ALL_METHODS

    data_path = ROOT / "data" / "toy_datasets.json"
    with open(data_path) as f:
        datasets = json.load(f)
    n = len(datasets)

    offset = args.offset
    limit = args.limit if args.limit is not None else n
    dataset_indices = list(range(offset, min(offset + limit, n)))

    tasks = [(i, m) for i in dataset_indices for m in methods]
    n_workers = min(len(tasks), args.max_workers)
    print(
        f"Running {len(tasks)} (dataset, method) tasks on up to {n_workers} workers...",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_dataset_method, i, m): (i, m) for i, m in tasks}
        for future in as_completed(futures):
            index, method, path, rc, stdout, stderr = future.result()
            if rc != 0:
                print(f"Dataset {index} method {method} FAILED (exit {rc}).", flush=True)
                print(stderr[-500:], file=sys.stderr)
            else:
                print(f"Dataset {index} method {method} done -> {path}", flush=True)

    # Merge outputs.
    merged = []
    missing = []
    for i in dataset_indices:
        for m in methods:
            path = OUT_DIR / f"ds{i:02d}_{m}.json"
            if path.is_file():
                with open(path) as f:
                    records = json.load(f)
                    # Expect one default + one oracle record per method.
                    merged.extend(records)
            else:
                missing.append((i, m))

    if missing:
        print(f"Warning: {len(missing)} (dataset, method) outputs missing", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote merged output: {out_path} ({len(merged)} records)", flush=True)


if __name__ == "__main__":
    main()
