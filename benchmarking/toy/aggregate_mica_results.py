"""Aggregate MICA simulation JSONs into benchmark_toydatasets_mica_tcpd.json.

Oracle: best F1 over all objectives and grids.
Practical/default: lowest objective value over all objectives and grids.
"""
import json
import glob
import os
from pathlib import Path

BASE = Path('benchmarking/Toy')
SIM_DIR = BASE / 'simulations'
OUT_PATH = BASE / 'data' / 'benchmark_toydatasets_mica_tcpd.json'

def load_records():
    records = []
    for model in ('ODE', 'LR', 'AR'):
        for path in sorted((SIM_DIR / model).glob('ds*.json')):
            with open(path) as f:
                rec = json.load(f)
            if 'error' in rec:
                continue
            records.append(rec)
    return records

def dataset_key(rec):
    return (rec['model'], rec['seed'], rec['noise_level'], rec['n'])

def make_mica_record(rec, method, config):
    return {
        'model': rec['model'],
        'seed': rec['seed'],
        'noise_level': rec['noise_level'],
        'n': rec['n'],
        'true_cps': rec['true_cps'],
        'method': method,
        'config': config,
        'objective': rec['objective'],
        'min_length': rec.get('min_length', 10),
        'step': rec.get('step', 10),
        'f1': rec['f1'],
        'precision': rec['precision'],
        'recall': rec['recall'],
        'covering': rec['covering'],
        'cps': rec['cps'],
        'obj_value': rec['obj_value'],
    }

def main():
    records = load_records()
    by_ds = {}
    for r in records:
        by_ds.setdefault(dataset_key(r), []).append(r)

    out = []
    for key in sorted(by_ds):
        recs = by_ds[key]
        # Oracle: best F1
        oracle = max(recs, key=lambda r: r['f1'])
        out.append(make_mica_record(oracle, 'MICA-O-TCPD', 'oracle'))
        # Practical/default: lowest objective value
        practical = min(recs, key=lambda r: r['obj_value'] if r['obj_value'] is not None else float('inf'))
        out.append(make_mica_record(practical, 'MICA-P-TCPD', 'default'))

    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {OUT_PATH} with {len(out)} records')

    for r in out:
        print(f"{r['model']} noise={r['noise_level']} {r['config']}: {r['method']} obj={r['objective']} F1={r['f1']:.3f} CPs={r['cps']}")

if __name__ == '__main__':
    main()
