#!/bin/bash
# Run D2D-style PLE locally for all 400-day COVID winners (global parameters only).
# Usage: bash run_covid_ple_local.sh [n_points] [max_iter]

set -euo pipefail

PROJECT=$(cd "$(dirname "$0")/../../.." && pwd)
JULIA=${JULIA:-$(command -v julia)}
SCRIPT=$PROJECT/applications/covid/scripts/covid_ple_winner.jl
OUT_ROOT=$PROJECT/applications/covid/results/ple_winners

export COVID_PLE_NPOINTS="${1:-20}"
export COVID_PLE_ITER="${2:-200}"
export COVID_PLE_PARAMS="global"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"

mkdir -p "$OUT_ROOT"

# Task G rows from cluster/covid_winners_plan.csv
winners=(
    "bic:$PROJECT/applications/covid/results/task_a/bic/covid_detected_cps_origset_bic.csv:$PROJECT/applications/covid/results/task_a/bic/covid_params_origset_bic.csv"
    "mdl:$PROJECT/applications/covid/results/task_a/mdl/covid_detected_cps_origset_mdl.csv:$PROJECT/applications/covid/results/task_a/mdl/covid_params_origset_mdl.csv"
    "invmean_boxcox_aic:$PROJECT/applications/covid/results/task_b_extended/invmean_boxcox_aic/covid_detected_cps_origset_invmean_boxcox_aic.csv:$PROJECT/applications/covid/results/task_b_extended/invmean_boxcox_aic/covid_params_origset_invmean_boxcox_aic.csv"
    "invstd_boxcox_aic:$PROJECT/applications/covid/results/task_b_extended/invstd_boxcox_aic/covid_detected_cps_origset_invstd_boxcox_aic.csv:$PROJECT/applications/covid/results/task_b_extended/invstd_boxcox_aic/covid_params_origset_invstd_boxcox_aic.csv"
    "invstd_sqrt_aic:$PROJECT/applications/covid/results/task_b_extended/invstd_sqrt_aic/covid_detected_cps_origset_invstd_sqrt_aic.csv:$PROJECT/applications/covid/results/task_b_extended/invstd_sqrt_aic/covid_params_origset_invstd_sqrt_aic.csv"
)

for entry in "${winners[@]}"; do
    IFS=':' read -r label cps params <<< "$entry"
    out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"
    echo "========================================"
    echo "Running PLE for $label"
    echo "CPs:   $cps"
    echo "Params: $params"
    echo "Output: $out_dir"
    echo "NPOINTS=$COVID_PLE_NPOINTS ITER=$COVID_PLE_ITER"
    echo "========================================"
    "$JULIA" --project="$PROJECT/Mica.jl" "$SCRIPT" \
        "$label" "$cps" "$params" "$out_dir" "$COVID_PLE_PARAMS" \
        2>&1 | tee "$out_dir/ple.log"
    echo "Finished $label"
done

echo "All PLE runs finished. Outputs in $OUT_ROOT"
