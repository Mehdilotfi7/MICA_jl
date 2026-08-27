#!/bin/bash
# Generate visualizations for every result directory under the COVID application.
# Directories without the required CP/parameter CSVs are skipped automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

for dir in "$APP_DIR"/results*; do
    [ -d "$dir" ] || continue
    echo "============================================================"
    echo "Plotting: $dir"
    echo "============================================================"
    julia plot_covid_objectives_origset.jl "$dir" || true
    echo ""
done

echo "All available plots generated."
