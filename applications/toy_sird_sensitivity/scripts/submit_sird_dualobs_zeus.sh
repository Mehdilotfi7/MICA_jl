#!/bin/bash
# Submit 4 dual-observable toy-SIRD detection jobs to Zeus (one per config).
# Each job runs all 9 (sigma x seed) combos for its configuration.

set -euo pipefail

BASE_DIR="/home/mlotfi/mica_revision"
SCRIPT_DIR="${BASE_DIR}/second_revision/dualobs/scripts"
OUT_DIR="${BASE_DIR}/second_revision/dualobs/results/data"
LOG_DIR="${BASE_DIR}/second_revision/dualobs/results/logs"
JULIA="/home/mlotfi/julia-1.12.5/bin/julia"

CPUS_PER_JOB=8
MEM_PER_JOB="32G"
WALLTIME="12:00:00"
PARTITION="batch"

mkdir -p "${LOG_DIR}"

for cfg in M1 M2 M3 M4; do
    job_name="sird_dualobs_${cfg}"
    tmp_script=$(mktemp)
    cat > "${tmp_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/slurm-%j.out
#SBATCH --error=${LOG_DIR}/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --time=${WALLTIME}
#SBATCH --partition=${PARTITION}
#SBATCH --chdir=${BASE_DIR}

set -euo pipefail

export JULIA_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "=== \$(date): ${job_name} on \$(hostname) with \${SLURM_CPUS_PER_TASK} CPUs ==="

"${JULIA}" --threads=\${SLURM_CPUS_PER_TASK} --project="${BASE_DIR}/Mica.jl" \
    "${SCRIPT_DIR}/run_sird_complexity_dualobs.jl" \
    "${cfg}" \
    "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${job_name}.log"

echo "=== \$(date): ${job_name} done ==="
EOF

    echo "Submitting ${job_name}"
    sbatch "${tmp_script}"
    rm "${tmp_script}"
done

echo "All 4 dual-observable detection jobs submitted."
