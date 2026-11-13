#!/bin/bash
# Submit dual-observable toy-SIRD PLE jobs to Zeus.
# 2 sets (true, detected) x 4 configs (M1-M4) x 2 modes (conditional, joint).
#
# Usage: bash submit_sird_ple_dualobs_zeus.sh [conditional|joint|all]
# Submit conditional first; submit joint after the conditional jobs finish
# (the joint runs adopt the conditional reference fit when it is better).

set -euo pipefail

BASE_DIR="/home/mlotfi/mica_revision"
SCRIPT_DIR="${BASE_DIR}/second_revision/dualobs/scripts"
JULIA="/home/mlotfi/julia-1.12.5/bin/julia"

CPUS_PER_JOB=8
MEM_PER_JOB="32G"
WALLTIME="3-00:00:00"
PARTITION="batch"

MODE_FILTER="${1:-all}"

for cpset in true detected; do
    if [ "${cpset}" = "true" ]; then
        out_base="${BASE_DIR}/second_revision/dualobs/results/ple"
    else
        out_base="${BASE_DIR}/second_revision/dualobs/results/ple_detected"
    fi
    for cfg in M1 M2 M3 M4; do
        for mode in conditional joint; do
            if [ "${MODE_FILTER}" != "all" ] && [ "${mode}" != "${MODE_FILTER}" ]; then
                continue
            fi
            job_name="sird_dualple_${cpset}_${cfg}_${mode}"
            out_dir="${out_base}/${cfg}_${mode}"
            mkdir -p "${out_dir}"

            tmp_script=$(mktemp)
            cat > "${tmp_script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${out_dir}/slurm-%j.out
#SBATCH --error=${out_dir}/slurm-%j.err
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
    "${SCRIPT_DIR}/run_sird_ple_dualobs.jl" \
    "${cfg}" \
    "${mode}" \
    "${cpset}" \
    "${out_dir}" \
    2>&1 | tee "${out_dir}/ple.log"

echo "=== \$(date): ${job_name} done ==="
EOF

            echo "Submitting ${job_name} -> ${out_dir}"
            sbatch "${tmp_script}"
            rm "${tmp_script}"
        done
    done
done

echo "Done (mode filter: ${MODE_FILTER})."
