#!/bin/bash
#SBATCH --job-name=covid_bic_multiseed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/mlotfi/mica_revision/second_revision/results/covid_multiseed_set2/slurm-%j.out
#SBATCH --error=/home/mlotfi/mica_revision/second_revision/results/covid_multiseed_set2/slurm-%j.err

set -e
cd /home/mlotfi/mica_revision

export PATH="/home/mlotfi/julia-1.12.5/bin:$PATH"

# Keep BLAS/Julia threading single so Distributed workers do not contend.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p second_revision/results/covid_multiseed_set2

stdbuf -oL /home/mlotfi/julia-1.12.5/bin/julia second_revision/scripts/run_covid_bic_multiseed_set2.jl \
    > second_revision/results/covid_multiseed_set2/run.log 2>&1
