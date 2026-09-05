#!/bin/bash
#SBATCH --job-name=gw_micro
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=micro_%j.out
#SBATCH --error=micro_%j.err

# Isolates the Session 4 16-bit anomaly from the experiment itself.
# See micro_bench.py's docstring.

set -euo pipefail

# Confine every thread pool to the CPUs SLURM granted. torch, OpenMP and
# MKL all size their pools from the machine core count (224 threads here)
# while the cgroup permits only --cpus-per-task, so the default is heavy
# oversubscription: 100% CPU, ~156ms CPU per generated token, 6x variance.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

source ~/greenweight_env.sh

echo "=== GPU state before ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv

python -u ~/greenAI/major-project/training/scripts/micro_bench.py

echo "=== GPU state after ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv
echo "=== other jobs on this node ==="
squeue -w "$(hostname)" -o "%.8i %.18j %.12u %.2t %.10M %b"
