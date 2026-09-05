#!/bin/bash
#SBATCH --job-name=gw_s4_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=session4_run_%j.out
#SBATCH --error=session4_run_%j.err

# Session 4 full run: 500 prompts x 3 tiers. Submit this THREE times on
# three different days (SESSION_4_PLAN.md 4.2-4.4) for the reproducibility
# gate; pass the run number so the outputs land in separate directories:
#
#   sbatch --export=ALL,RUN_ID=1 training/scripts/session4_full_run.sh
#
# 8h wall clock against a 4-6h budget: enough margin for a slow 8-bit tier
# without letting a wedged job sit on the GPU for a full day.

set -euo pipefail

RUN_ID="${RUN_ID:-1}"
OUT="$HOME/session4_run${RUN_ID}_j${SLURM_JOB_ID}_output"

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
cd ~/greenAI/major-project/backend/src/green_weight

# Record the exact code state alongside the numbers: three runs are only a
# reproducibility check if all three ran the same script (SESSION_4_PLAN.md
# note 4, "do not mix runs from different code versions").
mkdir -p "$OUT"
git -C ~/greenAI rev-parse HEAD > "$OUT/git_commit.txt" 2>/dev/null || \
  echo "not a git checkout" > "$OUT/git_commit.txt"
nvidia-smi > "$OUT/nvidia_smi.txt" 2>&1 || true

python -u ../../../training/scripts/kaggle_routing_experiment.py \
  --output-dir "$OUT"

echo "=== run ${RUN_ID} finished, exit $? ==="
ls -la "$OUT"
