#!/bin/bash
#SBATCH --job-name=gw_s4_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=session4_run_%j.out
#SBATCH --error=session4_run_%j.err

# Session 4 full run: 500 prompts x 3 tiers. Submit this THREE times on
# three different days (SESSION_4_PLAN.md 4.2-4.4) for the reproducibility
# gate; pass the run number so the outputs land in separate directories:
#
#   sbatch --export=ALL,RUN_ID=1 training/scripts/session4_full_run.sh
#
# 12h wall clock against a ~2.8h measured estimate (scaled from the
# 10-prompt dry run, job 1497: 4bit 14s + 8bit 27s + 16bit 163s per 10
# prompts). Worst case, if every prompt runs to the full 128 tokens AND
# host contention sits at the bad end of what micro_bench measured
# (4.4 tok/s), 16-bit alone is ~4h and the run is ~5h. SLURM bills actual
# usage rather than the request, so the extra headroom is free and stops
# a contended run from dying at the wall with Phase B never reached.
# phase_a_live.csv makes even that case recoverable.

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

# SMOKE_LIMIT lets us exercise THIS wrapper -- not a similar one -- on a
# handful of prompts before committing 4-6 unattended hours to it:
#   sbatch --export=ALL,RUN_ID=smoke,SMOKE_LIMIT=6 ... session4_full_run.sh
# Unset for real runs, which then evaluate the full 500-prompt set.
LIMIT_ARG=""
if [ -n "${SMOKE_LIMIT:-}" ]; then
  LIMIT_ARG="--limit ${SMOKE_LIMIT}"
  echo "SMOKE TEST: limiting to ${SMOKE_LIMIT} prompts -- not a real run"
fi

# Report what survived however we exit. Under `set -e` a failure in python
# would otherwise skip straight past the listing, which is exactly when we
# most want to know which partial results reached disk.
trap 'echo "=== run ${RUN_ID} exiting (status $?) ==="; ls -la "$OUT" || true' EXIT

python -u ../../../training/scripts/kaggle_routing_experiment.py \
  --output-dir "$OUT" $LIMIT_ARG
