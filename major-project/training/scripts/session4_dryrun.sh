#!/bin/bash
#SBATCH --job-name=gw_s4_dryrun
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=session4_dryrun_%j.out
#SBATCH --error=session4_dryrun_%j.err

# Session 4 dry run: 10 prompts x 3 tiers, all the way through Phase B.
#
# --time is deliberately 40 min, not 6h. With the fp16-activation fix in
# load_tier() this should finish in well under 10 minutes; if it is still
# running at 40 it is wedged, and SLURM kills it rather than letting it
# burn a GPU silently the way the Kaggle 8-bit stall did for ~4.7h.

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

# INTERLEAVE=1 measures each prompt across all tiers back-to-back in
# shuffled order rather than one whole tier at a time, so contention drift
# cannot land entirely on whichever tier runs last.
INTERLEAVE_ARG=""
if [ -n "${INTERLEAVE:-}" ]; then
  INTERLEAVE_ARG="--interleave"
  echo "INTERLEAVED ordering enabled"
fi

source ~/greenweight_env.sh

# green_weight/ must be cwd: router/fuzzy_controller.py does a bare
# `from config import get_config`, the same convention api.py relies on.
cd ~/greenAI/major-project/backend/src/green_weight

# -u matters: without unbuffered output the new per-prompt progress lines
# sit in a pipe buffer and the job looks hung even when it is fine.
python -u ../../../training/scripts/kaggle_routing_experiment.py \
  --limit 10 \
  --warmup 1 $INTERLEAVE_ARG \
  --output-dir "$HOME/session4_dryrun_${SLURM_JOB_ID}_output"

echo "=== dry run finished, exit $? ==="
ls -la "$HOME/session4_dryrun_${SLURM_JOB_ID}_output"
