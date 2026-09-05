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

source ~/greenweight_env.sh

# green_weight/ must be cwd: router/fuzzy_controller.py does a bare
# `from config import get_config`, the same convention api.py relies on.
cd ~/greenAI/major-project/backend/src/green_weight

# -u matters: without unbuffered output the new per-prompt progress lines
# sit in a pipe buffer and the job looks hung even when it is fine.
python -u ../../../training/scripts/kaggle_routing_experiment.py \
  --limit 10 \
  --warmup 1 \
  --output-dir "$HOME/session4_dryrun_${SLURM_JOB_ID}_output"

echo "=== dry run finished, exit $? ==="
ls -la "$HOME/session4_dryrun_${SLURM_JOB_ID}_output"
