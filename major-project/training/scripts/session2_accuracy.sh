#!/bin/bash
#SBATCH --job-name=gw_s2_acc
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=session2_%j.out
#SBATCH --error=session2_%j.err

# Session 2: per-tier accuracy via lm-eval, with and without the QAT
# adapters (6 conditions: {4,8,16}bit x {base, qat_adapter}).
#
# SESSION_2_BLOCKER.md diagnosed the original failure as srun environment
# isolation and concluded it needed containers or SLURM config changes.
# It did not: this script calls python DIRECTLY, with no srun, exactly as
# jobs 1494-1507 did successfully on 2026-09-05. For a single-node,
# single-task job sbatch has already allocated the node and the batch
# script runs on it -- srun was never needed.
#
# The second cause of "installed but the job cannot see it" was that the
# venv is Python 3.11 built with `uv venv` and has NO pip, so bare `pip`
# fell through to the system 3.9. Install with `uv pip install`, never pip.

set -euo pipefail

OUT="$HOME/session2_j${SLURM_JOB_ID}_output"

# Match the thread pools to the CPUs the cgroup actually permits; see
# session4_full_run.sh.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

source ~/greenweight_env.sh

echo "=== preflight ==="
python -c "import lm_eval, torch, peft, bitsandbytes; print('lm_eval', lm_eval.__version__)"

mkdir -p "$OUT"
cd "$OUT"          # the script writes its outputs into the working directory
git -C ~/greenAI rev-parse HEAD > git_commit.txt 2>/dev/null || true

trap 'echo "=== session 2 exiting (status $?) ==="; ls -la "$OUT" || true' EXIT

python -u ~/greenAI/major-project/training/scripts/kaggle_accuracy_eval.py
