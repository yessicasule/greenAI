#!/bin/bash
#SBATCH --job-name=gw_s3_train
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=session3_%j.out
#SBATCH --error=session3_%j.err

# Session 3: retrain the 3 QAT adapters. Trial first:
#   sbatch --export=ALL,SMOKE_LIMIT=8 training/scripts/session3_retrain.sh
# then the real run:
#   sbatch training/scripts/session3_retrain.sh
# New adapters go to $OUTPUT_DIR (default major-project/adapters_retrained/),
# never over major-project/adapters/.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

source ~/greenweight_env.sh

if [ -n "${SMOKE_LIMIT:-}" ]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/session3_smoke_j${SLURM_JOB_ID}}"
fi

echo "=== preflight ==="
python -c "import trl, peft, bitsandbytes, datasets; print('trl', trl.__version__, 'peft', peft.__version__)"

git -C ~/greenAI rev-parse HEAD || true

python -u ~/greenAI/major-project/training/scripts/train_adapters.py
