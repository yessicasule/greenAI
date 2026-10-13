#!/bin/bash
#SBATCH --job-name=gw_s5_vllm
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=session5_%j.out
#SBATCH --error=session5_%j.err

# Session 5: per-token energy under vLLM, one process per configuration.
# Trial first (10 prompts per config):
#   sbatch --export=ALL,RUN_ID=smoke,SMOKE_LIMIT=10 training/scripts/session5_vllm.sh
# Real run (submit 3x on different nights):
#   sbatch --export=ALL,RUN_ID=1 training/scripts/session5_vllm.sh
# Override the configuration list with CONFIGS="label|model|quant label|model|quant".

set -uo pipefail

RUN_ID="${RUN_ID:-1}"
OUT="$HOME/session5_run${RUN_ID}_j${SLURM_JOB_ID}"
BATCH_SIZES="${BATCH_SIZES:-1 8}"
CONFIGS="${CONFIGS:-\
1b_fp16|meta-llama/Llama-3.2-1B|none \
1b_fp8|meta-llama/Llama-3.2-1B|fp8 \
8b_fp16|meta-llama/Llama-3.1-8B-Instruct|none \
8b_fp8|meta-llama/Llama-3.1-8B-Instruct|fp8 \
8b_awq4|hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4|none}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING

source ~/vllm_env.sh

mkdir -p "$OUT"
git -C ~/greenAI rev-parse HEAD > "$OUT/git_commit.txt" 2>/dev/null || true
nvidia-smi > "$OUT/nvidia_smi.txt" 2>&1 || true
LIMIT_ARG=""
[ -n "${SMOKE_LIMIT:-}" ] && LIMIT_ARG="--limit ${SMOKE_LIMIT}" && echo "SMOKE TEST: ${SMOKE_LIMIT} prompts per config"

FAILED=0
for cfg in $CONFIGS; do
  IFS='|' read -r label model quant <<< "$cfg"
  QARG=""
  [ "$quant" != "none" ] && QARG="--quantization $quant"
  for bs in $BATCH_SIZES; do
    echo "===== $label  bs=$bs  ($model, quant=$quant)  $(date '+%F %T') ====="
    python -u ~/greenAI/major-project/training/scripts/vllm_energy_benchmark.py \
      --tier "$label" --model "$model" $QARG --batch-size "$bs" $LIMIT_ARG \
      --output-dir "$OUT/${label}_bs${bs}" || { echo "!!! $label bs=$bs FAILED"; FAILED=1; }
  done
done

echo "=== run ${RUN_ID} done $(date '+%F %T'); failures=$FAILED ==="
ls -R "$OUT" | head -60
exit $FAILED
