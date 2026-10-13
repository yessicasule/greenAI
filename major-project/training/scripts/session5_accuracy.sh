#!/bin/bash
#SBATCH --job-name=gw_s5_acc
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=session5acc_%j.out
#SBATCH --error=session5acc_%j.err

# Session 5 accuracy: tinyBenchmarks per vLLM configuration (lm-eval vllm backend).
# Accuracy does not depend on host load, so this can run in daytime.
# Trial (first configuration only):
#   sbatch --export=ALL,SMOKE=1 training/scripts/session5_accuracy.sh

set -uo pipefail

OUT="$HOME/session5_acc_j${SLURM_JOB_ID}"
CONFIGS="${CONFIGS:-\
1b_fp16|meta-llama/Llama-3.2-1B|none \
1b_fp8|meta-llama/Llama-3.2-1B|fp8 \
8b_fp16|meta-llama/Llama-3.1-8B-Instruct|none \
8b_fp8|meta-llama/Llama-3.1-8B-Instruct|fp8 \
8b_awq4|hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4|none}"

export TOKENIZERS_PARALLELISM=false
source ~/vllm_env.sh
mkdir -p "$OUT"
git -C ~/greenAI rev-parse HEAD > "$OUT/git_commit.txt" 2>/dev/null || true

FAILED=0
for cfg in $CONFIGS; do
  IFS='|' read -r label model quant <<< "$cfg"
  MARGS="pretrained=$model,dtype=float16,gpu_memory_utilization=0.85,max_model_len=2048,seed=42"
  [ "$quant" != "none" ] && MARGS="$MARGS,quantization=$quant"
  echo "===== $label ($MARGS) $(date '+%F %T') ====="
  lm_eval --model vllm --model_args "$MARGS" \
    --tasks tinyMMLU,tinyGSM8k,tinyHellaswag --batch_size auto --seed 42 \
    --output_path "$OUT/$label" || { echo "!!! $label FAILED"; FAILED=1; }
  [ -n "${SMOKE:-}" ] && break
done
echo "=== accuracy done $(date '+%F %T'); failures=$FAILED ==="
exit $FAILED
