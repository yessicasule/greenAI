#!/bin/bash
#SBATCH --job-name=gw_s2_test
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=test_session2_%j.out
#SBATCH --error=test_session2_%j.err

# Quick preflight for Session 2: test lm-eval import, model load, task availability.
# If this passes, the full session2_accuracy.sh is safe to submit.

set -euo pipefail

source ~/greenweight_env.sh

echo "=== Imports ==="
python -c "import lm_eval, torch, peft, bitsandbytes; print('lm_eval', lm_eval.__version__); print('torch', torch.__version__)"

echo "=== Load Llama-3.2-1B fp16 (no generation, just verify dtype) ==="
python <<'PYEOF'
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    dtype=torch.float16,
    device_map={"": 0}
)
elapsed = time.perf_counter() - t0
mem_gb = torch.cuda.memory_allocated() / 1e9

print(f"Loaded in {elapsed:.1f}s")
print(f"GPU memory: {mem_gb:.2f} GB (fp16 ~2.5, fp32 ~4.9)")
print(f"model.dtype = {model.dtype}")

print("\n=== Check lm-eval task availability ===")
from lm_eval.tasks import TaskManager
available = set(TaskManager().all_tasks)
print(f"Total tasks available: {len(available)}")

preferred = ['tinyMMLU', 'tinyGSM8k', 'tinyHellaswag']
fallback = ['arc_easy', 'gsm8k', 'hellaswag']

print("Preferred tiny* tasks:")
for task in preferred:
    status = '✓' if task in available else '✗'
    print(f"  {task}: {status}")

print("Fallback full tasks:")
for task in fallback:
    status = '✓' if task in available else '✗'
    print(f"  {task}: {status}")

if all(t in available for t in preferred):
    print("\n→ Session 2 will use tinyMMLU, tinyGSM8k, tinyHellaswag")
else:
    print("\n→ Session 2 will use fallback tasks with --limit 200 per task")

print("\n=== Preflight complete ===")
PYEOF

echo "If all above succeeded, submit: sbatch training/scripts/session2_accuracy.sh"
