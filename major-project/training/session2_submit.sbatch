#!/bin/bash
#SBATCH --job-name=greenai-session2-accuracy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=session2_%j.log
#SBATCH --error=session2_%j.err

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/major-project/training/scripts"
RESULTS_DIR="${REPO_ROOT}/major-project/training/results"

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "GREEN-WEIGHT SESSION 2: PER-TIER ACCURACY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Setup Python
echo "[1/3] Setting up Python environment..."
if [ ! -d "${REPO_ROOT}/venv" ]; then
    python3.11 -m venv "${REPO_ROOT}/venv"
fi
source "${REPO_ROOT}/venv/bin/activate"

# Install dependencies (use uv pip, NOT bare pip)
echo "[2/3] Installing dependencies..."
uv pip install -q torch transformers peft bitsandbytes pandas pynvml scikit-fuzzy textstat spacy lm-eval > /dev/null 2>&1 || true

# HF Token
echo "[3/3] Configuring HuggingFace..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi
export HF_TOKEN

# Main run (no srun — sbatch allocates GPU directly)
echo ""
echo "Starting accuracy evaluation (3 tiers × 3 tasks, estimated 4-6h)..."
echo ""

cd "${RESULTS_DIR}"

python3 "${SCRIPT_DIR}/kaggle_accuracy_eval.py"

EVAL_EXIT=$?

# Validate
echo ""
if [ $EVAL_EXIT -eq 0 ]; then
    echo "✓ Accuracy evaluation complete"

    echo ""
    echo "✓✓ SESSION 2 COMPLETE"
    echo "Results in: ${RESULTS_DIR}/"
    echo "  - accuracy_summary.csv (per-tier, per-task)"
    echo "  - accuracy_per_prompt.json (detailed, granular)"
    exit 0
else
    echo "✗ Accuracy evaluation failed"
    exit $EVAL_EXIT
fi
