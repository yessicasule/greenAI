#!/bin/bash
#SBATCH --job-name=greenai-session1-energy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=session1_%j.log
#SBATCH --error=session1_%j.err

set -e

# Setup paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/major-project/training/scripts"
DATA_DIR="${REPO_ROOT}/major-project/backend/src/green_weight/data"
RESULTS_DIR="${REPO_ROOT}/major-project/training/results"

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "GREEN-WEIGHT SESSION 1: ENERGY GROUND TRUTH"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo ""

# Python environment
echo "[1/4] Setting up Python environment..."
if [ ! -d "${REPO_ROOT}/venv" ]; then
    python3.11 -m venv "${REPO_ROOT}/venv"
fi
source "${REPO_ROOT}/venv/bin/activate"

# Dependencies
echo "[2/4] Installing dependencies..."
uv pip install -q torch transformers peft bitsandbytes pandas pynvml scikit-fuzzy textstat spacy 2>&1 | grep -v "already satisfied" || true
python3 -m spacy download en_core_web_sm > /dev/null 2>&1 || true

# HF Token
echo "[3/4] Configuring HuggingFace..."
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi
export HF_TOKEN

# Pre-flight check
echo "[4/4] Checking NVML energy counter support..."
python3 "${SCRIPT_DIR}/check_nvml_energy_support.py"

# Main run
echo ""
echo "Starting benchmark (3 runs × 3 tiers × ~167 prompts each)..."
echo "Estimated time: 2-3 hours"
echo ""

cd "${RESULTS_DIR}"

# Copy eval prompts if they exist (they should from Phase 1 prep)
if [ -f "${DATA_DIR}/eval_prompts.jsonl" ]; then
    cp "${DATA_DIR}/eval_prompts.jsonl" .
    echo "Using eval_prompts.jsonl (500 prompts)"
else
    echo "WARNING: eval_prompts.jsonl not found; will use fallback prompts"
fi

# Run benchmark (writes energy_per_inference.csv, energy_summary.csv, hardware_info.json to cwd)
python3 "${SCRIPT_DIR}/kaggle_energy_benchmark.py"
BENCHMARK_EXIT=$?

# Validate results
echo ""
echo "=========================================="
if [ $BENCHMARK_EXIT -eq 0 ]; then
    echo "✓ Benchmark complete. Validating..."
    python3 "${SCRIPT_DIR}/verify_results.py" energy_per_inference.csv
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓✓✓ SESSION 1 SUCCESS ✓✓✓"
        echo "Results in: ${RESULTS_DIR}/"
        echo "  - energy_per_inference.csv (detailed, one row per inference)"
        echo "  - energy_summary.csv (mean ± 95% CI per tier)"
        echo "  - hardware_info.json (GPU model, driver, etc)"
        exit 0
    else
        echo "✗ Validation failed — review output above"
        exit 1
    fi
else
    echo "✗ Benchmark failed"
    exit $BENCHMARK_EXIT
fi
