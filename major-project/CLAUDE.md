# Project Instructions

## Project Overview
**Green-Weight**: Dynamic Bit-Width Scaling for Energy-Proportional LLM Inference using Fuzzy Logic Controllers.

This research project implements an "Energy Gearbox" for AI - dynamically adjusting LLM precision based on prompt complexity to save up to 40% energy with <1% accuracy loss.

## Project Structure

**Full mapping and the 6-subagent routing table live in the repo-root
`CLAUDE.md` (`D:/Green AI/CLAUDE.md`) — read that first.** Summary of where
things live after the Aug 2026 reorg:

```
major-project/
├── backend/src/green_weight/  # Main Python package (import path unchanged: `green_weight.xxx`)
│   ├── config.py              # Configuration (BitWidth, Gears, Fuzzy params)
│   ├── core/
│   │   ├── prompt_complexity.py   # Complexity sensor/scorer
│   │   └── dynamic_inference.py   # Inference engine with gear selection
│   ├── controllers/
│   │   └── fuzzy_gearbox.py       # Fuzzy logic controller
│   ├── training/
│   │   ├── qat_trainer.py         # QAT training classes (library)
│   │   └── kaggle_qat_trainer.py  # QAT training entrypoint (see training/scripts/ note below)
│   ├── evaluation/
│   │   └── benchmark.py           # Energy/accuracy benchmarking
│   ├── api.py                     # FastAPI backend
│   ├── main.py                    # CLI entry point
│   └── __init__.py
├── backend/tests/              # pytest suite (test-agent's domain)
├── frontend/                   # React/Vite dashboard (was greenai-dashboard/)
│   ├── src/
│   └── tests/
├── training/
│   ├── scripts/                # All GPU/session entrypoints: kaggle_*.py,
│   │                           # verify_results.py, make_figures.py,
│   │                           # prepare_eval_dataset.py, finetune_judger.py,
│   │                           # + the 4 training notebooks
│   ├── configs/                # Intended home for externalized QAT hyperparams (not wired up yet)
│   └── logs/                   # Raw GPU-session transcripts
├── contracts/api-spec.yaml     # Backend/frontend API contract
├── verification/checklist.md   # Living sign-off checklist (verifier's domain)
├── paper/
│   ├── results.md               # Append-only experiment log (paper-writer's only numeric source)
│   ├── draft.md
│   └── figures/
├── dataset/
│   └── dataset_prep.py         # Dataset curation (Objective 1) — left in place, not moved
├── adapters/                    # Trained LoRA adapters — left in place, not moved
├── .gitignore
├── README.md
└── CLAUDE.md                  # This file
```

## Key Components

### 1. Complexity Sensor (`core/prompt_complexity.py`)
- `ComplexityScorer` - Analyzes prompts using:
  - Flesch-Kincaid reading difficulty
  - Code pattern detection
  - Math symbol counting
  - Reasoning keyword detection
- Returns 0-100 complexity score

### 2. Fuzzy Gearbox (`controllers/fuzzy_gearbox.py`)
- `FuzzyGearbox` - Smooth bit-width selection using triangular membership functions
- `GearDecision` - Contains selected gear + confidence scores
- Avoids binary decisions: "70% simple, 30% medium → 6-bit equivalent"

### 3. Dynamic Inference (`core/dynamic_inference.py`)
- `DynamicInferenceEngine` - Main orchestrator
- Pipeline: Score → Fuzzy Decide → Generate
- Supports forced gears for comparison

### 4. QAT Training (`training/`)
- `FakeQuantizer` - Simulates low-bit quantization during training
- `BitResilientTrainer` - Cycles through bit-widths during training
- `kaggle_qat_trainer.py` - Complete Kaggle notebook for training adapters

### 5. Benchmarking (`evaluation/benchmark.py`)
- `EnergyBenchmark` - Compares fuzzy vs static approaches
- `AccuracyBenchmark` - Evaluates per-gear accuracy
- Energy estimation based on bit-width scaling

## Running the Project

### Demo
```bash
cd backend/src   # green_weight is imported as a top-level package from here

# Analyze prompts
python -m green_weight.main analyze

# Run inference demo
python -m green_weight.main demo

# Run benchmark
python -m green_weight.main benchmark
```

### Training on Kaggle
1. Upload `combined_cleaned.csv` to Kaggle dataset
2. Open `backend/src/green_weight/training/kaggle_qat_trainer.py`
3. Run cells sequentially
4. Download `adapters/` folder

## Research Objectives

| Objective | Status | Description |
|-----------|--------|-------------|
| 1. Dataset | ✅ | Curated Alpaca + OpenOrca + CodeAlpaca |
| 2. QAT Training | 🔄 | Kaggle script ready, need to train adapters |
| 3. Controller | ✅ | Fuzzy logic + complexity sensor implemented |
| 4. Benchmark | 🔄 | Framework ready, need full evaluation |

## HuggingFace Token
Never store tokens in this repo. Set the `HF_TOKEN` environment variable locally,
or use a Kaggle secret named `HF_TOKEN` on Kaggle (see KAGGLE_MANUAL.md).

## Key Design Decisions

1. **Model**: Llama-3.2-1B (fits on Kaggle T4, good for research)
2. **Quantization**: QLoRA with fake quantization for QAT
3. **Fuzzy Logic**: Triangular membership functions, centroid defuzzification
4. **Energy Model**: The linear bit-width scaling (4-bit=0.25x, 8-bit=0.5x, 16-bit=1x)
   is a DEMO-ONLY assumption used by `evaluation/benchmark.py`. All publishable numbers
   must come from measured NVML energy (see `training/scripts/kaggle_energy_benchmark.py`).

## Results
No verified results yet. Energy savings, accuracy loss, and latency numbers are
produced by the Kaggle sessions in KAGGLE_MANUAL.md and validated by
`training/scripts/verify_results.py`; see CREDIBILITY_REPORT.md for status.

## Dependencies
```
torch >= 2.0
transformers >= 4.35
peft >= 0.7
bitsandbytes >= 0.41
trl >= 0.7
datasets, pandas
```

## Next Steps
1. Run Kaggle training to get 3 LoRA adapters
2. Integrate adapters into inference engine
3. Run full benchmark evaluation
4. Generate trade-off curves for paper
