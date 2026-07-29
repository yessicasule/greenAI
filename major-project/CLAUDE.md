# Project Instructions

## Project Overview
**Green-Weight**: Dynamic Bit-Width Scaling for Energy-Proportional LLM Inference using Fuzzy Logic Controllers.

This research project implements an "Energy Gearbox" for AI - dynamically adjusting LLM precision based on prompt complexity to save up to 40% energy with <1% accuracy loss.

## Project Structure
```
D:\Major Project\major-project
├── green_weight/              # Main package
│   ├── config.py              # Configuration (BitWidth, Gears, Fuzzy params)
│   ├── core/
│   │   ├── prompt_complexity.py   # Complexity sensor/scorer
│   │   └── dynamic_inference.py   # Inference engine with gear selection
│   ├── controllers/
│   │   └── fuzzy_gearbox.py       # Fuzzy logic controller
│   ├── training/
│   │   ├── qat_trainer.py         # QAT training classes
│   │   ├── kaggle_qat_trainer.py  # Kaggle notebook script
│   │   └── README.md              # Training instructions
│   ├── evaluation/
│   │   └── benchmark.py           # Energy/accuracy benchmarking
│   ├── main.py                    # CLI entry point
│   └── __init__.py
├── dataset/
│   └── dataset_prep.py        # Dataset curation (Objective 1)
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
# Analyze prompts
python -m green_weight.main analyze

# Run inference demo
python -m green_weight.main demo

# Run benchmark
python -m green_weight.main benchmark
```

### Training on Kaggle
1. Upload `combined_cleaned.csv` to Kaggle dataset
2. Open `green_weight/training/kaggle_qat_trainer.py`
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
   must come from measured NVML energy (see `scripts/kaggle_energy_benchmark.py`).

## Results
No verified results yet. Energy savings, accuracy loss, and latency numbers are
produced by the Kaggle sessions in KAGGLE_MANUAL.md and validated by
`scripts/verify_results.py`; see CREDIBILITY_REPORT.md for status.

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
