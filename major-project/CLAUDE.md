# Project Instructions

**Start here every session: read `NEW.md` (this folder) first.** It's the
live, checkbox-tracked roadmap (Phase 0-9) and single source of truth for
current status — which GPU sessions have run, what's verified, what's
next. Everything below is static project structure/design, not day-to-day
status.

## Project Overview
**Green-Weight**: Dynamic Bit-Width Scaling for Energy-Proportional LLM Inference using Fuzzy Logic Controllers.

This research project implements an "Energy Gearbox" for AI - dynamically
adjusting LLM precision based on prompt complexity. **"~40% energy savings,
<1% accuracy loss" is an unverified legacy number from an early linear
bit-width model, not a measured result — see NEW.md's "what NOT to do"
section. Do not state it as fact anywhere, including in conversation.** The
project's actual commitment (RESEARCH_PLAN.md) is to publish whatever the
real measured numbers turn out to be.

## Project Structure

**Full mapping and the 6-subagent routing table live in the repo-root
`CLAUDE.md` (`D:/Green AI/CLAUDE.md`) — read that first.** Summary of where
things live after the Aug 2026 reorg:

```
major-project/
├── backend/src/green_weight/  # Main Python package (import path unchanged: `green_weight.xxx`)
│   ├── config.py               # Config loader (config.yaml)
│   ├── router/
│   │   ├── complexity_scorer.py   # Complexity sensor (5 features)
│   │   ├── fuzzy_controller.py    # Fuzzy logic controller (scikit-fuzzy)
│   │   └── routellm_bridge.py     # Documented threshold pass-through (not a real RouteLLM call)
│   ├── models/
│   │   ├── model_pool.py          # Loads/registers the 3 quantized tiers + QAT adapters
│   │   └── local_llm_adapter.py   # Unified get_completion() interface over the 3 tiers
│   ├── benchmark/
│   │   ├── energy_tracker.py      # Energy measurement (live demo/API path)
│   │   └── accuracy_eval.py       # lm-eval-based accuracy measurement (live demo/API path)
│   ├── cascade/
│   │   └── frugal_cascade.py      # FrugalGPT-style cascade — escalation logic unbuilt/stubbed,
│   │                               # optional scope per NEW.md Phase 0, not required for the core claim
│   ├── evaluation/
│   │   └── __init__.py            # Stub only — explains where real logic now lives (see above);
│   │                               # the old AccuracyBenchmark/EnergyBenchmark classes that used
│   │                               # to live here were System-B-only and moved to _legacy/
│   ├── _legacy/                   # Archived System B (core/, controllers/, dynamic_inference.py,
│   │                               # evaluation_benchmark.py) — not imported by anything live
│   ├── training/
│   │   ├── qat_trainer.py         # QAT training classes (library)
│   │   └── kaggle_qat_trainer.py  # STALE, do not use for retraining — see
│   │                               # "QAT Training" note below; the real
│   │                               # script is training/scripts/adapter-training.ipynb
│   ├── api.py                     # FastAPI backend
│   ├── run_pipeline.py            # Main CLI orchestrator (routing + cascade + energy + accuracy + plots)
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

**Canonical system as of 2026-08-05** (see `_legacy/README.md` for the
divergent implementation that used to compete with this one — archived, not
imported by anything live):

### 1. Complexity Sensor (`router/complexity_scorer.py`)
- 5 features grounded in established metrics: Flesch-Kincaid (textstat),
  token length, Shannon entropy, spaCy syntax-tree depth, has_code_or_math
- `score(prompt)` returns a dict of the 5 normalized (0-1) features

### 2. Fuzzy Controller (`router/fuzzy_controller.py`)
- `FuzzyController` — a real Mamdani fuzzy-inference system built on the
  `scikit-fuzzy` library: independent membership functions per feature,
  a rule base (`ctrl.Rule`), library-computed defuzzification
- Breakpoints are read from `config.yaml` (supports ablation studies);
  normalization constants are calibrated against the real eval-prompt
  distribution, not arbitrary (see the comment in the file for the method)
- Output: complexity score → tier (4bit/8bit/16bit) + `win_probability`
  (despite the name, this is `complexity_score / 100`, not a RouteLLM
  classifier output — see below)

### 3. RouteLLM Bridge (`router/routellm_bridge.py`)
- Currently a **documented, provisional threshold pass-through** — RouteLLM's
  pretrained checkpoints are trained on human-preference battles between
  specific named models (GPT-4, Mixtral, ...) and can't be meaningfully
  applied to our quantization tiers of one model
- A real tier-preference router, trained on Session 4's
  `routing_per_prompt.csv`, is planned as a post-Session-4 addition
  (see `RESEARCH_PLAN.md` RQ3) — that will be the actual RouteLLM-methodology
  contribution, not this placeholder

### 4. QAT Training (`training/`)
- `FakeQuantizer` - Simulates low-bit quantization during training
- `BitResilientTrainer` - Cycles through bit-widths during training
- **The real training script is `training/scripts/adapter-training.ipynb`**
  — confirmed 2026-08-22 by diffing its `LoraConfig` calls against the
  actual adapters' `adapter_config.json`: exact match
  (`adapter_simple` r=16/α=32, `adapter_medium` r=8/α=16,
  `adapter_complex` r=4/α=8, `target_modules=[q_proj, v_proj]`).
  `training/scripts/kaggle_qat_trainer.py` and
  `training/scripts/major-project-v2.ipynb` are a **stale, unused pair**
  with different (wrong) hyperparameters — r=32/16/8, 7 target modules,
  and the wrong adapter names (`adapter_4bit/8bit/16bit` instead of
  `adapter_simple/medium/complex`). They were never archived to
  `_legacy/` the way the old `core/`/`controllers/` implementation was,
  so they still read as live. Do not follow them for retraining; do not
  externalize their hyperparameters into `training/configs/` either — see
  that directory's README.
  `training/scripts/majorproject-final.ipynb` and
  `majorproject-final (1).ipynb` are earlier, broader exploratory
  notebooks (install `routellm`/`codecarbon` directly, predate the
  `router/` refactor) — not wired into anything documented as canonical,
  likely safe to ignore but not yet formally archived either.

### 5. Real measurement (`benchmark/`, `training/scripts/kaggle_*.py`)
- `benchmark/energy_tracker.py` + `benchmark/accuracy_eval.py` — used by the
  live `api.py`/`run_pipeline.py` path
- `training/scripts/kaggle_energy_benchmark.py`,
  `kaggle_accuracy_eval.py`, `kaggle_routing_experiment.py` — the actual GPU
  measurement scripts that produce the paper's numbers. As of this
  consolidation, `kaggle_routing_experiment.py` imports the real
  `router/` modules directly (shipped to the GPU box) instead of embedding a
  second, drifted copy of the routing logic.

## Running the Project

There is no `main.py`/CLI-demo entrypoint (an earlier version of this doc
described one; it never existed in the live `router/`-based system). Every
module here uses bare same-directory imports (`from config import
get_config`, not `from green_weight.config import ...`) — see
`green_weight/__init__.py`'s docstring — so **cwd must be inside
`backend/src/green_weight/` itself**, not `backend/src/`; `python -m
green_weight.run_pipeline` from `backend/src` does not work (verified
2026-08-22 — it 404s on the bare imports).

```bash
cd backend/src/green_weight

# Routing-only sanity check — no GPU, no model loading, just the fuzzy
# controller's tier decisions over the eval prompt set
python run_pipeline.py --routing-only --dry-run

# Full pipeline (needs a CUDA GPU + HF_TOKEN): loads all 3 quantized tiers
# + QAT adapters, routes + runs cascade inference + energy tracking +
# accuracy eval + trade-off plots
python run_pipeline.py

# FastAPI backend
uvicorn api:app --reload
```

### Training on Kaggle
1. Upload `combined_cleaned.csv` to Kaggle dataset
2. Open `training/scripts/adapter-training.ipynb` (**not**
   `kaggle_qat_trainer.py` — that script is stale/unused, see "QAT
   Training" above)
3. Run cells sequentially
4. Download `adapters/` folder

## Research Objectives

| Objective | Status | Description |
|-----------|--------|-------------|
| 1. Dataset | ✅ | Curated Alpaca + OpenOrca + CodeAlpaca; 500-prompt stratified eval set built |
| 2. QAT Training | ✅ | 3 adapters exist and are fully verified loadable — real load test against actual base-model weights completed 2026-08-22 on CPU (`adapters/adapter_{simple,medium,complex}`, see NEW.md Phase 3). Real training script identified as `adapter-training.ipynb`, not the stale `kaggle_qat_trainer.py`. Their *effect* on accuracy vs. plain PTQ is still pending Session 2. |
| 3. Controller | ✅ | Fuzzy logic + complexity sensor implemented, recalibrated against the full 500-prompt set (2026-08-22) |
| 4. Benchmark | 🔄 | Framework fixed and wired to real `lm_eval` (2026-08-22; was previously fabricating numbers, see CREDIBILITY_REPORT.md) — no GPU measurement run yet |

## HuggingFace Token
Never store tokens in this repo. Set the `HF_TOKEN` environment variable locally,
or use a Kaggle secret named `HF_TOKEN` on Kaggle (see KAGGLE_MANUAL.md).

## Key Design Decisions

1. **Model**: Llama-3.2-1B (fits on Kaggle T4, good for research)
2. **Quantization**: QLoRA with fake quantization for QAT
3. **Fuzzy Logic**: Triangular membership functions, centroid defuzzification
4. **Energy Model**: The linear bit-width scaling (4-bit=0.25x, 8-bit=0.5x, 16-bit=1x)
   is a DEMO-ONLY assumption. It lived in the old System-B `evaluation/benchmark.py`,
   now archived at `_legacy/evaluation_benchmark.py` — `evaluation/__init__.py` is a
   stub explaining the move, not a live module. All publishable numbers must come
   from measured NVML energy (see `training/scripts/kaggle_energy_benchmark.py`).

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
See `NEW.md` for the live, ordered roadmap. Short version as of 2026-08-22:
the 3 LoRA adapters already exist and are trained; they're now correctly
wired into `models/model_pool.py` (a path bug that silently skipped loading
them was found and fixed today). What's left is almost entirely GPU-gated:
Session 1 (energy ground truth, the go/no-go gate), Session 2 (per-tier
accuracy), Session 4 (main routing experiment), all on the college GPU
cluster — see NEW.md Phase 1 onward.
