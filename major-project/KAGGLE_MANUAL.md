# Kaggle Manual — Training & Experiments, Step by Step

Every GPU experiment for this project runs on Kaggle's free tier. Follow the
sessions **in order**; each one tells you exactly what to click, what to run,
how long it takes, and which files to download. After each session, drop the
downloaded files into this repo and run the validator (Step 7).

---

## Step 0 — One-time setup (~15 min, no GPU)

1. **Kaggle account**: kaggle.com → sign up → verify your phone number
   (unverified accounts get no GPU). Free tier gives **30 GPU hours/week**.
2. **Hugging Face access to Llama**:
   - Create an account at huggingface.co.
   - Open https://huggingface.co/meta-llama/Llama-3.2-1B → "Agree and access
     repository" → wait for approval (usually minutes).
   - Profile → Settings → Access Tokens → New token (type: Read). Copy it.
3. **Store the token as a Kaggle secret** (never paste it into a notebook):
   - In any Kaggle notebook: Add-ons → Secrets → Add secret →
     Name: `HF_TOKEN`, Value: your token → attach it to each notebook you make.
   - First cell of every session notebook:
     ```python
     from kaggle_secrets import UserSecretsClient
     import os
     os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
     ```
4. **Accelerator**: in every session, Notebook → Settings → Accelerator →
   **GPU T4 x2** (or T4 x1). **Never P100** — it lacks the NVML hardware
   energy counter our measurements depend on.
5. **Upload two Kaggle datasets** (kaggle.com → Datasets → New Dataset):
   - `eval-prompts` — containing `eval_prompts.jsonl`
     (build it first: `python training/scripts/prepare_eval_dataset.py --num-prompts 500`,
     needs `pip install datasets` locally; or run that script in a Kaggle cell
     and download the output).
   - `greenweight-adapters` — containing the three folders from this repo's
     `adapters/` (`adapter_simple`, `adapter_medium`, `adapter_complex`).
     Re-upload after Session 3 if you retrain.

---

## Session 1 — Energy ground truth (~2–3 h GPU). RUN THIS FIRST.

Purpose: measure real joules-per-token for each precision tier. This is the
paper's Table 1 **and the go/no-go gate**: if 4-bit doesn't save energy vs
fp16, see "If Session 1 disappoints" below before spending more GPU hours.

1. New notebook → T4 → attach `HF_TOKEN` secret → add the `eval-prompts`
   dataset (Add Input → Datasets → yours).
2. Paste the secret cell (Step 0.3), then paste the entire contents of
   [`training/scripts/kaggle_energy_benchmark.py`](training/scripts/kaggle_energy_benchmark.py)
   into the next cell. Run all.
3. Watch for `NVML energy counter available: True` in the output. If it says
   `False` on a T4, restart the session (bad node).
4. When it finishes, download from the notebook's Output pane:
   - `energy_per_inference.csv`
   - `energy_summary.csv`
   - `hardware_info.json`
5. Put them in `backend/src/green_weight/results/energy_logs/` in this repo.

**If Session 1 disappoints** (4-bit ≥ fp16 energy): don't panic and don't
hide it. Options, per RESEARCH_PLAN.md §2: switch the low tiers to GPTQ/AWQ
checkpoints with real low-bit kernels, or reframe the paper around the
measurement study. Decide before running Session 4.

---

## Session 2 — Per-tier accuracy (~4–6 h GPU)

Purpose: benchmark accuracy of each tier, with and without the QAT adapters
(answers RQ4).

1. New notebook → T4 → `HF_TOKEN` secret → add the `greenweight-adapters`
   dataset as input.
2. Paste [`training/scripts/kaggle_accuracy_eval.py`](training/scripts/kaggle_accuracy_eval.py)
   and run. It evaluates 6 conditions (3 tiers × base/adapter) on
   tinyMMLU/tinyGSM8k/tinyHellaswag (falls back to arc_easy/gsm8k/hellaswag
   with a 200-example limit if tiny tasks are unavailable).
3. If the adapters fail to load (transformers/peft version drift since they
   were trained), the script logs a warning and runs base-only — in that
   case do Session 3 and re-run this session afterwards.
4. Download `accuracy_per_tier.json` and `accuracy_summary.csv` →
   `backend/src/green_weight/results/accuracy_logs/`.

---

## Session 3 — QAT adapter training (~6 h GPU, ONLY IF NEEDED)

Skip this if the existing `adapters/` loaded fine in Session 2 and the
adapter conditions scored ≥ base. Otherwise:

1. **Training data**: the trainer expects `combined_cleaned.csv` with a
   complexity split (built from Alpaca + OpenOrca + CodeAlpaca by
   `dataset/dataset_prep.py`). Build it locally (or in a Kaggle cell) and
   upload as a Kaggle dataset, e.g. `greenweight-training-data`.
2. New notebook → **T4 x2** → `HF_TOKEN` secret → add the training-data
   dataset as input.
3. Open [`backend/src/green_weight/training/kaggle_qat_trainer.py`](backend/src/green_weight/training/kaggle_qat_trainer.py).
   It is written as notebook cells (`CELL 1`, `CELL 2`, …): paste each CELL
   into its own notebook cell, in order.
4. In CELL 1, set `DATA_PATH` to your uploaded CSV, e.g.
   `/kaggle/input/greenweight-training-data/combined_cleaned.csv`.
   Leave `MODEL_NAME = "meta-llama/Llama-3.2-1B"` and
   `OUTPUT_PATH = "/kaggle/working/adapters"` as they are.
5. Run all cells. It trains three LoRA adapters (simple/medium/complex) with
   fake-quantization QAT, cycling bit-widths during training.
6. Kaggle sessions cap at ~12 h — this fits, but click "Save Version →
   Save & Run All (Commit)" so it survives your browser closing.
7. Download the `adapters/` folder from Output. Replace this repo's
   `adapters/` **and** update the `greenweight-adapters` Kaggle dataset
   (New Version) so Sessions 2 and 4 use the new weights.
8. Re-run Session 2.

---

## Session 4 — Main routing experiment (~4–6 h GPU)

Purpose: the paper's headline numbers — all 8 conditions (static 4/8/16-bit,
fuzzy router, matched-random, threshold, oracle, oracle-cascade) with
measured energy.

1. New notebook → T4 → `HF_TOKEN` secret → add BOTH `eval-prompts` and
   `greenweight-adapters` as inputs.
2. Paste [`training/scripts/kaggle_routing_experiment.py`](training/scripts/kaggle_routing_experiment.py)
   and run. Phase A measures every prompt on every tier (progress prints
   every 50 prompts); Phase B derives all routing conditions from those
   measurements on CPU.
3. Download all three outputs →  `backend/src/green_weight/results/routing_logs/`:
   - `routing_per_prompt.csv` (raw artifact — released with the paper)
   - `routing_conditions_summary.csv`
   - `routing_run_info.json`
4. **Repeat this session two more times on different days** (different
   physical nodes/thermals) and keep each run's files in subfolders
   `routing_logs/run1`, `run2`, `run3`. Cross-run agreement is your
   reproducibility evidence; put `run1`'s files at the top level for the
   scripts.

---

## Session 5 — Figures & validation (local, no GPU)

```powershell
cd "d:\Green AI\major-project"
pip install matplotlib
python training/scripts/verify_results.py      # credibility gate — read every WARN/FAIL
python training/scripts/make_figures.py        # writes backend/src/green_weight/results/figures/
```

`verify_results.py` writes `backend/src/green_weight/results/results_validation.md` with
a PASS/WARN/FAIL table. **A number may be quoted in the paper only if its
checks pass** — see CREDIBILITY_REPORT.md for the policy.

---

## Step 7 — After every session

1. Copy downloads into `backend/src/green_weight/results/…` as listed above.
2. `python training/scripts/verify_results.py` — fix FAILs before the next session.
3. Commit the CSVs (they are small) so results are versioned with the code.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `gated repo` loading Llama | HF license not approved yet, or secret not attached to this notebook. |
| `NVML energy counter available: False` | You're on P100 or an odd node — switch accelerator to T4 / restart. |
| OOM on 8-bit or fp16 | Another model wasn't freed: restart kernel; the scripts load one tier at a time. |
| Adapters fail to load in Session 2 | Version drift — run Session 3 to retrain, or pin `peft`/`transformers` to the training-time versions. |
| Session killed at 12 h | Use "Save & Run All (Commit)" mode; scripts flush CSVs incrementally, partial data is still usable. |
| lm-eval task not found | The script auto-falls back to `arc_easy/gsm8k/hellaswag` — check the printed task list. |
