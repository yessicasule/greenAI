# Session 2 Submission Guide — Per-Tier Accuracy & Correlation Study

**Goal:** Measure accuracy per precision tier (4-bit, 8-bit, 16-bit) on standard benchmarks, then analyze the relationship between prompt complexity and quantization sensitivity.

**Status:** UNBLOCKED as of Sept 5 (SLURM venv issues resolved). NOT YET SUBMITTED.

**Expected runtime:** 4–6 hours.

**Gate (NEW.md Phase 2):** Accuracy per tier measured; correlation analysis shows whether complexity features predict quantization sensitivity.

---

## What This Measures

1. **Accuracy per tier** on three tasks (via lm-eval):
   - **tinyMLU** (subset: ~1000 multiple-choice questions)
   - **GSM8K** (subset: math word problems)
   - **HellaSwag** (subset: commonsense reasoning)

2. **Per-prompt accuracy**: Store accuracy for each individual prompt × tier combination (not just aggregate averages) for later correlation analysis.

---

## Prerequisites

- [ ] Phase 1 (Session 1) completed successfully ✓
- [ ] On-campus with GPU access
- [ ] HF_TOKEN set: `echo $HF_TOKEN`
- [ ] Python 3.11+ on cluster
- [ ] lm-eval installed: `uv pip install lm-eval`

---

## The Sept 5 Fix (Why This Now Works)

**Problem:** Earlier submission attempts failed because:
- ❌ `srun` in sbatch isolated the venv, breaking Python discovery
- ❌ Bare `pip` command fell through to system Python 3.9 (no modules)

**Solution (verified working):**
- ✓ Use `sbatch` directly without `srun` — allocates GPU, venv works
- ✓ Install with `uv pip`, not bare `pip`
- ✓ Direct `python` command in the venv

This is already baked into the sbatch script below.

---

## Submission

```bash
cd major-project/training

chmod +x session2_submit.sbatch

sbatch session2_submit.sbatch
```

Expected output:
```
Submitted batch job <JOB_ID>
```

---

## Monitoring

Watch the log in real time:
```bash
tail -f session2_<JOB_ID>.log
```

Expected progress (should be steady):
```
[1/3] Setting up Python environment...
[2/3] Installing dependencies...
[3/3] Configuring HuggingFace...

Starting accuracy evaluation (3 tiers × 3 tasks, estimated 4-6h)...

=== Tier 4bit, Task tinyMLU ===
...progress...
=== Tier 8bit, Task tinyMLU ===
...progress...
=== Tier 16bit, Task tinyMLU ===
...
(repeat for GSM8K, HellaSwag)
```

If the job stalls:
- Check GPU utilization: `nvidia-smi` (should show non-zero VRAM in use)
- Check if lm-eval is downloading task datasets (first run of each task may be slow)
- Kill and resubmit if stuck >10 minutes on a single task

---

## Output Files

When complete, the `results/` directory contains:

| File | Purpose |
|------|---------|
| `accuracy_summary.csv` | Per-tier, per-task: mean accuracy, std, n samples |
| `accuracy_per_prompt.json` | Detailed: every prompt × tier × task combination and whether it was correct |
| `eval_run_info.json` | Metadata: lm-eval version, tasks run, seeds |

The `accuracy_summary.csv` becomes Table 2 in the paper.

---

## If Something Goes Wrong

### Error: `HF_TOKEN not set`
```bash
export HF_TOKEN=<your-token>
sbatch session2_submit.sbatch
```

### Error: `lm_eval` import fails
```bash
# Reinstall lm_eval
source venv/bin/activate
uv pip install --force-reinstall lm-eval
```

### Error: lm-eval datasets not downloading
lm-eval downloads task datasets from HuggingFace on first use. If the cluster doesn't have outbound internet from compute nodes:
- Pre-download: run evaluation on a login node before submitting (slower but works)
- Or: contact cluster admin about compute-node internet access

### Job times out (> 8 hours)
Increase the SLURM limit in the sbatch script:
```bash
# In session2_submit.sbatch, change:
#SBATCH --time=08:00:00
# to:
#SBATCH --time=10:00:00
```

### Validation shows low accuracy (< 0.1)
- Check that the model is loading correctly (log should show model name)
- Verify lm-eval version matches what `kaggle_accuracy_eval.py` expects
- Check if prompts are being preprocessed correctly (whitespace, special tokens)

---

## After Success

Once the job completes with ✓✓ SESSION 2 COMPLETE:

1. **Check the gate (NEW.md Phase 2):**
   - Does `accuracy_per_prompt.json` show that accuracy drops from fp16 → 4-bit?
   - Does the correlation study show features correlating with quantization sensitivity (Spearman ρ > 0.2)?
   - If yes → proceed to Phase 5 (Session 4, routing experiment)
   - If no → rethink the sensor design before Phase 5

2. **Commit results:**
   ```bash
   git add major-project/training/results/
   git commit -m "Session 2 complete: per-tier accuracy on tinyMLU, GSM8K, HellaSwag"
   ```

3. **Proceed to Phase 5:**
   Session 4 (routing experiment) will be submitted next, gated by Phase 2 results.

---

## Key Design Decisions

- **Stratified eval set:** 500 prompts (200 easy / 150 medium / 150 hard) from TriviaQA + Alpaca + GSM8K + CodeAlpaca
- **Greedy decoding:** Deterministic, reproducible
- **Fixed seed:** 42 across all runs
- **Per-prompt granularity:** Stored to enable later correlation study with complexity features

---

## Questions?

See:
- **NEW.md** Phase 2 — full context
- **CREDIBILITY_REPORT.md** § 3–4 — measurement methodology
- **major-project/CLAUDE.md** — project structure
