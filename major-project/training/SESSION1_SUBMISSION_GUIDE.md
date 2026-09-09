# Session 1 Submission Guide — Energy Ground Truth

**Goal:** Measure real J/token for 4-bit, 8-bit, 16-bit Llama-3.2-1B on the college GPU cluster.

**Status:** Ready to submit. Expected runtime: 2–3 hours. ✓ Dry run passed Sept 4.

---

## Prerequisites (Check These First)

- [ ] On-campus (cluster is on-campus only, no VPN)
- [ ] Have access to the SPIT cluster or equivalent RTX 6000 Ada GPU
- [ ] HF_TOKEN set in shell: `echo $HF_TOKEN` should return your token
  - If not set: `export HF_TOKEN=<your-huggingface-api-token>`
  - Token format: `hf_...` (generate at https://huggingface.co/settings/tokens)
- [ ] Llama-3.2-1B license accepted on HuggingFace
- [ ] Python 3.11+ available in cluster environment

---

## Submission

```bash
# Navigate to training directory
cd major-project/training

# Make script executable
chmod +x session1_submit.sbatch

# Submit the job
sbatch session1_submit.sbatch

# Should print: Submitted batch job <JOB_ID>
```

---

## Monitoring During the Run

The job runs in three phases:

### Phase 1: Setup (1–2 minutes)
```
[1/4] Setting up Python environment...
[2/4] Installing dependencies...
[3/4] Configuring HuggingFace...
[4/4] Checking NVML energy counter support...
```
✓ If all pass, the run is healthy.

### Phase 2: Benchmark (2–3 hours)
```
Starting benchmark (3 runs × 3 tiers × ~167 prompts each)...
```

**IMPORTANT MONITORING PERIOD: First 30–60 minutes**

Watch the log for the first 30 minutes:
```bash
# In another terminal on the cluster
tail -f session1_<JOB_ID>.log
```

Expected output (should see this quickly):
```
=== run 1 | tier 4bit ===
=== run 1 | tier 8bit ===
=== run 1 | tier 16bit ===
```

**HANG RISK (from original Kaggle failure):**
- If you see logs go silent during the 8-bit tier (after ~1–1.5h into run 1), this is the known risk
- Silent hang appears as: consistent log output, then a burst of repeated `MatMul8bitLt` warnings, then silence
- If this happens, kill the job: `scancel <JOB_ID>`
- The issue may have been fixed in the current code, but monitor to be safe

✓ If the 4-bit tier completes and moves to 8-bit without hanging, you're clear.

### Phase 3: Validation (2–5 minutes)
```
Validating...
✓✓✓ SESSION 1 SUCCESS ✓✓✓
Results in: major-project/training/results/
```

---

## Output Files

When complete, the `results/` directory contains:

| File | Purpose |
|------|---------|
| `energy_per_inference.csv` | One row per inference: tier, prompt_id, tokens_out, energy_j, j_per_token, latency_s |
| `energy_summary.csv` | Summary per tier: mean_j_per_token, std, ci95 (this is Table 1 data) |
| `hardware_info.json` | GPU model, driver, idle power baseline, timestamp |

The `energy_summary.csv` is what goes into the paper.

---

## If Something Goes Wrong

### Error: `HF_TOKEN not set`
```bash
export HF_TOKEN=<your-token>
sbatch session1_submit.sbatch
```

### Error: NVML check fails
```
FATAL: nvmlDeviceGetTotalEnergyConsumption not supported
```
This GPU doesn't have the NVML energy counter (Volta+ required, Pascal/older fails). Report this to the cluster admin — you may need a different GPU or a fallback platform.

### Job hangs silently
- Watch log in first hour; if silent during 8-bit tier, kill it
- Check if the issue is already fixed in `kaggle_energy_benchmark.py` (it was hardened Aug 22)
- If it persists, debug by running a single small-prompt test first:

```bash
cd major-project/backend/src/green_weight
python3 -c "
from training.scripts.kaggle_energy_benchmark import GpuEnergyMeter
meter = GpuEnergyMeter()
info = meter.info()
print(info)
"
```

### Job times out (> 4 hours)
The 4-hour SLURM limit is conservative. If you're seeing steady progress, you can increase it in the sbatch script:
```bash
# In session1_submit.sbatch, change:
#SBATCH --time=04:00:00
# to:
#SBATCH --time=05:00:00
```

### Validation fails
Read the validation output. Common issues:
- Empty CSV (no measurements recorded) → check if GPU was actually used
- J/token outside 0.05–50 range → likely a meter bug or wrong energy units
- Low n per tier → rerun; need ≥300 prompts per tier minimum

---

## After Success

Once the job completes with ✓✓✓ SUCCESS:

1. **Check the go/no-go gate (NEW.md Phase 1):**
   - If energy savings ≥15% over fp16 → proceed to Phase 2
   - If marginal → pivot to GPTQ/AWQ or reframe as latency/cost

2. **Commit results:**
   ```bash
   git add major-project/training/results/
   git add major-project/paper/results.md  # auto-append if results script does it
   git commit -m "Session 1 complete: energy ground truth on RTX 6000 Ada"
   ```

3. **Proceed to Session 2:**
   See `SESSION2_SUBMISSION_GUIDE.md`
