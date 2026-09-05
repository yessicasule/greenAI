# Session 4 Plan: Main Routing Experiment
**Status:** PLANNING (awaiting Session 2 completion or fallback decision)  
**Estimated GPU time:** 12–18 hours (4–6h × 3 runs on different days)  
**Expected output:** `routing_per_prompt.csv`, `routing_conditions_summary.csv`, paper's main Pareto plot  
**Research Questions answered:** RQ2 (does our router outperform static tiers?), RQ3 (does routing intelligence matter?)

---

## Pre-Session 4 Requirements

### Session 2 Status (Blocker)
**Current:** Session 2 is STOPPED due to SLURM environment issues on college cluster. See `SESSION_2_BLOCKER.md`.

**Critical decision:** Session 4 **does not hard-depend on Session 2** completing first:
- **Session 2 provides:** Per-tier accuracy numbers (needed for the Pareto plot's Y-axis in Phase 6)
- **Session 4 provides:** Per-tier energy numbers + routing-decision logs (needed for the Pareto plot's X-axis + tier distribution)

**Options:**
1. **Wait for Session 2 fix** (best for completeness) — unblock it using one of the 4 paths in `SESSION_2_BLOCKER.md`, then run Session 4
2. **Run Session 4 now, do Session 2 later** (maintains velocity) — gather energy data + routing logs now, fill in accuracy numbers once Session 2 works
3. **Use estimated/prior accuracy numbers temporarily** — if Session 2 remains blocked, use expected Llama-3.2-1B accuracy from public benchmarks as a placeholder during Phase 6, mark as `ESTIMATED`, then replace with real numbers when Session 2 completes

**Recommendation:** **Option 2** — Session 4 and Session 2 measure orthogonal things. Run Session 4 now to collect the energy/routing data; Session 2 can happen in parallel or shortly after, and Phase 6 (figures) combines them. This unblocks momentum.

### Code Readiness
✅ **All code is ready:**
- `training/scripts/kaggle_routing_experiment.py` — the main script (exists, fixed 2026-08-22)
- `router/` modules — complexity scorer, fuzzy controller, RouteLLM bridge (all imported by routing script)
- `backend/src/green_weight/benchmark/accuracy_eval.py` — fixed 2026-08-22 to call real lm_eval
- `training/scripts/verify_results.py` — validation script (ready)
- Adapters (`adapters/adapter_{simple,medium,complex}`) — verified loadable 2026-08-22

### Data Readiness
✅ **Eval set confirmed:**
- 500-prompt stratified set (easy 200 / medium 150 / hard 150) exists in `data/eval_prompts.jsonl`
- Fuzzy controller recalibrated 2026-08-22 against the full set (entropy/syntax_depth/token_length bounds fixed)

### Hardware Readiness (College cluster)
⚠️ **Session 1 (energy) succeeded on college RTX 6000 Ada** (proof that cluster access works)  
⚠️ **Session 2 failed due to srun environment isolation** (not a hardware issue)  
✅ **Session 4 should work on the same cluster** — the routing experiment script doesn't have the venv+srun dependency issues that Session 2 hit, since `kaggle_routing_experiment.py` is self-contained (no lm-eval wrapper, just model loading + inference)

---

## Session 4 Tasks

### 4.1 Dry-Run Sanity Check (30–60 min, same cluster)

**Before committing to full runs, validate on a 10–50 prompt subset:**

```bash
# On college cluster, submit:
python training/scripts/kaggle_routing_experiment.py \
  --dry-run \
  --limit 10 \
  --output-dir dryrun_session4
```

**Expected output:**
- `routing_per_prompt_DRY.csv` with columns:
  - `prompt_id`, `prompt_text`
  - `flesch_kincaid`, `token_length`, `entropy`, `syntax_depth`, `has_code_or_math` (complexity features)
  - For each of 8 conditions: `tier_chosen`, `energy_joules`, `tokens_generated`, `response`, `correctness`
  - `execution_time_seconds`

**Validations:**
1. **No crashes** — all 10 prompts complete without OOM or subprocess errors
2. **Energy values sensible** — 16-bit ~8–10 J/token, 8-bit ~3–4, 4-bit ~1–2 (rough ballpark)
3. **Tier distribution reasonable** — fuzzy router spreads prompts across tiers; random router does too (just differently)
4. **Correctness column populated** — not all NaN
5. **router() overhead in milliseconds** — check the separate `router_compute_time` if logged; should be <100ms

**If dry-run fails:**
- **OOM:** Same memory-cleanup fixes as Session 2 apply; but Session 4 loads models once (not per-task), so memory footprint is lower. If it still OOMs, reduce `--limit` further or request more GPU memory from admins.
- **srun PATH issues:** Session 4 doesn't use lm-eval's HFLM wrapper (only greedy generation), so it *should* work where Session 2 failed. If it doesn't, apply the same solution as Session 2.
- **Missing tier:** If 4-bit or 8-bit fail to load while 16-bit succeeds, it's a bitsandbytes config issue — check `load_tier()` in the script.

**If dry-run succeeds:** Proceed to 4.2.

---

### 4.1a The 8-bit "hang" — root-caused 2026-09-05

The dry run appearing to hang on the 8-bit tier, flooding the log with
`MatMul8bitLt: inputs will be cast from torch.float32 to float16`, was
**not** a hang and not a bitsandbytes bug. It is the same failure that
stalled the Kaggle Session 1 run for ~4.7h (recorded in `CLUSTER_MANUAL.md`
as unexplained).

**Cause:** `load_tier()` set `torch_dtype=torch.float16` only on the 16-bit
branch. For the 8-bit tier the modules bitsandbytes does *not* quantize
(embeddings, layernorms, `lm_head`) therefore stayed fp32, so every
`Linear8bitLt` forward pass had to cast its fp32 activations down to fp16 —
and warn about it. At 128 new tokens × 16 layers × ~7 projections that is
~140k warnings per 10 prompts. The job was bottlenecked on formatting
warning strings and flushing them to the SLURM log, not on GPU work. The
4-bit tier was unaffected only because `bnb_4bit_compute_dtype` already
pinned its compute path.

**Fixes applied to `kaggle_routing_experiment.py`:**
1. `torch_dtype=torch.float16` is now set for **every** tier (the real fix —
   removes the cast, the warning, and the slowdown together).
2. A `warnings.filterwarnings` guard on the same message, because
   transformers resets filters to `"always"` in places and defeats Python's
   normal once-per-site dedup.
3. Progress now prints every prompt on runs of ≤100 prompts (it was every
   50, so a 10-prompt dry run printed *nothing* between tiers — which is
   why slow was indistinguishable from hung), with elapsed/ETA, and the
   sbatch wrappers use `python -u`.
4. New flags: `--limit`, `--output-dir`, `--warmup`, `--tiers`. `--tiers`
   lets a dry run skip 8-bit entirely; Phase B clamps any router pick of an
   unmeasured tier onto the measured set so it cannot `KeyError` *after*
   the GPU work is spent (dry runs only — such numbers are not comparable
   to a full run).

**Submission wrappers:** `training/scripts/session4_dryrun.sh` and
`training/scripts/session4_full_run.sh`. The dry run is capped at
`--time=00:40:00` on purpose — with this fix it should finish in under 10
minutes, so a job still alive at 40 min is wedged and SLURM should reap it
rather than repeat the 4.7h silent burn.

---

### 4.2 Full Run #1 (4–6 hours, same cluster)

Once dry-run passes, submit the full 500-prompt evaluation on the cluster:

```bash
# Submit a 6-hour job (RTX 6000 Ada, ~200 prompts/hour estimated)
sbatch --time=6:00:00 --gpus=1 \
  -o training/logs/session4_run1.log \
  <<'EOF'
#!/bin/bash
source ~/greenweight_env.sh
cd ~/greenAI/major-project/backend/src/green_weight
python ../../../training/scripts/kaggle_routing_experiment.py \
  --output-dir /home/khushiwadhwa23/session4_run1_output
EOF
```

**Expected outputs:**
- `routing_per_prompt.csv` — full evaluation log, all 500 prompts × 8 conditions
- `routing_conditions_summary.csv` — aggregates per condition: mean energy, std dev, accuracy (once Session 2 provides it), tier distribution
- `energy_per_inference.csv` — per-tier, per-seed energy summary (for Phase 6 bootstrap CIs)
- `hardware_info.json` — GPU model, driver, timestamp (for reproducibility)

**Monitor for:**
- Errors in logs — save them to `training/logs/session4_run1_errors.txt`
- Expected duration: 4–6h on RTX 6000 Ada for 500 prompts (if faster, great; if slower, budget another hour)
- GPU memory trajectory — should stabilize after model load, not climb steadily

**On completion:**
- Download outputs to local machine
- Commit CSVs to repo with a "Session 4 Run 1" commit message
- **Do not** submit Run 2 immediately; wait at least 8–24 hours to ensure stability, then check logs for any warnings

---

### 4.3 Full Run #2 (4–6 hours, +24h later)

Submit identical job again, 1–2 days after Run 1:

```bash
# Same command as 4.2, but with run2 directory
```

**Purpose:** Validate reproducibility + collect a 2nd independent energy measurement. This is where you build confidence that your energy numbers aren't flukes.

**Expected:** Similar energy values to Run 1 (within 5–10% is good, >15% diff is a red flag for GPU variability or background processes).

---

### 4.4 Full Run #3 (4–6 hours, +48h after Run 1)

Submit once more, on a 3rd day:

```bash
# Same command as 4.2, but with run3 directory
```

**Gate requirement (from NEW.md):** ≥3 repeated runs on different days to build the 95% bootstrap CIs in Phase 6.

---

### 4.5 Validation & Commit (30–60 min, CPU, local)

Once all 3 runs complete:

```bash
cd training/scripts
python verify_results.py \
  --run1 /path/to/run1_output \
  --run2 /path/to/run2_output \
  --run3 /path/to/run3_output \
  --output-file ../logs/session4_validation.md
```

**Expected output:** `session4_validation.md` with:
- ✅ PASS rows for each condition (energy mean, std dev)
- ⚠️ WARN rows if energy variance is high or tier distribution is suspicious
- ❌ FAIL rows if any run has missing data, NaN energy, or correctness > 50% errors

**If PASS:** Append summary to `paper/results.md` with a new row:
```
| Session 4 (Routing Experiment) | 3 runs × 500 prompts | Energy per tier, routing logs | PASS | 2026-09-XX |
```

**If WARN:** Document in `paper/results.md` + `CREDIBILITY_REPORT.md` what the warning is and whether it's a limitation or a real finding.

**If FAIL:** Investigate failed run; do not proceed to Phase 6 until resolved.

---

## The 8 Conditions (What Session 4 Actually Measures)

From `RESEARCH_PLAN.md` and verified in `kaggle_routing_experiment.py` (fixed 2026-08-22):

| # | Name | Description | What it proves |
|---|------|-------------|---|
| 1 | `static_16bit` | Always use fp16 (full precision) | Baseline energy/accuracy reference |
| 2 | `static_8bit` | Always use 8-bit quantization | Baseline energy/accuracy for lower tier |
| 3 | `static_4bit` | Always use 4-bit quantization | Baseline energy/accuracy for lowest tier |
| 4 | `fuzzy_router` | Fuzzy controller routing (ours) | Main contribution: intelligent routing |
| 5 | `random_router` | Random tier selection, same distribution as #4 | Ablation: routing intelligence matters, not just tier mix |
| 6 | `threshold_router` | Simple mean of 5 features + 33/66 breakpoints (naive) | Ablation: fuzzy membership functions beat threshold logic |
| 7 | `oracle_router` | Always pick cheapest tier that answers correctly | Upper bound (theoretical best; requires perfect foresight) |
| 8 | `cascade_router` *(optional)* | FrugalGPT-style cascade (try 16-bit first, drop if slow) | Stretch goal; only if time permits |

**Key ablation insight:** Conditions 4, 5, 6 form a hierarchy:
- Condition 4 > 5 → routing *algorithm* matters (our fuzzy is better than random with same mix)
- Condition 4 > 6 → fuzzy membership *representation* matters (our fuzzy is better than threshold)
- Condition 4 < 7 → we're not oracle-optimal (expected; oracle is unfeasible)

---

## Key Metrics Collected

For **each prompt** × **each condition**:
1. **Energy** (`energy_joules`) — NVML counter (same as Session 1)
2. **Tokens generated** (`tokens_gen`) — length of model's response
3. **Latency** (`latency_ms`) — wall-clock time for inference
4. **Correctness** (`correctness`) — boolean (wrong until Session 2 accuracy eval merges it in)
5. **Tier chosen** (`tier`) — which of {4bit, 8bit, 16bit} was used
6. **Router decision time** (`router_ms`) — how long the complexity scorer + fuzzy controller took

For **each condition** × **each run** (aggregated):
1. **Mean energy** (J/token and J/prompt)
2. **Std dev** (for bootstrap CIs in Phase 6)
3. **Tier distribution** (% of prompts routed to each tier)
4. **Mean accuracy** (once Session 2 provides it)

---

## Expected Results (Sanity Check)

**Rough expectations, not commitments:**
- **Fuzzy router** should fall *between* static tiers on the energy axis, closer to the low-energy side (most prompts routed to 4bit/8bit)
- **Fuzzy router accuracy** should *exceed* static-4bit (the "obvious" cheap option), proving QAT adapters help
- **Random router** should have *similar* energy to fuzzy (same tier distribution by design) but *worse* accuracy (tier mismatches)
- **Threshold router** should be close to fuzzy but slightly worse (naive control)
- **Oracle** should have the best Pareto point (unachievable baseline)

If results invert (e.g., fuzzy is higher energy than static tiers, or accuracy is terrible), that's still a valid finding — it means:
1. The routing logic needs rethinking before publication, **or**
2. The energy model (bit-width → energy) is different from assumptions, **or**
3. The QAT adapters don't actually help much

All of these are publishable findings, just different papers (a negative result is still a contribution).

---

## Failure Modes & Mitigations

| Failure Mode | Signal | Mitigation |
|---|---|---|
| **OOM during model load** | `RuntimeError: CUDA out of memory` | Reduce batch size in lm_eval config, or request larger GPU |
| **srun PATH isolation** (like Session 2) | `-sh: python: command not found` | Use one of 4 solutions from SESSION_2_BLOCKER.md (likely same fix) |
| **Random seed inconsistency** | Energy/accuracy varies >15% between runs | Check that all 3 runs use the same `--seed`, same eval set, same model weights |
| **Tier not loading** | `ValueError: Unable to infer dtype from ...` in bitsandbytes config | Verify `load_tier()` function in `kaggle_routing_experiment.py` — may need torch dtype fixes |
| **Router crashes on edge prompts** | `ValueError: NaN in complexity score` | Check for inf/NaN in feature calculation; may need epsilon guards in `complexity_scorer.py` |
| **All prompts routed to same tier** | Fuzzy controller output is non-responsive | Check that breakpoints were recalibrated (done 2026-08-22) and ENTROPY_RANGE is widened (done) |
| **Correctness always False/True** | Evaluation shows 0% or 100% accuracy | Session 2 hasn't run yet; this is expected. Fill in real accuracy once Session 2 completes. |

---

## Files to Monitor

| File | Role | Status |
|------|------|--------|
| `training/scripts/kaggle_routing_experiment.py` | Main script | Ready (fixed 2026-08-22) |
| `backend/src/green_weight/router/complexity_scorer.py` | Feature extraction | Ready (recalibrated 2026-08-22) |
| `backend/src/green_weight/router/fuzzy_controller.py` | Tier decisions | Ready |
| `backend/src/green_weight/models/model_pool.py` | Model loading | Ready |
| `config.yaml` | Breakpoints, thresholds | Ready (updated 2026-08-22) |
| `adapters/adapter_{simple,medium,complex}` | LoRA adapters | Ready (verified 2026-08-22) |
| `training/logs/session4_run*.log` | Execution logs | To be generated |
| `paper/results.md` | Results registry | To be updated after validation |

---

## Expected Outputs (Commit These)

After all 3 runs validate:

1. **`training/logs/session4_run1_output/routing_per_prompt.csv`** — the full evaluation log (500 rows, 8 condition columns)
2. **`training/logs/session4_run1_output/hardware_info.json`** — GPU model, driver, timestamp
3. **`training/logs/session4_validation.md`** — `verify_results.py` output (shows all ✅ PASS checks)
4. **Updated `paper/results.md`** — new row documenting Session 4 completion + summary stats

Push all of the above to GitHub once validation passes.

---

## Timeline

```
Day 1 (T+0):
  - Dry-run (30-60 min) ← BLOCKER: must pass before proceeding
  - If pass: Submit Run #1 (4-6h job submitted, runs overnight)

Day 2 (T+24h):
  - Run #1 completes; download + spot-check logs
  - Submit Run #2 (4-6h job)

Day 3 (T+48h):
  - Run #2 completes; download + spot-check logs
  - Submit Run #3 (4-6h job)

Day 4 (T+72h):
  - Run #3 completes; download
  - Local validation: `verify_results.py` (30 min, CPU)
  - If PASS: commit + push to GitHub

Total wall-clock: ~4 days
Total GPU hours: 12–18 hours (billable if using AceCloud; free on Kaggle/college cluster)
Total user time: ~2–3 hours (monitoring logs, 30 min validation)
```

---

## Decision Gate: Proceed to Phase 6?

**PASS criteria (all must be true):**
- ✅ All 3 runs complete without crashes
- ✅ `verify_results.py` shows ✅ PASS (no ❌ FAIL rows)
- ✅ Energy values are sensible (16-bit > 8-bit > 4-bit, all in J/token range)
- ✅ Tier distributions are reasonable (not all prompts routed to same tier)
- ✅ `routing_per_prompt.csv` has all 500 rows with no NaN energy

**If PASS:** Commit results, proceed directly to Phase 6 (figures & stats). Phase 2 (accuracy) can continue in parallel.

**If FAIL or WARN:** Investigate; do not proceed until root cause identified + fixed.

---

## Notes for the Next Session

1. **Session 2 is not a hard blocker** — you can run Session 4 while Session 2 is being debugged. The two measure different things (accuracy vs. energy).

2. **Dry-run is non-negotiable** — do not skip it. The 30–60 min investment saves 4+ hours of debugging a crashing full run.

3. **3 runs are required** — this is your reproducibility gate. Even if Run 1 looks perfect, do Runs 2 & 3 on different days (GPU load varies, background processes vary).

4. **Do not modify the script mid-session** — if you need to fix a bug, freeze the current run, fix the bug, re-validate with dry-run, *then* restart from Run 1. Do not mix runs from different code versions.

5. **Energy numbers are ground truth** — they come directly from NVML (same as Session 1), so trust them. If they're weird, the hardware is the issue, not the code (in 99% of cases).

6. **Correctness is placeholder until Session 2 completes** — right now, all conditions will show similar correctness (because it's not measured per-prompt yet). Once Session 2 accuracy runs, you'll merge that data back in. This is expected and fine.

---

## Contact & References

- **College cluster contact:** gpu@spit.ac.in
- **Session 1 (energy reference):** Completed successfully; Session 4 uses same NVML tracking
- **Session 2 blocker:** See `SESSION_2_BLOCKER.md` for status + recovery options
- **Research plan:** `RESEARCH_PLAN.md` §4 (RQ2/RQ3/RQ4 definitions)
- **Script entry point:** `training/scripts/kaggle_routing_experiment.py`
- **Verification:** `training/scripts/verify_results.py`

---

**Last Updated:** 2026-09-05  
**Documented by:** Claude Code session  
**For:** Session 4 execution and Phase 5 gate decisions  
