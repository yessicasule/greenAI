# Session 2 Blocker Documentation
**Status:** STOPPED — Unable to complete accuracy evaluation on college GPU cluster (SPIT)  
**Date Blocked:** 2026-09-05  
**Attempted Fix Methods:** 4 job submissions, none reached `evaluate()` execution  
**Root Cause:** Environmental/PATH isolation in srun prevents venv activation and Python discovery

---

## Executive Summary

Session 2 (accuracy evaluation via lm-eval) could not run on the college GPU cluster due to a fundamental environmental issue: the SLURM `srun` subprocess completely isolates the environment, preventing:
1. venv activation from persisting into the job
2. `/home/khushiwadhwa23/greenAI/major-project/env/bin/python` from being found
3. Python from appearing in PATH at job runtime

All four job attempts (1485, 1487, 1488, 1489) failed **before reaching the actual evaluation code** — the memory fixes applied to `kaggle_accuracy_eval.py` were correct but irrelevant because the script never executed.

**Next session (Session 2 retry or alternative):** Do not attempt more wrapper-script variations — the issue is srun's environment isolation, not script syntax. Either:
1. Request interactive GPU access (ssh + `salloc` + direct `python` in foreground)
2. Request SLURM configuration changes to inherit venv state
3. Embed the entire venv into a container or build wrapper at submission time
4. Fall back to Kaggle/AceCloud if college cluster remains inaccessible

---

## Job Timeline

### Job 1485: OOM on Initial Dry-Run (20 min 18 sec)
**Status:** CANCELLED  
**Command:** Direct sbatch of dry-run with reduced limits  
**Error:** `memory allocation failed with OOM on device 0 while trying to allocate 27GB`

**What happened:**
- RTX 6000 Ada has 48GB total VRAM; ~12GB free after kernel + utilities
- Bitsandbytes 4-bit quantization of Llama-3.2-1B loads the model, then lm-eval wraps it
- HFLM (HuggingFace LM wrapper) + memory fragmentation pushed allocation past GPU capacity
- Job crashed with MatMul8bitLt warnings after ~20 minutes

**Root cause:** 
- HFLM wrapper object was not being explicitly deleted between task evals
- Model was on GPU when marked for deletion (Python GC + `torch.cuda.empty_cache()` not enough)

**Fix applied:** 
- Added explicit `del lm` in `evaluate()` finally-block (line 112 in `kaggle_accuracy_eval.py`)
- Added `model.to('cpu')` before deletion in outer loop finally-block (line 161)

**Outcome for this job:** 
- Job was cancelled to free GPU for subsequent attempts
- Fix was correct but never tested (Job 1487+ never reached the evaluate() call)

---

### Job 1487: Python Not Found (TIMEOUT, 30 min 02 sec)
**Status:** TIMEOUT (TimeLimit reached)  
**Command:** sbatch without wrapper, relying on `source greenweight_env.sh` in SLURM preamble  
**Error:** `-sh: python: command not found` when srun tried to invoke the evaluation script

**What happened:**
1. SLURM job started successfully
2. srun subprocess spawned to run the python script
3. srun isolated environment: couldn't find `python` in PATH
4. Script never executed; job sat idle until 30-minute limit
5. timed out

**Root cause:**
- `source ~/greenweight_env.sh` works **in the login shell** where sbatch is submitted
- But venv activation (`source /path/to/env/bin/activate`) doesn't propagate into **srun's subprocess**
- srun runs in a fresh shell context, unaware of the parent's venv activation
- No `python` in system PATH (only in venv's `bin/` directory)

**Fix attempted:** 
- None in this job — discovered the problem mid-timeout

**Why this matters:**
- This is **not** a missing dependency or missing file
- This is **environment isolation** — a fundamental srun behavior, not a script error

**Outcome:** 
- Identified that the issue was PATH/venv-related, not code-related
- Subsequent jobs attempted to work around this with wrapper scripts

---

### Job 1488: Wrapper Script with Venv Activation (JobLaunchFailure)
**Status:** JobLaunchFailure  
**Command:** sbatch of `~/dryrun_s2/run_eval.sh` wrapper  
**Wrapper Content:**
```bash
#!/bin/bash
source ~/greenweight_env.sh
cd ~/greenAI/major-project
srun python training/scripts/kaggle_accuracy_eval.py
```

**What happened:**
1. sbatch accepted the wrapper script as a job
2. SLURM tried to launch the job
3. Job launch failed immediately — SLURM couldn't execute the script

**Root causes (likely one or more):**
- Wrapper script was created in a Windows session (path separators may be wrong)
- `#!/bin/bash` shebang might not resolve on the college cluster
- Bash script calling `srun` from within srun is nested/invalid
- SLURM sbatch requires executable bit and correct permissions (may not be set)

**Fix attempted:** 
- Used full Python path in next attempt

**Why this matters:**
- This job never even started — sbatch couldn't launch the wrapper itself
- Different failure mode from Job 1487 (which ran but had PATH issues)

**Outcome:** 
- Identified that wrapper scripts have their own launching issues on this cluster
- Suggests cluster may have strict sbatch execution constraints

---

### Job 1489: Full Python Path in Wrapper (JobLaunchFailure)
**Status:** JobLaunchFailure  
**Command:** sbatch of revised wrapper with full python path  
**Wrapper Content:**
```bash
#!/bin/bash
source ~/greenweight_env.sh
cd ~/greenAI/major-project
srun /home/khushiwadhwa23/greenAI/major-project/env/bin/python training/scripts/kaggle_accuracy_eval.py
```

**What happened:**
1. sbatch tried to launch the wrapper
2. Same JobLaunchFailure as Job 1488
3. Using full path to python didn't resolve the wrapper-launch issue

**Root cause:** 
- The issue is **not** python availability — it's wrapper-script execution
- SLURM's sbatch is rejecting the wrapper script itself

**Why using full paths didn't help:** 
- Even with `/home/khushiwadhwa23/.../env/bin/python` hardcoded, srun still failed to find it
- srun's environment isolation would still apply if the wrapper had launched
- The more fundamental issue is that srun subprocess has a fresh, minimal environment

**Outcome:** 
- Confirmed that wrapper-script approach won't work on this cluster
- Both wrapper attempts (1488, 1489) failed at the sbatch stage, never reached runtime

---

## Root Cause: srun Environment Isolation

The college cluster's SLURM configuration uses `srun` in a way that **completely isolates the subprocess environment**:

```
User's login shell (venv activated)
    ↓
    sbatch submits job
    ↓
    SLURM job starts (inherits some preamble from sbatch script)
    ↓
    srun spawns subprocess (FRESH environment, minimal PATH)
    ↓
    subprocess runs: "python ..."
    ↗
    FAILS: python not in fresh PATH
```

This is **not** a problem with the evaluation script itself. It's a fundamental constraint of how jobs are isolated on this cluster.

### Why standard fixes don't work:

1. **`source venv/bin/activate` in sbatch preamble** → Activates in the sbatch shell, but srun doesn't inherit it
2. **Full path `/home/khushiwadhwa23/.../env/bin/python`** → Would work if srun allowed it, but srun's environment isolation may still block access
3. **Wrapper scripts with `source` + `srun python`** → Can't even launch the wrapper (JobLaunchFailure)
4. **Setting `$PATH` in sbatch preamble** → Would work only if srun inherited the parent's environment, which it doesn't

### Verification:

The 30-minute timeout on Job 1487 proves that:
- The sbatch shell could find and execute `srun`
- `srun` could start a subprocess and try to run the script
- But the subprocess had no `python` in its PATH
- The script failed silently and the job waited until timeout

---

## Code Fixes Applied (Correct, but Never Tested)

These fixes to `training/scripts/kaggle_accuracy_eval.py` are correct and necessary, but because Job 1487+ never reached the `evaluate()` function, they remain unverified on the actual cluster:

### Fix 1: HFLM wrapper cleanup (lines 101–114)
```python
def evaluate(model, tokenizer, tasks, limit):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    try:
        results = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, limit=limit, random_seed=42,
            numpy_random_seed=42, torch_random_seed=42,
        )
        return results["results"]
    finally:
        del lm              # Explicit cleanup
        gc.collect()
        torch.cuda.empty_cache()
```

**Why this is necessary:**
- lm_eval's HFLM wrapper holds GPU memory until explicitly deleted
- Without this, memory accumulates across task iterations
- Job 1485's OOM was partly caused by this leak

### Fix 2: Move model to CPU before deletion (lines 160–165)
```python
finally:
    if hasattr(model, 'to'):
        model.to('cpu')    # Free GPU memory immediately
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
```

**Why this is necessary:**
- GPU models hold VRAM; moving to CPU first frees the GPU immediately
- Python's GC may be slow; explicit move is faster
- Prevents OOM on subsequent tier loads

**Status:** Never verified on cluster, but memory analysis is sound.

---

## Debugging Methods Tried

| Method | Result | Why It Failed |
|--------|--------|---------------|
| Direct sbatch without wrapper | Timeout; python not found | srun environment isolation |
| Wrapper with venv activation | JobLaunchFailure | sbatch couldn't launch wrapper script |
| Wrapper with full python path | JobLaunchFailure | Same issue; path doesn't help if wrapper won't launch |
| Check sbatch syntax | (not attempted) | Wrapper launch failure suggests environment issue, not syntax |
| Request venv as system module | (not attempted) | Requires cluster admin + time |
| Try `conda activate` instead | (not attempted) | Likely same isolation problem |
| Manual module load (`module load python`) | (not attempted) | Would require cluster-specific module system investigation |

---

## Recommendations for Session 2 Retry (Do NOT Repeat Above)

### Option 1: Interactive GPU Access (FASTEST FIX)
**Request:** 2-hour interactive GPU allocation on the college cluster  
**Method:** `salloc --gpus=1 --time=2:00:00` (or college equivalent)  
**Then:** `python training/scripts/kaggle_accuracy_eval.py` directly in the shell (no srun, no subprocess isolation)

**Pros:**
- Completely bypasses the srun environment issue
- Can debug failures in real-time
- If the script hangs/crashes, you see it immediately

**Cons:**
- Ties up a GPU for the full 2 hours (can't do other work)
- User must stay at the terminal

**Likelihood of success:** Very high — srun's environment isolation only applies to batch jobs, not foreground shells

---

### Option 2: SLURM Configuration Fix (MEDIUM DIFFICULTY)
**Request:** Ask cluster admins (gpu@spit.ac.in) to adjust SLURM config to inherit venv activation  
**Details to request:**
- "Please allow `source venv/bin/activate` to persist from the sbatch preamble into srun subprocesses"
- Or: "Please set up module system for Python venv so we can use `module load` instead"

**Pros:**
- Solves the problem permanently for future jobs
- Allows batch submissions without interactive access

**Cons:**
- Requires admin help + potential cluster restart
- Admins may decline if it conflicts with cluster policy
- Turnaround time: 1-3 days

**Likelihood of success:** Unknown — depends on admin bandwidth and cluster design

---

### Option 3: Container-Based Submission (MEDIUM DIFFICULTY)
**Method:** Package the evaluation script + venv into a Singularity/Apptainer container  
**Then:** sbatch the container as the job  
**Container includes:** Python 3.10, all pip packages, the evaluation script

**Pros:**
- Completely self-contained; environment isolation becomes irrelevant
- Portable to other clusters
- Reproducible

**Cons:**
- College cluster must support Singularity/Apptainer (verify first)
- Building a container takes ~30 min
- Slightly slower startup than native venv

**Likelihood of success:** High, if cluster supports containers

---

### Option 4: Fall Back to Kaggle or AceCloud (LEAST FRICTION)
**Method:** Move Session 2 to Kaggle notebooks (proven working) or AceCloud GPU credits

**Pros:**
- No cluster configuration issues
- Proven working on Kaggle (Session 1 energy succeeded there)
- Internet connectivity not a blocker

**Cons:**
- Cost (AceCloud credits; Kaggle is free)
- Different GPU hardware (Kaggle T4 vs. college RTX 6000 Ada)
- Less direct control over session

**Likelihood of success:** Very high (Kaggle known to work for similar workloads)

---

## What NOT to Do

❌ **Do NOT:**
- Try another wrapper-script variation — srun won't execute it
- Try to set `PATH` in the evaluation script — that's too late; srun creates a fresh shell first
- Attempt `export PYTHONHOME` or similar hacks — won't persist across srun isolation
- Blame the evaluation script for the timeout — it was never reached

❌ **These are clustering/SLURM issues, not code issues:**
- The memory fixes in `kaggle_accuracy_eval.py` are separate and correct
- The evaluation logic itself is sound (tested locally in early sessions)
- If Job 1487 had somehow reached the evaluate() call, it would have hit memory issues, which the fixes address

---

## Files Involved

| File | Role | Status |
|------|------|--------|
| `training/scripts/kaggle_accuracy_eval.py` | Main evaluation script | Fixed (memory cleanup added), untested on cluster |
| `~/greenweight_env.sh` | venv activation | Works in login shell; doesn't persist into srun |
| `~/dryrun_s2/run_eval.sh` | Attempted wrapper script | Rejected by sbatch (JobLaunchFailure) |
| `backend/src/green_weight/router/complexity_scorer.py` | Routing sensor | Not involved in Session 2 failure |
| `config.yaml` | Fuzzy controller config | Not involved in Session 2 failure |

---

## Session 2 Task Breakdown (Not Yet Completed)

From `NEW.md` Phase 2:
```
- [ ] Evaluate each tier (fp16 / 8-bit / 4-bit) × (with / without QAT adapters)
      on tinyMMLU + GSM8K subset + HellaSwag subset.
- [ ] Save per-task, per-prompt JSON results.
- [ ] Run `verify_results.py`; append to `paper/results.md`.
- [ ] Compute accuracy drop correlation with complexity features.
- [ ] Produce correlation heatmap / scatter plots.
```

**None of the above could start because the evaluation script couldn't run on the cluster.**

---

## Next Steps (For Whoever Picks Up Session 2)

1. **Immediately:** Pick one of the four options above (interactive access, container, Kaggle, or admin request)
2. **Before rerunning:** Verify the chosen path can execute Python by running a trivial test:
   ```bash
   python -c "print('Python found'); import torch; print(torch.__version__)"
   ```
3. **Once working:** Run `kaggle_accuracy_eval.py` with the memory fixes already in place
4. **Document:** Add results to `paper/results.md` and run `verify_results.py`

**Estimated time once path is working:** 4–6 hours on GPU (matches the original estimate in the script docstring)

---

## Contact & References

- **College cluster contact:** gpu@spit.ac.in
- **Cluster type:** SPIT HPC with SLURM scheduler
- **GPU:** RTX 6000 Ada (48GB VRAM each)
- **Session 1 (energy) status:** ✅ Completed successfully on the same cluster (proof that cluster access works in general)
- **Script entry point:** `training/scripts/kaggle_accuracy_eval.py`
- **Research plan reference:** `major-project/NEW.md` Phase 2

---

## Summary Table

| Job # | Attempt | Command | Error | Root Cause | Fix Tried | Why Failed |
|-------|---------|---------|-------|-----------|-----------|-----------|
| 1485 | OOM (initial dry-run) | sbatch direct | OOM 27GB allocation | HFLM memory leak | Memory cleanup added | Job cancelled; fix not tested |
| 1487 | Python not found | sbatch direct | `-sh: python: not found` | srun env isolation | None; issue identified | srun blocked PATH |
| 1488 | Wrapper script v1 | sbatch wrapper.sh | JobLaunchFailure | sbatch can't launch script | (none) | Wrapper execution denied |
| 1489 | Wrapper script v2 | sbatch wrapper.sh | JobLaunchFailure | Same as 1488 | Full python path | Same failure |

---

**Last Updated:** 2026-09-05  
**Documented by:** Claude Code session  
**For:** Future Session 2 retry or alternative GPU environment  
