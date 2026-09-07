# Quick Start: GPU Session Submissions (Phases 1 & 2)

**TL;DR:** Two session scripts, ready to run. Phase 1 is the go/no-go gate; Phase 2 unlocks the paper's real contribution.

---

## Setup (One-Time)

```bash
# On the cluster, navigate to repo root
cd major-project

# Verify cluster access
ssh cluster-login  # (or already logged in)

# Check Python 3.11
python3.11 --version

# Check HF_TOKEN
echo $HF_TOKEN
# If empty: export HF_TOKEN=hf_<your-token>

# Make scripts executable
chmod +x training/session1_submit.sbatch
chmod +x training/session2_submit.sbatch
```

---

## Timeline & Gates

| Phase | What | GPU Hours | Gate | Status |
|-------|------|-----------|------|--------|
| **1** | Energy ground truth (3 tiers × 500 prompts) | 2–3h | **Energy savings ≥15%?** | ✅ Ready |
| **2** | Per-tier accuracy + correlation study | 4–6h | **Complexity features predict sensitivity?** | ✅ Ready |
| **5** | Routing experiment (8 conditions × 3 runs) | 12–18h | **Pareto curve defensible?** | ⏳ Gated by 1 & 2 |

---

## The Three Scripts

### Phase 1: Energy Measurement
```bash
cd major-project/training
sbatch session1_submit.sbatch
```
**Expected output:** `results/energy_summary.csv` (mean J/token per tier)

**What to watch:** First 30 minutes for the 8-bit hang risk (documented Kaggle failure). If progress stalls, kill and troubleshoot.

**Monitoring:**
```bash
tail -f session1_<JOB_ID>.log
```

**Expected runtime:** 2–3 hours

---

### Phase 2: Accuracy & Correlation
```bash
cd major-project/training
sbatch session2_submit.sbatch
```
**Expected output:** `results/accuracy_summary.csv` (mean accuracy per tier per task)

**What changed:** Sept 5 venv fix — no `srun`, use direct `python`, install with `uv pip`.

**Monitoring:**
```bash
tail -f session2_<JOB_ID>.log
```

**Expected runtime:** 4–6 hours

---

## What's Different This Time (vs. Kaggle / Earlier Failures)

| Issue | Kaggle | College Cluster (Now) |
|-------|--------|----------------------|
| Platform | T4, 12h cap, 8-bit hang | RTX 6000 Ada, no cap, hang risk monitored |
| Energy measurement | CodeCarbon (slow) | NVML hardware counter (validated) |
| Accuracy path | Fabricated hardcoded numbers | Real `lm_eval.simple_evaluate()` |
| Adapters | Path bugs, never loaded | Fixed, verified on CPU load test |
| SLURM venv | `srun` broke Python discovery | Direct `sbatch`, `uv pip` (Sept 5 fix) |
| Tests | None | 146 tests, all passing |
| Frontend | UTF-8 mojibake, wrong model name | Fixed + redesigned eco theme |

**Bottom line:** Implementation hardening pass (Aug 22) + venv fix (Sept 5) = production-ready.

---

## The Three Critical Gates

### Gate 1: Phase 1 (Energy)
**Question:** Does precision actually affect energy for a 1B model?

- **Pass:** Savings ≥15% over fp16 → proceed to Phases 2 & 5 as planned
- **Fail:** Marginal/negative savings → pivot:
  - Swap to GPTQ/AWQ with real low-bit kernels (ExLlama/Marlin)
  - Or reframe as latency/cost instead of energy
  - Or frame as honest measurement study (publish whatever numbers land)

### Gate 2: Phase 2 (Accuracy & Correlation)
**Question:** Do prompt complexity features justify the fuzzy router's design?

- **Pass:** Correlation heatmap shows features correlating with quantization sensitivity → proceed to Phase 5
- **Fail:** Features don't correlate (noise) → rethink sensor before Phase 5

### Gate 3: Phase 5 + 6 (Routing Experiment & Figures)
**Question:** Is the Pareto curve defensible for publication?

- **Pass:** Fuzzy router beats baselines → submit to TMLR / EuroSys
- **Fail:** Or findings are modest but honest → still publishable (that's the commitment)

---

## Detailed Guides

- **SESSION1_SUBMISSION_GUIDE.md** — full Phase 1 instructions, troubleshooting, hang monitoring
- **SESSION2_SUBMISSION_GUIDE.md** — full Phase 2 instructions, venv fix context

---

## After Each Session Completes

### After Phase 1:
```bash
# Check gate
# If go → continue to Phase 2
# If no-go → follow pivot options in SESSION1_SUBMISSION_GUIDE.md

# Commit results
git add major-project/training/results/
git commit -m "Session 1 complete: energy ground truth"
```

### After Phase 2:
```bash
# Check gate (correlation heatmap makes sense?)
# If yes → ready for Phase 5

# Commit results
git add major-project/training/results/
git commit -m "Session 2 complete: accuracy and complexity correlation"
```

### Phase 5 (Later, gated by 1 & 2):
```bash
# Will create session4_submit.sbatch once Phase 1 & 2 data exist
# This runs 8 routing conditions × 3 repeats (~12–18h total)
# Produces the paper's main Pareto curve
```

---

## What's Blocking → What's Now Unblocked

**✅ Phase 1 (Energy):** Ready to submit NOW
- Dry run passed Sept 4 ✓
- NVML counter works ✓
- Script is production-ready ✓

**✅ Phase 2 (Accuracy):** Ready to submit after Phase 1
- SLURM venv fix applied (Sept 5) ✓
- lm-eval wired correctly ✓
- Script is production-ready ✓

**⏳ Phase 5 (Routing):** Gated by Phases 1 & 2
- Will create session4_submit.sbatch once Phase 1 passes go/no-go gate
- Unlocks the paper's main contribution (Pareto curves, all routing conditions)

---

## Setup Checklist

Before submitting anything:

- [ ] On-campus (cluster is on-campus only)
- [ ] Python 3.11 available: `python3.11 --version`
- [ ] HF_TOKEN set: `echo $HF_TOKEN` → `hf_...`
- [ ] Llama-3.2-1B license accepted on HuggingFace
- [ ] GPU access confirmed: `nvidia-smi`
- [ ] sbatch scripts executable: `ls -l training/session*.sbatch`

---

## Next Steps

1. **Get on-campus** with GPU access
2. **Export HF_TOKEN** in your shell
3. **Submit Phase 1:** `sbatch training/session1_submit.sbatch`
4. **Monitor** first 30 minutes (8-bit hang risk)
5. **Wait** 2–3 hours
6. **Check gate:** Did energy savings meet the threshold?
7. **If yes:** Submit Phase 2: `sbatch training/session2_submit.sbatch`
8. **Wait** 4–6 hours
9. **Check gate:** Do features correlate with quantization sensitivity?
10. **If yes:** Phase 5 (routing) unlocked — create session4_submit.sbatch

---

## Files in This Directory

```
major-project/training/
├── session1_submit.sbatch              # Phase 1 job script
├── session2_submit.sbatch              # Phase 2 job script
├── SESSION1_SUBMISSION_GUIDE.md        # Detailed Phase 1 instructions
├── SESSION2_SUBMISSION_GUIDE.md        # Detailed Phase 2 instructions
├── SUBMISSION_QUICK_START.md           # This file
├── scripts/
│   ├── kaggle_energy_benchmark.py      # Phase 1 entrypoint
│   ├── kaggle_accuracy_eval.py         # Phase 2 entrypoint
│   ├── kaggle_routing_experiment.py    # Phase 5 entrypoint (later)
│   ├── verify_results.py               # Validation script
│   └── ...
├── results/                            # Auto-created on first run
│   ├── energy_per_inference.csv        # Phase 1 output
│   ├── accuracy_summary.csv            # Phase 2 output
│   └── ...
├── logs/                               # SLURM transcripts (optional)
└── configs/                            # Hyperparam externalization (not yet wired)
```

---

## Questions or Issues?

See the full documentation:
- **major-project/NEW.md** — live roadmap, Phases 0–9
- **major-project/CLAUDE.md** — project structure & design decisions
- **major-project/CREDIBILITY_REPORT.md** — measurement methodology & verification gates
- **major-project/backend/tests/** — test suite (146 tests, all passing)

Good luck! 🚀
