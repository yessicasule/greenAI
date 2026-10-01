# Experiment Log

Append-only. This is the **only** numeric source `paper-writer` may cite —
if a number isn't in this table with a PASS verdict, it doesn't exist yet
as far as the paper is concerned. `training-agent` appends a row after
every session (see `.claude/agents/training-agent.md`).

| Date | Session | Script run | Output CSV(s) | verify_results.py verdict | One-line finding |
|---|---|---|---|---|---|
| 2026-09-04 | 1 | `kaggle_energy_benchmark.py` | `session1_out/energy_logs/energy_{summary,per_inference}.csv` | PASS | Per-tier energy ground truth; 3 runs agree within 6.6%, gate passed |
| 2026-09-05 | 4 | `kaggle_routing_experiment.py` (job 1505, by-tier) | `routing_run1_{conditions,per_prompt}.csv` | CONTENDED | All 8 routing conditions; 4-bit/8-bit measured pre-contention and match Session 1, 16-bit inflated 45% |
| 2026-09-05 | 4 | `kaggle_routing_experiment.py` (job 1507, `--interleave`) | `routing_run2_{conditions,per_prompt}.csv` | CONTENDED | Same 8 conditions under full co-tenant load; 16-bit accurate to 1.6%, 4-bit/8-bit inflated 3-5x |
| 2026-09-09 | 4 | `kaggle_routing_experiment.py` (job 1529, by-tier) | `routing_run3_{conditions,per_prompt,info}` | CONTENDED | Contended throughout by job 1508; static_16bit 1581.35 J/req. Not gate-eligible |
| 2026-09-29 | 4 | `kaggle_routing_experiment.py` (job 1691, by-tier, daytime) | `routing_run4_{conditions,per_prompt,info}` | PASS routing checks (1 WARN) — but 16-bit phase DISRUPTED | 4/8-bit clean; 16-bit starved at 6.2 tok/s median, 81 W, 12.97 J/tok. See "Session 4 runs 4-5" below |
| 2026-09-30 | 4 | `kaggle_routing_experiment.py` (job 1692, by-tier, started 00:05) | `routing_run5_{conditions,per_prompt,info}` | PASS routing checks (1 WARN) — CLEAN | All tiers clean: 16-bit 80.7 tok/s, 136 W, 1.69 J/tok; 4-bit 1.71; 8-bit 3.12 J/tok |



## Session 1 — Energy ground truth (2026-09-04, SPIT cluster)



| Tier | n | Mean J/token | 95% CI | Notes |

|---|---|---|---|---|

| 4-bit | 1,500 | 1.414 | ±0.055 | baseline |

| 8-bit | 1,500 | 3.052 | ±0.039 | 2.16× higher energy |

| 16-bit | 1,500 | 8.124 | ±0.241 | 2.66× higher energy |



**Hardware:** NVIDIA RTX 6000 Ada Generation, driver 610.43.02, idle power 22.53W



**Methodology:** NVML energy counter, 500-prompt eval set, 3 full runs (seed=42)



**Known issues:** 117 rows (2.6%) with energy_j=0.0 due to prompts generating 1 token and counter rounding. These are real measurements, not meter failure. See threats-to-validity.



**Status:** VERIFIED by verify_results.py, all substantive checks PASS.


## Session 4 runs 4-5 — and a correction to Session 1's 16-bit number (logged 2026-09-30)

Both runs: RTX 6000 Ada (GPU 0, uuid GPU-c53791fb...), no co-tenant PID in the
end-of-run snapshot, 500 prompts, QAT adapters loaded on all tiers, seed 42,
same code. Accuracy and tier assignments are identical across the two runs
(deterministic); only timing/energy differ.

Per-tier medians over prompts with >=16 output tokens (computed from
`routing_run{4,5}_per_prompt.csv` and `results/energy_per_inference.csv`):

| Measurement | 4-bit tok/s · W · J/tok | 8-bit tok/s · W · J/tok | 16-bit tok/s · W · J/tok |
|---|---|---|---|
| Session 1, runs 1-3 (2026-09-04) | 67.3 · 90 · 1.33 | 26.9 · 81 · 3.00 | **10.5-11.8 · 85 · 7.2-7.8** |
| Run 4 (daytime) | 57.5 · 87 · 1.52 | 29.4 · 83 · 2.82 | **6.2 · 81 · 12.97** |
| Run 5 (overnight) | 56.4 · 96 · 1.71 | 29.0 · 91 · 3.12 | **80.7 · 136 · 1.69** |

**Finding.** 4-bit and 8-bit reproduce across all five measurements.
16-bit does not: run 5 is steady at 80.1-81.3 tok/s in every tenth of the
run, drawing 136 W (GPU busy). Run 4's 16-bit phase ran at 4.4-12.7 tok/s,
degrading over the run, at ~81 W (GPU mostly idle, i.e. starved). Session 1's
16-bit shows the same starved signature (~11 tok/s, ~85 W) in all three of
its runs — its "3 runs agree within 6.6%" gate measured consistency, not
cleanliness, because the runs were back-to-back under the same conditions.

Run 2 (`--interleave`, 2026-09-05) shows the disruption is not 16-bit
specific: under load, whichever tier is executing gets inflated. The
end-of-run co-tenant snapshot and verify_results.py both passed run 4, so
neither detects this; per-prompt tok/s and W do.

**Status of numbers:**
- Session 1 16-bit J/token (8.124) — **SUPERSEDED, do not cite as clean.**
  Session 1 4-bit/8-bit remain consistent with runs 4-5. Note: Session 1
  (`kaggle_energy_benchmark.py`) loads NO LoRA adapters; runs 4-5 load them
  on every tier — likely why Session 1's 4-bit is faster (67 vs 57 tok/s).
- Run 4 — 4-bit/8-bit usable; 16-bit phase and every condition that
  includes 16-bit prompts are **not** usable.
- Run 5 — the only clean measurement of all three tiers so far. Clean
  per-tier J/token: 4-bit 1.71, 8-bit 3.12, 16-bit 1.69. With
  bitsandbytes on this GPU, 4-bit ≈ fp16 and 8-bit ≈ 1.8x fp16.
- Needs run 6 (job 1707, queued for 2026-10-01 00:05) to agree with run 5
  before the clean 16-bit number is stated as reproducible.

Routing (identical in runs 4 and 5): fuzzy_router accuracy 0.156 vs
random_matched 0.1657 at the same tier mix — verify_results WARN, routing
adds no measurable intelligence. Accuracy metric is the reference-match
proxy (static tiers 0.11 / 0.186 / 0.162 for 4/8/16-bit).
