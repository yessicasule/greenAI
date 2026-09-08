# Session 4 Blocker: energy measurements are host-bound, not GPU-bound

**Status:** Dry run WORKS end to end. Full runs BLOCKED on a measurement-validity
problem, not a code problem.
**Found:** 2026-09-05, jobs 1494-1503 on hpc.spit.ac.in
**Reproducer:** `training/scripts/micro_bench.py` (30 lines, no project code)

---

## The finding

The GPU is idle while we measure it. Sampling `sm%` once a second through a
12-run generation benchmark:

```
samples=200  min=0  p50=1  p90=25  max=100  mean=6.9
```

Median GPU utilisation is **1%**. Meanwhile the process burns a full core
(`cpu/wall = 1.00` on every run) and throughput swings **4.4-29.4 tok/s on
identical work** — a 6.7x spread, bimodal rather than gradual: runs land
either at ~4-7 tok/s or ~18-29 tok/s with nothing in between.

Batch-1 generation of a 1B model is launch-bound: the GPU finishes each
token faster than Python can issue the next. Throughput is therefore set by
how fast the *host* can drive the GPU, and the host is shared. Another
user's job (`cruxr_worker`, 13h52m elapsed at time of measurement) sits on
the same node. When it leaves CPU headroom we get 29 tok/s; when it does not
we get 4.4.

## Why this blocks the full runs

Energy is integrated power over wall-clock. With the GPU at ~7% utilisation
and drawing 75-90 W against a ~23 W idle floor, **most of each measurement is
the GPU sitting idle at partial power while the host struggles.** So
`energy_per_request` is largely a proxy for host contention, not for the
computational cost of a precision tier.

Concretely, in the dry runs:

| Condition | job 1495 | job 1497 | swing |
|---|---|---|---|
| static_4bit | 114.49 J/req | 118.77 J/req | +4% |
| static_8bit | 240.30 J/req | 216.47 J/req | -10% |
| **static_16bit** | **874.68 J/req** | **1332.90 J/req** | **+52%** |

`SESSION_4_PLAN.md` sets the reproducibility gate at ">15% diff is a red
flag". 16-bit misses it by 3x between two runs 15 minutes apart on the same
GPU. Three runs on three days would not fix this — they would reproduce the
same bias three times and *look* consistent.

16-bit is worst hit because fp16 has no dequantisation kernels to hide
launch latency behind, making it the most host-sensitive tier. That is why
the path that should be fastest measured ~6x SLOWER than 4-bit.

## Ruled out

Each of these was tested and eliminated:

- **fp32 fallback** — `model.dtype = torch.float16`, 2.47 GB allocated (fp32
  would be ~4.9 GB). The `torch_dtype` deprecation warning is cosmetic.
- **Clock throttling / thermals** — 2760 MHz (full boost), 44-46 C,
  `clocks_throttle_reasons.active = 0x0`.
- **Contamination between tiers** — running the 16-bit tier alone (job 1498)
  reproduces the slowness, so it is not the earlier tiers failing to free.
- **Thread oversubscription** — pinning `OMP_NUM_THREADS` and
  `torch.set_num_threads()` to `SLURM_CPUS_PER_TASK` (job 1502, confirmed
  `intra-op=8`) changed nothing.
- **Same-GPU co-tenancy** — `cotenant_pids_on_this_gpu: []`; the neighbour's
  job is on GPU 1, ours on GPU 0. The contention is for host CPU, not the
  device.

Note: `cpu/wall = 1.00` is NOT evidence of CPU work. CUDA synchronisation
spin-waits by default, so a thread blocked on the GPU burns 100% CPU exactly
like a thread doing work. Only the `sm%` sampling separated the two.

## Options

1. **Exclusive node access** (`--exclusive`) for the three full runs. Removes
   the contention. Requires asking gpu@spit.ac.in whether that is available
   for 3 x 6h. This is the only option that fixes the measurement without
   changing what is being measured.
2. **Randomised tier interleaving.** Phase A currently runs all prompts on
   4-bit, then 8-bit, then 16-bit, so any drift over a 6h run lands entirely
   on 16-bit. All three tiers fit in ~5 GB of a 49 GB card, so they can be
   resident together and each prompt measured across tiers in shuffled order.
   Does not remove contention, but stops it loading onto one tier — worth
   doing regardless, as it converts an uncontrolled confound into a
   controlled one.
3. **Report GPU utilisation alongside energy.** At 7% utilisation the numbers
   describe a latency-bound serving regime dominated by idle power. That is a
   legitimate regime to study, but it must be stated, or a reviewer will read
   the energy deltas as computational cost.
4. **Batch inference.** Batch-1 on a 48 GB card is why we are launch-bound at
   all. Batching would make the workload GPU-bound and the energy numbers
   would reflect computation. This changes the research question (throughput
   serving rather than single-request latency), so it is a design decision,
   not a fix.

**Recommended:** 1 + 2 + 3. Ask for exclusivity, interleave tiers regardless
of the answer, and report utilisation in the paper.

## What is already fixed (dry run passes end to end)

Eight defects found and fixed while getting here, all on `main`:

1. **8-bit "hang"** — `torch_dtype` set only on the 16-bit branch, so 8-bit
   kept bf16 activations and every `Linear8bitLt` matmul warned about the
   cast: 1,103,631 warnings, a 93 MB log, bottlenecked on formatting warning
   text. Same failure stalled Kaggle Session 1 for ~4.7h.
2. **Phase B tuple bug** — `return rows[:LIMIT] if LIMIT else rows, complexity, ...`
   binds the ternary across the whole tuple, so any limited run returned a
   bare list and died unpacking.
3. **`pip` targeted the wrong interpreter** — the venv is Python 3.11 built
   with `uv venv` and has no `pip`, so bare `pip` fell through to the system
   3.9. Every install went to the wrong place. **Use `uv pip install`.**
4. **NVML metered the wrong GPU** — hardcoded physical index 0 while torch
   honours `CUDA_VISIBLE_DEVICES`. On a 2-GPU node this can silently measure
   another tenant's card.
5. **QAT adapters never loaded** — `ADAPTER_ROOT` only searched `/kaggle/*`.
   Every run before job 1497 was plain PTQ, reported as
   `adapters_used: false` and easy to miss.
6. **Output directories collided** — all dry runs wrote to one path; job 1497
   destroyed job 1495's results.
7. **Dependencies surfaced only after GPU time was spent** — `FuzzyController()`
   is constructed at the top of Phase B, so a missing `scikit-fuzzy` cost a
   full Phase A. Now checked in `preflight()` before any model loads.
8. **spaCy silently degrades** — missing `en_core_web_sm` makes
   `get_parse_depth` return a constant 5 with only a log warning, flattening
   one of the five routing features. Preflight now asserts against it.

## Open questions for Phase 6

- `oracle` costs more than `fuzzy_router` (696 vs 502 J/req). Expected: the
  placeholder correctness proxy scores ~0.1, so almost nothing is "correct"
  and the oracle falls back to 16-bit. Resolves when Session 2 lands.
- `threshold_router` (the naive baseline) currently beats `fuzzy_router` on
  energy at equal accuracy — 149 vs 502 J/req. On 10 prompts with placeholder
  correctness this is not a result, but if it holds at 500 prompts it inverts
  RQ3.
- Router overhead measured at 146 ms/prompt against the plan's <100 ms gate.
  Partly one-off spaCy model load amortised over only 10 prompts; recheck at
  500.

---

## Decision 2026-09-05: split the eval deliverable from the paper's gate

Agreed with the user, given an evaluation on Monday 2026-09-08 and blocked
off-campus cluster access.

**For the eval (now):** run #1 by-tier (job 1505) plus run #2 interleaved,
chained with `--dependency=afterok` so nothing runs concurrently and the
interleaved path is validated on a 10-prompt dry run before 500 prompts are
committed to it. These are two different methods, so they are deliberately
NOT the three-run reproducibility gate. What they give instead is real
500-prompt results plus a direct measurement of how much the tier-ordering
confound distorted `static_16bit` — which is a methods finding in its own
right.

**For the paper (after the eval):** three interleaved runs on different days
for the Phase 5 gate, per `SESSION_4_PLAN.md` 4.2-4.4. Run #1 is retained as
a documented by-tier comparison, not as one of the three.

This deliberately departs from `SESSION_4_PLAN.md` note 4 ("do not mix runs
from different code versions"). The note is right for the gate and is
honoured there; it is knowingly set aside for the eval deliverable, where
having two orderings is more informative than having one.

`routing_run_info.json` records `phase_a_ordering` (`by_tier` /
`interleaved`) for every run, so the two can never be conflated later.

---

## Confirmed at 500 prompts (runs #1 and #2, 2026-09-05)

Both full runs completed. Run #1 (job 1505, by-tier) finished 17:43; run #2
(job 1507, interleaved) finished 21:51. Same code, same 500 prompts, same
GPU, four hours apart:

| Condition | run #1 by-tier | run #2 interleaved | ratio |
|---|---|---|---|
| static_4bit | 125.08 J/req | 595.26 J/req | **4.76x** |
| static_8bit | 286.86 J/req | 1041.69 J/req | **3.63x** |
| static_16bit | 1160.36 J/req | 788.00 J/req | **0.68x** |

### The decisive detail: accuracy is identical in both runs

Every condition scores exactly the same in both runs — static_4bit 0.110,
static_8bit 0.186, static_16bit 0.162, fuzzy_router 0.130, oracle 0.258.

Generation is greedy (`do_sample=False`), so this is expected and it is the
point: **both runs produced token-for-token identical outputs.** The
computational work was the same to the token. Only the joules differed, by
up to 4.8x.

That rules out every workload-side explanation. The variance is not in what
the model did; it is in the environment the measurement was taken in.

### Interleaving did not fix it, and that is informative

`--interleave` was added to stop drift landing on whichever tier ran last.
It worked as designed — within run #2 the tiers no longer show the
monotonic 4bit < 8bit < 16bit ordering that time-confounding produced — but
it cannot help across runs, because the whole node's load moved between
17:43 and 21:51.

Note the ordering inverted rather than tightened: run #2 puts 8-bit
(1041 J) above 16-bit (788 J), which is physically implausible. Interleaving
converted a systematic bias into unsystematic noise. That is an improvement
in kind, not in magnitude.

### The co-tenant was constant across both runs

`squeue` on 2026-09-08 shows job 1508 (`rehanansari2`, 7-day limit) started
**2026-09-05T16:50** — before run #1 finished and before run #2 began. The
same neighbour was resident for both runs.

So the earlier framing in this document ("another user's job sits on the
same node") understates the problem. It is not that a co-tenant arrives or
leaves between runs. A *stable* set of co-tenants varies its own load enough
over four hours to swing our energy numbers by 4.8x. Scheduling runs on
different days cannot average this out, because there is no stationary
quantity to average.

**This is the argument for `--exclusive`.** Not "the node is sometimes
busy", but: identical deterministic work, measured twice on the same GPU
four hours apart, differs by up to 4.8x in energy while agreeing exactly on
every output token.

---

## CORRECTION 2026-09-08: the gate IS achievable on this node

The section above concludes that "there is no stationary quantity to
average" and that scheduling runs on separate days cannot help. **That is
wrong, and Session 1's own data disproves it.**

Session 1 (2026-09-04, `~/session1_out/energy_logs/`, 4,500 rows = 3 runs x
500 prompts x 3 tiers) agrees across its three runs to within 7%:

| Tier | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|
| 4-bit | 1.4648 | 1.3745 | 1.4021 | 6.6% |
| 8-bit | 3.0667 | 3.1163 | 2.9743 | 4.8% |
| 16-bit | 8.2237 | 7.8352 | 8.3141 | 6.1% |

Every tier passes the 15% gate. Reproducible energy measurement on this
node is not merely possible; it has already been achieved.

### What actually happened in Session 4

Job 1508 (`rehanansari2`, 7-day limit) started **2026-09-05T16:50**.
Session 1 ran the day before, on a quiet node. Session 4 straddled 1508's
arrival:

| | window | node state | 4-bit | 8-bit | 16-bit |
|---|---|---|---|---|---|
| Session 1 | 09-04 | quiet | 1.414 | 3.052 | 8.124 |
| run #1 by-tier | 15:04-17:43 | 1508 arrives 16:50 | 1.634 | 2.773 | 11.776 |
| run #2 interleaved | 17:48-21:51 | fully contended | 7.778 | 10.070 | 7.997 |

Run #1 measured 4-bit and 8-bit before 16:50 and they match Session 1
closely. Its 16-bit tier ran 15:50-17:43, straddling 1508's arrival, and is
inflated 45%.

Run #2 ran entirely under load, so 4-bit and 8-bit are inflated 5.5x and
3.3x. But its 16-bit lands at 7.997 against Session 1's 8.124 -- a 1.6%
error, the most accurate 16-bit measurement of the three sessions.

Latency confirms the mechanism: run #2's 16-bit was *faster* than run #1's
(9.42s vs 13.51s) while its 4-bit was 5x slower (7.43s vs 1.47s).

### Interleaving worked; it was asked to do too much

`--interleave` was designed to stop drift landing on whichever tier ran
last. It did exactly that: in run #1 the 16-bit tier ran last and absorbed
the entire contention, while in run #2 each tier's measurements were spread
across the whole run and 16-bit recovered to within 1.6% of ground truth.

What it cannot do is rescue a run in which *every* window is contended.
That is a scheduling problem, not an ordering problem.

### Revised recommendation

1. **Check the node before submitting.** `squeue -w hpc.spit.ac.in` costs
   nothing. Session 1 succeeded because the node happened to be quiet; that
   should be a precondition, not luck.
2. **Keep `--interleave` on.** It measurably improved the 16-bit estimate
   and costs ~5 GB of a 49 GB card.
3. **`--exclusive` remains preferable** where the queue allows it, but is no
   longer load-bearing for the claim that the gate is reachable.
4. **Session 1 stands as the per-tier energy ground truth.** Session 4's
   static-tier columns from runs #1 and #2 should be reported as contended
   measurements, not used to revise Session 1's numbers.
