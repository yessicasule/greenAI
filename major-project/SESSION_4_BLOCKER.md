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
