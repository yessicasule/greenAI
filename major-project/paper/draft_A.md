# Does Weight Quantization Save LLM Inference Energy? A Hardware Measurement Study, with Pitfalls and a Negative Routing Result

> **Working draft (paper A), started 2026-09-30.** Reframes the project as a
> measurement study; supersedes `draft.md`'s framing, which rested on
> Session 1's 16-bit number (8.124 J/token) — now known to be a disrupted
> measurement (see `results.md`, "Session 4 runs 4-5"). Every number here
> must trace to a row in `paper/results.md`. Sections marked **PENDING**
> wait on run 6 (job 1707) and Session 2 (jobs 1708/1709).
> IEEE conference conventions; convert to IEEEtran at submission.

## Abstract — PENDING (after run 6)

Draft skeleton: weight quantization is widely assumed to reduce LLM
inference energy. We measure per-token GPU energy for Llama-3.2-1B at 4-bit
(NF4), 8-bit (LLM.int8) and 16-bit (float16) with the GPU's on-board
energy counter on an NVIDIA RTX 6000 Ada. [Headline per-tier J/token from
runs 5-6.] We further show that on a shared cluster the 16-bit
measurement can be inflated up to ~8x by disruption that neither an
end-of-run co-tenant check nor our own validation gate detects, and that a
previously gate-passing baseline was affected. Finally, we report a
negative result: a fuzzy-logic per-prompt precision router does not beat a
random router with the same tier mix, and cannot save energy that the
quantization tiers themselves do not save.

**Keywords** — LLM inference energy, quantization, bitsandbytes, GPU
energy measurement, measurement methodology, precision routing, green AI.

## I. Introduction — PENDING (after run 6)

Planned structure:
1. The assumption: fewer bits → less energy. Why it is plausible (memory
   traffic) and why it may fail (dequantization kernels, small models,
   batch size 1).
2. What we measure and on what hardware; why the NVML counter.
3. Contributions (below).
4. Why a negative routing result is worth reporting.

**Contributions (draft):**
- A per-token energy measurement of three bitsandbytes precision tiers of
  a 1B-parameter LLM on a modern datacenter-class GPU, showing [4-bit ≈
  16-bit; 8-bit ≈ 1.8× 16-bit — confirm with run 6].
- A measurement-pitfall analysis on a shared GPU cluster (§IV): the
  disruption signature, why common safeguards miss it, and a simple
  per-prompt throughput/power check that catches it.
- A negative result for complexity-aware precision routing (§VI): a
  Mamdani fuzzy router is no better than random at the same tier mix, and
  with clean energy numbers routing cannot beat static 16-bit.

## II. Related Work

*Bibliographic details must be checked against primary sources before
submission.*

**Quantization.** GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2023)
are post-training weight-quantization methods; LLM.int8() (Dettmers et al.,
2022) introduced mixed-precision int8 matrix multiplication with outlier
handling; QLoRA (Dettmers et al., 2023) introduced 4-bit NormalFloat (NF4)
with double quantization for memory-efficient fine-tuning. These works
primarily report memory footprint and task quality. We study the
bitsandbytes implementations of LLM.int8() and NF4 — the most widely used
drop-in path in Hugging Face Transformers — and ask a different question:
what they cost in energy per generated token at batch size 1.

**Energy measurement of deep learning.** Zeus (Chung et al., NSDI 2023)
and LLMCarbon (Faiz et al., 2024) represent work on measuring and modelling
the energy and carbon cost of training and inference. We follow their
emphasis on measurement over analytic estimation, using the GPU's on-board
cumulative energy counter via NVML. Our pitfall analysis (§IV) is
complementary: it concerns how such measurements go wrong on shared
infrastructure, and how to detect it.

**Routing and cascades.** RouteLLM (Ong et al., 2024) learns to route
between a weak and a strong model from preference data; FrugalGPT (Chen,
Zaharia & Zou, 2023) cascades across API models with a learned judger.
Both choose among *different models*. Our router chooses among *precision
tiers of one model* from prompt-complexity features, which is only
worthwhile if lower-precision tiers are actually cheaper — the premise §V
tests.

**Adaptive computation.** Early exit, mixture-of-depths and speculative
decoding vary compute per input. Per-request precision is another such
axis; our results suggest its energy benefit depends on the quantization
kernel, not only on bit-width.

## III. Experimental Setup and Methodology

### A. Model, Tiers and Hardware

- **Model:** `meta-llama/Llama-3.2-1B`.
- **Tiers** (bitsandbytes via Hugging Face Transformers; non-quantized
  modules in float16 on every tier):
  - 4-bit: NF4 with double quantization, float16 compute dtype.
  - 8-bit: LLM.int8() (`load_in_8bit=True`).
  - 16-bit: float16, no quantization.
- **Adapters:** each tier carries a LoRA adapter (r = 16 / 8 / 4 for
  4 / 8 / 16-bit; targets `q_proj`, `v_proj`). Loaded in the Session 4
  runs (runs 1–6); **not** loaded in Session 1, whose benchmark script
  uses the base model only. This likely explains Session 1's higher 4-bit
  throughput (67 vs. 57 tok/s) and must be stated wherever the two are
  compared.
- **Hardware:** NVIDIA RTX 6000 Ada Generation (48 GB), driver
  610.43.02, in a shared two-GPU SLURM node (dual AMD EPYC, 224 threads).
  Jobs request one GPU and 8 CPU cores; library thread pools are pinned to
  the granted cores.

### B. Workload

500 prompts stratified by difficulty (200 easy / 150 medium / 150 hard),
drawn from TriviaQA, Alpaca, GSM8K and CodeAlpaca, deduplicated. Greedy
decoding (`do_sample=False`), `max_new_tokens = 128`, batch size 1,
seed 42.

### C. Energy Measurement

Energy per generation is the difference in the GPU's cumulative on-board
energy counter (`nvmlDeviceGetTotalEnergyConsumption`, millijoule
resolution) across the generation, bracketed by
`torch.cuda.synchronize()`. This is GPU-board energy only; CPU and DRAM
energy are excluded. Only one tier is resident in GPU memory at a time. Each
tier gets 5 discarded warmup generations. Per-prompt energy, generated
tokens, and latency are logged, so per-prompt throughput (tokens/s) and
mean power (W = J/s) can be derived; these two derived quantities are the
basis of the disruption check in §IV.

J/token for a tier is reported as the median over prompts with at least 16
generated tokens [final version: also mean ± 95% CI over all prompts].
Very short generations are excluded from the median because the counter's
update granularity makes their energy unreliable: 1-token generations can
read 0 J (117 of 4,500 rows in Session 1).

### D. Accuracy

Two measures, kept distinct:
1. **Reference-match proxy** on our own 500 prompts, used only to
   attribute correctness to routing decisions: numeric match for
   arithmetic, normalized substring match for short factual answers,
   token-level F1 ≥ 0.5 otherwise.
2. **Standard benchmarks** via lm-eval-harness (tinyMMLU, tinyGSM8k,
   tinyHellaswag), each tier with and without its adapter — PENDING
   (Session 2).

### E. Validation Gate

Every run is checked by an automated validator (`verify_results.py`):
completeness of the prompt × tier grid, positive energies, plausibility
bounds, and internal consistency (e.g. the oracle router must have the
highest accuracy). As §IV shows, passing this gate is necessary but not
sufficient: it cannot detect disruption.

## IV. Measurement Pitfalls on a Shared GPU Cluster

This section reports how our own measurements went wrong, because the
failure is easy to reproduce and hard to notice.

### A. The Observation

We measured the same code, prompts and GPU several times. 4-bit and 8-bit
reproduced closely every time. 16-bit did not:

| Measurement | 4-bit tok/s · W · J/tok | 8-bit tok/s · W · J/tok | 16-bit tok/s · W · J/tok |
|---|---|---|---|
| Session 1, 3 runs (2026-09-04) | 67.3 · 90 · 1.33 | 26.9 · 81 · 3.00 | 10.5–11.8 · 85 · 7.2–7.8 |
| Run 4 (daytime) | 57.5 · 87 · 1.52 | 29.4 · 83 · 2.82 | 6.2 · 81 · 12.97 |
| Run 5 (overnight) | 56.4 · 96 · 1.71 | 29.0 · 91 · 3.12 | 80.7 · 136 · 1.69 |

(Medians over prompts with ≥16 generated tokens. Session 1 ran without
LoRA adapters; runs 4–5 with them.)

The same 16-bit model consumed between 1.69 and 12.97 J/token, a factor
of about 8. Accuracy and every routing decision were identical across runs
4 and 5 — the model computed the same outputs; only time and energy changed.

### B. The Disruption Signature

The clean 16-bit run is fast *and* power-hungry: 80.7 tokens/s, steady to
within ±1 tokens/s across every tenth of the run, at 136 W. The disrupted
runs are slow *and* low-power: 4–13 tokens/s at 81–85 W — close to what
the GPU draws when mostly idle. The GPU was waiting, not working.

Because energy ≈ power × time, and the waiting GPU still draws a large
baseline power, a starved run accumulates energy while doing little work.
Energy per token therefore rises roughly in proportion to the slowdown.

### C. Disruption Is Not Specific to 16-bit

One earlier run interleaved the tiers prompt by prompt in shuffled order
under known heavy load from another job. There, 16-bit was measured
accurately and 4-bit and 8-bit were inflated 3–5× (results.md, 2026-09-05,
job 1507). Whichever tier executes during the disruption is affected. In
the by-tier runs, 16-bit ran last and happened to coincide with it.

### D. Why Standard Safeguards Missed It

1. **Co-tenant snapshot.** Each run records which processes share its
   GPU. It is taken once, at the end of the run, and was empty for both
   the disrupted run 4 and a run later confirmed as contended (run 3).
2. **Repeat-run agreement.** Session 1's three runs agreed within 6.6%
   and passed our pre-registered gate — because they ran back-to-back
   under the same conditions. Agreement measured consistency, not
   cleanliness. Its 16-bit value (8.124 J/token) has been withdrawn.
3. **Validation gate.** `verify_results.py` passed run 4. Its checks
   (completeness, positivity, plausibility, oracle consistency) cannot
   see a uniform slowdown.
4. **GPU isolation.** The GPU was not shared at the level SLURM
   allocates. A daytime diagnostic (job 1706) isolated plain float16
   generation — no router, adapters or energy meter — on the same GPU:
   8.5 tokens/s on average (4.4–25.6) against 80+ overnight, with GPU SM
   utilisation at 0% (median) and the process on-CPU 100% of the time.
   The GPU was waiting on a CPU-side launch path that had become ~10×
   slower per token. The disruption is therefore host-side, not GPU
   contention. We did not isolate the exact host mechanism (e.g.
   contention for shared CPU or memory resources from other jobs on the
   node). Batch-size-1 float16 decoding of a 1B model is the most
   launch-bound of the three tiers, which is consistent with it being the
   most visibly affected when it happens to run during disruption.

### E. A Cheap Check That Does Catch It

Per-prompt tokens/s and mean power, both derivable from the logs every
energy study already keeps, separate the clean run from the disrupted
ones without ambiguity (80.7 tok/s at 136 W vs. 4–13 tok/s at 81–85 W).
We recommend that energy studies on shared infrastructure:
- report per-tier throughput and mean power alongside J/token;
- check throughput stability across the run (e.g. per-decile medians),
  not just the final mean;
- re-measure any tier whose throughput departs from its clean baseline;
- record co-tenancy continuously, not at a single point.

## V. Results: Energy per Token by Precision

Two clean runs on different nights (runs 5 and 6) agree within 4%
(medians over prompts with ≥16 generated tokens; Fig. 2):

| Tier | Throughput (tok/s) | Mean power (W) | Energy (J/token) |
|---|---|---|---|
| 4-bit (NF4) | 56.4 / 56.5 | 96 / 97 | 1.71 / 1.71 |
| 8-bit (LLM.int8) | 29.0 / 29.4 | 91 / 91 | 3.12 / 3.10 |
| 16-bit (float16) | 80.7 / 81.8 | 136 / 142 | 1.69 / 1.73 |

(run 5 / run 6)

**Finding.** With bitsandbytes on an RTX 6000 Ada at batch size 1, 4-bit
weights do not reduce energy per token relative to float16 (1.71 vs.
~1.7 J/token), and 8-bit nearly doubles it (~3.1 J/token). float16 draws
the most power but is fastest by a wide margin, so it finishes each token
sooner; the quantized tiers draw less power but run 1.4× (4-bit) to 2.8×
(8-bit) slower, and that extra time costs more energy than the lower power
saves.

[TODO: mean ± 95% CI over all prompts alongside the medians.]

Planned explanation, to be checked against the measurements: at batch size
1 on this GPU, bitsandbytes 4-bit and 8-bit kernels dequantize weights on
the fly. The memory-traffic saving that should favour low precision is
offset (4-bit) or outweighed (8-bit: LLM.int8's mixed-precision outlier
path) by the extra compute, while float16 runs on the GPU's native path.

## VI. Case Study: Complexity-Aware Precision Routing — a Negative Result — PENDING

Planned content (numbers from run 5, confirm with run 6):
- System: 5-feature complexity sensor + Mamdani fuzzy controller choosing
  4/8/16-bit per prompt (describe briefly; full formalization is in
  `draft.md` §III and can be condensed).
- Accuracy: fuzzy router 0.156 vs. random router with the same tier mix
  0.1657 — no routing intelligence. Diagnosis: the sensor has no opinion on
  about half the evaluation set (ROUTER_DIAGNOSIS.md).
- Energy: with clean 16-bit, fuzzy router 238 J/request vs. static 16-bit
  166 J/request — routing costs more than never quantizing.
- Oracle upper bound: accuracy 0.258.
- Lesson: routing can only save what the tiers save.

**Run 5 analysis (`paper/analysis/routing_run5_*.csv`, Fig. 3). Confirmed by
run 6 (2026-10-01): paired energy difference +65.4 J/request (54.9–75.8),
same accuracy difference, oracle 141 J/request; tier mixes per difficulty
match run 5.**
- Paired per-prompt bootstrap, fuzzy router − static 16-bit: **+71.7 J/request
  (95% CI 61.0–82.2)**, accuracy **−0.006 (95% CI −0.034 to +0.020)** — about
  43% more energy for no measurable accuracy difference.
- The sensor does react to difficulty: the router sends 60.7% of hard
  prompts to 16-bit vs. 21% of easy ones. But it sends 51.6% of all prompts
  to 8-bit, the most expensive tier (323 J/request).
- Agreement with the oracle's tier: 26.6% of prompts (28.7% of the 129
  prompts any tier answers correctly) — below the 33% expected from a
  uniform random pick.
- Only 129/500 prompts are answered correctly by any tier under the
  reference-match proxy, which caps what any router can gain in accuracy.
- Oracle definition: we use the tier with the lowest *measured* energy
  among those that answer correctly (140 J/request, accuracy 0.258). The
  experiment script's own oracle assumes fewer bits = cheaper and reports
  184 J/request at the same accuracy; that assumption is false for 8-bit
  here. State the definition in the paper.

## VII. Threats to Validity — PENDING (partly draftable now)

To cover: single model and GPU; bitsandbytes only (other kernels such as
AWQ or FP8 may behave differently — the motivation for follow-up work);
GPU-board energy only; batch size 1 only; weak reference-match accuracy
proxy; adapters loaded on every tier; small number of clean runs;
unconfirmed cause of disruption.

## VIII. Conclusion — PENDING
