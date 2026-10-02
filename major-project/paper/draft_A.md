# When Fewer Bits Don't Save Energy: Measurement Pitfalls and a Negative Result for Precision Routing in LLM Inference

> **Working draft (paper A).** Target: AI-SPC workshop @ HiPC 2026
> (deadline 2026-10-09; 4 pages + 1 page references, IEEE two-column).
> Every number must trace to `paper/results.md`. Sections still marked
> PENDING: Session 2 adapter accuracy (jobs 1723/1724), conclusion,
> final page fitting. Bibliographic details must be verified before
> submission.

## Abstract

Weight quantization is widely assumed to reduce the energy of large
language model (LLM) inference. We measure per-token GPU energy for
Llama-3.2-1B at 4-bit (NF4), 8-bit (LLM.int8) and 16-bit (float16) using
bitsandbytes at batch size 1, read from the GPU's on-board energy counter
on an NVIDIA RTX 6000 Ada in a shared academic cluster. In two clean runs
that agree within 4%, 4-bit costs the same energy per token as float16
(≈1.7 J/token) and 8-bit costs 1.8× more (≈3.1 J/token): the quantized
kernels draw less power but run 1.4–2.8× slower. Our main finding concerns
measurement itself. On the shared node, the float16 measurement was
inflated by up to 7.7× in daytime runs, while the GPU was not shared and
sat idle on a slowed host-side launch path. An end-of-run co-tenancy
check, agreement across repeated runs, and an automated validation gate
all failed to detect this, and one previously accepted baseline was
affected. Per-prompt throughput and power expose it immediately. Finally,
we report a negative result: a fuzzy-logic router that picks a precision
tier per prompt is no more accurate than a random router with the same
tier mix, and uses 38–43% more energy than always serving float16.

**Keywords** — LLM inference energy, quantization, bitsandbytes, GPU
energy measurement, measurement methodology, precision routing.

## I. Introduction

Quantizing an LLM's weights to 8 or 4 bits shrinks its memory footprint
by 2–4×, and it is natural to expect a matching energy saving: fewer bits
moved per weight, less energy per token. Whether that holds depends on how
the low-precision arithmetic is executed. The most widely used drop-in
path, bitsandbytes in Hugging Face Transformers, dequantizes weights to
16-bit on the fly, and prior benchmarking has found that this can increase
inference energy rather than reduce it [Poddar et al.].

We set out to build on the opposite premise: a router that sends each
prompt to the cheapest precision tier of one resident model able to answer
it. Measuring that premise carefully on a shared GPU cluster led to three
findings, which are this paper's contributions:

1. **A measurement pitfall that standard safeguards miss (§IV).** The
   same float16 model on the same GPU consumed between 1.7 and 13.0
   J/token depending on when it ran. The GPU was not shared; it sat idle
   (median SM utilisation 0%) while a host-side launch path slowed ~10×.
   An end-of-run co-tenancy snapshot, agreement across back-to-back
   repeated runs, and an automated validation gate all accepted the
   inflated measurement, and a baseline we had treated as verified was
   affected. We show that per-prompt throughput and mean power — already
   derivable from standard energy logs — separate clean from disrupted
   runs without ambiguity, and we give a short checklist.
2. **Per-token energy of bitsandbytes tiers at batch size 1 (§V).** In
   two clean runs that agree within 4%, 4-bit NF4 breaks even with
   float16 (≈1.7 J/token) and 8-bit LLM.int8 costs 1.8× more. This refines
   prior batched measurements [Poddar et al.] for the interactive,
   single-request setting, using a hardware energy counter rather than a
   software estimator.
3. **A negative result for per-prompt precision routing (§VI).** A
   five-feature complexity sensor with a Mamdani fuzzy controller is no
   more accurate than a random router with the same tier mix, and costs
   65–72 J/request more than static float16 (paired 95% CIs exclude zero)
   with no measurable accuracy difference. Routing can only save what the
   tiers themselves save.

## II. Related Work

**Energy of LLM inference.** Poddar et al. benchmark inference energy
across NLP tasks and find that bitsandbytes 8-bit and 4-bit quantization
increases energy to almost 2× at equal batch size, because of conversions
to 16-bit; only larger batches made quantized models cheaper (A6000,
batch sizes 8–256, CodeCarbon/CarbonTracker). TokenPowerBench [Niu et al.]
measures joules per token across model families from 1B to 405B
parameters, varying batch size, context length, parallelism and
quantization. Vellaisamy et al. decompose request energy for Llama-3.2-1B
on H100/H200 into prefill, setup and per-token components. These works
characterise *what* inference costs; none examines how measurements are
corrupted by interference on shared infrastructure, which is our main
concern, and we study the batch-size-1 case with a hardware counter.

**Measurement tooling.** Zeus [Chung et al.] and LLMCarbon [Faiz et al.]
measure or model deep-learning energy and carbon. We read the GPU's
cumulative hardware counter via NVML, and show that even a hardware
counter yields wrong answers when the workload itself is slowed by the
host.

**Quantization.** LLM.int8() [Dettmers et al., 2022] and QLoRA's NF4
[Dettmers et al., 2023] — the bitsandbytes methods we measure — and GPTQ
[Frantar et al.] and AWQ [Lin et al.] target memory and accuracy. Kernels
that compute directly in low precision may behave differently from
bitsandbytes; we leave them to future work.

**Routing.** RouteLLM [Ong et al.] and FrugalGPT [Chen et al.] route or
cascade between *different* models. Our router chooses among *precision
tiers of one model*, which is worthwhile only if lower precision is
cheaper — the premise §V tests.

**References to verify:** Poddar et al., "Towards Sustainable NLP:
Insights from Benchmarking Inference Energy in Large Language Models,"
NAACL 2025 (arXiv:2502.05610); Niu et al., "TokenPowerBench," AAAI 2026
(arXiv:2512.03024); Vellaisamy et al., "Characterization of Request and
Token Energy Costs for LLM Inference Workloads on GPU Platforms,"
arXiv:2608.28044, 2026; plus Zeus, LLMCarbon, LLM.int8(), QLoRA, GPTQ,
AWQ, RouteLLM, FrugalGPT.

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

Means agree with the medians. Over prompts with ≥16 generated tokens,
mean J/token ± 95% CI is 1.692 ± 0.007 / 1.698 ± 0.008 (4-bit),
3.124 ± 0.006 / 3.095 ± 0.008 (8-bit) and 1.683 ± 0.009 / 1.762 ± 0.016
(16-bit) for runs 5 / 6. The 4-bit vs. 16-bit difference changes sign
between runs (+0.5%, −3.6%), so we report them as equal, not ranked.

**Short generations.** Very short outputs make J/token unreliable: 81–86
of 500 4-bit generations, 13–15 of 8-bit and 36–41 of 16-bit read 0 J (the
energy counter did not advance during a 1–2 token generation). Including
them inflates mean J/token and widens its CI (e.g. 4-bit, all prompts:
1.935 ± 0.173); we therefore report the ≥16-token subset and the median.

**Memory (calculated, not measured).** Llama-3.2-1B has 1.24 B parameters,
262.7 M of them in the tied embedding, which bitsandbytes leaves in
float16. Weights therefore occupy ≈2.47 GB in float16 (measured: 2.47 GB
allocated), ≈1.5 GB at 8-bit and ≈1.0 GB at 4-bit. The memory saving is
real (1.6–2.4×); the energy saving is not.

## VI. Case Study: Per-Prompt Precision Routing — a Negative Result

**Router.** A complexity sensor computes five features per prompt —
Flesch–Kincaid grade, approximate token length, character entropy,
dependency-parse depth, and a code/maths indicator — normalised to [0, 1]
with ranges calibrated on the evaluation set. A Mamdani fuzzy controller
(scikit-fuzzy; triangular low/medium/high membership functions, seven
rules, min/max operators, centroid defuzzification) maps them to a
complexity score in [0, 100], cut at 33/66 into 4-, 8- or 16-bit. The
router runs on the CPU in 10.1–10.3 ms per prompt, ≈0.4% of a routed
request's latency (its CPU energy is not included in our GPU
measurements).

**Baselines.** Static 4/8/16-bit; *random-matched*, which assigns tiers at
random with exactly the fuzzy router's tier mix (mean of 20 draws);
*threshold*, which applies the same 33/66 cuts to the plain mean of the
five raw features; and an *oracle* that picks, per prompt, the tier with
the lowest measured energy among those that answer correctly. Because
decoding is greedy, every condition is evaluated on the same per-prompt ×
per-tier measurement grid.

**Results (runs 5 and 6 identical in routing; energy from both; Fig. 3).**

| Condition | Accuracy | J/request (run 5 / 6) |
|---|---|---|
| Static 4-bit | 0.110 | 130 / 130 |
| Static 8-bit | 0.186 | 323 / 320 |
| Static 16-bit | 0.162 | 166 / 174 |
| Fuzzy router | 0.156 | 238 / 239 |
| Random, matched mix | 0.166 | 241 / 242 |
| Threshold | 0.120 | 185 / 184 |
| Oracle | 0.258 | 140 / 141 |

Paired per-prompt bootstrap, fuzzy router − static 16-bit: +71.7 J/request
(95% CI 61.0–82.2) in run 5 and +65.4 (54.9–75.8) in run 6, i.e. 38–43%
more energy; accuracy −0.006 (−0.034 to +0.020), not distinguishable from
zero.

**Why it fails.** The sensor does respond to difficulty — it sends 61% of
hard prompts to 16-bit against 21% of easy ones — but it sends 52% of all
prompts to 8-bit, the most expensive tier. Its tier choice matches the
oracle's on 27% of prompts (run 5), below the 33% expected from a uniform
random choice. The oracle
itself shows the ceiling: 0.258 accuracy at 140 J/request, mostly by
using 4-bit where it suffices and float16 otherwise, and almost never
8-bit. The lesson generalises beyond this router: per-request precision
routing can only save energy that the precision tiers themselves save, so
the tiers' measured costs must come first.

**Accuracy metric.** Accuracy here is a reference-match proxy (numeric
match, normalised substring match, or token-F1 ≥ 0.5) on our 500 prompts;
only 129 prompts are answered correctly by any tier, which caps what a
router can gain. Standard-benchmark accuracy per tier, with and without
the LoRA adapters, is in Table [PENDING Session 2, job 1731].

## VII. Threats to Validity

- **One model, one GPU, one library.** Llama-3.2-1B on an RTX 6000 Ada
  with bitsandbytes. Larger models, other GPUs, and kernels that compute
  in low precision (e.g. AWQ or FP8 paths) may show real energy savings;
  our result is a statement about this widely used configuration, not
  about quantization in general.
- **Batch size 1.** Interactive, single-request decoding. Prior work finds
  quantized models become cheaper at large batch sizes [Poddar et al.].
- **GPU-board energy only.** NVML reports GPU energy; CPU, DRAM and
  cooling are excluded, as is the router's CPU energy.
- **Disruption mechanism not isolated.** We show the slowdown is
  host-side and not GPU contention, but not which host resource caused it.
- **Few clean runs.** Two clean routing runs (a third is queued) and one
  daytime diagnostic.
- **Weak accuracy proxy.** The reference-match metric is coarse; routing
  conclusions rest mainly on energy, where the result is unambiguous.
- **Adapters.** All Session 4 tiers carry LoRA adapters, adding a small
  overhead (Session 1 without adapters: 67 vs. 57 tok/s at 4-bit).

## VIII. Conclusion

Measured with a hardware energy counter at batch size 1, bitsandbytes
4-bit quantization of a 1B-parameter LLM saves memory but not energy, and
8-bit costs 1.8× more than float16, because the quantized kernels' lower
power does not make up for their slower decoding. Getting this answer
right was harder than expected: on a shared cluster, a host-side slowdown
inflated float16 energy by up to 7.7×, passed every safeguard we had, and
had already contaminated a baseline. Reporting per-prompt throughput and
power alongside energy catches it. Finally, a fuzzy per-prompt precision
router, built on the assumption that fewer bits cost less, uses more
energy than never quantizing. Future work will test whether kernels that
compute natively in low precision (AWQ, FP8) and larger models change the
picture, and whether routing becomes worthwhile when they do.
