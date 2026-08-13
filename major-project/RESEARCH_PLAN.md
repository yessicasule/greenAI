# Green-Weight: Path to a Publishable Research Paper

*Plan written 2026-07-10. Working title: "Complexity-Aware Dynamic Precision
Routing for Energy-Proportional LLM Inference."*

## 0. Honest status audit (read this first)

**There are currently no verified energy or accuracy results.** As of the last
pipeline runs (May 2026):

- `results/energy_logs/energy_summary.csv` is empty (header only); every tier
  shows `count: 0` in the pipeline logs — CodeCarbon never captured a single
  measurement.
- `results/accuracy_logs/accuracy_results.json` is `{}` — `lm-eval` was not
  installed, so all accuracy phases errored out.
- The "40% energy savings" figure in CLAUDE.md comes from an *assumed* linear
  energy model (4-bit = 0.25x, 8-bit = 0.5x, 16-bit = 1x), not measurement.
  This assumption is very likely wrong on real hardware (see §2).
- The local machine has no NVIDIA GPU, so all measurements must run on Kaggle.

Bugs that must be fixed before any result is citable:

| Location | Problem |
|---|---|
| `backend/src/green_weight/benchmark/energy_tracker.py:104-110` | Converts CO₂ back to joules with an inverted constant (`1 kg CO₂ ≈ 0.17 kWh` is backwards). CodeCarbon reports energy directly (`energy_consumed`, kWh) — use that, or better, NVML (see script). |
| `backend/src/green_weight/benchmark/energy_tracker.py:205` | J/token divides by a hard-coded "50 tokens" instead of counting real generated tokens. |
| `backend/src/green_weight/core/dynamic_inference.py:91,114` | Responses can be *simulated*; token counts estimated by word-split × 1.3. Fine for demos, must never feed the paper. |
| `backend/src/green_weight/evaluation/benchmark.py:57` | Energy = 1 pJ/op analytic guess. Can be reported as a *model*, never as a *measurement*. |
| `backend/src/green_weight/config.yaml:9` vs adapters | Config says `Llama-2-7b-hf`; the three trained QAT adapters are based on `Llama-3.2-1B`. Unify on Llama-3.2-1B. |
| `backend/src/green_weight/router/routellm_bridge.py` | Broken against current RouteLLM API (`model_name_or_path` kwarg). Fix or drop to baseline-only. |
| `data/eval_prompts.jsonl` | Contains duplicate prompts — dedupe and re-stratify before evaluation. |
| `CLAUDE.md` "HuggingFace Token" line | Never commit a real token; use Kaggle secrets / env vars. |

## 1. The claim (pick one and defend it)

**Recommended core claim:** *A lightweight prompt-complexity sensor plus a
fuzzy-logic controller can route prompts across quantization tiers of a single
LLM, cutting measured energy per token by X% at ≤Y% accuracy loss versus
static fp16 inference.*

X and Y are whatever the measurements say. If measured savings are 18% instead
of 40%, publish 18% — a sound methodology with modest numbers is publishable;
a modeled 40% is not.

Novelty positioning against related work (all must be cited and differentiated):

- **RouteLLM** routes between *different models* (weak/strong). You route
  between *precision tiers of the same model* — no second model to host.
- **FrugalGPT** cascades across *paid APIs* with a stop-judger. Your cascade
  variant is a baseline/extension, not the core claim.
- **Quantization work** (GPTQ, AWQ, LLM.int8, QLoRA) picks *one* static
  precision. You make precision *dynamic per request*.
- **Adaptive computation** (early exit, mixture-of-depths, speculative
  decoding) adapts depth/steps; you adapt arithmetic precision.
- **Energy measurement work** (Zeus, MELODI, LLMCarbon) provides methodology
  you build on — cite for the measurement protocol.

Research questions:

1. **RQ1 (measurement):** What is the real J/token of 4-bit / 8-bit / fp16
   Llama-3.2-1B inference on a commodity GPU (T4)?
2. **RQ2 (routing):** How much energy does complexity-aware fuzzy routing save
   vs static fp16, and at what accuracy cost?
3. **RQ3 (baselines):** Does the fuzzy controller beat trivial routers
   (random with matched tier distribution, single threshold) and approach the
   oracle router? **Addendum (2026-08-05):** the `RouteLLMBridge` is
   currently a documented threshold pass-through, not a real RouteLLM
   integration (RouteLLM's pretrained checkpoints are trained on specific
   named-model pairs and don't transfer to our quantization tiers — see
   `CREDIBILITY_REPORT.md` §1). The actual RouteLLM-methodology
   contribution is a **real tier-preference router trained on Session 4's
   `routing_per_prompt.csv`** (comparing our own tiers' outputs the way
   RouteLLM compares model outputs) — planned as a post-Session-4 addition
   to this RQ, once that data exists to train it on.
4. **RQ4 (ablation):** Do the bit-resilient QAT LoRA adapters improve low-bit
   quality over plain post-training quantization?

## 2. The biggest scientific risk — face it early

On a T4, bitsandbytes 4-bit/8-bit **dequantizes weights to fp16 for compute**.
You save memory bandwidth, but LLM.int8 in particular is often *slower* than
fp16 for small models, and energy may not drop linearly with bit-width — for a
1B model that fits comfortably in fp16, savings can be small or even negative.

Mitigations, in order of preference:

1. **Measure first** (Session 1 below). If 4-bit shows real savings, proceed.
2. If bitsandbytes shows no savings, switch the low tiers to **GPTQ or AWQ
   checkpoints with real low-bit kernels** (ExLlama/Marlin) — these have
   genuine low-precision compute paths and typically do save energy.
3. If per-token energy barely differs, reframe the energy axis as
   **energy per request** (quantized tiers can use smaller batching head-room)
   or pivot the paper toward the *measurement study* (RQ1) plus routing as
   a latency/cost result. Decide only after Session 1 data exists.

## 3. Measurement methodology (what makes the results "verified")

- **Meter:** NVML `nvmlDeviceGetTotalEnergyConsumption` (hardware energy
  counter, mJ resolution, supported on T4/Turing). Fallback: integrate
  `nvmlDeviceGetPowerUsage` at 20 Hz. Do **not** use CodeCarbon's CO₂ output
  as an energy proxy. Use Kaggle **T4**, not P100 (Pascal has no counter).
- **Protocol per condition:** 5 warmup generations → N≥300 prompts, greedy
  decoding, fixed `max_new_tokens=128`, count *actual* generated tokens.
  Repeat the full sweep 3× (different session times), report mean ± 95% CI.
- **Controls:** record GPU model, driver, idle-power baseline, clocks; one
  tier in memory at a time; `torch.cuda.synchronize()` around timing.
- **Report:** J/token (primary), J/request, latency, peak memory — per tier
  and per routing condition. State the hardware and its limits in a
  "threats to validity" section (single GPU type, 1B model, greedy decoding).

## 4. Experimental conditions

| # | Condition | Purpose |
|---|---|---|
| 1 | Static fp16 | Quality/energy reference (baseline) |
| 2 | Static 8-bit | Static quantization point |
| 3 | Static 4-bit | Lower bound on energy & quality |
| 4 | **Fuzzy router** (ours) | The contribution |
| 5 | Random router, tier distribution matched to #4 | Proves routing *intelligence* matters, not just tier mix |
| 6 | Simple threshold on complexity score | Proves *fuzzy* beats naive control (or admit it doesn't) |
| 7 | Oracle router (cheapest tier that still answers correctly) | Upper bound / headroom |
| 8 | FrugalGPT-style cascade (existing `frugal_cascade`) | Optional strong baseline |

**Quality metrics:** lm-eval-harness — MMLU (use `tinyMMLU`/subset for GPU
budget), GSM8K, HellaSwag; plus RouteLLM's APGR/CPT metrics for routing
conditions. **Energy metrics:** from §3. **Headline figure:** accuracy vs
J/token Pareto plot with all 8 conditions.

**Ablations:** ± QAT adapters per tier; complexity-sensor feature ablation
(drop Flesch-Kincaid / entropy / syntax depth one at a time); sensitivity to
the 33/66 tier thresholds.

**Statistics:** 3 seeds/runs, bootstrap 95% CIs, paired comparison between
condition 4 and condition 1 accuracy.

## 5. GPU session guideline (Kaggle free tier, ~30 h/week)

> Implementation status: all session scripts exist. Click-by-click
> instructions live in **KAGGLE_MANUAL.md**; the verification policy and
> claim status live in **CREDIBILITY_REPORT.md**. Scripts:
> `training/scripts/kaggle_energy_benchmark.py` (S1), `training/scripts/kaggle_accuracy_eval.py`
> (S2), `backend/src/green_weight/training/kaggle_qat_trainer.py` (S3),
> `training/scripts/kaggle_routing_experiment.py` (S4), `training/scripts/make_figures.py` +
> `training/scripts/verify_results.py` (S5).

All sessions use **T4** accelerator. Estimated total: ~20–25 GPU hours.

**Session 1 — Energy ground truth (~2–3 h). DO THIS FIRST.**
Run `training/scripts/kaggle_energy_benchmark.py` (ready to paste). Produces
`energy_summary.csv` = the paper's Table 1 and the go/no-go signal for §2.
Upload `backend/src/green_weight/data/eval_prompts.jsonl` (script dedupes it).

**Session 2 — Per-tier accuracy (~4–6 h).**
`pip install lm-eval`; evaluate each tier (fp16 / 8-bit / 4-bit, with and
without your QAT adapters) on tinyMMLU + GSM8K subset + HellaSwag subset.
Save per-task JSON results. This yields RQ4 and the per-tier quality column.

**Session 3 — Adapter retraining only if needed (~6 h).**
The three LoRA adapters already exist (`adapters/adapter_{simple,medium,complex}`,
base Llama-3.2-1B). First try loading them with the current transformers/peft;
retrain via `backend/src/green_weight/training/kaggle_qat_trainer.py` only if they fail to
load or Session 2 shows them underperforming plain quantization.

**Session 4 — Routing experiment (~4–6 h).**
Build the deduped, stratified 500-prompt eval set (easy 200 / medium 150 /
hard 150). Run conditions 1–8 with the Session-1 energy meter wrapped around
generation. Log per-prompt: tier chosen, energy, tokens, response, correctness.

**Session 5 — Figures & stats (CPU, local).**
Pareto curve (accuracy vs J/token), tier-distribution bars per condition,
energy bars with CIs, ablation table. 300 dpi PNG + PDF.

## 6. Paper structure & target venues

Structure (6–8 pages): Abstract → Intro → Related Work → System (sensor,
fuzzy gearbox, tiers, QAT adapters) → Measurement Methodology → Results
(RQ1–4) → Threats to Validity → Conclusion. The Threats section is mandatory:
single GPU class, single 1B model, greedy decoding, English-only prompts.

Venues, in recommended order:

1. **SustaiNLP workshop (EMNLP)** — exact topical fit, workshop bar,
   welcomes honest negative/modest results.
2. **IEEE IGSC** (International Green and Sustainable Computing) — systems
   framing fits the measurement study.
3. **NeurIPS ENLSP / ICML ES-FoMo workshops** — efficient-inference fit.
4. **TMLR** (journal) — no novelty bar, but demands rigor; viable if
   Sessions 1–4 produce clean, well-CI'd results.

Check each venue's current CFP dates before committing; workshop deadlines
typically fall 2–3 months before the main conference.

## 7. Timeline (~7 weeks)

| Week | Milestone |
|---|---|
| 1 | Fix table-of-bugs above; run Session 1; **go/no-go decision on §2** |
| 2 | Session 2 + adapter check (Session 3 if needed) |
| 3 | Dedupe/stratify eval set; fix router bridge or drop; Session 4 |
| 4 | Repeat runs for CIs; ablations; Session 5 figures |
| 5 | Write methodology + results; related-work pass |
| 6 | Full draft; internal review (advisor) |
| 7 | Polish; submit to chosen venue; release code + CSVs on GitHub (reproducibility artifact) |

## 8. Open questions to resolve (owner: you)

1. Scope: is the paper the fuzzy precision-router alone (recommended), or the
   full router+cascade+QAT system? Narrower is easier to defend.
2. Venue/deadline: workshop this cycle vs journal (TMLR) later?
3. Confirm you can use Kaggle **T4** (not just P100/Colab) — the energy
   counter requirement.
4. Commitment to report measured numbers even if savings < 40%.
