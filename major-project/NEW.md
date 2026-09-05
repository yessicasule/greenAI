# Green-Weight — Roadmap to a Publishable Research Paper

**Domain:** Green / Sustainable AI — energy-proportional LLM inference.
**Working title:** *Complexity-Aware Dynamic Precision Routing for Energy-Proportional LLM Inference.*
**Status as of this roadmap:** Pre-measurement. Code/methodology built and
code-reviewed; **zero verified energy or accuracy numbers exist yet**
(`paper/results.md` is empty, `CREDIBILITY_REPORT.md` §1 = all PENDING).
This file is the single ordered path from here to a submitted paper.

**How to use this file:** work top to bottom. Each phase has a **gate** —
do not start the next phase until the gate condition is met. Check boxes as
you go; this doc is meant to be edited in place as a living tracker
(unlike `paper/results.md`, which is append-only).

---

## Phase 0 — Pre-flight (no GPU, ~1 day) — STATUS: complete (last open item, the Kaggle error text, resolved 2026-08-22)

Goal: remove every known blocker before spending GPU hours.

- [x] Fix the bug table in `RESEARCH_PLAN.md` §0 before any measurement run:
  - [x] `benchmark/energy_tracker.py:104-110` — inverted CO₂→joules constant.
        **Verified already fixed** (2026-08-13 code read): reads
        `tracker.final_emissions_data.energy_consumed` (kWh) directly, per
        the comment "never derived from CO2". No action needed.
  - [x] `benchmark/energy_tracker.py:205` — hard-coded "50 tokens" divisor.
        **Verified already fixed**: `joules_per_token` is computed from
        real summed `tokens_out` (actual generated-token counts), not a
        constant. No action needed.
  - [x] `config.yaml:9` vs adapters — unify model on `meta-llama/Llama-3.2-1B`.
        **Verified already fixed**: `config.yaml` `model.base_model_id` is
        already `meta-llama/Llama-3.2-1B`, matching the adapters. No
        action needed.
  - [x] `router/routellm_bridge.py` — confirmed already fixed to documented
        pass-through (no action needed, just don't re-break it).
  - [x] `data/eval_prompts.jsonl` — **file doesn't exist yet anywhere in
        the repo.** It's generated fresh, already deduped, by
        `training/scripts/prepare_eval_dataset.py` (script exists, not yet
        run — that run is genuinely Phase 1 work, not Phase 0; nothing to
        fix here now).
- [x] ~~Platform decision: Kaggle T4~~ **REVERSED 2026-08-13.** Kaggle's
      real 500-prompt Session 1 run errored out after reaching the 8-bit
      tier (~2.6h in) — exact error not yet captured, see open item below.
      **New platform decision: college GPU cluster, for every measurement
      session (1, 2, and 4)**, not just Session 3 as originally planned.
      Same single-platform rule applies as before — whichever platform is
      used, it must be used consistently across all four sessions, so this
      is now a full switch, not a mix.
      **Blocking constraint: cluster is only reachable on-campus** — no
      remote/VPN access confirmed, so nothing GPU-side can happen until
      physically at college. Non-GPU work can still proceed in the
      meantime (see "what to do today" below).
      **Not yet done, required before Session 1 can (re-)run on the
      cluster:** the hardware-compatibility check —
      `training/scripts/check_nvml_energy_support.py` (now saved in the
      repo; `pip install nvidia-ml-py` then run it) — confirms the
      cluster's GPU model and whether `nvmlDeviceGetTotalEnergyConsumption`
      is supported; Volta+ required, Pascal/older won't work. Run this the
      moment you're on-campus, before spending any GPU time on a real
      session.
      **Resolved 2026-08-22, together with the user (their Kaggle
      account, via the in-app Browser tool — user logged in themselves,
      no credentials handled by the assistant).** Found the notebook
      (`new_and_improved_greenai`, kernel id 130610077 — this notebook
      lives only on Kaggle, it is NOT one of the 4 `.ipynb` files in
      `training/scripts/`, so its exact code isn't locally auditable) and
      inspected its version history (5 versions) and logs directly:
      - **Version 4** ("Failed after 16 seconds"): a clean, quick
        `papermill.exceptions.PapermillExecutionError` /
        `BackendError` — `os.environ["HF_TOKEN"] =
        UserSecretsClient().get_secret("HF_TOKEN_NEW")` failed with "No
        user secrets exist for kernel id 130610077 and label
        HF_TOKEN_NEW". Simply the Kaggle secret not being attached under
        that label for that specific notebook/run — unrelated to the
        real incident, easily avoided by double-checking the secret is
        attached before starting a real session.
      - **Version 3** ("Cancelled after 12 hours") — **this is the real
        incident.** Log shows normal operation up to ~26218s (~7.3h in,
        during 8-bit-tier work), then a burst of ~700+ repeated
        `MatMul8bitLt: inputs will be cast from torch.bfloat16 to
        float16 during quantization` warnings all logged within a
        ~0.4-second window, then **complete silence** — no further log
        output at all — until Kaggle force-killed the session at
        exactly 43200.3s (12h, exit code 137, "exceeded the max allowed
        execution duration"). Version 5 (identical code to Version 4,
        which is identical to Version 3's code) repeated the same
        "cancelled after 12 hours" outcome.
      - **Answer to the original question:** this was NOT a clean
        Kaggle-side limit like OOM or quota (those produce distinct,
        immediate error messages) — the *proximate* trigger was Kaggle's
        12h session cap, but the *root cause* looks like a genuine
        **silent hang** in the script itself around 8-bit-tier inference,
        which is exactly the class of risk that **could recur on the
        college cluster too**, not something switching platforms
        automatically fixes.
      - **Actionable takeaway for Phase 1 on the cluster:** don't start a
        real session and walk away for hours unattended — check in
        periodically during the first hour or two (does it clear the
        4-bit tier and get meaningfully into 8-bit in a reasonable time,
        roughly matching the ~2-3h total budget?); if progress visibly
        stalls, kill and investigate rather than waiting for a platform
        timeout to do it for you. Could not confirm whether this exact
        hang risk still exists in the *current* `kaggle_energy_benchmark.py`
        (already reviewed and fixed today for an unrelated bytes-decode
        bug) since the old notebook's code isn't in the repo to diff
        against — treat as an open risk to watch for, not a confirmed
        non-issue.
- [x] **`HF_TOKEN` set up (confirmed 2026-08-13).** Token generated
      (Read scope), stored as a Kaggle secret, license granted on
      `meta-llama/Llama-3.2-1B` (HF model page confirms "Gated model — you
      have been granted access to this model").
      ⚠️ **One fix needed before Session 1**: the secret in the notebook
      is currently labeled `HF_TOKEN_NEW`, and the first-cell snippet
      assigned it to `os.environ["HF_TOKEN_NEW"]`. Every session script
      (`kaggle_energy_benchmark.py` etc.) reads the env var
      `HF_TOKEN` specifically (what `huggingface_hub` auto-detects) — the
      secret's Kaggle *label* can stay `HF_TOKEN_NEW`, but the first cell
      of every session notebook must be:
      ```python
      from kaggle_secrets import UserSecretsClient
      import os
      os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN_NEW")
      ```
      Apply this in Session 1's notebook (Phase 1) — not yet verified
      end-to-end (no session run yet), so treat as done-pending-first-run.
- [x] **Decided and committed (revised 2026-08-13 — IGSC ruled out, see below):**
  - ~~Target venue: IEEE IGSC~~ **Ruled out.** IGSC 2026's paper deadline
    was April 7, 2026 and the conference ran June 22–24, 2026 — both
    already passed as of this roadmap. IGSC 2027's CFP isn't published
    yet, and even on the historical pattern its conference would land
    ~June 2027 — after the hard April 2027 publish constraint below.
    Sources: [IGSC 2026 CFP](https://www.igscc.org/igsc26-cfp).
  - **New constraint from the user:** draft must be write-ready by
    **October 2026**; must be **published by April 2027 at the latest**.
  - **Target venue: TMLR (Transactions on Machine Learning Research).**
    Rolling submission — no fixed deadline, submit once ready (~Oct/Nov
    2026). No novelty bar; review criteria are technical correctness +
    clarity, not competitive excitement — explicitly the best fit for a
    paper whose central commitment is "publish the real measured numbers,
    even if modest." Typical review turnaround 2-4 months, sometimes
    longer — submitting by late October gives real margin against the
    April 2027 target, though TMLR does not *guarantee* a publish-by date
    the way a fixed conference would.
  - **Stretch/backstop option: EuroSys 2027, Fall cycle.** Deadline
    Sept 24, 2026; conference April 19–23, 2027 (matches the April cutoff
    almost exactly). CCF-A, highly competitive (~15-20% acceptance,
    reviewers expect production-scale systems evaluation) — a real reach
    for a single-model/single-GPU-class study. Only worth pursuing in
    parallel if Session 1-4 (Phases 1-6) results land unusually strong.
  - Ruled out on research: AAAI-27 IAAI track (wants already-deployed
    applications with a track record, not an academic measurement study —
    poor fit despite the "applied" framing), ICASSP 2027 (publish date
    slips to May 2027, past the cutoff; audience mismatch), NeurIPS 2026
    workshops incl. AXIOM/Climate-Change-AI/ENLSP (deadlines already in
    Aug/Sept 2026, too tight given GPU sessions haven't started; ENLSP not
    even confirmed to run in 2026), SustaiNLP 2026 (no confirmed CFP found
    for this cycle), MLSys 2026/2027 (2026 deadline already passed;
    2027 deadline unconfirmed and conference lands May 2027, past cutoff).
  - Explicit commitment: *we publish whatever the real measured numbers
    are, even if savings are modest, not just if they hit 40%.*
- [x] Scope decided: **fuzzy precision-router alone** (RESEARCH_PLAN.md §8,
      Q1) — narrower, easier to defend. The FrugalGPT-style cascade
      (condition 8 / Phase 5a) stays an optional side baseline, not core
      scope. This also means Phase 5's "full system" framing is
      de-prioritized; keep the paper's central claim to the complexity
      sensor + fuzzy controller routing across precision tiers.
- [x] Skip installing RouteLLM / FrugalGPT for now — neither is required
      for the core pipeline (see below, Phase 5b/Phase 8), and the chosen
      scope (fuzzy-router-alone) makes FrugalGPT's cascade optional at
      most, not planned.

**Gate:** ✅ **Phase 0 complete as of 2026-08-13.** All items done: bugs
verified fixed, platform locked (Kaggle T4, 30h available), token/license
set up, venue (TMLR) and scope (fuzzy-router-alone) decided. One thing to
carry into Phase 1: use the corrected `os.environ["HF_TOKEN"] = ...`
snippet above in the first cell of the Session 1 notebook — this hasn't
been exercised end-to-end yet, so confirm it works (no `401`) before
trusting the rest of the session.

---

## Phase 1 — Session 1: Energy ground truth (GPU, ~2-3h) — THE GO/NO-GO GATE

Goal: find out, on real hardware, whether precision even affects energy for
a 1B model. Everything downstream depends on this.

**STATUS 2026-09-04: dry run PASSED, real run NOT YET SUBMITTED.**
See `SESSION_LOG_2026-09-04.md` for the full account. Headline: on a
10-prompt smoke test the three tiers came out at 1.441 / 3.369 / 6.713
J/token with non-overlapping 95% CIs — **the go/no-go premise holds**, and
the Kaggle 8-bit hang did NOT reproduce. Those numbers are a smoke test
(n=30/tier), not a result; the 500-prompt run is the real one.

Note: the platform is the SPIT cluster, **not** a T4 — and the GPU is an
`NVIDIA RTX 6000 Ada Generation`, not the RTX A6000 `CLUSTER_MANUAL.md`
originally claimed.

- [x] **Done 2026-09-04, on-cluster.** Ran `prepare_eval_dataset.py` to build
      the real, deduped, stratified dataset — replaces the 30-prompt sample
      used for earlier calibration. **500 prompts, seed=42**, 200 easy /
      150 medium / 150 hard, from TriviaQA + Alpaca + GSM8K + CodeAlpaca-20k,
      no dedup failures and no repetition-padding. The laptop's copy was
      stale; this regenerated one is authoritative. Gitignored, so it lives
      on the cluster only — reproducible anywhere with the same seed.
- [x] ~~Upload deduped `eval_prompts.jsonl` to the Kaggle dataset.~~
      **Obsolete** — Kaggle is no longer the platform. On the cluster the
      file is copied into the job's working directory instead (the script
      resolves it relative to cwd).
- [ ] Run `training/scripts/kaggle_energy_benchmark.py`:
  - NVML `nvmlDeviceGetTotalEnergyConsumption` (hardware counter, mJ)
  - 5 warmup generations, then N≥300 prompts per tier, greedy decoding,
    `max_new_tokens=128`, real generated-token counts
  - One tier in GPU memory at a time, `torch.cuda.synchronize()` around timing
  - Record `hardware_info.json` (GPU model, driver, clocks, idle-power baseline)
- [ ] Run `training/scripts/verify_results.py` against the output.
- [ ] Append a row to `paper/results.md` with the verdict.
- [ ] Update `CREDIBILITY_REPORT.md` §1 row for "Per-tier J/token on T4".

**Gate — go/no-go decision (RESEARCH_PLAN.md §2):**
- If 4-bit/8-bit show real, non-trivial energy savings over fp16 →
  **proceed to Phase 2 as planned.**
- If savings are negligible or negative → do not proceed on the original
  plan. Pick one:
  1. Swap low tiers to GPTQ/AWQ checkpoints with real low-bit kernels
     (ExLlama/Marlin) and re-run Session 1, **or**
  2. Reframe the energy axis as *energy per request* (batching headroom),
     **or**
  3. Pivot the paper toward the measurement study itself (RQ1) plus
     routing framed as a latency/cost result instead of an energy result.
- Either way: **this negative/modest result is itself publishable and
  honest** — do not suppress it, write it up (see Phase 7, "Section 2 risk"
  becomes a real finding, not a hypothetical).

---

## Phase 2 — Session 2: Per-tier accuracy + the mechanistic correlation study (GPU, ~4-6h)

**STATUS: UNBLOCKED 2026-09-05 (evening) — ready to submit, not yet run.**

The `srun` diagnosis below was wrong. Jobs 1494-1507 that evening all ran
`python` DIRECTLY inside `sbatch` with no `srun`, and the venv resolved every
time. For a single-node single-task job `sbatch` has already allocated the
node — `srun` was never needed. A second, independent cause of the same
"installed but the job can't see it" symptom: the venv is Python 3.11 built
with `uv venv` and has **no pip**, so bare `pip` fell through to the system
Python 3.9. Install with `uv pip install`, never `pip`.

`kaggle_accuracy_eval.py` also carried the same 8-bit `torch_dtype` bug and
Kaggle-only adapter path found in the routing script — both fixed. Submit with
`training/scripts/session2_accuracy.sh` and WATCH it: the script has never
completed a run. Untested risk: lm-eval downloads its task datasets from
HuggingFace, which needs outbound internet from the compute node.

The original diagnosis is kept below for the record.

**Summary:** Attempted 4 job submissions on college GPU cluster; all failed before reaching evaluation code due to SLURM srun environment isolation preventing venv activation + Python discovery. Root cause: **not a code issue** — the evaluation script itself is sound (memory fixes applied to `kaggle_accuracy_eval.py` are correct). Next session must choose one of 4 recovery paths: (1) interactive GPU access, (2) SLURM admin config fix, (3) container submission, or (4) Kaggle/AceCloud fallback. See SESSION_2_BLOCKER.md §"Recommendations for Session 2 Retry" for details.

Goal (original): get real accuracy numbers per tier, **and** collect the evidence that
answers "why should prompt complexity predict quantization sensitivity?" —
this is the causal backbone the paper currently lacks.

- [x] `uv pip install lm-eval` — done 2026-09-05, lm_eval 0.4.13 confirmed
      importable on the venv interpreter. NOT `pip install`, see above.
- [ ] Evaluate each tier (fp16 / 8-bit / 4-bit) × (with / without QAT
      adapters) on tinyMMLU + GSM8K subset + HellaSwag subset.
- [ ] Save per-task, per-prompt JSON results (need per-prompt granularity,
      not just aggregate accuracy, for the correlation study below).
- [ ] Run `verify_results.py`; append to `paper/results.md`;
      update `CREDIBILITY_REPORT.md`.
- [ ] **New analysis (local, CPU, no extra GPU cost — reuses this session's
      output):** for every eval prompt, compute the accuracy drop from
      fp16→4-bit, and correlate it (Spearman) against each of the 5
      complexity features (Flesch-Kincaid, token length, entropy, syntax
      depth, has_code_or_math) individually. Produce a correlation
      heatmap / per-feature scatter. This is the evidence that justifies
      *why* the complexity sensor's specific features were chosen — write
      it up as new `paper/results.md` row + a figure for §3.1 of the draft.

**Gate:** RQ4 answerable from `accuracy_summary.csv`; correlation figure
exists and is at least directionally sensible (if it isn't — if complexity
features don't correlate with quantization sensitivity at all — treat that
as a serious finding requiring a rethink of the sensor before Session 4,
not something to bury).

**~~To unblock~~ (superseded 2026-09-05):** none of the four recovery paths
were needed — drop `srun`, install with `uv pip`. Original text: Once Python is runnable on GPU, this session will take 4–6 hours per the original estimate. Do not attempt more wrapper-script or PATH variations — the issue is environmental isolation in srun, not code or configuration.

---

## Phase 3 — Session 3: QAT adapter check — LOAD-TEST GATE PASSED 2026-08-22 (off-campus, CPU; GPU only needed if retraining)

Goal: confirm the three existing LoRA adapters (`adapters/adapter_{simple,medium,complex}`)
are usable, before spending more GPU time.

- [x] **Partially done 2026-08-22, off-campus (no GPU/HF_TOKEN available
      locally):** static/offline verification of the three adapters, ahead
      of the full on-GPU load test.
      - `PeftConfig.from_pretrained()` parses cleanly for all three against
        the locally installed `peft==0.19.1` (matches the adapters'
        recorded `peft_version`). `base_model_name_or_path` confirmed
        `meta-llama/Llama-3.2-1B` for all three, consistent with
        `config.yaml`.
      - `adapter_model.safetensors` opens cleanly for all three: 64
        tensors each (16 transformer layers × {q_proj, v_proj} ×
        {lora_A, lora_B}), all bfloat16, shapes consistent with
        Llama-3.2-1B's architecture (2048 hidden dim; v_proj's 512-dim
        output matches its GQA kv-head sizing).
      - **Confirmed the tier→adapter mapping** (from
        `kaggle_routing_experiment.py`/`kaggle_accuracy_eval.py`):
        `adapter_simple`→4bit, `adapter_medium`→8bit,
        `adapter_complex`→16bit. LoRA rank scales inversely with
        precision — r=16/α=32 (simple/4bit), r=8/α=16 (medium/8bit),
        r=4/α=8 (complex/16bit), constant α/r=2 across all three. This is
        the sensible direction: the most aggressively quantized tier gets
        the most adapter capacity to recover accuracy, the least-quantized
        tier needs the least. Not a red flag — a design choice worth
        stating explicitly in §3.4 of the paper.
- [x] **Full real load test completed 2026-08-22, off-campus, CPU (not
      GPU — a 1B model doesn't need one for a correctness check, just
      slower).** User provided `HF_TOKEN` (via a Windows user-scope env
      var, bridged into the test process from the registry — never pasted
      into the conversation). Downloaded the real gated
      `meta-llama/Llama-3.2-1B` weights (~141s on CPU) and ran
      `PeftModel.from_pretrained(base_model, adapter_path)` for real, for
      all three adapters, each followed by one short greedy generation
      (`"The capital of France is"` → `" Paris. It is the most populous
      city in France and the"`). **All three attached and generated
      without error**, base model included as a control. Identical output
      across base + all 3 adapters is expected, not a red flag: trivial
      factual completion, greedy decoding, small targeted LoRA deltas —
      no reason to expect a flipped argmax path on a case the base model
      already gets right. This is the real thing, not a proxy: adapters
      are now genuinely confirmed loadable and functional, ahead of any
      cluster access.
- [ ] Only retrain via `training/scripts/adapter-training.ipynb`
      if Phase 2's Session 2 results show them underperforming plain
      post-training quantization (the "fail to load" trigger no longer
      applies — they're confirmed loadable now). **Correction 2026-08-22:**
      previously documented as `backend/src/green_weight/training/
      kaggle_qat_trainer.py` — that script is confirmed **stale and
      unused**. Diffed its `LoraConfig` calls against the real adapters'
      `adapter_config.json`: they don't match (r=32/16/8 with 7 target
      modules and `adapter_4bit/8bit/16bit` naming here, vs. the real
      r=16/8/4 with `target_modules=[q_proj, v_proj]` and
      `adapter_simple/medium/complex` naming). `adapter-training.ipynb`
      matches the real adapters exactly and is the actual script used —
      confirmed by diffing its inline `train_adapter(..., r=16, alpha=32)`
      /`r=8/alpha=16`/`r=4/alpha=8` calls. `kaggle_qat_trainer.py` and its
      notebook counterpart `major-project-v2.ipynb` were never archived to
      `_legacy/` the way the old `core/`/`controllers/` implementation
      was, so they still read as live — this was a real, previously
      unnoticed documentation risk: if adapters had ever needed retraining
      before this correction, the wrong script would have been followed,
      producing adapters that don't match the ones already verified
      loadable. Fixed all references across `CLAUDE.md`, `RESEARCH_PLAN.md`,
      `KAGGLE_MANUAL.md`, and `training/configs/README.md`; added a
      prominent stale-script warning header to `kaggle_qat_trainer.py`
      itself.
- [ ] If retrained: re-run the relevant slice of Session 2 for the new adapters.

**Gate:** ✅ adapters confirmed loadable (real load test, not just static
checks, 2026-08-22). Re-verification against Session 2 accuracy numbers
still pending — that's the "do they actually help vs. plain PTQ" question,
separate from "do they load."

---

## Phase 4 — Eval-set finalization & recalibration (no GPU / light CPU) — CORE ITEMS DONE (started 2026-08-13, resolved 2026-08-22, both while blocked on cluster access; one new non-blocking item found)

Goal: move off the 30-prompt calibration sample before the main experiment.

- [x] Confirm the 500-prompt stratified set exists (easy 200 / medium 150 /
      hard 150) from Phase 1's `prepare_eval_dataset.py` run. Confirmed:
      200/150/150, verified unique.
- [x] Re-validate the fuzzy controller's feature-normalization bounds
      against the full 500-prompt set. Done for `ENTROPY_RANGE` (3.0,4.5)
      → **(3.3,5.0)** and `SYNTAX_DEPTH_RANGE` (1,8) → **(2,14)** in
      `router/complexity_scorer.py` — old bounds were clipping real data
      (entropy max 4.66 > old 4.5; syntax_depth max 12 > old 8, hitting
      exactly the hardest ~5% of prompts). Verified working (re-scored
      test prompts, no errors).
      **Finding worth carrying into the paper:** per-difficulty analysis
      shows `entropy` barely discriminates difficulty in this dataset
      (easy mean 4.11 vs hard mean 4.11 bits — nearly identical; medium
      3.96). `syntax_depth` does discriminate (easy 4.42 / medium 4.46 /
      hard 5.57). Range-widening fixes clipping but doesn't fix entropy's
      weak signal — that's a candidate finding for the Phase 5c feature-
      ablation study, not something a calibration tweak resolves.
      **New finding, not previously flagged:** `token_length` normalizes
      against a hardcoded max of 512 tokens, but the real 500-prompt set's
      max is only ~154 tokens (p95 = 62). This feature can barely ever
      cross the MID threshold (0.2 → 102 tokens) and never reaches HIGH
      (0.8 → 410 tokens) — it's structurally near-dead as a routing signal
      right now. **Not yet fixed — open decision, see below.**
- [x] **Decided and applied 2026-08-22 (off-campus session, no GPU access):**
      applied both proposed breakpoint updates in `config.yaml` —
      `syntax_depth_breakpoints`: `[3,4,6]` → `[4,5,8]` and
      `entropy_breakpoints`: `[3.3,3.7,4.1]` → `[3.8,4.1,4.5]`.
      **Correction to the original framing:** the entropy update was not
      merely cosmetic — leaving `[3.3,3.7,4.1]` in place against the
      widened `ENTROPY_RANGE=(3.3,5.0)` was a real bug: `bps[0]` (3.3)
      exactly equals the range floor, so
      `normalize_to_01(3.3, 3.3, 5.0) = 0`, collapsing the "low" membership
      triangle to zero width (`fuzz.trimf([0,0,0])`). The weak-signal
      finding (entropy barely discriminates difficulty) still stands and
      is still deferred to Phase 5c — this fix only resolves the
      degenerate-membership bug. The syntax_depth update was needed for
      the reason originally stated: leaving `[3,4,6]` against the widened
      `SYNTAX_DEPTH_RANGE=(2,14)` compressed lo/hi into the bottom ~17% of
      the [0,1] space, pushing nearly all real prompts into the "high"
      bucket regardless of actual difficulty.
- [x] **Decided and applied 2026-08-22:** fixed `token_length`'s
      normalization cap. Computed exact stats directly from the real
      `data/eval_prompts.jsonl` (500 prompts, confirmed to exist locally):
      min=5.25, max=153.75, p50=16.25, p90=48.5, p95=62.0, p99=109.25
      approx-tokens — matches the "~154 / p95=62" estimate exactly. Added
      `TOKEN_LENGTH_RANGE = (0, 154)` to `router/complexity_scorer.py`
      (same pattern as `ENTROPY_RANGE`/`SYNTAX_DEPTH_RANGE`), replacing the
      hardcoded `normalize_to_01(approx_tokens, 0, 512)`. Also updated
      `config.yaml`'s `max_token_length` 512 → 154 for documentation
      accuracy (confirmed via grep this field is not read anywhere at
      runtime — cosmetic-only fix, no behavior change from that edit
      alone; the real fix is the code constant).
- [x] Re-ran the fuzzy controller on 5 known prompts post-recalibration
      (trivial factual, summarization, code-generation, long technical
      explanation, factual). Sane: trivial prompts → 4bit, code-generation
      → 16bit, feature values now spread across the real 0–1 range instead
      of clustering near 0 as before the token_length fix. No crashes.
- [x] **Resolved 2026-08-22** (same session): `_trimf_low_mid_high()`'s
      dead-3rd-breakpoint issue. Considered implementing true 3-breakpoint
      asymmetric partitioning, but that's a real behavior change to every
      feature's membership shape (e.g. it would shift where syntax_depth's
      "high" region starts from raw depth 5 to raw depth 8) and would
      additionally require re-tuning `flesch_kincaid_breakpoints` (its old
      3rd value, 18, sits exactly on `FLESCH_KINCAID_RANGE`'s ceiling —
      same class of zero-width degenerate-membership bug just fixed for
      entropy) plus re-validating tier thresholds against the shifted
      output distribution. **Went with the lower-risk option instead:**
      dropped the dead 3rd value from all four `config.yaml` breakpoint
      lists (`flesch_kincaid_breakpoints` [6,12,18]→[6,12],
      `token_length_breakpoints` [0.2,0.5,0.8]→[0.2,0.5],
      `entropy_breakpoints` [3.8,4.1,4.5]→[3.8,4.1],
      `syntax_depth_breakpoints` [4,5,8]→[4,5]) and documented in both the
      config comments and `_trimf_low_mid_high`'s docstring that only 2
      edge breakpoints are used (MEDIUM's peak is their auto-computed
      midpoint). Zero behavior change by construction — re-ran the same 5
      sanity-check prompts, `win_probability` outputs identical to before
      this edit. True 3-breakpoint partitioning remains an option to
      revisit during Phase 5c if the feature-ablation study wants finer
      control over membership shape.

**Gate:** ✅ calibration re-validated against the full set, documented in
`CREDIBILITY_REPORT.md` (2026-08-22). Phase 5 is not blocked by the
remaining `_trimf_low_mid_high` item above — that's a decision to make
before Phase 5c specifically, not before Phase 5 starts.

---

## Implementation hardening — 2026-08-22, same off-campus session (not GPU-gated)

Goal: user asked to finish real implementation gaps before starting Phase 7
(paper drafting). Ran a codebase-wide survey (not previously done in this
roadmap) specifically for genuinely unfinished/broken code — not "not yet
measured," which is expected — in `api.py`, `dynamic_inference.py`,
`cascade/`, `evaluation/benchmark.py`, `frontend/`, `backend/tests/`.

- [x] **Fixed: `backend/src/green_weight/benchmark/accuracy_eval.py` was
      fabricating accuracy numbers.** `_evaluate_task_fixed_tier()` and
      `_evaluate_task_routed()` never called `lm_eval` at all — they
      returned hardcoded dicts (e.g. `{"mmlu": {"4bit": 0.42, "8bit": 0.58,
      "16bit": 0.72}}`) regardless of input, and
      `compute_routellm_metrics()` hardcoded `cpt = 0.5`. This is the live
      demo/API path (`run_pipeline.py` Phase 5) — **not** the real Session
      2 script (`training/scripts/kaggle_accuracy_eval.py`, confirmed
      still genuinely calls `lm_eval.simple_evaluate()`), so this bug never
      touched any paper number, but running `run_pipeline.py` outside
      dry-run would have silently written fabricated-looking numbers to
      `accuracy_results.json`. Rewrote the file to wrap the already-loaded
      tier models (via `models/model_pool.py`'s registry) in lm-eval's
      `HFLM` and call `lm_eval.simple_evaluate()` for real, matching the
      pattern already proven in `kaggle_accuracy_eval.py`. Added a new
      `RoutedLM` class implementing lm-eval's `LM` interface that performs
      genuine per-request routing: each request's prompt text is scored by
      the fuzzy controller, dispatched to that tier's `HFLM`, and
      reassembled in original order (grouped-batch dispatch, not one call
      per request). `compute_routellm_metrics()`'s CPT is now the real
      fraction of routed-condition calls sent to the 16bit tier (recorded
      during evaluation), replacing the placeholder — documented that this
      isn't a RouteLLM-style learned threshold, since this system doesn't
      have one (`routellm_bridge.py` is an intentional pass-through).
      Every failure path now raises loudly (`RuntimeError`) instead of
      defaulting to `0.0` or an empty dict — consistent with this
      project's core rule that nothing should imply a result exists until
      it's actually measured.
      **Verified without GPU** (no CUDA/lm-eval-on-real-model available
      locally): module imports cleanly, `lm_eval` installed locally and
      confirmed to expose the expected `LM`/`Instance` API used here;
      `AccuracyEvaluator()` constructs and correctly raises `RuntimeError`
      when no model is loaded (rather than returning fake data);
      `RoutedLM`'s dispatch/grouping/reassignment logic unit-tested against
      mock tier-LMs — confirmed order-preserving across mixed-tier batches,
      confirmed `_extract_scores()`'s metric-key preference and fallback,
      confirmed the not-loaded-tier fallback path warns and degrades
      instead of crashing. **Not yet verified: an actual `lm_eval` run
      against real loaded models** — needs the cluster; do this as part of
      Phase 5's Session 4, or as a quick standalone smoke test the first
      time GPU access is available.
- [x] **Fixed, found in the same pass: `models/model_pool.py`'s QAT
      adapters were silently never loading, in any run of the live
      pipeline, ever.** `load_pool()` computed `project_root =
      Path(__file__).parent.parent` (→ `backend/src/green_weight/`) and
      looked for `adapters/adapter_4bit`, `adapters/adapter_8bit`,
      `adapters/adapter_16bit` under it — but the real adapters live 3
      directories higher, at `major-project/adapters/`, under the names
      `adapter_simple`/`adapter_medium`/`adapter_complex` (see Phase 3
      above). Every `.exists()` check silently returned `False`, so
      `load_pool()` always fell back to `adapter_path=None` — the live
      pipeline has been running plain post-training quantization with no
      QAT adapter applied, with no error or warning, since this code was
      written. Fixed the path (`Path(__file__).resolve().parents[4]`,
      verified against the real directory layout) and the three adapter
      names to match `training/scripts/kaggle_routing_experiment.py`'s and
      `kaggle_accuracy_eval.py`'s existing tier→adapter mapping. Verified
      the corrected path resolves to real, existing directories on disk
      for all three tiers.
- [x] **Cleaned up: `backend/src/green_weight/run_pipeline.py:21-60`** had
      its entire import block pasted twice back-to-back (a botched-edit
      artifact, including a leftover placeholder comment `# ... (the rest
      of your file remains exactly the same)`). Harmless at runtime, but
      it's the main orchestrator entrypoint — removed the duplicate.
- [x] **Deleted 3 unused dead files in `frontend/`:** `api_updated.py`,
      `Playground_updated.jsx`, `fix_playground.py` — none referenced
      anywhere else in the repo (confirmed via repo-wide grep before
      deleting); `fix_playground.py` hardcoded a path
      (`c:\Users\Arwa\...\greenai-dashboard\...`) from a different
      machine/user that doesn't exist in this repo layout.
- [x] **Confirmed, left as-is (already correctly scoped as optional):**
      `cascade/frugal_cascade.py`'s escalation/judger logic is unbuilt —
      `run()` always does a single `get_completion()` call at the starting
      tier and hardcodes `"judger_score": 0.7`. Matches what `NEW.md`
      Phase 0 already documents as optional/deprioritized scope (condition
      5a-8) — not fixed, not a new finding, just confirmed still true.
- [x] **Fixed: `GpuEnergyMeter.info()` bytes-decoding crash risk in both
      `training/scripts/kaggle_energy_benchmark.py` (Session 1) and
      `kaggle_routing_experiment.py` (Session 4).** Neither decoded
      `pynvml.nvmlDeviceGetName()`/`nvmlSystemGetDriverVersion()`, even
      though the sibling script `check_nvml_energy_support.py` explicitly
      anticipates and handles these returning `bytes` on some pynvml
      versions. If that happens on the college cluster's pynvml version,
      `json.dumps(hw, ...)` would crash — in Session 1, right at the start
      of `main()`, before any measurement or `hardware_info.json` write;
      in Session 4, after Phase A+B have already spent hours of GPU time
      (though `routing_per_prompt.csv`/`routing_conditions_summary.csv`
      are written before that point, so only `routing_run_info.json` and
      the final summary print would be lost, not the core data). Fixed
      both with the same defensive decode `check_nvml_energy_support.py`
      already uses. Compile-checked; not runnable end-to-end without a
      GPU.
- [x] **`backend/tests/` test suite built, 146 tests, all passing.** User
      asked to keep going and build this too (superseding the earlier "not
      now" framing). Delegated to a `test-agent`-scoped subagent (isolated
      worktree). **Wrinkle:** since this repo isn't a git repository, the
      worktree isolation fell back to a plain directory snapshot taken at
      an earlier point — its own `CLAUDE.md`/`config.yaml`/application code
      were all a stale, pre-fix snapshot (predating even the 2026-08-05
      reorg in places). The subagent correctly detected this via a
      byte-for-byte diff, refused to touch application code per its scope,
      and wrote tests pinning down whatever was *actually* live in its
      worktree — including 3 `xfail(strict=True)` tests documenting bugs
      already fixed elsewhere, and a full `test_accuracy_eval.py` pinned to
      the OLD fabricating behavior. Reconciled by hand against the real
      (fixed) codebase: merged in the 5 files that were already fully
      compatible as-is (`test_verify_results.py`, `test_complexity_scorer.py`,
      `test_fuzzy_controller.py`, `test_routellm_bridge.py`,
      `test_model_pool.py` — the last of these already correctly asserted
      the real `adapter_simple/medium/complex` names, not the stale ones);
      added `TOKEN_LENGTH_RANGE` coverage to `test_complexity_scorer.py`
      (absent in the stale worktree since the constant didn't exist there
      yet); fixed 4 breakpoint-list assertions in `test_config.py` (stale
      3-value lists → real 2-value lists post-cleanup); deleted
      `test_known_issues.py` entirely (all 3 bugs it xfail-documented are
      fixed in the real repo, coverage now lives in the proper files
      instead); fully rewrote `test_accuracy_eval.py` against the real
      `RoutedLM`/`_extract_scores`/`RuntimeError`-on-missing-model behavior
      (25 tests, including dispatch-grouping, order-preservation, and the
      not-loaded-tier fallback path — formalizing the ad-hoc checks done
      manually earlier this session). **Verified: `python -m pytest
      backend/tests -q` → 146 passed, 0 failed, reproducible from a
      different cwd, no stray output in the real repo tree** (uses
      `tmp_path`/`monkeypatch.chdir` throughout).
- [x] **Fixed: `api.py`'s `/infer` endpoint could never run real GPU
      inference, ever, on any machine.** `_model_pool_loaded` was declared
      `False` at module level and never once set to `True` anywhere in the
      file (confirmed via repo-wide grep) — so `/infer`'s `elif
      _model_pool_loaded:` branch was permanently dead code, and every
      non-routing-only request silently fell through to the mock-response
      branch (honestly labeled `is_mock: True`, at least, unlike the
      accuracy_eval.py bug — but real inference was simply unreachable
      through the API regardless of GPU availability). `/health`'s
      `gpu_ready` field was permanently `False` too. Root cause: `startup()`
      only ever initialized the fuzzy controller/bridge, never called
      `models.model_pool.load_pool()`. Fixed by wiring `load_pool()` into
      `startup()`, gated behind an explicit `GREEN_WEIGHT_LOAD_MODELS=1`
      env var opt-in rather than loading unconditionally — `uvicorn
      --reload` restarts `startup()` on every file save, and eagerly
      loading a multi-GB model pool on each restart would wreck local
      dev ergonomics for anyone not actually testing real GPU inference.
      **Verified with FastAPI's `TestClient`** (no GPU available locally):
      default behavior unchanged (`gpu_ready: False`, `/infer` correctly
      mocks); with the env var set but no CUDA present, degrades
      gracefully (logs a warning, doesn't crash startup, still reports
      `gpu_ready: False`) — the real-GPU-and-opted-in path itself still
      needs a cluster to verify end-to-end.
- [x] **Frontend audit, 2026-08-22 (`frontend/`), user-requested follow-up.**
      Deep read of every `.jsx`/`.js` file (previously only checked for
      dead files + a light coherence pass). Found and fixed:
      - **`Playground.jsx` had real double-UTF-8-encoded mojibake baked
        into the file bytes** (confirmed via raw byte inspection, not a
        display artifact) — 7 distinct corrupted sequences, 14 occurrences
        total: `â€¦`→`…`, `âŒ˜â†µ`→`⌘↵`, `âš `→`⚠`, `â€”`→`—` (×3),
        `âœ•`→`✕`, `â”€â”€`→`──` (×2), `GÃ¶del`→`Gödel`. Fixed with a
        precise targeted-replacement script — deliberately did NOT do a
        blanket Latin-1/UTF-8 round-trip on the whole file, since that
        would have corrupted the many legitimately-encoded `·` (middle
        dot) characters used as separators throughout the same file.
        Scanned the rest of `frontend/src/**/*.{jsx,js}` for the same
        pattern — confirmed isolated to this one file.
      - **The exact same "Llama-2 vs Llama-3.2-1B" bug from the 2026-07-10
        backend audit, independently reintroduced twice in the frontend**:
        `Playground.jsx`'s "Active config.yaml" peek panel and
        `Settings.jsx`'s live-updating "config.yaml preview" both
        hardcoded `base_model_id: meta-llama/Llama-2-7b-hf` — the real
        config was fixed to `meta-llama/Llama-3.2-1B` back in the original
        2026-07-10 audit, but these two display-only strings were never
        updated to match. Fixed both.
      - **`api.py`'s `_append_trace()` didn't log `is_mock`**, so
        `pipeline_trace.jsonl` mixed real and mock `energy_joules` values
        with no way to tell them apart after the fact. Added the field.
      - **`Analytics.jsx`'s "Energy Saved vs 16-bit" KPI showed a
        fabricated "100% saved" result whenever no real energy had ever
        been measured** (e.g. pure routing-only usage, where every
        `energy_joules` is exactly `0`) — dividing by a nonzero traces
        count with `totalJ=0` always evaluates to 100%. This is exactly
        the class of bug this whole session has been hunting: implying a
        result exists when nothing was measured. Fixed: now filters to
        `!is_mock && energy_joules > 0` traces, and shows `—` / "no real
        measurements yet" when there are none, using the new `is_mock`
        field above.
      - **`Analytics.jsx`'s "Energy – Accuracy Trade-off" scatter chart
        labeled itself `"live"` whenever real accuracy data existed, but
        its energy axis is *always* the same hardcoded demo constants**
        (`36.8/8.5/28.5/105`J) regardless — `accuracy_results.json` only
        carries accuracy per condition, not energy, and no live
        per-condition energy endpoint exists yet (needs Session 4's
        `routing_conditions_summary.csv`, not built yet). This matches an
        *already-documented* known risk in
        `verification/checklist.md` about these exact mock placeholder
        values (`8.5/28.5/105.2`) leaking into anything presented as real.
        Relabeled to `"accuracy: live · energy: modeled (Session 4
        pending)"` instead of a flat, misleading `"live"`.
      **Verified all of the above for real, not just by reading code:**
      `npm install` + `npm run build` succeed cleanly (2728 modules, no
      errors); started the actual Vite dev server and drove it with the
      Browser tool. Playground's fixes confirmed directly (page loads by
      default). Settings/Analytics needed a workaround: the tool's browser
      tab runs with `document.hidden = true` (not composited), which
      pauses the requestAnimationFrame-driven animation-completion
      detection `AnimatePresence mode="wait"` depends on to swap pages —
      sidebar nav clicks correctly updated React state (confirmed via the
      active-nav-pill class) but the page transition itself never
      completed, purely a test-environment artifact, not an app bug
      (real, visible browser tabs run rAF normally). Worked around it by
      temporarily setting each page as the *initial* mount (no exit
      animation to wait for) instead of navigating via click — confirmed
      Settings' `base_model_id: meta-llama/Llama-3.2-1B` and Analytics'
      `—` / "no real measurements yet" both render correctly, then
      reverted the temporary change.

## Frontend visual polish pass — 2026-08-22, same session, user-requested

Goal: user asked whether frontend *design* work (not just bug fixes) could
also happen now, no GPU needed. Scoped to a visual polish pass (spacing,
typography, color consistency, responsiveness) — no new features, no
redesign. Design system itself (`index.css`) was already solid going in:
proper token scales (color, radius, shadow, transition), intentional
font pairing (display/body/mono) — this pass was about finding real
inconsistencies, not rebuilding anything.

- [x] Verified color/typography consistency: zero hardcoded hex colors
      anywhere outside `index.css` (grepped all of `frontend/src/**/*.css`)
      — every color in every page/component already goes through the CSS
      variable system. Nothing to fix here.
- [x] **Found and fixed a real layout bug**: `Playground.jsx`'s input-card
      footer (offline-warning text + "Run Pipeline" button) had no `gap`
      and no `flex-shrink: 0` on the button — reproduced visually at
      ~800-900px viewport width: the warning text wrapped to 3+ cramped
      lines and crowded directly against the button with near-zero
      breathing room. Fixed by adding `flex-wrap: wrap` + a real `gap` to
      `.input-card-footer` so the button now drops to its own line below
      the wrapped text at narrow widths instead of fighting for space in
      one cramped row — confirmed clean at 820px (button wraps below,
      readable) and unchanged at 1440px (still one row, no regression).
- [x] **Found and fixed a minor typographic issue**: `Settings.jsx`'s
      "Predicted Energy Savings" value wrapped mid-hyphen
      (`always-16-` / `bit` split across lines) at narrower widths.
      Wrapped the tier label in a new global `.nowrap` utility class
      (added to `index.css` alongside `.mono`/`.display`) so it now
      either fits whole on a line or wraps as a complete unit
      (`~51% vs` / `always-16-bit`) — confirmed visually.
- [x] Checked responsive behavior across the realistic desktop range
      (1024px-1920px) for all 3 pages (Playground, Analytics, Settings) —
      no other overlap/crowding bugs found. Confirmed via computed
      styles (not just eyeballing) that `.playground`'s grid and
      `.analytics`'s container both correctly fill available width at
      every width tested — the large empty space visible in some
      screenshots is the natural, expected empty state (no `result`/
      `history` card yet, since the demo backend isn't running), not a
      layout bug.
- [x] **Noted, not fixed:** zero `@media` queries exist anywhere in the
      codebase — there is no mobile/tablet responsive strategy at all
      (fixed-width 220px sidebar, fixed-width 340px right columns, no
      breakpoint-driven stacking). Judged out of scope for this pass:
      this is an internal research dashboard, not a public product, and
      building a real responsive system is a new capability, not
      "polish." Flagging in case that assumption is wrong.
- [x] **Interesting correction vs. the audit earlier this session**: this
      time, real sidebar-nav clicks (`.click()` via JS, same method as
      before) correctly triggered the `AnimatePresence` page transition
      and Analytics/Settings rendered directly — no repeat of the
      `document.hidden`/rAF-throttling stall that required the
      initial-mount workaround earlier. Possibly session-specific
      (repeated screenshot calls may have established real compositing
      this time); noting honestly rather than assuming either the
      earlier or this finding generalizes.
- [x] Verified: `npm run build` succeeds cleanly after all changes
      (production build, no errors).

## Frontend theme redesign — 2026-08-22, same session, user-requested

Goal: user pushback on the visual polish pass — the previous theme (near-
black background, neon-green glows, scan-line/pulse-ring HUD animations,
geometric `Syne` display font) read as futuristic/cyberpunk, contradicting
a *sustainable/green AI* research project's actual theme. User chose
"clean light/eco" over "organic dark" when asked to pick a direction.

- [x] **New design tokens in `index.css`** (propagates to nearly the whole
      app since every other file consumes these variables): warm
      cream/paper backgrounds (`#faf7f0` base, white surfaces) instead of
      near-black; warm charcoal-forest text instead of near-white; a moss/
      fern green primary accent (`#4f7942`) replacing the old neon
      `#22c55e`; secondary accents shifted to earth tones — terracotta,
      ochre, rust, muted teal — replacing indigo/amber/red/cyan (variable
      *names* kept as-is, e.g. `--indigo-400`, so every existing
      `var(--x)` reference still resolves; only the hues changed, to avoid
      a large, error-prone rename across every file). Shadows changed from
      dark-theme heavy blacks to soft warm-gray tints; `--shadow-glow`
      redefined from a literal neon blur to a gentle tinted lift.
- [x] **Display font swapped**: `Syne` (geometric, sci-fi-leaning sans) →
      `Fraunces` (warm, organic serif) — the single biggest lever away
      from "futuristic." Body (`Instrument Sans`) and mono (`JetBrains
      Mono`) kept — neither reads as sci-fi on its own, that came from the
      old dark+neon+glow combination, not the typefaces. Updated the
      Google Fonts `<link>` in `index.html` to match.
- [x] **Removed literal sci-fi/HUD motifs**: the SVG noise-grain texture
      overlay (`body::before` in `index.css`, would have looked like dirt
      on a light background anyway) deleted entirely; the scan-line sweep
      animation (`.scan-line`, a classic HUD-scanner effect on the
      "running" pipeline card) replaced with a calm pulsing bar (new
      `breathe` opacity keyframe) instead of a moving line.
      `pulse-ring`/`blink` kept — soft opacity/ring pulses read as normal
      UI feedback in any theme, the issue was never pulsing per se.
- [x] **Every hardcoded color reference updated to match**, not just the
      root tokens — grepped and fixed all of it: `Sidebar.css` (tier
      pills, status dots), `Playground.css` + `Playground.jsx`'s
      `TIER_META` (button glows, badges, complexity tags, error banner,
      mock-mode banner), `Analytics.css` + `Analytics.jsx`'s
      `TIER_COLORS` (chart grid strokes — flipped from white-based
      `rgba(255,255,255,0.04)` to dark-based, invisible-on-white
      otherwise — scatter point fills, error banner, highlighted row),
      `Settings.css` (savings banner). Final grep across all
      `.css`/`.jsx` files for the old palette's literal hex/rgba values
      came back clean.
- [x] **Caught and fixed a dark-theme-to-light-theme inversion bug**: two
      spots (`::selection` text color, `.savings-banner-val`) used
      `--green-300` (a light/medium shade) as *text* color — correct on
      the old dark theme (light text pops against dark bg) but low-
      contrast on the new light theme (light-ish green text on a near-
      white tinted background). Swapped both to `--green-600` (the
      darkest shade) for proper light-theme contrast. Checked every other
      usage of the tonal shade variables (`--green-300/500/600`) across
      the codebase for the same class of bug — the other three usages
      (a background gradient, a button hover background, a button hover
      background) were already directionally correct.
- [x] **Verified for real, not just by reading code**: ran the actual
      Vite dev server, drove it with the Browser tool at multiple
      viewport widths (1440px, 1024px, 820px — the last being where the
      responsive fix from the polish pass was found), screenshotted all
      three pages. Confirmed: cream/moss palette renders correctly
      throughout, Fraunces serif loads and displays on headlines
      (confirmed via computed `font-family`, not just visually), earlier
      responsive footer-wrap fix still holds with the new theme (no
      regression), config.yaml preview panels still show the correct
      `meta-llama/Llama-3.2-1B` (not reverted by the redesign).
      `npm run build` succeeds cleanly.

---

## Phase 5 — Session 4: Main routing experiment (GPU, ~4-6h × 3 runs on different days)

Goal: the actual contribution result — accuracy vs. energy across all
routing strategies, with the added baselines that make the "ours is
better" claim defensible.

### 5a. Core 8 conditions (already scoped in RESEARCH_PLAN.md §4)
- [ ] 1. Static fp16 (reference)
- [ ] 2. Static 8-bit
- [ ] 3. Static 4-bit
- [ ] 4. **Fuzzy router (ours)**
- [ ] 5. Random router, tier distribution matched to #4 (proves routing
      intelligence matters, not just tier mix)
- [ ] 6. Simple threshold on complexity score (proves fuzzy beats naive
      control — or admit it doesn't). **Bug found and fixed 2026-08-22
      (off-campus, before any Session 4 GPU time spent):**
      `training/scripts/kaggle_routing_experiment.py`'s `threshold_router`
      condition was tautological, not a real baseline. It fed
      `win_probability*100` — the fuzzy controller's own fully-defuzzified
      output — through the identical 33/66 cut points `FuzzyController`
      already applies internally, so `threshold_router`'s tier was
      mathematically guaranteed to equal the raw `fuzzy_tier` for every
      prompt (confirmed: 5/5 test prompts matched exactly before the fix).
      The script's own comment even said "instead of fuzzy membership,"
      contradicting what the code did. Fixed by adding
      `naive_complexity_score()` — a plain mean of the 5 raw features,
      computed without calling `FuzzyController` at all — and rewiring
      `threshold_router` to use it instead. Verified the fix actually
      breaks the tautology: naive tier now matches the fuzzy tier in only
      2/5 test prompts (real divergence, not coincidental agreement).
      Also added a `naive_complexity` column to `routing_per_prompt.csv`
      for auditability, matching the existing `fuzzy_tier`/`final_tier`
      transparency pattern. **User decision (2026-08-22):** chosen fix was
      "mean of raw features" over "single dominant feature" or "leave the
      code, just reframe the docstring" — see CREDIBILITY_REPORT.md for
      the full option comparison.
- [ ] 7. Oracle router (cheapest tier that still answers correctly —
      upper bound)
- [ ] 8. *(Optional)* FrugalGPT-style cascade (`cascade/frugal_cascade.py`)
      — only pursue if you `pip install` the FrugalGPT clone; not required
      for the core claim. Skip unless time/GPU budget allows.

### 5b. Added baseline: fuzzy vs. a trivial learned classifier
- [ ] Train a 1-layer logistic regression / small decision tree on the
      **same 5 features** → tier label, using this session's data.
- [ ] Compare against the fuzzy controller on the Pareto curve.
- [ ] This directly answers "why fuzzy logic and not just a classifier?" —
      report whichever way it lands:
      - fuzzy ≈ learned → argue interpretability/auditability (rule base
        is human-readable and editable; a logistic model isn't).
      - learned wins → report honestly; fuzzy becomes the interpretable
        baseline, learned router becomes a secondary contribution.
- [ ] Note: this is **separate** from the RouteLLM-style tier-preference
      router — that one (RQ3 addendum) is planned as a *post-Session-4*
      follow-up trained on this session's `routing_per_prompt.csv`, once
      it exists. RouteLLM's own pretrained checkpoints/library are not
      used anywhere in this pipeline — the clone in the repo root is kept
      only as citation/API reference for that future router's design.

### 5c. Feature ablation (ties back to Phase 2's correlation study)
- [ ] Drop each of the 5 features one at a time, re-run routing, measure
      Pareto-curve degradation.
- [ ] Cross-reference against Phase 2's correlation results: does the
      feature that hurts most when dropped match the feature that
      correlated most with quantization sensitivity? If yes, that's a
      tight, defensible causal story for the paper. If no, investigate
      before writing it up as settled.

### 5d. Logging
- [ ] Per-prompt log for every condition: tier chosen, energy, tokens,
      response, correctness — this is `routing_per_prompt.csv`.
- [ ] Router compute overhead measured and logged separately (it runs on
      CPU in milliseconds — quantify it, don't claim zero).

**Gate:** `verify_results.py` PASS on all conditions; ≥3 repeated runs on
different days; `paper/results.md` updated; `CREDIBILITY_REPORT.md` §1
updated for RQ2/RQ3.

---

## Phase 6 — Session 5: Figures, stats & credibility hardening (CPU, local, no GPU)

Goal: turn raw CSVs into the paper's actual evidence, plus the extra
robustness work that preempts reviewer pushback.

- [ ] Headline figure: accuracy vs. J/token Pareto plot, all conditions
      (300 dpi PNG + PDF).
- [ ] Tier-distribution bars per condition.
- [ ] Energy bars with 95% bootstrap CIs.
- [ ] Ablation table (feature ablation + QAT adapter ablation).
- [ ] Statistics: 3 seeds/runs, bootstrap 95% CIs, paired comparison
      between condition 4 (fuzzy) and condition 1 (static fp16) accuracy.
- [ ] **Threshold sensitivity sweep:** re-run routing decisions (not full
      GPU inference — this can reuse Phase 5's per-tier-per-prompt
      measurement grid, since greedy decoding is deterministic) at
      alternate breakpoints (e.g. 25/75, 40/60 instead of 33/66). Plot how
      much the Pareto curve shifts. Flat → robustness evidence. Steep →
      honest limitation, write into Threats to Validity.
- [ ] **Energy-accounting / break-even model:** formalize
      `E_total(tier) = E_router_compute + E_inference(tier)` using
      Phase 5d's router-overhead measurement; report the prompt-length
      break-even point below which routing isn't worth it.
- [ ] **Memory-footprint comparison (arithmetic only, no GPU needed):**
      base model + 3 LoRA adapters (this approach) vs. what hosting N
      separate models would cost (RouteLLM-style routing). This is an
      independent "why ours is better" axis that doesn't depend on the
      energy numbers landing favorably.

**Gate:** every figure/table traceable to a CSV with a PASS verdict; every
WARN from `verify_results.py` either resolved or written into Threats to
Validity.

---

## Phase 7 — Paper writing (gated section-by-section)

Goal: draft `paper/draft.md`, section by section, each gated by `verifier`
before being marked final — **no number enters the draft unless it has a
PASS row in `paper/results.md`.**

Writable now, no GPU data needed:
- [ ] §2 Related Work — RouteLLM / FrugalGPT / GPTQ-AWQ-LLM.int8-QLoRA /
      adaptive-computation / Zeus-MELODI-LLMCarbon positioning
      (RESEARCH_PLAN.md §1 has the differentiation already drafted).
- [ ] §3 System (3.1 Complexity Sensor, 3.2 Fuzzy Gearbox, 3.3 Precision
      Tiers, 3.4 QAT LoRA Adapters) — draftable from code inspection.
- [ ] §4 Measurement Methodology — draftable from `CREDIBILITY_REPORT.md`
      §3-4, finalize once Session 1 confirms NVML behavior on the real box.
- [ ] §6 Threats to Validity — draftable now from `CREDIBILITY_REPORT.md`
      §5, fold in real WARNs once sessions run.

Gated on GPU results existing:
- [ ] Abstract
- [ ] §1 Introduction
- [ ] §5 Results — RQ1 (Phase 1), RQ2 (Phase 5), RQ3 (Phase 5b + post-Session-4
      tier-preference router), RQ4 (Phase 2/3) — plus the new mechanistic
      correlation figure (Phase 2) and sensitivity/break-even results (Phase 6)
- [ ] §7 Conclusion

**Gate:** every section marked final only after `verifier` cross-checks its
claims against `paper/results.md` and `CREDIBILITY_REPORT.md`.

---

## Phase 8 — Internal review & polish (~1 week)

- [ ] Full draft read-through against `verification/checklist.md`.
- [ ] Advisor / internal review pass.
- [ ] Confirm no sentence in the paper contains a number that can't be
      traced to a committed CSV (`CREDIBILITY_REPORT.md` §7 sign-off list).
- [ ] Confirm `CREDIBILITY_REPORT.md` §1 has no claim left in
      UNVERIFIED/PENDING that also appears in the paper.
- [ ] Decide, based on actual results: was the FrugalGPT cascade baseline
      (5a-8) or RouteLLM clone ever actually needed in the end, or can
      both be dropped from the reproducibility artifact's dependencies?
      (Expectation: RouteLLM clone was reference-only throughout and can
      be dropped; FrugalGPT only stays if condition 8 was run.)

---

## Phase 9 — Submission & release

- [ ] Final venue CFP/deadline check (dates can move — re-verify before
      committing, per Phase 0's choice).
- [ ] Package reproducibility artifact:
  - all `training/scripts/*.py` (measurement, evaluation, routing,
    figures, validation), pinned seeds
  - raw per-inference CSVs (`energy_per_inference.csv`,
    `routing_per_prompt.csv`) + `hardware_info.json` per run
  - generated `results_validation.md` per run
  - `prepare_eval_dataset.py` with its seed (500-prompt set reconstructible)
- [ ] Submit to chosen venue.
- [ ] Release code + CSVs on GitHub.

---

## Quick reference — GPU budget

| Phase | Session | Est. GPU hours |
|---|---|---|
| 1 | Session 1 — energy ground truth | 2-3h |
| 2 | Session 2 — per-tier accuracy | 4-6h |
| 3 | Session 3 — adapter retrain (if needed) | 0 or ~6h |
| 5 | Session 4 — routing experiment, ×3 runs | 12-18h |
| 6 | Session 5 — figures & stats | 0 (CPU/local) |
| **Total** | | **~20-25h** (fits Kaggle free tier, ~30h/week) |

## Quick reference — what NOT to do

- Do not state "40% energy savings" or "<1% accuracy loss" anywhere — both
  are unverified legacy numbers from an assumed linear model, not measurement.
- Do not install/depend on RouteLLM's package for anything on this
  roadmap — it is citation/reference material only.
- Do not treat FrugalGPT as required — only install it if you commit to
  running condition 8 or the escalation extension.
- Do not let `paper-writer` cite anything not already a PASS row in
  `paper/results.md`.
