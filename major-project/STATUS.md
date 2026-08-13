# Green-Weight — Status & Handoff

**Last updated:** 2026-08-05. This is a living doc — update it (or ask for
an update) after every GPU session or major code change, rather than
re-deriving "where are we" from scratch each time.

---

## 1. What's done

### 1.1 Repo restructure + 6-subagent Claude Code setup
- Physically reorganized `major-project/` (`git mv`, history preserved):
  `green_weight/` → `backend/src/green_weight/`, `greenai-dashboard/` →
  `frontend/`, `scripts/` + 4 training notebooks → `training/scripts/`.
- New: `backend/tests/`, `frontend/tests/`, `training/configs/`,
  `training/logs/`, `contracts/`, `verification/`, `paper/`.
- Every path reference updated across `CLAUDE.md`, `RESEARCH_PLAN.md`,
  `CREDIBILITY_REPORT.md`, `KAGGLE_MANUAL.md`, `SETUP_GUIDE.md`, and the
  scripts themselves. Verified: `frontend` still builds, `verify_results.py`
  still runs, the FastAPI backend still imports cleanly.
- **6 subagents** at `D:/Green AI/.claude/agents/`: `training-agent`,
  `backend-agent`, `frontend-agent`, `test-agent`, `paper-writer`,
  `verifier` — each with explicit scope boundaries (what it may/must never
  touch). Routing table and path-mapping in the root `D:/Green AI/CLAUDE.md`.
- `contracts/api-spec.yaml` (real API contract), `verification/checklist.md`
  (living sign-off checklist), `paper/results.md` (append-only experiment
  log — the only thing `paper-writer` may cite) + `paper/draft.md`
  (section skeleton, everything currently `PENDING`).
- Fixed stale/fabricated content found along the way: `SETUP_GUIDE.md` had
  fake example numbers ("65% energy savings...") presented like real
  results — replaced with `PENDING` templates.

### 1.2 Legitimacy audit: two competing implementations → one canonical
While explaining the project framework, found the complexity sensor +
fuzzy controller (the paper's actual contribution) existed as **two
divergent implementations**. Resolved:
- **Canonical now:** `router/complexity_scorer.py` + `router/fuzzy_controller.py`
  — a real Mamdani fuzzy-inference system (`scikit-fuzzy`), 5 features
  grounded in established metrics. This is what the FastAPI backend and
  `run_pipeline.py` actually use.
- **Archived:** the older, weaker implementation (`core/`, `controllers/`,
  `main.py`, `evaluation/benchmark.py` — hand-rolled magic-number heuristic,
  no real fuzzy library, was broken) → `backend/src/green_weight/_legacy/`,
  not deleted, with a README explaining why.
- **`routellm_bridge.py` fixed honestly**: removed dead/broken RouteLLM
  `Controller` code (it was never actually called even when it loaded).
  Now a documented threshold pass-through. RouteLLM's pretrained checkpoints
  can't be applied to our quantization tiers (they're trained on
  human-preference battles between specific named models) — a **real**
  tier-preference router is planned *after* Session 4 produces the
  comparison data needed to train one (see `RESEARCH_PLAN.md` RQ3).
- **Real bug fixed**: the fuzzy controller's breakpoint-normalization
  formula didn't match its own feature-normalization formula — silently
  misaligning `config.yaml` thresholds against real feature values. Fixed
  to share one source of truth.
- **Calibrated** (against the current 30-prompt sample — needs re-check
  once the full 500-prompt set exists): entropy and syntax-depth bounds
  were so wide that real prompts barely varied within them. Confirmed
  empirically the fix makes those features actually discriminate.
- **`kaggle_routing_experiment.py` rewritten**: no longer embeds a second,
  drifted copy of the routing logic — imports the real `router/` modules
  directly. The GPU script and the demo are now provably the same code.
- `CREDIBILITY_REPORT.md`, `RESEARCH_PLAN.md`, `verifier.md` updated to
  reflect all of the above honestly (including a new explicit "NOT REAL —
  documented pass-through" status row for the RouteLLM bridge, rather than
  a silent gap).

### 1.3 Verification performed
`git status` shows clean renames (not deletions); FastAPI backend imports
cleanly; the fuzzy controller + bridge produce correct, unchanged tier
decisions on test prompts; the rewritten routing script's import mechanism
was dry-run locally (score → fuzzy → bridge, no GPU needed) and works;
`verify_results.py` runs clean (still correctly reports no real data yet).

**Nothing has been committed or pushed.** Everything above is sitting in
your working tree.

---

## 2. What's next (the plan, in order)

1. **Session 1 — energy ground truth** (~2-3h GPU). The go/no-go gate: if
   4-bit doesn't save energy vs fp16, stop and reconsider before anything else.
2. **Session 2 — per-tier accuracy** (~4-6h GPU).
3. **Session 3 — QAT adapter retrain**, only if Session 2 shows the existing
   adapters fail to load or underperform.
4. **Session 4 — main routing experiment** (~4-6h GPU, ×3 runs on different
   days). Now measures the real, canonical routing logic.
5. **Session 5 — figures & stats** (local, no GPU).
6. **Post-Session-4: train the real tier-preference router** on
   `routing_per_prompt.csv` — this is what actually delivers on the
   RouteLLM-methodology comparison (RQ3), replacing the current
   placeholder bridge.
7. **Re-validate the feature-normalization calibration** once the full
   500-prompt eval set exists (currently based on a 30-prompt sample).
8. **`paper-writer` drafts sections**, gated by `verifier` on every section
   before it's marked final — only reading from `paper/results.md` and
   `CREDIBILITY_REPORT.md`'s VERIFIED rows.
9. **Still-open strategic decisions** (from `RESEARCH_PLAN.md` §8, not yet
   answered): which venue/deadline to target (workshop this cycle vs. TMLR
   later); explicit commitment to publish whatever the real numbers turn
   out to be, even if savings are modest.

Full step-by-step GPU commands are in the message where the runbook was
handed over (and in `KAGGLE_MANUAL.md`, paths already updated) — ask if you
want them re-posted.

---

## 3. What you have to do

- [ ] **Review the diffs** — nothing's committed yet; look over the changes
      (especially `_legacy/` archival and the `kaggle_routing_experiment.py`
      rewrite) before they go in, and tell me if/when to commit.
- [ ] **Run Session 1 on the GPU box** — you're driving this over SSH; I
      hand you exact commands, you run them and paste back output (NVML
      check, session log, results). Say the word when ready to start.
- [ ] **Confirm HF_TOKEN + Llama-3.2-1B license** are set up on the box
      before Session 1 (one-time, ~5 min on huggingface.co if not done).
- [ ] **After Session 1**: tell me the `verify_results.py` verdict — that's
      the go/no-go gate before any more GPU time gets spent.
- [ ] **Decide on venue/timeline** (§2, item 9 above) — whenever convenient,
      not blocking Session 1.
- [ ] **Eventually**: run `prepare_eval_dataset.py` to build the real
      500-prompt set (needs `pip install datasets` + internet) so the
      calibration and all downstream sessions use the full set, not the
      30-prompt sample.
