# Repo orchestration (Green-Weight)

**Start here every session: read `major-project/NEW.md` first.** It's the
live, checkbox-tracked roadmap (Phase 0-9, pre-flight through submission)
and the single source of truth for "what's done and what's next" — more
current than the narrative below, which describes the static repo
structure, not day-to-day status.

This repo root (`D:/Green AI`) holds the real project, `major-project/`
(Green-Weight — complexity-aware dynamic precision routing for LLM
inference), plus two unrelated reference clones (`FrugalGPT/`, `RouteLLM/`)
kept for citation/API reference only — they are not part of the 6-subagent
structure below and shouldn't be edited as part of this project's work.

Project-specific detail (architecture, research objectives, dependencies)
lives in `major-project/CLAUDE.md` — read that too. This file is the
orchestration layer: where things live post-reorg, and which subagent owns
what.

## Path mapping (authoritative)

The 6 subagents below were designed against a target tree that implied
`backend/`, `frontend/`, `training/` at a bare "project-root". Since this
repo root also contains `FrugalGPT/`/`RouteLLM/`, those directories were
created **inside** `major-project/`, not at the repo root. `.claude/` itself
stays at the literal repo root (`D:/Green AI/.claude/`) since that's where
Claude Code actually reads it from.

| Directory | What's there |
|---|---|
| `major-project/backend/src/green_weight/` | Main Python package (FastAPI `api.py`, `router/`, `cascade/`, `models/`, `benchmark/`, `results/`; `evaluation/` is a stub pointing at `benchmark/`). The old `core/`/`controllers/` (System B) are archived under `_legacy/`, not live. All internal imports are bare same-directory (`from config import get_config`) — **cwd must be `backend/src/green_weight/` itself, not `backend/src/`** (verified 2026-08-22; `-m green_weight.x` from `backend/src` does not work). |
| `major-project/backend/tests/` | pytest suite |
| `major-project/frontend/` | React 18 + Vite 5 dashboard (was `greenai-dashboard/`) |
| `major-project/frontend/tests/` | frontend test suite (not created yet) |
| `major-project/training/scripts/` | All GPU/session entrypoints: `kaggle_energy_benchmark.py`, `kaggle_accuracy_eval.py`, `kaggle_routing_experiment.py`, `prepare_eval_dataset.py`, `finetune_judger.py`, `verify_results.py`, `make_figures.py`, plus the 4 training notebooks |
| `major-project/training/configs/` | Intended home for externalized QAT hyperparams — not wired up yet, see its README |
| `major-project/training/logs/` | Raw GPU-session transcripts |
| `major-project/contracts/api-spec.yaml` | Backend↔frontend API contract |
| `major-project/verification/checklist.md` | Living sign-off checklist |
| `major-project/paper/results.md` | Append-only experiment log — the only numeric source `paper-writer` may cite |
| `major-project/paper/draft.md` | The paper draft |
| `major-project/dataset/`, `major-project/adapters/` | Left at top level, not moved — no clean slot in the target tree and moving them risked breaking existing relative-path assumptions |

Do not create a `backend/`, `frontend/`, or top-level `tests/` directly
under the repo root — those live under `major-project/`.

## Subagent routing

| Task | Agent |
|---|---|
| Running/monitoring GPU sessions, pulling results, updating `paper/results.md` | `training-agent` |
| FastAPI backend changes (`backend/src/green_weight/`) | `backend-agent` |
| Dashboard changes (`frontend/`) | `frontend-agent` |
| Writing/running tests | `test-agent` |
| Drafting/revising `paper/draft.md` | `paper-writer` (must invoke `verifier` before finalizing any section) |
| Cross-checking any claim against code/logs before it's stated as fact | `verifier` |

Full scope boundaries (what each agent may/must never touch) are in its own
`.claude/agents/<name>.md` — this table is just the routing index.

## The one rule that matters most

The project's bottleneck is **real measured data**, not code or paper
polish. Nothing produced by `backend-agent`, `frontend-agent`, or
`paper-writer` should imply a result exists until `training-agent` +
`verify_results.py` say so. Check `major-project/NEW.md` for which GPU
sessions have actually been run and verified, and
`major-project/CREDIBILITY_REPORT.md` §1 for the live status of every claim.
