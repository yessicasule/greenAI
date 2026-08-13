# Legacy — not part of the live system

Moved here 2026-08-05 after a legitimacy audit found the project had **two
divergent implementations** of the complexity sensor + fuzzy controller:

- **`router/complexity_scorer.py` + `router/fuzzy_controller.py`**
  (canonical, live) — a real Mamdani fuzzy-inference system using the
  `scikit-fuzzy` library, 5 features grounded in established metrics
  (Flesch-Kincaid, spaCy syntax depth, Shannon entropy). Used by `api.py`
  (the FastAPI backend) and `run_pipeline.py` (the official full-evaluation
  entry point).
- **`core/prompt_complexity.py` + `controllers/fuzzy_gearbox.py`** (this
  folder) — a hand-rolled single-scalar heuristic with unexplained
  magic-number weights, fuzzy membership bolted on afterward via 3 fixed
  anchor points, no real fuzzy-logic library. Used only by `main.py` (also
  archived here), which was already broken (`BitWidth` imported from
  `config.py` but never defined there).

`evaluation_benchmark.py` (formerly `evaluation/benchmark.py`) is included
here too — it's built entirely around this system's `DynamicInferenceEngine`
and `BitWidth`, and contains the linear bit-width energy model
(`4-bit=0.25x, 8-bit=0.5x, 16-bit=1x`) that `CREDIBILITY_REPORT.md` already
flags as a demo-only assumption, never a measurement.

**Nothing here is imported by the live system.** Kept for history/reference,
not deleted. If you're looking for the real complexity sensor or fuzzy
controller, use `router/`. If you're looking for real energy/accuracy
measurement, use `benchmark/energy_tracker.py` + `benchmark/accuracy_eval.py`
or `training/scripts/kaggle_*.py`.
