"""Evaluation and benchmarking module.

The former contents of this module (AccuracyBenchmark, EnergyBenchmark,
estimate_energy_joules) were System-B-only — built around
_legacy/core/dynamic_inference.py's DynamicInferenceEngine and a linear
bit-width energy model that CREDIBILITY_REPORT.md explicitly flags as
demo-only, never a measurement. They moved to
green_weight/_legacy/evaluation_benchmark.py along with the rest of System B.

Real accuracy/energy evaluation now lives in:
- benchmark/energy_tracker.py + benchmark/accuracy_eval.py (used by the live
  api.py/run_pipeline.py path)
- training/scripts/kaggle_*.py (the actual GPU measurement scripts)
"""
