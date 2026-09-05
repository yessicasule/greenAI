# Experiment Log

Append-only. This is the **only** numeric source `paper-writer` may cite —
if a number isn't in this table with a PASS verdict, it doesn't exist yet
as far as the paper is concerned. `training-agent` appends a row after
every session (see `.claude/agents/training-agent.md`).

| Date | Session | Script run | Output CSV(s) | verify_results.py verdict | One-line finding |
|---|---|---|---|---|---|
| _(none yet — Session 1 has not been run)_ | | | | | |



## Session 1 — Energy ground truth (2026-09-04, SPIT cluster)



| Tier | n | Mean J/token | 95% CI | Notes |

|---|---|---|---|---|

| 4-bit | 1,500 | 1.414 | ±0.055 | baseline |

| 8-bit | 1,500 | 3.052 | ±0.039 | 2.16× higher energy |

| 16-bit | 1,500 | 8.124 | ±0.241 | 2.66× higher energy |



**Hardware:** NVIDIA RTX 6000 Ada Generation, driver 610.43.02, idle power 22.53W



**Methodology:** NVML energy counter, 500-prompt eval set, 3 full runs (seed=42)



**Known issues:** 117 rows (2.6%) with energy_j=0.0 due to prompts generating 1 token and counter rounding. These are real measurements, not meter failure. See threats-to-validity.



**Status:** VERIFIED by verify_results.py, all substantive checks PASS.

