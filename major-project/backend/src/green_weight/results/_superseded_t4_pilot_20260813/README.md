# SUPERSEDED — Kaggle Tesla T4 pilot, 2026-08-13

Not Session 1. Do not cite these numbers.

- Hardware: **Tesla T4** (Kaggle), idle 10.41 W — see `hardware_info.json`
- Size: 3 runs x 12 prompts x 3 tiers = 108 rows
- Means (J/token): 4-bit 1.4207, 8-bit 6.4424, 16-bit 1.5064

Session 1 is the ground truth: RTX 6000 Ada, idle 22.53 W, 3 runs x 500
prompts x 3 tiers = 4,500 rows, means 1.4138 / 3.0524 / 8.1244 J/token.
It lives in `major-project/results/` and is the row `paper/results.md`
cites.

The tier ordering differs between the two because the hardware differs, not
because either is in error: on the T4 16-bit is cheap (1.51) and 8-bit
expensive (6.44); on the Ada 16-bit is by far the most expensive (8.12).
Kept as a cross-hardware datapoint.
