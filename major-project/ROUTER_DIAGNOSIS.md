# Why the fuzzy router underperforms

**Found:** 2026-09-08, CPU-only analysis of `routing_run1_per_prompt.csv`
(500 prompts, job 1505). No GPU required to reproduce.

`verify_results.py` reports that the fuzzy router does not beat a
tier-matched random control, and the Pareto plot shows it dominated by
static 8-bit on both axes. This document is the diagnosis. There are two
separate defects, and only the first has a cheap fix.

---

## Defect 1: the bridge escalates every undecided prompt to fp16

`router/routellm_bridge.py` refines the fuzzy controller's tier choice by
thresholding `win_probability`. Its mid-zone tie-break is:

```python
# MID zone: >= 0.5 goes to 16-bit (complex-leaning), < 0.5 stays 8-bit
if win_probability >= 0.5:
    return "16bit"
return "8bit"
```

**235 of 500 prompts (47.0%) score exactly 50.000** — `win_probability`
exactly 0.500. That is the fuzzy controller's neutral centroid, the output
it produces when no rule fires decisively. It is not a complexity estimate;
it is the absence of one.

The `>=` sends all of them to the most expensive tier. Every one of those
235 prompts had `fuzzy_tier == "8bit"`.

| Tier | controller decides | after bridge |
|---|---|---|
| 4-bit | 31.6% | 31.6% |
| 8-bit | **54.2%** | **7.2%** |
| fp16 | **14.2%** | **61.2%** |

235 prompts are overridden and every override is 8bit to 16bit. No other
override occurs in the whole run. The router's headline energy figure is
therefore substantially a measurement of this one tie-break.

### Cost of the escalation

Modelled at Session 1's uncontended per-tier rates (1.4138 / 3.0524 /
8.1244 J per token) applied to each prompt's own measured token count, so
the fp16 contention in run #1 does not distort the comparison:

| Configuration | J/request | accuracy |
|---|---|---|
| router as shipped (post-bridge) | 530.30 | 0.1300 |
| bridge escalation removed | **311.84** | **0.1520** |
| static_8bit | 315.75 | 0.1860 |
| static_4bit | 108.20 | 0.1100 |
| static_16bit | 800.56 | 0.1620 |

Removing the escalation cuts energy 41.2% **and** raises accuracy by 0.022.
The component is strictly harmful: it costs more and answers worse. On the
raw contended run-1 measurements the saving reads 52.2%, but that figure is
inflated because run #1's fp16 tier was measured under load; 41.2% is the
defensible number.

With the escalation removed the router is no longer dominated on both axes:
it sits level with static 8-bit on energy (311.84 against 315.75) while
still trailing it on accuracy.

### Fix

Change `>=` to `>`, or better, route the neutral mass explicitly. A score of
exactly 50.0 means "no rule fired", and the honest response to that is the
middle tier, not the most expensive one. **Do not apply this before deciding
the run schedule** — see "Consequences for the gate" below.

---

## Defect 2: the sensor does not discriminate, and fixing the bridge does not fix that

Recomputing the tier-matched random control against each distribution
(200 shuffles of the same tier multiset across prompts):

| Configuration | router acc | matched random acc | verdict |
|---|---|---|---|
| post-bridge | 0.1300 | 0.1475 | loses |
| bridge removed | 0.1520 | 0.1581 | still loses |

Removing the escalation narrows the gap from 0.018 to 0.006 but does not
close it. **Assigning the same tiers at random still scores as well as the
router's own assignment.** By the project's own criterion that is the check
which separates a routing policy from an arbitrary one, and it fails in
both configurations.

The 47% figure is the likely reason. If the sensor emits an identical
neutral score for nearly half the evaluation set, it cannot be carrying
information about those prompts, and no downstream threshold can recover
what was never measured. Supporting evidence: the sensor does track the
dataset's own difficulty labels in aggregate (mean complexity 35.6 easy /
41.1 medium / 58.1 hard) but with standard deviations near 20, so the
per-prompt signal is weak even where it exists.

This is the substantive research problem, and it is not a bug. Candidate
causes, in rough order of how cheaply they can be tested — all CPU-only:

1. Membership-function breakpoints in `config.yaml` are too narrow, so most
   feature values fall outside every band and no rule fires.
2. The rule base does not cover the region of feature space the real prompt
   distribution occupies.
3. The five features genuinely do not predict quantization sensitivity, in
   which case the negative result is the finding and the sensor needs
   rethinking rather than recalibrating.

Distinguishing (1)/(2) from (3) needs no GPU: the per-prompt correctness at
each tier is already in `routing_run1_per_prompt.csv`. Whether *any*
function of the five features predicts which tier a prompt needs can be
tested directly against that.

Caveat: correctness here is the placeholder reference-match proxy, which
scores 0.11-0.26 across conditions. A metric that coarse limits what any
correlation study can conclude, so Session 2 is a prerequisite for treating
(3) as settled.

---

## Consequences for the run schedule

Runs #1 and #2 measured the system *with* the escalation. Fixing it changes
the system under test, so:

- Fixing before runs #3-5 means those runs measure the corrected router,
  and runs #1-2 become the documented "before" case. This is preferable —
  the gate should certify the system the paper actually proposes.
- Fixing between gate runs would violate `SESSION_4_PLAN.md` note 4 ("do not
  mix runs from different code versions") and invalidate the gate.

Neither run #1 nor #2 counts toward the gate anyway, both having been
measured under host contention, so there is no sunk cost in fixing now.
