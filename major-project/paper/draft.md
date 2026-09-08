# Complexity-Aware Dynamic Precision Routing for Energy-Proportional LLM Inference

> **Editorial note.** This is a markdown working draft. It is written to
> IEEE-conference conventions (numbered sections, numbered equations,
> lettered subsections, itemized contributions, keywords line) so that it
> converts cleanly to an IEEEtran LaTeX template at submission time; no
> attempt is made here to fake a two-column PDF layout in markdown. Section
> numbering follows Roman numerals (I–VII) per IEEE convention; the file
> previously used Arabic numbering (1–7) — this revision keeps the same
> section order and content but relabels headers for IEEE consistency.
> *Author names and affiliations are omitted from this working draft and
> will be added at submission.*

*Working title, per `RESEARCH_PLAN.md`. Every section is written strictly
against `paper/results.md` (the append-only, verifier-gated experiment
log) and `CREDIBILITY_REPORT.md` (VERIFIED rows only). A number not present
in `paper/results.md` with a PASS verdict does not appear in this paper.*

## Abstract

Large language model (LLM) inference is typically served at a single,
statically chosen numeric precision for an entire deployment. This paper
studies an alternative: routing each individual request, at inference
time, to one of several precision tiers of the *same* base model — chosen
per-prompt by a lightweight complexity sensor and a Mamdani fuzzy-logic
controller — rather than hosting multiple distinct models or committing to
one static bit-width. Unlike learned inter-model routers (e.g., RouteLLM),
this design requires only one set of base weights plus small per-tier
low-rank adapters resident in memory. We present the full system design
(complexity sensor, fuzzy inference system, precision tiers, quantization-
aware-trained LoRA adapters) with explicit mathematical formalization, and
a hardware-grounded measurement methodology built on GPU on-board NVML
energy counters together with an automated validation gate that must pass
before any number is eligible to be cited as a result. This paper's
empirical contribution at the present stage is a direct, hardware-measured
answer to the precondition the rest of the system depends on: does
energy-per-token actually differ across precision tiers on real hardware,
and by how much? Measuring Llama-3.2-1B on an NVIDIA RTX 6000 Ada
Generation GPU across 4-bit, 8-bit, and 16-bit tiers (n = 1,500 prompts per
tier, 3 repeated runs, seed = 42), we find mean energy of 1.414 ± 0.055,
3.052 ± 0.039, and 8.124 ± 0.241 J/token respectively, with non-overlapping
95% confidence intervals across all three tiers — a real, monotonic,
statistically separated energy gradient across precision. This result
passes this project's own pre-registered go/no-go gate for whether dynamic
precision routing is worth building at all. The routing energy/accuracy
tradeoff, baseline comparisons against naive routers, and a quantization-
aware-training adapter ablation remain open empirical questions at the
time of this draft: the measurement sessions that will answer them are
implemented and scripted but not yet executed. We report their exact
protocol so the remaining results can be added without redesigning the
system, and we are explicit throughout about what is and is not yet known.

**Keywords** — energy-proportional inference, dynamic quantization,
fuzzy logic control, LLM inference routing, green AI, hardware energy
measurement.

## I. Introduction

Serving LLM inference at a single, statically chosen numeric precision
treats every request identically regardless of how difficult it actually
is: a two-word factual query and a multi-step chain-of-reasoning math
problem pay the same per-token energy cost if both are served at, say,
16-bit. Prior work has separately explored (a) routing requests across
*different models* by predicted preference (RouteLLM), (b) cascading
across *paid API tiers* with a stop/escalate judger (FrugalGPT), and (c)
*static* post-training or quantization-aware compression of a single model
(GPTQ, AWQ, LLM.int8, QLoRA). None of these directly asks whether the
numeric precision of a single, already-resident model can itself be
treated as a per-request routing decision — chosen automatically from an
estimate of how complex a given prompt is, before generation begins. That
is the central design question of this paper's system, Green-Weight: a
lightweight, five-feature complexity sensor and Mamdani fuzzy inference
system route each prompt to one of three precision tiers (4-bit, 8-bit,
16-bit) of a single base model, so only one set of base weights plus small
per-tier LoRA adapters must be resident, rather than multiple hosted
models.

Before any such system's energy claim can be trusted, however, its
underlying physical premise has to be checked on real hardware rather than
assumed: does energy-per-token actually fall meaningfully as bit-width
drops for this model class, or does GPU dequantize-to-fp16 compute (as it
does for some quantization kernels) erase the expected savings? This
project's own history is a caution here — an earlier internal estimate of
end-to-end savings was derived from an assumed linear bit-width energy
model rather than measurement, and is explicitly retracted and not relied
upon anywhere in this paper (see `CREDIBILITY_REPORT.md` §2 for the full
account and `major-project/CLAUDE.md` for the standing project rule against
restating it). This paper's methodological stance is the direct response
to that history: every claim below is either (i) traceable to a specific
row of a verifier-gated, append-only experiment log, or (ii) explicitly
marked as not yet measured. We do not report an aggregate "energy savings"
or "accuracy loss" percentage anywhere in this paper, now or as a
projection, because no such number is licensed by measured data at the
time of writing.

Concretely, this paper reports:

1. A formal system design for complexity-aware dynamic precision routing
   — a five-feature complexity sensor (§III-A) and a Mamdani fuzzy
   inference system with an explicit rule base, aggregation operator, and
   centroid defuzzification (§III-B), both specified with numbered
   equations directly against the deployed implementation rather than
   described only in prose.
2. A hardware-grounded measurement methodology (§IV): GPU on-board NVML
   energy counters (not a software or emissions-derived estimate), a fixed
   greedy-decoding protocol, and an automated validation gate
   (`verify_results.py`) that every number in this paper must pass before
   it may be cited.
3. The first hardware-measured result in this research program (RQ1, §V-A):
   joules-per-token differs substantially and monotonically across the
   4-bit / 8-bit / 16-bit precision tiers of Llama-3.2-1B on an NVIDIA RTX
   6000 Ada Generation GPU, with non-overlapping 95% confidence intervals
   across all three tiers. This is this project's pre-registered go/no-go
   gate (`major-project/NEW.md`, Phase 1) for whether the rest of the
   routing system is worth evaluating, and it passes.
4. A fully specified, not-yet-executed remainder of the empirical program
   — routing energy/accuracy tradeoff (RQ2), baseline comparison against
   naive routers and an oracle (RQ3), and a QAT-adapter ablation (RQ4) —
   with the exact GPU sessions that will produce each (§V-B–D), so this
   paper's present evidentiary boundary is unambiguous to the reader.

**Contributions:**
- A complete mathematical formalization of the complexity sensor and
  Mamdani fuzzy routing controller (§III), derived directly from the
  deployed implementation rather than approximated.
- A formal energy-accounting model (§IV-B) that separates router compute
  overhead from tier inference cost, used as the basis for a
  break-even framing carried into future work (§VII).
- An automated, pre-registered validation gate that must pass before any
  number is eligible to appear as a result in this paper (§IV-D), and a
  full accounting of what has and has not yet passed it (§V).
- A hardware-measured, statistically significant, monotonic energy
  separation across three precision tiers of a real 1B-parameter LLM on a
  commodity workstation GPU (§V-A) — the empirical premise the rest of
  this research program depends on.
- Honest, explicit scoping of the three research questions (RQ2–RQ4) that
  remain open at the time of this draft, with no projected or implied
  outcome for any of them (§V-B–D, §VII).

## II. Related Work

*Note: bibliographic details (author lists, years, venues) below are given
to the best of the drafting agent's knowledge and should be checked against
primary sources before submission; none of this section depends on this
project's own measured numbers.*

### A. Learned Inter-Model Routing

RouteLLM (Ong et al., 2024) trains a preference-predicting router that,
given a prompt, decides whether to send it to a cheap "weak" model or an
expensive "strong" model, learned from human-preference battle data
between specific named model pairs (e.g., GPT-4 vs. Mixtral). Green-Weight
addresses a related but structurally different problem: rather than
choosing between distinct models that must each be hosted, it routes a
single prompt across precision tiers of one base model
(`meta-llama/Llama-3.2-1B`, §III-C), so only one set of base weights plus
small per-tier adapters (§III-D) needs to be resident. This sidesteps the
multi-model hosting cost central to RouteLLM's trade-off, but also means
RouteLLM's own pretrained routers cannot be reused as-is: its checkpoints
are trained on preference battles between specific named models drawn from
a closed model-identity set and carry no learned signal for "should this
prompt run at 4-bit, 8-bit, or 16-bit of the same model." Accordingly, the
RouteLLM-shaped component of this project's codebase
(`router/routellm_bridge.py`) is currently a documented, provisional
threshold pass-through over the fuzzy controller's own output (§III-B) —
it does not load or call a RouteLLM model, and this paper makes no claim
of a working RouteLLM integration. A real tier-preference router, trained
on this project's own per-prompt routing data once that data exists, is
planned future work and is the actual point of comparison to RouteLLM's
training methodology; it is not part of the present contribution.

### B. Cascade-Based Cost Reduction

FrugalGPT (Chen, Zaharia & Zou, 2023) reduces the cost of querying paid
third-party LLM APIs by cascading: starting with a cheap model and using a
learned generation judger to decide whether an answer is good enough or
whether to escalate to a more expensive model. This project's codebase
includes an analogous path (`cascade/frugal_cascade.py`) that wraps
FrugalGPT's own `LLMCascade` class around this project's local precision
tiers, but its escalation logic is not built out: the current
implementation performs a single call at the router-assigned starting tier
and returns a hardcoded judger score rather than actually invoking
FrugalGPT's judger and conditionally escalating to a higher tier. This
cascade path is treated as optional, deprioritized scope — the paper's
core claim is complexity-aware precision routing via the fuzzy controller
(§III-B), not the cascade — and, if included at all, serves at most as a
secondary baseline, not the core contribution.

### C. Static and Post-Training Quantization

GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2023), LLM.int8() (Dettmers
et al., 2022), and QLoRA (Dettmers et al., 2023) address a different
question: given a fixed precision budget, how to quantize (or fine-tune
under quantization) with minimal quality loss. Each picks one static
precision for an entire deployment. This project's approach is
complementary rather than competing: it uses bitsandbytes' NF4 4-bit and
int8 8-bit quantization — the same family of technique as
LLM.int8()/QLoRA — as two of its three precision tiers (§III-C), but
treats precision itself as a per-request routing decision rather than a
fixed deployment-time choice, and trains a separate QAT LoRA adapter for
each tier (§III-D) rather than a single adapter tuned for one quantization
level, applying QLoRA's adapter-based recovery strategy independently at
each of the three tiers.

### D. Adaptive Computation

This work sits inside the broader adaptive-computation paradigm —
early-exit networks, mixture-of-depths, speculative decoding, and other
methods that vary the amount of compute spent on a given input based on
its estimated difficulty. Where that literature typically adapts network
depth, number of decoding steps, or which draft/verify model pair is used,
this project adapts numeric precision: the same architecture and layer
depth run for every request, but the arithmetic precision of the weights
(and, correspondingly, memory-bandwidth demand) is selected per request
from a prompt-complexity estimate computed before generation begins
(§III-A).

### E. Energy and Carbon Measurement Tooling

Zeus (Chung et al., NSDI 2023) and LLMCarbon (Faiz et al., 2024) are
representative of a growing body of work on measuring and modeling the
energy/carbon footprint of deep learning workloads; MELODI-style GPU
energy-profiling tooling occupies similar territory (citation to be
confirmed before submission). This project follows their emphasis on
grounding energy claims in real measurement rather than an analytic model,
but differs in instrument: rather than software-level power estimation or
CO2-emissions-derived accounting (an earlier version of this project's own
pipeline mis-derived energy by inverting CodeCarbon's CO2 output with an
incorrect constant — see `CREDIBILITY_REPORT.md` §2 — a defect since
fixed), this project reads the GPU's onboard cumulative hardware energy
counter directly via NVML's `nvmlDeviceGetTotalEnergyConsumption` (§IV), a
hardware measurement rather than a software estimate. This is a narrower
but more directly verifiable instrument than a full carbon-accounting
framework: it does not model upstream grid carbon intensity or embodied
hardware carbon the way LLMCarbon does, and it reports GPU-board joules
per token only, explicitly excluding CPU/DRAM host energy (§IV, §VI).

## III. System

### A. Prompt-Complexity Sensor

The complexity sensor (`router/complexity_scorer.py`) converts a raw
prompt string $p$ into five features that the fuzzy controller (§III-B)
consumes, four of them normalized to $[0,1]$ and one binary:

1. **Flesch–Kincaid grade level** (via `textstat.flesch_kincaid_grade`),
   normalized against `FLESCH_KINCAID_RANGE = (0, 18)`.
2. **Token length**, approximated as prompt character count divided by 4
   (a standard English characters-per-token heuristic), normalized against
   `TOKEN_LENGTH_RANGE = (0, 154)`.
3. **Character-level Shannon entropy**, normalized against
   `ENTROPY_RANGE = (3.3, 5.0)` bits.
4. **Syntax-tree depth**, the maximum dependency-parse depth returned by
   spaCy's `en_core_web_sm` model, normalized against
   `SYNTAX_DEPTH_RANGE = (2, 14)`.
5. **`has_code_or_math`**, a binary (unnormalized) feature set by regex
   matching against markdown/fenced code blocks, inline code, LaTeX
   delimiters and commands, common mathematical operators (∑, ∫, ∂, √, ∞,
   and relational symbols), and a keyword list of natural-language
   code/algorithm requests (e.g. "implement," "algorithm," "recursion").

All continuous features share a single `normalize_to_01(value, min, max)`
helper that clips to $[0,1]$; the fuzzy controller (§III-B) reuses the
same range constants when normalizing its own membership-function
breakpoints, so the two stay aligned by construction rather than by
convention.

**Formal definition.** Let the normalization operator be

$$\text{norm}(x; a, b) = \text{clip}\!\left(\frac{x - a}{b - a},\ 0,\ 1\right) \tag{1}$$

used identically for every continuous feature below. The sensor produces a
5-dimensional feature vector for prompt $p$:

$$f(p) = [f_{FK},\ f_{TL},\ f_{H},\ f_{SD},\ f_{CM}] \tag{2}$$

where

$$f_{FK} = \text{norm}(FK(p);\ 0,\ 18),\quad FK(p) = \text{Flesch–Kincaid grade level of } p \tag{3}$$

$$f_{TL} = \text{norm}\!\left(\left\lfloor \frac{\text{len}(p)}{4} \right\rfloor;\ 0,\ 154\right) \tag{4}$$

$$f_{H} = \text{norm}(H(p);\ 3.3,\ 5.0),\quad H(p) = -\sum_{c \in \Sigma} p(c)\log_2 p(c) \tag{5}$$

with $H(p)$ the character-level Shannon entropy over the empirical
character distribution of $p$ ($\Sigma$ = the set of distinct characters
appearing in $p$);

$$f_{SD} = \text{norm}(D(p);\ 2,\ 14),\quad D(p) = \text{max. depth of the spaCy dependency parse of } p \tag{6}$$

$$f_{CM} \in \{0, 1\},\ \text{an indicator over the regex/keyword match described above.} \tag{7}$$

$f_{CM}$ is explicitly *not* passed through $\text{norm}(\cdot)$ — it is a
binary indicator, not a continuous normalized feature.

The range constants in Eqs. (3), (5), (6), (4) — $(0,18)$, $(3.3,5.0)$,
$(2,14)$, $(0,154)$ — are not arbitrary defaults; they are empirically
calibrated against this project's real 500-prompt stratified evaluation
set (200 easy / 150 medium / 150 hard prompts, seed = 42,
`data/eval_prompts.jsonl`), following an earlier 30-prompt calibration
pass. The original entropy and syntax-depth ranges clipped the real
distribution's upper tail — syntax depth in particular exceeded the old
bound for roughly the hardest 5% of prompts, collapsing them to an
identical flat 1.0 regardless of their actual relative difficulty — and
the original token-length normalization used a hardcoded 512-token cap
that the real prompt set never approached, leaving that feature
structurally unable to signal difficulty in practice.
`TOKEN_LENGTH_RANGE = (0, 154)` was derived directly from the observed
distribution of the real evaluation set. The values in Eqs. (3)–(6) are
the corrected, re-calibrated constants currently in use (see
`CREDIBILITY_REPORT.md`, row "Fuzzy controller's feature-normalization
bounds," for the full calibration history).

One caveat carried into the routing design rather than resolved by range
calibration alone: an early per-difficulty analysis of this dataset
suggested character-level entropy ($f_H$) discriminates prompt difficulty
only weakly, while syntax depth ($f_{SD}$) showed clearer separation
between easy and hard prompts. This motivates treating entropy as a
candidate for a feature-ablation study rather than assuming all five
features contribute equally to routing quality; we flag it here as a
design consideration, and defer any quantitative ablation result to §V-D
(not yet measured).

### B. Fuzzy Gearbox Controller

The routing decision is made by a Mamdani-type fuzzy inference system
(`router/fuzzy_controller.py`), implemented with the `scikit-fuzzy`
library rather than a hand-rolled approximation of one. Each of the five
complexity-sensor features becomes a fuzzy antecedent over the universe
$[0,1]$:

- `flesch_kincaid`, `token_length`, `entropy`, and `syntax_depth` each get
  three triangular membership functions — LOW, MEDIUM, HIGH — built from
  two edge breakpoints read from `config.yaml` (LOW spans $[0,0,lo]$,
  MEDIUM spans $[lo, mid, hi]$ with $mid$ as their midpoint, HIGH spans
  $[hi,1,1]$). Breakpoints are stored in each feature's raw units (e.g.
  entropy bits, parse-tree depth) and normalized through the *same* range
  constants as §III-A (Eqs. (3), (5), (6)) before being used as fuzzy-set
  edges, so the membership functions stay aligned with the features they
  are compared against — a real design property enforced in code, not an
  incidental convenience.
- `has_code_or_math` gets two membership functions ("no"/"yes") split at
  0.5, matching its effectively binary nature.

**Formal definition.** The single membership-function primitive used
throughout is the triangular function

$$\text{tri}(x; a, b, c) = \max\!\left(\min\!\left(\frac{x-a}{b-a},\ \frac{c-x}{c-b}\right),\ 0\right) \tag{8}$$

For each continuous antecedent feature $x \in \{f_{FK}, f_{TL}, f_H,
f_{SD}\}$ with calibrated breakpoints $(lo, hi)$ read from `config.yaml`
and normalized into the same $[0,1]$ space as $x$ via Eq. (1), and
$mid = (lo+hi)/2$:

$$\mu_{LOW}(x) = \text{tri}(x;\ 0,\ 0,\ lo) \tag{9}$$

$$\mu_{MEDIUM}(x) = \text{tri}(x;\ lo,\ mid,\ hi) \tag{10}$$

$$\mu_{HIGH}(x) = \text{tri}(x;\ hi,\ 1,\ 1) \tag{11}$$

For the binary antecedent $f_{CM}$:

$$\mu_{NO}(f_{CM}) = \text{tri}(f_{CM};\ 0,\ 0,\ 0.5) \tag{12}$$

$$\mu_{YES}(f_{CM}) = \text{tri}(f_{CM};\ 0.5,\ 1,\ 1) \tag{13}$$

The consequent "complexity" $y \in [0,100]$ has three fixed membership
functions, matching the deployed configuration exactly:

$$\mu_{Z=LOW}(y) = \text{tri}(y;\ 0,\ 0,\ 33) \tag{14}$$

$$\mu_{Z=MEDIUM}(y) = \text{tri}(y;\ 25,\ 50,\ 75) \tag{15}$$

$$\mu_{Z=HIGH}(y) = \text{tri}(y;\ 67,\ 100,\ 100) \tag{16}$$

The rule base consists of seven Mamdani IF-THEN rules $R_1,\dots,R_7$:
$R_1$ — low readability, low token length, low syntax depth, and no
code/math $\Rightarrow$ LOW; $R_2$ — (high syntax depth OR high entropy)
AND high readability grade $\Rightarrow$ HIGH; $R_3$ — code or math
detected $\Rightarrow$ HIGH; $R_4$ — high token length $\Rightarrow$
MEDIUM; $R_5$ — high entropy AND high readability grade $\Rightarrow$
HIGH; $R_6$ — medium token length OR medium syntax depth $\Rightarrow$
MEDIUM; $R_7$ — low readability AND low token length $\Rightarrow$ LOW.
For each rule $R_k$, its firing strength combines antecedent memberships
via the standard Mamdani/Zadeh operators — $\min$ for AND ($\wedge$),
$\max$ for OR ($\vee$):

$$\alpha_k = T\big(\mu_{A_{i_1}}(x_{i_1}), \dots, \mu_{A_{i_n}}(x_{i_n})\big) \tag{17}$$

Mamdani implication clips each rule's consequent membership function at
its firing strength, and rule outputs are aggregated across all seven
rules via $\max$:

$$\mu_{agg}(y) = \max_k \min\big(\alpha_k,\ \mu_{Z_k}(y)\big) \tag{18}$$

Centroid (center-of-gravity) defuzzification, computed by `scikit-fuzzy`'s
own library implementation rather than custom project code, produces the
crisp complexity score:

$$y^{*} = \frac{\int y \cdot \mu_{agg}(y)\, dy}{\int \mu_{agg}(y)\, dy} \tag{19}$$

The crisp score maps to a precision tier via configurable cut points
(default 33/66):

$$\text{tier}(y^{*}) = \begin{cases} \text{4-bit} & y^{*} \le 33 \\ \text{8-bit} & 33 < y^{*} \le 66 \\ \text{16-bit} & y^{*} > 66 \end{cases} \tag{20}$$

and

$$\text{win\_probability} = y^{*}/100 \tag{21}$$

Despite the name — kept for interface compatibility with the
RouteLLM-shaped bridge described in §II-A — Eq. (21) is *not* a RouteLLM
classifier's output; it is a deterministic rescaling of this system's own
fuzzy defuzzification result, consumed by `router/routellm_bridge.py`'s
threshold pass-through to make the final tier decision.

### C. Precision Tiers

Each prompt is routed to one of three precision tiers of a single base
model, `meta-llama/Llama-3.2-1B`:

- **4-bit**: bitsandbytes NF4 quantization with double quantization
  enabled (`BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`).
- **8-bit**: bitsandbytes LLM.int8() quantization (`load_in_8bit=True`).
- **16-bit**: bfloat16, loaded without quantization as the full-precision
  reference tier.

During measurement (§IV), only one tier is held in GPU memory at a time,
so that each tier's energy and latency behavior is measured in isolation
rather than under any shared-memory or co-residency effect.

### D. QAT LoRA Adapters

Each precision tier is paired with its own quantization-aware-training
(QAT) LoRA adapter, trained independently rather than sharing a single
adapter across tiers: `adapter_simple` for the 4-bit tier (rank $r=16$,
$\alpha=32$), `adapter_medium` for the 8-bit tier ($r=8$, $\alpha=16$),
and `adapter_complex` for the 16-bit tier ($r=4$, $\alpha=8$) — a constant
$\alpha/r$ ratio of 2 across all three, with LoRA rank scaling inversely
with precision. The design rationale is that the most aggressively
quantized tier (4-bit) has the most quantization error to recover from and
is given the most adapter capacity to do so, while the least-quantized
tier (16-bit) needs the least. All three adapters target the same two
attention projections (`q_proj`, `v_proj`) and share the
`meta-llama/Llama-3.2-1B` base.

The three adapters' existence, base-model consistency, and loadability
have been confirmed by a real, non-simulated load test: the actual gated
`meta-llama/Llama-3.2-1B` weights were downloaded, each adapter was
attached via `PeftModel.from_pretrained()`, and each was exercised with a
real greedy-decoding generation, with no errors across all three. This
confirms the adapters are usable, correctly-shaped artifacts; it does not
by itself establish that they improve accuracy over plain post-training
quantization at each tier — that comparison is a measurement question
(per-tier accuracy evaluation, not yet complete as of this draft; see
§V-D) and is not claimed here.

## IV. Measurement Methodology

### A. Energy Measurement

Energy is measured with the GPU's onboard hardware energy counter, read
via NVML's `nvmlDeviceGetTotalEnergyConsumption` (millijoule resolution) —
a direct hardware measurement, not a software estimate or an
emissions-derived proxy (see §II-E's note on this project's earlier,
now-fixed CO2-inversion bug). If a node's driver does not expose the
counter, the measurement script falls back to integrating 20 Hz power
samples and the affected run is flagged for disclosure rather than
reported as equivalent to a counter-based measurement.

The protocol for each tier is: 5 warmup generations (discarded from
measurement), followed by greedy (deterministic) decoding over the
evaluation prompt set with a fixed `max_new_tokens=128` and real generated
token counts (not an estimate). Only one precision tier is resident in GPU
memory at a time (§III-C). `torch.cuda.synchronize()` brackets each
measured generation to avoid attributing asynchronous CUDA work to the
wrong measurement window. An idle-power baseline and the GPU/driver
identity (model, driver version, clocks) are recorded to
`hardware_info.json` for every run. The full sweep across tiers is
repeated three times on different days, and results are reported as mean ±
95% confidence interval.

Because decoding is greedy and therefore deterministic, a router's
per-prompt output and energy cost are identical to those of whichever tier
it selects for that prompt; routing-condition results are accordingly
derived from the same per-tier, per-prompt measurement grid rather than
re-run independently.

### B. Energy Accounting Model

We formalize the per-request energy accounting used throughout this paper
and its remaining measurement program. For a prompt $p$ routed to tier
$\text{tier}(p)$ (Eq. (20)), total energy is

$$E_{total}(p) = E_{router}(p) + E_{inference}(\text{tier}(p),\ p) \tag{22}$$

where $E_{router}(p)$ is the fuzzy controller's own CPU-side compute cost
(feature extraction plus fuzzy inference) and $E_{inference}$ is the GPU
energy consumed generating the response at the chosen tier. Per-tier mean
joules-per-token, exactly what §V-A reports, is

$$\bar{e}_{tier} = \frac{1}{N}\sum_{i=1}^{N} \frac{E_i}{T_i},\quad \text{reported as } \bar{e}_{tier} \pm 1.96\frac{s}{\sqrt{N}}\ \text{(95\% CI)} \tag{23}$$

with $N = 1{,}500$ per tier in the measurement reported in §V-A. Eq. (22)
makes explicit that $E_{router}$ is measured and reported separately
rather than assumed to be zero, since it runs on CPU in milliseconds and
is not free: quantifying it is necessary to determine whether, and at what
prompt lengths, routing overhead is worth paying relative to always
serving at a single static tier (the break-even framing carried into §VII
as future work). As of this draft, $E_{router}$ has not yet been measured
on the college GPU cluster and no number is reported for it; this is a
planned, required measurement (`major-project/NEW.md`, Phase 6), not an
assumption of zero cost.

### C. Accuracy Measurement

Per-tier benchmark accuracy uses `lm-eval-harness`, evaluating
community-standard, versioned tasks with a fixed seed (42): tinyMMLU, a
GSM8K subset, and a HellaSwag subset, chosen to fit the measurement budget
while remaining standard, citable benchmarks. Raw per-task JSON output is
retained.

A separate, explicitly-labeled proxy is used to score correctness on this
project's own routed evaluation set (as distinct from the standard
lm-eval-harness benchmark tasks above): a reference-match check — numeric
match for arithmetic/math answers, normalized substring match for short
factual answers, and token-level F1 ≥ 0.5 otherwise. This proxy is used
only to attribute per-prompt correctness to a routing decision on this
project's own prompt set; it is not a substitute for, and is kept
explicitly distinct from, the lm-eval-harness benchmark accuracy numbers,
which remain this paper's accuracy claim of record.

### D. Automated Validation Gate

Every results file produced by a measurement session is checked by
`training/scripts/verify_results.py` before any number from it may be
cited anywhere in this paper. The gate checks: the output is non-empty;
each tier has at least $n=30$ samples; all recorded energies and token
counts are positive; joules-per-token values fall inside a physical
plausibility window for a roughly 1-billion-parameter model
(0.05–50 J/token); confidence intervals are acceptably tight; at least 3
repeated runs exist; and a set of internal-consistency checks hold (e.g.,
an oracle router's accuracy must be at least as high as any non-oracle
condition's; a result implying 4-bit inference costs more energy than
16-bit would trigger a disclosed risk warning rather than being silently
accepted). Only rows in `paper/results.md` with a passing verdict from
this gate are eligible to be cited as results in this paper; any WARN is
either resolved before citing the affected number or explicitly carried
into §VI (Threats to Validity).

### E. Current Status

All measurements described above run on a college GPU cluster (an NVIDIA
RTX 6000 Ada Generation card), not the T4 originally planned for; this
platform switch is documented in the project roadmap. As of this draft,
per `paper/results.md`, the energy-measurement protocol above (Session 1)
has been executed as a full sweep over the 500-prompt evaluation set,
repeated three times, and has passed the automated validation gate
described above; the resulting per-tier energy figures are reported in
§V-A, not here, since this section is confined to methodology. Per-tier
accuracy evaluation (Session 2, via lm-eval-harness, §IV-C) and the main
routing experiment (Session 4) had not, as of this draft, produced a
validator-passing, citable result — no accuracy or routing numbers are
stated anywhere in this paper until they do.

## V. Results

### A. RQ1 (Measurement) — Answered

**Finding.** Energy per token differs substantially and monotonically
across the three precision tiers of Llama-3.2-1B on an NVIDIA RTX 6000 Ada
Generation GPU. Table I reports the full result, drawn verbatim from
Session 1 of `paper/results.md` (2026-09-04, SPIT cluster; driver
610.43.02, idle power 22.53 W; 500-prompt evaluation set, 3 full sweeps,
seed = 42).

**Table I. Measured energy per token by precision tier (Eq. (23)).**

| Tier | $n$ | Mean J/token | 95% CI | 95% CI bounds |
|---|---|---|---|---|
| 4-bit  | 1,500 | 1.414 | ±0.055 | [1.359, 1.469] |
| 8-bit  | 1,500 | 3.052 | ±0.039 | [3.013, 3.091] |
| 16-bit | 1,500 | 8.124 | ±0.241 | [7.883, 8.365] |

Pairwise ratios computed from these means:

$$\frac{\bar{e}_{8bit}}{\bar{e}_{4bit}} = \frac{3.052}{1.414} \approx 2.16\times,\quad \frac{\bar{e}_{16bit}}{\bar{e}_{8bit}} = \frac{8.124}{3.052} \approx 2.66\times,\quad \frac{\bar{e}_{16bit}}{\bar{e}_{4bit}} = \frac{8.124}{1.414} \approx 5.75\times$$

The three tiers' 95% confidence intervals do not overlap at any adjacent
pair: 4-bit's upper bound (1.469) is below 8-bit's lower bound (3.013),
and 8-bit's upper bound (3.091) is below 16-bit's lower bound (7.883).
This is the statistical basis for treating the separation as real rather
than measurement noise: energy-per-token increases monotonically and
significantly from 4-bit to 8-bit to 16-bit.

**Interpretation against the go/no-go gate.** `major-project/NEW.md`
Phase 1 defines this project's go/no-go criterion as: "if 4-bit/8-bit show
real, non-trivial energy savings over fp16 → proceed." The measured,
non-overlapping, monotonic separation above satisfies that criterion on
real hardware — this is not the assumed linear bit-width model referenced
in §I as retracted, but a directly measured hardware result. This licenses
proceeding with the routing system's remaining evaluation (RQ2–RQ4 below):
the physical premise that precision tier materially changes per-token
energy cost on this hardware is empirically established.

**Scope.** This result is a single-GPU-class (NVIDIA RTX 6000 Ada
Generation), single-model (Llama-3.2-1B) measurement. We do not claim it
generalizes to other GPU architectures, other model sizes/families, or
other decoding configurations; see §VI for the full scope discussion.

### B. RQ2 (Routing) — Pending

RQ2 asks how much energy complexity-aware fuzzy routing (§III-B) uses
relative to static fp16 inference, and at what accuracy cost, across the
full 500-prompt evaluation set. Per `RESEARCH_PLAN.md`, this requires
running all routing conditions (static tiers, fuzzy router, and the
baseline/oracle conditions listed under RQ3 below) over the same
prompt×tier measurement grid described in §IV, produced by Session 4
(`training/scripts/kaggle_routing_experiment.py`). This session is
implemented and scripted but has not yet been run on the GPU cluster as of
this draft. No energy, accuracy, or energy/accuracy-tradeoff number for
routing is stated anywhere in this paper; this subsection will be filled
in from a validator-passing row of `paper/results.md` once Session 4
completes.

### C. RQ3 (Baselines) — Pending

RQ3 asks whether the fuzzy controller (§III-B) outperforms trivial
routing strategies — a random router with tier distribution matched to the
fuzzy router, and a naive threshold router over the mean of the five raw
complexity features (`naive_complexity_score()`, deliberately built as a
non-tautological baseline distinct from the fuzzy controller's own
defuzzified output, per `CREDIBILITY_REPORT.md`) — and how close it
approaches an oracle router (cheapest tier that still answers correctly).
This also requires Session 4's routing experiment, evaluated over the same
conditions as RQ2. No baseline-comparison result is stated anywhere in
this paper as of this draft.

### D. RQ4 (Ablation) — Pending

RQ4 asks whether the per-tier QAT LoRA adapters (§III-D) improve accuracy
at each precision tier relative to plain post-training quantization
without an adapter. This requires Session 2
(`training/scripts/kaggle_accuracy_eval.py`), which evaluates each tier
with and without its adapter via lm-eval-harness (§IV-C). Session 2 is
implemented and scripted but has not yet been run on the GPU cluster as of
this draft. No accuracy or ablation number is stated anywhere in this
paper; §III-D's adapter load-test result (adapters attach and generate
without error) is a loadability check only and is not an accuracy claim.

## VI. Threats to Validity

Drawn from `CREDIBILITY_REPORT.md` §5, restated against the RQ1 result
reported in §V-A and this paper's known measurement artifacts.

1. **Hardware scope.** All energy measurements reported in this paper are
   from one GPU model, the NVIDIA RTX 6000 Ada Generation (§IV-E, a
   platform switch from the T4 originally planned in `RESEARCH_PLAN.md`).
   Findings may differ on other GPU architectures (e.g., A100, H100, or
   older Turing-class cards) or on CPU inference. Claims in this paper are
   scoped to this specific card and are not asserted to generalize across
   GPU classes without further measurement.
2. **Model scope.** All results are from one model, Llama-3.2-1B. No claim
   is made about other model sizes or families without measurement.
3. **Quantization-kernel reality vs. theoretical bit-width scaling.**
   bitsandbytes' 4-bit and 8-bit kernels dequantize weights to a higher
   precision for compute; energy is not guaranteed to scale linearly, or
   even monotonically, with nominal bit-width for every model/hardware
   combination. §V-A's result confirms a real, monotonic, statistically
   significant separation on this hardware for this model, but this is
   reported as a *measured* outcome specific to this configuration, not as
   evidence that bit-width and energy scale proportionally in general.
4. **Correctness proxy.** The reference-match proxy used to score
   correctness on this project's own routed evaluation set (§IV-C) — exact
   numeric match, normalized substring match, or token-F1 ≥ 0.5 — is an
   approximation of true answer correctness, not a substitute for
   lm-eval-harness's standard benchmark scoring. Any oracle-router
   condition (RQ3, §V-C) that depends on this proxy inherits its noise.
5. **Shared-infrastructure noise.** The college GPU cluster used for
   measurement is shared infrastructure; thermal and host conditions can
   vary between sessions. This is mitigated, not eliminated, by repeating
   the full sweep three times on different days and reporting 95%
   confidence intervals (Eq. (23)); the non-overlapping CIs in Table I are
   evidence the observed tier separation exceeds this noise floor, but
   session-to-session variation beyond what three repeats capture cannot
   be ruled out.
6. **Energy counter scope (GPU-board-only).** NVML's
   `nvmlDeviceGetTotalEnergyConsumption` measures GPU-board energy only;
   host CPU and DRAM energy are excluded from every number in this paper.
   Total system energy for a request is therefore higher than the reported
   J/token figures, which should be read as GPU-board energy specifically,
   not end-to-end system energy.
7. **NVML counter resolution at very short generations.** 117 of 4,500
   measured rows (2.6%, across all three tiers) recorded `energy_j = 0.0`.
   These come from prompts that generated only a single output token,
   where the elapsed generation window is short enough that the NVML
   counter's millijoule-scale rounding resolution can report zero net
   energy change over that window. This is a real measurement artifact of
   counter resolution at short durations, not a meter failure or a
   discarded/invalid reading, and it is disclosed here as a limitation of
   the instrument at very short generations rather than silently excluded
   from the reported means; `verify_results.py` flags but does not reject
   these rows, and Table I's means are computed over the full 1,500-row
   per-tier set inclusive of them.

## VII. Conclusion

This paper reports the design and initial hardware-measured evaluation of
Green-Weight, a system that routes individual LLM requests across
precision tiers of a single base model using a five-feature complexity
sensor and a Mamdani fuzzy inference controller, formalized in §III with
numbered equations against the deployed implementation rather than
described only in prose. We built a measurement methodology grounded in
GPU on-board hardware energy counters and an automated, pre-registered
validation gate (§IV) so that no number in this paper is asserted without
a machine-checked, traceable source.

At the time of this draft, one of this paper's four research questions is
fully answered: RQ1 establishes, on real hardware, that energy-per-token
differs substantially and monotonically across the 4-bit, 8-bit, and
16-bit precision tiers of Llama-3.2-1B on an NVIDIA RTX 6000 Ada
Generation GPU, with statistically non-overlapping confidence intervals
across all three tiers (§V-A). This is this project's own pre-registered
go/no-go criterion for whether dynamic precision routing is a physically
grounded idea worth building the rest of the system around, and it passes.
RQ2 (routing energy/accuracy tradeoff), RQ3 (comparison against naive and
oracle baseline routers), and RQ4 (QAT adapter ablation) remain open: the
GPU sessions that will answer them (Session 4 for RQ2/RQ3, Session 2 for
RQ4) are implemented and scripted against the same verifier-gated
methodology used for RQ1, but have not yet been executed as of this draft.
We report their exact protocol in §IV and §V rather than any projected
outcome, so this paper's evidentiary boundary is unambiguous.

Near-term future work is therefore concrete rather than speculative:
running Sessions 2 and 4 to answer RQ2–RQ4; measuring the router's own
compute overhead, $E_{router}$ in Eq. (22), which is currently reported as
an un-measured but explicitly non-zero quantity, to complete the
energy-accounting/break-even model sketched in §IV-B (the prompt-length
point below which routing overhead outweighs its tier-selection savings);
training a real tier-preference router on Session 4's per-prompt routing
data as the genuine RouteLLM-methodology comparison point discussed in
§II-A, superseding the current documented threshold pass-through; and a
memory-footprint comparison of this project's single-base-model-plus-
adapters design against the cost of hosting multiple separate models under
a RouteLLM-style routing scheme. We view a paper that honestly reports one
fully measured, statistically significant finding alongside a fully
specified but not-yet-executed remainder as the appropriate way to report
work at this project's current stage, and we commit to reporting whatever
RQ2–RQ4 show once measured, whether or not the resulting numbers are as
favorable as an earlier, retracted, non-measured estimate once assumed.
