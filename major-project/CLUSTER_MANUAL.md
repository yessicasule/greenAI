# SPIT GPU Cluster — Runbook

Parallel to `KAGGLE_MANUAL.md`, for the college GPU cluster (the platform
NEW.md Phase 0 switched to after Kaggle's Session 1 run hung — see
`CREDIBILITY_REPORT.md` for that investigation). Source docs: the GPU
manual PDF (registration + web portal walkthrough) and
[Rio-0912/SPIT-GPU-Docs](https://github.com/Rio-0912/SPIT-GPU-Docs)
(technical SLURM usage), both read 2026-08-22. This file is the
synthesis — re-check the source repo if something here seems off, it may
have been updated since.

## Hardware

- 2x NVIDIA RTX A6000 (Ampere generation — confirmed newer than the
  Volta-minimum this project's NVML-energy-counter methodology requires;
  `check_nvml_energy_support.py` should pass, but still run it first per
  Phase 1's gate — don't skip on the assumption).
- Dual AMD EPYC, 224 threads / 112 cores, 250GB RAM.
- Max job time: 7 days (partition `general`).

## Access — CONFIRMED vs. UNCERTAIN

**Confirmed**: a web portal at `cluster.gpu.spit.ac.in` (Open OnDemand —
note the PDF manual says `cluster.spit.ac.in` without "gpu.", the GitHub
README says `cluster.gpu.spit.ac.in`; try the README's version first,
it's the more technical/recent source) offers **Interactive Apps**
(VS Code, Jupyter, Desktop) launched through a browser UI, each capped at
**1.5-2h continuous use** (the source README states both numbers in
different places — treat 1.5h as the real cap, be conservative).

**Uncertain**: whether a plain SSH login-node hostname exists for this
cluster (separate from the "interactive GPU via SSH" that's explicitly
forbidden — that rule is about not grabbing a *GPU* node outside the
scheduler, not necessarily about SSH to a login node for job submission).
Neither source document gives an SSH hostname. **Do not assume one and
do not guess a hostname** — confirm with whoever's actually at the
keyboard before any workflow that depends on it. If no SSH login-node
access exists, all cluster interaction (git clone, sbatch, squeue,
tailing logs) has to happen through the browser portal's Interactive
Apps / shell access feature, with a human relaying commands and output
back and forth — slower, but the only confirmed-safe path absent more
information.

## Rules of Engagement (given directly by the user, treat as absolute)

- **Local validation first**: code must be validated on a local
  environment before it ever touches the cluster. (This project already
  has a head start here — `kaggle_energy_benchmark.py` and
  `kaggle_routing_experiment.py` were reviewed and bug-fixed 2026-08-22
  without a GPU; the QAT adapters were load-tested for real on CPU. That
  is NOT the same as a GPU dry run, which has never happened for these
  scripts — treat the first real cluster run as the first real test of
  GPU-path code, watch it closely.)
- **No direct interactive SSH to a GPU node.** Use `srun` (short,
  scheduler-mediated interactive commands) or `sbatch` (batch jobs) —
  never try to grab a GPU node outside the scheduler.
- **Interactive Apps (VS Code/Jupyter/Desktop) capped at 1.5h
  continuous.** Fine for setup/editing/quick checks; the actual multi-hour
  energy benchmark MUST go through `sbatch`, not an interactive app.
- **Two consecutive unanswered administrative warnings → 2-year ban.**
  If any email/notification from cluster admins arrives, respond
  immediately — this is not a "get to it later" situation.
- **Expired directories are purged on the 1st of every month.**
  Download/back up any results (CSVs, JSONs, logs) off the cluster
  promptly after each session — don't leave them sitting there.
- **No cryptomining / commercial use** — not applicable to this project,
  noting for completeness.

## Software environment

Module system, no manual Python/Conda install needed.

```sh
module avail                      # see what's available
module load python/3.11.14        # load Python
uv venv env                       # fast venv (recommended over conda)
source env/bin/activate
uv pip install torch transformers accelerate bitsandbytes nvidia-ml-py
```

Note: install `nvidia-ml-py` (the current PyPI package providing the
`pynvml` importable module), not the older standalone `pynvml` package —
matches what NEW.md Phase 0 already specified for
`check_nvml_energy_support.py`.

Apptainer (containers) is available if a plain venv ever proves
insufficient (e.g. a finicky `bitsandbytes` CUDA build):
```sh
apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:24.02-py3
apptainer exec --nv pytorch.sif python script.py
```
Not expected to be needed — start with the plain venv approach, it's what
`kaggle_energy_benchmark.py` etc. already assume.

## Getting the code onto the cluster

The project repo (`yessicasule/greenAI`) is on GitHub and up to date as
of 2026-08-22 (5 commits pushed today — all the implementation-hardening
fixes, the new test suite, and the frontend redesign). Once you have a
cluster shell (via whichever access method got confirmed above):

```sh
git clone https://github.com/yessicasule/greenAI.git
cd greenAI/major-project
```

No rsync-from-laptop needed — `kaggle_routing_experiment.py`'s docstring
still says "SSH GPU box... rsync" from before this repo existed; that's
now stale, `git clone` directly on the cluster is simpler. (Worth fixing
that docstring in a future pass — not urgent.)

## Session 1 (energy ground truth) on this cluster — the immediate goal

1. **Hardware check first, via `srun` (not sbatch — this only takes
   seconds):**
   ```sh
   cd major-project
   srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:05:00 \
     python training/scripts/check_nvml_energy_support.py
   ```
   Needs `nvidia-ml-py` installed first (see Software environment above).
   **Gate**: if this reports the energy counter is NOT supported, STOP —
   don't proceed to the real benchmark. Re-read NEW.md Phase 1's go/no-go
   section for the fallback options. (Expected to pass — A6000 is
   Ampere — but confirm, don't assume.)

2. **Get the real eval prompt set onto the cluster.** The repo doesn't
   include `data/eval_prompts.jsonl` (it's generated, not committed —
   check `.gitignore`). Either:
   - Run `training/scripts/prepare_eval_dataset.py` on the cluster to
     regenerate it fresh (needs internet access + `datasets` package —
     confirm the cluster's compute nodes have outbound internet before
     relying on this), or
   - Copy the already-generated one from this laptop
     (`major-project/backend/src/green_weight/data/eval_prompts.jsonl`)
     up via whatever file-transfer the portal offers (Open OnDemand
     usually has a Files app with upload).
   `kaggle_energy_benchmark.py` looks for it at
   `eval_prompts.jsonl` (relative to cwd) as a fallback path — place it
   in the job's working directory, or edit `EVAL_FILE_CANDIDATES` if it
   ends up somewhere else. **Without this file, the script silently falls
   back to 12 built-in prompts — Phase 1 needs the real 500, watch for
   the "Loaded N unique prompts from ..." log line to confirm it found
   the real file, not the fallback.**

3. **Set `HF_TOKEN`** as an environment variable in the cluster shell
   session (`export HF_TOKEN=...`, typed directly by the person at the
   keyboard — never paste it through an AI assistant into a shared
   conversation, same rule as we followed for the local machine earlier
   today). `transformers`/`huggingface_hub` picks up `HF_TOKEN` from the
   environment automatically — no code change needed.

4. **Submit the real run via `sbatch`** — adapt the `gpu_single` template
   (`SPIT-GPU-Docs/templates/gpu_single/submit.sh`):
   ```bash
   #!/bin/bash
   #SBATCH --job-name=greenweight_session1
   #SBATCH --partition=general
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --gres=gpu:1
   #SBATCH --time=06:00:00
   #SBATCH --output=session1_%j.out
   #SBATCH --error=session1_%j.err

   cd "${SLURM_SUBMIT_DIR:?}"
   module load python/3.11.14
   uv venv env
   source env/bin/activate
   uv pip install torch transformers accelerate bitsandbytes nvidia-ml-py

   export HF_TOKEN="$HF_TOKEN"   # inherited from the submitting shell's env

   python -u training/scripts/kaggle_energy_benchmark.py
   ```
   `--time=06:00:00` is deliberate: NEW.md's budget for Session 1 is
   ~2-3h, so 6h gives real margin without leaving it able to run
   silently for a full day if something's wrong — direct lesson from the
   Kaggle investigation (that run wasn't caught for ~5 hours after it
   actually stalled, because nothing was watching and the platform
   timeout was the only backstop).
   ```sh
   sbatch session1.sh
   squeue -l                          # check it's running
   ```

5. **Actively monitor — do not fire-and-forget.** This is the single
   biggest lesson from the Kaggle Session 1 postmortem: that run hung
   silently at the 8-bit tier for ~4.7 hours before a platform timeout
   caught it, because nobody was watching. This time:
   - Check the log (`tail -f session1_<jobid>.out`) within the first
     15-30 minutes to confirm it's actually progressing (model loading,
     warmup, first tier's measurements starting).
   - Check again every 30-60 minutes. Expect roughly even progress
     across the three tiers (4bit → 8bit → 16bit) within the ~2-3h
     budget. If output stalls for more than ~20-30 minutes with no new
     log lines, treat that as a real warning sign (matches exactly what
     happened on Kaggle) — don't just wait for the 6h timeout, go look.
   - `watch -n 5 nvidia-smi` (if you have a way to run it against the
     right node) to confirm the GPU is actually being utilized, not
     idling.

6. **On completion**: run `training/scripts/verify_results.py` against
   the output (per NEW.md Phase 1), download `energy_per_inference.csv`,
   `energy_summary.csv`, `hardware_info.json` off the cluster promptly
   (monthly purge policy), and follow NEW.md Phase 1's go/no-go decision
   tree based on what the numbers show.

## What's NOT resolved yet (flag if you hit these)

- Exact mechanism for getting a cluster shell if no SSH login-node
  access exists (see Access section above).
- Whether cluster compute nodes have outbound internet (needed for
  `pip`/`uv pip install` to reach PyPI, and for downloading the gated
  Llama-3.2-1B weights from HuggingFace, and for `prepare_eval_dataset.py`
  if regenerating the eval set on-cluster). Untested — find out early,
  it blocks almost everything if node-level internet is restricted and
  only the login node has it.
