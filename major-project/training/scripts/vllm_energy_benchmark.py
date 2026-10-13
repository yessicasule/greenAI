"""
Session 5: per-token GPU energy of one precision tier under vLLM.

One invocation = one (model, quantization, batch size). Run each tier in its
own process (session5_vllm.sh does this) so GPU memory is fully released
between tiers.

Same energy protocol as kaggle_routing_experiment.py: NVML cumulative
energy counter around each generate() call, greedy decoding, 128 new
tokens, same 500 prompts, same correctness proxy. Adds a monitor thread
that logs co-tenants, GPU state and host state every 2 s for the whole run
(the end-of-run snapshot used before missed a disrupted run).

Outputs in --output-dir:
  per_prompt.csv  one row per prompt (batch energy split by output tokens)
  per_batch.csv   one row per generate() call
  monitor.csv     2 s samples of GPU + host state
  info.json       hardware, software, model and summary
"""

import argparse
import csv
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "backend" / "src" / "green_weight" / "data" / "eval_prompts.jsonl"
MAX_NEW_TOKENS = 128
SEED = 42

# ---------------------------------------------------------------- correctness
# Copied verbatim from kaggle_routing_experiment.py so results stay comparable.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def is_correct(response, reference):
    if not reference:
        return False
    ref_n, resp_n = _norm(reference), _norm(response)
    ref_nums = _NUM_RE.findall(reference)
    if ref_nums and len(ref_n.split()) <= 3:
        return ref_nums[-1] in _NUM_RE.findall(response)
    if len(ref_n.split()) <= 6:
        return ref_n in resp_n

    def stems(tokens):
        return {t[:5] for t in tokens}
    ref_t, resp_t = stems(ref_n.split()), stems(resp_n.split())
    if not ref_t or not resp_t:
        return False
    overlap = len(ref_t & resp_t)
    p, r = overlap / len(resp_t), overlap / len(ref_t)
    return (2 * p * r / (p + r) if p + r else 0.0) >= 0.5


# ---------------------------------------------------------------- energy meter
def _resolve_nvml_index():
    """Physical NVML index of cuda:0 (NVML ignores CUDA_VISIBLE_DEVICES)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return 0
    entry = vis.split(",")[0].strip()
    if entry.isdigit():
        return int(entry)
    for i in range(pynvml.nvmlDeviceGetCount()):
        uuid = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
        if (uuid.decode() if isinstance(uuid, bytes) else uuid) == entry:
            return i
    raise RuntimeError(f"No NVML device matches {entry!r}")


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


class Meter:
    def __init__(self):
        pynvml.nvmlInit()
        self.index = _resolve_nvml_index()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)  # fail fast if unsupported

    def joules(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0

    def info(self):
        return {"gpu": _s(pynvml.nvmlDeviceGetName(self.h)),
                "driver": _s(pynvml.nvmlSystemGetDriverVersion()),
                "nvml_index": self.index,
                "gpu_uuid": _s(pynvml.nvmlDeviceGetUUID(self.h)),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


# ---------------------------------------------------------------- monitor
def _our_pids():
    """This process and all its descendants (vLLM runs its engine in a child)."""
    me = os.getpid()
    children = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(d))
            except (OSError, IndexError, ValueError):
                pass
    out, stack = set(), [me]
    while stack:
        p = stack.pop()
        out.add(p)
        stack.extend(children.get(p, []))
    return out


def _cpu_mhz(cpus):
    mhz, cur = {}, None
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("processor"):
                cur = int(line.split(":")[1])
            elif line.startswith("cpu MHz") and cur is not None:
                mhz[cur] = float(line.split(":")[1])
    vals = [mhz[c] for c in cpus if c in mhz]
    return (sum(vals) / len(vals)) if vals else float("nan"), \
        (sum(mhz.values()) / len(mhz)) if mhz else float("nan")


class Monitor:
    FIELDS = ["t", "gpu_util", "gpu_mem_util", "power_w", "sm_mhz", "cotenant_pids",
              "load1", "mhz_our_cpus", "mhz_all_cpus"]

    def __init__(self, meter, path, period=2.0):
        self.m, self.period, self.stop_flag = meter, period, False
        self.cpus = sorted(os.sched_getaffinity(0))
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()
        self.cotenants_seen = set()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        h = self.m.h
        ut = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            ours = _our_pids()
            others = sorted(p.pid for p in procs if p.pid not in ours)
        except pynvml.NVMLError:
            others = []
        self.cotenants_seen.update(others)
        ours_mhz, all_mhz = _cpu_mhz(self.cpus)
        self.w.writerow({
            "t": round(time.time(), 1), "gpu_util": ut.gpu, "gpu_mem_util": ut.memory,
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1),
            "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "cotenant_pids": " ".join(map(str, others)),
            "load1": round(os.getloadavg()[0], 2),
            "mhz_our_cpus": round(ours_mhz), "mhz_all_cpus": round(all_mhz)})
        self.f.flush()

    def _loop(self):
        while not self.stop_flag:
            try:
                self._sample()
            except Exception as e:  # monitoring must never kill a measurement
                print(f"[monitor] {e!r}", flush=True)
            time.sleep(self.period)

    def start(self):
        self.th.start()

    def stop(self):
        self.stop_flag = True
        self.th.join()
        self.f.close()


# ---------------------------------------------------------------- run
def load_prompts(limit):
    rows, seen = [], set()
    with open(PROMPTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["prompt"] not in seen:
                    seen.add(r["prompt"])
                    rows.append(r)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, help="label, e.g. fp16 / fp8 / awq4")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization (e.g. fp8); omit for fp16 or pre-quantized checkpoints")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke test)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.limit)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS}", flush=True)

    import vllm
    from vllm import LLM, SamplingParams

    meter = Meter()
    mon = Monitor(meter, out / "monitor.csv")
    mon.start()

    t0 = time.time()
    llm = LLM(model=args.model, quantization=args.quantization, dtype="float16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              seed=SEED)
    load_s = time.time() - t0
    cfg = llm.llm_engine.model_config
    print(f"loaded {args.model} in {load_s:.1f}s; quantization={cfg.quantization} "
          f"dtype={cfg.dtype}", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=SEED)

    for i in range(args.warmup):
        llm.generate([prompts[i % len(prompts)]["prompt"]] * args.batch_size, sp, use_tqdm=False)
    print(f"warmed up ({args.warmup} generations)", flush=True)

    pp_f = open(out / "per_prompt.csv", "w", newline="", encoding="utf-8")
    pp = csv.DictWriter(pp_f, fieldnames=["prompt_id", "tier", "batch_size", "difficulty",
                                          "energy_j", "tokens_out", "j_per_token", "latency_s",
                                          "correct", "response"])
    pp.writeheader()
    pb_f = open(out / "per_batch.csv", "w", newline="")
    pb = csv.DictWriter(pb_f, fieldnames=["batch", "n_prompts", "energy_j", "tokens_out",
                                          "latency_s", "j_per_token", "tok_per_s", "watts"])
    pb.writeheader()

    tot_e = tot_tok = tot_t = 0.0
    bs = args.batch_size
    for b, start in enumerate(range(0, len(prompts), bs)):
        chunk = prompts[start:start + bs]
        # generate() returns only when outputs are complete; no cuda sync here,
        # since vLLM may run the model in a separate engine process.
        e0, t1 = meter.joules(), time.perf_counter()
        outs = llm.generate([p["prompt"] for p in chunk], sp, use_tqdm=False)
        dt, e = time.perf_counter() - t1, meter.joules() - e0
        toks = [len(o.outputs[0].token_ids) for o in outs]
        n_tok = sum(toks)
        pb.writerow({"batch": b, "n_prompts": len(chunk), "energy_j": round(e, 3),
                     "tokens_out": n_tok, "latency_s": round(dt, 4),
                     "j_per_token": round(e / n_tok, 4) if n_tok else "",
                     "tok_per_s": round(n_tok / dt, 2), "watts": round(e / dt, 1) if dt else ""})
        for k, (p, o, nt) in enumerate(zip(chunk, outs, toks)):
            share = e * nt / n_tok if n_tok else 0.0
            text = o.outputs[0].text
            pp.writerow({"prompt_id": start + k, "tier": args.tier, "batch_size": bs,
                         "difficulty": p.get("difficulty_label", "?"),
                         "energy_j": round(share, 3), "tokens_out": nt,
                         "j_per_token": round(share / nt, 4) if nt else "",
                         "latency_s": round(dt, 4),
                         "correct": is_correct(text, p.get("reference_answer", "")),
                         "response": text})
        pp_f.flush()
        pb_f.flush()
        tot_e, tot_tok, tot_t = tot_e + e, tot_tok + n_tok, tot_t + dt
        if (b + 1) % max(1, 50 // bs) == 0:
            print(f"  {start + len(chunk)}/{len(prompts)} prompts, "
                  f"{tot_tok / tot_t:.1f} tok/s, {tot_e / tot_tok:.3f} J/tok", flush=True)

    mon.stop()
    pp_f.close()
    pb_f.close()
    info = {**meter.info(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "tier": args.tier, "model": args.model, "quantization_arg": args.quantization,
            "quantization_resolved": str(cfg.quantization), "dtype": str(cfg.dtype),
            "batch_size": bs, "n_prompts": len(prompts), "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED, "load_s": round(load_s, 1), "vllm": vllm.__version__,
            "cpus": mon.cpus,
            "cotenant_pids_seen_during_run": sorted(mon.cotenants_seen),
            "summary": {"energy_j": round(tot_e, 1), "tokens_out": int(tot_tok),
                        "j_per_token": round(tot_e / tot_tok, 4) if tot_tok else None,
                        "tok_per_s": round(tot_tok / tot_t, 2) if tot_t else None,
                        "watts": round(tot_e / tot_t, 1) if tot_t else None}}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info["summary"]), flush=True)
    print(f"cotenants seen during run: {info['cotenant_pids_seen_during_run'] or 'none'}",
          flush=True)


if __name__ == "__main__":
    main()
