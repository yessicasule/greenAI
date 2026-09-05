"""Minimal fp16 throughput probe for the Session 4 16-bit anomaly.

Session 4's 16-bit tier measures ~9 tok/s on an RTX 6000 Ada for
Llama-3.2-1B, roughly 6x SLOWER than the 4-bit tier -- backwards, since
bitsandbytes 4-bit carries dequantization overhead that fp16 does not.
Running the 16-bit tier alone (job 1498) reproduced it, ruling out
contamination from the earlier tiers.

This bypasses the experiment entirely: no router, no energy meter, no
adapters, no quantization. Just load the model and time generation, so
the result separates "fp16 is slow on this node" from "something in
phase_a is slow". It also prints the resolved dtype and allocated memory,
which is the direct check that torch_dtype/dtype was actually honoured
rather than silently falling back to fp32.

Run via micro_bench.sh. Takes ~2 minutes.
"""

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
PROMPT = "Explain the theory of gravity in one paragraph."
MAX_NEW = 128


def main():
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.perf_counter()
    # `dtype=` is the current spelling; `torch_dtype=` is deprecated in this
    # transformers version and warns on every load.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map={"": 0})
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # The direct check: if this says float32, the dtype argument was
    # ignored and every "fp16" measurement in Session 4 is really fp32.
    print(f"model.dtype = {model.dtype}", flush=True)
    print(f"first param dtype = {next(model.parameters()).dtype}", flush=True)
    print(f"allocated = {torch.cuda.memory_allocated() / 1e9:.2f} GB "
          f"(fp16 ~2.5, fp32 ~4.9)", flush=True)

    inputs = tok(PROMPT, return_tensors="pt").to("cuda")

    def gen(n):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=n, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return out.shape[1] - inputs["input_ids"].shape[1]

    for _ in range(3):
        gen(32)

    # Several timed runs: the level tells us whether fp16 is slow at all,
    # the spread tells us whether the node is contended. Session 4 saw
    # 11-19s for identical 128-token prompts, so variance is the point.
    print("\n--- timed runs ---", flush=True)
    print("  cpu_s is this process's own CPU time. Generation at batch 1 is "
          "launch-bound, so:", flush=True)
    print("    cpu_s flat while wall_s grows  -> we are WAITING "
          "(descheduled by a co-tenant; environmental)", flush=True)
    print("    cpu_s grows with wall_s        -> we are doing more work "
          "per token (internal: our stack)", flush=True)
    rates = []
    for r in range(12):
        c0, t0 = time.process_time(), time.perf_counter()
        n = gen(MAX_NEW)
        dt = time.perf_counter() - t0
        dc = time.process_time() - c0
        rates.append(n / dt)
        print(f"  run {r + 1:2d}: {n} tok  wall {dt:6.2f}s  cpu {dc:6.2f}s  "
              f"cpu/wall {dc / dt:.2f}  {n / dt:5.1f} tok/s", flush=True)

    lo, hi = min(rates), max(rates)
    print(f"\nmean {sum(rates) / len(rates):.1f} tok/s, "
          f"range {lo:.1f}-{hi:.1f} ({hi / lo:.2f}x spread)", flush=True)
    print("\nExpectation: a 1B model in fp16 on an RTX 6000 Ada should reach "
          "roughly 40-80 tok/s with a tight spread.\n"
          "  ~40-80 tok/s, tight  -> fp16 is fine; the slowness is in phase_a\n"
          "  ~10 tok/s            -> the node cannot run fp16 at speed "
          "(environmental: CPU starvation from a co-tenant)\n"
          "  wide spread          -> contention, and the 3-run "
          "reproducibility gate is not achievable as scheduled", flush=True)


if __name__ == "__main__":
    main()
