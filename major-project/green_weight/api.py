"""
api.py — Green-Weight FastAPI Backend
======================================
Run from green_weight/ directory:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /infer          — route + infer a single prompt
    POST /route          — routing-only (no GPU needed)
    GET  /results        — load pipeline_trace.jsonl
    GET  /energy         — load energy_summary.csv
    GET  /accuracy       — load accuracy_results.json
    GET  /health         — health check
"""

import json
import csv
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_config
from router.complexity_scorer import score as score_complexity
from router.fuzzy_controller import FuzzyController
from router.routellm_bridge import RouteLLMBridge

logger = logging.getLogger(__name__)

app = FastAPI(title="Green-Weight API", version="0.1.0")

# Allow frontend dev server (Vite default: 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (initialised once at startup) ──────────────────────────────────
_fuzzy = None
_bridge = None
_model_pool_loaded = False


@app.on_event("startup")
def startup():
    global _fuzzy, _bridge
    _fuzzy = FuzzyController()
    _bridge = RouteLLMBridge()
    logger.info("[OK] Routing layer initialised")


# ── Request / Response schemas ────────────────────────────────────────────────

class InferRequest(BaseModel):
    prompt: str
    routing_only: bool = True      # set False when GPU available
    max_new_tokens: int = 256


class RouteResponse(BaseModel):
    prompt: str
    features: dict
    fuzzy_tier: str
    win_probability: float
    final_tier: str
    energy_joules: float
    duration_s: float
    response: Optional[str] = None
    cost_usd: float
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "routing_layer": "ready",
        "gpu_ready": _model_pool_loaded,
    }


@app.post("/route", response_model=RouteResponse)
def route_only(req: InferRequest):
    """
    Routing-only endpoint — no GPU needed.
    Returns complexity features, fuzzy tier, RouteLLM-refined tier.
    """
    try:
        import time
        t0 = time.time()

        features = score_complexity(req.prompt)
        fuzzy_tier, win_prob = _fuzzy.route(features)
        final_tier = _bridge.decide(req.prompt, win_prob)

        latency_ms = (time.time() - t0) * 1000

        # Energy is 0 in routing-only — no inference ran
        return RouteResponse(
            prompt=req.prompt,
            features=features,
            fuzzy_tier=fuzzy_tier,
            win_probability=round(win_prob, 4),
            final_tier=final_tier,
            energy_joules=0.0,
            duration_s=round(latency_ms / 1000, 3),
            response=None,
            cost_usd=0.0,
            latency_ms=round(latency_ms, 1),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer", response_model=RouteResponse)
def infer(req: InferRequest):
    """
    Full inference endpoint.
    In routing_only=True mode it behaves like /route.
    In routing_only=False mode it calls the model pool (requires GPU).
    """
    import time

    try:
        t0 = time.time()

        features = score_complexity(req.prompt)
        fuzzy_tier, win_prob = _fuzzy.route(features)
        final_tier = _bridge.decide(req.prompt, win_prob)

        if req.routing_only:
            response = None
            energy_joules = 0.0
        else:
            # Full inference — requires model pool to be loaded
            from benchmark.energy_tracker import EnergyTracker
            tracker = EnergyTracker()
            response, energy_info = tracker.track_inference(
                req.prompt, final_tier,
                max_new_tokens=req.max_new_tokens
            )
            energy_joules = energy_info.get("joules", 0.0)

        latency_ms = (time.time() - t0) * 1000

        # Rough cost estimate: $0.0001 per 4-bit call, $0.0004 8-bit, $0.0009 16-bit
        cost_map = {"4bit": 0.0001, "8bit": 0.0004, "16bit": 0.0009}
        cost_usd = cost_map.get(final_tier, 0.0004)

        # Append to pipeline trace
        _append_trace({
            "prompt": req.prompt,
            "features": features,
            "fuzzy_tier": fuzzy_tier,
            "win_probability": round(win_prob, 4),
            "final_tier": final_tier,
            "energy_joules": round(energy_joules, 4),
            "latency_ms": round(latency_ms, 1),
            "response": (response or "")[:300],
        })

        return RouteResponse(
            prompt=req.prompt,
            features=features,
            fuzzy_tier=fuzzy_tier,
            win_probability=round(win_prob, 4),
            final_tier=final_tier,
            energy_joules=round(energy_joules, 4),
            duration_s=round(latency_ms / 1000, 3),
            response=response,
            cost_usd=cost_usd,
            latency_ms=round(latency_ms, 1),
        )

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
def get_results():
    """Return all entries from pipeline_trace.jsonl."""
    config = get_config()
    trace_path = Path(config.get_log_output_dir()) / "pipeline_trace.jsonl"

    if not trace_path.exists():
        return {"entries": []}

    entries = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return {"entries": entries}


@app.get("/energy")
def get_energy():
    """Return energy summary CSV as JSON."""
    config = get_config()
    summary_path = Path(config.get_energy_output_dir()) / "energy_summary.csv"

    if not summary_path.exists():
        return {"tiers": []}

    rows = []
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return {"tiers": rows}


@app.get("/accuracy")
def get_accuracy():
    """Return accuracy_results.json."""
    config = get_config()
    acc_path = (
        Path(config.benchmark.get("accuracy_eval", {})
             .get("output_dir", "results/accuracy_logs"))
        / "accuracy_results.json"
    )

    if not acc_path.exists():
        return {}

    with open(acc_path) as f:
        return json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _append_trace(entry: dict):
    """Append a single entry to pipeline_trace.jsonl."""
    try:
        config = get_config()
        log_dir = Path(config.get_log_output_dir())
        log_dir.mkdir(parents=True, exist_ok=True)
        trace_path = log_dir / "pipeline_trace.jsonl"
        with open(trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Could not append to trace: {e}")