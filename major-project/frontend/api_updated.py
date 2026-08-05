"""
api.py — Green-Weight FastAPI Backend
======================================
Run from green_weight/ directory:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /infer          — route + infer a single prompt (real or mock)
    POST /route          — routing-only (no GPU needed)
    GET  /results        — load pipeline_trace.jsonl
    GET  /energy         — load energy_summary.csv
    GET  /accuracy       — load accuracy_results.json
    GET  /health         — health check
"""

import json
import csv
import time
import logging
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_config
from router.complexity_scorer import score as score_complexity
from router.fuzzy_controller import FuzzyController
from router.routellm_bridge import RouteLLMBridge

logger = logging.getLogger(__name__)

app = FastAPI(title="Green-Weight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────
_fuzzy = None
_bridge = None
_model_pool_loaded = False


@app.on_event("startup")
def startup():
    global _fuzzy, _bridge, _model_pool_loaded
    _fuzzy = FuzzyController()
    _bridge = RouteLLMBridge()

    # Try to load model pool if GPU is available
    if torch.cuda.is_available():
        try:
            from models import model_pool
            model_pool.load_pool(lazy_16bit=True)
            _model_pool_loaded = True
            logger.info("[OK] Model pool loaded — GPU inference available")
        except Exception as e:
            logger.warning(f"Model pool failed to load: {e} — falling back to mock inference")
            _model_pool_loaded = False
    else:
        logger.info("[OK] No GPU detected — routing-only mode, mock responses will be used")
        _model_pool_loaded = False

    logger.info("[OK] Routing layer initialised")


# ── Mock response generator ───────────────────────────────────────────────────

MOCK_RESPONSES = {
    "4bit": [
        "The answer is straightforward: {prompt_end}",
        "Simply put: this is a basic factual question. The answer is well-established.",
        "This is a simple query. The response is concise and direct.",
    ],
    "8bit": [
        "This is a moderately complex topic. {prompt_start} involves several key concepts that "
        "are worth unpacking carefully. To understand this properly, we need to consider the "
        "underlying principles and how they interact with each other in practice.",
        "A good explanation of this topic covers the main ideas and their relationships. "
        "The subject has both theoretical foundations and practical applications that are "
        "worth exploring in some depth.",
    ],
    "16bit": [
        "This is a complex topic requiring detailed analysis. {prompt_start} encompasses "
        "multiple dimensions of understanding. From a theoretical perspective, we need to "
        "examine the foundational principles carefully. The practical implications are "
        "equally significant, as they affect how we approach related problems. Furthermore, "
        "there are nuanced edge cases and counterexamples that a thorough treatment must address. "
        "The literature on this subject is rich and multifaceted, drawing from several disciplines.",
        "Addressing this comprehensively requires examining multiple angles. The question of "
        "{prompt_start} is non-trivial and has been the subject of significant scholarly "
        "discussion. At its core, the issue involves balancing competing considerations "
        "while maintaining logical consistency throughout the analysis.",
    ],
}

def _mock_response(prompt: str, tier: str) -> str:
    import random
    templates = MOCK_RESPONSES.get(tier, MOCK_RESPONSES["8bit"])
    template = random.choice(templates)
    words = prompt.strip().split()
    prompt_start = " ".join(words[:6]) if len(words) >= 6 else prompt
    prompt_end = " ".join(words[-3:]) if len(words) >= 3 else prompt
    return (
        template
        .replace("{prompt_start}", prompt_start)
        .replace("{prompt_end}", prompt_end)
        + f"\n\n[Mock response — GPU not available · tier: {tier}]"
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    prompt: str
    routing_only: bool = False   # default False — always try to get a response
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
    is_mock: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "routing_layer": "ready",
        "gpu_ready": _model_pool_loaded,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/route", response_model=RouteResponse)
def route_only(req: InferRequest):
    """Routing-only — no inference, no GPU needed."""
    try:
        t0 = time.time()
        features = score_complexity(req.prompt)
        fuzzy_tier, win_prob = _fuzzy.route(features)
        final_tier = _bridge.decide(req.prompt, win_prob)
        latency_ms = (time.time() - t0) * 1000

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
            is_mock=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer", response_model=RouteResponse)
def infer(req: InferRequest):
    """
    Full inference endpoint.

    - If GPU is available and model pool loaded → real inference via EnergyTracker
    - If GPU not available or routing_only=True → mock response based on tier
    """
    try:
        t0 = time.time()

        # Always run routing (CPU, fast)
        features = score_complexity(req.prompt)
        fuzzy_tier, win_prob = _fuzzy.route(features)
        final_tier = _bridge.decide(req.prompt, win_prob)

        is_mock = False
        response = None
        energy_joules = 0.0

        if req.routing_only:
            # Caller explicitly asked for no inference
            response = None
            is_mock = False

        elif _model_pool_loaded:
            # Real GPU inference
            try:
                from benchmark.energy_tracker import EnergyTracker
                tracker = EnergyTracker()
                response, energy_info = tracker.track_inference(
                    req.prompt,
                    final_tier,
                    max_new_tokens=req.max_new_tokens,
                )
                energy_joules = energy_info.get("joules", 0.0)
                is_mock = False
            except Exception as e:
                logger.warning(f"Real inference failed: {e} — falling back to mock")
                response = _mock_response(req.prompt, final_tier)
                is_mock = True
                energy_joules = {"4bit": 8.5, "8bit": 28.5, "16bit": 105.2}.get(final_tier, 28.5)

        else:
            # No GPU — use mock with realistic energy estimates
            response = _mock_response(req.prompt, final_tier)
            is_mock = True
            energy_joules = {"4bit": 8.5, "8bit": 28.5, "16bit": 105.2}.get(final_tier, 28.5)

        latency_ms = (time.time() - t0) * 1000

        cost_map = {"4bit": 0.0001, "8bit": 0.0004, "16bit": 0.0009}
        cost_usd = cost_map.get(final_tier, 0.0004)

        _append_trace({
            "prompt": req.prompt,
            "features": features,
            "fuzzy_tier": fuzzy_tier,
            "win_probability": round(win_prob, 4),
            "final_tier": final_tier,
            "energy_joules": round(energy_joules, 4),
            "latency_ms": round(latency_ms, 1),
            "response": (response or "")[:300],
            "is_mock": is_mock,
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
            is_mock=is_mock,
        )

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
def get_results():
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
    try:
        config = get_config()
        log_dir = Path(config.get_log_output_dir())
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "pipeline_trace.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Could not append to trace: {e}")
