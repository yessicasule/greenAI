"""
api.py — Green-Weight FastAPI Backend
======================================
Run from backend/src/green_weight/ directory:
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
    routing_only: bool = False     # default: always infer (mock if no GPU)
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
    is_mock: Optional[bool] = False
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
            is_mock = False
            energy_joules = 0.0
        elif _model_pool_loaded:
            try:
                from benchmark.energy_tracker import EnergyTracker
                tracker = EnergyTracker()
                response, energy_info = tracker.track_inference(
                    req.prompt, final_tier,
                    max_new_tokens=req.max_new_tokens,
                )
                energy_joules = energy_info.get("joules", 0.0)
                is_mock = False
            except Exception as e:
                logger.warning(f"Real inference failed: {e}")
                response = _mock_response(req.prompt, final_tier)
                is_mock = True
                energy_joules = {"4bit": 8.5, "8bit": 28.5, "16bit": 105.2}.get(final_tier, 28.5)
        else:
            response = _mock_response(req.prompt, final_tier)
            is_mock = True
            energy_joules = {"4bit": 8.5, "8bit": 28.5, "16bit": 105.2}.get(final_tier, 28.5)

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
            is_mock=is_mock,
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

def _mock_response(prompt: str, tier: str) -> str:
    """Answer the prompt using math eval or pre-built keyword answers (no external LLM)."""
    import ast, operator, re

    p = prompt.strip()
    pl = p.lower()

    # Safe math evaluator
    _ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported")

    math_expr = re.sub(r"[^\d\s\+\-\*\/\(\)\.\^%]", "", p).strip()
    if math_expr:
        try:
            result = _eval(ast.parse(math_expr, mode="eval").body)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except Exception:
            pass

    # Keyword-based direct answers
    if re.search(r"\bdays?\s+in\s+a\s+week\b", pl): return "7"
    if re.search(r"\bdays?\s+in\s+a\s+year\b", pl): return "365 (366 in a leap year)"
    if re.search(r"\bhours?\s+in\s+a\s+day\b", pl): return "24"
    if re.search(r"\bminutes?\s+in\s+an?\s+hour\b", pl): return "60"
    if re.search(r"\bseconds?\s+in\s+a\s+minute\b", pl): return "60"
    if re.search(r"\bcapital\s+of\s+france\b", pl): return "Paris"
    if re.search(r"\bcapital\s+of\s+india\b", pl): return "New Delhi"
    if re.search(r"\bcapital\s+of\s+usa\b", pl): return "Washington, D.C."
    if re.search(r"\bsupply\s+and\s+demand\b", pl):
        return ("Supply and demand is a fundamental economic model. Demand represents how much consumers want a product; supply represents how much producers offer. When demand exceeds supply, prices rise. When supply exceeds demand, prices fall. The equilibrium price is where the two curves intersect.")
    if re.search(r"\bwater\s+cycle\b", pl):
        return ("The water cycle (hydrological cycle) describes the continuous movement of water on Earth: Evaporation (water turns to vapour from oceans/lakes) -> Condensation (vapour forms clouds) -> Precipitation (rain/snow falls) -> Collection (water returns to oceans/rivers) -> repeat.")
    if re.search(r"\bgodel|incompleteness\b", pl):
        return ("Goedel incompleteness theorems (1931): 1st - Any consistent formal system powerful enough to express arithmetic contains true statements unprovable within the system. 2nd - Such a system cannot prove its own consistency.")
    if re.search(r"\bquicksort\b", pl):
        return "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"

    # Generic fallback
    return f"Routed to {tier} tier. Connect a GPU model pool for live inference, or use one of the sample prompts above."


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
