"""
Shared pytest fixtures / sys.path setup for the backend test suite.

The application code under backend/src/green_weight/ uses bare
same-directory imports everywhere (e.g. `from config import get_config`,
`from router import complexity_scorer`) — see green_weight/__init__.py's
module docstring: it expects cwd/sys.path to include
`backend/src/green_weight/` itself, not `backend/src/`.

training/scripts/verify_results.py is a standalone script (not part of the
green_weight package) living under major-project/training/scripts/ — it's
added to sys.path separately so it can be imported directly by name.

No GPU / network / gated-model access is required to import any of these
modules: torch, transformers, scikit-fuzzy, spacy (with en_core_web_sm),
textstat, and lm-eval are all installed in this environment, and none of
the modules under test call out to a live model at import time.
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent          # backend/tests
_BACKEND_DIR = _TESTS_DIR.parent                        # backend
_MAJOR_PROJECT_DIR = _BACKEND_DIR.parent                 # major-project

GREEN_WEIGHT_SRC = _BACKEND_DIR / "src" / "green_weight"
TRAINING_SCRIPTS = _MAJOR_PROJECT_DIR / "training" / "scripts"

for _p in (str(GREEN_WEIGHT_SRC), str(TRAINING_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def major_project_dir() -> Path:
    """Absolute path to major-project/, for tests that need to locate
    on-disk artifacts (e.g. adapters/) relative to the real repo layout."""
    return _MAJOR_PROJECT_DIR


@pytest.fixture
def green_weight_src_dir() -> Path:
    return GREEN_WEIGHT_SRC


@pytest.fixture(autouse=True)
def _clear_model_registry():
    """The model registry (models/local_llm_adapter.py's _model_registry)
    is module-level global state populated by model_pool.register_model().
    No test in this suite calls load_pool() for real (GPU boundary), but
    clear it before and after every test anyway so nothing can leak state
    between tests that construct AccuracyEvaluator / RoutedLM-like objects."""
    from models import local_llm_adapter
    local_llm_adapter._model_registry.clear()
    yield
    local_llm_adapter._model_registry.clear()
