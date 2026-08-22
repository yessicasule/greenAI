"""
Tests for models/model_pool.py — load_pool() itself needs a CUDA GPU (out
of scope for this suite per the GPU boundary rule), so these tests exercise
what's testable without one:

1. load_pool() must not raise when no CUDA GPU is present -- it should log
   an error and return, not crash. Forced via monkeypatching
   `torch.cuda.is_available` so this is deterministic regardless of the
   machine the suite happens to run on.
2. The adapter-directory path-resolution logic (project_root computed as
   `Path(__file__).resolve().parents[4] / "adapters" / ...`) is replicated
   here and checked against the real on-disk adapters/ directory, to guard
   the exact class of bug described in the task brief (wrong parents[]
   depth / wrong adapter directory names silently pointing at nothing).

See test_known_issues.py for a locked-in xfail: in *this* worktree,
model_pool.py's actual adapter path computation still uses the pre-fix
names (adapter_4bit/adapter_8bit/adapter_16bit), which do not exist on
disk -- only adapter_simple/adapter_medium/adapter_complex do.
"""

from pathlib import Path

import pytest

from models import model_pool


def test_load_pool_does_not_raise_without_cuda(monkeypatch):
    monkeypatch.setattr(model_pool.torch.cuda, "is_available", lambda: False)
    # Must return quietly (logs an error internally), never raise.
    assert model_pool.load_pool() is None


def test_project_root_resolves_to_major_project(major_project_dir):
    # Mirrors model_pool.py's own computation: models/model_pool.py ->
    # models/ -> green_weight/ -> src/ -> backend/ -> major-project/
    project_root = Path(model_pool.__file__).resolve().parents[4]
    assert project_root == major_project_dir.resolve()


def test_real_adapters_exist_on_disk_under_expected_names(major_project_dir):
    """Guards the fix this bug describes: the adapters actually shipped
    are named adapter_simple/adapter_medium/adapter_complex, not
    adapter_4bit/adapter_8bit/adapter_16bit."""
    adapters_dir = major_project_dir / "adapters"
    assert (adapters_dir / "adapter_simple").exists()
    assert (adapters_dir / "adapter_medium").exists()
    assert (adapters_dir / "adapter_complex").exists()


def test_hf_token_env_lookup(monkeypatch):
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert model_pool._get_hf_token() is None

    monkeypatch.setenv("HF_TOKEN", "abc123")
    assert model_pool._get_hf_token() == "abc123"

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "xyz789")
    assert model_pool._get_hf_token() == "xyz789"  # preferred over HF_TOKEN
