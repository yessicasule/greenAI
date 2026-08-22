"""
Tests for benchmark/accuracy_eval.py.

This module was rewritten 2026-08-22 to call real lm_eval (via HFLM wrapping
the already-loaded tier models) instead of returning hardcoded placeholder
scores — every failure path now raises RuntimeError rather than fabricating
a number. These tests cover what's exercisable without a GPU or a real
loaded model:

- AccuracyEvaluator raising RuntimeError (never a fake score) when no tier
  model is registered.
- RoutedLM's dispatch/grouping/reassembly logic, using fake Instance-like
  objects and fake tier-LMs (no real lm_eval model needed).
- _extract_scores()'s metric-key preference order and fallback behavior.
- compute_routellm_metrics()'s CPT/APGR math, exercised by populating
  evaluator.results / evaluator._last_routing_log directly rather than via
  evaluate_condition (which requires a real model).

AccuracyEvaluator.__init__ creates self.output_dir via mkdir(parents=True)
using a relative path from config.yaml (default "results/accuracy_logs") --
every test that constructs one runs inside monkeypatch.chdir(tmp_path) so
nothing is written into the real repo tree.
"""

import json

import pytest

from benchmark.accuracy_eval import AccuracyEvaluator, RoutedLM, TIER_TO_SERVICE


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return AccuracyEvaluator()


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

class TestInit:
    def test_construction_succeeds(self, evaluator):
        assert evaluator.tasks == ["mmlu", "hellaswag"]
        assert evaluator.results == {}

    def test_output_dir_created_under_cwd(self, evaluator, tmp_path):
        assert evaluator.output_dir.exists()
        assert evaluator.output_dir.resolve() == (tmp_path / "results" / "accuracy_logs").resolve()


# ----------------------------------------------------------------------
# No model loaded -> RuntimeError, never a fabricated score
# ----------------------------------------------------------------------

class TestRaisesWithoutModel:
    """The whole point of the 2026-08-22 rewrite: a failed/impossible
    evaluation must fail loudly, not silently return a placeholder."""

    def test_always_tier_raises_when_no_model_loaded(self, evaluator):
        with pytest.raises(RuntimeError, match="not loaded"):
            evaluator.evaluate_condition("always_4bit", "4bit")

    def test_routed_raises_when_no_model_loaded(self, evaluator):
        with pytest.raises(RuntimeError, match="No tier models are loaded"):
            evaluator.evaluate_condition("routed")

    def test_get_tier_lm_raises_for_unregistered_service(self, evaluator):
        with pytest.raises(RuntimeError, match="local/8bit"):
            evaluator._get_tier_lm("8bit")

    def test_run_all_conditions_raises_on_first_missing_model(self, evaluator):
        with pytest.raises(RuntimeError):
            evaluator.run_all_conditions()


# ----------------------------------------------------------------------
# RoutedLM dispatch / grouping / reassembly
# ----------------------------------------------------------------------

class _FakeInstance:
    """Minimal stand-in for lm_eval.api.instance.Instance — RoutedLM only
    reads req.args[0] (the context/text to route on)."""
    def __init__(self, context):
        self.args = (context,)


class _FakeFuzzyController:
    """Deterministic router: looks up the tier for each context string from
    a fixed mapping, bypassing the real fuzzy inference entirely."""
    def __init__(self, tier_for_context):
        self._tier_for_context = tier_for_context

    def route(self, features):
        return features["_tier"], 0.5


class _FakeTierLM:
    """Records which requests it was asked to handle, tags each result with
    its own tier label so dispatch/reassembly can be checked precisely."""
    def __init__(self, tag):
        self.tag = tag
        self.received = []

    def _handle(self, requests):
        self.received.append(list(requests))
        return [(self.tag, r.args[0]) for r in requests]

    def loglikelihood(self, requests):
        return self._handle(requests)

    def loglikelihood_rolling(self, requests):
        return self._handle(requests)

    def generate_until(self, requests):
        return self._handle(requests)


@pytest.fixture
def fake_score_complexity(monkeypatch):
    """Patches accuracy_eval.score_complexity so each prompt's routed tier
    is controlled directly by the test, without touching the real
    complexity scorer or fuzzy controller."""
    import benchmark.accuracy_eval as ae

    mapping = {}
    monkeypatch.setattr(ae, "score_complexity", lambda text: {"_tier": mapping[text]})
    return mapping


class TestRoutedLMConstruction:
    def test_raises_without_any_tier_lm(self):
        with pytest.raises(RuntimeError):
            RoutedLM({}, _FakeFuzzyController({}))


class TestRoutedLMDispatch:
    def test_order_preserving_across_mixed_tiers(self, fake_score_complexity):
        prompts = ["a", "b", "c", "d", "e"]
        tiers = ["4bit", "16bit", "8bit", "4bit", "16bit"]
        for p, t in zip(prompts, tiers):
            fake_score_complexity[p] = t

        tier_lms = {t: _FakeTierLM(t) for t in ("4bit", "8bit", "16bit")}
        routed = RoutedLM(tier_lms, _FakeFuzzyController(fake_score_complexity))
        requests = [_FakeInstance(p) for p in prompts]

        results = routed.loglikelihood(requests)

        assert results == [(t, p) for p, t in zip(prompts, tiers)]

    def test_requests_grouped_into_one_batch_per_tier(self, fake_score_complexity):
        prompts = ["a", "b", "c", "d"]
        tiers = ["4bit", "4bit", "16bit", "4bit"]
        for p, t in zip(prompts, tiers):
            fake_score_complexity[p] = t

        tier_lms = {t: _FakeTierLM(t) for t in ("4bit", "8bit", "16bit")}
        routed = RoutedLM(tier_lms, _FakeFuzzyController(fake_score_complexity))
        routed.loglikelihood([_FakeInstance(p) for p in prompts])

        # 3 prompts routed to 4bit should arrive as a single batched call,
        # not 3 separate calls.
        assert len(tier_lms["4bit"].received) == 1
        assert len(tier_lms["4bit"].received[0]) == 3
        assert len(tier_lms["16bit"].received) == 1
        assert len(tier_lms["16bit"].received[0]) == 1
        assert tier_lms["8bit"].received == []

    def test_routing_log_records_one_entry_per_request(self, fake_score_complexity):
        prompts = ["a", "b", "c"]
        for p, t in zip(prompts, ["4bit", "8bit", "16bit"]):
            fake_score_complexity[p] = t

        tier_lms = {t: _FakeTierLM(t) for t in ("4bit", "8bit", "16bit")}
        routed = RoutedLM(tier_lms, _FakeFuzzyController(fake_score_complexity))
        routed.loglikelihood([_FakeInstance(p) for p in prompts])

        assert len(routed.routing_log) == 3
        assert {r["tier"] for r in routed.routing_log} == {"4bit", "8bit", "16bit"}

    def test_falls_back_to_loaded_tier_when_routed_tier_unavailable(self, fake_score_complexity):
        fake_score_complexity["x"] = "16bit"  # routed tier, but not loaded below
        tier_lms = {"4bit": _FakeTierLM("4bit")}  # only 4bit is "loaded"

        routed = RoutedLM(tier_lms, _FakeFuzzyController(fake_score_complexity))
        results = routed.loglikelihood([_FakeInstance("x")])

        assert results == [("4bit", "x")]
        assert routed.routing_log[0]["tier"] == "4bit"

    def test_loglikelihood_rolling_dispatches_correctly(self, fake_score_complexity):
        fake_score_complexity["p"] = "8bit"
        tier_lms = {"8bit": _FakeTierLM("8bit")}
        routed = RoutedLM(tier_lms, _FakeFuzzyController(fake_score_complexity))
        assert routed.loglikelihood_rolling([_FakeInstance("p")]) == [("8bit", "p")]

    def test_generate_until_dispatches_correctly(self, fake_score_complexity):
        fake_score_complexity["p"] = "16bit"
        tier_lms = {"16bit": _FakeTierLM("16bit")}
        routed = RoutedLM(tier_lms, _FakeFuzzyController(fake_score_complexity))
        assert routed.generate_until([_FakeInstance("p")]) == [("16bit", "p")]


# ----------------------------------------------------------------------
# _extract_scores
# ----------------------------------------------------------------------

class TestExtractScores:
    def test_prefers_acc_none_over_other_keys(self):
        task_results = {
            "mmlu": {"acc,none": 0.55, "acc_norm,none": 0.60, "acc_stderr,none": 0.02},
        }
        scores = AccuracyEvaluator._extract_scores(task_results)
        assert scores["mmlu"] == pytest.approx(0.55)

    def test_falls_back_to_acc_norm_none(self):
        task_results = {"hellaswag": {"acc_norm,none": 0.61, "acc_norm_stderr,none": 0.01}}
        scores = AccuracyEvaluator._extract_scores(task_results)
        assert scores["hellaswag"] == pytest.approx(0.61)

    def test_falls_back_to_first_numeric_metric_for_unknown_keys(self):
        task_results = {"weird_task": {"some_custom_metric,none": 0.77}}
        scores = AccuracyEvaluator._extract_scores(task_results)
        assert scores["weird_task"] == pytest.approx(0.77)

    def test_raises_when_no_numeric_metric_present(self):
        task_results = {"broken_task": {"note": "not a number"}}
        with pytest.raises(RuntimeError, match="no numeric metric"):
            AccuracyEvaluator._extract_scores(task_results)

    def test_multiple_tasks_reduced_independently(self):
        task_results = {
            "mmlu": {"acc,none": 0.5},
            "hellaswag": {"acc,none": 0.6},
        }
        scores = AccuracyEvaluator._extract_scores(task_results)
        assert scores == {"mmlu": pytest.approx(0.5), "hellaswag": pytest.approx(0.6)}


# ----------------------------------------------------------------------
# compute_routellm_metrics — populate .results / ._last_routing_log
# directly since evaluate_condition() requires a real model
# ----------------------------------------------------------------------

class TestComputeRoutellmMetrics:
    def test_returns_empty_without_prerequisite_conditions(self, evaluator):
        assert evaluator.compute_routellm_metrics() == {}

    def test_cpt_is_fraction_of_routed_calls_sent_to_16bit(self, evaluator):
        evaluator.results = {
            "always_4bit": {"overall": 0.50},
            "always_16bit": {"overall": 0.75},
            "routed": {"overall": 0.68},
        }
        evaluator._last_routing_log = [
            {"tier": "4bit"}, {"tier": "16bit"}, {"tier": "16bit"}, {"tier": "8bit"},
        ]
        metrics = evaluator.compute_routellm_metrics()
        assert metrics["CPT"] == pytest.approx(2 / 4)
        assert metrics["APGR"] == pytest.approx((0.68 - 0.50) / (0.75 - 0.50))
        assert metrics["strong_performance"] == pytest.approx(0.75)
        assert metrics["weak_performance"] == pytest.approx(0.50)
        assert metrics["routed_performance"] == pytest.approx(0.68)

    def test_cpt_is_zero_when_no_routing_log_recorded(self, evaluator):
        evaluator.results = {
            "always_4bit": {"overall": 0.50},
            "always_16bit": {"overall": 0.75},
            "routed": {"overall": 0.68},
        }
        # evaluator._last_routing_log left at its default empty list
        metrics = evaluator.compute_routellm_metrics()
        assert metrics["CPT"] == 0.0

    def test_apgr_is_zero_when_strong_does_not_beat_weak(self, evaluator):
        evaluator.results = {
            "always_4bit": {"overall": 0.60},
            "always_16bit": {"overall": 0.60},  # tied, not strictly greater
            "routed": {"overall": 0.60},
        }
        metrics = evaluator.compute_routellm_metrics()
        assert metrics["APGR"] == 0.0


# ----------------------------------------------------------------------
# save_results / get_summary — populate .results directly
# ----------------------------------------------------------------------

class TestSaveResults:
    def test_writes_expected_json_structure(self, evaluator, tmp_path):
        evaluator.results = {
            "always_4bit": {"mmlu": 0.4, "hellaswag": 0.5, "overall": 0.45},
            "always_16bit": {"mmlu": 0.7, "hellaswag": 0.8, "overall": 0.75},
            "routed": {"mmlu": 0.6, "hellaswag": 0.65, "overall": 0.625},
        }
        evaluator._last_routing_log = [{"tier": "16bit"}, {"tier": "4bit"}]

        evaluator.save_results()

        out_file = evaluator.output_dir / "accuracy_results.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "accuracy_by_condition" in data
        assert "routellm_metrics" in data
        assert data["accuracy_by_condition"]["always_4bit"]["overall"] == pytest.approx(0.45)
        assert data["routellm_metrics"]["CPT"] == pytest.approx(0.5)


class TestGetSummary:
    def test_summary_maps_condition_to_overall(self, evaluator):
        evaluator.results = {"always_4bit": {"mmlu": 0.4, "overall": 0.4}}
        assert evaluator.get_summary() == pytest.approx({"always_4bit": 0.4})

    def test_summary_empty_when_no_conditions_evaluated(self, evaluator):
        assert evaluator.get_summary() == {}


# ----------------------------------------------------------------------
# TIER_TO_SERVICE mapping sanity — guards against drifting away from
# models/model_pool.py's registration names
# ----------------------------------------------------------------------

def test_tier_to_service_matches_model_pool_registration_convention():
    assert TIER_TO_SERVICE == {
        "4bit": "local/4bit",
        "8bit": "local/8bit",
        "16bit": "local/16bit",
    }
