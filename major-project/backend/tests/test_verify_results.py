"""
Tests for training/scripts/verify_results.py — the automated credibility
gate. Pure functions over CSV/JSON files, no GPU dependency. This is the
highest-priority target per test-agent.md: a bug here would let a bogus
result get reported as PASS/publishable.

`findings` is module-level global state (accumulated by check()), so every
test clears it first via the `clean_findings` autouse fixture.
"""

import csv
import json

import pytest

import verify_results as vr


@pytest.fixture(autouse=True)
def clean_findings():
    vr.findings.clear()
    yield
    vr.findings.clear()


# --------------------------------------------------------------------------
# helpers to build the on-disk fixture layout verify_results.py expects
# --------------------------------------------------------------------------

ENERGY_FIELDS = ["tier", "energy_j", "tokens_out", "j_per_token", "run"]
ACCURACY_FIELDS = ["metric", "value", "task", "tier", "variant"]
ROUTING_SUMMARY_FIELDS = ["condition", "accuracy", "j_per_request"]


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_energy_rows(tier, n, energy_j=50.0, tokens_out=50, j_per_token=1.0, run="1"):
    return [
        {
            "tier": tier,
            "energy_j": energy_j,
            "tokens_out": tokens_out,
            "j_per_token": j_per_token,
            "run": run,
        }
        for _ in range(n)
    ]


def write_full_energy_fixture(results_dir, hw_energy_counter=True, runs=("1", "2", "3")):
    """A complete, plausible energy fixture that should PASS every check."""
    rows = []
    for tier in ["4bit", "8bit", "16bit"]:
        for run in runs:
            rows += make_energy_rows(tier, 10, run=run)  # 10 * 3 runs = 30/tier
    _write_csv(results_dir / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
    hw = {"gpu": "Tesla T4", "driver": "550.54", "energy_counter": hw_energy_counter}
    (results_dir / "energy_logs" / "hardware_info.json").write_text(json.dumps(hw))
    return rows


def write_full_accuracy_fixture(results_dir):
    rows = [
        {"metric": "acc,none", "value": 0.72, "task": "mmlu", "tier": "16bit", "variant": "base"},
        {"metric": "acc,none", "value": 0.42, "task": "mmlu", "tier": "4bit", "variant": "base"},
        {"metric": "acc,none", "value": 0.58, "task": "mmlu", "tier": "8bit", "variant": "base"},
    ]
    _write_csv(results_dir / "accuracy_logs" / "accuracy_summary.csv", ACCURACY_FIELDS, rows)
    return rows


def write_full_routing_fixture(results_dir, n_prompts=300):
    conditions = ["static_4bit", "static_8bit", "static_16bit", "fuzzy_router",
                  "random_matched", "threshold_router", "oracle", "oracle_cascade"]
    # accuracies: oracle must be the max; static_4bit energy <= static_16bit;
    # fuzzy_router >= random_matched
    acc = {
        "static_4bit": 0.50, "static_8bit": 0.65, "static_16bit": 0.75,
        "fuzzy_router": 0.70, "random_matched": 0.60, "threshold_router": 0.68,
        "oracle": 0.80, "oracle_cascade": 0.78,
    }
    energy = {
        "static_4bit": 10.0, "static_8bit": 20.0, "static_16bit": 40.0,
        "fuzzy_router": 18.0, "random_matched": 22.0, "threshold_router": 19.0,
        "oracle": 25.0, "oracle_cascade": 24.0,
    }
    rows = [{"condition": c, "accuracy": acc[c], "j_per_request": energy[c]} for c in conditions]
    _write_csv(results_dir / "routing_logs" / "routing_conditions_summary.csv",
               ROUTING_SUMMARY_FIELDS, rows)

    praws = []
    for i in range(n_prompts):
        for tier in ["4bit", "8bit", "16bit"]:
            praws.append({"prompt_id": str(i), "tier": tier})
    _write_csv(results_dir / "routing_logs" / "routing_per_prompt.csv",
               ["prompt_id", "tier"], praws)
    return rows, praws


def levels(area_substr=None):
    """Return [(level, area, msg), ...] from vr.findings, optionally
    filtered to areas containing area_substr."""
    if area_substr is None:
        return list(vr.findings)
    return [f for f in vr.findings if area_substr in f[1]]


def has_fail():
    return any(lvl == "FAIL" for lvl, _, _ in vr.findings)


def has_warn():
    return any(lvl == "WARN" for lvl, _, _ in vr.findings)


# --------------------------------------------------------------------------
# check() bookkeeping
# --------------------------------------------------------------------------

class TestCheck:
    def test_pass_appends_pass_level(self):
        ok = vr.check("FAIL", "area", True, "good", "bad")
        assert ok is True
        assert vr.findings == [("PASS", "area", "good")]

    def test_fail_appends_given_level_and_fail_message(self):
        ok = vr.check("FAIL", "area", False, "good", "bad")
        assert ok is False
        assert vr.findings == [("FAIL", "area", "bad")]

    def test_warn_level_on_failure_is_warn_not_fail(self):
        ok = vr.check("WARN", "area", False, "good", "bad")
        assert ok is False
        assert vr.findings == [("WARN", "area", "bad")]

    def test_returns_the_ok_value(self):
        assert vr.check("FAIL", "a", True, "p", "f") is True
        assert vr.check("FAIL", "a", False, "p", "f") is False


# --------------------------------------------------------------------------
# verify_energy
# --------------------------------------------------------------------------

class TestVerifyEnergy:
    def test_missing_file_is_fail_and_returns_none(self, tmp_path):
        result = vr.verify_energy(tmp_path)
        assert result is None
        assert has_fail()
        assert "missing" in levels("energy")[0][2]

    def test_empty_file_header_only_is_fail(self, tmp_path):
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, [])
        vr.verify_energy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("EMPTY" in m for m in fails)

    def test_too_few_samples_per_tier_is_fail(self, tmp_path):
        rows = make_energy_rows("4bit", vr.MIN_N_PER_TIER - 1)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("only n=" in m and "4bit" in m for m in fails)

    def test_exactly_min_n_per_tier_passes_that_check(self, tmp_path):
        rows = make_energy_rows("4bit", vr.MIN_N_PER_TIER)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        n_msgs = [m for lvl, area, m in vr.findings if lvl == "PASS" and "n=" in m and "4bit" in m]
        assert any(f"n={vr.MIN_N_PER_TIER}" in m for m in n_msgs)

    def test_zero_or_negative_energy_is_fail(self, tmp_path):
        rows = make_energy_rows("4bit", 30, energy_j=0.0)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("zero/negative energy" in m for m in fails)

    def test_zero_tokens_is_fail(self, tmp_path):
        rows = make_energy_rows("4bit", 30, tokens_out=0)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("zero token counts" in m for m in fails)

    def test_jpt_below_window_is_fail(self, tmp_path):
        rows = make_energy_rows("4bit", 30, j_per_token=vr.JPT_MIN / 10)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("OUTSIDE plausible" in m for m in fails)

    def test_jpt_above_window_is_fail(self, tmp_path):
        rows = make_energy_rows("4bit", 30, j_per_token=vr.JPT_MAX * 10)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("OUTSIDE plausible" in m for m in fails)

    def test_jpt_within_window_passes(self, tmp_path):
        rows = make_energy_rows("4bit", 30, j_per_token=(vr.JPT_MIN + vr.JPT_MAX) / 2)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert not any("plausib" in m.lower() and "OUTSIDE" in m for m in fails)

    def test_noisy_ci_is_warn(self, tmp_path):
        # Alternate two very different j_per_token values -> high stdev
        # relative to mean -> CI half-width > 25% of mean.
        rows = []
        for i in range(30):
            jpt = 1.0 if i % 2 == 0 else 20.0
            rows += make_energy_rows("4bit", 1, j_per_token=jpt)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("noisy" in m for m in warns)

    def test_tight_ci_identical_values_passes(self, tmp_path):
        rows = make_energy_rows("4bit", 30, j_per_token=1.0)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        passes = [m for lvl, area, m in vr.findings if lvl == "PASS"]
        assert any("is tight" in m for m in passes)

    def test_fewer_than_three_runs_is_warn(self, tmp_path):
        rows = make_energy_rows("4bit", 30, run="1")  # single run id
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        vr.verify_energy(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("repeat 3x" in m for m in warns)

    def test_missing_hardware_info_is_warn_and_returns_none(self, tmp_path):
        rows = make_energy_rows("4bit", 30)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        result = vr.verify_energy(tmp_path)
        assert result is None
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("hardware provenance unrecorded" in m for m in warns)

    def test_power_sampling_fallback_is_warn(self, tmp_path):
        rows = make_energy_rows("4bit", 30)
        _write_csv(tmp_path / "energy_logs" / "energy_per_inference.csv", ENERGY_FIELDS, rows)
        hw = {"gpu": "Tesla T4", "energy_counter": False}
        (tmp_path / "energy_logs" / "hardware_info.json").write_text(json.dumps(hw))
        result = vr.verify_energy(tmp_path)
        assert result == hw
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("power-sampling fallback" in m for m in warns)

    def test_full_pass_path_all_three_tiers(self, tmp_path):
        write_full_energy_fixture(tmp_path)
        hw = vr.verify_energy(tmp_path)
        assert hw is not None and hw["energy_counter"] is True
        assert not has_fail()
        assert not has_warn()  # tight CI, 3 runs, hw counter present -> no warnings either


# --------------------------------------------------------------------------
# verify_accuracy
# --------------------------------------------------------------------------

class TestVerifyAccuracy:
    def test_missing_file_is_fail(self, tmp_path):
        vr.verify_accuracy(tmp_path)
        assert has_fail()

    def test_no_accuracy_rows_is_fail(self, tmp_path):
        # only stderr rows -> filtered out -> zero acc rows
        rows = [{"metric": "acc_stderr,none", "value": 0.01, "task": "mmlu",
                  "tier": "4bit", "variant": "base"}]
        _write_csv(tmp_path / "accuracy_logs" / "accuracy_summary.csv", ACCURACY_FIELDS, rows)
        vr.verify_accuracy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("no accuracy metrics" in m for m in fails)

    def test_out_of_range_value_is_fail(self, tmp_path):
        rows = [{"metric": "acc,none", "value": 1.5, "task": "mmlu",
                  "tier": "4bit", "variant": "base"}]
        _write_csv(tmp_path / "accuracy_logs" / "accuracy_summary.csv", ACCURACY_FIELDS, rows)
        vr.verify_accuracy(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("outside [0, 1]" in m for m in fails)

    def test_4bit_beats_16bit_by_more_than_2pts_is_warn(self, tmp_path):
        rows = [
            {"metric": "acc,none", "value": 0.90, "task": "mmlu", "tier": "4bit", "variant": "base"},
            {"metric": "acc,none", "value": 0.50, "task": "mmlu", "tier": "16bit", "variant": "base"},
        ]
        _write_csv(tmp_path / "accuracy_logs" / "accuracy_summary.csv", ACCURACY_FIELDS, rows)
        vr.verify_accuracy(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("beats fp16" in m for m in warns)

    def test_sanity_check_only_applies_to_base_variant(self, tmp_path):
        # A variant != "base" row where 4-bit beats 16-bit should NOT
        # trigger the fp16-vs-4bit sanity WARN (only "base" rows are
        # compared) -- and since no "base" rows exist for this task at
        # all, the per-task loop should have nothing to compare.
        rows = [
            {"metric": "acc,none", "value": 0.90, "task": "mmlu", "tier": "4bit", "variant": "finetuned"},
            {"metric": "acc,none", "value": 0.50, "task": "mmlu", "tier": "16bit", "variant": "finetuned"},
        ]
        _write_csv(tmp_path / "accuracy_logs" / "accuracy_summary.csv", ACCURACY_FIELDS, rows)
        vr.verify_accuracy(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert not any("beats fp16" in m for m in warns)

    def test_full_pass_path(self, tmp_path):
        write_full_accuracy_fixture(tmp_path)
        vr.verify_accuracy(tmp_path)
        assert not has_fail()
        assert not has_warn()


# --------------------------------------------------------------------------
# verify_routing
# --------------------------------------------------------------------------

class TestVerifyRouting:
    def test_missing_summary_is_fail(self, tmp_path):
        vr.verify_routing(tmp_path)
        assert has_fail()

    def test_missing_condition_is_fail_and_stops_further_checks(self, tmp_path):
        rows = [{"condition": "static_4bit", "accuracy": 0.5, "j_per_request": 10.0}]
        _write_csv(tmp_path / "routing_logs" / "routing_conditions_summary.csv",
                   ROUTING_SUMMARY_FIELDS, rows)
        vr.verify_routing(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("MISSING from summary" in m for m in fails)
        # per_prompt.csv was never even looked at -> no finding mentions it
        assert not any("per-prompt" in m for _, _, m in vr.findings)

    def test_zero_energy_condition_is_fail(self, tmp_path):
        write_full_routing_fixture(tmp_path)
        # overwrite with a zero-energy condition
        conditions = ["static_4bit", "static_8bit", "static_16bit", "fuzzy_router",
                      "random_matched", "threshold_router", "oracle", "oracle_cascade"]
        rows = [{"condition": c, "accuracy": 0.5, "j_per_request": 0.0} for c in conditions]
        rows[-1]["accuracy"] = 0.9  # keep oracle-esque max plausible elsewhere
        _write_csv(tmp_path / "routing_logs" / "routing_conditions_summary.csv",
                   ROUTING_SUMMARY_FIELDS, rows)
        vr.verify_routing(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("zero-energy condition" in m for m in fails)

    def test_oracle_not_max_is_warn(self, tmp_path):
        write_full_routing_fixture(tmp_path)
        conditions = ["static_4bit", "static_8bit", "static_16bit", "fuzzy_router",
                      "random_matched", "threshold_router", "oracle", "oracle_cascade"]
        rows = [{"condition": c, "accuracy": 0.5, "j_per_request": 10.0} for c in conditions]
        for r in rows:
            if r["condition"] == "fuzzy_router":
                r["accuracy"] = 0.99  # beats oracle -> derivation bug
            if r["condition"] == "oracle":
                r["accuracy"] = 0.80
        _write_csv(tmp_path / "routing_logs" / "routing_conditions_summary.csv",
                   ROUTING_SUMMARY_FIELDS, rows)
        vr.verify_routing(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("derivation bug" in m for m in warns)

    def test_4bit_more_energy_than_16bit_is_warn(self, tmp_path):
        rows, praws = write_full_routing_fixture(tmp_path)
        conditions = ["static_4bit", "static_8bit", "static_16bit", "fuzzy_router",
                      "random_matched", "threshold_router", "oracle", "oracle_cascade"]
        energy = {c: 20.0 for c in conditions}
        energy["static_4bit"] = 100.0  # 4-bit now MORE expensive than 16-bit
        energy["static_16bit"] = 20.0
        acc = {c: 0.6 for c in conditions}
        acc["oracle"] = 0.9
        new_rows = [{"condition": c, "accuracy": acc[c], "j_per_request": energy[c]} for c in conditions]
        _write_csv(tmp_path / "routing_logs" / "routing_conditions_summary.csv",
                   ROUTING_SUMMARY_FIELDS, new_rows)
        vr.verify_routing(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("uses MORE energy than fp16" in m for m in warns)

    def test_fuzzy_router_below_random_matched_is_warn(self, tmp_path):
        conditions = ["static_4bit", "static_8bit", "static_16bit", "fuzzy_router",
                      "random_matched", "threshold_router", "oracle", "oracle_cascade"]
        acc = {c: 0.6 for c in conditions}
        acc["oracle"] = 0.9
        acc["fuzzy_router"] = 0.40
        acc["random_matched"] = 0.65
        energy = {c: 20.0 for c in conditions}
        rows = [{"condition": c, "accuracy": acc[c], "j_per_request": energy[c]} for c in conditions]
        _write_csv(tmp_path / "routing_logs" / "routing_conditions_summary.csv",
                   ROUTING_SUMMARY_FIELDS, rows)
        vr.verify_routing(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("does NOT beat matched random" in m for m in warns)

    def test_missing_per_prompt_csv_is_warn(self, tmp_path):
        write_full_routing_fixture(tmp_path)
        (tmp_path / "routing_logs" / "routing_per_prompt.csv").unlink()
        vr.verify_routing(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any("raw artifact needed for release" in m for m in warns)

    def test_incomplete_grid_is_fail(self, tmp_path):
        write_full_routing_fixture(tmp_path, n_prompts=10)
        # Drop one row so it's not a clean 3x multiple of distinct prompt ids
        per_prompt_path = tmp_path / "routing_logs" / "routing_per_prompt.csv"
        praws = list(csv.DictReader(open(per_prompt_path, encoding="utf-8")))
        praws.pop()
        _write_csv(per_prompt_path, ["prompt_id", "tier"], praws)
        vr.verify_routing(tmp_path)
        fails = [m for lvl, area, m in vr.findings if lvl == "FAIL"]
        assert any("incomplete measurement grid" in m for m in fails)

    def test_fewer_than_300_prompts_is_warn(self, tmp_path):
        write_full_routing_fixture(tmp_path, n_prompts=50)
        vr.verify_routing(tmp_path)
        warns = [m for lvl, area, m in vr.findings if lvl == "WARN"]
        assert any(">= 300" in m for m in warns)

    def test_full_pass_path_300_prompts(self, tmp_path):
        write_full_routing_fixture(tmp_path, n_prompts=300)
        vr.verify_routing(tmp_path)
        assert not has_fail()
        assert not has_warn()


# --------------------------------------------------------------------------
# write_report()
# --------------------------------------------------------------------------

class TestWriteReport:
    def test_all_pass_verdict(self, tmp_path):
        (tmp_path).mkdir(exist_ok=True)
        vr.findings.append(("PASS", "area", "ok"))
        vr.write_report(tmp_path, hw=None)
        text = (tmp_path / "results_validation.md").read_text(encoding="utf-8")
        assert "ALL CHECKS PASSED" in text

    def test_warn_verdict_when_no_fail_present(self, tmp_path):
        vr.findings.append(("PASS", "area", "ok"))
        vr.findings.append(("WARN", "area", "careful"))
        vr.write_report(tmp_path, hw=None)
        text = (tmp_path / "results_validation.md").read_text(encoding="utf-8")
        assert "PUBLISHABLE WITH CAVEATS" in text
        assert "NOT PUBLISHABLE" not in text

    def test_fail_verdict_when_any_fail_present(self, tmp_path):
        vr.findings.append(("PASS", "area", "ok"))
        vr.findings.append(("WARN", "area", "careful"))
        vr.findings.append(("FAIL", "area", "broken"))
        vr.write_report(tmp_path, hw=None)
        text = (tmp_path / "results_validation.md").read_text(encoding="utf-8")
        assert "NOT PUBLISHABLE" in text

    def test_report_includes_hardware_line_when_provided(self, tmp_path):
        vr.findings.append(("PASS", "area", "ok"))
        vr.write_report(tmp_path, hw={"gpu": "Tesla T4", "driver": "550.54", "energy_counter": True})
        text = (tmp_path / "results_validation.md").read_text(encoding="utf-8")
        assert "Tesla T4" in text
        assert "550.54" in text
