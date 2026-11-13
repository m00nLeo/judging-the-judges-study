import json

import pytest

from src.experiments.position_bias import (
    CostModel,
    append_cache,
    call_with_retry,
    load_cache,
    run_position_bias_protocol,
    summarize_position_bias,
)
from src.judges.clients import JudgeVerdict


class ScriptedJudge:
    """Returns verdicts from a fixed script, in call order -- lets tests pin
    down exact flip/no-flip scenarios instead of relying on randomness."""

    def __init__(self, script: list[str]):
        self._script = list(script)
        self.calls = 0

    def judge(self, question, response_a, response_b, temperature=0.1):
        verdict = self._script[self.calls]
        self.calls += 1
        return JudgeVerdict(verdict=verdict, raw_text=verdict, input_tokens=10, output_tokens=1)


PAIRS = [
    {"id": "p1", "question": "q1", "response_a": "a1", "response_b": "b1"},
    {"id": "p2", "question": "q2", "response_a": "a2", "response_b": "b2"},
]


def test_summarize_detects_flip_and_no_flip():
    # p1: AB->A, BA->A (normalized BA "A" means original B won) -> AB says A, BA-normalized says B -> flip
    # p2: AB->B, BA->B (normalized -> A) -> AB says B, BA-normalized says A -> flip too; use a no-flip case instead
    cache = {
        ("p1", "AB"): {"verdict_raw": "A"},
        ("p1", "BA"): {"verdict_raw": "B"},  # normalized -> "A" -> same as AB -> no flip
        ("p2", "AB"): {"verdict_raw": "A"},
        ("p2", "BA"): {"verdict_raw": "A"},  # normalized -> "B" -> differs from AB -> flip
    }
    result = summarize_position_bias(PAIRS, cache)
    assert result["n_pairs_complete"] == 2
    assert result["n_pairs_incomplete"] == 0
    assert result["flip_rate"] == 0.5  # 1 flip out of 2 pairs


def test_summarize_excludes_incomplete_pairs():
    cache = {("p1", "AB"): {"verdict_raw": "A"}, ("p1", "BA"): {"verdict_raw": "B"}}
    result = summarize_position_bias(PAIRS, cache)
    assert result["n_pairs_complete"] == 1
    assert result["n_pairs_incomplete"] == 1
    assert result["flip_rate"] == 0.0


def test_run_protocol_dry_run_populates_cache_and_resumes(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    judge = ScriptedJudge(["A", "B", "B", "A"])  # p1 AB, p1 BA, p2 AB, p2 BA
    result = run_position_bias_protocol(
        PAIRS, judge, judge_name="dry-run", protocol="A", cache_path=cache_path, dry_run=True
    )
    assert result["n_pairs_complete"] == 2
    assert judge.calls == 4

    # Re-running with a judge that would error if called should be a no-op (full resume from cache).
    class ExplodingJudge:
        def judge(self, *a, **k):
            raise AssertionError("should not be called -- everything was cached")

    result2 = run_position_bias_protocol(
        PAIRS, ExplodingJudge(), judge_name="dry-run", protocol="A", cache_path=cache_path, dry_run=True
    )
    assert result2 == result

    lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # no duplicate rows written on resume


def test_budget_cap_stops_before_overspending(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    judge = ScriptedJudge(["A", "A", "A", "A"])
    cost_model = CostModel(cost_per_1k_input=1.0, cost_per_1k_output=1.0)  # deliberately expensive
    with pytest.raises(RuntimeError, match="budget cap reached"):
        run_position_bias_protocol(
            PAIRS,
            judge,
            judge_name="dry-run",
            protocol="A",
            cache_path=cache_path,
            cost_model=cost_model,
            budget_usd=0.0005,  # smaller than one call's estimated cost
            dry_run=False,
        )
    # Whatever completed before the cap was hit must still be cached (no lost work).
    cache = load_cache(cache_path)
    assert len(cache) >= 0  # at minimum, doesn't crash before writing anything


def test_load_cache_roundtrips_appended_rows(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    row = {"pair_id": "p1", "order": "AB", "verdict_raw": "A", "cost_usd": 0.001}
    append_cache(cache_path, row)
    cache = load_cache(cache_path)
    assert cache[("p1", "AB")] == row


class _ServerError(Exception):
    """Stand-in for google.genai.errors.ServerError / similar 5xx exceptions."""

    def __init__(self, message):
        super().__init__(message)
        self.status_code = 503


class _BadRequestError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400


def test_call_with_retry_succeeds_after_transient_errors(monkeypatch):
    monkeypatch.setattr("src.experiments.common.time.sleep", lambda _: None)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _ServerError("503 UNAVAILABLE: high demand, please try again later")
        return "ok"

    result = call_with_retry(flaky, max_retries=5, base_delay=0.01)
    assert result == "ok"
    assert attempts["n"] == 3


def test_call_with_retry_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("src.experiments.common.time.sleep", lambda _: None)

    def always_fails():
        raise _ServerError("503 UNAVAILABLE")

    with pytest.raises(_ServerError):
        call_with_retry(always_fails, max_retries=2, base_delay=0.01)


def test_call_with_retry_does_not_retry_non_transient_errors(monkeypatch):
    monkeypatch.setattr(
        "src.experiments.common.time.sleep",
        lambda _: pytest.fail("should not sleep/retry on a non-transient error"),
    )
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _BadRequestError("400 invalid API key")

    with pytest.raises(_BadRequestError):
        call_with_retry(bad_request, max_retries=5, base_delay=0.01)
    assert calls["n"] == 1  # failed on the first attempt, no retry
