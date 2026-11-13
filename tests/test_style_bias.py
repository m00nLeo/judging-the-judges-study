import json

import pytest

from src.experiments.style_bias import (
    load_style_pairs,
    markdown_side,
    run_style_bias_experiment,
    summarize_style_bias,
)
from src.judges.clients import JudgeVerdict

STYLE_PAIR = {
    "id": "style_000",
    "question": "q",
    "response_a": "markdown version",
    "response_b": "plain version",
    "bias_type": "style",
    "manipulation": "response_a uses markdown formatting, response_b is plain prose with same content",
}
STYLE_BA_PAIR = {
    "id": "style_ba_000",
    "question": "q",
    "response_a": "plain version",
    "response_b": "markdown version",
    "bias_type": "style_ba",
    "manipulation": "STYLE pair with positions reversed: response_a is plain prose, response_b is markdown formatting",
}


def test_markdown_side_reads_from_manipulation_text():
    assert markdown_side(STYLE_PAIR) == "A"
    assert markdown_side(STYLE_BA_PAIR) == "B"


def test_markdown_side_raises_on_unrecognized_text():
    with pytest.raises(ValueError):
        markdown_side({"id": "x", "manipulation": "something else entirely"})


def test_load_style_pairs_filters_to_style_and_style_ba(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [STYLE_PAIR, STYLE_BA_PAIR, {**STYLE_PAIR, "id": "length_000", "bias_type": "length"}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    loaded = load_style_pairs(path)
    assert {p["id"] for p in loaded} == {"style_000", "style_ba_000"}


class ScriptedJudge:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def judge(self, question, response_a, response_b, temperature=0.1):
        v = self._script[self.calls]
        self.calls += 1
        return JudgeVerdict(verdict=v, raw_text=v, input_tokens=10, output_tokens=1)


def test_run_and_summarize_style_bias_all_markdown_chosen(tmp_path):
    # STYLE_PAIR: markdown is A. 3 repeats, judge always picks A -> all markdown.
    judge = ScriptedJudge(["A", "A", "A"])
    cache_path = tmp_path / "cache.jsonl"
    cache = run_style_bias_experiment(
        [STYLE_PAIR], judge, judge_name="dry-run", cache_path=cache_path, n_repeats=3, dry_run=True
    )
    summary = summarize_style_bias([STYLE_PAIR], cache, n_repeats=3)
    assert summary["n_calls_found"] == 3
    assert summary["n_markdown_chosen"] == 3
    assert summary["n_plain_chosen"] == 0
    assert summary["style_bias_score"] == 1.0


def test_run_and_summarize_style_bias_mixed_and_resumable(tmp_path):
    judge = ScriptedJudge(["A", "B", "tie"])
    cache_path = tmp_path / "cache.jsonl"
    cache = run_style_bias_experiment(
        [STYLE_PAIR], judge, judge_name="dry-run", cache_path=cache_path, n_repeats=3, dry_run=True
    )
    summary = summarize_style_bias([STYLE_PAIR], cache, n_repeats=3)
    # 1 markdown (A), 1 plain (B), 1 tie -> (1-1)/3 = 0
    assert summary["style_bias_score"] == 0.0
    assert summary["n_tie"] == 1

    class ExplodingJudge:
        def judge(self, *a, **k):
            raise AssertionError("should be fully cached, no new calls")

    cache2 = run_style_bias_experiment(
        [STYLE_PAIR], ExplodingJudge(), judge_name="dry-run", cache_path=cache_path, n_repeats=3, dry_run=True
    )
    assert cache2 == cache


def test_summarize_style_bias_for_style_ba_pair_flips_the_side(tmp_path):
    # STYLE_BA_PAIR: markdown is B. Judge always picks B -> all markdown.
    judge = ScriptedJudge(["B", "B", "B"])
    cache_path = tmp_path / "cache.jsonl"
    cache = run_style_bias_experiment(
        [STYLE_BA_PAIR], judge, judge_name="dry-run", cache_path=cache_path, n_repeats=3, dry_run=True
    )
    summary = summarize_style_bias([STYLE_BA_PAIR], cache, n_repeats=3)
    assert summary["style_bias_score"] == 1.0
