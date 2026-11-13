from src.experiments.format_normalization import (
    STRATEGIES,
    _parse_last_line,
    _parse_simple,
    run_format_normalization_experiment,
    strip_markdown,
    summarize_agreement_by_strategy,
    summarize_style_bias_by_strategy,
)
from src.judges.clients import JudgeVerdict


def test_strip_markdown_removes_common_syntax():
    text = "# Header\n\n**bold** and *italic* and `code` and\n\n* bullet one\n* bullet two\n\n---\n[link](http://x)"
    stripped = strip_markdown(text)
    assert "#" not in stripped
    assert "**" not in stripped
    assert "`" not in stripped
    assert "bold" in stripped and "italic" in stripped and "code" in stripped
    assert "link" in stripped and "http://x" not in stripped


def test_parse_last_line_extracts_final_verdict_from_cot_text():
    text = "The response A is more detailed and correct.\n\nFinal verdict: A"
    assert _parse_last_line(text) == "A"
    assert _parse_last_line("blah blah\nFinal verdict: tie") == "tie"


def test_parse_simple_matches_first_letter():
    assert _parse_simple("A") == "A"
    assert _parse_simple("B, because...") == "B"
    assert _parse_simple("tie") == "tie"


STYLE_PAIR = {
    "id": "style_000",
    "question": "q",
    "response_a": "**markdown** version",
    "response_b": "plain version",
    "bias_type": "style",
    "manipulation": "response_a uses markdown formatting, response_b is plain prose with same content",
}

MT_BENCH_PAIR = {
    "id": "mt_bench_1",
    "question": "q",
    "response_a": "a",
    "response_b": "b",
    "human_preference": "A",
}


class ScriptedJudge:
    """complete_raw-based fake, keyed off a dict so test order doesn't matter."""

    def __init__(self, responses_by_call_index):
        self._responses = list(responses_by_call_index)
        self.calls = 0
        self.seen_prompts = []
        self.seen_max_tokens = []

    def complete_raw(self, prompt, temperature=0.1, max_tokens=5):
        self.seen_prompts.append(prompt)
        self.seen_max_tokens.append(max_tokens)
        text = self._responses[self.calls]
        self.calls += 1
        return text, 10, 2


def test_run_format_normalization_uses_right_prompt_and_max_tokens_per_strategy(tmp_path):
    # Order of (pair, strategy) iteration is pair-major then strategy-major per build_items.
    judge = ScriptedJudge(["A", "A", "A", "Final verdict: A"])  # B0, N1, N2, S5
    cache_path = tmp_path / "cache.jsonl"
    cache = run_format_normalization_experiment(
        [STYLE_PAIR], judge, judge_name="dry-run", cache_path=cache_path,
        strategies=("B0", "N1", "N2", "S5"), dry_run=True,
    )
    assert len(cache) == 4
    assert judge.seen_max_tokens == [5, 5, 5, 300]
    # N1's prompt must have stripped markdown out of response_a.
    n1_prompt = judge.seen_prompts[1]
    assert "**" not in n1_prompt
    assert "markdown" in n1_prompt  # text content preserved, just syntax stripped
    # N2's prompt must contain the ignore-formatting instruction.
    n2_prompt = judge.seen_prompts[2]
    assert "Ignore all formatting" in n2_prompt
    # All four cached verdicts should parse to "A".
    assert all(row["verdict_raw"] == "A" for row in cache.values())


def test_summarize_style_bias_by_strategy(tmp_path):
    cache = {
        ("style_000", "B0"): {"verdict_raw": "A"},  # A is markdown side -> markdown chosen
        ("style_000", "N2"): {"verdict_raw": "B"},  # plain chosen
    }
    result = summarize_style_bias_by_strategy([STYLE_PAIR], cache, strategies=("B0", "N2"))
    assert result["B0"]["style_bias_score"] == 1.0
    assert result["N2"]["style_bias_score"] == -1.0


def test_summarize_agreement_by_strategy_skips_pairs_without_human_preference(tmp_path):
    cache = {
        ("mt_bench_1", "B0"): {"verdict_raw": "A"},
        ("mt_bench_1", "N1"): {"verdict_raw": "B"},
    }
    result = summarize_agreement_by_strategy([MT_BENCH_PAIR], cache, strategies=("B0", "N1"))
    assert result["B0"]["agreement_rate"] == 1.0
    assert result["N1"]["agreement_rate"] == 0.0
