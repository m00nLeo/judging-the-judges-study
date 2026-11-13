"""E5: does format normalization (N1, N2) beat CoT (S5) at reducing style
bias without hurting agreement? (master plan mục 8, W3)

Master plan: "Thêm chiến lược N1: bỏ markdown trước khi chấm, và N2: chỉ dẫn
'bỏ qua định dạng'; so với B0 và S5 trên STYLE, MT-Bench 400, LLMBar 200."

Risk the master plan itself flags: stripping markdown (N1) can destroy
structural content (tables, code blocks) on MT-Bench -- strip_markdown()
below is a crude regex pass, NOT a markdown parser; hand-check a sample of
its output on MT-Bench/LLMBar pairs before trusting N1 at scale there. It is
safer on the STYLE set, whose pairs are prose-only by construction.
"""

import re
from pathlib import Path

from src.experiments.common import CostModel, run_cached_calls
from src.judges.clients import JudgeVerdict

_MARKDOWN_PATTERNS = [
    (re.compile(r"```.*?```", re.DOTALL), " "),  # fenced code blocks
    (re.compile(r"`([^`]+)`"), r"\1"),  # inline code
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),  # headers
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),  # bold
    (re.compile(r"\*([^*]+)\*"), r"\1"),  # italic
    (re.compile(r"^[*\-+]\s+", re.MULTILINE), ""),  # bullet markers
    (re.compile(r"^\d+\.\s+", re.MULTILINE), ""),  # numbered list markers
    (re.compile(r"^-{3,}\s*$", re.MULTILINE), ""),  # horizontal rules
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # links -> visible text
    (re.compile(r"\|", re.MULTILINE), " "),  # table pipes (crude, flattens tables)
]


def strip_markdown(text: str) -> str:
    """Crude regex-based markdown stripper. See module docstring for the
    known limitation on tables/code blocks.
    """
    out = text
    for pattern, repl in _MARKDOWN_PATTERNS:
        out = pattern.sub(repl, out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


BASE_PROMPT = (
    "You are judging which of two responses better answers the question. "
    'Reply with exactly one token: "A", "B", or "tie".\n\n'
    "Question: {question}\n\nResponse A: {response_a}\n\nResponse B: {response_b}\n\n"
    "Verdict (A, B, or tie):"
)

IGNORE_FORMAT_PROMPT = (
    "You are judging which of two responses better answers the question. "
    "Ignore all formatting, markdown, bullet points, bold text, and visual "
    "style -- judge ONLY the substance of the content. "
    'Reply with exactly one token: "A", "B", or "tie".\n\n'
    "Question: {question}\n\nResponse A: {response_a}\n\nResponse B: {response_b}\n\n"
    "Verdict (A, B, or tie):"
)

COT_PROMPT = (
    "You are judging which of two responses better answers the question. "
    "Think step by step about the content quality, accuracy, and helpfulness "
    "of each response, ignoring formatting. Then end your reply with a final "
    'line in exactly this form: "Final verdict: A" or "Final verdict: B" '
    'or "Final verdict: tie".\n\n'
    "Question: {question}\n\nResponse A: {response_a}\n\nResponse B: {response_b}\n\n"
    "Reasoning:"
)

# strip_md: whether to run strip_markdown() on response_a/b before formatting
# the prompt; parse: "simple" (first-letter token) or "last_line" (CoT -- the
# model is asked to end with "Final verdict: X" on its own line).
STRATEGIES = {
    "B0": {"prompt": BASE_PROMPT, "max_tokens": 5, "parse": "simple", "strip_md": False},
    "N1": {"prompt": BASE_PROMPT, "max_tokens": 5, "parse": "simple", "strip_md": True},
    "N2": {"prompt": IGNORE_FORMAT_PROMPT, "max_tokens": 5, "parse": "simple", "strip_md": False},
    "S5": {"prompt": COT_PROMPT, "max_tokens": 300, "parse": "last_line", "strip_md": False},
}


def _parse_simple(text: str) -> str:
    t = text.strip().upper()
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return "tie"


def _parse_last_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else text
    last = re.sub(r"(?i)^final verdict:\s*", "", last)
    return _parse_simple(last)


def build_items(pairs: list[dict], strategies: tuple[str, ...]) -> list[dict]:
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"unknown strategy {s!r}, expected one of {sorted(STRATEGIES)}")
    return [
        {"pair_id": pair["id"], "strategy": strategy, "pair": pair}
        for pair in pairs
        for strategy in strategies
    ]


def run_format_normalization_experiment(
    pairs: list[dict],
    judge,
    judge_name: str,
    cache_path: Path,
    strategies: tuple[str, ...] = ("B0", "N1", "N2", "S5"),
    cost_model: CostModel | None = None,
    budget_usd: float = 1.0,
    dry_run: bool = False,
    verbose: bool = True,
    max_workers: int = 1,
) -> dict:
    """Judges every pair under every strategy, caching each call keyed by
    (pair_id, strategy). `judge` must implement complete_raw(prompt,
    temperature, max_tokens) -> (text, input_tokens, output_tokens) -- see
    src.judges.clients, every provider class there already has it.
    """
    items = build_items(pairs, strategies)

    def call_fn(item):
        pair = item["pair"]
        spec = STRATEGIES[item["strategy"]]
        resp_a = strip_markdown(pair["response_a"]) if spec["strip_md"] else pair["response_a"]
        resp_b = strip_markdown(pair["response_b"]) if spec["strip_md"] else pair["response_b"]
        prompt = spec["prompt"].format(question=pair["question"], response_a=resp_a, response_b=resp_b)
        text, in_tok, out_tok = judge.complete_raw(prompt, temperature=0.1, max_tokens=spec["max_tokens"])
        verdict = _parse_last_line(text) if spec["parse"] == "last_line" else _parse_simple(text)
        return JudgeVerdict(verdict=verdict, raw_text=text, input_tokens=in_tok, output_tokens=out_tok)

    def row_fn(item, result, cost):
        return {
            "pair_id": item["pair_id"],
            "strategy": item["strategy"],
            "judge": judge_name,
            "verdict_raw": result.verdict,
            "raw_text": result.raw_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": cost,
        }

    return run_cached_calls(
        items,
        key_fields=("pair_id", "strategy"),
        call_fn=call_fn,
        row_fn=row_fn,
        cache_path=cache_path,
        cost_model=cost_model,
        budget_usd=budget_usd,
        dry_run=dry_run,
        verbose=verbose,
        label=f"E5/{judge_name}",
        max_workers=max_workers,
    )


def summarize_style_bias_by_strategy(
    style_pairs: list[dict], cache: dict, strategies: tuple[str, ...]
) -> dict:
    """Style bias score (same convention as src.experiments.style_bias) per
    strategy, restricted to pairs with bias_type in {style, style_ba}.
    """
    from src.experiments.style_bias import markdown_side

    by_id = {pair["id"]: pair for pair in style_pairs}
    out = {}
    for strategy in strategies:
        n_markdown = n_plain = n_tie = 0
        for (pair_id, s), row in cache.items():
            if s != strategy or pair_id not in by_id:
                continue
            side = markdown_side(by_id[pair_id])
            verdict = row["verdict_raw"]
            if verdict == "tie":
                n_tie += 1
            elif verdict == side:
                n_markdown += 1
            else:
                n_plain += 1
        total = n_markdown + n_plain + n_tie
        out[strategy] = {
            "n_calls": total,
            "style_bias_score": ((n_markdown - n_plain) / total) if total else None,
        }
    return out


def summarize_agreement_by_strategy(
    benchmark_pairs: list[dict], cache: dict, strategies: tuple[str, ...]
) -> dict:
    """Agreement rate with human_preference per strategy, for benchmark
    pairs that carry that field (MT-Bench, LLMBar). Pairs with
    human_preference is None are skipped, not counted as disagreement.
    """
    by_id = {pair["id"]: pair for pair in benchmark_pairs}
    out = {}
    for strategy in strategies:
        n_agree = n_total = 0
        for (pair_id, s), row in cache.items():
            if s != strategy or pair_id not in by_id:
                continue
            human = by_id[pair_id].get("human_preference")
            if human is None:
                continue
            n_total += 1
            if row["verdict_raw"] == human:
                n_agree += 1
        out[strategy] = {"n_calls": n_total, "agreement_rate": (n_agree / n_total) if n_total else None}
    return out
