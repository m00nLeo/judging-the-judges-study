"""E2: position bias via two protocols (master plan mục 8, W9).

Protocol A ("giống Soumik"): the 50 bias_type == "position" pairs from
data/custom/controlled_pairs.jsonl -- each constructed so an unbiased judge
should answer "tie".
Protocol B ("AB+BA kiểu Norman"): a random sample of data/raw/mt_bench/mt_bench.jsonl,
ordinary pairs not specially constructed for position bias.

Both protocols call the judge twice per pair -- once as (A, B), once swapped
as (B, A) -- and compute:
  - p_first: fraction of calls where the judge picked whichever response was
    shown FIRST in that call (0.5 = unbiased; this is W9's "|P(A) - 0.5|" up
    to the 0.5 recentering)
  - flip_rate: fraction of pairs where swapping order changed the verdict,
    after normalizing the swapped-order verdict back to original A/B labels

Every call is appended to a per-run jsonl cache keyed by (pair_id, order), so
an interrupted Colab session resumes without re-paying for completed calls.
A hard USD budget cap stops the run before the next call would exceed it.
Transient provider errors (rate limits, 5xx, "model overloaded") are retried
with exponential backoff (see call_with_retry); anything else fails immediately.
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_MESSAGE_SNIPPETS = (
    "503",
    "429",
    "unavailable",
    "overloaded",
    "rate limit",
    "rate_limit",
    "timeout",
    "temporarily",
    "high demand",
)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in RETRYABLE_STATUS_CODES:
        return True
    text = str(exc).lower()
    return any(snippet in text for snippet in RETRYABLE_MESSAGE_SNIPPETS)


def call_with_retry(fn, max_retries: int = 5, base_delay: float = 2.0):
    """Retries `fn()` with exponential backoff on transient provider errors
    (rate limits, 5xx, "model overloaded") -- not on anything else, so a bad
    API key or a malformed request still fails immediately instead of
    burning 5 retries on something that will never succeed.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_retryable(e) or attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))


@dataclass
class CostModel:
    cost_per_1k_input: float
    cost_per_1k_output: float

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * self.cost_per_1k_input + (
            output_tokens / 1000
        ) * self.cost_per_1k_output


def load_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cache[(row["pair_id"], row["order"])] = row
    return cache


def append_cache(cache_path: Path, row: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_position_pairs_protocol_a(controlled_pairs_path: Path) -> list[dict]:
    pairs = []
    with controlled_pairs_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["bias_type"] == "position":
                pairs.append(row)
    return pairs


def load_mt_bench_sample_protocol_b(mt_bench_path: Path, n: int, seed: int = 42) -> list[dict]:
    rows = []
    with mt_bench_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def run_position_bias_protocol(
    pairs: list[dict],
    judge,
    judge_name: str,
    protocol: str,
    cache_path: Path,
    cost_model: CostModel | None = None,
    budget_usd: float = 1.0,
    dry_run: bool = False,
) -> dict:
    """Runs both orders for every pair, respecting cache and budget cap.

    `judge` must already be constructed (see src.judges.clients.build_judge)
    so tests can inject a fake without this function knowing about dry_run
    internals; `dry_run` here only controls whether cost is tracked.

    Raises RuntimeError before a call that would push spend over budget_usd.
    """
    cache = load_cache(cache_path)
    spent_usd = sum(row.get("cost_usd", 0.0) for row in cache.values())

    for pair in pairs:
        pair_id = pair["id"]
        for order, (resp_first, resp_second) in [
            ("AB", (pair["response_a"], pair["response_b"])),
            ("BA", (pair["response_b"], pair["response_a"])),
        ]:
            key = (pair_id, order)
            if key in cache:
                continue
            if not dry_run and cost_model is not None:
                # Flat pre-call estimate (actual accounting uses real usage after the call).
                est = cost_model.estimate(1000, 5)
                if spent_usd + est > budget_usd:
                    raise RuntimeError(
                        f"budget cap reached: spent ${spent_usd:.4f}, next call estimated "
                        f"${est:.4f}, cap ${budget_usd:.2f}. Raise budget_usd to continue."
                    )
            verdict = call_with_retry(
                lambda: judge.judge(pair["question"], resp_first, resp_second, temperature=0.1)
            )
            cost = (
                0.0
                if dry_run or cost_model is None
                else cost_model.estimate(verdict.input_tokens, verdict.output_tokens)
            )
            spent_usd += cost
            row = {
                "pair_id": pair_id,
                "protocol": protocol,
                "judge": judge_name,
                "order": order,
                "verdict_raw": verdict.verdict,
                "raw_text": verdict.raw_text,
                "input_tokens": verdict.input_tokens,
                "output_tokens": verdict.output_tokens,
                "cost_usd": cost,
            }
            append_cache(cache_path, row)
            cache[key] = row

    return summarize_position_bias(pairs, cache)


def summarize_position_bias(pairs: list[dict], cache: dict) -> dict:
    """p_first and flip_rate from cached AB/BA verdicts. Pairs missing either
    order (e.g. budget cap hit mid-run) are counted as incomplete and
    excluded from the metrics rather than silently treated as ties.
    """
    first_position_count = 0
    total_calls = 0
    flips = 0
    complete_pairs = 0
    incomplete_pairs = 0

    for pair in pairs:
        pair_id = pair["id"]
        ab = cache.get((pair_id, "AB"))
        ba = cache.get((pair_id, "BA"))
        if ab is None or ba is None:
            incomplete_pairs += 1
            continue
        complete_pairs += 1
        for row in (ab, ba):
            total_calls += 1
            if row["verdict_raw"] == "A":
                first_position_count += 1

        ba_normalized = {"A": "B", "B": "A", "tie": "tie"}[ba["verdict_raw"]]
        if ab["verdict_raw"] != ba_normalized:
            flips += 1

    return {
        "n_pairs_total": len(pairs),
        "n_pairs_complete": complete_pairs,
        "n_pairs_incomplete": incomplete_pairs,
        "p_first": (first_position_count / total_calls) if total_calls else None,
        "flip_rate": (flips / complete_pairs) if complete_pairs else None,
    }
