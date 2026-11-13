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
from pathlib import Path

from src.experiments.common import CostModel, call_with_retry, run_cached_calls
from src.experiments.common import append_cache as _common_append_cache
from src.experiments.common import load_cache as _common_load_cache

# Re-exported for backward compatibility: call_with_retry and CostModel used
# to be defined here before being factored out to src.experiments.common
# (now shared with E3/E5/E4). Behavior is unchanged.
__all__ = [
    "CostModel",
    "call_with_retry",
    "load_cache",
    "append_cache",
    "load_position_pairs_protocol_a",
    "load_mt_bench_sample_protocol_b",
    "run_position_bias_protocol",
    "summarize_position_bias",
]

_KEY_FIELDS = ("pair_id", "order")


def load_cache(cache_path: Path) -> dict:
    return _common_load_cache(cache_path, _KEY_FIELDS)


def append_cache(cache_path: Path, row: dict) -> None:
    _common_append_cache(cache_path, row)


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
    verbose: bool = True,
    max_workers: int = 1,
) -> dict:
    """Runs both orders for every pair, respecting cache and budget cap.

    `judge` must already be constructed (see src.judges.clients.build_judge)
    so tests can inject a fake without this function knowing about dry_run
    internals; `dry_run` here only controls whether cost is tracked.

    Raises RuntimeError before a call that would push spend over budget_usd.
    With `verbose=True` (default), prints one line per API call so a slow run
    is visibly making progress instead of looking hung for an hour.
    `max_workers > 1` runs pending calls concurrently -- see
    src.experiments.common.run_cached_calls for what that changes about the
    budget-cap guarantee (still bounded, no longer exact).
    """
    items = []
    for pair in pairs:
        items.append(
            {
                "pair_id": pair["id"],
                "order": "AB",
                "question": pair["question"],
                "resp_first": pair["response_a"],
                "resp_second": pair["response_b"],
            }
        )
        items.append(
            {
                "pair_id": pair["id"],
                "order": "BA",
                "question": pair["question"],
                "resp_first": pair["response_b"],
                "resp_second": pair["response_a"],
            }
        )

    def call_fn(item):
        return judge.judge(item["question"], item["resp_first"], item["resp_second"], temperature=0.1)

    def row_fn(item, result, cost):
        return {
            "pair_id": item["pair_id"],
            "protocol": protocol,
            "judge": judge_name,
            "order": item["order"],
            "verdict_raw": result.verdict,
            "raw_text": result.raw_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": cost,
        }

    cache = run_cached_calls(
        items,
        key_fields=_KEY_FIELDS,
        call_fn=call_fn,
        row_fn=row_fn,
        cache_path=cache_path,
        cost_model=cost_model,
        budget_usd=budget_usd,
        dry_run=dry_run,
        verbose=verbose,
        label=f"{protocol}/{judge_name}",
        max_workers=max_workers,
    )
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
