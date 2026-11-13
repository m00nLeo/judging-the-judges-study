"""Shared caching/budget/retry infrastructure for E2+ experiments that call a
real judge API. E2 (position_bias.py) predates this module and keeps its own
copy of call_with_retry/CostModel for its AB/BA-per-pair shape -- this module
is for experiments with one call per (item, repeat) instead, used by E3
(style_bias.py) and E5 (format_normalization.py).
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


def call_with_retry(fn: Callable, max_retries: int = 5, base_delay: float = 2.0):
    """Retries `fn()` with exponential backoff on transient provider errors;
    anything non-transient (bad key, malformed request) fails immediately.
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


def load_cache(cache_path: Path, key_fields: tuple[str, ...]) -> dict:
    """Loads a jsonl cache into {tuple(row[f] for f in key_fields): row}."""
    cache = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cache[tuple(row[field] for field in key_fields)] = row
    return cache


def append_cache(cache_path: Path, row: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cached_calls(
    items: list[dict],
    key_fields: tuple[str, ...],
    call_fn: Callable[[dict], object],
    row_fn: Callable[[dict, object, float], dict],
    cache_path: Path,
    cost_model: CostModel | None = None,
    budget_usd: float = 1.0,
    dry_run: bool = False,
    verbose: bool = True,
    label: str = "",
    max_workers: int = 1,
) -> dict:
    """Generic one-call-per-item runner with caching, budget cap, retry, and
    progress logging -- shared by E3 and E5. Each item must already carry its
    own key_fields as dict entries (e.g. {"pair_id": ..., "repeat": 0, ...}).

    `call_fn(item)` makes the actual API call and returns a result object with
    `.input_tokens` / `.output_tokens` (see src.judges.clients.JudgeVerdict).
    `row_fn(item, result, cost)` builds the dict to cache; it MUST include
    all of key_fields as literal entries so the cache can be reloaded later.

    Raises RuntimeError before a call that would push spend over budget_usd.
    Resumable: items whose key is already in the cache are skipped.

    `max_workers > 1` runs pending calls concurrently in a thread pool -- the
    network wait, not CPU, dominates each call, so threads (not processes)
    are enough. Free-tier APIs have per-minute rate limits, so more workers
    isn't always faster; start small (e.g. 3-5) and watch for a spike in 429s.
    Under concurrency the budget check is approximate: each worker checks the
    same flat per-call cost estimate against the budget before its own call,
    without reserving it first, so spend can briefly overshoot by up to
    roughly (max_workers - 1) call-estimates before the cap stops new calls --
    bounded, but not exact. Cache writes are serialized, so the jsonl file
    itself never gets corrupted by concurrent appends.
    """
    cache = load_cache(cache_path, key_fields)
    lock = threading.Lock()
    state = {"spent_usd": sum(row.get("cost_usd", 0.0) for row in cache.values()), "made": 0}
    total = len(items)
    done_already = len(cache)

    pending = [item for item in items if tuple(item[field] for field in key_fields) not in cache]

    def process_one(item):
        key = tuple(item[field] for field in key_fields)
        if not dry_run and cost_model is not None:
            with lock:
                est = cost_model.estimate(1000, 5)
                if state["spent_usd"] + est > budget_usd:
                    raise RuntimeError(
                        f"budget cap reached: spent ${state['spent_usd']:.4f}, next call estimated "
                        f"${est:.4f}, cap ${budget_usd:.2f}. Raise budget_usd to continue."
                    )
        call_start = time.monotonic()
        result = call_with_retry(lambda: call_fn(item))
        cost = (
            0.0
            if dry_run or cost_model is None
            else cost_model.estimate(
                getattr(result, "input_tokens", 0), getattr(result, "output_tokens", 0)
            )
        )
        row = row_fn(item, result, cost)
        with lock:
            state["spent_usd"] += cost
            state["made"] += 1
            if verbose:
                elapsed = time.monotonic() - call_start
                done = done_already + state["made"]
                print(f"[{label}] {done}/{total} key={key} ({elapsed:.1f}s, ${cost:.5f})")
            append_cache(cache_path, row)
            cache[key] = row

    if max_workers <= 1:
        for item in pending:
            process_one(item)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, item) for item in pending]
            for future in futures:
                future.result()  # re-raises in the main thread (e.g. budget cap, a real API error)

    return cache
