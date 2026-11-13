import time

import pytest

from src.experiments.common import CostModel, append_cache, load_cache, run_cached_calls


class SlowFakeResult:
    def __init__(self, verdict):
        self.verdict = verdict
        self.input_tokens = 10
        self.output_tokens = 2


def _slow_call_fn(delay):
    def call_fn(item):
        time.sleep(delay)
        return SlowFakeResult(item["id"])

    return call_fn


def _row_fn(item, result, cost):
    return {"id": item["id"], "verdict_raw": result.verdict, "cost_usd": cost}


def test_run_cached_calls_sequential_vs_concurrent_results_match(tmp_path):
    items = [{"id": f"item_{i}"} for i in range(6)]

    seq_path = tmp_path / "seq.jsonl"
    seq_cache = run_cached_calls(
        items, ("id",), _slow_call_fn(0), _row_fn, seq_path, dry_run=True, verbose=False, max_workers=1
    )

    par_path = tmp_path / "par.jsonl"
    par_cache = run_cached_calls(
        items, ("id",), _slow_call_fn(0), _row_fn, par_path, dry_run=True, verbose=False, max_workers=4
    )

    assert {k: v["verdict_raw"] for k, v in seq_cache.items()} == {
        k: v["verdict_raw"] for k, v in par_cache.items()
    }
    assert len(par_cache) == 6


def test_run_cached_calls_concurrency_is_actually_faster(tmp_path):
    items = [{"id": f"item_{i}"} for i in range(8)]
    delay = 0.1

    seq_path = tmp_path / "seq.jsonl"
    start = time.monotonic()
    run_cached_calls(
        items, ("id",), _slow_call_fn(delay), _row_fn, seq_path, dry_run=True, verbose=False, max_workers=1
    )
    sequential_elapsed = time.monotonic() - start

    par_path = tmp_path / "par.jsonl"
    start = time.monotonic()
    run_cached_calls(
        items, ("id",), _slow_call_fn(delay), _row_fn, par_path, dry_run=True, verbose=False, max_workers=4
    )
    concurrent_elapsed = time.monotonic() - start

    # 8 items * 0.1s sequential ~= 0.8s; with 4 workers ~= 0.2s. Generous margin for CI jitter.
    assert concurrent_elapsed < sequential_elapsed * 0.6


def test_run_cached_calls_resumes_without_recalling_cached_items(tmp_path):
    items = [{"id": f"item_{i}"} for i in range(5)]
    cache_path = tmp_path / "cache.jsonl"
    run_cached_calls(items, ("id",), _slow_call_fn(0), _row_fn, cache_path, dry_run=True, verbose=False)

    def exploding_call_fn(item):
        raise AssertionError("should be fully cached, no new calls expected")

    cache2 = run_cached_calls(
        items, ("id",), exploding_call_fn, _row_fn, cache_path, dry_run=True, verbose=False, max_workers=3
    )
    assert len(cache2) == 5


def test_run_cached_calls_budget_cap_stops_concurrent_run(tmp_path):
    items = [{"id": f"item_{i}"} for i in range(10)]
    cache_path = tmp_path / "cache.jsonl"
    cost_model = CostModel(cost_per_1k_input=1.0, cost_per_1k_output=1.0)  # deliberately expensive

    with pytest.raises(RuntimeError, match="budget cap reached"):
        run_cached_calls(
            items,
            ("id",),
            _slow_call_fn(0),
            _row_fn,
            cache_path,
            cost_model=cost_model,
            budget_usd=0.0005,  # smaller than one call's estimated cost
            dry_run=False,
            verbose=False,
            max_workers=4,
        )
    # Whatever got through before the cap tripped must be cached, not lost.
    cache = load_cache(cache_path, ("id",))
    assert len(cache) < 10


def test_append_cache_writes_are_not_corrupted_by_concurrent_access(tmp_path):
    # Each row written under the lock should still be one valid JSON line.
    items = [{"id": f"item_{i}"} for i in range(20)]
    cache_path = tmp_path / "cache.jsonl"
    run_cached_calls(
        items, ("id",), _slow_call_fn(0.01), _row_fn, cache_path, dry_run=True, verbose=False, max_workers=8
    )
    lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20
    for line in lines:
        import json

        json.loads(line)  # raises if any line got interleaved/corrupted
