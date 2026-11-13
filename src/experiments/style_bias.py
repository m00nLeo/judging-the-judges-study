"""E3: does style bias survive on newer judges? (master plan mục 8, W3)

Master plan: "Chạy lại 100 cặp STYLE (50 gốc, 50 đảo vị trí) với 2-3 judge
mới, lặp 3 lần, nhiệt độ 0,1; bias có dấu và bootstrap CI."

Data: data/custom/controlled_pairs.jsonl, bias_type in {"style", "style_ba"}.
Confirmed from the actual file (not assumed): for bias_type=="style", the
markdown-formatted response is always response_a ("response_a uses markdown
formatting, response_b is plain prose" in every row's `manipulation` field);
for "style_ba" it's always response_b (the position-reversed mirror). This
function reads that from `manipulation` text directly rather than trusting
bias_type as a shortcut, so a future row that doesn't fit the pattern raises
instead of being silently mis-scored.

Style bias score convention used here (ours, not guaranteed identical to the
paper's undisclosed exact tie handling -- same caveat as Table 2 in notebook 01):
    score = (n_markdown_chosen - n_plain_chosen) / n_total_calls
range [-1, +1], 0 = unbiased, ties count toward the denominator but neither
side's numerator (so an all-tie judge scores 0, matching the +0.76-style
scale already seen in the paper's own Table 2).
"""

import json
from pathlib import Path

from src.experiments.common import CostModel, run_cached_calls


def markdown_side(pair: dict) -> str:
    text = pair.get("manipulation", "")
    if "response_b is markdown formatting" in text:
        return "B"
    if "response_a uses markdown formatting" in text:
        return "A"
    raise ValueError(f"cannot determine markdown side for pair {pair['id']!r} from manipulation text")


def load_style_pairs(controlled_pairs_path: Path) -> list[dict]:
    pairs = []
    with controlled_pairs_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["bias_type"] in ("style", "style_ba"):
                pairs.append(row)
    return pairs


def run_style_bias_experiment(
    pairs: list[dict],
    judge,
    judge_name: str,
    cache_path: Path,
    n_repeats: int = 3,
    cost_model: CostModel | None = None,
    budget_usd: float = 1.0,
    dry_run: bool = False,
    verbose: bool = True,
    max_workers: int = 1,
) -> dict:
    """Judges every pair `n_repeats` times (master plan: "lặp 3 lần"), caches
    each call keyed by (pair_id, repeat), and returns the resulting cache.
    Call `summarize_style_bias(pairs, cache)` on the result for the metric.
    """
    items = [
        {"pair_id": pair["id"], "repeat": r, "pair": pair}
        for pair in pairs
        for r in range(n_repeats)
    ]

    def call_fn(item):
        pair = item["pair"]
        return judge.judge(pair["question"], pair["response_a"], pair["response_b"], temperature=0.1)

    def row_fn(item, result, cost):
        return {
            "pair_id": item["pair_id"],
            "repeat": item["repeat"],
            "judge": judge_name,
            "verdict_raw": result.verdict,
            "raw_text": result.raw_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": cost,
        }

    return run_cached_calls(
        items,
        key_fields=("pair_id", "repeat"),
        call_fn=call_fn,
        row_fn=row_fn,
        cache_path=cache_path,
        cost_model=cost_model,
        budget_usd=budget_usd,
        dry_run=dry_run,
        verbose=verbose,
        label=f"E3/{judge_name}",
        max_workers=max_workers,
    )


def summarize_style_bias(pairs: list[dict], cache: dict, n_repeats: int = 3) -> dict:
    """Style bias score (see module docstring) plus how many (pair, repeat)
    calls were actually found in the cache, so a partial run is visible
    rather than silently treated as complete.
    """
    by_id = {pair["id"]: pair for pair in pairs}
    n_markdown = 0
    n_plain = 0
    n_tie = 0
    n_found = 0
    n_expected = len(pairs) * n_repeats

    for key, row in cache.items():
        pair_id, _repeat = key
        pair = by_id.get(pair_id)
        if pair is None:
            continue  # cache from a different pair set; not part of this summary
        n_found += 1
        side = markdown_side(pair)
        verdict = row["verdict_raw"]
        if verdict == "tie":
            n_tie += 1
        elif verdict == side:
            n_markdown += 1
        else:
            n_plain += 1

    total = n_markdown + n_plain + n_tie
    return {
        "n_calls_expected": n_expected,
        "n_calls_found": n_found,
        "n_markdown_chosen": n_markdown,
        "n_plain_chosen": n_plain,
        "n_tie": n_tie,
        "style_bias_score": ((n_markdown - n_plain) / total) if total else None,
    }
