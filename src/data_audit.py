"""Load and reconcile the upstream mirror of sksoumik/judging-the-judges.

Scope note: results/raw/ (per-instance cache) and data/human_study/ are
.gitignore'd upstream and were never available in the public repo, so this
module only works with results/aggregated/*_metrics.json (summary counts,
no per-instance verdicts). That rules out recomputing McNemar, Holm-Bonferroni,
or tie-split breakdowns here -- it only lets us check that the aggregated
agreement_rate values match the numbers published in analysis/figures/.
"""

import json
import re
from pathlib import Path

import pandas as pd

AGGREGATED_FILENAME_RE = re.compile(
    r"^(?P<model>.+)_(?P<strategy>B0_naive|S1_position_swap|S2_ensemble_same|"
    r"S3_ensemble_cross|S4_calibrated_rubric|S5_cot_forcing|S6_reference_guided|"
    r"S7_combined_full|S8_combined_budget)_(?P<benchmark>mt_bench|llmbar|custom|"
    r"alpaca_eval|faireval)_metrics\.json$"
)

# Table 1 in the paper (and analysis/figures/table1_main_results.csv) only
# covers these five strategies, keyed by the model display name it uses.
STRATEGY_TO_TABLE1_COLUMN = {
    "B0_naive": "Baseline",
    "S1_position_swap": "Pos. Swap",
    "S4_calibrated_rubric": "Rubric",
    "S5_cot_forcing": "CoT",
    "S8_combined_budget": "Combined Budget",
}

MODEL_SLUG_TO_DISPLAY = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-sonnet-4": "Claude Sonnet 4",
    "gpt-4o": "GPT-4o",
    "llama-3.3-70b": "Llama 3.3-70B",
    # Not one of the 5 judges described in the paper / master plan.
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
}


def load_aggregated_metrics(upstream_dir: Path) -> pd.DataFrame:
    """Parse every results/aggregated/*_metrics.json file into one tidy DataFrame.

    Raises ValueError if a filename doesn't match the expected
    `<model>_<strategy>_<benchmark>_metrics.json` pattern, instead of silently
    skipping it -- unexpected files should surface, not be guessed away.
    """
    agg_dir = Path(upstream_dir) / "results" / "aggregated"
    files = sorted(agg_dir.glob("*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"no aggregated metrics files found under {agg_dir}")

    rows = []
    for fp in files:
        m = AGGREGATED_FILENAME_RE.match(fp.name)
        if m is None:
            raise ValueError(f"filename does not match expected pattern: {fp.name}")
        payload = json.loads(fp.read_text(encoding="utf-8"))
        row = {
            "model": m.group("model"),
            "strategy": m.group("strategy"),
            "benchmark": m.group("benchmark"),
            **payload,
        }
        # The paper's model/strategy/benchmark fields inside the JSON must agree
        # with what the filename encodes -- otherwise the mirrored file is corrupt
        # or was renamed, and we should not silently trust the filename.
        for key in ("model", "strategy", "benchmark"):
            if payload.get(key) != row[key]:
                raise ValueError(
                    f"{fp.name}: filename says {key}={row[key]!r} but "
                    f"JSON payload says {key}={payload.get(key)!r}"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def reconcile_table1(
    aggregated: pd.DataFrame, table1_csv_path: Path, tolerance: float = 0.005
) -> pd.DataFrame:
    """Compare agreement_rate in the aggregated cache against the published
    Table 1 (MT-Bench) point estimates. Returns one row per (model, strategy)
    that Table 1 covers, with a `within_tolerance` flag.

    Table 1 also carries bootstrap CIs; those cannot be recomputed here since
    results/raw/ is unavailable, so only the point estimate is reconciled.
    """
    published = pd.read_csv(table1_csv_path)
    published = published.set_index("Model")

    mt = aggregated[aggregated["benchmark"] == "mt_bench"].copy()
    mt["model_display"] = mt["model"].map(MODEL_SLUG_TO_DISPLAY)
    mt["table1_column"] = mt["strategy"].map(STRATEGY_TO_TABLE1_COLUMN)
    mt = mt.dropna(subset=["model_display", "table1_column"])

    out_rows = []
    for _, r in mt.iterrows():
        model_display, col = r["model_display"], r["table1_column"]
        if model_display not in published.index or col not in published.columns:
            continue
        cell = published.loc[model_display, col]
        published_point = float(str(cell).split("[")[0].strip())
        recomputed = float(r["agreement_rate"])
        out_rows.append(
            {
                "model": model_display,
                "strategy": r["strategy"],
                "table1_column": col,
                "published": published_point,
                "aggregated_cache": recomputed,
                "diff": round(recomputed - published_point, 4),
                "within_tolerance": abs(recomputed - published_point) <= tolerance,
            }
        )
    return pd.DataFrame(out_rows).sort_values(["model", "strategy"]).reset_index(drop=True)


def tie_rate(aggregated: pd.DataFrame, model: str, strategy: str, benchmark: str) -> float:
    """Fraction of verdicts that were "tie" for one (model, strategy, benchmark)
    cell. Computable from aggregated data alone (verdict_distribution counts),
    unlike the rest of W8 (decomposing the agreement-rate change into tie vs
    non-tie instances needs joint verdict/correctness data that only exists
    per-instance, i.e. in results/raw/, which is unavailable).
    """
    rows = aggregated[
        (aggregated["model"] == model)
        & (aggregated["strategy"] == strategy)
        & (aggregated["benchmark"] == benchmark)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly 1 row for ({model}, {strategy}, {benchmark}), found {len(rows)}"
        )
    row = rows.iloc[0]
    return row["verdict_distribution"]["tie"] / row["n_instances"]
