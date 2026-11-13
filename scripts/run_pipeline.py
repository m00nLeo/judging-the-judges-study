"""Unified CLI runner for Phase 1 (E0-E5) + Phase 2 (E6 Counterfactual GLM).

Usage examples:
  1. Run all no-API checks (E0, E1, E6 RQ1 confounding audit) + dry-run all API pipelines:
     python scripts/run_pipeline.py --mode dry-run

  2. Run pilot API experiments (small sample, checks cost & OpenRouter connectivity):
     python scripts/run_pipeline.py --mode pilot

  3. Run full API experiments + E6 Counterfactual GLM calibration on cached verdicts:
     python scripts/run_pipeline.py --mode full

  4. Re-run only E6 (Counterfactual GLM analysis) on existing caches ($0 API cost):
     python scripts/run_pipeline.py --mode e6-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import load_aggregated_metrics, reconcile_table1, tie_rate  # noqa: E402
from src.experiments.common import CostModel  # noqa: E402
from src.experiments.counterfactual_glm import (  # noqa: E402
    audit_dataset_confounding,
    counterfactual_debias_verdicts,
    evaluate_style_bias_subgroups,
    fit_counterfactual_glm,
    summarize_style_confounding,
)
from src.experiments.format_normalization import (  # noqa: E402
    run_format_normalization_experiment,
    summarize_agreement_by_strategy,
    summarize_style_bias_by_strategy,
)
from src.experiments.position_bias import (  # noqa: E402
    load_mt_bench_sample_protocol_b,
    load_position_pairs_protocol_a,
    run_position_bias_protocol,
)
from src.experiments.style_bias import (  # noqa: E402
    load_style_pairs,
    run_style_bias_experiment,
    summarize_style_bias,
)
from src.judges.clients import build_judge  # noqa: E402
from src.stats_checks import check_mde_claim, ci_overlap, two_proportion_ztest  # noqa: E402


def _total_cost(cache_path: Path) -> float:
    if not cache_path.exists():
        return 0.0
    return sum(
        json.loads(line).get("cost_usd", 0.0)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def run_no_api_phase(upstream_dir: Path) -> None:
    print("=" * 72)
    print("PHASE 1 (NO API): E0 Data Audit & E1 Statistical Checks")
    print("=" * 72)
    agg = load_aggregated_metrics(upstream_dir)
    table1_csv = upstream_dir / "analysis" / "figures" / "table1_main_results.csv"
    t1_report = reconcile_table1(agg, table1_csv)
    matched = int(t1_report["within_tolerance"].sum())
    print(f"[E0] Table 1 reconciliation: {matched}/{len(t1_report)} cells match within 0.005")

    s1_tie = tie_rate(agg, "claude-sonnet-4", "S1_position_swap", "mt_bench")
    s8_tie = tie_rate(agg, "claude-sonnet-4", "S8_combined_budget", "mt_bench")
    print(f"[E0/W8] Claude Sonnet 4 tie rate on MT-Bench: S1={s1_tie:.2%}, S8={s8_tie:.2%}")

    mde_report = check_mde_claim(table1_csv, n=400, tolerance=0.015)
    print(
        f"[E1/W2] Wald half-width check (N=400): "
        f"{int(mde_report['within_tolerance'].sum())}/{len(mde_report)} CIs within 1.5 pts"
    )

    df_t1 = pd.read_csv(table1_csv).set_index("Model")
    ov = ci_overlap(
        df_t1.loc["Gemini 2.5 Flash", "Combined Budget"],
        df_t1.loc["Claude Sonnet 4", "Combined Budget"],
    )
    ztest = two_proportion_ztest(n1=400, p1=0.710, n2=400, p2=0.695)
    print(f"[E1/W7] Flash S8 vs Claude S8 CI overlap: {ov['overlaps']} (amount={ov['overlap_amount']})")
    print(f"[E1/W7] Unpaired two-proportion z-test: z={ztest['z']}, p={ztest['p_value']} (sig={ztest['significant_at_0.05']})")

    print("\n" + "=" * 72)
    print("PHASE 2 (RQ1, NO API): Confounding Audit on Upstream STYLE Pairs")
    print("=" * 72)
    style_pairs = load_style_pairs(upstream_dir / "data" / "custom" / "controlled_pairs.jsonl")
    conf = summarize_style_confounding(style_pairs)
    for k, v in conf.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    df_art = audit_dataset_confounding(style_pairs)
    corr_cols = [
        "log_word_ratio",
        "delta_bold",
        "delta_header",
        "delta_list",
        "delta_struct",
        "delta_completeness",
    ]
    print("\nArtifact correlation with log_word_ratio & delta_completeness:")
    print(df_art[corr_cols].corr()[["log_word_ratio", "delta_completeness"]].round(3).to_string())


def run_api_experiments(config: dict, upstream_dir: Path, results_root: Path, mode: str) -> None:
    dry_run = mode == "dry-run"
    is_pilot = mode == "pilot"
    max_workers = 1 if dry_run else int(config["concurrency"]["max_workers"])
    seed = int(config["seed"])

    controlled_path = upstream_dir / "data" / "custom" / "controlled_pairs.jsonl"
    mt_bench_path = upstream_dir / "data" / "raw" / "mt_bench" / "mt_bench.jsonl"
    llmbar_path = upstream_dir / "data" / "raw" / "llmbar" / "llmbar.jsonl"

    position_pairs = load_position_pairs_protocol_a(controlled_path)
    style_pairs = load_style_pairs(controlled_path)

    # --- E2: Position bias ---
    print("\n" + "=" * 72)
    print(f"RUNNING E2 (Position Bias Protocols A & B) [mode={mode}]")
    print("=" * 72)
    e2_dir = results_root / "e2_position_bias"
    n_mt_e2 = (
        5
        if dry_run
        else (
            config["sampling"]["e2_position_bias"]["mt_bench_pilot_sample"]
            if is_pilot
            else config["sampling"]["e2_position_bias"]["mt_bench_full_sample"]
        )
    )
    mt_e2_pairs = load_mt_bench_sample_protocol_b(mt_bench_path, n=n_mt_e2, seed=seed)
    pos_sample = position_pairs[:5] if dry_run else position_pairs

    judges_e2 = ["dry-run"] if dry_run else config["judges"]["e2_position_bias"]
    for jname in judges_e2:
        judge = build_judge(jname, dry_run=dry_run)
        cmodel = None
        if not dry_run:
            p = config["cost_per_1k_tokens"][jname]
            cmodel = CostModel(cost_per_1k_input=p["input"], cost_per_1k_output=p["output"])
        budget = float(config["budget"]["e2_position_bias_usd"])
        tag = "dry" if dry_run else ("pilot" if is_pilot else "full")
        ca = e2_dir / f"{jname}_protocolA_position_{tag}.jsonl"
        cb = e2_dir / f"{jname}_protocolB_mtbench_{tag}.jsonl"
        if dry_run:
            ca.unlink(missing_ok=True)
            cb.unlink(missing_ok=True)
        sum_a = run_position_bias_protocol(
            pos_sample, judge, jname, "A", ca, cmodel, budget, dry_run=dry_run, verbose=not dry_run, max_workers=max_workers
        )
        sum_b = run_position_bias_protocol(
            mt_e2_pairs, judge, jname, "B", cb, cmodel, budget, dry_run=dry_run, verbose=not dry_run, max_workers=max_workers
        )
        print(f"[E2/{jname}] Protocol A: {sum_a} (${_total_cost(ca):.4f})")
        print(f"[E2/{jname}] Protocol B ({tag}): {sum_b} (${_total_cost(cb):.4f})")

    # --- E3: Style bias ---
    print("\n" + "=" * 72)
    print(f"RUNNING E3 (Style Bias on Modern Judges) [mode={mode}]")
    print("=" * 72)
    e3_dir = results_root / "e3_style_bias"
    n_repeats = 1 if (dry_run or is_pilot) else int(config["sampling"]["e3_style_bias"]["n_repeats"])
    e3_sample = style_pairs[:5] if (dry_run or is_pilot) else style_pairs
    judges_e3 = ["dry-run"] if dry_run else config["judges"]["e3_style_bias"]
    for jname in judges_e3:
        judge = build_judge(jname, dry_run=dry_run)
        cmodel = None
        if not dry_run:
            p = config["cost_per_1k_tokens"][jname]
            cmodel = CostModel(cost_per_1k_input=p["input"], cost_per_1k_output=p["output"])
        budget = float(config["budget"]["e3_style_bias_usd"])
        tag = "dry" if dry_run else ("pilot" if is_pilot else "full")
        cpath = e3_dir / f"{jname}_style_{tag}.jsonl"
        if dry_run:
            cpath.unlink(missing_ok=True)
        cache = run_style_bias_experiment(
            e3_sample, judge, jname, cpath, n_repeats=n_repeats, cost_model=cmodel,
            budget_usd=budget, dry_run=dry_run, verbose=not dry_run, max_workers=max_workers
        )
        summary = summarize_style_bias(e3_sample, cache, n_repeats=n_repeats)
        print(f"[E3/{jname}] ({tag}): {summary} (${_total_cost(cpath):.4f})")

    # --- E5: Format normalization ---
    print("\n" + "=" * 72)
    print(f"RUNNING E5 (Format Normalization B0, N1, N2, S5) [mode={mode}]")
    print("=" * 72)
    e5_dir = results_root / "e5_format_normalization"
    e5_cfg = config["sampling"]["e5_format_normalization"]
    strategies = tuple(e5_cfg["strategies"])
    n_mt_e5 = 3 if dry_run else (e5_cfg["mt_bench_pilot_sample"] if is_pilot else e5_cfg["mt_bench_full_sample"])
    n_lb_e5 = 3 if dry_run else (e5_cfg["llmbar_pilot_sample"] if is_pilot else e5_cfg["llmbar_full_sample"])
    style_e5 = style_pairs[:4] if dry_run else (style_pairs[:10] if is_pilot else style_pairs)
    mt_e5 = load_mt_bench_sample_protocol_b(mt_bench_path, n=n_mt_e5, seed=seed)
    llmbar_rows = [json.loads(line) for line in llmbar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lb_e5 = random.Random(seed).sample(llmbar_rows, min(n_lb_e5, len(llmbar_rows)))

    judges_e5 = ["dry-run"] if dry_run else config["judges"]["e5_format_normalization"]
    for jname in judges_e5:
        judge = build_judge(jname, dry_run=dry_run)
        cmodel = None
        if not dry_run:
            p = config["cost_per_1k_tokens"][jname]
            cmodel = CostModel(cost_per_1k_input=p["input"], cost_per_1k_output=p["output"])
        budget = float(config["budget"]["e5_format_normalization_usd"])
        tag = "dry" if dry_run else ("pilot" if is_pilot else "full")

        for bench_name, bench_pairs, is_style in [
            ("style", style_e5, True),
            ("mtbench", mt_e5, False),
            ("llmbar", lb_e5, False),
        ]:
            cpath = e5_dir / f"{jname}_{bench_name}_{tag}.jsonl"
            if dry_run:
                cpath.unlink(missing_ok=True)
            cache = run_format_normalization_experiment(
                bench_pairs, judge, jname, cpath, strategies=strategies,
                cost_model=cmodel, budget_usd=budget, dry_run=dry_run,
                verbose=not dry_run, max_workers=max_workers
            )
            if is_style:
                res = summarize_style_bias_by_strategy(bench_pairs, cache, strategies=strategies)
            else:
                res = summarize_agreement_by_strategy(bench_pairs, cache, strategies=strategies)
            print(f"[E5/{jname}/{bench_name}] ({tag}): {res} (${_total_cost(cpath):.4f})")


def run_e6_counterfactual_glm(config: dict, upstream_dir: Path, results_root: Path) -> None:
    print("\n" + "=" * 72)
    print("PHASE 2 (RQ2 & RQ3): E6 Counterfactual Multi-Bias GLM (MAC-GLM)")
    print("=" * 72)
    style_pairs = load_style_pairs(upstream_dir / "data" / "custom" / "controlled_pairs.jsonl")
    e3_dir = results_root / "e3_style_bias"

    found_any = False
    for jname in config["judges"]["e3_style_bias"]:
        for tag in ("full", "pilot"):
            cpath = e3_dir / f"{jname}_style_{tag}.jsonl"
            if not cpath.exists():
                continue
            found_any = True
            rows = [json.loads(line) for line in cpath.read_text(encoding="utf-8").splitlines() if line.strip()]
            by_pair = {r["pair_id"]: r["verdict_raw"] for r in rows if r.get("repeat", 0) == 0}
            paired = [p for p in style_pairs if p["id"] in by_pair]
            if len(paired) < 10:
                print(f"[E6/{jname}] Only {len(paired)} pairs in {cpath.name}; run full E3 (100 pairs) for full GLM fit.")
                continue
            verdicts = [by_pair[p["id"]] for p in paired]
            fit = fit_counterfactual_glm(paired, verdicts)
            cf = counterfactual_debias_verdicts(paired, verdicts, n_folds=5)
            raw_sub = evaluate_style_bias_subgroups(paired, verdicts)
            cf_sub = evaluate_style_bias_subgroups(paired, cf["debiased_verdicts"])
            print(f"\n--- Judge: {jname} ({cpath.name}, n={len(paired)}) ---")
            print("Bias Fingerprint (GLM coefficients):")
            print(fit["coefficients"].to_string(index=False))
            print(f"Raw Style Bias Slices:      {raw_sub}")
            print(f"MAC-GLM Debiased Slices:    {cf_sub}")
            break

    if not found_any:
        print("[E6] Running dry-run demonstration on 100 STYLE pairs (no real API cache found yet):")
        dry_judge = build_judge("dry-run", dry_run=True)
        dry_verdicts = [dry_judge.judge(p["question"], p["response_a"], p["response_b"]).verdict for p in style_pairs]
        fit = fit_counterfactual_glm(style_pairs, dry_verdicts)
        cf = counterfactual_debias_verdicts(style_pairs, dry_verdicts, n_folds=5)
        print(fit["coefficients"].to_string(index=False))
        print("Dry-run subgroup evaluation:", evaluate_style_bias_subgroups(style_pairs, cf["debiased_verdicts"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Judging-the-Judges study pipeline.")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "pilot", "full", "e6-only"),
        default="dry-run",
        help="Execution mode: dry-run ($0 fake judge), pilot (small real API run), full (complete API run), or e6-only ($0 GLM analysis).",
    )
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    upstream_dir = PROJECT_ROOT / "data" / "upstream"
    results_root = PROJECT_ROOT / "results"

    run_no_api_phase(upstream_dir)
    if args.mode in ("dry-run", "pilot", "full"):
        run_api_experiments(config, upstream_dir, results_root, mode=args.mode)
    run_e6_counterfactual_glm(config, upstream_dir, results_root)


if __name__ == "__main__":
    main()

