from pathlib import Path

from src.stats_checks import (
    check_mde_claim,
    ci_overlap,
    two_proportion_ztest,
    wald_ci_half_width,
)

UPSTREAM = Path(__file__).resolve().parents[1] / "data" / "upstream"
TABLE1 = UPSTREAM / "analysis" / "figures" / "table1_main_results.csv"
TABLE_LLMBAR = UPSTREAM / "analysis" / "figures" / "table_llmbar_with_ci.csv"


def test_wald_half_width_matches_paper_claim_of_4_to_5_points():
    # Master plan W2: "N=400 chi phat hien duoc muc chenh khoang 4-5 diem"
    hw = wald_ci_half_width(n=400, p=0.6)
    assert 0.04 <= hw <= 0.05


def test_mde_claim_reconciles_with_published_table1_cis():
    report = check_mde_claim(TABLE1, n=400, tolerance=0.015)
    assert len(report) == 25
    mismatches = report[~report["within_tolerance"]]
    assert mismatches.empty, (
        "theoretical N=400 Wald half-width diverges from the published "
        f"bootstrap CI by more than tolerance:\n{mismatches.to_string()}"
    )


def test_w7_flash_s8_vs_claude_s8_cis_overlap():
    # W7: the master plan's own claim that these two CIs overlap a lot.
    import pandas as pd

    df = pd.read_csv(TABLE1).set_index("Model")
    result = ci_overlap(
        df.loc["Gemini 2.5 Flash", "Combined Budget"],
        df.loc["Claude Sonnet 4", "Combined Budget"],
    )
    assert result["overlaps"] is True
    assert result["overlap_amount"] > 0.05


def test_w7_unpaired_proportion_test_finds_no_significant_difference():
    # Weaker substitute for the paired McNemar the master plan specifies;
    # see BLOCKED_ANALYSES in src/stats_checks.py for why.
    result = two_proportion_ztest(n1=400, p1=0.710, n2=400, p2=0.695)
    assert result["significant_at_0.05"] is False


def test_llmbar_s8_regression_is_real_for_most_models():
    # Sanity check unrelated to W7: S8 should sit below Baseline on LLMBar
    # for the models the master plan says regress (Pro, GPT-4o, Llama).
    import pandas as pd

    from src.stats_checks import parse_ci_cell

    df = pd.read_csv(TABLE_LLMBAR).set_index("Model")
    for model in ["Gemini 2.5 Pro", "GPT-4o", "Llama 3.3-70B"]:
        base_point, _, _ = parse_ci_cell(df.loc[model, "Baseline"])
        s8_point, _, _ = parse_ci_cell(df.loc[model, "Combined Budget"])
        assert s8_point < base_point, f"{model}: expected S8 < Baseline on LLMBar"
