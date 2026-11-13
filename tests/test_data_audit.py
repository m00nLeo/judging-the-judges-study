from pathlib import Path

import pytest

from src.data_audit import load_aggregated_metrics, reconcile_table1, tie_rate

UPSTREAM = Path(__file__).resolve().parents[1] / "data" / "upstream"


@pytest.fixture(scope="module")
def aggregated():
    return load_aggregated_metrics(UPSTREAM)


def test_aggregated_metrics_loaded(aggregated):
    # 88 files mirrored by scripts/fetch_upstream.sh at the time this was written.
    assert len(aggregated) == 88
    assert set(aggregated["benchmark"]) <= {"mt_bench", "llmbar", "custom"}
    # gemini-2.5-flash-lite is not one of the 5 judges described in the paper;
    # flag its presence rather than silently dropping or merging it.
    assert "gemini-2.5-flash-lite" in set(aggregated["model"])


def test_table1_mt_bench_reconciles_within_tolerance(aggregated):
    table1_csv = UPSTREAM / "analysis" / "figures" / "table1_main_results.csv"
    report = reconcile_table1(aggregated, table1_csv, tolerance=0.005)

    # All 5 published models x 5 published strategies = 25 cells should be
    # present in the aggregated cache (frontier/mid-tier judges have full
    # B0/S1/S4/S5/S8 coverage on mt_bench per the repo's aggregated/ listing).
    assert len(report) == 25

    mismatches = report[~report["within_tolerance"]]
    assert mismatches.empty, (
        "aggregated cache disagrees with published Table 1 beyond tolerance:\n"
        f"{mismatches.to_string()}"
    )


def test_w8_tie_rate_matches_papers_own_claim(aggregated):
    # W8: paper's text claims S1 has 32.8% tie and S8 has 12.5% tie for
    # Claude Sonnet 4 on MT-Bench. Only the tie-rate half of W8 is checkable
    # from aggregated data; the agreement-rate-excluding-ties decomposition
    # needs per-instance data (results/raw/), which is unavailable.
    s1_tie = tie_rate(aggregated, "claude-sonnet-4", "S1_position_swap", "mt_bench")
    s8_tie = tie_rate(aggregated, "claude-sonnet-4", "S8_combined_budget", "mt_bench")
    assert abs(s1_tie - 0.328) < 0.005
    assert abs(s8_tie - 0.125) < 0.005
