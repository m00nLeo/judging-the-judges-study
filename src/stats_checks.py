"""E1 checks (power / CI overlap) that work from aggregated-only data.

Scope reminder: results/raw/ (per-instance verdicts) is not in the public
repo (see src/data_audit.py docstring). That blocks the master plan's E1
design for two of its three pieces:

- Paired McNemar test between Flash S8 and Claude S8 (needs per-instance
  paired agreement) -- NOT POSSIBLE here. `two_proportion_ztest` below is an
  UNPAIRED approximation using only the aggregated n and agreement_rate; it
  is weaker evidence than McNemar and is labeled as such everywhere it's used.
- Stratification by MT-Bench category (needs a per-instance category field)
  -- NOT POSSIBLE here; aggregated/*_metrics.json carries no category
  breakdown. See BLOCKED_ANALYSES.

What IS possible from aggregated-only data: the Wald CI half-width implied by
N=400 can be computed from first principles and checked against the
bootstrap CIs already published in table1_main_results.csv (W2), and CI
overlap between any two published cells can be checked directly (W7).
"""

import math
import re
from pathlib import Path

import pandas as pd
from scipy.stats import norm

CI_CELL_RE = re.compile(r"^\s*([\d.]+)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*$")

BLOCKED_ANALYSES = {
    "W7_mcnemar_paired": (
        "Paired McNemar test (Flash S8 vs Claude S8) needs per-instance "
        "paired verdicts from results/raw/, which is .gitignore'd upstream. "
        "Substituted with an unpaired two-proportion z-test (weaker)."
    ),
    "E1_category_stratification": (
        "Stratifying MT-Bench agreement by category needs a per-instance "
        "category field. results/aggregated/*_metrics.json has no such "
        "field (only n_instances, verdict_distribution, agreement_rate, "
        "cost/latency/token totals). Not possible from the public cache."
    ),
}


def parse_ci_cell(cell: str) -> tuple[float, float, float]:
    """Parse a "0.710 [0.665, 0.755]" style cell into (point, lo, hi)."""
    m = CI_CELL_RE.match(cell)
    if m is None:
        raise ValueError(f"cell does not match '<point> [<lo>, <hi>]' pattern: {cell!r}")
    point, lo, hi = (float(g) for g in m.groups())
    return point, lo, hi


def ci_half_width(cell: str) -> float:
    _, lo, hi = parse_ci_cell(cell)
    return (hi - lo) / 2


def wald_ci_half_width(n: int, p: float, confidence: float = 0.95) -> float:
    """Theoretical half-width of a Wald CI for one proportion at the given n, p."""
    z = norm.ppf(1 - (1 - confidence) / 2)
    return z * math.sqrt(p * (1 - p) / n)


def check_mde_claim(
    table1_csv_path: Path, n: int = 400, tolerance: float = 0.015
) -> pd.DataFrame:
    """W2: does the paper's "~4-5 point" MDE claim for N=400 match a
    first-principles Wald half-width computed from each cell's own point
    estimate? Returns one row per table cell with both values and the diff.
    """
    df = pd.read_csv(table1_csv_path).set_index("Model")
    rows = []
    for model in df.index:
        for strategy in df.columns:
            point, lo, hi = parse_ci_cell(df.loc[model, strategy])
            published_half_width = (hi - lo) / 2
            theoretical_half_width = wald_ci_half_width(n, point)
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "point_estimate": point,
                    "published_half_width": round(published_half_width, 4),
                    "theoretical_half_width_n400": round(theoretical_half_width, 4),
                    "diff": round(published_half_width - theoretical_half_width, 4),
                    "within_tolerance": abs(published_half_width - theoretical_half_width)
                    <= tolerance,
                }
            )
    return pd.DataFrame(rows)


def ci_overlap(cell_a: str, cell_b: str) -> dict:
    """Do two published CI cells overlap? Used for W7 (Flash S8 vs Claude S8)."""
    pa, lo_a, hi_a = parse_ci_cell(cell_a)
    pb, lo_b, hi_b = parse_ci_cell(cell_b)
    overlap_lo, overlap_hi = max(lo_a, lo_b), min(hi_a, hi_b)
    overlaps = overlap_lo <= overlap_hi
    return {
        "point_a": pa,
        "point_b": pb,
        "overlaps": overlaps,
        "overlap_amount": round(overlap_hi - overlap_lo, 4) if overlaps else 0.0,
    }


def two_proportion_ztest(n1: int, p1: float, n2: int, p2: float) -> dict:
    """UNPAIRED two-proportion z-test. See module docstring: this is a weaker
    substitute for the paired McNemar test the master plan specifies for W7,
    used only because per-instance paired data is unavailable.
    """
    pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {
        "z": round(z, 4),
        "p_value": round(float(p_value), 4),
        "significant_at_0.05": bool(p_value < 0.05),
    }
