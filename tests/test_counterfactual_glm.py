from pathlib import Path

import pytest

from src.experiments.counterfactual_glm import (
    audit_dataset_confounding,
    counterfactual_debias_verdicts,
    evaluate_style_bias_subgroups,
    extract_pair_artifact_vector,
    fit_counterfactual_glm,
    is_truncated_response,
    summarize_style_confounding,
)
from src.experiments.style_bias import load_style_pairs, summarize_style_bias

UPSTREAM = Path(__file__).resolve().parents[1] / "data" / "upstream"


def test_is_truncated_response_detects_mid_sentence_cutoffs():
    assert is_truncated_response("This sentence ends properly.") is False
    assert is_truncated_response("Result is **42**.") is False
    assert is_truncated_response("and the outcome of one die does not affect the outcome") is True
    assert is_truncated_response("### Summarizing the Outcomes:\n\n") is True
    assert is_truncated_response("") is True


def test_extract_pair_artifact_vector_is_antisymmetric():
    pair_ab = {
        "id": "p1",
        "response_a": "# Heading\n\n* **Bold item 1**\n* **Bold item 2**\n\nComplete conclusion.",
        "response_b": "Short plain text cut off mid sentence without period",
    }
    pair_ba = {
        "id": "p1_swap",
        "response_a": pair_ab["response_b"],
        "response_b": pair_ab["response_a"],
    }
    v_ab = extract_pair_artifact_vector(pair_ab)
    v_ba = extract_pair_artifact_vector(pair_ba)

    for key in (
        "log_word_ratio",
        "delta_bold",
        "delta_header",
        "delta_list",
        "delta_struct",
        "delta_completeness",
    ):
        assert v_ab[key] == pytest.approx(-v_ba[key])
    assert v_ab["delta_bold"] == 2.0
    assert v_ab["delta_header"] == 1.0
    assert v_ab["delta_list"] == 2.0
    assert v_ab["delta_completeness"] == 1.0


def test_summarize_style_confounding_on_upstream_data():
    style_pairs = load_style_pairs(UPSTREAM / "data" / "custom" / "controlled_pairs.jsonl")
    summary = summarize_style_confounding(style_pairs)
    assert summary["n_pairs"] == 100
    # Confirmed upstream artifact: the markdown side is systematically longer
    # and a non-trivial fraction of pairs have mid-sentence truncation.
    assert summary["frac_markdown_longer"] > 0.70
    assert summary["frac_asymmetric_truncation"] > 0.10


def test_fit_and_counterfactual_debias_reduces_synthetic_style_bias():
    pairs = []
    verdicts = []
    for i in range(20):
        is_ab = i % 2 == 0
        md_text = f"## Title {i}\n\n* **Key point one** with extra words.\n* **Key point two** with details."
        plain_text = f"Point one and point two for item {i}."
        if is_ab:
            pair = {
                "id": f"s_{i}",
                "question": "q",
                "response_a": md_text,
                "response_b": plain_text,
                "bias_type": "style",
                "manipulation": "response_a uses markdown formatting, response_b is plain prose with same content",
            }
            verdicts.append("A")  # biased judge always picks markdown side
        else:
            pair = {
                "id": f"s_{i}",
                "question": "q",
                "response_a": plain_text,
                "response_b": md_text,
                "bias_type": "style_ba",
                "manipulation": "STYLE pair with positions reversed: response_a is plain prose, response_b is markdown formatting",
            }
            verdicts.append("B")  # biased judge always picks markdown side
        pairs.append(pair)

    fit = fit_counterfactual_glm(pairs, verdicts)
    assert fit["converged"] is True
    assert len(fit["coefficients"]) == 7  # intercept + 6 features

    raw_subgroups = evaluate_style_bias_subgroups(pairs, verdicts)
    assert raw_subgroups["all_pairs"]["style_bias_score"] == 1.0

    cf = counterfactual_debias_verdicts(pairs, verdicts, n_folds=5)
    debiased_subgroups = evaluate_style_bias_subgroups(pairs, cf["debiased_verdicts"])
    # Counterfactual subtraction of the surface artifact bias pulls all-markdown
    # preference down toward zero (tie).
    assert abs(debiased_subgroups["all_pairs"]["style_bias_score"]) < 0.25

