import csv
import json

import pytest

from src.experiments.vietnamese_style import (
    _extract_json_object,
    build_translated_pairs,
    cohen_kappa,
    generate_annotation_sheet,
    human_markdown_preference_rate,
    ingest_annotations,
    pairwise_kappas,
    run_translation_experiment,
)
STYLE_PAIR = {
    "id": "style_000",
    "question": "why is sqrt(2) irrational?",
    "response_a": "**markdown** answer",
    "response_b": "plain answer",
    "bias_type": "style",
    "manipulation": "response_a uses markdown formatting, response_b is plain prose with same content",
}


def test_extract_json_object_from_messy_model_output():
    text = 'Here you go:\n{"question": "q", "response_a": "a", "response_b": "b"}\nHope that helps!'
    assert _extract_json_object(text) == {"question": "q", "response_a": "a", "response_b": "b"}


def test_extract_json_object_raises_on_no_json():
    with pytest.raises(ValueError):
        _extract_json_object("no json here at all")


class FakeTranslatorJudge:
    def complete_raw(self, prompt, temperature=0.1, max_tokens=4000):
        payload = {"question": "cau hoi", "response_a": "**markdown** tra loi", "response_b": "tra loi thuong"}
        return json.dumps(payload, ensure_ascii=False), 50, 80


def test_run_translation_and_build_translated_pairs(tmp_path):
    cache_path = tmp_path / "translate_cache.jsonl"
    cache = run_translation_experiment(
        [STYLE_PAIR], FakeTranslatorJudge(), judge_name="fake", cache_path=cache_path, dry_run=True
    )
    translated = build_translated_pairs([STYLE_PAIR], cache)
    assert len(translated) == 1
    assert translated[0]["question"] == "cau hoi"
    assert translated[0]["bias_type"] == "style"  # carried over from original, not overwritten
    assert translated[0]["manipulation"] == STYLE_PAIR["manipulation"]


def test_build_translated_pairs_raises_on_missing_translation():
    with pytest.raises(KeyError):
        build_translated_pairs([STYLE_PAIR], translation_cache={})


def test_generate_and_ingest_annotation_sheet_roundtrip(tmp_path):
    csv_path = tmp_path / "sheet.csv"
    key_path = tmp_path / "key.json"
    generate_annotation_sheet([STYLE_PAIR], csv_path, key_path, seed=42)

    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert key["style_000"] in ("A", "B")

    # Simulate an annotator filling in "1" (whichever side that decodes to).
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 1
    filled_path = tmp_path / "annotator1_filled.csv"
    with filled_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "response_1", "response_2", "choice (1, 2, or tie)"])
        writer.writerow([rows[0]["id"], rows[0]["question"], rows[0]["response_1"], rows[0]["response_2"], "1"])

    records = ingest_annotations({"annotator1": filled_path}, key_path)
    assert len(records) == 1
    assert records[0]["choice_original_side"] == key["style_000"]


def test_ingest_annotations_rejects_invalid_choice(tmp_path):
    csv_path = tmp_path / "sheet.csv"
    key_path = tmp_path / "key.json"
    generate_annotation_sheet([STYLE_PAIR], csv_path, key_path, seed=42)
    bad_path = tmp_path / "bad.csv"
    with bad_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "response_1", "response_2", "choice (1, 2, or tie)"])
        writer.writerow(["style_000", "q", "r1", "r2", "maybe"])
    with pytest.raises(ValueError):
        ingest_annotations({"a": bad_path}, key_path)


def test_cohen_kappa_perfect_and_chance_agreement():
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    # Same label used every time by both -- treated as perfect agreement by convention.
    assert cohen_kappa(["A", "A", "A"], ["A", "A", "A"]) == 1.0


def test_cohen_kappa_below_chance_agreement_is_negative():
    # Both use A and B equally often, but always disagree -> below-chance agreement.
    assert cohen_kappa(["A", "B", "A", "B"], ["B", "A", "B", "A"]) < 0


def test_cohen_kappa_requires_equal_length():
    with pytest.raises(ValueError):
        cohen_kappa(["A"], ["A", "B"])


def test_pairwise_kappas_only_over_shared_pair_ids():
    records = [
        {"pair_id": "p1", "annotator": "x", "choice_original_side": "A"},
        {"pair_id": "p2", "annotator": "x", "choice_original_side": "B"},
        {"pair_id": "p1", "annotator": "y", "choice_original_side": "A"},
        {"pair_id": "p1", "annotator": "z", "choice_original_side": "B"},
    ]
    result = pairwise_kappas(records)
    assert ("x", "y") in result
    assert result[("x", "y")] == 1.0  # both said A on the one shared item
    assert ("x", "z") in result
    # Disagreed on their one shared item; with n=1 and no category overlap, pe=0, so kappa=0
    # (not -1 -- Cohen's kappa needs >=2 items to show negative agreement-below-chance).
    assert result[("x", "z")] == 0.0


def test_human_markdown_preference_rate():
    records = [
        {"pair_id": "style_000", "choice_original_side": "A"},  # A is markdown -> counted as markdown
        {"pair_id": "style_000", "choice_original_side": "B"},
    ]
    result = human_markdown_preference_rate(records, [STYLE_PAIR])
    assert result["n_markdown_chosen"] == 1
    assert result["n_plain_chosen"] == 1
    assert result["markdown_preference_rate"] == 0.5
