"""E4: does style bias carry over into Vietnamese, and do humans really
prefer markdown? (master plan mục 8, W1, W10)

This experiment has a hard human-in-the-loop step that CANNOT run "in one
go" on Colab: master plan requires >= 3 Vietnamese annotators (mục 7, 8).
What this module automates:
  1. Translate the 50 STYLE + 50 STYLE_BA pairs to Vietnamese (LLM call).
  2. Generate a blind annotation sheet (CSV) with response order randomized
     per pair, plus a separate decoding key -- same split the upstream repo
     itself uses (results/human_study/ held answers+keys, gitignored; see
     src/data_audit.py module docstring).
  3. Run the judge(s) on the translated pairs (reuses src.experiments.style_bias
     logic once translation is done, since schema is unchanged).
What it does NOT automate: getting >= 3 people to actually fill in the sheet.
`ingest_annotations()` picks up after that happens.
"""

import csv
import json
import random
from pathlib import Path

from src.experiments.common import CostModel, run_cached_calls

TRANSLATE_PROMPT = (
    "Translate the following question and two responses into natural, fluent "
    "Vietnamese. Preserve all markdown syntax exactly (headers, bold, bullet "
    "points, code blocks, tables) -- translate only the prose content, not the "
    "markdown markers. Preserve mathematical notation and code unchanged.\n\n"
    "Reply with ONLY a JSON object with exactly these keys: "
    '"question", "response_a", "response_b". No other text.\n\n'
    "Question: {question}\n\nResponse A: {response_a}\n\nResponse B: {response_b}"
)


def _extract_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in translation output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def run_translation_experiment(
    pairs: list[dict],
    judge,
    judge_name: str,
    cache_path: Path,
    cost_model: CostModel | None = None,
    budget_usd: float = 1.0,
    dry_run: bool = False,
    verbose: bool = True,
    max_workers: int = 1,
) -> dict:
    """Translates every pair's question/response_a/response_b to Vietnamese,
    caching each call keyed by pair_id. `judge` must implement complete_raw().
    """
    items = [{"pair_id": pair["id"], "pair": pair} for pair in pairs]

    def call_fn(item):
        from src.judges.clients import JudgeVerdict

        pair = item["pair"]
        prompt = TRANSLATE_PROMPT.format(
            question=pair["question"], response_a=pair["response_a"], response_b=pair["response_b"]
        )
        text, in_tok, out_tok = judge.complete_raw(prompt, temperature=0.1, max_tokens=4000)
        translated = _extract_json_object(text)  # raises loudly if the model didn't return valid JSON
        for key in ("question", "response_a", "response_b"):
            if key not in translated:
                raise ValueError(f"translation for {item['pair_id']} missing key {key!r}: {translated}")
        return JudgeVerdict(verdict=json.dumps(translated, ensure_ascii=False), raw_text=text,
                             input_tokens=in_tok, output_tokens=out_tok)

    def row_fn(item, result, cost):
        return {
            "pair_id": item["pair_id"],
            "judge": judge_name,
            "translated_json": result.verdict,  # reusing the .verdict field to carry the JSON payload
            "raw_text": result.raw_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": cost,
        }

    return run_cached_calls(
        items,
        key_fields=("pair_id",),
        call_fn=call_fn,
        row_fn=row_fn,
        cache_path=cache_path,
        cost_model=cost_model,
        budget_usd=budget_usd,
        dry_run=dry_run,
        verbose=verbose,
        label=f"E4-translate/{judge_name}",
        max_workers=max_workers,
    )


def build_translated_pairs(original_pairs: list[dict], translation_cache: dict) -> list[dict]:
    """Merges each original pair's bias_type/manipulation (needed by
    markdown_side()) with its translated text. Raises if a pair's translation
    is missing rather than silently dropping it from the Vietnamese set.
    """
    out = []
    for pair in original_pairs:
        row = translation_cache.get((pair["id"],))
        if row is None:
            raise KeyError(f"no translation cached for pair {pair['id']!r}")
        translated = json.loads(row["translated_json"])
        out.append(
            {
                **pair,
                "question": translated["question"],
                "response_a": translated["response_a"],
                "response_b": translated["response_b"],
            }
        )
    return out


def generate_annotation_sheet(
    pairs: list[dict], out_csv_path: Path, key_json_path: Path, seed: int = 42
) -> None:
    """Writes a blind CSV (id, question, response_1, response_2, choice) for
    human annotators -- which original side ended up as response_1 vs
    response_2 is randomized per pair and saved SEPARATELY in key_json_path,
    never in the sheet itself (annotators must not see which side is markdown).
    """
    rng = random.Random(seed)
    key = {}
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "response_1", "response_2", "choice (1, 2, or tie)"])
        for pair in pairs:
            response_1_is_a = rng.random() < 0.5
            r1 = pair["response_a"] if response_1_is_a else pair["response_b"]
            r2 = pair["response_b"] if response_1_is_a else pair["response_a"]
            key[pair["id"]] = "A" if response_1_is_a else "B"
            writer.writerow([pair["id"], pair["question"], r1, r2, ""])
    key_json_path.parent.mkdir(parents=True, exist_ok=True)
    key_json_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest_annotations(filled_csv_paths: dict[str, Path], key_json_path: Path) -> list[dict]:
    """filled_csv_paths: {annotator_name: path to their filled-in copy of the
    sheet generate_annotation_sheet() produced}. Returns tidy records:
    {"pair_id", "annotator", "choice_original_side"} with "1"/"2" decoded back
    to "A"/"B" via the key, and "tie" passed through unchanged.
    """
    key = json.loads(key_json_path.read_text(encoding="utf-8"))
    records = []
    for annotator, path in filled_csv_paths.items():
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair_id = row["id"]
                raw_choice = row["choice (1, 2, or tie)"].strip().lower()
                if raw_choice not in ("1", "2", "tie"):
                    raise ValueError(
                        f"{annotator}'s sheet, pair {pair_id}: choice must be '1', '2', or 'tie', "
                        f"got {raw_choice!r}"
                    )
                if raw_choice == "tie":
                    original_side = "tie"
                else:
                    response_1_is = key[pair_id]
                    original_side = response_1_is if raw_choice == "1" else (
                        "B" if response_1_is == "A" else "A"
                    )
                records.append(
                    {"pair_id": pair_id, "annotator": annotator, "choice_original_side": original_side}
                )
    return records


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Plain Cohen's kappa for two annotators who labeled the same items in
    the same order -- no external dependency, since this is one small formula.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must be the same length (same items, same order)")
    n = len(labels_a)
    if n == 0:
        raise ValueError("no labels to compute kappa over")
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n
    categories = set(labels_a) | set(labels_b)
    pe = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if pe == 1.0:
        return 1.0  # both annotators used exactly one, identical category -- perfect agreement by definition
    return (po - pe) / (1 - pe)


def pairwise_kappas(records: list[dict]) -> dict:
    """{(annotator_a, annotator_b): kappa} for every pair of annotators,
    restricted to pair_ids both of them labeled.
    """
    by_annotator: dict[str, dict[str, str]] = {}
    for r in records:
        by_annotator.setdefault(r["annotator"], {})[r["pair_id"]] = r["choice_original_side"]

    annotators = sorted(by_annotator)
    out = {}
    for i, a in enumerate(annotators):
        for b in annotators[i + 1 :]:
            shared_ids = sorted(set(by_annotator[a]) & set(by_annotator[b]))
            if not shared_ids:
                continue
            labels_a = [by_annotator[a][pid] for pid in shared_ids]
            labels_b = [by_annotator[b][pid] for pid in shared_ids]
            out[(a, b)] = cohen_kappa(labels_a, labels_b)
    return out


def human_markdown_preference_rate(records: list[dict], pairs: list[dict]) -> dict:
    """Fraction of (annotator, pair) choices where the human picked the
    markdown-formatted side -- the direct analogue of the judge's
    style_bias_score numerator, for comparing against E3's judge results.
    """
    from src.experiments.style_bias import markdown_side

    by_id = {pair["id"]: pair for pair in pairs}
    n_markdown = n_plain = n_tie = 0
    for r in records:
        pair = by_id.get(r["pair_id"])
        if pair is None:
            continue
        side = markdown_side(pair)
        choice = r["choice_original_side"]
        if choice == "tie":
            n_tie += 1
        elif choice == side:
            n_markdown += 1
        else:
            n_plain += 1
    total = n_markdown + n_plain + n_tie
    return {
        "n_markdown_chosen": n_markdown,
        "n_plain_chosen": n_plain,
        "n_tie": n_tie,
        "markdown_preference_rate": (n_markdown / total) if total else None,
    }
