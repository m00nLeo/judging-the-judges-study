"""E6: Multi-Artifact Counterfactual GLM Debiasing & Confounding Audit (MAC-GLM).

Extends post-hoc counterfactual debiasing (e.g. Length-Controlled AlpacaEval,
Dubois et al. 2024; Bayesian GLM Calibration, 2025-2026) from a single length
dimension to a multi-dimensional surface artifact space:
  - Position intercept (preference for slot A at zero artifact difference)
  - Length asymmetry: log((words(A) + 1) / (words(B) + 1))
  - Markdown formatting density differences: bold (**), headers (#), lists,
    and structural blocks (code fences / tables)
  - Completeness / truncation asymmetry: whether one response is cut off
    mid-sentence without terminal punctuation (discovered in upstream
    data/custom/controlled_pairs.jsonl).

Operates entirely on pair texts and cached judge verdicts ($0 API cost).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from src.experiments.style_bias import markdown_side

_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
_HEADER_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_LIST_RE = re.compile(r"^\s*(?:[*\-+]|\d+\.)\s+\S", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_TERMINAL_CHARS = tuple(".!?\"')]}*`$:;|")

ARTIFACT_FEATURES = (
    "log_word_ratio",
    "delta_bold",
    "delta_header",
    "delta_list",
    "delta_struct",
    "delta_completeness",
)


def is_truncated_response(text: str) -> bool:
    """True if a non-empty response appears cut off mid-sentence (no terminal
    punctuation at the end, or ends with an empty section header / colon).
    """
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(":"):
        return True
    return not stripped.endswith(_TERMINAL_CHARS)


def extract_surface_features(text: str) -> dict[str, float]:
    """Counts surface length, markdown formatting elements, and completeness
    for a single response string.
    """
    words = text.split()
    n_words = len(words)
    n_bold = len(_BOLD_RE.findall(text))
    n_header = len(_HEADER_RE.findall(text))
    n_list = len(_LIST_RE.findall(text))
    n_struct = len(_CODE_FENCE_RE.findall(text)) + len(_TABLE_ROW_RE.findall(text))
    complete = 0.0 if is_truncated_response(text) else 1.0
    return {
        "word_count": float(n_words),
        "char_count": float(len(text)),
        "bold_count": float(n_bold),
        "header_count": float(n_header),
        "list_count": float(n_list),
        "struct_count": float(n_struct),
        "markdown_total": float(n_bold + n_header + n_list + n_struct),
        "is_complete": complete,
    }


def extract_pair_artifact_vector(pair: dict) -> dict[str, float]:
    """Computes the antisymmetric surface artifact vector z_AB = f(A) - f(B)
    for a (response_a, response_b) pair. Flipping A and B negates every entry.
    """
    fa = extract_surface_features(pair["response_a"])
    fb = extract_surface_features(pair["response_b"])
    return {
        "log_word_ratio": math.log((fa["word_count"] + 1.0) / (fb["word_count"] + 1.0)),
        "delta_bold": fa["bold_count"] - fb["bold_count"],
        "delta_header": fa["header_count"] - fb["header_count"],
        "delta_list": fa["list_count"] - fb["list_count"],
        "delta_struct": fa["struct_count"] - fb["struct_count"],
        "delta_markdown_total": fa["markdown_total"] - fb["markdown_total"],
        "delta_completeness": fa["is_complete"] - fb["is_complete"],
        "words_a": fa["word_count"],
        "words_b": fb["word_count"],
        "complete_a": fa["is_complete"],
        "complete_b": fb["is_complete"],
    }


def audit_dataset_confounding(pairs: list[dict]) -> pd.DataFrame:
    """Builds a tidy DataFrame of surface artifacts for every pair in a dataset."""
    rows = []
    for pair in pairs:
        vec = extract_pair_artifact_vector(pair)
        rows.append(
            {
                "id": pair["id"],
                "bias_type": pair.get("bias_type", "benchmark"),
                "human_preference": pair.get("human_preference"),
                **vec,
            }
        )
    return pd.DataFrame(rows)


def summarize_style_confounding(style_pairs: list[dict]) -> dict:
    """Quantifies the hidden confounding (length asymmetry and mid-sentence
    truncation) between the Markdown side and Plain Prose side in the upstream
    STYLE + STYLE_BA controlled pairs.
    """
    n_total = len(style_pairs)
    if n_total == 0:
        raise ValueError("style_pairs must not be empty")

    md_longer = 0
    plain_truncated_only = 0
    md_truncated_only = 0
    both_complete = 0
    word_ratios = []

    for pair in style_pairs:
        side = markdown_side(pair)
        fa = extract_surface_features(pair["response_a"])
        fb = extract_surface_features(pair["response_b"])
        f_md, f_plain = (fa, fb) if side == "A" else (fb, fa)

        ratio = (f_md["word_count"] + 1.0) / (f_plain["word_count"] + 1.0)
        word_ratios.append(ratio)
        if f_md["word_count"] > f_plain["word_count"]:
            md_longer += 1
        if f_md["is_complete"] == 1.0 and f_plain["is_complete"] == 0.0:
            plain_truncated_only += 1
        elif f_md["is_complete"] == 0.0 and f_plain["is_complete"] == 1.0:
            md_truncated_only += 1
        elif f_md["is_complete"] == 1.0 and f_plain["is_complete"] == 1.0:
            both_complete += 1

    return {
        "n_pairs": n_total,
        "frac_markdown_longer": md_longer / n_total,
        "mean_word_ratio_md_over_plain": float(np.mean(word_ratios)),
        "median_word_ratio_md_over_plain": float(np.median(word_ratios)),
        "n_plain_truncated_only": plain_truncated_only,
        "n_markdown_truncated_only": md_truncated_only,
        "n_both_complete": both_complete,
        "frac_asymmetric_truncation": (plain_truncated_only + md_truncated_only) / n_total,
    }


def _verdict_to_numeric(verdict: str) -> float:
    if verdict == "A":
        return 1.0
    if verdict == "B":
        return 0.0
    if verdict == "tie":
        return 0.5
    raise ValueError(f"unexpected verdict {verdict!r}, expected 'A', 'B', or 'tie'")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_counterfactual_glm(
    pairs: list[dict],
    verdicts: list[str],
    features: tuple[str, ...] = ARTIFACT_FEATURES,
    l2_penalty: float = 1.0,
    standardize: bool = True,
) -> dict:
    """Fits a quasi-binomial (fractional logit) Generalized Linear Model with
    L2 regularization:
        logit P(Judge picks A) = gamma_0 + sum_k gamma_k * z_{i,k}

    Returns a dictionary containing:
      - `coefficients`: DataFrame with feature name, estimate, std_error, z_score,
        p_value, ci_95_lo, ci_95_hi (Bias Fingerprint for RQ2)
      - `intercept`: position bias log-odds at z = 0
      - `feature_means`, `feature_stds`: scaling parameters used
      - `raw_weights`: unstandardized weights for direct counterfactual subtraction
    """
    if len(pairs) != len(verdicts):
        raise ValueError("pairs and verdicts must have the same length")
    if len(pairs) < 4:
        raise ValueError("need at least 4 observations to fit counterfactual GLM")

    df_art = audit_dataset_confounding(pairs)
    X_raw = df_art[list(features)].to_numpy(dtype=float)
    y = np.array([_verdict_to_numeric(v) for v in verdicts], dtype=float)

    n, k = X_raw.shape
    if standardize:
        means = np.zeros(k, dtype=float)  # keep z=0 centered at zero physical difference!
        stds = np.std(X_raw, axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)
        X_scaled = (X_raw - means) / stds
    else:
        means = np.zeros(k, dtype=float)
        stds = np.ones(k, dtype=float)
        X_scaled = X_raw.copy()

    # Design matrix: column 0 is intercept (position bias), columns 1..k are features
    X_design = np.column_stack([np.ones(n, dtype=float), X_scaled])
    reg_mask = np.array([0.0] + [1.0] * k, dtype=float)

    def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        logits = X_design @ w
        p = _sigmoid(logits)
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        pen = 0.5 * l2_penalty * float(np.sum((w * reg_mask) ** 2))
        grad = X_design.T @ (p - y) + l2_penalty * (w * reg_mask)
        return float(nll + pen), grad

    res = minimize(
        loss_and_grad,
        x0=np.zeros(k + 1, dtype=float),
        jac=True,
        method="L-BFGS-B",
    )
    w_hat = res.x
    p_hat = _sigmoid(X_design @ w_hat)
    weights_diag = np.clip(p_hat * (1.0 - p_hat), 1e-6, None)
    hessian = (X_design.T * weights_diag) @ X_design + np.diag(l2_penalty * reg_mask + 1e-6)
    cov = np.linalg.inv(hessian)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    names = ["position_intercept"] + list(features)
    rows = []
    for idx, name in enumerate(names):
        est = float(w_hat[idx])
        err = float(se[idx])
        z_val = est / err if err > 0 else 0.0
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z_val))))
        rows.append(
            {
                "feature": name,
                "estimate": round(est, 4),
                "std_error": round(err, 4),
                "z_score": round(z_val, 4),
                "p_value": round(p_val, 4),
                "ci_95_lo": round(est - 1.96 * err, 4),
                "ci_95_hi": round(est + 1.96 * err, 4),
                "significant_at_0.05": bool(p_val < 0.05),
            }
        )

    unscaled_feature_weights = w_hat[1:] / stds
    return {
        "coefficients": pd.DataFrame(rows),
        "intercept": float(w_hat[0]),
        "unscaled_weights": {f: float(w) for f, w in zip(features, unscaled_feature_weights)},
        "feature_stds": {f: float(s) for f, s in zip(features, stds)},
        "n_obs": n,
        "converged": bool(res.success),
    }


def counterfactual_debias_verdicts(
    pairs: list[dict],
    verdicts: list[str],
    features: tuple[str, ...] = ARTIFACT_FEATURES,
    l2_penalty: float = 1.0,
    n_folds: int = 5,
    tie_band: float = 0.12,
) -> dict:
    """Out-of-fold post-hoc counterfactual debiasing (do(z = 0)).

    Uses the Neyman-orthogonal residual formulation:
      1. Fits the GLM on the other (n_folds - 1) folds to estimate the expected
         surface-artifact preference probability:
             p_artifact(z_i) = sigmoid(gamma_0 + sum_k gamma_k * z_{i,k})
      2. On held-out items with observed numeric verdict y_i in {1.0, 0.5, 0.0},
         computes the counterfactual preference probability at z = 0:
             p_i^cf = clip(0.5 + (y_i - p_artifact(z_i)), 0.0, 1.0)
         So a verdict explained entirely by surface artifacts (y_i ~= p_artifact)
         collapses to 0.5 ("tie") instead of over-correcting to the opposite side,
         while a verdict that goes against or exceeds surface bias is preserved.
      3. Assigns debiased verdict: 'A' if p_i^cf > 0.5 + tie_band,
         'B' if p_i^cf < 0.5 - tie_band, else 'tie'.
    """
    n = len(pairs)
    if n != len(verdicts):
        raise ValueError("pairs and verdicts must have the same length")
    if n < n_folds or n_folds < 2:
        raise ValueError(f"need len(pairs) >= n_folds >= 2 (got n={n}, n_folds={n_folds})")

    df_art = audit_dataset_confounding(pairs)
    X_raw = df_art[list(features)].to_numpy(dtype=float)

    indices = np.arange(n)
    folds = np.array_split(indices, n_folds)
    debiased_probs = np.zeros(n, dtype=float)
    bias_shifts = np.zeros(n, dtype=float)

    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.setdiff1d(indices, test_idx)
        train_pairs = [pairs[i] for i in train_idx]
        train_verdicts = [verdicts[i] for i in train_idx]

        fit = fit_counterfactual_glm(
            train_pairs, train_verdicts, features=features, l2_penalty=l2_penalty
        )
        w_unscaled = np.array([fit["unscaled_weights"][f] for f in features], dtype=float)
        intercept = fit["intercept"]

        for i in test_idx:
            y_obs = _verdict_to_numeric(verdicts[i])
            artifact_logit = intercept + float(X_raw[i] @ w_unscaled)
            p_artifact = float(_sigmoid(np.array([artifact_logit]))[0])
            bias_shifts[i] = artifact_logit
            debiased_probs[i] = float(np.clip(0.5 + (y_obs - p_artifact), 0.0, 1.0))

    debiased_verdicts = []
    for p in debiased_probs:
        if p > 0.5 + tie_band:
            debiased_verdicts.append("A")
        elif p < 0.5 - tie_band:
            debiased_verdicts.append("B")
        else:
            debiased_verdicts.append("tie")

    return {
        "debiased_verdicts": debiased_verdicts,
        "debiased_probs": [round(float(p), 4) for p in debiased_probs],
        "estimated_bias_logit": [round(float(b), 4) for b in bias_shifts],
    }


def evaluate_style_bias_subgroups(
    style_pairs: list[dict], verdicts: list[str]
) -> dict:
    """Computes style bias score across three slices of the STYLE dataset:
      1. `all_pairs`: all 100 pairs (Soumik's headline metric)
      2. `clean_complete_pairs`: pairs where BOTH responses are complete
         (not truncated mid-sentence) and within 1.5x word length of each other
      3. `confounded_pairs`: pairs with asymmetric truncation or >1.5x length gap
    """
    if len(style_pairs) != len(verdicts):
        raise ValueError("style_pairs and verdicts must have the same length")

    def _score(subset_indices: list[int]) -> dict:
        if not subset_indices:
            return {"n": 0, "style_bias_score": None, "n_markdown": 0, "n_plain": 0, "n_tie": 0}
        n_md = n_pl = n_tie = 0
        for i in subset_indices:
            side = markdown_side(style_pairs[i])
            v = verdicts[i]
            if v == "tie":
                n_tie += 1
            elif v == side:
                n_md += 1
            else:
                n_pl += 1
        tot = len(subset_indices)
        return {
            "n": tot,
            "style_bias_score": round((n_md - n_pl) / tot, 4),
            "n_markdown": n_md,
            "n_plain": n_pl,
            "n_tie": n_tie,
        }

    all_idx = list(range(len(style_pairs)))
    clean_idx = []
    confounded_idx = []
    for i, pair in enumerate(style_pairs):
        vec = extract_pair_artifact_vector(pair)
        both_complete = vec["complete_a"] == 1.0 and vec["complete_b"] == 1.0
        length_balanced = abs(vec["log_word_ratio"]) <= math.log(1.5)
        if both_complete and length_balanced:
            clean_idx.append(i)
        else:
            confounded_idx.append(i)

    return {
        "all_pairs": _score(all_idx),
        "clean_complete_pairs": _score(clean_idx),
        "confounded_pairs": _score(confounded_idx),
    }

