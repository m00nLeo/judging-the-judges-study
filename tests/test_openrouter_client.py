import pytest

from src.judges.clients import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL_ALIASES,
    DryRunJudge,
    OpenRouterJudge,
    build_judge,
)


def test_build_judge_dry_run_returns_dry_run_judge():
    judge = build_judge("gemini-3.8-flash", dry_run=True)
    assert isinstance(judge, DryRunJudge)


def test_build_judge_routes_to_openrouter_when_openrouter_key_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    j_gemini = build_judge("gemini-3.8-flash", dry_run=False)
    assert isinstance(j_gemini, OpenRouterJudge)
    assert j_gemini.name == "gemini-3.8-flash"
    assert j_gemini.model_id == OPENROUTER_MODEL_ALIASES["gemini-3.8-flash"]
    assert str(j_gemini._client.base_url).rstrip("/") == OPENROUTER_BASE_URL

    j_claude = build_judge("claude-sonnet-4-5", dry_run=False)
    assert isinstance(j_claude, OpenRouterJudge)
    assert j_claude.model_id == OPENROUTER_MODEL_ALIASES["claude-sonnet-4-5"]

    j_explicit = build_judge("google/gemini-2.5-flash", dry_run=False)
    assert isinstance(j_explicit, OpenRouterJudge)
    assert j_explicit.model_id == "google/gemini-2.5-flash"


def test_openrouter_judge_raises_clear_error_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_judge("gemini-3.8-flash", dry_run=False)

