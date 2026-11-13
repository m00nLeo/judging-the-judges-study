"""Unified judge client interface for E2+ (first experiments that call a real API).

Uses direct provider APIs (Anthropic, OpenAI, Google) instead of the original
paper's Vertex AI endpoints -- per master plan section 7, this is a recorded
confound (endpoint differences), not something hidden.

Keys are read ONLY from environment variables or, on Colab, from
`google.colab.userdata` secrets -- never hardcoded in a notebook cell.
Provider SDKs are imported lazily inside each class so that importing this
module (and running in dry-run mode) never requires any of them installed.
"""

import os
import random
from dataclasses import dataclass


def get_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        try:
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    except ImportError:
        pass
    return os.environ.get(name)


@dataclass
class JudgeVerdict:
    verdict: str  # "A" | "B" | "tie"
    raw_text: str
    input_tokens: int
    output_tokens: int


PROMPT_TEMPLATE = (
    "You are judging which of two responses better answers the question. "
    'Reply with exactly one token: "A", "B", or "tie".\n\n'
    "Question: {question}\n\n"
    "Response A: {response_a}\n\n"
    "Response B: {response_b}\n\n"
    "Verdict (A, B, or tie):"
)


def _parse_verdict(text: str) -> str:
    t = text.strip().upper()
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return "tie"


class DryRunJudge:
    """Deterministic-seeded fake judge. No network, no key, zero cost --
    used to validate the pipeline (caching, budget cap, metric math) before
    any real API key is involved.
    """

    name = "dry-run"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        verdict = self._rng.choice(["A", "B", "tie"])
        return JudgeVerdict(verdict=verdict, raw_text=f"[dry-run] {verdict}", input_tokens=0, output_tokens=0)


class AnthropicJudge:
    def __init__(self, model_id: str = "claude-sonnet-4-5", api_key: str | None = None):
        import anthropic

        self.name = model_id
        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=api_key or get_secret("ANTHROPIC_API_KEY"))

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        prompt = PROMPT_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=5,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        return JudgeVerdict(
            verdict=_parse_verdict(text),
            raw_text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class OpenAIJudge:
    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        import openai

        self.name = model_id
        self.model_id = model_id
        self._client = openai.OpenAI(api_key=api_key or get_secret("OPENAI_API_KEY"))

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        prompt = PROMPT_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=5,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return JudgeVerdict(
            verdict=_parse_verdict(text),
            raw_text=text,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )


class GeminiJudge:
    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: str | None = None):
        from google import genai

        self.name = model_id
        self.model_id = model_id
        self._client = genai.Client(api_key=api_key or get_secret("GOOGLE_API_KEY"))

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        from google.genai import types

        prompt = PROMPT_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=5),
        )
        text = resp.text or ""
        usage = resp.usage_metadata
        return JudgeVerdict(
            verdict=_parse_verdict(text),
            raw_text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


def build_judge(name: str, dry_run: bool = False):
    if dry_run:
        return DryRunJudge()
    if name.startswith("claude"):
        return AnthropicJudge(model_id=name)
    if name.startswith("gpt"):
        return OpenAIJudge(model_id=name)
    if name.startswith("gemini"):
        return GeminiJudge(model_id=name)
    raise ValueError(f"unknown judge name: {name!r} (expected a claude-/gpt-/gemini- prefixed model id)")
