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

# Without an explicit timeout, a hung connection blocks forever -- no retry,
# no error, the call just never returns. This is what "running for an hour
# with no progress and no error" usually means in practice.
DEFAULT_TIMEOUT_SECONDS = 60.0


def get_secret(name: str) -> str | None:
    """Look up a secret from Colab Secrets (if running on Colab), else env vars.

    Only "the secret isn't set at all" falls through silently to env vars --
    that's the expected case when running locally. Anything else Colab-specific
    (notebook access toggled off for this secret, etc.) raises immediately with
    an actionable message, instead of silently returning None and producing a
    confusing error two layers down in some provider SDK.
    """
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return os.environ.get(name)

    try:
        value = userdata.get(name)
        if value:
            return value
    except userdata.SecretNotFoundError:
        pass  # not set as a Colab secret -- fall through to env vars below
    except userdata.NotebookAccessError as e:
        raise RuntimeError(
            f"Colab secret {name!r} exists but this notebook doesn't have access to it. "
            f"Click the key icon in the left sidebar and toggle 'Notebook access' ON for {name!r}."
        ) from e
    return os.environ.get(name)


def require_secret(secret_name: str) -> str:
    """get_secret(), but raises immediately with where-to-fix instructions if
    nothing was found -- instead of passing None into a provider SDK and
    getting a confusing error from inside that library instead.
    """
    value = get_secret(secret_name)
    if not value:
        raise ValueError(
            f"No value found for {secret_name!r}. On Colab: click the key icon in the left "
            f"sidebar, add a secret named exactly {secret_name!r}, and toggle 'Notebook access' "
            f"ON for it. Locally: set it as an environment variable before running."
        )
    return value


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

    def complete_raw(self, prompt: str, temperature: float = 0.1, max_tokens: int = 5):
        verdict = self._rng.choice(["A", "B", "tie"])
        return f"[dry-run] {verdict}", 0, 0

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        text, in_tok, out_tok = self.complete_raw("", temperature, max_tokens=5)
        return JudgeVerdict(verdict=_parse_verdict(text.replace("[dry-run] ", "")), raw_text=text, input_tokens=in_tok, output_tokens=out_tok)


class AnthropicJudge:
    def __init__(self, model_id: str = "claude-sonnet-4-5", api_key: str | None = None):
        import anthropic

        self.name = model_id
        self.model_id = model_id
        self._client = anthropic.Anthropic(
            api_key=api_key or require_secret("ANTHROPIC_API_KEY"), timeout=DEFAULT_TIMEOUT_SECONDS
        )

    def complete_raw(self, prompt: str, temperature: float = 0.1, max_tokens: int = 5):
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        prompt = PROMPT_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
        text, in_tok, out_tok = self.complete_raw(prompt, temperature, max_tokens=5)
        return JudgeVerdict(verdict=_parse_verdict(text), raw_text=text, input_tokens=in_tok, output_tokens=out_tok)


class OpenAIJudge:
    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        import openai

        self.name = model_id
        self.model_id = model_id
        self._client = openai.OpenAI(
            api_key=api_key or require_secret("OPENAI_API_KEY"), timeout=DEFAULT_TIMEOUT_SECONDS
        )

    def complete_raw(self, prompt: str, temperature: float = 0.1, max_tokens: int = 5):
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        prompt = PROMPT_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
        text, in_tok, out_tok = self.complete_raw(prompt, temperature, max_tokens=5)
        return JudgeVerdict(verdict=_parse_verdict(text), raw_text=text, input_tokens=in_tok, output_tokens=out_tok)


class GroqJudge:
    """Groq's OpenAI-compatible endpoint, used to run Llama 3.3-70B for free
    (no credit card) -- a stand-in for the Anthropic/OpenAI judges when
    budget is $0. Llama 3.3-70B is already one of the 5 judges in the paper's
    own roster, so this isn't an extra model, just a different (free) endpoint
    for it -- another confound to record alongside Vertex vs direct-API (see
    module docstring), since Groq likely serves a different quantization than
    Vertex AI Model Garden did.
    """

    def __init__(self, model_id: str = "llama-3.3-70b-versatile", api_key: str | None = None):
        import openai

        self.name = model_id
        self.model_id = model_id
        self._client = openai.OpenAI(
            api_key=api_key or require_secret("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

    def complete_raw(self, prompt: str, temperature: float = 0.1, max_tokens: int = 5):
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        prompt = PROMPT_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
        text, in_tok, out_tok = self.complete_raw(prompt, temperature, max_tokens=5)
        return JudgeVerdict(verdict=_parse_verdict(text), raw_text=text, input_tokens=in_tok, output_tokens=out_tok)


class GeminiJudge:
    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: str | None = None):
        from google import genai
        from google.genai import types

        self.name = model_id
        self.model_id = model_id
        self._client = genai.Client(
            api_key=api_key or require_secret("GOOGLE_API_KEY"),
            # google-genai's HttpOptions.timeout is in milliseconds, not seconds
            # (unlike anthropic/openai above) -- double-check this against the
            # installed SDK version if timeouts don't seem to take effect.
            http_options=types.HttpOptions(timeout=int(DEFAULT_TIMEOUT_SECONDS * 1000)),
        )

    def complete_raw(self, prompt: str, temperature: float = 0.1, max_tokens: int = 5):
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens),
        )
        text = resp.text or ""
        usage = resp.usage_metadata
        return (
            text,
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
        )

    def judge(
        self, question: str, response_a: str, response_b: str, temperature: float = 0.1
    ) -> JudgeVerdict:
        prompt = PROMPT_TEMPLATE.format(question=question, response_a=response_a, response_b=response_b)
        text, in_tok, out_tok = self.complete_raw(prompt, temperature, max_tokens=5)
        return JudgeVerdict(verdict=_parse_verdict(text), raw_text=text, input_tokens=in_tok, output_tokens=out_tok)


def build_judge(name: str, dry_run: bool = False):
    if dry_run:
        return DryRunJudge()
    if name.startswith("claude"):
        return AnthropicJudge(model_id=name)
    if name.startswith("gpt"):
        return OpenAIJudge(model_id=name)
    if name.startswith("gemini"):
        return GeminiJudge(model_id=name)
    if name.startswith("llama"):
        return GroqJudge(model_id=name)
    raise ValueError(
        f"unknown judge name: {name!r} (expected a claude-/gpt-/gemini-/llama- prefixed model id)"
    )
