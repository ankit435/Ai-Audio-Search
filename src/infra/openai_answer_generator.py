"""Optional `AnswerGenerator` adapter backed by a real LLM (Stretch
Goal). Behind the same `AnswerGenerator` port as
`ExtractiveAnswerGenerator` — swappable via the composition root with
zero changes to `SearchAnswerService` or the `/search/answer` route.

**Not limited to OpenAI**: the `openai` Python SDK talks to any
OpenAI-compatible chat-completions endpoint via its `base_url` param,
so this one adapter also covers:
- **NVIDIA NIM** (`base_url="https://integrate.api.nvidia.com/v1"`,
  e.g. `model="meta/llama-3.1-8b-instruct"`) — set
  `AUDIO_SEARCH_ANSWER_BASE_URL` / `AUDIO_SEARCH_ANSWER_MODEL` env vars
  and `NVIDIA_API_KEY` (checked as a fallback to `OPENAI_API_KEY`).
- Any self-hosted OpenAI-compatible server (vLLM, Ollama, LM Studio,
  Together, Groq, etc.) serving open Hugging Face models — same idea,
  point `AUDIO_SEARCH_ANSWER_BASE_URL` at it.
- Hosted OpenAI models — leave `AUDIO_SEARCH_ANSWER_BASE_URL` unset.

New providers don't need a new class at all — just different
base_url/model/api-key configuration wired at the composition root
(`src/api/main.py::_build_answer_generator`). Only a genuinely
different wire protocol (not OpenAI-compatible) would need a new
`AnswerGenerator` adapter.

**Unverified in this environment**: instantiating this adapter requires
an API key and outbound HTTPS, which this environment's corporate proxy
has been blocking for every non-Azure-CDN host tested this session (see
SOLUTION.md "Known limitations"). The code follows the official
`openai` Python SDK's chat-completions usage and fails fast/typed if the
dependency or key is missing, but no live call has been made against it.
"""

from __future__ import annotations

import os

from src.domain.exceptions import EmbeddingFailedError
from src.domain.models import Chunk

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided transcript excerpts. "
    "Cite the speaker and timestamp for any claim. If the provided context "
    "does not contain the answer to the question, reply strictly with: "
    "'I do not know based on the provided context.' Do not guess or fabricate details."
)


class OpenAIAnswerGenerator:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not self._api_key:
            raise EmbeddingFailedError(
                context="OpenAIAnswerGenerator.__init__",
                reason=(
                    "No API key found — set OPENAI_API_KEY or NVIDIA_API_KEY "
                    "(or pass api_key=...); see docstring for why this adapter is optional/gated"
                ),
            )

    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EmbeddingFailedError(
                context="OpenAIAnswerGenerator.generate", reason=f"openai package not installed: {exc}"
            ) from exc

        excerpts = "\n".join(
            f"[{c.speaker_id} @ {c.start_time:.1f}s] {c.text}" for c in chunks
        )
        try:
            client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {query}\n\nTranscript excerpts:\n{excerpts}"},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception so
            # `ResilientAnswerGenerator` (and any other caller) can react to a single,
            # documented failure type regardless of *why* the API call failed (network
            # block, invalid/expired key, rate limit, model deprecation, etc.) — the
            # same "fail loud with typed context" contract every other adapter follows.
            raise EmbeddingFailedError(
                context="OpenAIAnswerGenerator.generate", reason=str(exc)
            ) from exc
