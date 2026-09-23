"""Generic `Embedder` adapter for any OpenAI-compatible `/v1/embeddings`
endpoint — NOT limited to OpenAI itself. This is the extensibility point
for using a hosted/GPU-accelerated embedding model instead of running
`sentence-transformers` locally, e.g.:

- **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`, model
  `nvidia/nv-embedqa-e5-v5` or similar) — set
  `AUDIO_SEARCH_EMBEDDING_PROVIDER=openai-compatible`,
  `AUDIO_SEARCH_EMBEDDING_BASE_URL=https://integrate.api.nvidia.com/v1`,
  `AUDIO_SEARCH_EMBEDDING_MODEL_NAME=nvidia/nv-embedqa-e5-v5`,
  `NVIDIA_API_KEY=...` (or `OPENAI_API_KEY`, checked as a fallback).
- Any self-hosted OpenAI-compatible embeddings server (vLLM,
  Text-Embeddings-Inference, Ollama, LM Studio, etc.) — same idea,
  point `AUDIO_SEARCH_EMBEDDING_BASE_URL` at it.
- Hosted OpenAI embeddings (`text-embedding-3-small`) — leave
  `AUDIO_SEARCH_EMBEDDING_BASE_URL` unset.

Kept behind the same `Embedder` port as `SentenceTransformerEmbedder`
(`src/infra/sentence_transformer_embedder.py`), so the composition root
(`src/api/main.py`) is the only place that decides which one is active —
zero changes needed in `application`/`domain` to switch providers, and
new providers can be added the same way (implement `embed`/`embed_batch`,
register in `_build_embedder()`).

**Vector dimension note**: `db/schema.sql` declares `vector(384)` to
match the default `all-MiniLM-L6-v2` model. Switching to a model with a
different output dimension (most hosted models are 768/1024/1536+)
requires updating that column's dimension accordingly — see SETUP.md.
"""

from __future__ import annotations

import os

from src.domain.exceptions import EmbeddingFailedError

DEFAULT_MODEL_NAME = "text-embedding-3-small"


class OpenAICompatibleEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, base_url: str | None = None, api_key: str | None = None) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not self._api_key:
            raise EmbeddingFailedError(
                context="OpenAICompatibleEmbedder.__init__",
                reason="No API key found — set OPENAI_API_KEY or NVIDIA_API_KEY (or pass api_key=...)",
            )
        self._client = None  # lazy-loaded on first use

    def _load_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # local import: optional dep, shared with OpenAIAnswerGenerator

            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._load_client()
            response = await client.embeddings.create(model=self._model_name, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception
            raise EmbeddingFailedError(
                context=f"model={self._model_name} base_url={self._base_url}", reason=str(exc)
            ) from exc
