"""SentenceTransformer-based `Embedder` adapter — local, no hosted API call.

Model loaded lazily (once) and reused; `encode` is CPU/GPU-bound and
synchronous in the underlying library, so it's offloaded to a thread via
`asyncio.to_thread` to keep the adapter's `async def` contract honest
(non-blocking from the event loop's perspective).
"""

from __future__ import annotations

import asyncio

from src.domain.exceptions import EmbeddingFailedError

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, matches db/schema.sql vector(384)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded on first use

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: heavy/optional dep

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            model = self._load_model()
            vectors = await asyncio.to_thread(model.encode, texts, convert_to_numpy=True)
            return [vector.tolist() for vector in vectors]
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception
            raise EmbeddingFailedError(context=f"model={self._model_name}", reason=str(exc)) from exc
