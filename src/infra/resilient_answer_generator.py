"""`AnswerGenerator` decorator (Stretch/robustness) — automatic fallback
from a real LLM adapter (`OpenAIAnswerGenerator`) to the offline
`ExtractiveAnswerGenerator` when the LLM is unavailable for *any* reason
(no API key, no network, blocked corporate proxy, rate limit, model
deprecation, etc.). Reacts only to the typed `EmbeddingFailedError` the
`AnswerGenerator`/adapter contract guarantees — portable across
environments, not hardcoded to this session's specific proxy behavior.

Same "sticky" circuit-breaker as `ResilientDiarizer`/`ResilientTranscriber`:
once the primary has failed once, every subsequent call goes straight to
the fallback for the lifetime of the instance.
"""

from __future__ import annotations

import logging

from src.domain.exceptions import EmbeddingFailedError
from src.domain.models import Chunk
from src.domain.ports import AnswerGenerator
from src.infra.logging_config import log_event

logger = logging.getLogger(__name__)


class ResilientAnswerGenerator:
    def __init__(self, primary: AnswerGenerator, fallback: AnswerGenerator) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_available = True

    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        if self._primary_available:
            try:
                return await self._primary.generate(query, chunks)
            except EmbeddingFailedError as exc:
                self._primary_available = False
                log_event(
                    logger, logging.WARNING, "resilient_answer_generator.primary_unavailable",
                    primary=type(self._primary).__name__, fallback=type(self._fallback).__name__,
                    reason=exc.reason,
                )
        return await self._fallback.generate(query, chunks)
