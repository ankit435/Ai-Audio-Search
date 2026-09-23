"""`Transcriber` decorator (Stretch/robustness) — automatic fallback
between two Whisper backends when the preferred one's model download is
blocked. Same "sticky" circuit-breaker pattern as `ResilientDiarizer`:
reacts only to the typed `TranscriptionFailedError` the `Transcriber`
port contract guarantees, so it's portable across whichever backend/
environment combination is actually blocked (Hugging Face Hub vs. an
LLM-hosting CDN vs. no network at all) — not hardcoded to this
session's specific proxy behavior.
"""

from __future__ import annotations

import logging

from src.domain.exceptions import TranscriptionFailedError
from src.domain.models import SpeakerTurn
from src.domain.ports import Transcriber
from src.infra.logging_config import log_event

logger = logging.getLogger(__name__)


class ResilientTranscriber:
    def __init__(self, primary: Transcriber, fallback: Transcriber) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_available = True

    async def transcribe(self, audio_file_path: str) -> list[SpeakerTurn]:
        if self._primary_available:
            try:
                return await self._primary.transcribe(audio_file_path)
            except TranscriptionFailedError as exc:
                self._primary_available = False
                log_event(
                    logger, logging.WARNING, "resilient_transcriber.primary_unavailable",
                    primary=type(self._primary).__name__, fallback=type(self._fallback).__name__,
                    reason=exc.reason,
                )
        return await self._fallback.transcribe(audio_file_path)
