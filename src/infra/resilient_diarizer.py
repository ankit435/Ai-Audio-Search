"""`Diarizer` decorator (Stretch/robustness) — automatic fallback when
the primary diarizer's underlying model is blocked for *any* reason
(gated Hugging Face model, missing `HF_TOKEN`, no network, corporate
TLS-inspecting proxy, out-of-memory, etc.). This makes the "model
blocked" scenario something the system self-heals from at the port
boundary instead of something every caller has to special-case —
portable across environments/proxies, since it only reacts to the
typed `DiarizationFailedError` the port contract already guarantees
(see `src/domain/ports.py::Diarizer`), never to a specific error string.

"Sticky" circuit breaker: once the primary has failed once, this
adapter stops retrying it for the lifetime of the instance (composition
root builds one instance per process) — avoids paying a slow
network-timeout retry cost on every single ingestion request once we
already know the model is unavailable in this environment.
"""

from __future__ import annotations

import logging

from src.domain.exceptions import DiarizationFailedError
from src.domain.models import SpeakerTurn
from src.domain.ports import Diarizer
from src.infra.logging_config import log_event

logger = logging.getLogger(__name__)


class ResilientDiarizer:
    def __init__(self, primary: Diarizer, fallback: Diarizer) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_available = True

    async def diarize(self, audio_file_path: str) -> list[SpeakerTurn]:
        if self._primary_available:
            try:
                return await self._primary.diarize(audio_file_path)
            except DiarizationFailedError as exc:
                self._primary_available = False
                log_event(
                    logger, logging.WARNING, "resilient_diarizer.primary_unavailable",
                    primary=type(self._primary).__name__, fallback=type(self._fallback).__name__,
                    reason=exc.reason,
                )
        return await self._fallback.diarize(audio_file_path)
