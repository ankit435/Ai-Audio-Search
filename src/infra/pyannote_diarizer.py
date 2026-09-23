"""pyannote.audio-based `Diarizer` adapter (local).

Requires a HuggingFace access token accepted for the pyannote model
license, read from env (`HF_TOKEN`) — never hardcoded (see plan
"Core Engineering Principles": credentials via env vars only).
"""

from __future__ import annotations

import asyncio
import os

from src.domain.exceptions import DiarizationFailedError
from src.domain.models import SpeakerTurn

DEFAULT_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizer:
    def __init__(self, pipeline_name: str = DEFAULT_PIPELINE_NAME) -> None:
        self._pipeline_name = pipeline_name
        self._pipeline = None  # lazy-loaded on first use

    def _load_pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline  # local import: heavy/optional dep

            hf_token = os.environ.get("HF_TOKEN")
            self._pipeline = Pipeline.from_pretrained(self._pipeline_name, use_auth_token=hf_token)
        return self._pipeline

    def _diarize_sync(self, audio_file_path: str) -> list[SpeakerTurn]:
        pipeline = self._load_pipeline()
        diarization = pipeline(audio_file_path)
        return [
            SpeakerTurn(speaker_id=speaker, text="", start_time=turn.start, end_time=turn.end)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

    async def diarize(self, audio_file_path: str) -> list[SpeakerTurn]:
        try:
            return await asyncio.to_thread(self._diarize_sync, audio_file_path)
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception
            raise DiarizationFailedError(audio_file_id=audio_file_path, reason=str(exc)) from exc
