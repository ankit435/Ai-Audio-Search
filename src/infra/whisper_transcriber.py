"""Whisper-based `Transcriber` adapter.

Two backends are supported behind the same `Transcriber` port:
- `openai-whisper` (default) — downloads its checkpoint from OpenAI's
  Azure blob CDN (`openaipublic.azureedge.net`) rather than the Hugging
  Face Hub. In this project's environment, Hugging Face downloads are
  blocked by a TLS-inspecting corporate proxy while the Azure CDN is
  reachable, so this backend is the one actually verified end-to-end
  here. Requires the `ffmpeg` binary on PATH for audio decoding.
- `faster-whisper` (opt-in via `backend="faster-whisper"`) — downloads
  its converted checkpoint from the Hugging Face Hub
  (`Systran/faster-whisper-*`); kept as an alternative for environments
  where HF Hub access is available (e.g. it is typically faster/lighter
  than openai-whisper when its model download succeeds).

Runs the blocking `transcribe` call in a thread to keep the adapter's
`async def` contract non-blocking (see `SentenceTransformerEmbedder` for
the same pattern).
"""

from __future__ import annotations

import asyncio

from src.domain.exceptions import TranscriptionFailedError
from src.domain.models import SpeakerTurn

DEFAULT_MODEL_SIZE = "base"
DEFAULT_BACKEND = "openai-whisper"


class WhisperTranscriber:
    """Speaker is left as a placeholder ("UNKNOWN") — diarization alignment
    happens later in `ingest_pipeline._align_transcript_to_speakers`.
    """

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, backend: str = DEFAULT_BACKEND) -> None:
        self._model_size = model_size
        self._backend = backend
        self._model = None  # lazy-loaded on first use

    def _load_model(self):
        if self._model is None:
            if self._backend == "faster-whisper":
                from faster_whisper import WhisperModel  # local import: heavy/optional dep

                self._model = WhisperModel(self._model_size)
            else:
                import whisper  # local import: heavy/optional dep

                self._model = whisper.load_model(self._model_size)
        return self._model

    def _transcribe_sync(self, audio_file_path: str) -> list[SpeakerTurn]:
        model = self._load_model()
        if self._backend == "faster-whisper":
            segments, _info = model.transcribe(audio_file_path)
            return [
                SpeakerTurn(speaker_id="UNKNOWN", text=seg.text.strip(), start_time=seg.start, end_time=seg.end)
                for seg in segments
            ]
        result = model.transcribe(audio_file_path)
        return [
            SpeakerTurn(
                speaker_id="UNKNOWN",
                text=seg["text"].strip(),
                start_time=seg["start"],
                end_time=seg["end"],
            )
            for seg in result["segments"]
        ]

    async def transcribe(self, audio_file_path: str) -> list[SpeakerTurn]:
        try:
            return await asyncio.to_thread(self._transcribe_sync, audio_file_path)
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception
            raise TranscriptionFailedError(audio_file_id=audio_file_path, reason=str(exc)) from exc
