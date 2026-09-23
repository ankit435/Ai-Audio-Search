"""Production-safe `Diarizer` fallback — NOT a ground-truth substitute
like `ScriptedDiarizer` (which only exists because the golden dataset's
answer is already known). This adapter works on **any** real audio file
with no external model/network/credentials at all: it treats the whole
recording as a single speaker turn.

Used as the automatic fallback behind `ResilientDiarizer` when the real
diarization model (`pyannote.audio`) is unavailable for *any* reason —
gated-model access denied, missing `HF_TOKEN`, no network, blocked
proxy, out-of-memory, etc. Speaker attribution quality degrades to "one
speaker for the whole file" in that case, but the rest of the pipeline
(transcription, chunking, embedding, search) keeps working instead of
the request failing outright.
"""

from __future__ import annotations

import contextlib
import wave

from src.domain.models import SpeakerTurn

FALLBACK_SPEAKER_ID = "SPEAKER_UNKNOWN"
# Used when duration can't be determined (e.g. non-WAV input); large enough
# that alignment-by-overlap always finds this turn for any real segment.
FALLBACK_DURATION_SECONDS = 24 * 60 * 60.0


def _probe_wav_duration(audio_file_path: str) -> float | None:
    with contextlib.suppress(Exception):
        with wave.open(audio_file_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate > 0:
                return frames / float(rate)
    return None


class SingleSpeakerDiarizer:
    async def diarize(self, audio_file_path: str) -> list[SpeakerTurn]:
        duration = _probe_wav_duration(audio_file_path) or FALLBACK_DURATION_SECONDS
        return [SpeakerTurn(speaker_id=FALLBACK_SPEAKER_ID, text="", start_time=0.0, end_time=duration)]
