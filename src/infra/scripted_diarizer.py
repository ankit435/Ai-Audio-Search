"""`Diarizer` adapter backed by ground-truth turn timing, used only for
constructing and evaluating the synthetic golden dataset.

Why this exists (and why it's not the production diarizer): real
diarization here would be `PyannoteDiarizer`
(`src/infra/pyannote_diarizer.py`), but `pyannote.audio` requires a
gated Hugging Face model and a personal `HF_TOKEN`, unavailable in this
environment (see SOLUTION.md "Known limitations"). Since the golden
dataset is synthesized (see `scripts/generate_golden_dataset.py`), the
exact speaker/time-range ground truth is already known at generation
time and written to a `<name>.turns.json` sidecar file. This adapter
simply reads that sidecar back in, implementing the same `Diarizer`
port so it can be swapped into `IngestPipeline` without any pipeline
code changes — a direct demonstration of the ports-and-adapters
extensibility the architecture is built for.

This adapter must never be wired into `src/api/main.py`'s production
composition root; it only makes sense for the specific audio files it
was generated alongside.
"""

from __future__ import annotations

import json
import os

from src.domain.exceptions import DiarizationFailedError
from src.domain.models import SpeakerTurn


class ScriptedDiarizer:
    """Reads `<audio_file_path with .wav replaced by .turns.json>` and
    returns its turns as ground-truth `SpeakerTurn`s (`text` included,
    though `IngestPipeline` only uses `speaker_id`/timing from the
    diarizer — text comes from the real transcriber and is aligned
    separately)."""

    async def diarize(self, audio_file_path: str) -> list[SpeakerTurn]:
        sidecar_path = _sidecar_path(audio_file_path)
        if not os.path.exists(sidecar_path):
            raise DiarizationFailedError(
                audio_file_id=audio_file_path,
                reason=f"no ground-truth sidecar found at {sidecar_path}",
            )
        with open(sidecar_path) as f:
            data = json.load(f)
        return [
            SpeakerTurn(
                speaker_id=t["speaker_id"],
                text=t["text"],
                start_time=t["start_time"],
                end_time=t["end_time"],
            )
            for t in data["turns"]
        ]


def _sidecar_path(audio_file_path: str) -> str:
    base, _ext = os.path.splitext(audio_file_path)
    return f"{base}.turns.json"
