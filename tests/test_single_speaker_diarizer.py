"""Tests for `SingleSpeakerDiarizer` — the production-safe `Diarizer`
fallback used when real diarization models are unavailable. Uses a real
(tiny, synthesized) WAV file rather than a fake, since this adapter's
whole job is to read real audio duration.
"""

from __future__ import annotations

import wave

import pytest

from src.infra.single_speaker_diarizer import FALLBACK_DURATION_SECONDS, SingleSpeakerDiarizer


def _write_silent_wav(path: str, duration_seconds: float, sample_rate: int = 16000) -> None:
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.asyncio
async def test_single_speaker_diarizer_reads_real_wav_duration(tmp_path):
    wav_path = tmp_path / "silent.wav"
    _write_silent_wav(str(wav_path), duration_seconds=3.0)

    turns = await SingleSpeakerDiarizer().diarize(str(wav_path))

    assert len(turns) == 1
    assert turns[0].speaker_id == "SPEAKER_UNKNOWN"
    assert turns[0].start_time == 0.0
    assert turns[0].end_time == pytest.approx(3.0, abs=0.01)


@pytest.mark.asyncio
async def test_single_speaker_diarizer_falls_back_to_sentinel_duration_on_bad_file(tmp_path):
    not_a_wav = tmp_path / "not_audio.wav"
    not_a_wav.write_bytes(b"this is not a real wav file")

    turns = await SingleSpeakerDiarizer().diarize(str(not_a_wav))

    assert len(turns) == 1
    assert turns[0].end_time == FALLBACK_DURATION_SECONDS
