"""Tests for the resilient decorator adapters (`ResilientDiarizer`,
`ResilientTranscriber`, `ResilientAnswerGenerator`) — the "model
blocked" fallback logic. Uses tiny fake primaries/fallbacks so these
tests run anywhere, independent of which real ML model is actually
blocked in a given environment.
"""

from __future__ import annotations

import pytest

from src.domain.exceptions import DiarizationFailedError, EmbeddingFailedError, TranscriptionFailedError
from src.domain.models import Chunk, SpeakerTurn
from src.infra.resilient_answer_generator import ResilientAnswerGenerator
from src.infra.resilient_diarizer import ResilientDiarizer
from src.infra.resilient_transcriber import ResilientTranscriber


class _AlwaysFailsDiarizer:
    def __init__(self) -> None:
        self.call_count = 0

    async def diarize(self, audio_file_path: str):
        self.call_count += 1
        raise DiarizationFailedError(audio_file_id=audio_file_path, reason="model blocked (simulated)")


class _WorkingDiarizer:
    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = turns
        self.call_count = 0

    async def diarize(self, audio_file_path: str):
        self.call_count += 1
        return list(self._turns)


class _AlwaysFailsTranscriber:
    def __init__(self) -> None:
        self.call_count = 0

    async def transcribe(self, audio_file_path: str):
        self.call_count += 1
        raise TranscriptionFailedError(audio_file_id=audio_file_path, reason="model blocked (simulated)")


class _WorkingTranscriber:
    def __init__(self, segments: list[SpeakerTurn]) -> None:
        self._segments = segments
        self.call_count = 0

    async def transcribe(self, audio_file_path: str):
        self.call_count += 1
        return list(self._segments)


class _AlwaysFailsAnswerGenerator:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        self.call_count += 1
        raise EmbeddingFailedError(context="test", reason="model blocked (simulated)")


class _WorkingAnswerGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.call_count = 0

    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        self.call_count += 1
        return self._answer


@pytest.mark.asyncio
async def test_resilient_diarizer_uses_primary_when_available():
    primary = _WorkingDiarizer([SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=5.0)])
    fallback = _AlwaysFailsDiarizer()
    resilient = ResilientDiarizer(primary=primary, fallback=fallback)

    result = await resilient.diarize("audio.wav")

    assert result[0].speaker_id == "SPEAKER_00"
    assert primary.call_count == 1


@pytest.mark.asyncio
async def test_resilient_diarizer_falls_back_when_primary_blocked():
    primary = _AlwaysFailsDiarizer()
    fallback = _WorkingDiarizer([SpeakerTurn(speaker_id="SPEAKER_UNKNOWN", text="", start_time=0.0, end_time=5.0)])
    resilient = ResilientDiarizer(primary=primary, fallback=fallback)

    result = await resilient.diarize("audio.wav")

    assert result[0].speaker_id == "SPEAKER_UNKNOWN"
    assert primary.call_count == 1
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_resilient_diarizer_is_sticky_after_first_failure():
    """Once primary fails, subsequent calls must skip straight to
    fallback — no repeated slow-retry cost against a known-blocked model.
    """
    primary = _AlwaysFailsDiarizer()
    fallback = _WorkingDiarizer([SpeakerTurn(speaker_id="SPEAKER_UNKNOWN", text="", start_time=0.0, end_time=5.0)])
    resilient = ResilientDiarizer(primary=primary, fallback=fallback)

    await resilient.diarize("audio1.wav")
    await resilient.diarize("audio2.wav")
    await resilient.diarize("audio3.wav")

    assert primary.call_count == 1  # only tried once, ever
    assert fallback.call_count == 3


@pytest.mark.asyncio
async def test_resilient_transcriber_falls_back_when_primary_blocked():
    primary = _AlwaysFailsTranscriber()
    fallback = _WorkingTranscriber(
        [SpeakerTurn(speaker_id="UNKNOWN", text="fallback transcript", start_time=0.0, end_time=5.0)]
    )
    resilient = ResilientTranscriber(primary=primary, fallback=fallback)

    result = await resilient.transcribe("audio.wav")

    assert result[0].text == "fallback transcript"
    assert primary.call_count == 1
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_resilient_transcriber_uses_primary_when_available():
    primary = _WorkingTranscriber([SpeakerTurn(speaker_id="UNKNOWN", text="primary transcript", start_time=0.0, end_time=5.0)])
    fallback = _AlwaysFailsTranscriber()
    resilient = ResilientTranscriber(primary=primary, fallback=fallback)

    result = await resilient.transcribe("audio.wav")

    assert result[0].text == "primary transcript"
    assert fallback.call_count == 0


@pytest.mark.asyncio
async def test_resilient_answer_generator_falls_back_when_primary_blocked():
    primary = _AlwaysFailsAnswerGenerator()
    fallback = _WorkingAnswerGenerator("extractive answer")
    resilient = ResilientAnswerGenerator(primary=primary, fallback=fallback)

    answer = await resilient.generate("query", [])

    assert answer == "extractive answer"
    assert primary.call_count == 1
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_resilient_answer_generator_is_sticky_after_first_failure():
    primary = _AlwaysFailsAnswerGenerator()
    fallback = _WorkingAnswerGenerator("extractive answer")
    resilient = ResilientAnswerGenerator(primary=primary, fallback=fallback)

    await resilient.generate("q1", [])
    await resilient.generate("q2", [])

    assert primary.call_count == 1
    assert fallback.call_count == 2
