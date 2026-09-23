"""Application service tests using in-memory fake port implementations
(no Postgres, no real ML models) — see `tests/fakes.py`.
"""

from __future__ import annotations

import uuid

import pytest

from src.application.ingest_pipeline import IngestPipeline
from src.domain.models import SpeakerTurn
from tests.fakes import FakeAudioFileRepository, FakeChunkRepository, FakeDiarizer, FakeEmbedder, FakeTranscriber


@pytest.fixture
def tmp_audio_file(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"fake-audio-bytes-for-checksum-purposes")
    return str(path)


def _build_pipeline(transcript_segments, diarization_turns):
    return IngestPipeline(
        transcriber=FakeTranscriber(transcript_segments),
        diarizer=FakeDiarizer(diarization_turns),
        embedder=FakeEmbedder(),
        chunk_repository=FakeChunkRepository(),
        audio_file_repository=FakeAudioFileRepository(),
    )


@pytest.mark.asyncio
async def test_ingest_pipeline_produces_chunks_and_embeddings(tmp_audio_file):
    transcript = [
        SpeakerTurn(speaker_id="UNKNOWN", text="Hello, welcome to the show.", start_time=0.0, end_time=3.0),
        SpeakerTurn(speaker_id="UNKNOWN", text="Thanks for having me.", start_time=3.0, end_time=6.0),
    ]
    diarization = [
        SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=3.0),
        SpeakerTurn(speaker_id="SPEAKER_01", text="", start_time=3.0, end_time=6.0),
    ]
    pipeline = _build_pipeline(transcript, diarization)

    result = await pipeline.run(tmp_audio_file)

    assert result.chunk_count == 2
    assert result.skipped_existing is False
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_ingest_pipeline_aligns_speaker_by_time_overlap(tmp_audio_file):
    transcript = [SpeakerTurn(speaker_id="UNKNOWN", text="Only one segment.", start_time=0.0, end_time=3.0)]
    diarization = [
        SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=1.0),
        SpeakerTurn(speaker_id="SPEAKER_01", text="", start_time=1.0, end_time=3.0),
    ]
    audio_file_repo = FakeAudioFileRepository()
    chunk_repo = FakeChunkRepository()
    pipeline = IngestPipeline(
        transcriber=FakeTranscriber(transcript),
        diarizer=FakeDiarizer(diarization),
        embedder=FakeEmbedder(),
        chunk_repository=chunk_repo,
        audio_file_repository=audio_file_repo,
    )

    await pipeline.run(tmp_audio_file)

    saved_chunks = list(chunk_repo._chunks.values())
    assert len(saved_chunks) == 1
    # SPEAKER_01 overlaps [1.0, 3.0] = 2s vs SPEAKER_00 overlaps [0.0, 1.0] = 1s -> SPEAKER_01 wins.
    assert saved_chunks[0].speaker_id == "SPEAKER_01"


@pytest.mark.asyncio
async def test_ingest_pipeline_is_idempotent_on_same_file(tmp_audio_file):
    transcript = [SpeakerTurn(speaker_id="UNKNOWN", text="Hi.", start_time=0.0, end_time=1.0)]
    diarization = [SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=1.0)]
    pipeline = _build_pipeline(transcript, diarization)

    first = await pipeline.run(tmp_audio_file)
    second = await pipeline.run(tmp_audio_file)

    assert second.skipped_existing is True
    assert second.audio_file_id == first.audio_file_id
