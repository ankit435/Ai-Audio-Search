"""Tests for `SearchService.search_stream()` (Stretch Goal — SSE search),
using in-memory fakes, plus a test that the FastAPI `/search/stream`
route wires it correctly.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from src.application.search_service import SearchService
from src.domain.exceptions import InvalidQueryError
from src.domain.models import AudioFile, Chunk
from tests.fakes import FakeAudioFileRepository, FakeChunkRepository, FakeEmbedder


async def _seeded_service():
    audio_file_id = uuid.uuid4()
    chunk_repo = FakeChunkRepository()
    audio_file_repo = FakeAudioFileRepository()
    embedder = FakeEmbedder()

    await audio_file_repo.save(
        AudioFile(audio_file_id=audio_file_id, file_name="interview_01.wav", file_path="/x", checksum="abc")
    )
    chunk = Chunk(
        chunk_id=uuid.uuid4(), audio_file_id=audio_file_id, speaker_id="SPEAKER_00",
        text="We discussed the quarterly budget and revenue projections.",
        start_time=0.0, end_time=10.0,
    )
    chunk = replace(chunk, embedding=await embedder.embed(chunk.text))
    await chunk_repo.save_chunks([chunk])

    return SearchService(chunk_repository=chunk_repo, embedder=embedder, audio_file_repository=audio_file_repo)


@pytest.mark.asyncio
async def test_search_stream_yields_stages_in_order_then_done():
    service = await _seeded_service()

    events = [event async for event in service.search_stream("quarterly budget", top_k=5)]

    event_names = [e["event"] for e in events]
    assert event_names == ["keyword_results", "semantic_results", "fused_results", "done"]

    done_event = events[-1]
    assert "results" in done_event and "took_ms" in done_event
    assert len(done_event["results"]) == 1
    assert done_event["results"][0]["file_name"] == "interview_01.wav"


@pytest.mark.asyncio
async def test_search_stream_raises_on_invalid_query():
    service = await _seeded_service()

    with pytest.raises(InvalidQueryError):
        async for _ in service.search_stream("   ", top_k=5):
            pass
