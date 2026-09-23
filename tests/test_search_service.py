"""Search service tests using in-memory fake port implementations."""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from src.application.search_service import SearchService
from src.domain.exceptions import InvalidQueryError
from src.domain.models import AudioFile, Chunk
from tests.fakes import FakeAudioFileRepository, FakeChunkRepository, FakeEmbedder


@pytest.fixture
def audio_file_id():
    return uuid.uuid4()


async def _seeded_service(audio_file_id):
    chunk_repo = FakeChunkRepository()
    audio_file_repo = FakeAudioFileRepository()
    embedder = FakeEmbedder()

    await audio_file_repo.save(
        AudioFile(audio_file_id=audio_file_id, file_name="interview_01.wav", file_path="/x", checksum="abc")
    )

    chunks = [
        Chunk(
            chunk_id=uuid.uuid4(), audio_file_id=audio_file_id, speaker_id="SPEAKER_00",
            text="We discussed the quarterly budget and revenue projections.",
            start_time=0.0, end_time=10.0,
        ),
        Chunk(
            chunk_id=uuid.uuid4(), audio_file_id=audio_file_id, speaker_id="SPEAKER_01",
            text="The weather today is sunny with a light breeze.",
            start_time=10.0, end_time=20.0,
        ),
    ]
    for chunk in chunks:
        chunk = replace(chunk, embedding=await embedder.embed(chunk.text))
        await chunk_repo.save_chunks([chunk])

    service = SearchService(chunk_repository=chunk_repo, embedder=embedder, audio_file_repository=audio_file_repo)
    return service, chunks


@pytest.mark.asyncio
async def test_search_returns_keyword_relevant_result_first(audio_file_id):
    service, chunks = await _seeded_service(audio_file_id)

    results = await service.search("quarterly budget revenue", top_k=5)

    assert len(results) >= 1
    assert results[0].text_snippet.startswith("We discussed")
    assert results[0].file_name == "interview_01.wav"
    assert results[0].speaker_id == "SPEAKER_00"


@pytest.mark.asyncio
async def test_search_rejects_empty_query(audio_file_id):
    service, _ = await _seeded_service(audio_file_id)

    with pytest.raises(InvalidQueryError):
        await service.search("   ")


@pytest.mark.asyncio
async def test_search_rejects_overlong_query(audio_file_id):
    service, _ = await _seeded_service(audio_file_id)

    with pytest.raises(InvalidQueryError):
        await service.search("x" * 2000)


@pytest.mark.asyncio
async def test_search_respects_top_k(audio_file_id):
    service, _ = await _seeded_service(audio_file_id)

    results = await service.search("weather budget", top_k=1)

    assert len(results) <= 1
