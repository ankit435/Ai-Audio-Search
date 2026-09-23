"""Tests for the extractive `AnswerGenerator` adapter and
`SearchAnswerService` (Stretch Goal), using in-memory fakes for search.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from src.application.search_answer_service import SearchAnswerService
from src.application.search_service import SearchService
from src.domain.models import AudioFile, Chunk
from src.infra.extractive_answer_generator import ExtractiveAnswerGenerator
from tests.fakes import FakeAudioFileRepository, FakeChunkRepository, FakeEmbedder


async def _seeded_search_answer_service():
    audio_file_id = uuid.uuid4()
    chunk_repo = FakeChunkRepository()
    audio_file_repo = FakeAudioFileRepository()
    embedder = FakeEmbedder()

    await audio_file_repo.save(
        AudioFile(audio_file_id=audio_file_id, file_name="onboarding_session.wav", file_path="/x", checksum="abc")
    )
    chunks = [
        Chunk(
            chunk_id=uuid.uuid4(), audio_file_id=audio_file_id, speaker_id="SPEAKER_00",
            text="New hires should complete the benefits enrollment within the first week.",
            start_time=12.0, end_time=18.0,
        ),
        Chunk(
            chunk_id=uuid.uuid4(), audio_file_id=audio_file_id, speaker_id="SPEAKER_01",
            text="The weather today is sunny with a light breeze.",
            start_time=20.0, end_time=25.0,
        ),
    ]
    for chunk in chunks:
        chunk = replace(chunk, embedding=await embedder.embed(chunk.text))
        await chunk_repo.save_chunks([chunk])

    search_service = SearchService(
        chunk_repository=chunk_repo, embedder=embedder, audio_file_repository=audio_file_repo
    )
    return SearchAnswerService(search_service=search_service, answer_generator=ExtractiveAnswerGenerator())


@pytest.mark.asyncio
async def test_answer_cites_relevant_chunk():
    service = await _seeded_search_answer_service()

    result = await service.answer("when should new hires enroll in benefits", top_k=5)

    assert "benefits enrollment" in result.answer
    assert "SPEAKER_00" in result.answer
    assert len(result.citations) >= 1
    assert result.citations[0].file_name == "onboarding_session.wav"


@pytest.mark.asyncio
async def test_extractive_generator_handles_no_results():
    generator = ExtractiveAnswerGenerator()
    answer = await generator.generate("anything", [])
    assert "No relevant results" in answer
