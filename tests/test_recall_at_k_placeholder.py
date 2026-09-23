"""Placeholder recall@k evaluation harness (Phase 2 "first-pass against a
placeholder query set" — see plan Implementation Phasing).

Once the real golden dataset + labeled query set (plan Evaluation Plan §2)
exist, this file should be replaced/extended to load them from a fixture
(e.g. `tests/fixtures/golden_query_set.json`) instead of the inline
synthetic corpus below. The recall@k computation and assertion pattern
already matches what that real test will look like — only the fixture
data source changes.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from src.application.search_service import SearchService
from src.domain.models import AudioFile, Chunk
from tests.fakes import FakeAudioFileRepository, FakeChunkRepository, FakeEmbedder

# Tiny synthetic corpus standing in for the real golden dataset until it's sourced.
_CORPUS = [
    ("SPEAKER_00", "The quarterly revenue grew by twelve percent this year."),
    ("SPEAKER_01", "Our biggest competitor launched a similar product last month."),
    ("SPEAKER_00", "We should hire two more engineers for the platform team."),
    ("SPEAKER_01", "The weather has been unusually warm for this time of year."),
    ("SPEAKER_00", "Customer churn dropped after we improved onboarding."),
]

# (query, expected_relevant_text_substring) — placeholder labeled query set.
_LABELED_QUERIES = [
    ("revenue growth this year", "revenue grew"),
    ("competitor product launch", "competitor launched"),
    ("hiring engineers", "hire two more engineers"),
    ("customer churn onboarding", "Customer churn"),
]


async def _build_service():
    audio_file_id = uuid.uuid4()
    chunk_repo = FakeChunkRepository()
    audio_file_repo = FakeAudioFileRepository()
    embedder = FakeEmbedder()

    await audio_file_repo.save(
        AudioFile(audio_file_id=audio_file_id, file_name="placeholder.wav", file_path="/x", checksum="corpus-1")
    )

    chunks_by_text = {}
    for i, (speaker, text) in enumerate(_CORPUS):
        chunk = Chunk(
            chunk_id=uuid.uuid4(), audio_file_id=audio_file_id, speaker_id=speaker,
            text=text, start_time=float(i * 10), end_time=float(i * 10 + 8),
        )
        chunk = replace(chunk, embedding=await embedder.embed(chunk.text))
        await chunk_repo.save_chunks([chunk])
        chunks_by_text[text] = chunk

    service = SearchService(chunk_repository=chunk_repo, embedder=embedder, audio_file_repository=audio_file_repo)
    return service, chunks_by_text


@pytest.mark.asyncio
async def test_placeholder_recall_at_k():
    """Not a substitute for the real Evaluation Plan §2 suite — asserts the
    harness itself (recall@k computation + search_service wiring) works
    end-to-end, using a synthetic corpus/query set as a stand-in.
    """
    service, chunks_by_text = await _build_service()
    top_k = 3
    hits = 0

    for query, expected_substring in _LABELED_QUERIES:
        results = await service.search(query, top_k=top_k)
        matched = any(expected_substring in r.text_snippet for r in results)
        hits += int(matched)

    recall_at_k = hits / len(_LABELED_QUERIES)
    assert recall_at_k >= 0.75, f"placeholder recall@{top_k}={recall_at_k:.2f} below sanity threshold"
