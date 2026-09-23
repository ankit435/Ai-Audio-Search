"""Task 8 (pytest wrapper) — asserts speaker attribution accuracy and
search latency thresholds against the real golden dataset in Postgres.

The heavy lifting lives in `scripts/measure_speaker_accuracy_and_latency.py`
(kept runnable standalone for humans); this test just re-uses those
functions so the thresholds are enforced by `pytest -q` like every other
success criterion in the plan.

Skips (does not fail) if the golden dataset has not been ingested yet —
see `tests/test_recall_at_k_golden.py` for the same convention.
"""

from __future__ import annotations

import logging
import os

import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

import asyncpg

from scripts.measure_speaker_accuracy_and_latency import measure_latency, measure_speaker_accuracy
from scripts.golden_query_set import QUERY_SET
from src.application.search_service import SearchService
from src.infra.postgres_repositories import PostgresAudioFileRepository, PostgresChunkRepository
from src.infra.sentence_transformer_embedder import SentenceTransformerEmbedder

DATABASE_URL = os.environ.get(
    "AUDIO_SEARCH_DATABASE_URL", "postgresql://audio_search:audio_search@localhost:5432/audio_search"
)
SPEAKER_ACCURACY_THRESHOLD = 0.90
LATENCY_P95_THRESHOLD_MS = 500.0


@pytest_asyncio.fixture
async def pool():
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    p = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4, init=_init_connection)
    row_count = await p.fetchval("SELECT count(*) FROM audio_file")
    if row_count < 5:
        await p.close()
        pytest.skip("golden dataset not ingested — run scripts/ingest_golden_dataset.py first")
    yield p
    await p.close()


@pytest.mark.asyncio
async def test_speaker_attribution_accuracy(pool):
    correct, total = await measure_speaker_accuracy(pool)
    accuracy = correct / total if total else 0.0
    print(f"\nspeaker attribution accuracy: {correct}/{total} = {accuracy:.3f}")
    assert accuracy >= SPEAKER_ACCURACY_THRESHOLD


@pytest.mark.asyncio
async def test_search_latency_p95(pool):
    logging.getLogger("src.application.search_service").setLevel(logging.INFO)
    search_service = SearchService(
        chunk_repository=PostgresChunkRepository(pool),
        embedder=SentenceTransformerEmbedder(),
        audio_file_repository=PostgresAudioFileRepository(pool),
    )
    await search_service.search(QUERY_SET[0]["query"], top_k=10)  # warm-up, excluded

    durations = sorted(await measure_latency(search_service))
    p95_idx = min(len(durations) - 1, int(round(0.95 * (len(durations) - 1))))
    p95 = durations[p95_idx]
    print(f"\nsearch latency p95={p95:.1f}ms (n={len(durations)})")
    assert p95 < LATENCY_P95_THRESHOLD_MS
