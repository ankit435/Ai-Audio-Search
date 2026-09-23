"""Integration test — runs the REAL Postgres/pgvector infra adapters
(`PostgresAudioFileRepository`, `PostgresChunkRepository`) against a live
database, plus the real `IngestPipeline`/`SearchService` orchestration
(with `FakeEmbedder`/`FakeTranscriber`/`FakeDiarizer` standing in only for
the ML models, which aren't downloaded in this environment).

Requires a reachable Postgres with pgvector + db/schema.sql loaded (see
`AUDIO_SEARCH_DATABASE_URL` env var). Skipped automatically if the DB
isn't reachable, so `pytest -q` (unit suite) is unaffected.

Each test uses a unique audio-file payload (uuid4-tagged) so checksums
never collide across runs, and explicitly deletes the rows it created by
`audio_file_id` in a `finally` block — no reliance on fragile string
matching for cleanup.
"""

from __future__ import annotations

import os
import uuid

import asyncpg
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from src.application.ingest_pipeline import IngestPipeline
from src.application.search_service import SearchService
from src.domain.models import SpeakerTurn
from src.infra.postgres_repositories import PostgresAudioFileRepository, PostgresChunkRepository
from tests.fakes import FakeDiarizer, FakeEmbedder, FakeTranscriber

DATABASE_URL = os.environ.get(
    "AUDIO_SEARCH_DATABASE_URL", "postgresql://audio_search:audio_search@localhost:5432/audio_search"
)


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


@pytest_asyncio.fixture
async def pool():
    try:
        p = await asyncpg.create_pool(DATABASE_URL, init=_init_connection, min_size=1, max_size=2)
        async with p.acquire() as conn:
            await conn.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - environment-dependent skip
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}: {exc}")
    yield p
    await p.close()


async def _delete_audio_file(pool: asyncpg.Pool, audio_file_id: uuid.UUID) -> None:
    # ON DELETE CASCADE on chunk.audio_file_id handles chunk rows too.
    await pool.execute("DELETE FROM audio_file WHERE audio_file_id = $1", audio_file_id)


def _unique_audio_bytes() -> bytes:
    return f"integration-test-payload-{uuid.uuid4()}".encode()


@pytest.mark.asyncio
async def test_ingest_and_search_against_real_postgres_pgvector(pool, tmp_path):
    chunk_repo = PostgresChunkRepository(pool)
    audio_file_repo = PostgresAudioFileRepository(pool)
    embedder = FakeEmbedder(dim=384)  # must match db/schema.sql's vector(384) column

    audio_path = tmp_path / "itest_sample.wav"
    audio_path.write_bytes(_unique_audio_bytes())

    transcript = [
        SpeakerTurn(speaker_id="UNKNOWN", text="We reviewed the quarterly budget in detail.", start_time=0.0, end_time=4.0),
        SpeakerTurn(speaker_id="UNKNOWN", text="Then we talked about hiring new engineers.", start_time=4.0, end_time=8.0),
    ]
    diarization = [
        SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=4.0),
        SpeakerTurn(speaker_id="SPEAKER_01", text="", start_time=4.0, end_time=8.0),
    ]

    pipeline = IngestPipeline(
        transcriber=FakeTranscriber(transcript), diarizer=FakeDiarizer(diarization), embedder=embedder,
        chunk_repository=chunk_repo, audio_file_repository=audio_file_repo,
    )

    result = await pipeline.run(str(audio_path))
    try:
        assert result.chunk_count == 2
        assert result.skipped_existing is False

        search_service = SearchService(
            chunk_repository=chunk_repo, embedder=embedder, audio_file_repository=audio_file_repo
        )

        results = await search_service.search("quarterly budget", top_k=5)
        assert len(results) >= 1
        assert "budget" in results[0].text_snippet.lower()
        assert results[0].file_name == os.path.basename(str(audio_path))
        assert results[0].speaker_id == "SPEAKER_00"

        hiring_results = await search_service.search("hiring engineers", top_k=5)
        assert any("engineers" in r.text_snippet.lower() for r in hiring_results)
    finally:
        await _delete_audio_file(pool, result.audio_file_id)


@pytest.mark.asyncio
async def test_ingest_pipeline_idempotent_against_real_postgres(pool, tmp_path):
    audio_path = tmp_path / "itest_idempotent.wav"
    audio_path.write_bytes(_unique_audio_bytes())

    transcript = [SpeakerTurn(speaker_id="UNKNOWN", text="Single segment.", start_time=0.0, end_time=2.0)]
    diarization = [SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=2.0)]

    chunk_repo = PostgresChunkRepository(pool)
    audio_file_repo = PostgresAudioFileRepository(pool)
    pipeline = IngestPipeline(
        transcriber=FakeTranscriber(transcript), diarizer=FakeDiarizer(diarization), embedder=FakeEmbedder(dim=384),
        chunk_repository=chunk_repo, audio_file_repository=audio_file_repo,
    )

    first = await pipeline.run(str(audio_path))
    try:
        second = await pipeline.run(str(audio_path))
        assert second.skipped_existing is True
        assert second.audio_file_id == first.audio_file_id
    finally:
        await _delete_audio_file(pool, first.audio_file_id)
