"""Integration test — REAL Postgres-backed `PostgresJobQueue` +
`IngestionWorker` (Stretch Goal). Uses `FakeTranscriber`/`FakeDiarizer`/
`FakeEmbedder` for the ML models (not available offline in this
environment) but the queue itself, its `SKIP LOCKED` claim semantics,
and the worker's polling/persistence logic are all real.

Skipped automatically if Postgres isn't reachable (same convention as
`tests/test_postgres_integration.py`).
"""

from __future__ import annotations

import os
import uuid

import asyncpg
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from src.application.ingest_pipeline import IngestPipeline
from src.application.ingestion_worker import IngestionWorker
from src.domain.models import SpeakerTurn
from src.infra.postgres_job_queue import PostgresJobQueue
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


async def _cleanup_job(pool: asyncpg.Pool, job_id: uuid.UUID) -> None:
    await pool.execute("DELETE FROM ingestion_job WHERE job_id = $1", job_id)


async def _cleanup_audio_file_by_path(pool: asyncpg.Pool, file_path: str) -> None:
    await pool.execute("DELETE FROM audio_file WHERE file_path = $1", file_path)


@pytest.mark.asyncio
async def test_worker_processes_enqueued_job_end_to_end(pool, tmp_path):
    audio_path = tmp_path / f"job_queue_itest_{uuid.uuid4()}.wav"
    audio_path.write_bytes(f"job-queue-test-{uuid.uuid4()}".encode())

    job_queue = PostgresJobQueue(pool)
    pipeline = IngestPipeline(
        transcriber=FakeTranscriber(
            segments=[SpeakerTurn(speaker_id="UNKNOWN", text="Async ingestion worked.", start_time=0.0, end_time=3.0)]
        ),
        diarizer=FakeDiarizer(turns=[SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=3.0)]),
        embedder=FakeEmbedder(dim=384),
        chunk_repository=PostgresChunkRepository(pool),
        audio_file_repository=PostgresAudioFileRepository(pool),
    )
    worker = IngestionWorker(job_queue=job_queue, ingest_pipeline=pipeline, worker_id="test-worker")

    job_id = await job_queue.enqueue(str(audio_path))
    try:
        row = await pool.fetchrow("SELECT status FROM ingestion_job WHERE job_id = $1", job_id)
        assert row["status"] == "pending"

        processed = await worker.run_once()
        assert processed is True

        row = await pool.fetchrow(
            "SELECT status, attempts, locked_by FROM ingestion_job WHERE job_id = $1", job_id
        )
        assert row["status"] == "done"
        assert row["attempts"] == 1
        assert row["locked_by"] == "test-worker"

        audio_row = await pool.fetchrow("SELECT audio_file_id FROM audio_file WHERE file_path = $1", str(audio_path))
        assert audio_row is not None
        chunk_count = await pool.fetchval(
            "SELECT count(*) FROM chunk WHERE audio_file_id = $1", audio_row["audio_file_id"]
        )
        assert chunk_count >= 1

        # Queue is now empty — a second run_once() should be a no-op.
        assert await worker.run_once() is False
    finally:
        await _cleanup_audio_file_by_path(pool, str(audio_path))  # cascades to chunk
        await _cleanup_job(pool, job_id)


@pytest.mark.asyncio
async def test_claim_next_skips_locked_rows_for_concurrent_workers(pool, tmp_path):
    """Two jobs enqueued; claiming one must not block/duplicate the other —
    this is the entire point of `FOR UPDATE SKIP LOCKED`."""
    path_a = str(tmp_path / f"a_{uuid.uuid4()}.wav")
    path_b = str(tmp_path / f"b_{uuid.uuid4()}.wav")
    job_queue = PostgresJobQueue(pool)
    job_a = await job_queue.enqueue(path_a)
    job_b = await job_queue.enqueue(path_b)
    try:
        claimed_a = await job_queue.claim_next("worker-a")
        claimed_b = await job_queue.claim_next("worker-b")
        assert claimed_a is not None and claimed_b is not None
        assert claimed_a["job_id"] != claimed_b["job_id"]
        assert {claimed_a["job_id"], claimed_b["job_id"]} == {job_a, job_b}

        # Both now 'processing' — a third claim finds nothing left.
        assert await job_queue.claim_next("worker-c") is None
    finally:
        await _cleanup_job(pool, job_a)
        await _cleanup_job(pool, job_b)


@pytest.mark.asyncio
async def test_worker_marks_job_failed_on_domain_error(pool, tmp_path):
    """A transcriber that raises should result in status='failed' with
    `last_error` populated, not a silently swallowed exception."""
    from src.domain.exceptions import TranscriptionFailedError

    class _FailingTranscriber:
        async def transcribe(self, audio_file_path: str):
            raise TranscriptionFailedError(audio_file_id="n/a", reason="simulated failure")

    audio_path_obj = tmp_path / f"failing_{uuid.uuid4()}.wav"
    audio_path_obj.write_bytes(f"failing-job-test-{uuid.uuid4()}".encode())
    audio_path = str(audio_path_obj)
    job_queue = PostgresJobQueue(pool)
    pipeline = IngestPipeline(
        transcriber=_FailingTranscriber(),
        diarizer=FakeDiarizer(turns=[]),
        embedder=FakeEmbedder(dim=384),
        chunk_repository=PostgresChunkRepository(pool),
        audio_file_repository=PostgresAudioFileRepository(pool),
    )
    worker = IngestionWorker(job_queue=job_queue, ingest_pipeline=pipeline, worker_id="test-worker")

    job_id = await job_queue.enqueue(audio_path)
    try:
        assert await worker.run_once() is True
        row = await pool.fetchrow("SELECT status, last_error FROM ingestion_job WHERE job_id = $1", job_id)
        assert row["status"] == "failed"
        assert "simulated failure" in row["last_error"]
    finally:
        await _cleanup_job(pool, job_id)
