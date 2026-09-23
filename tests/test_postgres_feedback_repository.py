"""Integration test — REAL Postgres-backed `PostgresFeedbackRepository`
against the `search_feedback` table + a real chunk row's `tsv` column
(needed for the keyword/semantic attribution heuristic). Skipped
automatically if Postgres isn't reachable.
"""

from __future__ import annotations

import os
import uuid

import asyncpg
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from src.infra.postgres_feedback_repository import PostgresFeedbackRepository

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


async def _make_audio_file_and_chunk(pool: asyncpg.Pool, text: str) -> tuple[uuid.UUID, uuid.UUID]:
    audio_file_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO audio_file (audio_file_id, file_name, file_path, checksum) VALUES ($1, $2, $3, $4)",
        audio_file_id, "feedback_itest.wav", f"/tmp/feedback_itest_{uuid.uuid4()}.wav", str(uuid.uuid4()),
    )
    await pool.execute(
        """
        INSERT INTO chunk (chunk_id, audio_file_id, speaker_id, text, start_time, end_time)
        VALUES ($1, $2, 'SPEAKER_00', $3, 0.0, 5.0)
        """,
        chunk_id, audio_file_id, text,
    )
    return audio_file_id, chunk_id


@pytest.mark.asyncio
async def test_record_and_recall_feedback_row(pool):
    repo = PostgresFeedbackRepository(pool)
    audio_file_id, chunk_id = await _make_audio_file_and_chunk(pool, "the onboarding schedule for new hires")
    try:
        await repo.record("onboarding schedule", chunk_id, rank_shown=1, signal="click")
        row = await pool.fetchrow(
            "SELECT query_text, rank_shown, signal FROM search_feedback WHERE chunk_id = $1", chunk_id
        )
        assert row["query_text"] == "onboarding schedule"
        assert row["rank_shown"] == 1
        assert row["signal"] == "click"
    finally:
        await pool.execute("DELETE FROM audio_file WHERE audio_file_id = $1", audio_file_id)


@pytest.mark.asyncio
async def test_positive_signal_on_keyword_matching_chunk_raises_keyword_weight(pool):
    repo = PostgresFeedbackRepository(pool)
    # Chunk text lexically overlaps the query -> heuristic should attribute to "keyword".
    audio_file_id, chunk_id = await _make_audio_file_and_chunk(pool, "quarterly revenue projections update")
    try:
        for _ in range(3):
            await repo.record("quarterly revenue projections", chunk_id, rank_shown=1, signal="click")
        weights = await repo.get_branch_weights("default")
        assert weights["keyword"] > 1.0
        assert 0.5 <= weights["semantic"] <= 2.0
    finally:
        await pool.execute("DELETE FROM audio_file WHERE audio_file_id = $1", audio_file_id)


@pytest.mark.asyncio
async def test_positive_signal_on_non_matching_chunk_raises_semantic_weight(pool):
    repo = PostgresFeedbackRepository(pool)
    # Chunk text has zero lexical overlap with the query -> heuristic should attribute to "semantic".
    audio_file_id, chunk_id = await _make_audio_file_and_chunk(pool, "the weather today is sunny and mild")
    try:
        for _ in range(3):
            await repo.record("quarterly revenue projections", chunk_id, rank_shown=1, signal="click")
        weights = await repo.get_branch_weights("default")
        assert weights["semantic"] > 1.0
    finally:
        await pool.execute("DELETE FROM audio_file WHERE audio_file_id = $1", audio_file_id)
