"""Integration test for async job queue API endpoints (`POST /ingest/async` & `GET /ingest/jobs/{job_id}`).
"""

from __future__ import annotations

import os
import uuid
import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pgvector.asyncpg import register_vector

from src.api.main import app

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
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}: {exc}")
    
    app.state.pool = p
    yield p
    await p.close()


@pytest.mark.asyncio
async def test_async_ingest_api_enqueue_and_check_status(pool, tmp_path):
    audio_path = tmp_path / f"async_api_test_{uuid.uuid4()}.wav"
    audio_path.write_bytes(b"dummy audio content")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Enqueue via POST /ingest/async
        response = await client.post("/ingest/async", json={"audio_file_path": str(audio_path)})
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        job_id = data["job_id"]

        # Check status via GET /ingest/jobs/{job_id}
        status_resp = await client.get(f"/ingest/jobs/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == job_id
        assert status_data["audio_file_path"] == str(audio_path)
        assert status_data["status"] in ("pending", "processing", "done")

    # Cleanup DB row
    await pool.execute("DELETE FROM ingestion_job WHERE job_id = $1", uuid.UUID(job_id))
