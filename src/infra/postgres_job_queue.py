"""Postgres-backed `JobQueue` adapter (Stretch Goal — async ingestion).

Uses `SELECT ... FOR UPDATE SKIP LOCKED` so multiple `IngestionWorker`
processes can poll the same `ingestion_job` table concurrently without
double-processing a row (the standard Postgres job-queue pattern).
"""

from __future__ import annotations

from uuid import UUID

import asyncpg


class PostgresJobQueue:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def enqueue(self, audio_file_path: str) -> UUID:
        row = await self._pool.fetchrow(
            "INSERT INTO ingestion_job (audio_file_path) VALUES ($1) RETURNING job_id",
            audio_file_path,
        )
        return row["job_id"]

    async def claim_next(self, worker_id: str) -> dict | None:
        """Atomically claim the oldest pending (or stale-locked) job.

        `FOR UPDATE SKIP LOCKED` means concurrent workers each get a
        distinct row instead of blocking on/duplicating one another.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT job_id, audio_file_path, attempts
                FROM ingestion_job
                WHERE status = 'pending'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            )
            if row is None:
                return None
            await conn.execute(
                """
                UPDATE ingestion_job
                SET status = 'processing', locked_by = $2, locked_at = now(),
                    attempts = attempts + 1, updated_at = now()
                WHERE job_id = $1
                """,
                row["job_id"], worker_id,
            )
            return {"job_id": row["job_id"], "audio_file_path": row["audio_file_path"], "attempts": row["attempts"] + 1}

    async def mark_done(self, job_id: UUID) -> None:
        await self._pool.execute(
            "UPDATE ingestion_job SET status = 'done', updated_at = now() WHERE job_id = $1",
            job_id,
        )

    async def mark_failed(self, job_id: UUID, error: str) -> None:
        await self._pool.execute(
            "UPDATE ingestion_job SET status = 'failed', last_error = $2, updated_at = now() WHERE job_id = $1",
            job_id, error,
        )
