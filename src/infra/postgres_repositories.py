"""Postgres + pgvector adapter for `AudioFileRepository` and `ChunkRepository`.

Only place in the codebase allowed to contain raw SQL. Uses asyncpg with
parameterized queries throughout (no string-interpolated SQL — OWASP
basics per plan "Core Engineering Principles").
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from src.domain.exceptions import ChunkPersistenceError
from src.domain.models import AudioFile, Chunk, RankedChunk


class PostgresAudioFileRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def find_by_checksum(self, checksum: str) -> AudioFile | None:
        row = await self._pool.fetchrow(
            "SELECT audio_file_id, file_name, file_path, checksum, duration_s, created_at "
            "FROM audio_file WHERE checksum = $1",
            checksum,
        )
        return _row_to_audio_file(row) if row else None

    async def save(self, audio_file: AudioFile) -> None:
        await self._pool.execute(
            "INSERT INTO audio_file (audio_file_id, file_name, file_path, checksum, duration_s) "
            "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (checksum) DO NOTHING",
            audio_file.audio_file_id, audio_file.file_name, audio_file.file_path,
            audio_file.checksum, audio_file.duration_s,
        )

    async def get(self, audio_file_id: UUID) -> AudioFile | None:
        row = await self._pool.fetchrow(
            "SELECT audio_file_id, file_name, file_path, checksum, duration_s, created_at "
            "FROM audio_file WHERE audio_file_id = $1",
            audio_file_id,
        )
        return _row_to_audio_file(row) if row else None


class PostgresChunkRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO chunk (
                        chunk_id, audio_file_id, speaker_id, text, start_time, end_time,
                        embedding, prev_chunk_id, next_chunk_id, token_count, char_count
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    [
                        (
                            c.chunk_id, c.audio_file_id, c.speaker_id, c.text, c.start_time, c.end_time,
                            c.embedding, c.prev_chunk_id, c.next_chunk_id, c.token_count, c.char_count,
                        )
                        for c in chunks
                    ],
                )
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception
            raise ChunkPersistenceError(audio_file_id=str(chunks[0].audio_file_id), reason=str(exc)) from exc

    async def search_keyword(self, query_text: str, top_k: int) -> list[RankedChunk]:
        rows = await self._pool.fetch(
            """
            SELECT chunk_id, ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score
            FROM chunk
            WHERE tsv @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            query_text, top_k,
        )
        return [
            RankedChunk(chunk_id=row["chunk_id"], rank=i + 1, raw_score=row["score"])
            for i, row in enumerate(rows)
        ]

    async def search_semantic(self, query_embedding: list[float], top_k: int) -> list[RankedChunk]:
        rows = await self._pool.fetch(
            """
            SELECT chunk_id, 1 - (embedding <=> $1) AS score
            FROM chunk
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            query_embedding, top_k,
        )
        return [
            RankedChunk(chunk_id=row["chunk_id"], rank=i + 1, raw_score=row["score"])
            for i, row in enumerate(rows)
        ]

    async def get_by_ids(self, chunk_ids: list[UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        rows = await self._pool.fetch(
            """
            SELECT chunk_id, audio_file_id, speaker_id, text, start_time, end_time,
                   embedding, prev_chunk_id, next_chunk_id, token_count, char_count, created_at
            FROM chunk WHERE chunk_id = ANY($1::uuid[])
            """,
            chunk_ids,
        )
        return [_row_to_chunk(row) for row in rows]


def _row_to_audio_file(row) -> AudioFile:
    return AudioFile(
        audio_file_id=row["audio_file_id"],
        file_name=row["file_name"],
        file_path=row["file_path"],
        checksum=row["checksum"],
        duration_s=row["duration_s"],
        created_at=row["created_at"],
    )


def _row_to_chunk(row) -> Chunk:
    # asyncpg + pgvector's registered codec returns a `pgvector.Vector`
    # wrapper, not a plain iterable — `.to_list()` unwraps it into the
    # domain model's `list[float]` embedding representation.
    raw_embedding = row["embedding"]
    embedding = raw_embedding.to_list() if raw_embedding is not None else None

    return Chunk(
        chunk_id=row["chunk_id"],
        audio_file_id=row["audio_file_id"],
        speaker_id=row["speaker_id"],
        text=row["text"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        embedding=embedding,
        prev_chunk_id=row["prev_chunk_id"],
        next_chunk_id=row["next_chunk_id"],
        token_count=row["token_count"],
        char_count=row["char_count"],
        created_at=row["created_at"],
    )
