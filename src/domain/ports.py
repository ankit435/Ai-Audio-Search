"""Domain ports (Protocols) — Hexagonal Architecture boundary.

Application code depends only on these interfaces, never on concrete
infra adapters (Dependency Inversion). Each port is implemented by
exactly one infra adapter today, but swapping implementations (e.g.
Whisper -> hosted API, pgvector -> another vector store) requires zero
changes here or in `application/`.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from src.domain.models import AudioFile, Chunk, RankedChunk, SpeakerTurn


class Transcriber(Protocol):
    """Produces timestamped transcript segments from an audio file."""

    async def transcribe(self, audio_file_path: str) -> list[SpeakerTurn]:
        """Return raw (pre-diarization-aligned) transcript segments.

        Raises `domain.exceptions.TranscriptionFailedError` on failure.
        """
        ...


class Diarizer(Protocol):
    """Identifies speaker turns (who spoke when) in an audio file."""

    async def diarize(self, audio_file_path: str) -> list[SpeakerTurn]:
        """Return speaker-labeled time ranges (text may be empty; filled
        in later by aligning with `Transcriber` output).

        Raises `domain.exceptions.DiarizationFailedError` on failure.
        """
        ...


class Embedder(Protocol):
    """Computes a dense vector embedding for a piece of text."""

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for `text`.

        Raises `domain.exceptions.EmbeddingFailedError` on failure.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch variant, used during ingestion for efficiency."""
        ...


class ChunkRepository(Protocol):
    """Persists and queries `Chunk` records (text, vector, metadata)."""

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist chunks + embeddings atomically (single transaction).

        Raises `domain.exceptions.ChunkPersistenceError` on failure.
        """
        ...

    async def search_keyword(self, query_text: str, top_k: int) -> list[RankedChunk]:
        """Return top_k chunks ranked by `ts_rank_cd` on the query."""
        ...

    async def search_semantic(self, query_embedding: list[float], top_k: int) -> list[RankedChunk]:
        """Return top_k chunks ranked by cosine similarity to the query embedding."""
        ...

    async def get_by_ids(self, chunk_ids: list[UUID]) -> list[Chunk]:
        """Fetch full chunk records (for result hydration/context stitching)."""
        ...


class AudioFileRepository(Protocol):
    """Persists and queries `AudioFile` records."""

    async def find_by_checksum(self, checksum: str) -> AudioFile | None:
        """Used by the ingestion pipeline for idempotent dedup."""
        ...

    async def save(self, audio_file: AudioFile) -> None:
        ...

    async def get(self, audio_file_id: UUID) -> AudioFile | None:
        ...


class JobQueue(Protocol):
    """Stretch goal port — Postgres-backed ingestion job queue (SKIP LOCKED)."""

    async def enqueue(self, audio_file_path: str) -> UUID:
        ...

    async def claim_next(self, worker_id: str) -> dict | None:
        ...

    async def mark_done(self, job_id: UUID) -> None:
        ...

    async def mark_failed(self, job_id: UUID, error: str) -> None:
        ...


class FeedbackRepository(Protocol):
    """Stretch goal port — records user relevance feedback signals."""

    async def record(self, query_text: str, chunk_id: UUID, rank_shown: int, signal: str) -> None:
        ...

    async def get_branch_weights(self, query_type: str) -> dict[str, float]:
        """Returns e.g. {"keyword": 1.0, "semantic": 1.2} for RRF weighting."""
        ...


class AnswerGenerator(Protocol):
    """Stretch goal port — RAG-style answer generation over search results."""

    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        ...
