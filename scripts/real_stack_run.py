"""Full real-stack, end-to-end run of the ingest + search pipeline.

Uses REAL infra adapters everywhere it is possible to in this
environment:
  - `WhisperTranscriber` (openai-whisper backend, real speech-to-text)
  - `SentenceTransformerEmbedder` (real all-MiniLM-L6-v2 embeddings)
  - `PostgresAudioFileRepository` / `PostgresChunkRepository` (real
    asyncpg + pgvector against a live Postgres database)

The only stand-in is the diarizer: `pyannote.audio` requires a gated
Hugging Face model + a personal `HF_TOKEN`, which is out of reach in
this environment (documented in SOLUTION.md / PROGRESS.md). A minimal
single-speaker `FakeDiarizer` is used instead so the full pipeline
(transcribe -> diarize -> align -> chunk -> embed -> index -> search)
can be exercised end-to-end against real infrastructure.

Cleans up the rows it inserts before exiting.
"""

from __future__ import annotations

import asyncio
import os
import sys

import asyncpg
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.application.ingest_pipeline import IngestPipeline
from src.application.search_service import SearchService
from src.domain.models import SpeakerTurn
from src.infra.postgres_repositories import PostgresAudioFileRepository, PostgresChunkRepository
from src.infra.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infra.whisper_transcriber import WhisperTranscriber
from tests.fakes import FakeDiarizer

DATABASE_URL = os.environ.get(
    "AUDIO_SEARCH_DATABASE_URL",
    "postgresql://audio_search:audio_search@localhost:5432/audio_search",
)


async def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/real_stack_audio.wav"
    query = sys.argv[2] if len(sys.argv) > 2 else "audio search system"

    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2, init=_init_connection)

    audio_repo = PostgresAudioFileRepository(pool)
    chunk_repo = PostgresChunkRepository(pool)
    embedder = SentenceTransformerEmbedder()
    transcriber = WhisperTranscriber(model_size="tiny")
    diarizer = FakeDiarizer([SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=60.0)])

    pipeline = IngestPipeline(
        transcriber=transcriber,
        diarizer=diarizer,
        embedder=embedder,
        chunk_repository=chunk_repo,
        audio_file_repository=audio_repo,
    )
    search_service = SearchService(chunk_repository=chunk_repo, embedder=embedder, audio_file_repository=audio_repo)

    print(f"=== INGEST: {audio_path} ===")
    result = None
    try:
        result = await pipeline.run(audio_path)
        print(
            f"audio_file_id={result.audio_file_id} chunk_count={result.chunk_count} "
            f"duration_ms={result.duration_ms:.1f} skipped_existing={result.skipped_existing}"
        )

        print(f"\n=== SEARCH: {query!r} ===")
        results = await search_service.search(query, top_k=5)
        for r in results:
            print(f"[{r.score:.4f}] {r.file_name} {r.start_time:.1f}-{r.end_time:.1f} {r.speaker_id}: {r.text_snippet}")
        if not results:
            print("(no results)")
    finally:
        if result is not None and not result.skipped_existing:
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM audio_file WHERE audio_file_id = $1", result.audio_file_id)
            print(f"\n(cleaned up audio_file_id={result.audio_file_id})")
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
