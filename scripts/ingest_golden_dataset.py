"""Ingests the synthetic golden dataset (`data/golden/*.wav`) into the
real Postgres+pgvector database using the real production pipeline:

    real WhisperTranscriber -> ScriptedDiarizer (ground truth, see
    src/infra/scripted_diarizer.py) -> real chunking/alignment ->
    real SentenceTransformerEmbedder -> real Postgres repositories

This is the actual product ingestion path (`IngestPipeline`), with only
the diarizer swapped for a ground-truth adapter (documented reason:
`pyannote.audio` is unavailable in this environment). Idempotent via
the pipeline's existing checksum check — safe to re-run.

Usage:
    AUDIO_SEARCH_DATABASE_URL=postgresql://... python3 scripts/ingest_golden_dataset.py
"""

from __future__ import annotations

import asyncio
import glob
import os
import sys

import asyncpg
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.application.ingest_pipeline import IngestPipeline
from src.infra.postgres_repositories import PostgresAudioFileRepository, PostgresChunkRepository
from src.infra.scripted_diarizer import ScriptedDiarizer
from src.infra.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infra.whisper_transcriber import WhisperTranscriber

DATABASE_URL = os.environ.get(
    "AUDIO_SEARCH_DATABASE_URL",
    "postgresql://audio_search:audio_search@localhost:5432/audio_search",
)
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "golden")
WHISPER_MODEL_SIZE = os.environ.get("AUDIO_SEARCH_WHISPER_MODEL_SIZE", "base")


async def main() -> None:
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4, init=_init_connection)

    audio_repo = PostgresAudioFileRepository(pool)
    chunk_repo = PostgresChunkRepository(pool)
    embedder = SentenceTransformerEmbedder()
    transcriber = WhisperTranscriber(model_size=WHISPER_MODEL_SIZE)
    diarizer = ScriptedDiarizer()

    pipeline = IngestPipeline(
        transcriber=transcriber,
        diarizer=diarizer,
        embedder=embedder,
        chunk_repository=chunk_repo,
        audio_file_repository=audio_repo,
    )

    wav_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.wav")))
    if not wav_paths:
        print(f"No .wav files found in {GOLDEN_DIR}. Run scripts/generate_golden_dataset.py first.")
        await pool.close()
        return

    for wav_path in wav_paths:
        result = await pipeline.run(wav_path)
        status = "already ingested (skipped)" if result.skipped_existing else "ingested"
        print(
            f"{os.path.basename(wav_path)}: {status} "
            f"audio_file_id={result.audio_file_id} chunk_count={result.chunk_count} "
            f"duration_ms={result.duration_ms:.1f}"
        )

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
