"""Demo server — runs the real FastAPI app wired to in-memory fake adapters
instead of Postgres/Whisper/pyannote/SentenceTransformer.

Purpose: let anyone (no Docker/GPU/model downloads required) exercise the
real `/search` and `/ingest` HTTP contracts end-to-end, seeded with a small
synthetic corpus. This is NOT how the app runs in production (see
`src/api/main.py` for the real composition root wired to Postgres) — it's
a demo/dev convenience, kept out of `src/` since it's not shipped code.

Usage:
    python3 scripts/demo_server.py
    curl "http://127.0.0.1:8000/search?q=quarterly+budget"
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import Depends, FastAPI, HTTPException

from src.api.schemas import IngestRequest, IngestResponse, SearchResponse, SearchResultItemResponse
from src.application.ingest_pipeline import IngestPipeline
from src.application.search_service import SearchService
from src.domain.exceptions import DomainError, InvalidQueryError
from src.domain.models import AudioFile, Chunk
from src.infra.logging_config import configure_logging
from tests.fakes import FakeAudioFileRepository, FakeChunkRepository, FakeDiarizer, FakeEmbedder, FakeTranscriber

configure_logging("INFO")

chunk_repo = FakeChunkRepository()
audio_file_repo = FakeAudioFileRepository()
embedder = FakeEmbedder()

_SEED_AUDIO_FILE_ID = uuid.uuid4()
_SEED_CORPUS = [
    ("SPEAKER_00", "We discussed the quarterly budget and revenue projections.", 0.0, 10.0),
    ("SPEAKER_01", "The weather today is sunny with a light breeze.", 10.0, 20.0),
    ("SPEAKER_00", "Our biggest competitor launched a similar product last month.", 20.0, 30.0),
    ("SPEAKER_01", "We should hire two more engineers for the platform team.", 30.0, 40.0),
]


async def _seed() -> None:
    await audio_file_repo.save(
        AudioFile(
            audio_file_id=_SEED_AUDIO_FILE_ID, file_name="demo_interview.wav",
            file_path="/demo/demo_interview.wav", checksum="demo-seed-checksum",
        )
    )
    for speaker, text, start, end in _SEED_CORPUS:
        chunk = Chunk(
            chunk_id=uuid.uuid4(), audio_file_id=_SEED_AUDIO_FILE_ID, speaker_id=speaker,
            text=text, start_time=start, end_time=end,
        )
        chunk = replace(chunk, embedding=await embedder.embed(chunk.text))
        await chunk_repo.save_chunks([chunk])


app = FastAPI(title="Audio Search — Hybrid Retrieval (DEMO, in-memory fakes)", version="0.1.0-demo")


def get_search_service() -> SearchService:
    return SearchService(chunk_repository=chunk_repo, embedder=embedder, audio_file_repository=audio_file_repo)


def get_ingest_pipeline() -> IngestPipeline:
    # Demo ingestion always "transcribes" a fixed two-line exchange, regardless
    # of the file path given — real transcription/diarization require Whisper
    # + pyannote (see src/infra/), intentionally not exercised in this demo.
    from src.domain.models import SpeakerTurn

    transcript = [SpeakerTurn(speaker_id="UNKNOWN", text="This is a demo ingestion.", start_time=0.0, end_time=3.0)]
    diarization = [SpeakerTurn(speaker_id="SPEAKER_00", text="", start_time=0.0, end_time=3.0)]
    return IngestPipeline(
        transcriber=FakeTranscriber(transcript), diarizer=FakeDiarizer(diarization), embedder=embedder,
        chunk_repository=chunk_repo, audio_file_repository=audio_file_repo,
    )


@app.get("/search", response_model=SearchResponse, tags=["search"])
async def search(q: str, top_k: int = 10, service: SearchService = Depends(get_search_service)) -> SearchResponse:
    import time

    start = time.perf_counter()
    try:
        results = await service.search(q, top_k=top_k)
    except InvalidQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DomainError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(
        query=q,
        results=[
            SearchResultItemResponse(
                chunk_id=r.chunk_id, file_name=r.file_name, speaker_id=r.speaker_id,
                text_snippet=r.text_snippet, start_time=r.start_time, end_time=r.end_time, score=r.score,
            )
            for r in results
        ],
        took_ms=(time.perf_counter() - start) * 1000,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest(request: IngestRequest, pipeline: IngestPipeline = Depends(get_ingest_pipeline)) -> IngestResponse:
    result = await pipeline.run(request.audio_file_path)
    return IngestResponse(
        audio_file_id=result.audio_file_id, chunk_count=result.chunk_count,
        duration_ms=result.duration_ms, skipped_existing=result.skipped_existing,
    )


if __name__ == "__main__":
    asyncio.run(_seed())
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
