"""Composition root — the only module that imports both `infra.*` and
`application.*`. Wires concrete adapters into application services via
FastAPI's `Depends(...)`, maps domain exceptions to HTTP status codes,
and configures structured logging + app-level OpenAPI/Swagger metadata.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import asyncpg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ErrorResponse,
    FeedbackRequest,
    IngestRequest,
    IngestResponse,
    SearchAnswerResponse,
    SearchResponse,
    SearchResultItemResponse,
)
from src.api.settings import Settings
from src.application.feedback_service import FeedbackService
from src.application.ingest_pipeline import IngestPipeline
from src.application.search_answer_service import SearchAnswerService
from src.application.search_service import SearchService
from src.domain.exceptions import ChunkPersistenceError, DomainError, EmbeddingFailedError, InvalidQueryError
from src.infra.extractive_answer_generator import ExtractiveAnswerGenerator
from src.infra.logging_config import configure_logging, log_event
from src.infra.openai_answer_generator import OpenAIAnswerGenerator
from src.infra.openai_compatible_embedder import OpenAICompatibleEmbedder
from src.infra.postgres_feedback_repository import PostgresFeedbackRepository
from src.infra.postgres_repositories import PostgresAudioFileRepository, PostgresChunkRepository
from src.infra.pyannote_diarizer import PyannoteDiarizer
from src.infra.resilient_answer_generator import ResilientAnswerGenerator
from src.infra.resilient_diarizer import ResilientDiarizer
from src.infra.resilient_transcriber import ResilientTranscriber
from src.infra.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infra.single_speaker_diarizer import SingleSpeakerDiarizer
from src.infra.whisper_transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _build_embedder(settings: Settings):
    """Embedding provider is fully swappable via `AUDIO_SEARCH_EMBEDDING_*`
    env vars — no code change needed to point at a different model/provider
    (e.g. an NVIDIA NIM hosted embedding model, or any other HF/local
    `sentence-transformers`-compatible checkpoint). Add a new `elif` branch
    here (and a new adapter under `src/infra/`) for a provider that doesn't
    fit either shape (e.g. a gRPC-based service).
    """
    if settings.embedding_provider == "openai-compatible":
        return OpenAICompatibleEmbedder(
            model_name=settings.embedding_model_name, base_url=settings.embedding_base_url
        )
    return SentenceTransformerEmbedder(settings.embedding_model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    async def _init_connection(conn: asyncpg.Connection) -> None:
        from pgvector.asyncpg import register_vector

        await register_vector(conn)

    app.state.pool = await asyncpg.create_pool(settings.database_url, init=_init_connection)
    app.state.embedder = _build_embedder(settings)

    # Transcriber: primary/fallback backends are configurable
    # (`AUDIO_SEARCH_TRANSCRIPTION_PRIMARY_BACKEND` /
    # `..._FALLBACK_BACKEND`, default faster-whisper -> openai-whisper).
    # `whisper_model_size` also accepts a local path or HF repo id (e.g. a
    # community/NVIDIA CTranslate2-converted checkpoint) when using the
    # `faster-whisper` backend. Automatically falls back if the primary
    # backend's model download is blocked (gated/network/proxy). See
    # `ResilientTranscriber` — reacts to the typed `TranscriptionFailedError`
    # only, so this works regardless of *why* a given backend is unavailable.
    app.state.transcriber = ResilientTranscriber(
        primary=WhisperTranscriber(settings.whisper_model_size, backend=settings.transcription_primary_backend),
        fallback=WhisperTranscriber(settings.whisper_model_size, backend=settings.transcription_fallback_backend),
    )

    # Diarizer: prefer real `pyannote.audio`; automatically fall back to
    # `SingleSpeakerDiarizer` (no model/network/credentials required) if the
    # gated model / HF_TOKEN / network is unavailable, so `/ingest` degrades
    # gracefully (single-speaker attribution) instead of failing outright.
    app.state.diarizer = ResilientDiarizer(primary=PyannoteDiarizer(), fallback=SingleSpeakerDiarizer())

    yield
    await app.state.pool.close()


app = FastAPI(
    title="Audio Search — Hybrid Retrieval",
    description=(
        "Hybrid (keyword + semantic) search over diarized, transcribed audio "
        "conversations. Returns file, timestamp, and speaker for each hit."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def get_search_service() -> SearchService:
    settings = get_settings()
    pool = app.state.pool
    return SearchService(
        chunk_repository=PostgresChunkRepository(pool),
        embedder=app.state.embedder,
        audio_file_repository=PostgresAudioFileRepository(pool),
        rrf_k=settings.rrf_k,
    )


def get_ingest_pipeline() -> IngestPipeline:
    pool = app.state.pool
    return IngestPipeline(
        transcriber=app.state.transcriber,
        diarizer=app.state.diarizer,
        embedder=app.state.embedder,
        chunk_repository=PostgresChunkRepository(pool),
        audio_file_repository=PostgresAudioFileRepository(pool),
    )


def get_feedback_service() -> FeedbackService:
    return FeedbackService(feedback_repository=PostgresFeedbackRepository(app.state.pool))


def _build_answer_generator():
    """LLM adapter requires an API key at construction time; if it's
    missing there's nothing to retry, so skip the resilient wrapper
    entirely rather than pretending there's a primary to fall back from.
    If it *is* configured, still wrap it — a working key doesn't
    guarantee a working network path at call time (see
    `ResilientAnswerGenerator` docstring).

    `model`/`base_url` are read from settings so this works with any
    OpenAI-compatible provider (NVIDIA NIM, self-hosted vLLM/Ollama,
    hosted OpenAI, ...) purely via configuration — see
    `src/infra/openai_answer_generator.py` for provider examples.
    """
    settings = get_settings()
    try:
        primary = OpenAIAnswerGenerator(model=settings.answer_model, base_url=settings.answer_base_url)
    except EmbeddingFailedError as exc:
        log_event(
            logger, logging.INFO, "answer_generator.openai_unavailable",
            reason=exc.reason, using="ExtractiveAnswerGenerator",
        )
        return ExtractiveAnswerGenerator()
    return ResilientAnswerGenerator(primary=primary, fallback=ExtractiveAnswerGenerator())


def get_search_answer_service() -> SearchAnswerService:
    pool = app.state.pool
    settings = get_settings()
    return SearchAnswerService(
        search_service=SearchService(
            chunk_repository=PostgresChunkRepository(pool),
            embedder=app.state.embedder,
            audio_file_repository=PostgresAudioFileRepository(pool),
            rrf_k=settings.rrf_k,
        ),
        answer_generator=_build_answer_generator(),
    )


@app.exception_handler(DomainError)
async def domain_error_handler(request, exc: DomainError):
    status_code = 400 if isinstance(exc, InvalidQueryError) else 500
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(error_type=type(exc).__name__, message=str(exc)).model_dump(),
    )


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid keyword + semantic search",
    description=(
        "Runs keyword (Postgres ts_rank_cd) and semantic (pgvector cosine) "
        "branches in parallel and fuses them via Reciprocal Rank Fusion. "
        "Single-response contract — used by the recall@k evaluation suite."
    ),
    tags=["search"],
)
async def search(
    q: str, top_k: int = 10, service: SearchService = Depends(get_search_service)
) -> SearchResponse:
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


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest an audio file",
    description="Runs transcribe -> diarize -> chunk -> embed -> index. Idempotent per checksum.",
    tags=["ingest"],
)
async def ingest(
    request: IngestRequest, pipeline: IngestPipeline = Depends(get_ingest_pipeline)
) -> IngestResponse:
    try:
        result = await pipeline.run(request.audio_file_path)
    except ChunkPersistenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except DomainError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        audio_file_id=result.audio_file_id, chunk_count=result.chunk_count,
        duration_ms=result.duration_ms, skipped_existing=result.skipped_existing,
    )


@app.get(
    "/search/stream",
    summary="Stretch Goal — incremental hybrid search via Server-Sent Events",
    description=(
        "Streams `keyword_results` -> `semantic_results` -> `fused_results` -> "
        "`done` events as each stage completes, instead of waiting for the "
        "whole request. Useful for UIs that want to show progress on slower "
        "queries/larger corpora. The single-response `GET /search` above "
        "remains the contract used by the recall@k evaluation suite."
    ),
    tags=["search"],
)
async def search_stream(
    q: str, top_k: int = 10, service: SearchService = Depends(get_search_service)
):
    async def _event_source():
        try:
            async for event in service.search_stream(q, top_k=top_k):
                yield f"event: {event['event']}\ndata: {json.dumps(event)}\n\n"
        except InvalidQueryError as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        except DomainError as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_event_source(), media_type="text/event-stream")


@app.post(
    "/search/feedback",
    status_code=204,
    summary="Stretch Goal — record relevance feedback on a search result",
    description=(
        "Records a click/thumbs_up/thumbs_down/dwell_time_ms signal for a "
        "(query, chunk, rank_shown) triple. Aggregated feedback can be used "
        "to adjust RRF branch weights via `FeedbackService.get_branch_weights`."
    ),
    tags=["search"],
)
async def search_feedback(
    request: FeedbackRequest, service: FeedbackService = Depends(get_feedback_service)
) -> None:
    try:
        await service.record_feedback(
            query_text=request.query_text,
            chunk_id=request.chunk_id,
            rank_shown=request.rank_shown,
            signal=request.signal,
        )
    except InvalidQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/search/answer",
    response_model=SearchAnswerResponse,
    summary="Stretch Goal — extractive RAG-style answer over search results",
    description=(
        "Runs hybrid search, then generates a cited answer by extracting the "
        "most relevant snippet(s) from the top results — no external LLM API "
        "key required (see `src/infra/extractive_answer_generator.py`). A "
        "gated `OpenAIAnswerGenerator` adapter also exists for real LLM "
        "generation if `OPENAI_API_KEY` is configured, behind the same "
        "`AnswerGenerator` port."
    ),
    tags=["search"],
)
async def search_answer(
    q: str, top_k: int = 5, service: SearchAnswerService = Depends(get_search_answer_service)
) -> SearchAnswerResponse:
    try:
        result = await service.answer(q, top_k=top_k)
    except InvalidQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DomainError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchAnswerResponse(
        query=q,
        answer=result.answer,
        citations=[
            SearchResultItemResponse(
                chunk_id=c.chunk_id, file_name=c.file_name, speaker_id=c.speaker_id,
                text_snippet=c.text_snippet, start_time=c.start_time, end_time=c.end_time, score=c.score,
            )
            for c in result.citations
        ],
    )
