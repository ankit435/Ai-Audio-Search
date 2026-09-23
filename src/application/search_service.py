"""Hybrid search application service.

Fuses keyword (`ts_rank_cd`) and semantic (pgvector cosine) rank lists
via the pure RRF function in `application.fusion`. Depends only on
domain ports — `ChunkRepository`, `Embedder`, `AudioFileRepository` —
never on infra.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from src.application.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.domain.exceptions import InvalidQueryError
from src.domain.models import Chunk, RankedChunk, SearchResultItem
from src.domain.ports import AudioFileRepository, ChunkRepository, Embedder
from src.infra.logging_config import log_event

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 1000
SNIPPET_MAX_CHARS = 280


def _validate_query(query_text: str) -> str:
    """Fail fast at the system boundary — no silent truncation/guessing."""
    stripped = query_text.strip()
    if not stripped:
        raise InvalidQueryError("query text must not be empty")
    if len(stripped) > MAX_QUERY_LENGTH:
        raise InvalidQueryError(f"query text exceeds max length of {MAX_QUERY_LENGTH} chars")
    return stripped


def _snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _hydrate_results(
    fused: list[tuple], chunks_by_id: dict, file_names_by_audio_id: dict, top_k: int
) -> list[SearchResultItem]:
    results: list[SearchResultItem] = []
    for chunk_id, score in fused[:top_k]:
        chunk: Chunk | None = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue  # defensive: repository/fusion disagreement should never happen
        results.append(
            SearchResultItem(
                chunk_id=chunk.chunk_id,
                audio_file_id=chunk.audio_file_id,
                file_name=file_names_by_audio_id.get(chunk.audio_file_id, ""),
                speaker_id=chunk.speaker_id,
                text_snippet=_snippet(chunk.text),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                score=score,
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class BranchWeights:
    keyword: float = 1.0
    semantic: float = 1.0


class SearchService:
    def __init__(
        self,
        chunk_repository: ChunkRepository,
        embedder: Embedder,
        audio_file_repository: AudioFileRepository,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._chunk_repository = chunk_repository
        self._embedder = embedder
        self._audio_file_repository = audio_file_repository
        self._rrf_k = rrf_k

    async def search(
        self, query_text: str, top_k: int = 10, branch_weights: BranchWeights | None = None
    ) -> list[SearchResultItem]:
        """Single-response hybrid search: keyword + semantic -> RRF fusion.

        This is the path evaluated by recall@k tests (see Evaluation Plan
        §2) — always returns one complete, deterministic result set.
        """
        request_start = time.perf_counter()
        query_text = _validate_query(query_text)
        query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]
        log_event(logger, logging.INFO, "search.request", query_hash=query_hash, top_k=top_k)

        branch_start = time.perf_counter()
        keyword_ranks: list[RankedChunk] = await self._chunk_repository.search_keyword(query_text, top_k)
        log_event(
            logger, logging.INFO, "search.keyword_branch.end", query_hash=query_hash,
            duration_ms=(time.perf_counter() - branch_start) * 1000, result_count=len(keyword_ranks),
        )

        branch_start = time.perf_counter()
        query_embedding = await self._embedder.embed(query_text)
        semantic_ranks: list[RankedChunk] = await self._chunk_repository.search_semantic(query_embedding, top_k)
        log_event(
            logger, logging.INFO, "search.semantic_branch.end", query_hash=query_hash,
            duration_ms=(time.perf_counter() - branch_start) * 1000, result_count=len(semantic_ranks),
        )

        weights = branch_weights or BranchWeights()
        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"keyword": keyword_ranks, "semantic": semantic_ranks},
            k=self._rrf_k,
            branch_weights={"keyword": weights.keyword, "semantic": weights.semantic},
        )
        log_event(
            logger, logging.INFO, "search.fusion.end", query_hash=query_hash,
            duration_ms=(time.perf_counter() - fusion_start) * 1000, result_count=len(fused),
        )

        chunk_ids = [chunk_id for chunk_id, _ in fused[:top_k]]
        chunks = await self._chunk_repository.get_by_ids(chunk_ids)
        chunks_by_id = {c.chunk_id: c for c in chunks}

        unique_audio_file_ids = {c.audio_file_id for c in chunks}
        file_names_by_audio_id = {}
        for audio_file_id in unique_audio_file_ids:
            audio_file = await self._audio_file_repository.get(audio_file_id)
            if audio_file is not None:
                file_names_by_audio_id[audio_file_id] = audio_file.file_name

        results = _hydrate_results(fused, chunks_by_id, file_names_by_audio_id, top_k)

        log_event(
            logger, logging.INFO, "search.response", query_hash=query_hash,
            total_duration_ms=(time.perf_counter() - request_start) * 1000,
            top_result_score=results[0].score if results else None,
        )
        return results

    async def search_stream(
        self, query_text: str, top_k: int = 10, branch_weights: BranchWeights | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Stretch Goal — SSE-friendly incremental search.

        Yields one dict per stage as soon as it's available (keyword
        branch, then semantic branch, then fused/hydrated results),
        instead of blocking until the whole request completes like
        `search()`. Each dict has an `"event"` key naming the stage, so
        the API layer (`GET /search/stream`) can forward it directly as
        an SSE `event:`/`data:` pair. This is intentionally a *separate*
        method from `search()` (not a refactor of it) so the
        single-response contract relied on by the recall@k evaluation
        suite is never accidentally changed by streaming concerns.
        """
        request_start = time.perf_counter()
        query_text = _validate_query(query_text)
        query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]
        log_event(logger, logging.INFO, "search.request", query_hash=query_hash, top_k=top_k)

        keyword_ranks: list[RankedChunk] = await self._chunk_repository.search_keyword(query_text, top_k)
        yield {"event": "keyword_results", "count": len(keyword_ranks)}

        query_embedding = await self._embedder.embed(query_text)
        semantic_ranks: list[RankedChunk] = await self._chunk_repository.search_semantic(query_embedding, top_k)
        yield {"event": "semantic_results", "count": len(semantic_ranks)}

        weights = branch_weights or BranchWeights()
        fused = reciprocal_rank_fusion(
            {"keyword": keyword_ranks, "semantic": semantic_ranks},
            k=self._rrf_k,
            branch_weights={"keyword": weights.keyword, "semantic": weights.semantic},
        )
        yield {"event": "fused_results", "count": len(fused)}

        chunk_ids = [chunk_id for chunk_id, _ in fused[:top_k]]
        chunks = await self._chunk_repository.get_by_ids(chunk_ids)
        chunks_by_id = {c.chunk_id: c for c in chunks}

        unique_audio_file_ids = {c.audio_file_id for c in chunks}
        file_names_by_audio_id = {}
        for audio_file_id in unique_audio_file_ids:
            audio_file = await self._audio_file_repository.get(audio_file_id)
            if audio_file is not None:
                file_names_by_audio_id[audio_file_id] = audio_file.file_name

        results = _hydrate_results(fused, chunks_by_id, file_names_by_audio_id, top_k)
        log_event(
            logger, logging.INFO, "search.response", query_hash=query_hash,
            total_duration_ms=(time.perf_counter() - request_start) * 1000,
            top_result_score=results[0].score if results else None,
        )
        yield {
            "event": "done",
            "results": [
                {
                    "chunk_id": str(r.chunk_id),
                    "file_name": r.file_name,
                    "speaker_id": r.speaker_id,
                    "text_snippet": r.text_snippet,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "score": r.score,
                }
                for r in results
            ],
            "took_ms": (time.perf_counter() - request_start) * 1000,
        }
