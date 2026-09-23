"""Search-and-answer application service (Stretch Goal).

Composes `SearchService` (hybrid retrieval) with an `AnswerGenerator`
port — kept as a distinct service (not a method on `SearchService`) so
plain search's single-response contract, relied on by the recall@k
evaluation suite, is never coupled to answer-generation concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.application.search_service import SearchService
from src.domain.models import Chunk, SearchResultItem
from src.domain.ports import AnswerGenerator


@dataclass(frozen=True, slots=True)
class SearchAnswerResult:
    answer: str
    citations: list[SearchResultItem]


class SearchAnswerService:
    def __init__(self, search_service: SearchService, answer_generator: AnswerGenerator) -> None:
        self._search_service = search_service
        self._answer_generator = answer_generator

    async def answer(self, query_text: str, top_k: int = 5) -> SearchAnswerResult:
        results = await self._search_service.search(query_text, top_k=top_k)
        chunks = [_result_to_chunk(r) for r in results]
        answer_text = await self._answer_generator.generate(query_text, chunks)
        return SearchAnswerResult(answer=answer_text, citations=results)


def _result_to_chunk(result: SearchResultItem) -> Chunk:
    """`AnswerGenerator.generate` takes `Chunk`s (domain model with full
    text), but `SearchService.search` returns `SearchResultItem`s
    (API-shaped, truncated `text_snippet`). Since the answer generator
    only needs speaker/timestamp/text for citation purposes, the
    (possibly truncated) snippet is an acceptable stand-in here rather
    than doing a second `get_by_ids` round-trip — documented tradeoff.
    """
    return Chunk(
        chunk_id=result.chunk_id,
        audio_file_id=result.audio_file_id,
        speaker_id=result.speaker_id,
        text=result.text_snippet,
        start_time=result.start_time,
        end_time=result.end_time,
    )
