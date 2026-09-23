"""Feedback application service (Stretch Goal).

Thin validation + orchestration layer over the `FeedbackRepository`
port — kept separate from `SearchService` so recording feedback and
computing adjusted RRF weights doesn't couple to the search request
path itself.
"""

from __future__ import annotations

from uuid import UUID

from src.domain.exceptions import InvalidQueryError
from src.domain.ports import FeedbackRepository

ALLOWED_SIGNALS = {"click", "thumbs_up", "thumbs_down", "dwell_time_ms"}


class FeedbackService:
    def __init__(self, feedback_repository: FeedbackRepository) -> None:
        self._feedback_repository = feedback_repository

    async def record_feedback(self, query_text: str, chunk_id: UUID, rank_shown: int, signal: str) -> None:
        stripped = query_text.strip()
        if not stripped:
            raise InvalidQueryError("query text must not be empty")
        if signal not in ALLOWED_SIGNALS:
            raise InvalidQueryError(f"signal must be one of {sorted(ALLOWED_SIGNALS)}, got {signal!r}")
        if rank_shown < 1:
            raise InvalidQueryError("rank_shown must be >= 1")
        await self._feedback_repository.record(stripped, chunk_id, rank_shown, signal)

    async def get_branch_weights(self, query_type: str = "default") -> dict[str, float]:
        return await self._feedback_repository.get_branch_weights(query_type)
