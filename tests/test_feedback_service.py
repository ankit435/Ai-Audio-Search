"""Tests for `FeedbackService` (Stretch Goal) using an in-memory fake
`FeedbackRepository` — the real Postgres-backed heuristic in
`PostgresFeedbackRepository.get_branch_weights` is covered separately
against a real DB (see `test_postgres_feedback_repository.py`).
"""

from __future__ import annotations

import uuid

import pytest

from src.application.feedback_service import FeedbackService
from src.domain.exceptions import InvalidQueryError


class _FakeFeedbackRepository:
    def __init__(self) -> None:
        self.recorded: list[tuple] = []

    async def record(self, query_text, chunk_id, rank_shown, signal) -> None:
        self.recorded.append((query_text, chunk_id, rank_shown, signal))

    async def get_branch_weights(self, query_type: str) -> dict[str, float]:
        return {"keyword": 1.1, "semantic": 0.9}


@pytest.mark.asyncio
async def test_record_feedback_delegates_to_repository():
    repo = _FakeFeedbackRepository()
    service = FeedbackService(feedback_repository=repo)
    chunk_id = uuid.uuid4()

    await service.record_feedback("onboarding steps", chunk_id, rank_shown=2, signal="click")

    assert repo.recorded == [("onboarding steps", chunk_id, 2, "click")]


@pytest.mark.asyncio
async def test_record_feedback_rejects_empty_query():
    service = FeedbackService(feedback_repository=_FakeFeedbackRepository())
    with pytest.raises(InvalidQueryError):
        await service.record_feedback("   ", uuid.uuid4(), rank_shown=1, signal="click")


@pytest.mark.asyncio
async def test_record_feedback_rejects_unknown_signal():
    service = FeedbackService(feedback_repository=_FakeFeedbackRepository())
    with pytest.raises(InvalidQueryError):
        await service.record_feedback("query", uuid.uuid4(), rank_shown=1, signal="not_a_real_signal")


@pytest.mark.asyncio
async def test_record_feedback_rejects_non_positive_rank():
    service = FeedbackService(feedback_repository=_FakeFeedbackRepository())
    with pytest.raises(InvalidQueryError):
        await service.record_feedback("query", uuid.uuid4(), rank_shown=0, signal="click")


@pytest.mark.asyncio
async def test_get_branch_weights_delegates_to_repository():
    service = FeedbackService(feedback_repository=_FakeFeedbackRepository())
    weights = await service.get_branch_weights("keyword")
    assert weights == {"keyword": 1.1, "semantic": 0.9}
