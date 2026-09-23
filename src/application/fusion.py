"""Reciprocal Rank Fusion (RRF) — pure, unit-testable fusion strategy.

See plan "Architecture Decisions -> Fusion" and "Relevance Feedback Loop"
(branch weighting hook). Swappable for weighted-sum fusion without
touching `search_service`'s orchestration logic (Strategy pattern).
"""

from __future__ import annotations

from uuid import UUID

from src.domain.models import RankedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    branch_rank_lists: dict[str, list[RankedChunk]],
    k: int = DEFAULT_RRF_K,
    branch_weights: dict[str, float] | None = None,
) -> list[tuple[UUID, float]]:
    """Fuse N ranked lists (one per named branch, e.g. "keyword"/"semantic")
    into a single ranked list of (chunk_id, fused_score) via RRF.

    RRF score for a chunk = sum over branches of weight_b / (k + rank_b),
    where rank_b is 1-indexed within that branch (chunks absent from a
    branch simply don't contribute a term for it). Higher score = better.

    `branch_weights` defaults to 1.0 for every branch (equal-weighted),
    matching the plan's default; the relevance-feedback loop (stretch
    goal) can pass in learned weights instead without changing this
    function's contract.
    """
    weights = branch_weights or {}
    scores: dict[UUID, float] = {}

    for branch_name, ranked_list in branch_rank_lists.items():
        weight = weights.get(branch_name, 1.0)
        for ranked in ranked_list:
            scores[ranked.chunk_id] = scores.get(ranked.chunk_id, 0.0) + weight / (k + ranked.rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
