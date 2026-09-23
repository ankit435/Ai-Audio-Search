"""Unit tests for the pure Reciprocal Rank Fusion function."""

from __future__ import annotations

import uuid

from src.application.fusion import reciprocal_rank_fusion
from src.domain.models import RankedChunk


def test_chunk_ranked_first_in_both_branches_wins():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    keyword = [RankedChunk(chunk_id=a, rank=1), RankedChunk(chunk_id=b, rank=2)]
    semantic = [RankedChunk(chunk_id=a, rank=1), RankedChunk(chunk_id=c, rank=2)]

    fused = reciprocal_rank_fusion({"keyword": keyword, "semantic": semantic})

    assert fused[0][0] == a  # present + top-ranked in both branches


def test_chunk_present_in_both_branches_outranks_single_branch_top_rank():
    a, b = uuid.uuid4(), uuid.uuid4()
    keyword = [RankedChunk(chunk_id=b, rank=1), RankedChunk(chunk_id=a, rank=2)]
    semantic = [RankedChunk(chunk_id=a, rank=1)]

    fused = reciprocal_rank_fusion({"keyword": keyword, "semantic": semantic})
    fused_ids = [chunk_id for chunk_id, _ in fused]

    assert fused_ids[0] == a  # 1/(60+2) + 1/(60+1) > 1/(60+1) alone


def test_empty_branch_does_not_error():
    a = uuid.uuid4()
    keyword = [RankedChunk(chunk_id=a, rank=1)]

    fused = reciprocal_rank_fusion({"keyword": keyword, "semantic": []})

    assert fused == [(a, 1 / 61)]


def test_branch_weights_bias_fusion():
    a, b = uuid.uuid4(), uuid.uuid4()
    keyword = [RankedChunk(chunk_id=a, rank=1)]
    semantic = [RankedChunk(chunk_id=b, rank=1)]

    fused = reciprocal_rank_fusion(
        {"keyword": keyword, "semantic": semantic}, branch_weights={"keyword": 0.1, "semantic": 2.0}
    )

    assert fused[0][0] == b  # semantic weighted much higher


def test_no_branches_returns_empty_list():
    assert reciprocal_rank_fusion({}) == []
