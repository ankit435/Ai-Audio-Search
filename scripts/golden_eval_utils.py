"""Shared utilities for evaluating the golden dataset against the real
Postgres+pgvector database: resolving ground-truth turn indices to
persisted `chunk_id`s, and computing recall@k / MRR.

Used by both `tests/test_recall_at_k_golden.py` (Task 6, real version)
and `scripts/measure_speaker_accuracy_and_latency.py` (Task 8).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from uuid import UUID

import asyncpg

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "golden")


@dataclass(frozen=True)
class GroundTruthTurn:
    turn_index: int
    speaker_id: str
    text: str
    start_time: float
    end_time: float


def load_turns(file_name: str) -> list[GroundTruthTurn]:
    """Load the `<file_name>.turns.json` sidecar (file_name includes .wav)."""
    base, _ext = os.path.splitext(file_name)
    with open(os.path.join(GOLDEN_DIR, f"{base}.turns.json")) as f:
        data = json.load(f)
    return [
        GroundTruthTurn(
            turn_index=t["turn_index"],
            speaker_id=t["speaker_id"],
            text=t["text"],
            start_time=t["start_time"],
            end_time=t["end_time"],
        )
        for t in data["turns"]
    ]


async def resolve_turn_to_chunk_ids(
    pool: asyncpg.Pool, file_name: str, turn_index: int
) -> list[UUID]:
    """Find the real persisted chunk(s) whose [start_time, end_time] range
    overlaps the ground-truth turn's midpoint — robust to minor Whisper
    wording differences and to however turns were merged/split by the
    chunker, since it matches on time, not text.
    """
    turns = load_turns(file_name)
    turn = next(t for t in turns if t.turn_index == turn_index)
    midpoint = (turn.start_time + turn.end_time) / 2

    rows = await pool.fetch(
        """
        SELECT c.chunk_id
        FROM chunk c
        JOIN audio_file a ON a.audio_file_id = c.audio_file_id
        WHERE a.file_name = $1 AND c.start_time <= $2 AND c.end_time >= $2
        """,
        file_name,
        midpoint,
    )
    return [r["chunk_id"] for r in rows]


async def resolve_expected_chunk_ids(
    pool: asyncpg.Pool, file_name: str, expected_turn_indices: list[int]
) -> set[UUID]:
    expected: set[UUID] = set()
    for ti in expected_turn_indices:
        expected.update(await resolve_turn_to_chunk_ids(pool, file_name, ti))
    return expected


def reciprocal_rank(ranked_chunk_ids: list[UUID], expected: set[UUID]) -> float:
    for rank, cid in enumerate(ranked_chunk_ids, start=1):
        if cid in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_chunk_ids: list[UUID], expected: set[UUID], k: int) -> float:
    if not expected:
        return 0.0
    top_k = set(ranked_chunk_ids[:k])
    return 1.0 if top_k & expected else 0.0
