"""Postgres-backed `FeedbackRepository` adapter (Stretch Goal).

`search_feedback` (see `db/schema.sql`) does not persist which retrieval
branch (keyword/semantic) surfaced a chunk — only `query_text`,
`chunk_id`, `rank_shown`, and `signal`. `get_branch_weights()` therefore
infers branch attribution post-hoc: for each feedback row, if the
chunk's Postgres full-text rank against `query_text` is non-trivial, the
positive/negative signal is attributed to the keyword branch; otherwise
to the semantic branch. This is a documented heuristic, not ground
truth — a production system would additionally persist
`retrieved_via` on `search_feedback` to make this exact.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

POSITIVE_SIGNALS = {"click", "thumbs_up"}
NEGATIVE_SIGNALS = {"thumbs_down"}
WEIGHT_STEP = 0.02
MIN_WEIGHT = 0.5
MAX_WEIGHT = 2.0
KEYWORD_RANK_THRESHOLD = 0.01
MAX_FEEDBACK_ROWS_CONSIDERED = 500


class PostgresFeedbackRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(self, query_text: str, chunk_id: UUID, rank_shown: int, signal: str) -> None:
        await self._pool.execute(
            "INSERT INTO search_feedback (query_text, chunk_id, rank_shown, signal) VALUES ($1, $2, $3, $4)",
            query_text, chunk_id, rank_shown, signal,
        )

    async def get_branch_weights(self, query_type: str) -> dict[str, float]:
        """`query_type` is accepted for `FeedbackRepository` port
        compatibility (e.g. future per-query-type tuning) but the current
        schema aggregates across all feedback globally — see module
        docstring for the reason.
        """
        rows = await self._pool.fetch(
            f"""
            SELECT f.signal,
                   ts_rank_cd(c.tsv, plainto_tsquery('english', f.query_text)) AS keyword_score
            FROM search_feedback f
            JOIN chunk c ON c.chunk_id = f.chunk_id
            ORDER BY f.created_at DESC
            LIMIT {MAX_FEEDBACK_ROWS_CONSIDERED}
            """
        )
        keyword_weight, semantic_weight = 1.0, 1.0
        for row in rows:
            if row["signal"] in POSITIVE_SIGNALS:
                sign = 1.0
            elif row["signal"] in NEGATIVE_SIGNALS:
                sign = -1.0
            else:
                continue  # e.g. dwell_time_ms — no clear positive/negative direction here
            delta = WEIGHT_STEP * sign
            if row["keyword_score"] and row["keyword_score"] > KEYWORD_RANK_THRESHOLD:
                keyword_weight += delta
            else:
                semantic_weight += delta

        return {
            "keyword": max(MIN_WEIGHT, min(MAX_WEIGHT, keyword_weight)),
            "semantic": max(MIN_WEIGHT, min(MAX_WEIGHT, semantic_weight)),
        }
