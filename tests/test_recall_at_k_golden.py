"""Real recall@k / MRR evaluation (Task 6, real version; Evaluation Plan
§2) — the labeled query set (`scripts/golden_query_set.py`) run against
the actual golden dataset (`data/golden/`) already ingested into a real
Postgres+pgvector database via `scripts/ingest_golden_dataset.py`, using
the real `SentenceTransformerEmbedder` for query embedding.

Ground-truth relevance is resolved by real time-range overlap against
persisted chunks (`scripts/golden_eval_utils.py`), not by string
matching — robust to Whisper's minor ASR wording differences.

Skipped automatically if:
  - Postgres isn't reachable, or
  - the golden dataset hasn't been ingested yet (run
    `scripts/generate_golden_dataset.py` then
    `scripts/ingest_golden_dataset.py` first).

This test does NOT delete the golden dataset afterwards — it is
persistent reference data (like a fixture corpus), not throwaway test
output; `tests/test_postgres_integration.py` covers the
insert/cleanup-per-test path separately with unrelated synthetic data.
"""

from __future__ import annotations

import os
import sys

import asyncpg
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from golden_eval_utils import recall_at_k, reciprocal_rank, resolve_expected_chunk_ids  # noqa: E402
from golden_query_set import QUERY_SET  # noqa: E402

from src.application.search_service import SearchService  # noqa: E402
from src.infra.postgres_repositories import PostgresAudioFileRepository, PostgresChunkRepository  # noqa: E402
from src.infra.sentence_transformer_embedder import SentenceTransformerEmbedder  # noqa: E402

DATABASE_URL = os.environ.get(
    "AUDIO_SEARCH_DATABASE_URL", "postgresql://audio_search:audio_search@localhost:5432/audio_search"
)

# Real success criteria from the plan (Evaluation Plan / Success Criteria section).
RECALL_AT_5_THRESHOLD = 0.80
RECALL_AT_10_THRESHOLD = 0.90


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


@pytest_asyncio.fixture
async def pool():
    try:
        p = await asyncpg.create_pool(DATABASE_URL, init=_init_connection, min_size=1, max_size=4)
        async with p.acquire() as conn:
            await conn.execute("SELECT 1")
            count = await conn.fetchval("SELECT count(*) FROM audio_file")
    except Exception as exc:  # noqa: BLE001 - environment-dependent skip
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}: {exc}")
        return
    if count < 5:
        await p.close()
        pytest.skip(
            "Golden dataset not ingested (expected 5 audio_file rows). Run "
            "scripts/generate_golden_dataset.py then scripts/ingest_golden_dataset.py first."
        )
    yield p
    await p.close()


@pytest_asyncio.fixture
async def search_service(pool):
    return SearchService(
        chunk_repository=PostgresChunkRepository(pool),
        embedder=SentenceTransformerEmbedder(),
        audio_file_repository=PostgresAudioFileRepository(pool),
    )


@pytest.mark.asyncio
async def test_recall_at_k_and_mrr_on_golden_dataset(pool, search_service):
    per_query_results = []
    for entry in QUERY_SET:
        expected = await resolve_expected_chunk_ids(pool, entry["file"], entry["expected_turn_indices"])
        assert expected, f"no ground-truth chunk resolved for {entry['file']} turns {entry['expected_turn_indices']}"

        results = await search_service.search(entry["query"], top_k=10)
        ranked_ids = [r.chunk_id for r in results]

        per_query_results.append(
            {
                "query": entry["query"],
                "query_type": entry["query_type"],
                "file": entry["file"],
                "recall_5": recall_at_k(ranked_ids, expected, 5),
                "recall_10": recall_at_k(ranked_ids, expected, 10),
                "rr": reciprocal_rank(ranked_ids, expected),
            }
        )

    def _agg(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "recall_5": sum(r["recall_5"] for r in rows) / n,
            "recall_10": sum(r["recall_10"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / n,
        }

    overall = _agg(per_query_results)
    by_keyword = _agg([r for r in per_query_results if r["query_type"] == "keyword"])
    by_semantic = _agg([r for r in per_query_results if r["query_type"] == "semantic"])

    print("\n=== Recall@k / MRR — golden dataset ===")
    print(f"Overall  (n={overall['n']:>2}): recall@5={overall['recall_5']:.2f} recall@10={overall['recall_10']:.2f} MRR={overall['mrr']:.2f}")
    print(f"Keyword  (n={by_keyword['n']:>2}): recall@5={by_keyword['recall_5']:.2f} recall@10={by_keyword['recall_10']:.2f} MRR={by_keyword['mrr']:.2f}")
    print(f"Semantic (n={by_semantic['n']:>2}): recall@5={by_semantic['recall_5']:.2f} recall@10={by_semantic['recall_10']:.2f} MRR={by_semantic['mrr']:.2f}")

    misses = [r for r in per_query_results if r["recall_10"] == 0.0]
    if misses:
        print(f"\n{len(misses)} quer{'y' if len(misses) == 1 else 'ies'} missed entirely (recall@10=0):")
        for m in misses:
            print(f"  - [{m['query_type']}] {m['file']}: {m['query']!r}")

    assert overall["recall_5"] >= RECALL_AT_5_THRESHOLD, (
        f"recall@5={overall['recall_5']:.2f} below threshold {RECALL_AT_5_THRESHOLD}"
    )
    assert overall["recall_10"] >= RECALL_AT_10_THRESHOLD, (
        f"recall@10={overall['recall_10']:.2f} below threshold {RECALL_AT_10_THRESHOLD}"
    )
