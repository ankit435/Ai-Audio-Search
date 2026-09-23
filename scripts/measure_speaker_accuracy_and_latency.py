"""Task 8 — speaker attribution accuracy + latency (p50/p95/p99)
measurement, run against the real golden dataset in Postgres.

Speaker attribution accuracy: compares each persisted chunk's
`speaker_id` against the ground-truth speaker for that time range
(`data/golden/*.turns.json`). Because `ScriptedDiarizer` (see
`src/infra/scripted_diarizer.py`) fed the *real* ground truth into the
ingestion pipeline, this measures the correctness of the
transcript-to-diarization time-overlap alignment logic
(`ingest_pipeline._align_transcript_to_speakers`) and the chunker's
speaker-purity guarantee — NOT real diarization-model accuracy, since
`pyannote.audio` is unavailable in this environment (see SOLUTION.md).
That caveat is intentional and stated up front rather than hidden.

Latency: runs the full labeled query set (`scripts/golden_query_set.py`)
several times each through the real `SearchService.search()` — real
embedding + real pgvector cosine search + real Postgres full-text search
+ RRF fusion — and aggregates the structured `search.response`
log events' `total_duration_ms` field (Evaluation Plan §4: "Computed by
aggregating structured `search.response` log events instead of separate
ad hoc timing code").

Usage:
    AUDIO_SEARCH_DATABASE_URL=postgresql://... python3 scripts/measure_speaker_accuracy_and_latency.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import asyncpg
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.golden_eval_utils import load_turns  # noqa: E402
from scripts.golden_query_set import QUERY_SET  # noqa: E402
from src.application.search_service import SearchService  # noqa: E402
from src.infra.postgres_repositories import PostgresAudioFileRepository, PostgresChunkRepository  # noqa: E402
from src.infra.sentence_transformer_embedder import SentenceTransformerEmbedder  # noqa: E402

DATABASE_URL = os.environ.get(
    "AUDIO_SEARCH_DATABASE_URL", "postgresql://audio_search:audio_search@localhost:5432/audio_search"
)
LATENCY_REPS = int(os.environ.get("LATENCY_REPS", "3"))
LATENCY_P95_THRESHOLD_MS = 500.0


class _SearchResponseCapture(logging.Handler):
    """Captures `total_duration_ms` from `search.response` structured log
    events instead of timing calls separately in this script."""

    def __init__(self) -> None:
        super().__init__()
        self.durations_ms: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "search.response":
            self.durations_ms.append(record.__dict__["total_duration_ms"])


async def measure_speaker_accuracy(pool: asyncpg.Pool) -> tuple[int, int]:
    correct = 0
    total = 0
    file_names = {q["file"] for q in QUERY_SET}
    for file_name in file_names:
        turns = load_turns(file_name)
        for turn in turns:
            midpoint = (turn.start_time + turn.end_time) / 2
            rows = await pool.fetch(
                """
                SELECT c.speaker_id FROM chunk c
                JOIN audio_file a ON a.audio_file_id = c.audio_file_id
                WHERE a.file_name = $1 AND c.start_time <= $2 AND c.end_time >= $2
                """,
                file_name,
                midpoint,
            )
            for row in rows:
                total += 1
                if row["speaker_id"] == turn.speaker_id:
                    correct += 1
    return correct, total


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, int(round(p / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


async def measure_latency(search_service: SearchService) -> list[float]:
    capture = _SearchResponseCapture()
    logging.getLogger("src.application.search_service").addHandler(capture)
    try:
        for _ in range(LATENCY_REPS):
            for entry in QUERY_SET:
                await search_service.search(entry["query"], top_k=10)
    finally:
        logging.getLogger("src.application.search_service").removeHandler(capture)
    return capture.durations_ms


async def main() -> None:
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4, init=_init_connection)
    logging.getLogger("src.application.search_service").setLevel(logging.INFO)

    correct, total = await measure_speaker_accuracy(pool)
    accuracy = correct / total if total else 0.0
    print("=== Speaker Attribution Accuracy (alignment logic, ground-truth diarization) ===")
    print(f"{correct}/{total} chunks correctly attributed = {accuracy:.3f}")

    search_service = SearchService(
        chunk_repository=PostgresChunkRepository(pool),
        embedder=SentenceTransformerEmbedder(),
        audio_file_repository=PostgresAudioFileRepository(pool),
    )
    # one warm-up pass excluded from measurement (cold start / model already loaded here anyway)
    await search_service.search(QUERY_SET[0]["query"], top_k=10)

    durations = sorted(await measure_latency(search_service))
    p50, p95, p99 = _percentile(durations, 50), _percentile(durations, 95), _percentile(durations, 99)
    print(f"\n=== Search Latency (n={len(durations)} requests, {LATENCY_REPS} reps x {len(QUERY_SET)} queries) ===")
    print(f"p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  max={max(durations):.1f}ms")

    await pool.close()

    print("\n=== Success Criteria ===")
    print(f"speaker attribution accuracy >= 0.90: {'PASS' if accuracy >= 0.90 else 'FAIL'} ({accuracy:.3f})")
    print(f"search latency p95 < 500ms: {'PASS' if p95 < LATENCY_P95_THRESHOLD_MS else 'FAIL'} ({p95:.1f}ms)")


if __name__ == "__main__":
    asyncio.run(main())
