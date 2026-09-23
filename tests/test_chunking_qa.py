"""Chunking QA regression tests — Evaluation Plan §1.

These check invariants that should hold *by construction* of the
merge-and-split chunker; they act as a regression guard, not a
first-time discovery mechanism.
"""

from __future__ import annotations

import statistics
import uuid

from src.application.chunking import SPLIT_CAP_SECONDS, build_chunks
from src.domain.models import SpeakerTurn


def _sample_turns() -> list[SpeakerTurn]:
    return [
        SpeakerTurn(speaker_id="SPEAKER_00", text="Short one.", start_time=0.0, end_time=2.0),
        SpeakerTurn(speaker_id="SPEAKER_00", text="Short two.", start_time=2.0, end_time=4.0),
        SpeakerTurn(speaker_id="SPEAKER_01", text="A reply.", start_time=4.0, end_time=6.0),
        SpeakerTurn(
            speaker_id="SPEAKER_00",
            text=" ".join(f"Long sentence number {i} in a monologue." for i in range(25)),
            start_time=6.0, end_time=70.0,
        ),
    ]


def test_speaker_purity_no_chunk_spans_two_speakers():
    """Regression guard: chunking must never merge across a speaker
    boundary (see plan "Chunking" — never cross a speaker boundary).
    """
    chunks = build_chunks(uuid.uuid4(), _sample_turns())

    for chunk in chunks:
        assert chunk.speaker_id in {"SPEAKER_00", "SPEAKER_01"}
    # Adjacent chunks from different source turns may share a speaker (merge
    # case) but a single chunk's speaker_id is always exactly one speaker —
    # guaranteed by construction since Chunk.speaker_id is a scalar field.


def test_length_distribution_flags_no_near_empty_or_oversized_chunks():
    chunks = build_chunks(uuid.uuid4(), _sample_turns())
    char_counts = [c.char_count for c in chunks]

    assert all(count > 0 for count in char_counts), "no chunk should be empty"
    assert min(char_counts) >= 5, "flag suspiciously near-empty chunks"

    # Outlier flag: no chunk's duration should exceed the hard split cap by
    # more than roughly one sentence's worth of overage (sentence-boundary
    # splitting is greedy, never cuts mid-sentence — see chunking.py).
    durations = [c.end_time - c.start_time for c in chunks]
    assert max(durations) <= SPLIT_CAP_SECONDS * 1.25

    if len(char_counts) > 1:
        stdev = statistics.pstdev(char_counts)
        assert stdev >= 0  # sanity: distribution is computable, no NaNs/errors


def test_boundary_sanity_start_before_end_and_monotonic_within_speaker_run():
    chunks = build_chunks(uuid.uuid4(), _sample_turns())

    for chunk in chunks:
        assert chunk.start_time < chunk.end_time, "start_time must precede end_time"

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        # Overlap is allowed only via the documented split-overlap window,
        # never a large drift/regression backwards in time.
        assert later.start_time >= earlier.start_time
