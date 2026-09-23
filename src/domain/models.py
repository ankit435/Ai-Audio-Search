"""Domain models — plain data structures, no infra/framework imports.

See .github/prompts/plan-audioSearchHybridRetrieval.prompt.md
  "Data Model — Chunk" section for field-by-field rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class AudioFile:
    """A source audio file that has been (or is being) ingested."""

    audio_file_id: UUID
    file_name: str
    file_path: str
    checksum: str
    duration_s: float | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A single diarized speaker turn, prior to chunking.

    Produced by aligning `Transcriber` segments to `Diarizer` output.
    """

    speaker_id: str
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class Chunk:
    """A merge-and-split, speaker-bounded transcript chunk.

    Never spans more than one speaker (see plan "Chunking" section).
    """

    chunk_id: UUID
    audio_file_id: UUID
    speaker_id: str
    text: str
    start_time: float
    end_time: float
    embedding: list[float] | None = None
    prev_chunk_id: UUID | None = None
    next_chunk_id: UUID | None = None
    token_count: int | None = None
    char_count: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SearchResultItem:
    """A single fused, ranked search result, ready for API serialization."""

    chunk_id: UUID
    audio_file_id: UUID
    file_name: str
    speaker_id: str
    text_snippet: str
    start_time: float
    end_time: float
    score: float


@dataclass(frozen=True, slots=True)
class RankedChunk:
    """A chunk id with its rank/score in a single retrieval branch.

    Used as the common currency between the keyword branch, semantic
    branch, and the pure RRF fusion function — none of which need to
    know about each other's scoring scale.
    """

    chunk_id: UUID
    rank: int
    raw_score: float = field(compare=False, default=0.0)
