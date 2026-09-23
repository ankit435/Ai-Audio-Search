"""Merge-and-split, speaker-bounded chunking strategy.

Pure functions only — no I/O, no DB, fully unit-testable in isolation.
See plan "Architecture Decisions -> Chunking" for the algorithm spec.
"""

from __future__ import annotations

import re
import uuid
from uuid import UUID

from src.domain.models import Chunk, SpeakerTurn

MERGE_TARGET_SECONDS = 15.0
MERGE_MAX_SECONDS = 30.0
SPLIT_CAP_SECONDS = 45.0
SPLIT_OVERLAP_SECONDS = 2.5

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _merge_consecutive_same_speaker(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Merge consecutive turns from the same speaker until MERGE_TARGET_SECONDS
    is reached, never exceeding MERGE_MAX_SECONDS, and never crossing a
    speaker boundary.
    """
    if not turns:
        return []

    merged: list[SpeakerTurn] = []
    current = turns[0]

    for nxt in turns[1:]:
        same_speaker = nxt.speaker_id == current.speaker_id
        current_duration = current.end_time - current.start_time
        combined_duration = nxt.end_time - current.start_time

        if same_speaker and current_duration < MERGE_TARGET_SECONDS and combined_duration <= MERGE_MAX_SECONDS:
            current = SpeakerTurn(
                speaker_id=current.speaker_id,
                text=f"{current.text} {nxt.text}".strip(),
                start_time=current.start_time,
                end_time=nxt.end_time,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def _split_long_turn(turn: SpeakerTurn) -> list[SpeakerTurn]:
    """Split a single-speaker turn longer than SPLIT_CAP_SECONDS at sentence
    boundaries into sub-turns capped at SPLIT_CAP_SECONDS, with overlap.

    Since word-level timestamps aren't available at this stage, time is
    apportioned proportionally to sentence character length — an
    approximation documented as a known limitation (see SOLUTION.md).
    """
    duration = turn.end_time - turn.start_time
    if duration <= SPLIT_CAP_SECONDS:
        return [turn]

    sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(turn.text) if s.strip()]
    if len(sentences) <= 1:
        return [turn]

    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return [turn]

    sub_turns: list[SpeakerTurn] = []
    cursor_time = turn.start_time
    buffer_sentences: list[str] = []
    buffer_start = turn.start_time

    for sentence in sentences:
        sentence_duration = duration * (len(sentence) / total_chars)
        buffer_sentences.append(sentence)
        cursor_time += sentence_duration
        buffer_duration = cursor_time - buffer_start

        if buffer_duration >= SPLIT_CAP_SECONDS:
            sub_turns.append(
                SpeakerTurn(
                    speaker_id=turn.speaker_id,
                    text=" ".join(buffer_sentences).strip(),
                    start_time=buffer_start,
                    end_time=cursor_time,
                )
            )
            # Overlap: next sub-chunk starts slightly before this one ends.
            buffer_start = max(turn.start_time, cursor_time - SPLIT_OVERLAP_SECONDS)
            buffer_sentences = []

    if buffer_sentences:
        sub_turns.append(
            SpeakerTurn(
                speaker_id=turn.speaker_id,
                text=" ".join(buffer_sentences).strip(),
                start_time=buffer_start,
                end_time=turn.end_time,
            )
        )

    return sub_turns or [turn]


def build_chunks(audio_file_id: UUID, turns: list[SpeakerTurn]) -> list[Chunk]:
    """Full merge-and-split pipeline: merge short same-speaker turns, split
    long ones, then materialize `Chunk` domain objects with prev/next
    linkage for query-time context stitching.
    """
    merged = _merge_consecutive_same_speaker(turns)

    final_turns: list[SpeakerTurn] = []
    for turn in merged:
        final_turns.extend(_split_long_turn(turn))

    chunk_ids = [uuid.uuid4() for _ in final_turns]
    chunks: list[Chunk] = []
    for i, turn in enumerate(final_turns):
        chunks.append(
            Chunk(
                chunk_id=chunk_ids[i],
                audio_file_id=audio_file_id,
                speaker_id=turn.speaker_id,
                text=turn.text,
                start_time=turn.start_time,
                end_time=turn.end_time,
                prev_chunk_id=chunk_ids[i - 1] if i > 0 else None,
                next_chunk_id=chunk_ids[i + 1] if i < len(chunk_ids) - 1 else None,
                token_count=len(turn.text.split()),
                char_count=len(turn.text),
            )
        )
    return chunks
