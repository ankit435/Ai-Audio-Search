"""Unit tests for the pure merge-and-split chunking functions.

Pure, no I/O — see plan "Chunking Quality" evaluation section.
"""

from __future__ import annotations

import uuid

from src.application.chunking import build_chunks
from src.domain.models import SpeakerTurn


def test_merges_short_consecutive_same_speaker_turns():
    audio_file_id = uuid.uuid4()
    turns = [
        SpeakerTurn(speaker_id="A", text="Hello there.", start_time=0.0, end_time=2.0),
        SpeakerTurn(speaker_id="A", text="How are you?", start_time=2.0, end_time=4.0),
        SpeakerTurn(speaker_id="A", text="Doing well I hope.", start_time=4.0, end_time=6.0),
    ]

    chunks = build_chunks(audio_file_id, turns)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello there. How are you? Doing well I hope."
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 6.0


def test_never_merges_across_speaker_boundary():
    audio_file_id = uuid.uuid4()
    turns = [
        SpeakerTurn(speaker_id="A", text="Hi.", start_time=0.0, end_time=1.0),
        SpeakerTurn(speaker_id="B", text="Hello.", start_time=1.0, end_time=2.0),
    ]

    chunks = build_chunks(audio_file_id, turns)

    assert len(chunks) == 2
    assert chunks[0].speaker_id == "A"
    assert chunks[1].speaker_id == "B"


def test_stops_merging_once_target_reached():
    audio_file_id = uuid.uuid4()
    # Each turn is 20s; first turn alone already exceeds MERGE_TARGET_SECONDS (15s),
    # so the second turn must start a new chunk instead of merging.
    turns = [
        SpeakerTurn(speaker_id="A", text="Long turn one.", start_time=0.0, end_time=20.0),
        SpeakerTurn(speaker_id="A", text="Long turn two.", start_time=20.0, end_time=40.0),
    ]

    chunks = build_chunks(audio_file_id, turns)

    assert len(chunks) == 2


def test_splits_long_single_speaker_turn_at_sentence_boundaries():
    audio_file_id = uuid.uuid4()
    long_text = " ".join(f"Sentence number {i} in this long monologue." for i in range(20))
    turns = [SpeakerTurn(speaker_id="A", text=long_text, start_time=0.0, end_time=60.0)]

    chunks = build_chunks(audio_file_id, turns)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.speaker_id == "A"
        # Sentence-boundary splitting is greedy, not a hard cutoff: a chunk
        # may exceed the cap by up to roughly one sentence's duration since
        # we never cut mid-sentence. Assert against a generous tolerance
        # rather than the exact cap.
        assert chunk.end_time - chunk.start_time <= 45.0 * 1.25


def test_prev_next_chunk_linkage():
    audio_file_id = uuid.uuid4()
    turns = [
        SpeakerTurn(speaker_id="A", text="First.", start_time=0.0, end_time=20.0),
        SpeakerTurn(speaker_id="B", text="Second.", start_time=20.0, end_time=40.0),
        SpeakerTurn(speaker_id="A", text="Third.", start_time=40.0, end_time=60.0),
    ]

    chunks = build_chunks(audio_file_id, turns)

    assert chunks[0].prev_chunk_id is None
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].prev_chunk_id == chunks[0].chunk_id
    assert chunks[1].next_chunk_id == chunks[2].chunk_id
    assert chunks[2].next_chunk_id is None


def test_empty_input_returns_empty_list():
    assert build_chunks(uuid.uuid4(), []) == []
