"""Ingestion pipeline application service.

Depends only on domain ports (Transcriber, Diarizer, Embedder,
ChunkRepository, AudioFileRepository) — never imports infra modules
directly. Concrete adapters are injected via the constructor by the
composition root (`src/api/main.py` or a `container.py`).

Idempotent per audio file (checksum-based dedup): if an `AudioFile` with
the same checksum already exists, ingestion is a no-op and the existing
record is returned.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, replace

from src.application.chunking import build_chunks
from src.domain.exceptions import ChunkPersistenceError, EmbeddingFailedError
from src.domain.models import AudioFile, SpeakerTurn
from src.domain.ports import AudioFileRepository, ChunkRepository, Diarizer, Embedder, Transcriber
from src.infra.logging_config import log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    audio_file_id: uuid.UUID
    chunk_count: int
    duration_ms: float
    skipped_existing: bool


def _checksum_file(audio_file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(audio_file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _with_embedding(chunk, embedding: list[float]):
    """Chunk is a frozen dataclass; return a copy with `embedding` attached."""
    return replace(chunk, embedding=embedding)


def _align_transcript_to_speakers(
    transcript_segments: list[SpeakerTurn], diarization_turns: list[SpeakerTurn]
) -> list[SpeakerTurn]:
    """Assign each transcript segment the speaker whose diarized turn has
    the greatest time overlap with it. If diarization has no overlapping
    turn (e.g. silence-only segment misclassified), falls back to the
    nearest diarized turn by start_time.
    """
    if not diarization_turns:
        return transcript_segments

    aligned: list[SpeakerTurn] = []
    for seg in transcript_segments:

        def overlap(turn: SpeakerTurn) -> float:
            return max(0.0, min(seg.end_time, turn.end_time) - max(seg.start_time, turn.start_time))

        best_turn = max(diarization_turns, key=overlap)
        if overlap(best_turn) <= 0.0:
            best_turn = min(diarization_turns, key=lambda t: abs(t.start_time - seg.start_time))

        aligned.append(
            SpeakerTurn(
                speaker_id=best_turn.speaker_id,
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
            )
        )
    return aligned


class IngestPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        diarizer: Diarizer,
        embedder: Embedder,
        chunk_repository: ChunkRepository,
        audio_file_repository: AudioFileRepository,
    ) -> None:
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._embedder = embedder
        self._chunk_repository = chunk_repository
        self._audio_file_repository = audio_file_repository

    async def run(self, audio_file_path: str) -> IngestResult:
        pipeline_start = time.perf_counter()
        checksum = _checksum_file(audio_file_path)

        existing = await self._audio_file_repository.find_by_checksum(checksum)
        if existing is not None:
            log_event(
                logger, logging.INFO, "ingest.skipped_existing",
                audio_file_id=str(existing.audio_file_id), checksum=checksum,
            )
            return IngestResult(
                audio_file_id=existing.audio_file_id, chunk_count=0, duration_ms=0.0, skipped_existing=True,
            )

        audio_file = AudioFile(
            audio_file_id=uuid.uuid4(),
            file_name=os.path.basename(audio_file_path),
            file_path=audio_file_path,
            checksum=checksum,
        )
        file_id = str(audio_file.audio_file_id)

        stage_start = time.perf_counter()
        transcript_segments = await self._transcriber.transcribe(audio_file_path)
        log_event(
            logger, logging.INFO, "ingest.transcribe.end", audio_file_id=file_id,
            duration_ms=(time.perf_counter() - stage_start) * 1000, segment_count=len(transcript_segments),
        )

        stage_start = time.perf_counter()
        diarization_turns = await self._diarizer.diarize(audio_file_path)
        speaker_count = len({t.speaker_id for t in diarization_turns})
        log_event(
            logger, logging.INFO, "ingest.diarize.end", audio_file_id=file_id,
            duration_ms=(time.perf_counter() - stage_start) * 1000, speaker_count=speaker_count,
        )

        aligned_turns = _align_transcript_to_speakers(transcript_segments, diarization_turns)

        stage_start = time.perf_counter()
        chunks = build_chunks(audio_file.audio_file_id, aligned_turns)
        log_event(
            logger, logging.INFO, "ingest.chunk.end", audio_file_id=file_id,
            duration_ms=(time.perf_counter() - stage_start) * 1000, chunk_count=len(chunks),
        )

        stage_start = time.perf_counter()
        try:
            embeddings = await self._embedder.embed_batch([c.text for c in chunks])
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception
            raise EmbeddingFailedError(context=f"audio_file_id={file_id}", reason=str(exc)) from exc
        chunks = [
            _with_embedding(chunk, embedding) for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        log_event(
            logger, logging.INFO, "ingest.embed.end", audio_file_id=file_id,
            duration_ms=(time.perf_counter() - stage_start) * 1000, chunk_count=len(chunks),
        )

        stage_start = time.perf_counter()
        try:
            await self._audio_file_repository.save(audio_file)
            await self._chunk_repository.save_chunks(chunks)
        except Exception as exc:  # noqa: BLE001 - re-raised as typed domain exception
            raise ChunkPersistenceError(audio_file_id=file_id, reason=str(exc)) from exc
        log_event(
            logger, logging.INFO, "ingest.index.end", audio_file_id=file_id,
            duration_ms=(time.perf_counter() - stage_start) * 1000, chunk_count=len(chunks),
        )

        total_duration_ms = (time.perf_counter() - pipeline_start) * 1000
        return IngestResult(
            audio_file_id=audio_file.audio_file_id,
            chunk_count=len(chunks),
            duration_ms=total_duration_ms,
            skipped_existing=False,
        )
