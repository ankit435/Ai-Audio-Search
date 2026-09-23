"""In-memory fake adapters implementing the domain ports, for fast,
dependency-free unit testing of `application` services (no Postgres, no
ML models). Test-only: lives outside `src/` since it's not shipped code.

Per plan "Testability first": every application service must be
constructible with fake/in-memory port implementations.
"""

from __future__ import annotations

import math
import re
from uuid import UUID

from src.domain.models import AudioFile, Chunk, RankedChunk, SpeakerTurn

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder — no ML model, no I/O.

    Good enough to exercise fusion/ranking logic in tests without pulling
    in sentence-transformers; NOT representative of real semantic quality.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokenize(text):
            vec[hash(token) % self._dim] += 1.0
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class FakeTranscriber:
    """Returns pre-canned segments supplied at construction time."""

    def __init__(self, segments: list[SpeakerTurn]) -> None:
        self._segments = segments

    async def transcribe(self, audio_file_path: str) -> list[SpeakerTurn]:
        return list(self._segments)


class FakeDiarizer:
    """Returns pre-canned speaker turns supplied at construction time."""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = turns

    async def diarize(self, audio_file_path: str) -> list[SpeakerTurn]:
        return list(self._turns)


class FakeAudioFileRepository:
    def __init__(self) -> None:
        self._by_id: dict[UUID, AudioFile] = {}
        self._by_checksum: dict[str, AudioFile] = {}

    async def find_by_checksum(self, checksum: str) -> AudioFile | None:
        return self._by_checksum.get(checksum)

    async def save(self, audio_file: AudioFile) -> None:
        self._by_id[audio_file.audio_file_id] = audio_file
        self._by_checksum[audio_file.checksum] = audio_file

    async def get(self, audio_file_id: UUID) -> AudioFile | None:
        return self._by_id.get(audio_file_id)


class FakeChunkRepository:
    """In-memory chunk store with naive keyword-overlap and cosine-similarity
    ranking, standing in for `ts_rank_cd` / pgvector cosine search.
    """

    def __init__(self) -> None:
        self._chunks: dict[UUID, Chunk] = {}

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def search_keyword(self, query_text: str, top_k: int) -> list[RankedChunk]:
        query_tokens = _tokenize(query_text)
        scored = []
        for chunk in self._chunks.values():
            overlap = len(query_tokens & _tokenize(chunk.text))
            if overlap > 0:
                scored.append((chunk.chunk_id, overlap))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RankedChunk(chunk_id=chunk_id, rank=i + 1, raw_score=score)
            for i, (chunk_id, score) in enumerate(scored[:top_k])
        ]

    async def search_semantic(self, query_embedding: list[float], top_k: int) -> list[RankedChunk]:
        scored = []
        for chunk in self._chunks.values():
            if chunk.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk.chunk_id, sim))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RankedChunk(chunk_id=chunk_id, rank=i + 1, raw_score=score)
            for i, (chunk_id, score) in enumerate(scored[:top_k])
        ]

    async def get_by_ids(self, chunk_ids: list[UUID]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]
