"""Domain exceptions — typed, stage-aware failures.

Infra adapters must catch underlying library exceptions and re-raise as
one of these, carrying stage/file context. Application and API layers
only ever see these types, never raw infra exceptions (see plan
"Core Engineering Principles": fail loud, log context).
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all typed domain exceptions in this system."""


class TranscriptionFailedError(DomainError):
    """Raised when a `Transcriber` adapter fails to produce a transcript."""

    def __init__(self, audio_file_id: str, reason: str) -> None:
        self.audio_file_id = audio_file_id
        self.reason = reason
        super().__init__(f"Transcription failed for audio_file_id={audio_file_id}: {reason}")


class DiarizationFailedError(DomainError):
    """Raised when a `Diarizer` adapter fails to produce speaker turns."""

    def __init__(self, audio_file_id: str, reason: str) -> None:
        self.audio_file_id = audio_file_id
        self.reason = reason
        super().__init__(f"Diarization failed for audio_file_id={audio_file_id}: {reason}")


class EmbeddingFailedError(DomainError):
    """Raised when an `Embedder` adapter fails to produce a vector."""

    def __init__(self, context: str, reason: str) -> None:
        self.context = context
        self.reason = reason
        super().__init__(f"Embedding failed ({context}): {reason}")


class ChunkPersistenceError(DomainError):
    """Raised when a `ChunkRepository` fails to save chunks/vectors atomically."""

    def __init__(self, audio_file_id: str, reason: str) -> None:
        self.audio_file_id = audio_file_id
        self.reason = reason
        super().__init__(f"Chunk persistence failed for audio_file_id={audio_file_id}: {reason}")


class AudioFileNotFoundError(DomainError):
    """Raised when a referenced audio file id/path does not exist."""

    def __init__(self, identifier: str) -> None:
        self.identifier = identifier
        super().__init__(f"Audio file not found: {identifier}")


class InvalidQueryError(DomainError):
    """Raised when user-supplied query text fails validation/sanitization."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid search query: {reason}")
