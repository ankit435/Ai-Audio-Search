"""Pydantic request/response models for the API layer.

FastAPI derives OpenAPI/Swagger docs from these automatically (see plan
"API Documentation — Swagger/OpenAPI").
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class SearchResultItemResponse(BaseModel):
    chunk_id: UUID
    file_name: str = Field(examples=["interview_01.wav"])
    speaker_id: str = Field(examples=["SPEAKER_00"])
    text_snippet: str
    start_time: float = Field(examples=[42.5])
    end_time: float = Field(examples=[58.1])
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItemResponse]
    took_ms: float


class IngestRequest(BaseModel):
    audio_file_path: str = Field(examples=["/data/audio/interview_01.wav"])


class IngestResponse(BaseModel):
    audio_file_id: UUID
    chunk_count: int
    duration_ms: float
    skipped_existing: bool


class ErrorResponse(BaseModel):
    error_type: str
    message: str


class FeedbackRequest(BaseModel):
    """Stretch Goal — relevance feedback payload for `POST /search/feedback`."""

    query_text: str = Field(examples=["what are the onboarding steps"])
    chunk_id: UUID
    rank_shown: int = Field(ge=1, examples=[1])
    signal: str = Field(examples=["click", "thumbs_up", "thumbs_down", "dwell_time_ms"])


class SearchAnswerResponse(BaseModel):
    """Stretch Goal — extractive RAG-style answer for `POST /search/answer`."""

    query: str
    answer: str
    citations: list[SearchResultItemResponse]
