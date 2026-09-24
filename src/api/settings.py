"""Env-backed application settings — read once at the composition root.

Never read env vars directly inside `application`/`domain` services; they
receive configuration only via this object (or constructor params derived
from it), consistent with the plan's DI/composition-root approach.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_SEARCH_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://audio_search:audio_search@localhost:5432/audio_search"
    # --- Embedding model (semantic search) ---
    # `sentence-transformers`: runs any local/HF sentence-transformers model
    # (default). `openai-compatible`: calls a hosted/self-hosted
    # `/v1/embeddings` endpoint instead — NVIDIA NIM, OpenAI, vLLM, TEI, etc.
    # See `src/infra/openai_compatible_embedder.py` for provider examples.
    embedding_provider: str = "sentence-transformers"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_base_url: str | None = None  # only used when embedding_provider="openai-compatible"

    # --- Transcription (Whisper) ---
    # `backend` selects the underlying library; `model_size` is passed through
    # as-is, so it also accepts a local path or HF repo id for a converted
    # model (e.g. a CTranslate2-converted NVIDIA/community Whisper variant)
    # when using the `faster-whisper` backend.
    whisper_model_size: str = "base"
    transcription_primary_backend: str = "faster-whisper"
    transcription_fallback_backend: str = "openai-whisper"

    # --- Answer generation (`/search/answer` LLM mode, optional) ---
    # Any OpenAI-compatible chat-completions endpoint works here — NVIDIA
    # NIM, OpenAI, vLLM/Ollama/local servers, etc. Leave `answer_base_url`
    # unset for hosted OpenAI. See `src/infra/openai_answer_generator.py`.
    answer_model: str = "meta/muse-glimmer-30b"
    answer_base_url: str | None = "https://integrate.api.nvidia.com/v1"
    openai_api_key: str | None = None
    nvidia_api_key: str | None = None

    log_level: str = "INFO"
    default_top_k: int = 10
    rrf_k: int = 60
