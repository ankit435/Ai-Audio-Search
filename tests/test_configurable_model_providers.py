"""Tests for the multi-provider configuration surface added for
"configurable models" (NVIDIA NIM / other Hugging Face models / any
OpenAI-compatible endpoint): `Settings` field defaults + overrides, the
`_build_embedder` provider-selection factory, and
`OpenAICompatibleEmbedder`'s fail-fast behavior when no API key is
available. No network calls are made — key presence/absence and
provider routing are exercised directly.
"""

from __future__ import annotations

import pytest

from src.api.main import _build_embedder
from src.api.settings import Settings
from src.domain.exceptions import EmbeddingFailedError
from src.infra.openai_compatible_embedder import OpenAICompatibleEmbedder
from src.infra.sentence_transformer_embedder import SentenceTransformerEmbedder


def test_settings_defaults_use_local_sentence_transformers_provider():
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "sentence-transformers"
    assert settings.embedding_base_url is None
    assert settings.transcription_primary_backend == "faster-whisper"
    assert settings.transcription_fallback_backend == "openai-whisper"
    assert settings.answer_model == "meta/muse-glimmer-30b"
    assert settings.answer_base_url == "https://integrate.api.nvidia.com/v1"


def test_settings_can_be_overridden_for_a_hosted_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("AUDIO_SEARCH_EMBEDDING_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AUDIO_SEARCH_EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("AUDIO_SEARCH_EMBEDDING_MODEL_NAME", "nvidia/nv-embedqa-e5-v5")
    monkeypatch.setenv("AUDIO_SEARCH_ANSWER_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("AUDIO_SEARCH_ANSWER_MODEL", "meta/llama-3.1-8b-instruct")

    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "openai-compatible"
    assert settings.embedding_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.embedding_model_name == "nvidia/nv-embedqa-e5-v5"
    assert settings.answer_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.answer_model == "meta/llama-3.1-8b-instruct"


def test_build_embedder_defaults_to_sentence_transformers():
    settings = Settings(_env_file=None)
    embedder = _build_embedder(settings)
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_build_embedder_selects_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    settings = Settings(_env_file=None, embedding_provider="openai-compatible", embedding_base_url="https://example.test/v1")
    embedder = _build_embedder(settings)
    assert isinstance(embedder, OpenAICompatibleEmbedder)


def test_openai_compatible_embedder_fails_fast_without_any_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(EmbeddingFailedError):
        OpenAICompatibleEmbedder(model_name="whatever")


def test_openai_compatible_embedder_accepts_nvidia_api_key_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    embedder = OpenAICompatibleEmbedder(model_name="nvidia/nv-embedqa-e5-v5", base_url="https://integrate.api.nvidia.com/v1")
    assert embedder is not None  # construction succeeds; no network call made
