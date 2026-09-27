"""Provider routing from env settings through chat and embedding requests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cuda_sft_qgen import config, keys
from cuda_sft_qgen.config_v2 import load_config_v2
from cuda_sft_qgen.embeddings import NvidiaEmbedder
from cuda_sft_qgen.llm import OpenAILLM
from cuda_sft_qgen.runtime import build_runtime


def _settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in tuple(config.os.environ):
        role_name = name.split("_", 1)[0]
        if (role_name in {"WRITER", "EVOLVER", "REVIEW", "SALVAGE", "KNOWLEDGE", "EMBEDDING"}
                and (name.endswith(("_PROVIDER", "_BASE_URL", "_MODEL")) or "_API_KEY" in name)):
            monkeypatch.delenv(name)
    monkeypatch.setenv("WRITER_PROVIDER", "openai")
    monkeypatch.setenv("WRITER_API_KEY", "chat-secret")
    monkeypatch.setenv("WRITER_BASE_URL", "https://www.poke2api.com/v1/")
    monkeypatch.setenv("WRITER_MODEL", "gpt-6-luna")
    monkeypatch.setenv("REVIEW_MODEL", "gpt-6-sol")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nvidia")
    monkeypatch.setenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
    monkeypatch.setenv("THINKING_LEVEL", "medium")
    return config.load_settings(data_dir=tmp_path)


def test_openai_chat_and_nvidia_embedding_route_separately(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PER_KEY_CONCURRENCY", "4")
    settings = _settings(monkeypatch, tmp_path)
    calls = []

    class MockOpenAI:
        def __init__(self, *, api_key, base_url, **kwargs):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_create))
            self.embeddings = SimpleNamespace(create=self.embed_create)

        def chat_create(self, **kwargs):
            calls.append((self.api_key, self.base_url, "chat", kwargs))
            return [SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=5, total_tokens=9),
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(content='{"ok": true}', reasoning_content=None),
                )],
            )]

        def embed_create(self, **kwargs):
            calls.append((self.api_key, self.base_url, "embed", kwargs))
            return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0])], usage=None)

    monkeypatch.setattr(keys, "OpenAI", MockOpenAI)
    rt = build_runtime(settings)
    try:
        assert isinstance(rt.llm, OpenAILLM)
        assert isinstance(rt.review_llm, OpenAILLM)
        assert rt.review_llm.pool is rt.pool
        assert isinstance(rt.embedder, NvidiaEmbedder)
        assert rt.pool is not rt.embed_key_pool
        assert rt.pool.llm_per_key == 4
        assert rt.embed_key_pool.llm_per_key == 4
        assert rt.llm.complete_json([{"role": "user", "content": "Return JSON"}], retries=0) == {"ok": True}
        assert rt.review_llm.complete_json([{"role": "user", "content": "Return JSON"}], retries=0) == {"ok": True}
        assert np.allclose(rt.embedder.embed(["CUDA test"]), [[0.6, 0.8]])
    finally:
        rt.close()

    chat = calls[0]
    assert chat[:3] == ("chat-secret", "https://www.poke2api.com/v1", "chat")
    assert chat[3]["model"] == "gpt-6-luna"
    assert chat[3]["reasoning_effort"] == "medium"
    assert chat[3]["max_completion_tokens"] == settings.max_tokens
    assert "extra_body" not in chat[3]
    assert "temperature" not in chat[3]
    assert calls[1][:3] == ("chat-secret", "https://www.poke2api.com/v1", "chat")
    assert calls[1][3]["model"] == "gpt-6-sol"
    assert calls[2][:3] == ("embed-secret", "https://integrate.api.nvidia.com/v1", "embed")
    assert calls[2][3]["model"] == "nvidia/nemotron-3-embed-1b"
    assert calls[2][3]["extra_body"] == {"input_type": "passage"}


def test_openai_config_and_missing_key_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("WRITER_GATE_MAX_TOKENS", "2048")
    settings = _settings(monkeypatch, tmp_path)
    typed = load_config_v2(data_dir=tmp_path)
    assert typed.llm.provider == "openai"
    assert typed.llm.model == settings.llm_model == "gpt-6-luna"
    assert typed.agents["review"].model == "gpt-6-sol"
    assert typed.embedding.model == settings.embedding_model
    assert settings.generation["writer_gate_max_tokens"] == 2048
    assert "chat-secret" not in repr(settings)
    assert "chat-secret" not in repr(typed)

    monkeypatch.delenv("WRITER_API_KEY")
    settings = config.load_settings(data_dir=tmp_path)
    assert settings.missing_api_keys == ["WRITER_API_KEY"]
    monkeypatch.delenv("EMBEDDING_API_KEY")
    settings = config.load_settings(data_dir=tmp_path)
    assert settings.missing_api_keys == ["WRITER_API_KEY", "EMBEDDING_API_KEY"]


def test_each_agent_routes_by_explicit_protocol_and_reuses_matching_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in tuple(config.os.environ):
        if name.startswith(("WRITER_", "EVOLVER_", "REVIEW_", "SALVAGE_", "KNOWLEDGE_", "EMBEDDING_")):
            monkeypatch.delenv(name)
    values = {
        "WRITER_PROVIDER": "nvidia", "WRITER_API_KEY": "writer-key",
        "WRITER_BASE_URL": "https://writer.example/v1", "WRITER_MODEL": "nvidia/writer",
        "EVOLVER_PROVIDER": "openai", "EVOLVER_API_KEY": "evolver-key",
        "EVOLVER_BASE_URL": "https://evolver.example/v1", "EVOLVER_MODEL": "nvidia/named-by-gateway",
        "REVIEW_PROVIDER": "nvidia", "REVIEW_API_KEY": "review-key",
        "REVIEW_BASE_URL": "https://review.example/v1", "REVIEW_MODEL": "gpt-named-by-gateway",
        "SALVAGE_MODEL": "salvage-model", "KNOWLEDGE_MODEL": "knowledge-model",
        "EMBEDDING_PROVIDER": "nvidia", "EMBEDDING_API_KEY": "writer-key",
        "EMBEDDING_BASE_URL": "https://writer.example/v1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = config.load_settings(data_dir=tmp_path)
    assert settings.missing_api_keys == settings.missing_api_fields == []
    assert settings.agent_apis["salvage"].api_keys == ["writer-key"]
    calls = []

    class MockOpenAI:
        def __init__(self, *, api_key, base_url, **kwargs):
            self.api_key, self.base_url = api_key, base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append((self.api_key, self.base_url, kwargs))
            return [SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(finish_reason="stop", delta=SimpleNamespace(content="ok", reasoning_content=None))],
            )]

    monkeypatch.setattr(keys, "OpenAI", MockOpenAI)
    rt = build_runtime(settings)
    try:
        assert rt.embed_key_pool is rt.pool
        assert rt.role_pools["salvage"] is rt.role_pools["knowledge"] is rt.pool
        assert rt.role_pools["evolver"] is not rt.role_pools["review"]
        for client in (rt.llm, rt.evolver_llm, rt.review_llm, rt.salvage_llm, rt.knowledge_llm):
            assert client.complete([{"role": "user", "content": "test"}], thinking_level="none") == "ok"
    finally:
        rt.close()
    assert [(key, url, payload["model"]) for key, url, payload in calls] == [
        ("writer-key", "https://writer.example/v1", "nvidia/writer"),
        ("evolver-key", "https://evolver.example/v1", "nvidia/named-by-gateway"),
        ("review-key", "https://review.example/v1", "gpt-named-by-gateway"),
        ("writer-key", "https://writer.example/v1", "salvage-model"),
        ("writer-key", "https://writer.example/v1", "knowledge-model"),
    ]
    assert "reasoning_effort" in calls[1][2]
    assert "extra_body" in calls[2][2]


def test_changed_agent_connection_requires_own_url_and_key(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("WRITER_PROVIDER", "nvidia")
    monkeypatch.setenv("WRITER_BASE_URL", "https://writer.example/v1")
    monkeypatch.setenv("WRITER_API_KEY", "writer-key")
    monkeypatch.setenv("EVOLVER_PROVIDER", "openai")
    monkeypatch.delenv("EVOLVER_BASE_URL", raising=False)
    monkeypatch.delenv("EVOLVER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="EVOLVER_BASE_URL and EVOLVER_API_KEY"):
        config.load_settings(data_dir=tmp_path)


def test_legacy_provider_variables_are_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in tuple(config.os.environ):
        if name.startswith(("WRITER_", "EVOLVER_", "REVIEW_", "SALVAGE_", "KNOWLEDGE_", "EMBEDDING_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "old-nvidia-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://old-nvidia.example/v1")
    settings = config.load_settings(data_dir=tmp_path)
    assert settings.llm_provider == "nvidia"
    assert settings.llm_base_url == ""
    assert settings.missing_api_keys == ["WRITER_API_KEY", "EMBEDDING_API_KEY"]
    assert "WRITER_BASE_URL" in settings.missing_api_fields


def test_fake_mode_allows_unconfigured_role_connections(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("WRITER_PROVIDER", "nvidia")
    monkeypatch.setenv("REVIEW_PROVIDER", "openai")
    monkeypatch.delenv("REVIEW_BASE_URL", raising=False)
    monkeypatch.delenv("REVIEW_API_KEY", raising=False)
    settings = config.load_settings(fake_llm=True, data_dir=tmp_path)
    assert settings.missing_api_keys == settings.missing_api_fields == []
    runtime = build_runtime(settings)
    try:
        assert runtime.pool is None
        assert runtime.review_llm is not None
    finally:
        runtime.close()
