"""Tests for config profiles and CLI > env > YAML precedence."""

from __future__ import annotations

import os
import pytest

from cuda_sft_qgen import config
from cuda_sft_qgen.config_v2 import load_config_v2


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Clear NVIDIA/profile env vars so profile YAML is the only override."""
    root = config.find_project_root()
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for key in tuple(os.environ):
        if key.startswith(("NVIDIA_", "CUDA_SFT_", "KNOWLEDGE_")) or key in {
            "REVIEW_MODEL", "REVIEW_BATCH_SIZE", "TEMPERATURE", "TOP_P", "WRITER_CONCURRENCY",
        }:
            monkeypatch.delenv(key)
    monkeypatch.setenv("CUDA_SFT_DATA_DIR", str(tmp_path / "outputs"))
    return root


def test_profile_is_identical_for_runtime_and_diagnostics(isolated_config, monkeypatch):
    monkeypatch.setenv("CUDA_SFT_PROFILE", "test")
    legacy = config.load_settings(fake_llm=True)
    typed = load_config_v2(fake_llm=True)
    assert legacy.review_batch_size == typed.review.batch_size == 5
    assert legacy.writer_concurrency == typed.planner.writer_concurrency == 2
    assert legacy.taxonomy_manager_cfg["enabled"] is typed.taxonomy.manager_enabled is False
    assert legacy.data_dir == typed.data_dir
    assert not legacy.data_dir.exists()
    assert typed.default_yaml == legacy.default_yaml
    assert typed.quotas == legacy.quotas
    assert typed.personas == legacy.personas
    assert legacy.review["top_k"] == 8
    assert legacy.taxonomy_manager_cfg["leaf_cosine_dup"] == .92


def test_cli_environment_and_review_model_precedence(isolated_config, monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_BATCH_SIZE", "7")
    monkeypatch.setenv("TEMPERATURE", "0.41")
    monkeypatch.setenv("REVIEW_MODEL", "independent-reviewer")
    legacy = config.load_settings(profile="test", review_batch_size=9, target=12, data_dir=tmp_path / "cli")
    typed = load_config_v2(profile="test", target=12, data_dir=tmp_path / "cli")
    assert legacy.review_batch_size == 9
    assert typed.review.batch_size == 7
    assert legacy.temperature == typed.llm.temperature == .41
    assert legacy.review_model == typed.review.model == "independent-reviewer"
    assert legacy.target_question_count == typed.target_question_count == 12
    assert legacy.data_dir == typed.data_dir == tmp_path / "cli"
    assert load_config_v2(profile="test", review_batch_size=9).review.batch_size == 9


def test_secrets_hidden_and_knowledge_config_shared(isolated_config, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-for-test-only")
    monkeypatch.setenv("KNOWLEDGE_AUDIT_BATCH_SIZE", "13")
    legacy = config.load_settings()
    typed = load_config_v2()
    assert "secret-for-test-only" not in repr(legacy)
    assert "secret-for-test-only" not in repr(typed)
    assert legacy.knowledge_cfg == typed.knowledge
    assert typed.knowledge["audit_batch_size"] == 13


def test_unknown_profile_fails_before_creating_directory(isolated_config):
    with pytest.raises(ValueError, match="Unknown configuration profile"):
        config.load_settings(profile="missing-audit-profile")


def test_dotenv_loaded_for_both_entrypoints(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "load_dotenv", lambda path, override: calls.append((path, override)))
    config.load_settings(fake_llm=True, data_dir=tmp_path)
    load_config_v2(fake_llm=True, data_dir=tmp_path)
    assert len(calls) == 2
    assert all(path.name == ".env" and override is False for path, override in calls)
