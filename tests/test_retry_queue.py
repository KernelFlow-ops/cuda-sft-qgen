"""Tests for Archive retry-queue lease, due items, and cooldown."""

from __future__ import annotations

import time
from pathlib import Path

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.store.archive import Archive


def test_retry_due_items_roundtrip(tmp_path: Path):
    store = Archive(tmp_path)
    store.enqueue_retry(
        {"brief": {"l3_id": "a"}, "reason": "429", "attempts": 1, "next_ok_at": 0}
    )
    store.enqueue_retry(
        {
            "brief": {"l3_id": "b"},
            "reason": "429",
            "attempts": 1,
            "next_ok_at": time.time() + 3600,
        }
    )
    assert store.retry_queue_size() == 2
    assert store.due_retry_count() == 1
    due = store.due_retry_items(limit=10)
    assert len(due) == 1
    assert due[0]["brief"]["l3_id"] == "a"
    assert store.retry_queue_size() == 1


def test_settings_hyperparams_from_env(monkeypatch):
    monkeypatch.setenv("SALVAGE_USE_LLM", "0")
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("JSON_KEEP_TRUNCATED", "0")
    settings = load_settings(fake_llm=True)
    assert settings.salvage_use_llm is False
    assert settings.retry_max_attempts == 3
    assert settings.json_keep_truncated is False
