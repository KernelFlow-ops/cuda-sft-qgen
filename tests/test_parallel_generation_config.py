"""Environment controls for parallel writer requests."""

from __future__ import annotations

import threading

import pytest

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.graph.nodes.generate import generate_batch_node
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.runtime import reset_runtime, set_runtime


def test_parallel_writer_settings_from_env(monkeypatch):
    monkeypatch.setenv("WRITER_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_PER_KEY_CONCURRENCY", "4")
    monkeypatch.setenv("WRITER_SUBMIT_STAGGER_SEC", "0")
    monkeypatch.setenv("MIN_GENERATE_BATCH", "4")
    settings = load_settings(fake_llm=True)
    assert settings.writer_concurrency == 4
    assert settings.llm_per_key_concurrency == 4
    assert settings.generation["submit_stagger_sec"] == 0
    assert settings.min_generate_batch == 4


def test_negative_writer_stagger_is_rejected(monkeypatch):
    monkeypatch.setenv("WRITER_SUBMIT_STAGGER_SEC", "-0.1")
    with pytest.raises(ValueError, match="writer_submit_stagger_sec"):
        load_settings(fake_llm=True)


def test_four_writer_calls_overlap(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph.nodes import generate
    from cuda_sft_qgen.graph.nodes.generate_core import GenerateOutcome

    barrier = threading.Barrier(4)
    calls = []

    def generate_one(brief, steering, **kwargs):
        calls.append(1)
        barrier.wait(timeout=3)
        return GenerateOutcome(status="hard_fail", brief=brief)

    monkeypatch.setattr(generate, "generate_or_salvage", generate_one)
    fake_runtime.settings.writer_concurrency = 4
    fake_runtime.settings.generation["submit_stagger_sec"] = 0
    briefs = [sample_brief(l3_id=f"parallel_{i}").model_dump() for i in range(4)]
    token = set_runtime(fake_runtime)
    try:
        generate_batch_node({"batch_briefs": briefs, "steering_brief": SteeringBrief().model_dump(), "pending": []})
    finally:
        reset_runtime(token)
    assert len(calls) == 4
