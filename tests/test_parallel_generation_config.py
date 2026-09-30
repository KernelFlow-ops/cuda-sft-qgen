"""Environment controls for parallel writer requests."""

from __future__ import annotations

import threading
import time

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


def test_writer_submission_is_bounded_and_preserves_runtime(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph.nodes import generate
    from cuda_sft_qgen.graph.nodes.generate_core import GenerateOutcome
    from cuda_sft_qgen.runtime import get_runtime

    first_started = threading.Event()
    release_first = threading.Event()
    queue_filled = threading.Event()
    submitted = []
    errors = []
    seen = []
    real_executor = generate.ThreadPoolExecutor

    class TrackingPool(real_executor):
        def submit(self, *args, **kwargs):
            submitted.append(args)
            if len(submitted) == 2:
                queue_filled.set()
            return super().submit(*args, **kwargs)

    def generate_one(brief, steering, **kwargs):
        assert get_runtime() is fake_runtime
        seen.append(brief["l3_id"])
        if brief["l3_id"] == "bounded_0":
            first_started.set()
            assert release_first.wait(3)
        return GenerateOutcome(status="hard_fail", brief=brief)

    monkeypatch.setattr(generate, "ThreadPoolExecutor", TrackingPool)
    monkeypatch.setattr(generate, "generate_or_salvage", generate_one)
    monkeypatch.setattr(generate, "apply_outcome", lambda *args, **kwargs: "hard_fail")
    fake_runtime.settings.writer_concurrency = 1
    fake_runtime.settings.generation["priority_queue_size"] = 1
    fake_runtime.settings.generation["submit_stagger_sec"] = 0
    briefs = [sample_brief(l3_id=f"bounded_{i}").model_dump() for i in range(4)]

    def run_batch():
        token = set_runtime(fake_runtime)
        try:
            generate_batch_node({"batch_briefs": briefs, "steering_brief": {}, "pending": []})
        except Exception as exc:
            errors.append(exc)
        finally:
            reset_runtime(token)

    thread = threading.Thread(target=run_batch)
    thread.start()
    try:
        assert first_started.wait(2)
        assert queue_filled.wait(2)
        time.sleep(0.05)
        assert len(submitted) == 2
    finally:
        release_first.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert len(submitted) == 4
    assert set(seen) == {f"bounded_{i}" for i in range(4)}
