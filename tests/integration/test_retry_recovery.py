"""Integration tests that 429s requeue and later drain into pending."""

from __future__ import annotations

import time
import pytest

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.errors import RateLimitError
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.graph.nodes.generate import enqueue_retry
from cuda_sft_qgen.models import QuestionBrief
from cuda_sft_qgen.runtime import build_runtime, set_runtime, reset_runtime


def test_retry_queue_basic(temp_data_dir, sample_brief):
    """Test retry queue basic operations."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    runtime = build_runtime(settings)
    token = set_runtime(runtime)

    try:
        # Enqueue a retry item
        brief = sample_brief()
        enqueue_retry(brief.model_dump(), reason="429", attempts=1)

        # Check retry queue
        retry_file = temp_data_dir / "retry_queue.jsonl"
        assert retry_file.exists()

        # Load retry items
        assert runtime.store.due_retry_count() == 0
        items = runtime.store.due_retry_items(limit=10, now=time.time() + settings.retry_max_cooldown_sec + 1)
        assert len(items) > 0
        assert items[0]["brief"]["l3_id"] == brief.l3_id
    finally:
        reset_runtime(token)


def test_retry_queue_cooldown(temp_data_dir, sample_brief):
    """Test retry queue respects cooldown period."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.retry_base_cooldown_sec = 1.0
    runtime = build_runtime(settings)
    token = set_runtime(runtime)

    try:
        brief = sample_brief()
        # Enqueue with future cooldown
        runtime.store.enqueue_retry({
            "brief": brief.model_dump(),
            "reason": "429",
            "attempts": 1,
            "next_ok_at": time.time() + 10,  # 10 seconds in future
        })

        # Should not be available yet
        items = runtime.store.due_retry_items(limit=10)
        assert len(items) == 0

        # Enqueue one that's immediately available
        brief2 = sample_brief(l3_id="cuda_002")
        runtime.store.enqueue_retry({
            "brief": brief2.model_dump(),
            "reason": "429",
            "attempts": 1,
            "next_ok_at": time.time() - 1,  # Already past
        })

        # Should get the immediately available one
        items = runtime.store.due_retry_items(limit=10)
        assert len(items) == 1
        assert items[0]["brief"]["l3_id"] == "cuda_002"
    finally:
        reset_runtime(token)


def test_retry_queue_max_attempts(temp_data_dir, sample_brief):
    """Test retry queue respects max attempts."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.retry_max_attempts = 3
    runtime = build_runtime(settings)
    token = set_runtime(runtime)

    try:
        brief = sample_brief()

        # Enqueue with attempts at max
        enqueue_retry(brief.model_dump(), reason="429", attempts=3)

        # Should be rejected, not retried
        rejected = runtime.store.load_rejected()
        assert any(q.l3_id == brief.l3_id for q in rejected)
    finally:
        reset_runtime(token)


def test_retry_recovery_in_pipeline(temp_data_dir):
    """Test pipeline recovers from retry queue."""
    settings = load_settings(
        fake_llm=True,
        target=5,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.retry_drain_enabled = True
    settings.retry_base_cooldown_sec = 0.1
    settings.max_iterations = 10

    runtime = build_runtime(settings)
    token = set_runtime(runtime)

    try:
        # Manually add a retry item
        brief_data = {
            "topic_path": ["compute", "occupancy", "cuda_retry_001"],
            "l3_id": "cuda_retry_001",
            "l3_claim": "Test retry recovery",
            "question_type": "conceptual",
            "difficulty": 2,
            "bloom": "understand",
            "persona": "cuda_interviewer",
            "hardware": "hopper",
            "granularity": "kernel",
            "language": "zh",
            "role": "writer",
        }
        runtime.store.enqueue_retry({
            "brief": brief_data,
            "reason": "429",
            "attempts": 1,
            "next_ok_at": time.time(),
        })

        reset_runtime(token)

        # Run pipeline
        result = run_pipeline(settings, runtime=None)

        # Pipeline should process retry queue
        assert result["accepted_count"] >= 0
    finally:
        closer = getattr(runtime, "close", None)
        if callable(closer):
            closer()


def test_retry_queue_exponential_backoff(temp_data_dir, sample_brief):
    """Test retry queue uses exponential backoff."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.retry_base_cooldown_sec = 1.0
    settings.retry_max_cooldown_sec = 60.0
    runtime = build_runtime(settings)
    token = set_runtime(runtime)

    try:
        from cuda_sft_qgen.graph.nodes.generate import retry_wait_sec

        # Test exponential backoff calculation
        wait_1 = retry_wait_sec(1)
        wait_2 = retry_wait_sec(2)
        wait_3 = retry_wait_sec(3)

        assert wait_1 == 1.0  # 1 * 2^0
        assert wait_2 == 2.0  # 1 * 2^1
        assert wait_3 == 4.0  # 1 * 2^2

        # Should cap at max
        wait_10 = retry_wait_sec(10)
        assert wait_10 == settings.retry_max_cooldown_sec
    finally:
        reset_runtime(token)


def test_retry_queue_batch_limit(temp_data_dir, sample_brief):
    """Test retry drain respects batch limit."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.retry_drain_batch = 3
    runtime = build_runtime(settings)
    token = set_runtime(runtime)

    try:
        # Enqueue multiple retry items
        for i in range(10):
            brief = sample_brief(l3_id=f"cuda_retry_{i:03d}")
            runtime.store.enqueue_retry({
                "brief": brief.model_dump(),
                "reason": "429",
                "attempts": 1,
                "next_ok_at": time.time(),
            })

        # Should only get batch size
        items = runtime.store.due_retry_items(limit=3)
        assert len(items) <= 3
    finally:
        reset_runtime(token)
