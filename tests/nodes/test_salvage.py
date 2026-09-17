"""Unit tests for salvage_node and retry_drain_node."""

from __future__ import annotations

import pytest

from cuda_sft_qgen.graph.nodes.salvage import salvage_node, retry_drain_node
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.runtime import set_runtime


def test_salvage_node_empty(fake_runtime):
    """Test salvage node with no items."""
    set_runtime(fake_runtime)

    state = {
        "salvage_items": [],
        "pending": [],
    }

    result = salvage_node(state)

    assert result["status"] == "salvage_skipped"
    assert result["salvage_items"] == []
    assert result["pending"] == []


def test_salvage_node_disabled(fake_runtime, sample_brief):
    """Test salvage node when disabled."""
    set_runtime(fake_runtime)
    fake_runtime.settings.salvage_enabled = False

    salvage_items = [
        {
            "brief": sample_brief().model_dump(),
            "raw": '{"question": "incomplete',
        }
    ]

    state = {
        "salvage_items": salvage_items,
        "pending": [],
    }

    result = salvage_node(state)

    assert result["status"] == "salvage_skipped"
    assert result["salvage_items"] == []


def test_salvage_node_basic(fake_runtime, sample_brief):
    """Test salvage node recovers questions."""
    set_runtime(fake_runtime)
    fake_runtime.settings.salvage_enabled = True
    fake_runtime.settings.salvage_use_llm = False

    brief = sample_brief()
    salvage_items = [
        {
            "brief": brief.model_dump(),
            "raw": '{"question": "What is CUDA?", "tests_claim": "' + brief.l3_claim + '", "language": "zh"}',
        }
    ]

    state = {
        "salvage_items": salvage_items,
        "pending": [],
    }

    result = salvage_node(state)

    # Should either recover or retry
    assert result["salvage_items"] == []
    assert "salvage" in result["status"]


def test_retry_drain_node_empty(fake_runtime):
    """Test retry drain node with no retry items."""
    set_runtime(fake_runtime)

    state = {
        "pending": [],
        "steering_brief": SteeringBrief().model_dump(),
        "review_batch_size": 20,
    }

    result = retry_drain_node(state)

    assert result["status"] == "retry_drain_empty"
    assert result["pending"] == []


def test_retry_drain_node_disabled(fake_runtime):
    """Test retry drain node when disabled."""
    set_runtime(fake_runtime)
    fake_runtime.settings.retry_drain_enabled = False

    state = {
        "pending": [],
        "steering_brief": SteeringBrief().model_dump(),
        "review_batch_size": 20,
    }

    result = retry_drain_node(state)

    assert result["status"] == "retry_drain_skipped"


def test_retry_drain_node_full_pending(fake_runtime, sample_question):
    """Test retry drain node when pending is full."""
    set_runtime(fake_runtime)
    fake_runtime.settings.review_batch_size = 20

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(20)]

    state = {
        "pending": pending,
        "steering_brief": SteeringBrief().model_dump(),
        "review_batch_size": 20,
    }

    result = retry_drain_node(state)

    assert result["status"] == "retry_drain_full"
    assert len(result["pending"]) == 20


def test_retry_drain_node_with_retry_items(fake_runtime, sample_brief, sample_steering_brief):
    """Test retry drain node processes retry items."""
    set_runtime(fake_runtime)
    fake_runtime.settings.retry_drain_enabled = True

    # Add a retry item to the store
    brief = sample_brief()
    fake_runtime.store.enqueue_retry({
        "brief": brief.model_dump(),
        "reason": "429",
        "attempts": 1,
        "next_ok_at": 0,  # Immediately available
    })

    state = {
        "pending": [],
        "steering_brief": sample_steering_brief().model_dump(),
        "review_batch_size": 20,
    }

    result = retry_drain_node(state)

    assert result["status"] == "retry_drained"
    # Should have attempted to process the retry item
    assert "retry_drain" in result["log"][0]
