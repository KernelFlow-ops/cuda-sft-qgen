"""Unit tests for salvage_node and retry_drain_node."""

from __future__ import annotations

import importlib
import json
from queue import Queue
from types import SimpleNamespace

import pytest

from cuda_sft_qgen.graph.nodes.generate_core import GenerateOutcome
from cuda_sft_qgen.graph.nodes.salvage import repair_batch_node, salvage_node, retry_drain_node
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.runtime import reset_runtime, set_runtime


def test_llm_salvage_cannot_change_recovered_stem(fake_runtime, sample_brief, monkeypatch):
    core = importlib.import_module("cuda_sft_qgen.graph.nodes.generate_core")
    brief = sample_brief(kind="numeric")
    original = "A CUDA kernel launches 32 threads. Which lane reads index 7?"
    raw = '{"question": ' + json.dumps(original) + ', broken}'
    parse = core.parse_question_payload
    monkeypatch.setattr(core, "parse_question_payload",
                        lambda text, **kwargs: None if text == raw else parse(text, **kwargs))
    called = []
    fake_runtime.salvage_llm = SimpleNamespace(complete=lambda *args, **kwargs: (
        called.append(True) or json.dumps({"question": original + " Changed."})
    ))
    fake_runtime.settings.salvage_use_llm = True
    token = set_runtime(fake_runtime)
    try:
        outcome = core.salvage_item({"brief": brief.model_dump(), "raw": raw,
                                     "writer_real_api": True})
    finally:
        reset_runtime(token)
    assert called
    assert outcome.status == "retry"
    assert fake_runtime.store.load_pending() == []


def test_llm_salvage_skips_unrecoverable_stem(fake_runtime, sample_brief):
    core = importlib.import_module("cuda_sft_qgen.graph.nodes.generate_core")
    fake_runtime.settings.salvage_use_llm = True
    token = set_runtime(fake_runtime)
    try:
        outcome = core.salvage_item({"brief": sample_brief(kind="numeric").model_dump(),
                                     "raw": '{"question":', "writer_real_api": True})
    finally:
        reset_runtime(token)
    assert outcome.status == "retry"
    assert outcome.reason == "salvage_question_unverifiable"


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


def test_retry_lease_survives_handoff_until_repair_finishes(
    fake_runtime, sample_brief, sample_question, monkeypatch
):
    from cuda_sft_qgen.graph.nodes import salvage

    brief = sample_brief().model_dump()
    fake_runtime.store.enqueue_retry({"brief": brief, "reason": "429", "attempts": 1, "next_ok_at": 0})
    original = sample_question(id="q_original")
    repaired = sample_question(id="q_repaired")
    monkeypatch.setattr(salvage, "generate_or_salvage", lambda *a, **k: GenerateOutcome(
        status="repair", question=original, brief=brief, reason="writer_gate"
    ))
    monkeypatch.setattr(salvage, "rewrite_item", lambda item: GenerateOutcome(
        status="ok", question=repaired
    ))
    queue = Queue()
    token = set_runtime(fake_runtime)
    try:
        retry_drain_node({"pending": [], "steering_brief": SteeringBrief().model_dump(),
                          "review_batch_size": 1, "_repair_queue": queue})
        assert fake_runtime.store.retry_queue_size() == 1
        repair_batch_node([queue.get_nowait()])
    finally:
        reset_runtime(token)
    assert fake_runtime.store.retry_queue_size() == 0
    assert [q.id for q in fake_runtime.store.load_pending()] == ["q_repaired"]
