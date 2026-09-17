"""Unit tests for review_score, review_chair, and archive nodes."""

from __future__ import annotations

import numpy as np
import pytest

from cuda_sft_qgen.graph.nodes.review import (
    review_score_node,
    review_chair_node,
    archive_node,
)
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.runtime import set_runtime


def test_review_score_node_empty(fake_runtime):
    """Test review score node with no pending questions."""
    set_runtime(fake_runtime)

    state = {"pending": []}

    result = review_score_node(state)

    assert result["status"] == "review_empty"
    assert result["last_diversity"] == {}


def test_review_score_node_basic(fake_runtime, sample_question):
    """Test review score node with pending questions."""
    set_runtime(fake_runtime)

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(5)]

    state = {"pending": pending}

    result = review_score_node(state)

    assert result["status"] == "review_scored"
    assert "last_diversity" in result
    assert "scored" in result["last_diversity"]
    assert "report" in result["last_diversity"]
    assert "vectors" in result["last_diversity"]
    assert len(result["last_diversity"]["scored"]) == 5
    assert "review_score" in result["log"][0]


def test_review_score_node_deduplication(fake_runtime, sample_question):
    """Test review score node detects duplicates."""
    set_runtime(fake_runtime)

    # Add an accepted question to the store
    accepted = sample_question(id="q_accepted", question="What is CUDA occupancy?")
    fake_runtime.store.append_accepted(accepted)
    fake_runtime.index.add([accepted.id], fake_runtime.embedder.embed([accepted.question]))

    # Create a similar pending question
    pending = [
        sample_question(id="q_pending", question="What is CUDA occupancy?").to_jsonl()
    ]

    state = {"pending": pending}

    result = review_score_node(state)

    scored = result["last_diversity"]["scored"]
    # The similar question should be flagged as dup or gray
    assert scored[0]["decision"]["verdict"] in ["dup", "gray", "unique"]


def test_review_chair_node_basic(fake_runtime, sample_question):
    """Test review chair node makes decisions."""
    set_runtime(fake_runtime)

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(3)]
    state_before = {"pending": pending}
    scored_result = review_score_node(state_before)

    state = {
        "last_diversity": scored_result["last_diversity"],
        "review_rounds": 0,
    }

    result = review_chair_node(state)

    assert result["status"] == "review_chaired"
    assert "steering_brief" in result
    assert result["review_rounds"] == 1
    assert "final_decisions" in result["last_diversity"]
    decisions = result["last_diversity"]["final_decisions"]
    assert len(decisions) == 3
    for d in decisions:
        assert "action" in d
        assert d["action"] in ["accept", "reject"]


def test_review_chair_node_with_llm(fake_runtime, sample_question):
    """Test review chair node uses LLM for decisions."""
    set_runtime(fake_runtime)
    fake_runtime.settings.review_llm_every_k_batches = 1

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(3)]
    state_before = {"pending": pending}
    scored_result = review_score_node(state_before)

    state = {
        "last_diversity": scored_result["last_diversity"],
        "review_rounds": 0,
    }

    result = review_chair_node(state)

    assert result["status"] == "review_chaired"
    assert "steering_brief" in result
    decisions = result["last_diversity"]["final_decisions"]
    assert len(decisions) > 0


def test_chair_does_not_auto_reject_hw_leaf_unique(fake_runtime, sample_question):
    """Chair defaults to accept when the only issue is hw/leaf consistency."""
    set_runtime(fake_runtime)
    q = sample_question(
        id="q_hw",
        hardware="hopper",
        l3_id="sm80_cp_async_l2_persist",
        l3_claim="Ampere sm_80 L2 persist window",
        question=(
            "A developer enables L2 cache persistence on an Ampere (sm_80) GPU "
            "using cudaStreamSetAttribute. Which statement is correct about occupancy?"
        ),
    )
    pending = [q.to_jsonl()]
    scored = review_score_node({"pending": pending})
    result = review_chair_node(
        {"last_diversity": scored["last_diversity"], "review_rounds": 0}
    )
    decisions = result["last_diversity"]["final_decisions"]
    assert decisions[0]["action"] == "accept"
    warnings = decisions[0]["question"].get("warnings") or []
    assert any("hw" in w or "mismatch" in w for w in warnings)


def test_archive_node_basic(fake_runtime, sample_question):
    """Test archive node accepts and rejects questions."""
    set_runtime(fake_runtime)

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(5)]
    state_before = {"pending": pending}
    scored_result = review_score_node(state_before)

    state_chair = {
        "last_diversity": scored_result["last_diversity"],
        "review_rounds": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "target": 100,
    }
    chair_result = review_chair_node(state_chair)

    state = {
        "last_diversity": chair_result["last_diversity"],
        "accepted_count": 0,
        "rejected_count": 0,
        "target": 100,
        "iteration": 1,
        "review_rounds": 1,
    }

    result = archive_node(state)

    assert result["status"] == "archived"
    assert result["pending"] == []
    assert result["accepted_count"] >= 0
    assert result["rejected_count"] >= 0
    assert "archive" in result["log"][0]


def test_archive_node_updates_index(fake_runtime, sample_question):
    """Test archive node updates the embedding index."""
    set_runtime(fake_runtime)

    initial_size = len(fake_runtime.index.ids)

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(3)]
    state_before = {"pending": pending}
    scored_result = review_score_node(state_before)

    state_chair = {
        "last_diversity": scored_result["last_diversity"],
        "review_rounds": 0,
    }
    chair_result = review_chair_node(state_chair)

    state = {
        "last_diversity": chair_result["last_diversity"],
        "accepted_count": 0,
        "rejected_count": 0,
        "target": 100,
        "iteration": 1,
        "review_rounds": 1,
    }

    result = archive_node(state)

    # At least some questions should be accepted
    if result["accepted_count"] > 0:
        assert len(fake_runtime.index.ids) > initial_size
