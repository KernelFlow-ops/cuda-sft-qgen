"""Unit tests for planner_node batch sizing and skip_plan."""

from __future__ import annotations

import pytest

from cuda_sft_qgen.graph.nodes.planner import planner_node
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.runtime import set_runtime


def test_planner_node_target_reached(fake_runtime, sample_question):
    """Test planner node when target is already reached."""
    set_runtime(fake_runtime)
    fake_runtime.settings.target_question_count = 10

    state = {
        "accepted_count": 10,
        "pending": [],
        "target": 10,
        "review_batch_size": 20,
        "steering_brief": SteeringBrief().model_dump(),
        "iteration": 5,
    }

    result = planner_node(state)

    assert result["done"] is True
    assert result["batch_briefs"] == []
    assert result["status"] == "target_reached"
    assert result["iteration"] == 6


def test_planner_node_basic_plan(fake_runtime, sample_question, sample_steering_brief):
    """Test planner node generates briefs."""
    set_runtime(fake_runtime)
    fake_runtime.settings.target_question_count = 100
    fake_runtime.settings.review_batch_size = 20

    state = {
        "accepted_count": 0,
        "pending": [],
        "target": 100,
        "review_batch_size": 20,
        "steering_brief": sample_steering_brief().model_dump(),
        "iteration": 0,
    }

    result = planner_node(state)

    assert result["done"] is False
    assert len(result["batch_briefs"]) > 0
    assert result["iteration"] == 1
    assert "planned_" in result["status"]


def test_planner_node_with_pending(fake_runtime, sample_question, sample_steering_brief):
    """Test planner node with existing pending questions."""
    set_runtime(fake_runtime)
    fake_runtime.settings.target_question_count = 100
    fake_runtime.settings.review_batch_size = 20

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(15)]

    state = {
        "accepted_count": 0,
        "pending": pending,
        "target": 100,
        "review_batch_size": 20,
        "steering_brief": sample_steering_brief().model_dump(),
        "iteration": 2,
    }

    result = planner_node(state)

    assert result["done"] is False
    fill = 5
    min_batch = fake_runtime.settings.min_generate_batch
    # May overshoot the leftover fill so the writer pool stays busy.
    assert fill <= len(result["batch_briefs"]) <= max(fill, min_batch)
    assert result["iteration"] == 3


def test_planner_node_skip_plan_when_full(fake_runtime, sample_question, sample_steering_brief):
    """Test planner node skips when pending is already full."""
    set_runtime(fake_runtime)
    fake_runtime.settings.target_question_count = 100
    fake_runtime.settings.review_batch_size = 20

    pending = [sample_question(id=f"q{i}").to_jsonl() for i in range(20)]

    state = {
        "accepted_count": 0,
        "pending": pending,
        "target": 100,
        "review_batch_size": 20,
        "steering_brief": sample_steering_brief().model_dump(),
        "iteration": 1,
    }

    result = planner_node(state)

    assert result["batch_briefs"] == []
    assert result["status"] == "skip_plan"
    assert result["done"] is False


def test_planner_node_near_target(fake_runtime, sample_question, sample_steering_brief):
    """Test planner node when close to target."""
    set_runtime(fake_runtime)
    fake_runtime.settings.target_question_count = 100
    fake_runtime.settings.review_batch_size = 20

    state = {
        "accepted_count": 95,
        "pending": [],
        "target": 100,
        "review_batch_size": 20,
        "steering_brief": sample_steering_brief().model_dump(),
        "iteration": 10,
    }

    result = planner_node(state)

    min_batch = fake_runtime.settings.min_generate_batch
    assert 1 <= len(result["batch_briefs"]) <= max(5, min_batch)
    assert result["done"] is False
