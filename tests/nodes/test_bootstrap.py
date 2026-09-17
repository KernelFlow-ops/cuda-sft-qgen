"""Unit tests for bootstrap_node archive hydration."""

from __future__ import annotations

import pytest

from cuda_sft_qgen.graph.nodes.bootstrap import bootstrap_node
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.runtime import set_runtime


def test_bootstrap_node_empty(fake_runtime, temp_data_dir):
    """Test bootstrap node with empty store."""
    set_runtime(fake_runtime)
    state = {}
    result = bootstrap_node(state)

    assert result["target"] == fake_runtime.settings.target_question_count
    assert result["iteration"] == 0
    assert result["review_rounds"] == 0
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 0
    assert result["pending"] == []
    assert result["done"] is False
    assert "bootstrap" in result["log"][0]


def test_bootstrap_node_with_data(fake_runtime, sample_question):
    """Test bootstrap node with existing data."""
    set_runtime(fake_runtime)

    # Add some accepted and rejected questions
    fake_runtime.store.append_accepted(sample_question(id="q1"))
    fake_runtime.store.append_accepted(sample_question(id="q2"))
    fake_runtime.store.append_rejected(sample_question(id="q3", question=""))

    pending_q = sample_question(id="q4")
    fake_runtime.store.save_pending([pending_q])

    fake_runtime.store.save_run_state({"iteration": 5, "review_rounds": 3})

    state = {}
    result = bootstrap_node(state)

    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 1
    assert len(result["pending"]) == 1
    assert result["iteration"] == 5
    assert result["review_rounds"] == 3
    assert result["done"] is False


def test_bootstrap_node_target_reached(fake_runtime, sample_question):
    """Test bootstrap node when target is reached."""
    set_runtime(fake_runtime)
    fake_runtime.settings.target_question_count = 2

    fake_runtime.store.append_accepted(sample_question(id="q1"))
    fake_runtime.store.append_accepted(sample_question(id="q2"))

    state = {}
    result = bootstrap_node(state)

    assert result["accepted_count"] == 2
    assert result["done"] is True
    assert result["target"] == 2
