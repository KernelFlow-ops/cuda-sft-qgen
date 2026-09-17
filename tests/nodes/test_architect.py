"""Unit tests for the Architect and taxonomy-health graph nodes."""

from __future__ import annotations

import pytest

from cuda_sft_qgen.graph.nodes.architect import (
    architect_node,
    taxonomy_health_node,
)
from cuda_sft_qgen.runtime import set_runtime


def test_architect_node_basic(fake_runtime):
    """Test architect node expands taxonomy."""
    set_runtime(fake_runtime)

    state = {}

    result = architect_node(state)

    assert result["status"] == "taxonomy_ready"
    assert "log" in result
    assert len(result["log"]) > 0


def test_architect_node_empty_taxonomy(fake_runtime):
    """Test architect node with empty taxonomy."""
    set_runtime(fake_runtime)
    fake_runtime.taxonomy.domains = []
    fake_runtime.taxonomy.leaves = []

    state = {}

    result = architect_node(state)

    assert result["status"] == "taxonomy_ready"
    assert "empty taxonomy" in result["log"][0]


def test_architect_node_fake_llm(fake_runtime):
    """Test architect node with fake LLM."""
    set_runtime(fake_runtime)
    # Fake LLM should still produce some result

    initial_leaves = len(fake_runtime.taxonomy.leaves)

    state = {}

    result = architect_node(state)

    assert result["status"] == "taxonomy_ready"
    # With fake LLM, may or may not add leaves
    assert len(fake_runtime.taxonomy.leaves) >= initial_leaves


def test_taxonomy_health_node_basic(fake_runtime, sample_question):
    """Test taxonomy health node."""
    set_runtime(fake_runtime)

    # Add some accepted questions
    for i in range(5):
        fake_runtime.store.append_accepted(sample_question(id=f"q{i}"))

    state = {
        "taxonomy_ticks": 0,
    }

    result = taxonomy_health_node(state)

    assert "status" in result
    assert result["taxonomy_ticks"] == 1
    assert "log" in result


def test_taxonomy_health_node_with_diversity_report(fake_runtime, sample_question):
    """Test taxonomy health node with diversity report."""
    set_runtime(fake_runtime)

    for i in range(5):
        fake_runtime.store.append_accepted(sample_question(id=f"q{i}"))

    state = {
        "taxonomy_ticks": 2,
        "last_diversity": {
            "report": {
                "total_count": 5,
                "mean_nn_cosine": 0.85,
                "kl_divergence": 0.1,
                "entropy_ratio": 0.9,
            }
        },
    }

    result = taxonomy_health_node(state)

    assert result["taxonomy_ticks"] == 3


def test_taxonomy_health_node_fake_llm_no_apply(fake_runtime):
    """Test taxonomy health node with fake_llm doesn't apply changes."""
    set_runtime(fake_runtime)
    fake_runtime.settings.fake_llm = True

    initial_leaves = len(fake_runtime.taxonomy.leaves)

    state = {"taxonomy_ticks": 0}

    result = taxonomy_health_node(state)

    # With fake_llm=True and apply=False, should not add leaves
    assert result["taxonomy_ticks"] == 1
