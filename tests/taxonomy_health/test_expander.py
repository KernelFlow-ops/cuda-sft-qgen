"""Tests for taxonomy leaf expander."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch

from cuda_sft_qgen.taxonomy import Taxonomy, Leaf
from cuda_sft_qgen.taxonomy_health import HealthReport, L2Health
from cuda_sft_qgen.taxonomy_health.expander import LeafExpander


@pytest.fixture
def sample_taxonomy():
    """Create a sample taxonomy for testing."""
    domains = [
        {
            "id": "mem",
            "name": "Memory",
            "children": [
                {
                    "id": "shared",
                    "name": "Shared Memory",
                    "leaves": [
                        {"id": "shared_001", "claim": "Shared memory is faster than global", "types": ["conceptual"], "difficulty": [2]},
                    ],
                },
            ],
        },
    ]

    leaves = [
        Leaf("mem", "Memory", "shared", "Shared Memory", "shared_001",
             "Shared memory is faster than global", ("conceptual",), (2,)),
    ]

    return Taxonomy(
        version=1,
        name="test",
        leaves=leaves,
        domains=domains,
    )


@pytest.fixture
def sample_health_report():
    """Create a sample health report."""
    l2_metrics = {
        "mem.shared": L2Health(
            l2_key="mem.shared",
            l1_id="mem",
            l2_id="shared",
            leaf_count=5,
            question_count=3,
            expected_quota=10,
            coverage=0.3,
            accept_rate=0.75,
            avg_diversity=0.8,
            health_score=45.0,
        ),
        "mem.global": L2Health(
            l2_key="mem.global",
            l1_id="mem",
            l2_id="global",
            leaf_count=10,
            question_count=8,
            expected_quota=10,
            coverage=0.8,
            accept_rate=0.5,
            avg_diversity=0.7,
            health_score=65.0,
        ),
    }

    return HealthReport(
        l2_metrics=l2_metrics,
        overall_coverage=0.55,
        overall_accept_rate=0.625,
        overall_diversity=0.75,
        overall_health=55.0,
        underperforming=["mem.shared"],
        expansion_targets=["mem.shared"],
    )


def test_leaf_expander_identify_targets(sample_taxonomy, sample_health_report):
    """Test expansion target identification."""
    llm = Mock()
    expander = LeafExpander(sample_taxonomy, llm)

    targets = expander.identify_expansion_targets(sample_health_report)

    assert len(targets) > 0
    assert targets[0][0] == "mem.shared"
    assert "low_coverage" in targets[0][1]["reason"]
    assert "good_accept_rate" in targets[0][1]["reason"]


def test_leaf_expander_preview(sample_taxonomy, sample_health_report):
    """Test expansion preview."""
    llm = Mock()
    expander = LeafExpander(sample_taxonomy, llm)

    preview = expander.preview_expansion(sample_health_report, limit=5)

    assert len(preview) > 0
    assert preview[0]["l2_key"] == "mem.shared"
    assert "current_leaves" in preview[0]
    assert "suggested_new_leaves" in preview[0]
    assert preview[0]["coverage"] == 0.3
    assert preview[0]["accept_rate"] == 0.75


@patch("cuda_sft_qgen.taxonomy_health.expander.load_prompt")
def test_leaf_expander_expand_l2_dry_run(
    mock_load_prompt,
    sample_taxonomy,
):
    """Test L2 expansion in dry-run mode."""
    llm = Mock()
    expander = LeafExpander(sample_taxonomy, llm)

    result = expander.expand_l2("mem.shared", max_new=5, dry_run=True)

    assert len(result) == 1
    assert result[0]["dry_run"] is True
    assert result[0]["l2_key"] == "mem.shared"
    assert result[0]["max_new"] == 5


@patch("cuda_sft_qgen.taxonomy_health.expander.load_prompt")
def test_leaf_expander_expand_l2_with_llm(
    mock_load_prompt,
    sample_taxonomy,
):
    """Test L2 expansion with LLM."""
    mock_load_prompt.side_effect = lambda x: f"PROMPT_{x}"

    llm = Mock()
    llm.complete_json.return_value = {
        "new_leaves": [
            {
                "id": "shared_002",
                "claim": "Bank conflicts occur when multiple threads access the same bank",
                "types": ["practical"],
                "difficulty": [3],
                "kind": "mechanism",
            },
            {
                "id": "shared_003",
                "claim": "Shared memory has 32 banks on modern GPUs",
                "types": ["numeric"],
                "difficulty": [2],
                "kind": "numeric",
            },
        ]
    }

    expander = LeafExpander(sample_taxonomy, llm, priority=50)
    result = expander.expand_l2("mem.shared", max_new=5, facet_id="mechanism")

    assert len(result) == 2
    assert result[0]["l3_id"] == "shared_002"
    assert result[0]["domain_id"] == "mem"
    assert result[0]["l2_id"] == "shared"
    assert "Bank conflicts" in result[0]["claim"]

    llm.complete_json.assert_called_once()
    call_args = llm.complete_json.call_args
    assert call_args[1]["priority"] == 50


@patch("cuda_sft_qgen.taxonomy_health.expander.load_prompt")
def test_leaf_expander_invalid_l2_key(mock_load_prompt, sample_taxonomy):
    """Test expansion with invalid L2 key."""
    llm = Mock()
    expander = LeafExpander(sample_taxonomy, llm)

    result = expander.expand_l2("invalid.key", max_new=5)
    assert result == []

    result = expander.expand_l2("mem.nonexistent", max_new=5)
    assert result == []


@patch("cuda_sft_qgen.taxonomy_health.expander.load_prompt")
def test_leaf_expander_validation_filtering(
    mock_load_prompt,
    sample_taxonomy,
):
    """Test that invalid leaves are filtered out."""
    mock_load_prompt.side_effect = lambda x: f"PROMPT_{x}"

    llm = Mock()
    llm.complete_json.return_value = {
        "new_leaves": [
            {
                "id": "shared_001",  # Duplicate ID (already exists in taxonomy)
                "claim": "Shared memory is faster than global memory access",
                "types": ["conceptual"],
                "difficulty": [2],
                "kind": "mechanism",
            },
            {
                "id": "shared_002",
                "claim": "Valid new leaf with sufficient length for validation",
                "types": ["practical"],
                "difficulty": [3],
                "kind": "mechanism",
            },
            {
                # Missing ID
                "claim": "No ID leaf but has enough text to pass length check",
                "types": ["conceptual"],
                "difficulty": [2],
                "kind": "mechanism",
            },
        ]
    }

    expander = LeafExpander(sample_taxonomy, llm)
    result = expander.expand_l2("mem.shared", max_new=10)

    # Only the valid leaf should be returned
    assert len(result) == 1
    assert result[0]["l3_id"] == "shared_002"


@patch("cuda_sft_qgen.taxonomy_health.expander.load_prompt")
def test_leaf_expander_max_new_limit(
    mock_load_prompt,
    sample_taxonomy,
):
    """Test that max_new limit is enforced."""
    mock_load_prompt.side_effect = lambda x: f"PROMPT_{x}"

    llm = Mock()
    # Generate diverse claims to avoid near-duplicate filtering
    diverse_claims = [
        "Bank conflicts occur when threads access same memory bank simultaneously",
        "Shared memory provides low-latency inter-thread communication within block",
        "Memory coalescing improves global memory access patterns significantly",
        "Warp divergence happens when threads take different execution paths",
        "Register spilling moves data to local memory when registers exhausted",
        "Atomic operations on shared memory enable thread synchronization primitives",
        "Memory transactions combine multiple accesses for bandwidth efficiency",
        "Texture memory provides cached read-only access with spatial locality",
        "Constant memory broadcasts single value to all threads in warp",
        "Local memory resides in global memory but private to each thread",
    ]

    llm.complete_json.return_value = {
        "new_leaves": [
            {
                "id": f"shared_{i:03d}",
                "claim": diverse_claims[i - 2],
                "types": ["conceptual"],
                "difficulty": [2],
                "kind": "mechanism",
            }
            for i in range(2, 12)
        ]
    }

    expander = LeafExpander(sample_taxonomy, llm)
    result = expander.expand_l2("mem.shared", max_new=5)

    # Should return at most 5 leaves (some may be filtered for validation reasons)
    assert len(result) <= 5
    assert len(result) >= 3  # Should get at least a few valid ones


def test_leaf_expander_get_facet_info(sample_taxonomy):
    """Test facet info retrieval."""
    llm = Mock()
    expander = LeafExpander(sample_taxonomy, llm)

    # The method should not depend on load_prompt_yaml since facets come from runtime
    facet = expander._get_facet_info("formula")
    # Without runtime setup, this will return a default empty dict or None
    assert facet is None or isinstance(facet, dict)
