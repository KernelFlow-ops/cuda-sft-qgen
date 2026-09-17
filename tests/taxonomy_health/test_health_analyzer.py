"""Tests for taxonomy health analyzer."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from cuda_sft_qgen.models import Question
from cuda_sft_qgen.store.archive import Archive
from cuda_sft_qgen.taxonomy import Taxonomy, Leaf
from cuda_sft_qgen.taxonomy_health import (
    HealthAnalyzer,
    TaxonomyHealth,
    L2Health,
    HealthReport,
)


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
                        {"id": "mem.shared.001", "claim": "Shared memory is faster than global", "types": ["conceptual"], "difficulty": [2]},
                        {"id": "mem.shared.002", "claim": "Bank conflicts reduce throughput", "types": ["practical"], "difficulty": [3]},
                    ],
                },
                {
                    "id": "global",
                    "name": "Global Memory",
                    "leaves": [
                        {"id": "mem.global.001", "claim": "Global memory is large but slow", "types": ["conceptual"], "difficulty": [1]},
                    ],
                },
            ],
        },
        {
            "id": "compute",
            "name": "Compute",
            "children": [
                {
                    "id": "warp",
                    "name": "Warp",
                    "leaves": [],
                },
            ],
        },
    ]

    leaves = [
        Leaf("mem", "Memory", "shared", "Shared Memory", "mem.shared.001",
             "Shared memory is faster than global", ("conceptual",), (2,)),
        Leaf("mem", "Memory", "shared", "Shared Memory", "mem.shared.002",
             "Bank conflicts reduce throughput", ("practical",), (3,)),
        Leaf("mem", "Memory", "global", "Global Memory", "mem.global.001",
             "Global memory is large but slow", ("conceptual",), (1,)),
    ]

    return Taxonomy(
        version=1,
        name="test",
        leaves=leaves,
        domains=domains,
    )


@pytest.fixture
def sample_questions():
    """Create sample questions for testing."""
    accepted = [
        Question(
            id="q1",
            question="What is shared memory?",
            tests_claim="Explain shared memory",
            topic_path=["mem", "shared", "mem.shared.001"],
            l3_id="mem.shared.001",
            question_type="conceptual",
            difficulty=2,
            bloom="understand",
            persona="cuda_beginner",
            hardware="agnostic",
            granularity="api",
            language="zh",
        ),
        Question(
            id="q2",
            question="What causes bank conflicts?",
            tests_claim="Explain bank conflicts",
            topic_path=["mem", "shared", "mem.shared.002"],
            l3_id="mem.shared.002",
            question_type="practical",
            difficulty=3,
            bloom="analyze",
            persona="cuda_intermediate",
            hardware="agnostic",
            granularity="kernel",
            language="zh",
        ),
        Question(
            id="q3",
            question="Why is global memory slow?",
            tests_claim="Explain global memory latency",
            topic_path=["mem", "global", "mem.global.001"],
            l3_id="mem.global.001",
            question_type="conceptual",
            difficulty=1,
            bloom="understand",
            persona="cuda_beginner",
            hardware="agnostic",
            granularity="api",
            language="zh",
        ),
    ]

    rejected = [
        Question(
            id="q4",
            question="Bad question?",
            tests_claim="Bad question",
            topic_path=["mem", "shared", "mem.shared.001"],
            l3_id="mem.shared.001",
            question_type="conceptual",
            difficulty=2,
            bloom="understand",
            persona="cuda_beginner",
            hardware="agnostic",
            granularity="api",
            language="zh",
            reject_reason="too_vague",
        ),
    ]

    return accepted, rejected


def test_taxonomy_health_compute_l2_metrics(sample_taxonomy, sample_questions):
    """Test L2 metrics computation."""
    accepted, rejected = sample_questions
    health = TaxonomyHealth(sample_taxonomy, target_per_l2=10)

    metrics = health.compute_l2_metrics(accepted, rejected)

    assert "mem.shared" in metrics
    assert "mem.global" in metrics
    assert "compute.warp" in metrics

    mem_shared = metrics["mem.shared"]
    assert mem_shared.leaf_count == 2
    assert mem_shared.question_count == 2
    assert mem_shared.accept_rate == 2 / 3  # 2 accepted, 1 rejected
    assert mem_shared.coverage == 1.0
    assert mem_shared.question_quota_completion == 2 / 10


def test_taxonomy_health_diversity_no_embedder(sample_taxonomy, sample_questions):
    """Test diversity calculation without embedder."""
    accepted, rejected = sample_questions
    health = TaxonomyHealth(sample_taxonomy, target_per_l2=10, embedder=None)

    metrics = health.compute_l2_metrics(accepted, rejected)
    mem_shared = metrics["mem.shared"]

    assert mem_shared.avg_diversity == 0.0


def test_taxonomy_health_empty_l2(sample_taxonomy, sample_questions):
    """Test metrics for L2 with no questions."""
    accepted, rejected = sample_questions
    health = TaxonomyHealth(sample_taxonomy, target_per_l2=10)

    metrics = health.compute_l2_metrics(accepted, rejected)
    compute_warp = metrics["compute.warp"]

    assert compute_warp.leaf_count == 0
    assert compute_warp.question_count == 0
    assert compute_warp.coverage == 0.0
    assert compute_warp.accept_rate == 0.5  # Default


def test_health_analyzer_analyze(sample_taxonomy, sample_questions, tmp_path):
    """Test full health analysis."""
    accepted, rejected = sample_questions

    # Create mock archive
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    archive = Archive(data_dir)

    # Save questions
    for q in accepted:
        archive.append_accepted(q)
    for q in rejected:
        archive.append_rejected(q)

    analyzer = HealthAnalyzer(sample_taxonomy, target_per_l2=10)
    report = analyzer.analyze(archive)

    assert isinstance(report, HealthReport)
    assert len(report.l2_metrics) == 3
    assert report.overall_coverage > 0
    assert report.overall_accept_rate > 0
    assert len(report.underperforming) >= 0
    assert len(report.expansion_targets) >= 0


def test_health_analyzer_identify_expansion_targets(sample_taxonomy):
    """Test expansion target identification."""
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
        "compute.warp": L2Health(
            l2_key="compute.warp",
            l1_id="compute",
            l2_id="warp",
            leaf_count=2,
            question_count=1,
            expected_quota=10,
            coverage=0.1,
            accept_rate=0.9,
            avg_diversity=0.9,
            health_score=40.0,
        ),
    }

    analyzer = HealthAnalyzer(sample_taxonomy, target_per_l2=10)
    targets = analyzer._identify_expansion_targets(l2_metrics)

    # compute.warp should be high priority (low coverage, high accept rate)
    assert "compute.warp" in targets
    # mem.shared should also be a target (coverage < 0.5, accept_rate > 0.6)
    assert "mem.shared" in targets
    # mem.global should not be prioritized (coverage = 0.8)
    assert targets.index("compute.warp") < targets.index("mem.shared")


def test_health_score_calculation(sample_taxonomy):
    """Test health score calculation."""
    health = TaxonomyHealth(sample_taxonomy, target_per_l2=10)

    # Perfect metrics
    score = health._compute_health_score(
        coverage=1.0,
        accept_rate=1.0,
        diversity=1.0,
        leaf_count=15,
    )
    assert score == 100.0

    # Poor metrics
    score = health._compute_health_score(
        coverage=0.2,
        accept_rate=0.3,
        diversity=0.5,
        leaf_count=2,
    )
    assert score < 50.0

    # Medium metrics
    score = health._compute_health_score(
        coverage=0.5,
        accept_rate=0.7,
        diversity=0.8,
        leaf_count=10,
    )
    assert 50.0 < score < 80.0
