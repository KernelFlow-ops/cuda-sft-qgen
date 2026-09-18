"""Tests for epsilon-greedy adaptive leaf sampling (planner_v2)."""

from __future__ import annotations

import random

import pytest

from cuda_sft_qgen.models import Question
from cuda_sft_qgen.planner_v2 import (
    AdaptiveSampler,
    LeafHealthScore,
    compute_leaf_stats,
)


def test_leaf_health_score_accept_rate():
    """Test accept_rate calculation."""
    # No samples: neutral prior
    leaf = LeafHealthScore(
        l3_id="cuda.memory.l3_001",
        accepted=0,
        rejected=0,
        total_generated=0,
        avg_difficulty=2.0,
        target_quota=3,
    )
    assert leaf.accept_rate == 0.5

    # 8 accepted, 2 rejected
    leaf = LeafHealthScore(
        l3_id="cuda.memory.l3_001",
        accepted=8,
        rejected=2,
        total_generated=10,
        avg_difficulty=2.5,
        target_quota=3,
    )
    assert leaf.accept_rate == 0.8


def test_leaf_health_score_coverage():
    """Test coverage calculation."""
    # Under quota
    leaf = LeafHealthScore(
        l3_id="cuda.memory.l3_001",
        accepted=1,
        rejected=0,
        total_generated=1,
        avg_difficulty=2.0,
        target_quota=3,
    )
    assert leaf.coverage == pytest.approx(1 / 3)
    assert leaf.coverage_gap == pytest.approx(2 / 3)

    # Met quota
    leaf = LeafHealthScore(
        l3_id="cuda.memory.l3_002",
        accepted=3,
        rejected=0,
        total_generated=3,
        avg_difficulty=2.0,
        target_quota=3,
    )
    assert leaf.coverage == 1.0
    assert leaf.coverage_gap == 0.0

    # Over quota (capped at 1.0)
    leaf = LeafHealthScore(
        l3_id="cuda.memory.l3_003",
        accepted=5,
        rejected=0,
        total_generated=5,
        avg_difficulty=2.0,
        target_quota=3,
    )
    assert leaf.coverage == 1.0
    assert leaf.coverage_gap == 0.0


def test_adaptive_sampler_health_score():
    """Coverage gaps score higher than easy saturated leaves."""
    sampler = AdaptiveSampler(epsilon=0.15, accept_weight=0.6, coverage_weight=0.4, min_samples=5)

    easy_full = LeafHealthScore(
        l3_id="cuda.memory.l3_001",
        accepted=8,
        rejected=0,
        total_generated=8,
        avg_difficulty=2.0,
        target_quota=3,
    )
    # gap=0, easy=1.0 → score 0
    assert sampler.compute_health_score(easy_full) == pytest.approx(0.0)

    unseen = LeafHealthScore(
        l3_id="cuda.memory.l3_003",
        accepted=0,
        rejected=0,
        total_generated=0,
        avg_difficulty=2.0,
        target_quota=3,
    )
    assert sampler.compute_health_score(unseen) == pytest.approx(1.0)

    undercovered = LeafHealthScore(
        l3_id="cuda.memory.l3_002",
        accepted=1,
        rejected=1,
        total_generated=5,
        avg_difficulty=3.0,
        target_quota=3,
    )
    # gap=2/3, easy=0.5*0.5=0.25 → 0.4*(2/3)+0.6*0.75
    assert sampler.compute_health_score(undercovered) > sampler.compute_health_score(easy_full)


def test_adaptive_sampler_adjust_weights():
    """Test weight adjustment with epsilon-greedy."""
    sampler = AdaptiveSampler(epsilon=0.0, accept_weight=0.6, coverage_weight=0.4)
    rng = random.Random(42)

    base_weights = {
        "cuda": 1.0,
        "memory": 1.0,
        "compute": 1.0,
    }

    # Create health scores: cuda is unhealthy, memory is healthy
    health_scores = {
        "cuda.sync.l3_001": LeafHealthScore(
            l3_id="cuda.sync.l3_001",
            domain_id="cuda",
            accepted=1,
            rejected=4,  # Low accept rate
            total_generated=5,
            avg_difficulty=2.0,
            target_quota=3,
        ),
        "memory.shared.l3_002": LeafHealthScore(
            l3_id="memory.shared.l3_002",
            domain_id="memory",
            accepted=3,
            rejected=0,  # High accept rate
            total_generated=3,
            avg_difficulty=2.0,
            target_quota=3,
        ),
        "compute.warp.l3_003": LeafHealthScore(
            l3_id="compute.warp.l3_003",
            domain_id="compute",
            accepted=0,
            rejected=0,  # No data, neutral prior
            total_generated=0,
            avg_difficulty=2.0,
            target_quota=3,
        ),
    }

    adjusted = sampler.adjust_weights(base_weights, health_scores, rng=rng)

    # Easy fully-covered memory must not outrank unseen compute or under-covered cuda.
    assert adjusted["memory"] < adjusted["compute"]
    assert adjusted["cuda"] > adjusted["memory"]


def test_adaptive_sampler_epsilon_exploration():
    """Test epsilon-greedy exploration returns base weights."""
    sampler = AdaptiveSampler(epsilon=1.0)  # Always explore
    rng = random.Random(42)

    base_weights = {"cuda": 1.0, "memory": 2.0}
    health_scores = {
        "cuda.sync.l3_001": LeafHealthScore(
            l3_id="cuda.sync.l3_001",
            accepted=0,
            rejected=10,
            total_generated=10,
            avg_difficulty=2.0,
            target_quota=3,
        ),
    }

    adjusted = sampler.adjust_weights(base_weights, health_scores, rng=rng)
    # With epsilon=1.0, should return base weights unchanged
    assert adjusted == base_weights


def test_compute_leaf_stats():
    """Test leaf statistics computation from questions."""
    accepted = [
        Question(
            id="q1",
            l3_id="cuda.memory.l3_001",
            question="Test question 1",
            tests_claim="claim1",
            difficulty=2,
            topic_path=["cuda", "memory", "shared"],
            question_type="conceptual",
            bloom="understand",
            persona="cuda_interviewer",
            hardware="agnostic",
            granularity="kernel",
            language="zh",
            review_status="accepted",
        ),
        Question(
            id="q2",
            l3_id="cuda.memory.l3_001",
            question="Test question 2",
            tests_claim="claim2",
            difficulty=3,
            topic_path=["cuda", "memory", "shared"],
            question_type="conceptual",
            bloom="apply",
            persona="cuda_interviewer",
            hardware="agnostic",
            granularity="kernel",
            language="zh",
            review_status="accepted",
        ),
        Question(
            id="q3",
            l3_id="cuda.compute.l3_002",
            question="Test question 3",
            tests_claim="claim3",
            difficulty=4,
            topic_path=["cuda", "compute", "warp"],
            question_type="code_complete",
            bloom="analyze",
            persona="cuda_interviewer",
            hardware="ampere",
            granularity="api",
            language="en",
            review_status="accepted",
        ),
    ]

    rejected = [
        Question(
            id="q4",
            l3_id="cuda.memory.l3_001",
            question="Bad question",
            tests_claim="bad_claim",
            difficulty=1,
            topic_path=["cuda", "memory", "shared"],
            question_type="conceptual",
            bloom="remember",
            persona="cuda_interviewer",
            hardware="agnostic",
            granularity="kernel",
            language="zh",
            review_status="rejected",
            reject_reason="too_vague",
        ),
    ]

    stats = compute_leaf_stats(accepted, rejected, target_per_leaf=3)

    # Check cuda.memory.l3_001: 2 accepted, 1 rejected
    leaf1 = stats["cuda.memory.l3_001"]
    assert leaf1.accepted == 2
    assert leaf1.rejected == 1
    assert leaf1.total_generated == 3
    assert leaf1.accept_rate == pytest.approx(2 / 3)
    assert leaf1.avg_difficulty == pytest.approx(2.5)  # (2+3)/2
    assert leaf1.coverage == pytest.approx(2 / 3)
    assert leaf1.target_quota == 3

    # Check cuda.compute.l3_002: 1 accepted, 0 rejected
    leaf2 = stats["cuda.compute.l3_002"]
    assert leaf2.accepted == 1
    assert leaf2.rejected == 0
    assert leaf2.total_generated == 1
    assert leaf2.accept_rate == 1.0
    assert leaf2.avg_difficulty == 4.0
    assert leaf2.coverage == pytest.approx(1 / 3)


def test_low_accept_rate_reduces_weight():
    """Reject storms are down-weighted versus under-covered moderate-accept leaves."""
    sampler = AdaptiveSampler(epsilon=0.0, accept_weight=0.6, coverage_weight=0.4, min_samples=5)
    rng = random.Random(42)

    base_weights = {"domain_reject_storm": 1.0, "domain_undercovered": 1.0}

    health_scores = {
        "domain_reject_storm.l2.l3_001": LeafHealthScore(
            l3_id="domain_reject_storm.l2.l3_001",
            domain_id="domain_reject_storm",
            accepted=1,
            rejected=9,
            total_generated=10,
            avg_difficulty=2.0,
            target_quota=3,
        ),
        "domain_undercovered.l2.l3_002": LeafHealthScore(
            l3_id="domain_undercovered.l2.l3_002",
            domain_id="domain_undercovered",
            accepted=2,
            rejected=3,
            total_generated=5,
            avg_difficulty=3.0,
            target_quota=8,
        ),
    }

    adjusted = sampler.adjust_weights(base_weights, health_scores, rng=rng)
    assert adjusted["domain_reject_storm"] < adjusted["domain_undercovered"]


def test_uncovered_leaves_increase_weight():
    """Integration test: uncovered leaves should get higher weight."""
    sampler = AdaptiveSampler(epsilon=0.0, accept_weight=0.6, coverage_weight=0.4)
    rng = random.Random(42)

    base_weights = {"domain_covered": 1.0, "domain_uncovered": 1.0}

    health_scores = {
        # Covered domain: reached target
        "domain_covered.l2.l3_001": LeafHealthScore(
            l3_id="domain_covered.l2.l3_001",
            domain_id="domain_covered",
            accepted=3,
            rejected=0,
            total_generated=3,
            avg_difficulty=2.0,
            target_quota=3,
        ),
        # Uncovered domain: far from target
        "domain_uncovered.l2.l3_002": LeafHealthScore(
            l3_id="domain_uncovered.l2.l3_002",
            domain_id="domain_uncovered",
            accepted=0,
            rejected=0,
            total_generated=0,
            avg_difficulty=2.0,
            target_quota=3,
        ),
    }

    adjusted = sampler.adjust_weights(base_weights, health_scores, rng=rng)

    # Uncovered/unseen domain should outrank a fully covered easy leaf.
    assert adjusted["domain_uncovered"] > adjusted["domain_covered"]


def test_real_taxonomy_domains_and_zero_sample_leaves(fake_settings, sample_question):
    from cuda_sft_qgen.taxonomy import load_taxonomy

    taxonomy = load_taxonomy(fake_settings.taxonomy_path)
    leaf = next(leaf for leaf in taxonomy.leaves if not leaf.retired)
    accepted = [sample_question(id=f"q_{i}", l3_id=leaf.l3_id, topic_path=list(leaf.path)) for i in range(3)]
    stats = compute_leaf_stats(accepted, [], taxonomy=taxonomy)
    active = [leaf for leaf in taxonomy.leaves if not leaf.retired]
    assert len(stats) == len(active)
    assert all(stats[leaf.l3_id].domain_id == leaf.domain_id for leaf in active)
    unseen = next(stats[item.l3_id] for item in active if item.l3_id != leaf.l3_id)
    assert unseen.total_generated == 0
    assert unseen.coverage_gap == 1
    sampler = AdaptiveSampler(epsilon=0)
    base = {domain: 1.0 for domain in taxonomy.l1_ids()}
    adjusted = sampler.adjust_weights(base, stats)
    assert adjusted != base
    assert adjusted[leaf.domain_id] < max(adjusted.values())
