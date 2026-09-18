"""Epsilon-greedy leaf sampling that boosts coverage gaps, not easy high-accept leaves.

Unseen leaves get a coverage-gap prior. Leaves with enough samples and a very
low accept rate are treated as likely-unanswerable and down-weighted. This
module does not assign hardware labels or emit writer briefs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

from cuda_sft_qgen.models import Question

if TYPE_CHECKING:
    from cuda_sft_qgen.taxonomy import Taxonomy


@dataclass
class LeafHealthScore:
    """Health metrics for a single leaf node."""

    l3_id: str
    accepted: int
    rejected: int
    total_generated: int
    avg_difficulty: float
    target_quota: int
    domain_id: str = ""

    @property
    def accept_rate(self) -> float:
        """Acceptance rate: accepted / (accepted + rejected)."""
        total = self.accepted + self.rejected
        if total == 0:
            return 0.5  # neutral prior for unseen leaves
        return self.accepted / total

    @property
    def coverage(self) -> float:
        """Coverage ratio: accepted / target_quota."""
        if self.target_quota <= 0:
            return 1.0
        return min(self.accepted / self.target_quota, 1.0)

    @property
    def coverage_gap(self) -> float:
        """Gap to target: 1 - coverage."""
        return 1.0 - self.coverage


class AdaptiveSampler:
    """Epsilon-greedy adaptive sampler based on leaf health scores."""

    def __init__(
        self,
        epsilon: float = 0.15,
        accept_weight: float = 0.6,
        coverage_weight: float = 0.4,
        min_samples: int = 5,
    ):
        """
        Args:
            epsilon: Exploration probability (0-1). Default 0.15 means 15% random.
            accept_weight: Weight for the anti-easy term (1 - easy_score).
            coverage_weight: Weight for coverage_gap.
            min_samples: Samples before treating accept_rate as evidence of easiness.
        """
        self.epsilon = epsilon
        self.accept_weight = accept_weight
        self.coverage_weight = coverage_weight
        self.min_samples = max(1, int(min_samples))

    def compute_health_score(self, leaf: LeafHealthScore) -> float:
        """Score under-covered and harder leaves; do not reward easy high accept rates.

        Unseen leaves (fewer than ``min_samples``) use coverage_gap plus a small
        explore prior. Saturated easy leaves score near zero.
        """
        gap = leaf.coverage_gap
        if leaf.total_generated < self.min_samples:
            return 0.7 * gap + 0.3
        easy = leaf.accept_rate if leaf.avg_difficulty <= 2.5 else leaf.accept_rate * 0.5
        score = gap * self.coverage_weight + (1.0 - easy) * self.accept_weight
        if leaf.accept_rate < 0.15:
            score *= 0.25
        return score

    def adjust_weights(
        self,
        base_weights: dict[str, float],
        health_scores: dict[str, LeafHealthScore],
        rng: random.Random | None = None,
    ) -> dict[str, float]:
        """
        Adjust L1 domain weights based on leaf health scores.

        Args:
            base_weights: Original L1 weights from planner
            health_scores: Health metrics per leaf
            rng: Random number generator

        Returns:
            Adjusted L1 weights incorporating health information
        """
        rng = rng or random.Random()

        # Group health scores by L1 domain
        l1_health: dict[str, list[float]] = {}
        for health in health_scores.values():
            l1 = health.domain_id
            if not l1:
                continue
            score = self.compute_health_score(health)
            l1_health.setdefault(l1, []).append(score)

        # Compute average health per L1
        l1_avg_health = {k: sum(v) / len(v) for k, v in l1_health.items() if v}

        if not l1_avg_health:
            return base_weights

        # Epsilon-greedy: explore vs exploit
        if rng.random() < self.epsilon:
            # Exploration: return base weights unchanged
            return dict(base_weights)

        # Exploitation: boost weights by health score
        adjusted = {}
        for l1, base_w in base_weights.items():
            health = l1_avg_health.get(l1, 0.5)  # neutral default
            # Reward usable outputs and coverage gaps without deriving domains from IDs.
            adjusted[l1] = base_w * (0.5 + health)

        return adjusted


def compute_leaf_stats(
    accepted: Sequence[Question],
    rejected: Sequence[Question],
    target_per_leaf: int = 3,
    taxonomy: Taxonomy | None = None,
) -> dict[str, LeafHealthScore]:
    """
    Compute health statistics for each leaf node.

    Args:
        accepted: Accepted questions
        rejected: Rejected questions
        target_per_leaf: Target quota per leaf (default 3)

    Returns:
        Dictionary mapping l3_id to LeafHealthScore
    """
    from collections import defaultdict

    leaf_accepted: dict[str, int] = defaultdict(int)
    leaf_rejected: dict[str, int] = defaultdict(int)
    leaf_difficulties: dict[str, list[int]] = defaultdict(list)
    domains = {leaf.l3_id: leaf.domain_id for leaf in taxonomy.leaves if not leaf.retired} if taxonomy else {}

    for q in accepted:
        if q.l3_id:
            if q.topic_path:
                domains.setdefault(q.l3_id, q.topic_path[0])
            leaf_accepted[q.l3_id] += 1
            if q.difficulty:
                leaf_difficulties[q.l3_id].append(q.difficulty)

    for q in rejected:
        if q.l3_id:
            if q.topic_path:
                domains.setdefault(q.l3_id, q.topic_path[0])
            leaf_rejected[q.l3_id] += 1

    all_l3_ids = set(domains) | set(leaf_accepted) | set(leaf_rejected)

    stats = {}
    for l3_id in all_l3_ids:
        acc = leaf_accepted[l3_id]
        rej = leaf_rejected[l3_id]
        diffs = leaf_difficulties[l3_id]
        avg_diff = sum(diffs) / len(diffs) if diffs else 2.0

        stats[l3_id] = LeafHealthScore(
            l3_id=l3_id,
            accepted=acc,
            rejected=rej,
            total_generated=acc + rej,
            avg_difficulty=avg_diff,
            target_quota=target_per_leaf,
            domain_id=domains.get(l3_id, ""),
        )

    return stats
