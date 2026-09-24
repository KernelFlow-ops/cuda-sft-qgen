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
from cuda_sft_qgen.planner.pair_mining import cuda_difficulty_features, monotonicity_ok

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
    avg_feature_score: float = 2.0

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


def _mixup_mode(question: Question) -> str | None:
    """Read optional mixup provenance without requiring a model schema change."""
    direct = getattr(question, "mixup_mode", None)
    if direct:
        normalized = str(direct).strip().lower()
        return normalized if normalized in {"decomposed", "hybrid", "original"} else None
    metadata = getattr(question, "requested_metadata", {}) or {}
    mode = metadata.get("mixup_mode") or metadata.get("mode")
    if not mode:
        details = getattr(question, "review_details", {}) or {}
        mode = details.get("mixup_mode") or details.get("mode")
    if not mode and str(getattr(question, "seed_id", "") or "").startswith("mixup:"):
        parts = str(question.seed_id).split(":", 2)
        mode = parts[1] if len(parts) > 1 else None
    mode = str(mode or "").strip().lower()
    return mode if mode in {"decomposed", "hybrid", "original"} else None


def mixup_difficulty_report(questions: Sequence[Question]) -> dict[str, float | int]:
    """Check difficulty monotonicity independently for each parent family.

    A family is the sorted pair of parent ids.  Each generated child carries
    the two parent difficulty values, so an ``original`` middle sample is not
    required in the archive: the high parent is the reference upper bound.
    Aggregating all modes globally would hide a bad pair behind an unrelated
    good pair, hence every family is checked separately.
    """
    families: dict[tuple[str, ...], dict[str, list[float]]] = {}
    for question in questions:
        mode = _mixup_mode(question)
        parent_ids = list(getattr(question, "parent_ids", ()) or ())
        if not parent_ids:
            context = getattr(question, "knowledge_context", {}) or {}
            mixup = context.get("mixup") if isinstance(context, dict) else {}
            parent_ids = list((mixup or {}).get("parent_ids") or [])
        if not mode or len(parent_ids) < 2:
            continue
        family = tuple(sorted(str(parent_id) for parent_id in parent_ids))
        families.setdefault(family, {"decomposed": [], "original": [], "hybrid": []})[mode].append(
            float(getattr(question, "difficulty", 0) or 0)
        )
    if not families:
        return {
            "complete_groups": 0,
            "monotonic_groups": 0,
            "monotonicity_rate": 0.0,
            "missing_modes": 2,
            "families": 0,
        }
    complete = 0
    monotonic = 0
    raw_decomp: list[float] = []
    raw_original: list[float] = []
    raw_hybrid: list[float] = []
    for family, modes in families.items():
        if not modes["decomposed"] or not modes["hybrid"]:
            continue
        complete += 1
        decomp = sum(modes["decomposed"]) / len(modes["decomposed"])
        hybrid = sum(modes["hybrid"]) / len(modes["hybrid"])
        # Pair-specific parent bounds are attached to each question.  A family
        # with only child records still gets a conservative 1..4 check.
        low = 1.0
        high = 4.0
        for question in questions:
            ids = tuple(sorted(str(parent_id) for parent_id in (getattr(question, "parent_ids", ()) or ())))
            if ids != family:
                continue
            parent_difficulties = list(getattr(question, "parent_difficulties", ()) or ())
            if len(parent_difficulties) >= 2:
                low = float(min(parent_difficulties))
                high = float(max(parent_difficulties))
                break
        raw_decomp.append(decomp)
        middle = (
            sum(modes["original"]) / len(modes["original"])
            if modes["original"] else high
        )
        raw_original.append(middle)
        raw_hybrid.append(hybrid)
        # Every child must satisfy its own family bounds; averaging can conceal
        # a too-hard decomposed item or a hybrid that failed to add difficulty.
        if (all(low <= d <= high for d in modes["decomposed"])
                and all(h >= min(4.0, high + 1) for h in modes["hybrid"])
                and low <= middle <= high):
            monotonic += 1
    return {
        "complete_groups": complete,
        "monotonic_groups": monotonic,
        "monotonicity_rate": monotonic / complete if complete else 0.0,
        "missing_modes": sum(
            not any(modes[mode] for modes in families.values())
            for mode in ("decomposed", "hybrid")
        ),
        "families": len(families),
        "mean_decomposed": sum(raw_decomp) / len(raw_decomp) if raw_decomp else 0.0,
        "mean_original": sum(raw_original) / len(raw_original) if raw_original else 0.0,
        "mean_hybrid": sum(raw_hybrid) / len(raw_hybrid) if raw_hybrid else 0.0,
    }


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
    leaf_feature_scores: dict[str, list[float]] = defaultdict(list)
    domains = {leaf.l3_id: leaf.domain_id for leaf in taxonomy.leaves if not leaf.retired} if taxonomy else {}

    for q in accepted:
        if q.l3_id:
            if q.topic_path:
                domains.setdefault(q.l3_id, q.topic_path[0])
            leaf_accepted[q.l3_id] += 1
            if q.difficulty:
                leaf_difficulties[q.l3_id].append(q.difficulty)
            leaf_feature_scores[q.l3_id].append(cuda_difficulty_features(q).score)

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
            avg_feature_score=(
                sum(leaf_feature_scores[l3_id]) / len(leaf_feature_scores[l3_id])
                if leaf_feature_scores[l3_id]
                else 2.0
            ),
        )

    return stats
