"""Taxonomy health analysis module."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from cuda_sft_qgen.models import Question
from cuda_sft_qgen.store.archive import Archive
from cuda_sft_qgen.taxonomy import Taxonomy

logger = logging.getLogger(__name__)


@dataclass
class L2Health:
    """Health metrics for a single L2 node."""

    l2_key: str
    l1_id: str
    l2_id: str
    leaf_count: int
    question_count: int
    expected_quota: int
    coverage: float
    accept_rate: float
    avg_diversity: float
    health_score: float
    question_quota_completion: float = 0.0
    knowledge_gap: bool = False


@dataclass
class HealthReport:
    """Aggregate L2 metrics, expansion targets, and overall health score."""

    l2_metrics: dict[str, L2Health]
    overall_coverage: float
    overall_accept_rate: float
    overall_diversity: float
    overall_health: float
    underperforming: list[str]
    expansion_targets: list[str]


class TaxonomyHealth:
    """Compute per-L2 health metrics."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        *,
        target_per_l2: int = 30,
        embedder: Any = None,
    ):
        """Bind taxonomy, L2 target, and optional embedder for diversity scores."""
        self.taxonomy = taxonomy
        self.target_per_l2 = target_per_l2
        self.embedder = embedder

    def compute_l2_metrics(
        self,
        accepted: Sequence[Question],
        rejected: Sequence[Question],
    ) -> dict[str, L2Health]:
        """Compute health metrics for each L2 node."""
        l2_accepted: dict[str, int] = defaultdict(int)
        l2_rejected: dict[str, int] = defaultdict(int)
        l2_leaf_ids: dict[str, set[str]] = defaultdict(set)

        for q in accepted:
            if len(q.topic_path) >= 2:
                l2_key = f"{q.topic_path[0]}.{q.topic_path[1]}"
                l2_accepted[l2_key] += 1
                if q.l3_id:
                    l2_leaf_ids[l2_key].add(q.l3_id)

        for q in rejected:
            if len(q.topic_path) >= 2:
                l2_key = f"{q.topic_path[0]}.{q.topic_path[1]}"
                l2_rejected[l2_key] += 1

        l2_leaf_counts: dict[str, int] = {}
        l2_info: dict[str, tuple[str, str]] = {}
        for domain in self.taxonomy.domains:
            for child in domain.get("children", []):
                l2_key = f"{domain['id']}.{child['id']}"
                l2_leaf_counts[l2_key] = sum(
                    1
                    for item in child.get("leaves", [])
                    if not item.get("retired")
                )
                l2_info[l2_key] = (domain["id"], child["id"])

        metrics: dict[str, L2Health] = {}
        for l2_key, leaf_count in l2_leaf_counts.items():
            acc = l2_accepted.get(l2_key, 0)
            rej = l2_rejected.get(l2_key, 0)
            total = acc + rej
            accept_rate = acc / total if total > 0 else 0.5
            active_ids = {leaf.l3_id for leaf in self.taxonomy.leaves if not leaf.retired and f"{leaf.domain_id}.{leaf.l2_id}" == l2_key}
            covered = l2_leaf_ids.get(l2_key, set()) & active_ids
            coverage = len(covered) / leaf_count if leaf_count else 0.0
            quota_completion = min(acc / self.target_per_l2, 1.0) if self.target_per_l2 > 0 else 1.0

            diversity = self._compute_diversity(l2_key, l2_leaf_ids.get(l2_key, set()))

            health_score = self._compute_health_score(
                coverage, accept_rate, diversity, leaf_count
            )

            l1_id, l2_id = l2_info.get(l2_key, ("", ""))
            metrics[l2_key] = L2Health(
                l2_key=l2_key,
                l1_id=l1_id,
                l2_id=l2_id,
                leaf_count=leaf_count,
                question_count=acc,
                expected_quota=self.target_per_l2,
                coverage=coverage,
                accept_rate=accept_rate,
                avg_diversity=diversity,
                health_score=health_score,
                question_quota_completion=quota_completion,
                knowledge_gap=leaf_count < 10,
            )

        return metrics

    def _compute_diversity(self, l2_key: str, leaf_ids: set[str]) -> float:
        """Compute average diversity based on embedding nearest neighbor distances."""
        if not leaf_ids or len(leaf_ids) < 2:
            return 0.0

        if self.embedder is None:
            return 0.0

        try:
            from cuda_sft_qgen.taxonomy_embed import leaf_embed_text

            leaves = [
                lf for lf in self.taxonomy.leaves
                if lf.l3_id in leaf_ids and not lf.retired
            ]
            if len(leaves) < 2:
                return 0.0

            texts = [leaf_embed_text(lf.kind, lf.l2_id, lf.claim) for lf in leaves]
            embeddings = self.embedder.embed(texts)
            embeddings = np.asarray(embeddings, dtype=np.float32)

            from cuda_sft_qgen.embeddings import l2_normalize
            embeddings = l2_normalize(embeddings)

            similarities = embeddings @ embeddings.T
            np.fill_diagonal(similarities, -1)

            nearest_similarities = np.max(similarities, axis=1)
            avg_nearest = float(np.mean(nearest_similarities))

            diversity = 1.0 - avg_nearest
            return max(0.0, min(1.0, diversity))

        except Exception as exc:
            logger.warning("diversity computation failed for %s: %s", l2_key, exc)
            return 0.0

    def _compute_health_score(
        self,
        coverage: float,
        accept_rate: float,
        diversity: float,
        leaf_count: int,
    ) -> float:
        """Compute overall health score for an L2 node."""
        score = 0.0

        score += coverage * 40.0
        score += accept_rate * 30.0
        score += diversity * 20.0

        if leaf_count >= 10:
            score += 10.0
        elif leaf_count >= 5:
            score += 5.0

        return min(100.0, score)


class HealthAnalyzer:
    """Analyze taxonomy health and generate reports."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        *,
        target_per_l2: int = 30,
        embedder: Any = None,
    ):
        """Construct the L2 health calculator used by :meth:`analyze`."""
        self.taxonomy = taxonomy
        self.target_per_l2 = target_per_l2
        self.embedder = embedder
        self.health_calculator = TaxonomyHealth(
            taxonomy, target_per_l2=target_per_l2, embedder=embedder
        )

    def analyze(
        self,
        archive: Archive,
    ) -> HealthReport:
        """Analyze taxonomy health from archive data."""
        accepted = archive.load_accepted()
        rejected = archive.load_rejected()

        l2_metrics = self.health_calculator.compute_l2_metrics(accepted, rejected)

        overall_coverage = np.mean([m.coverage for m in l2_metrics.values()]) if l2_metrics else 0.0
        overall_accept_rate = np.mean([m.accept_rate for m in l2_metrics.values()]) if l2_metrics else 0.5
        overall_diversity = np.mean([m.avg_diversity for m in l2_metrics.values()]) if l2_metrics else 0.8
        overall_health = np.mean([m.health_score for m in l2_metrics.values()]) if l2_metrics else 50.0

        underperforming = [
            l2_key for l2_key, m in l2_metrics.items()
            if m.health_score < 50.0
        ]

        expansion_targets = self._identify_expansion_targets(l2_metrics)

        return HealthReport(
            l2_metrics=l2_metrics,
            overall_coverage=float(overall_coverage),
            overall_accept_rate=float(overall_accept_rate),
            overall_diversity=float(overall_diversity),
            overall_health=float(overall_health),
            underperforming=sorted(underperforming, key=lambda k: l2_metrics[k].health_score),
            expansion_targets=expansion_targets,
        )

    def _identify_expansion_targets(
        self, l2_metrics: dict[str, L2Health]
    ) -> list[str]:
        """Identify L2 nodes that should be expanded."""
        targets: list[tuple[str, float]] = []

        for l2_key, m in l2_metrics.items():
            if not m.knowledge_gap and m.leaf_count >= 10:
                continue
            if m.coverage < 0.5 and m.accept_rate > 0.6:
                priority = (1.0 - m.coverage) * m.accept_rate * 100
                targets.append((l2_key, priority))
            elif m.coverage < 0.3:
                priority = (1.0 - m.coverage) * 50
                targets.append((l2_key, priority))

        targets.sort(key=lambda x: x[1], reverse=True)
        return [l2_key for l2_key, _ in targets[:20]]
