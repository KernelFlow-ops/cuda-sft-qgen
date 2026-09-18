"""Taxonomy leaf expansion module."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from cuda_sft_qgen.llm import LLMClient
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.taxonomy import Taxonomy
from cuda_sft_qgen.taxonomy_health.health import HealthReport
from cuda_sft_qgen.taxonomy_expand import (
    existing_leaves_payload,
    normalize_leaf_item,
    validate_leaf,
    render_expand_user,
)

logger = logging.getLogger(__name__)


def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)


def _render(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value)
    return out


class LeafExpander:
    """Expand taxonomy leaves based on health analysis."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        llm: LLMClient,
        *,
        priority: int = 50,
        embedder: Any = None,
    ):
        """Bind taxonomy, LLM, and optional embedder used to filter near-dup leaves."""
        self.taxonomy = taxonomy
        self.llm = llm
        self.priority = priority
        self.embedder = embedder

    def identify_expansion_targets(
        self,
        health_report: HealthReport,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Identify L2 nodes that need expansion based on health report."""
        targets: list[tuple[str, dict[str, Any]]] = []

        for l2_key in health_report.expansion_targets:
            metric = health_report.l2_metrics.get(l2_key)
            if not metric:
                continue

            reason = []
            if metric.coverage < 0.5:
                reason.append(f"low_coverage_{metric.coverage:.2f}")
            if metric.accept_rate > 0.6:
                reason.append(f"good_accept_rate_{metric.accept_rate:.2f}")
            if metric.leaf_count < 10:
                reason.append(f"sparse_leaves_{metric.leaf_count}")

            targets.append((
                l2_key,
                {
                    "reason": ",".join(reason),
                    "coverage": metric.coverage,
                    "accept_rate": metric.accept_rate,
                    "current_leaves": metric.leaf_count,
                    "target_leaves": metric.expected_quota,
                }
            ))

        return targets

    def expand_l2(
        self,
        l2_key: str,
        *,
        max_new: int = 10,
        facet_id: str = "mechanism",
        dry_run: bool = False,
        near_dup_warnings: str = "",
    ) -> list[dict[str, Any]]:
        """Expand a specific L2 node with new leaves."""
        parts = l2_key.split(".", 1)
        if len(parts) != 2:
            logger.warning("invalid l2_key: %s", l2_key)
            return []

        domain_id, l2_id = parts
        domain = next(
            (d for d in self.taxonomy.domains if d["id"] == domain_id),
            None,
        )
        if not domain:
            logger.warning("domain not found: %s", domain_id)
            return []

        child = next(
            (c for c in domain.get("children", []) if c["id"] == l2_id),
            None,
        )
        if not child:
            logger.warning("l2 not found: %s", l2_id)
            return []

        existing_leaves = existing_leaves_payload(child)
        negative_examples = [
            lf["claim"] for lf in existing_leaves
            if not lf.get("retired")
        ][:20]

        try:
            curator = load_prompt("curator")
            template = load_prompt("expand")
        except FileNotFoundError:
            logger.warning("expand prompt not found")
            return []

        facet_info = self._get_facet_info(facet_id)
        system = self._build_system_prompt(curator, facet_info)

        user = render_expand_user(
            domain=domain,
            child=child,
            facet=facet_info,
            max_new=max_new,
            template=template,
            near_dup_warnings=near_dup_warnings,
        )

        if dry_run:
            return [{
                "dry_run": True,
                "l2_key": l2_key,
                "max_new": max_new,
                "facet": facet_id,
                "existing_count": len(existing_leaves),
            }]

        try:
            result = self.llm.complete_json(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                retries=1,
                priority=self.priority,
            )
        except Exception as exc:
            logger.warning("llm expansion failed for %s: %s", l2_key, exc)
            return []

        if not isinstance(result, dict):
            return []

        raw_leaves = result.get("new_leaves") or result.get("leaves") or []
        if not isinstance(raw_leaves, list):
            return []

        validated = self._validate_and_normalize(
            raw_leaves,
            domain=domain,
            child=child,
            max_new=max_new,
        )

        logger.info(
            "expanded %s: %d candidates -> %d validated",
            l2_key,
            len(raw_leaves),
            len(validated),
        )

        return validated

    def _get_facet_info(self, facet_id: str) -> dict[str, Any]:
        """Get facet configuration."""
        try:
            from cuda_sft_qgen.prompts import load_prompt_yaml
            raw = load_prompt_yaml("facets")
            if isinstance(raw, dict):
                facets = raw.get("facets", [])
                for f in facets:
                    if isinstance(f, dict) and f.get("id") == facet_id:
                        return f
        except Exception as exc:
            logger.warning("facets load failed: %s", exc)

        return {"id": facet_id}

    def _build_system_prompt(self, curator: str, facet: dict[str, Any]) -> str:
        """Build system prompt with facet information."""
        extra = (
            f"\n\nEXPANSION FACET: {facet.get('id')}\n"
            f"WHEN: {facet.get('when', '')}\n"
            f"MUST: {facet.get('must', '')}\n"
            f"MUST NOT: {facet.get('must_not', '')}\n"
            f"EXAMPLE: {facet.get('example_claim', '')}\n"
            f"BANNED: {facet.get('banned_claim', '')}\n"
        )
        return curator + extra

    def _validate_and_normalize(
        self,
        raw_leaves: list[dict[str, Any]],
        *,
        domain: dict[str, Any],
        child: dict[str, Any],
        max_new: int,
    ) -> list[dict[str, Any]]:
        """Validate and normalize leaf candidates."""
        existing_ids = {leaf.l3_id for leaf in self.taxonomy.leaves}
        existing_claims = [leaf.claim for leaf in self.taxonomy.leaves]

        validated: list[dict[str, Any]] = []
        for raw in raw_leaves:
            if len(validated) >= max_new:
                break

            item = dict(raw)
            item["domain_id"] = domain["id"]
            item["domain_name"] = domain.get("name", domain["id"])
            item["l2_id"] = child["id"]
            item["l2_name"] = child.get("name", child["id"])

            item = normalize_leaf_item(item, l2_id=child["id"])

            # Ensure l3_id is set (normalize_leaf_item may leave it in 'id')
            if "l3_id" not in item and "id" in item:
                item["l3_id"] = item["id"]

            reason = validate_leaf(
                item,
                l2_id=child["id"],
                existing_ids=existing_ids | {
                    x.get("l3_id") or x.get("id") for x in validated
                },
                existing_claims=existing_claims + [
                    x.get("claim", "") for x in validated
                ],
            )

            if reason:
                logger.debug("dropped leaf %s: %s", item.get("l3_id"), reason)
                continue

            validated.append(item)

        if self.embedder is not None and validated:
            from cuda_sft_qgen.taxonomy_embed import TaxonomyEmbedIndex
            from pathlib import Path
            index = TaxonomyEmbedIndex(Path("unused-taxonomy-index.npz"), self.embedder)
            index.rebuild(self.taxonomy.leaves, save=False)
            validated, _, _ = index.filter_candidates(validated)
        return validated

    def preview_expansion(
        self,
        health_report: HealthReport,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Preview expansion suggestions without executing."""
        targets = self.identify_expansion_targets(health_report)[:limit]

        preview: list[dict[str, Any]] = []
        for l2_key, metadata in targets:
            metric = health_report.l2_metrics.get(l2_key)
            if not metric:
                continue

            need = max(0, metric.expected_quota - metric.leaf_count)
            need = min(need, 20)

            preview.append({
                "l2_key": l2_key,
                "current_leaves": metadata["current_leaves"],
                "current_questions": metric.question_count,
                "target_questions": metric.expected_quota,
                "suggested_new_leaves": need,
                "reason": metadata["reason"],
                "coverage": metadata["coverage"],
                "accept_rate": metadata["accept_rate"],
                "health_score": metric.health_score,
            })

        return preview
