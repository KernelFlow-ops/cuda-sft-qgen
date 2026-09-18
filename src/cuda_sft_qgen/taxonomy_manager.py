"""Strategic taxonomy manager: health reports, facet expansion, and merge/refine."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from cuda_sft_qgen.keys import PRI_TAXONOMY
from cuda_sft_qgen.llm import LLMClient
from cuda_sft_qgen.models import DiversityReport, Question
from cuda_sft_qgen.prompts import load_prompt, load_prompt_yaml
from cuda_sft_qgen.taxonomy import Taxonomy, dump_taxonomy, merge_leaves, patch_leaf
from cuda_sft_qgen.taxonomy_embed import TaxonomyEmbedIndex
from cuda_sft_qgen.taxonomy_expand import (
    existing_leaves_payload,
    normalize_leaf_item,
    render_expand_user,
    validate_leaf,
)

logger = logging.getLogger(__name__)

DEFAULT_FACET_IDS = (
    "formula",
    "derivation",
    "failure",
    "feature",
    "mechanism",
    "api",
    "numeric",
    "layout",
    "profiling",
    "arch_migrate",
    "cross_factor",
)


def load_facets() -> list[dict]:
    """Load expansion facet cards from ``prompts/taxonomy/facets.yaml``."""
    try:
        raw = load_prompt_yaml("facets")
    except FileNotFoundError:
        logger.warning("taxonomy facets.yaml missing")
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("taxonomy facets.yaml failed: %s", exc)
        return []
    if not isinstance(raw, dict):
        return []
    return [x for x in (raw.get("facets") or []) if isinstance(x, dict)]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _render(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _active_leaf_count(child: dict[str, Any]) -> int:
    return sum(1 for item in (child.get("leaves") or []) if not item.get("retired"))


def _new_leaves(obj: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    raw = obj.get("new_leaves") or obj.get("leaves") or []
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _manager_cfg(settings: Any | None) -> dict[str, Any]:
    if settings is None:
        return {}
    tax_cfg = (getattr(settings, "default_yaml", {}) or {}).get("taxonomy") or {}
    return dict(tax_cfg.get("manager") or {})


@dataclass
class TaxonomyHealthReport:
    """Comprehensive taxonomy health metrics."""

    # Coverage metrics
    total_leaves: int
    total_l2_nodes: int
    empty_l2_nodes: list[str]  # L2 with 0 leaves
    sparse_l2_nodes: list[tuple[str, int]]  # L2 with <10 leaves: (l2_key, count)
    dense_l2_nodes: list[tuple[str, int]]  # L2 with >50 leaves: (l2_key, count)

    # Balance metrics
    l1_leaf_distribution: dict[str, int]  # L1 -> leaf count
    l1_imbalance_ratio: float  # max(l1_counts) / mean(l1_counts)

    # Quality metrics
    leaves_without_hints: int  # Leaves with empty hints field
    leaves_with_short_claims: list[str]  # l3_id with claim <30 chars
    leaves_with_generic_claims: list[str]  # Claims that don't specify mechanism

    # Kind distribution
    kind_distribution: dict[str, int]  # kind -> count
    kind_imbalance: bool  # True if any kind is >50% of active leaves

    # Generation performance (from question data)
    leaves_with_no_questions: list[str]  # l3_id with 0 accepted questions
    leaves_with_high_reject_rate: list[tuple[str, float]]  # (l3_id, reject_rate) >0.7

    # Recommendations
    expansion_priority: list[str]  # L2 keys to expand first
    refinement_needed: list[str]  # L3 IDs that need claim rewrites

    # Overall health score (0-100)
    health_score: float
    question_coverage: float = 0.0
    knowledge_gap_l2: list[str] | None = None


def manager_from_runtime(rt: Any) -> "TaxonomyManager":
    """Construct a :class:`TaxonomyManager` from the live runtime."""
    tax_cfg = (getattr(rt.settings, "default_yaml", {}) or {}).get("taxonomy") or {}
    return TaxonomyManager(
        rt.taxonomy,
        rt.llm,
        embedder=getattr(rt, "embedder", None),
        settings=rt.settings,
        target_leaves_per_l2=int(tax_cfg.get("target_leaves_per_l2", 30) or 30),
        min_leaves_per_l2=10,
        max_leaves_per_l2=50,
    )


class TaxonomyManager:
    """Strategic taxonomy management with health analysis and expansion planning."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        llm: LLMClient,
        *,
        embedder: Any = None,
        embed_index: TaxonomyEmbedIndex | None = None,
        settings: Any = None,
        target_leaves_per_l2: int = 30,
        min_leaves_per_l2: int = 10,
        max_leaves_per_l2: int = 50,
    ):
        """Bind taxonomy, LLM, leaf-count targets, and manager YAML knobs."""
        self.taxonomy = taxonomy
        self.llm = llm
        self.embedder = embedder
        self.settings = settings
        self.target_leaves_per_l2 = int(target_leaves_per_l2)
        self.min_leaves_per_l2 = int(min_leaves_per_l2)
        self.max_leaves_per_l2 = int(max_leaves_per_l2)

        mgr = _manager_cfg(settings)
        self.enabled = bool(mgr.get("enabled", True))
        self.every_k_reviews = int(mgr.get("every_k_reviews", 5))
        self.max_llm_calls_per_tick = int(mgr.get("max_llm_calls_per_tick", 2))
        self.max_leaves_per_tick = int(mgr.get("max_leaves_per_tick", 12))
        self.health_expand_threshold = float(mgr.get("health_expand_threshold", 70))
        self.leaf_cosine_dup = float(mgr.get("leaf_cosine_dup", 0.92))
        self.leaf_cosine_gray = float(mgr.get("leaf_cosine_gray", 0.85))
        self.allow_refine_in_pipeline = bool(mgr.get("allow_refine_in_pipeline", False))
        self.allow_merge_in_pipeline = bool(mgr.get("allow_merge_in_pipeline", False))
        self.allow_new_domains = bool(mgr.get("allow_new_domains", False))
        facets = mgr.get("facets") or list(DEFAULT_FACET_IDS)
        self.facet_ids = [str(x) for x in facets] or list(DEFAULT_FACET_IDS)

        self.embed_index = embed_index
        if self.embed_index is None and embedder is not None and settings is not None:
            data_dir = getattr(settings, "data_dir", None)
            if data_dir is not None:
                self.embed_index = TaxonomyEmbedIndex(
                    Path(data_dir) / "taxonomy_embed.npz",
                    embedder,
                    cosine_dup=self.leaf_cosine_dup,
                    cosine_gray=self.leaf_cosine_gray,
                )
                try:
                    self.embed_index.load()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("taxonomy embed load failed: %s", exc)

        self._llm_calls = 0
        self._in_tick = False
        self._facets_cache: list[dict[str, Any]] | None = None
        self._last_gray_pairs: list[dict[str, Any]] = []
        self._last_dropped = 0

    def analyze_health(
        self,
        accepted_questions: Sequence[Question] | None = None,
        rejected_questions: Sequence[Question] | None = None,
        *,
        accepted: Sequence[Question] | None = None,
        rejected: Sequence[Question] | None = None,
    ) -> TaxonomyHealthReport:
        """Comprehensive taxonomy health analysis."""
        if accepted_questions is None:
            accepted_questions = accepted
        if rejected_questions is None:
            rejected_questions = rejected

        l2_leaf_counts: dict[str, int] = {}
        for domain in self.taxonomy.domains:
            for child in domain.get("children", []):
                l2_key = f"{domain['id']}.{child['id']}"
                l2_leaf_counts[l2_key] = _active_leaf_count(child)

        empty_l2 = [k for k, v in l2_leaf_counts.items() if v == 0]
        sparse_l2 = [
            (k, v) for k, v in l2_leaf_counts.items() if 0 < v < self.min_leaves_per_l2
        ]
        dense_l2 = [
            (k, v) for k, v in l2_leaf_counts.items() if v > self.max_leaves_per_l2
        ]

        l1_counts: dict[str, int] = {}
        for leaf in self.taxonomy.leaves:
            if leaf.retired:
                continue
            l1_counts[leaf.domain_id] = l1_counts.get(leaf.domain_id, 0) + 1

        mean_l1 = sum(l1_counts.values()) / max(len(l1_counts), 1)
        max_l1 = max(l1_counts.values()) if l1_counts else 0
        imbalance_ratio = max_l1 / mean_l1 if mean_l1 > 0 else 1.0

        no_hints = sum(
            1 for leaf in self.taxonomy.leaves if not leaf.retired and not leaf.hints
        )
        short_claims = [
            leaf.l3_id
            for leaf in self.taxonomy.leaves
            if not leaf.retired and len(leaf.claim) < 30
        ]
        generic_claims = self._find_generic_claims()

        kind_counts: dict[str, int] = {}
        for leaf in self.taxonomy.leaves:
            if leaf.retired:
                continue
            k = leaf.kind or "mechanism"
            kind_counts[k] = kind_counts.get(k, 0) + 1

        total_leaves = sum(1 for leaf in self.taxonomy.leaves if not leaf.retired)
        kind_imbalance = any(
            count > 0.5 * total_leaves for count in kind_counts.values()
        ) if total_leaves else False

        no_questions: list[str] = []
        high_reject: list[tuple[str, float]] = []

        if accepted_questions is not None:
            accepted_by_leaf: dict[str, int] = {}
            for q in accepted_questions:
                accepted_by_leaf[q.l3_id] = accepted_by_leaf.get(q.l3_id, 0) + 1

            no_questions = [
                leaf.l3_id
                for leaf in self.taxonomy.leaves
                if not leaf.retired and accepted_by_leaf.get(leaf.l3_id, 0) == 0
            ]

            if rejected_questions is not None:
                rejected_by_leaf: dict[str, int] = {}
                for q in rejected_questions:
                    rejected_by_leaf[q.l3_id] = rejected_by_leaf.get(q.l3_id, 0) + 1

                for l3_id in set(accepted_by_leaf) | set(rejected_by_leaf):
                    acc = accepted_by_leaf.get(l3_id, 0)
                    rej = rejected_by_leaf.get(l3_id, 0)
                    total = acc + rej
                    if total >= 3 and rej / total > 0.7:
                        high_reject.append((l3_id, rej / total))

        priority = self._compute_expansion_priority(
            empty_l2, sparse_l2, l2_leaf_counts, accepted_questions
        )
        refinement = short_claims + generic_claims + [l3_id for l3_id, _ in high_reject]
        score = self._compute_health_score(
            empty_l2, sparse_l2, imbalance_ratio, no_hints,
            short_claims, kind_imbalance, no_questions
        )

        return TaxonomyHealthReport(
            total_leaves=total_leaves,
            total_l2_nodes=len(l2_leaf_counts),
            empty_l2_nodes=empty_l2,
            sparse_l2_nodes=sorted(sparse_l2, key=lambda x: x[1]),
            dense_l2_nodes=sorted(dense_l2, key=lambda x: x[1], reverse=True),
            l1_leaf_distribution=l1_counts,
            l1_imbalance_ratio=imbalance_ratio,
            leaves_without_hints=no_hints,
            leaves_with_short_claims=short_claims[:20],
            leaves_with_generic_claims=generic_claims[:20],
            kind_distribution=kind_counts,
            kind_imbalance=kind_imbalance,
            leaves_with_no_questions=no_questions[:20],
            leaves_with_high_reject_rate=sorted(high_reject, key=lambda x: x[1], reverse=True)[:10],
            expansion_priority=priority,
            refinement_needed=refinement[:20],
            health_score=score,
            question_coverage=(1.0 - len(no_questions) / total_leaves) if accepted_questions is not None and total_leaves else 0.0,
            knowledge_gap_l2=empty_l2 + [key for key, _ in sparse_l2],
        )

    def _find_generic_claims(self) -> list[str]:
        """Find claims that are too generic (no specific mechanism/constraint)."""
        generic_patterns = [
            r"^(提升|improves?) (performance|性能)",
            r"^(是一种|is a (type|kind) of)",
            r"^(用于|used for|可以|can be used)",
            r"^CUDA (提供|provides|支持|supports)",
        ]
        generic = []
        for leaf in self.taxonomy.leaves:
            if leaf.retired:
                continue
            claim_lower = leaf.claim.lower()
            if any(re.search(pat, claim_lower, re.IGNORECASE) for pat in generic_patterns):
                if not re.search(r"\d+|sm_\d+|hopper|ampere|blackwell", claim_lower):
                    generic.append(leaf.l3_id)
        return generic

    def _compute_expansion_priority(
        self,
        empty_l2: list[str],
        sparse_l2: list[tuple[str, int]],
        l2_counts: dict[str, int],
        questions: Sequence[Question] | None,
    ) -> list[str]:
        """Compute expansion priority order based on multiple factors."""
        priority = list(empty_l2)

        if questions:
            l2_with_questions = set()
            for q in questions:
                if len(q.topic_path) >= 2:
                    l2_key = f"{q.topic_path[0]}.{q.topic_path[1]}"
                    l2_with_questions.add(l2_key)

            sparse_no_questions = [
                k for k, count in sparse_l2 if k not in l2_with_questions
            ]
            priority.extend(sparse_no_questions)

        remaining_sparse = [k for k, _ in sparse_l2 if k not in priority]
        priority.extend(remaining_sparse)

        below_target = [
            k
            for k, v in sorted(l2_counts.items(), key=lambda kv: kv[1])
            if v < self.target_leaves_per_l2 and k not in priority
        ]
        priority.extend(below_target)

        return priority[:20]

    def _compute_health_score(
        self,
        empty_l2: list[str],
        sparse_l2: list[tuple[str, int]],
        imbalance_ratio: float,
        no_hints: int,
        short_claims: list[str],
        kind_imbalance: bool,
        no_questions: list[str],
    ) -> float:
        """Compute overall health score 0-100."""
        score = 100.0
        score -= min(len(empty_l2) * 2.0, 20.0)
        score -= min(len(sparse_l2) * 0.5, 25.0)
        if imbalance_ratio > 3.0:
            score -= 10.0
        elif imbalance_ratio > 2.0:
            score -= 5.0
        score -= min(no_hints * 0.1, 10.0)
        score -= min(len(short_claims) * 0.5, 10.0)
        if kind_imbalance:
            score -= 5.0
        return max(0.0, score)

    def choose_facet(
        self,
        health: TaxonomyHealthReport,
        last_facet: str | None = None,
    ) -> str:
        """Pick one facet id. Rotate away from last_facet."""
        ids = list(self.facet_ids) or list(DEFAULT_FACET_IDS)
        formula_n = int((health.kind_distribution or {}).get("formula", 0) or 0)
        total = max(int(health.total_leaves or 0), 1)
        few_formula = formula_n < 0.05 * total
        if health.kind_imbalance or few_formula:
            preferred = [x for x in ids if x in {"formula", "numeric"}] or [
                "formula",
                "numeric",
            ]
        elif health.leaves_with_high_reject_rate:
            preferred = [x for x in ids if x == "failure"] or ["failure"]
        else:
            preferred = ids
        if preferred == ids and last_facet in ids:
            return ids[(ids.index(last_facet) + 1) % len(ids)]
        options = [x for x in preferred if x != last_facet]
        return (options or preferred or ids)[0]

    def _facet_defs(self) -> list[dict[str, Any]]:
        if self._facets_cache is None:
            self._facets_cache = load_facets()
        return self._facets_cache

    def _facet_dict(self, facet_id: str | None) -> dict[str, Any]:
        fid = str(facet_id or "")
        for item in self._facet_defs():
            if str(item.get("id")) == fid:
                return item
        return {"id": fid}

    def _system_for_facet(self, facet: dict) -> str:
        curator = load_prompt("curator")
        extra = (
            f"\n\nEXPANSION FACET: {facet.get('id')}\n"
            f"WHEN: {facet.get('when')}\n"
            f"MUST: {facet.get('must')}\n"
            f"MUST NOT: {facet.get('must_not')}\n"
            f"EXAMPLE: {facet.get('example_claim')}\n"
            f"BANNED: {facet.get('banned_claim')}\n"
        )
        return curator + extra

    def _complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        count: bool | None = None,
    ) -> dict[str, Any]:
        use_budget = self._in_tick if count is None else count
        if use_budget:
            if self._llm_calls >= self.max_llm_calls_per_tick:
                logger.warning("taxonomy llm budget exhausted")
                return {}
            self._llm_calls += 1
        try:
            obj = self.llm.complete_json(
                messages, retries=1, priority=PRI_TAXONOMY
            )
            return obj if isinstance(obj, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("taxonomy llm failed: %s", exc)
            return {}

    def _lookup_l2(
        self, l2_key: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        parts = str(l2_key).split(".", 1)
        if len(parts) != 2:
            return None
        domain_id, l2_id = parts
        domain = next(
            (d for d in self.taxonomy.domains if str(d.get("id")) == domain_id),
            None,
        )
        if not domain:
            return None
        child = next(
            (
                c
                for c in (domain.get("children") or [])
                if str(c.get("id")) == l2_id
            ),
            None,
        )
        if not child:
            return None
        return domain, child

    def _underrep_keys(self, diversity_report: Any) -> list[str]:
        if diversity_report is None:
            return []
        if isinstance(diversity_report, dict):
            return [str(x) for x in (diversity_report.get("underrepresented") or [])]
        return [str(x) for x in (getattr(diversity_report, "underrepresented", None) or [])]

    def _reject_reason_hist(self, rejected: Sequence[Question] | None) -> dict[str, int]:
        hist: Counter[str] = Counter()
        for q in rejected or []:
            reason = getattr(q, "reject_reason", None) or ""
            if reason:
                hist[str(reason)] += 1
        return dict(hist)

    def _sample_leaves(
        self, health: TaxonomyHealthReport, n: int = 24
    ) -> list[dict[str, Any]]:
        wanted = set(health.leaves_with_short_claims[:8])
        wanted.update(health.leaves_with_generic_claims[:8])
        wanted.update(health.refinement_needed[:8])
        wanted.update(l3_id for l3_id, _ in health.leaves_with_high_reject_rate)
        samples: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _row(leaf: Any) -> dict[str, Any]:
            return {
                "l3_id": leaf.l3_id,
                "l2": f"{leaf.domain_id}.{leaf.l2_id}",
                "kind": leaf.kind or "mechanism",
                "claim": leaf.claim,
                "retired": bool(leaf.retired),
            }

        for leaf in self.taxonomy.leaves:
            if leaf.l3_id not in wanted:
                continue
            samples.append(_row(leaf))
            seen.add(leaf.l3_id)
            if len(samples) >= n:
                return samples
        for leaf in self.taxonomy.leaves:
            if leaf.l3_id in seen or leaf.retired:
                continue
            samples.append(_row(leaf))
            if len(samples) >= n:
                break
        return samples

    def _health_summary(self, health: TaxonomyHealthReport) -> dict[str, Any]:
        return {
            "health_score": health.health_score,
            "total_leaves": health.total_leaves,
            "empty_l2_nodes": health.empty_l2_nodes,
            "sparse_l2_nodes": [list(x) for x in health.sparse_l2_nodes],
            "dense_l2_nodes": [list(x) for x in health.dense_l2_nodes],
            "kind_distribution": health.kind_distribution,
            "kind_imbalance": health.kind_imbalance,
            "leaves_with_short_claims": health.leaves_with_short_claims,
            "leaves_with_generic_claims": health.leaves_with_generic_claims,
            "leaves_with_high_reject_rate": [list(x) for x in health.leaves_with_high_reject_rate],
            "leaves_with_no_questions": health.leaves_with_no_questions,
            "expansion_priority": health.expansion_priority,
            "l1_imbalance_ratio": health.l1_imbalance_ratio,
            "question_coverage": health.question_coverage,
            "knowledge_gap_l2": health.knowledge_gap_l2,
        }

    def _validate_and_filter(
        self,
        raw_leaves: list[dict[str, Any]],
        *,
        domain: dict[str, Any],
        child: dict[str, Any],
        max_new: int,
    ) -> list[dict[str, Any]]:
        if max_new <= 0:
            return []
        existing_ids = {leaf.l3_id for leaf in self.taxonomy.leaves}
        claims = [leaf.claim for leaf in self.taxonomy.leaves]
        accepted: list[dict[str, Any]] = []
        drop_reasons: list[str] = []
        for raw in raw_leaves:
            item = dict(raw)
            item["domain_id"] = domain.get("id")
            item["domain_name"] = domain.get("name")
            item["l2_id"] = child.get("id")
            item["l2_name"] = child.get("name")
            item = normalize_leaf_item(item, l2_id=str(child.get("id") or ""))
            reason = validate_leaf(
                item,
                l2_id=str(child.get("id")),
                existing_ids=existing_ids
                | {str(x.get("l3_id") or x.get("id") or "") for x in accepted},
                existing_claims=claims + [str(x.get("claim") or "") for x in accepted],
            )
            if reason:
                drop_reasons.append(f"{item.get('l3_id')}:{reason}")
                logger.info("drop leaf %s: %s", item.get("l3_id"), reason)
                continue
            accepted.append(item)
            if len(accepted) >= max_new:
                break
        if raw_leaves and not accepted:
            logger.warning(
                "taxonomy filter dropped all %s candidates (%s)",
                len(raw_leaves),
                "; ".join(drop_reasons[:12]) or "empty_payload",
            )
        gray_pairs: list[dict[str, Any]] = []
        dropped = 0
        if self.embed_index is not None and accepted:
            try:
                keep, dropped_dups, gray_pairs = self.embed_index.filter_candidates(
                    accepted
                )
                dropped = len(dropped_dups)
                accepted = list(keep)
            except Exception as exc:  # noqa: BLE001
                logger.warning("taxonomy embed filter failed: %s", exc)
        if len(accepted) > max_new:
            accepted = accepted[:max_new]
        self._last_gray_pairs = gray_pairs
        self._last_dropped = dropped
        return accepted

    def _commit_leaves(self, accepted: list[dict[str, Any]], *, apply: bool) -> int:
        if not accepted:
            return 0
        added = merge_leaves(
            self.taxonomy,
            accepted,
            allow_new_domains=self.allow_new_domains,
        )
        if added <= 0:
            return 0
        if self.embed_index is not None:
            try:
                had_corpus = self.embed_index.ensure_loaded()
                self.embed_index.add_leaves(accepted)
                if apply and had_corpus:
                    self.embed_index.save()
            except Exception as exc:  # noqa: BLE001
                logger.warning("taxonomy embed update failed: %s", exc)
        if apply:
            self.taxonomy.version = int(self.taxonomy.version) + 1
            try:
                dump_taxonomy(self.taxonomy)
            except Exception as exc:  # noqa: BLE001
                logger.warning("taxonomy dump failed: %s", exc)
        return added

    def _expand_l2(
        self,
        *,
        domain: dict[str, Any],
        child: dict[str, Any],
        facet: dict[str, Any],
        max_new: int,
        near_dup_warnings: str = "",
    ) -> list[dict[str, Any]]:
        from cuda_sft_qgen.keys import PRI_TAXONOMY
        from cuda_sft_qgen.taxonomy_health.expander import LeafExpander

        expander = LeafExpander(
            self.taxonomy,
            self.llm,
            embedder=self.embedder,
            priority=PRI_TAXONOMY,
        )
        l2_key = f"{domain.get('id')}.{child.get('id')}"
        facet_id = str(facet.get("id") or "mechanism")
        return expander.expand_l2(
            l2_key,
            max_new=max_new,
            facet_id=facet_id,
            near_dup_warnings=near_dup_warnings,
        )

    def llm_health_review(
        self,
        health: TaxonomyHealthReport | None = None,
        *,
        report: TaxonomyHealthReport | None = None,
        accepted: Sequence[Question] | None = None,
        rejected: Sequence[Question] | None = None,
    ) -> dict:
        """Ask the curator LLM for health notes given a TaxonomyHealthReport."""
        health = health or report
        if health is None:
            return {}
        try:
            curator = load_prompt("curator")
            template = load_prompt("health")
        except FileNotFoundError:
            logger.warning("taxonomy health prompt missing")
            return {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("taxonomy health prompt failed: %s", exc)
            return {}
        user = _render(
            template,
            {
                "health_summary_json": _dumps(self._health_summary(health)),
                "sample_leaves_json": _dumps(self._sample_leaves(health)),
                "reject_reason_hist_json": _dumps(self._reject_reason_hist(rejected)),
            },
        )
        return self._complete_json(
            [
                {"role": "system", "content": curator},
                {"role": "user", "content": user},
            ]
        )

    def gap_search(
        self,
        *,
        target_l2_key: str,
        health: TaxonomyHealthReport,
        rejected: Sequence[Question] | None,
        max_new: int,
        facet: str | dict[str, Any] | None,
    ) -> list[dict]:
        """Propose leaves that fill a coverage/reject gap on one L2 node."""
        pair = self._lookup_l2(target_l2_key)
        if pair is None:
            return []
        domain, child = pair
        if isinstance(facet, dict):
            facet_d = facet
        else:
            facet_d = self._facet_dict(str(facet or ""))
        try:
            system = self._system_for_facet(facet_d)
            template = load_prompt("gap_search")
        except FileNotFoundError:
            logger.warning("taxonomy gap_search prompt missing")
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("taxonomy gap_search prompt failed: %s", exc)
            return []
        under = list(self._underrep_kinds(health))
        queries = self._heuristic_search_queries(health, child, target_l2_key)
        high_reject = []
        for l3_id, rate in health.leaves_with_high_reject_rate:
            leaf = self.taxonomy.leaf_by_key(l3_id)
            high_reject.append(
                {
                    "l3_id": l3_id,
                    "claim": leaf.claim if leaf else "",
                    "reject_rate": rate,
                }
            )
        user = _render(
            template,
            {
                "target_l2_id": str(child.get("id") or ""),
                "target_l2_name": str(child.get("name") or child.get("id") or ""),
                "domain_id": str(domain.get("id") or ""),
                "domain_name": str(domain.get("name") or domain.get("id") or ""),
                "search_queries": _dumps(queries),
                "underrepresented": _dumps(under),
                "high_reject_leaves_json": _dumps(high_reject),
                "reject_reason_hist_json": _dumps(self._reject_reason_hist(rejected)),
                "missing_token_notes": _dumps(self._missing_token_notes(health)),
                "existing_leaves_json": _dumps(existing_leaves_payload(child)),
                "facet_must": str(facet_d.get("must") or ""),
                "max_new_leaves": str(max_new),
            },
        )
        obj = self._complete_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        raw = _new_leaves(obj)
        logger.info(
            "taxonomy llm returned %s new_leaves for %s.%s",
            len(raw),
            domain.get("id"),
            child.get("id"),
        )
        return self._validate_and_filter(
            raw, domain=domain, child=child, max_new=max_new
        )

    def _underrep_kinds(self, health: TaxonomyHealthReport) -> list[str]:
        out: list[str] = []
        total = max(int(health.total_leaves or 0), 1)
        dist = health.kind_distribution or {}
        if health.kind_imbalance:
            out.append("kind_imbalance")
        for kind in ("formula", "failure", "feature", "numeric"):
            if dist.get(kind, 0) < 0.05 * total:
                out.append(kind)
        return out

    def _missing_token_notes(self, health: TaxonomyHealthReport) -> list[str]:
        notes: list[str] = []
        if health.leaves_with_generic_claims:
            notes.append("generic claims lack specific CUDA tokens")
        if health.kind_imbalance or (health.kind_distribution or {}).get("formula", 0) == 0:
            notes.append("formula identities with = or → underrepresented")
        if health.leaves_with_high_reject_rate:
            notes.append("high-reject leaves may lack answerable constraints")
        return notes

    def _heuristic_search_queries(
        self,
        health: TaxonomyHealthReport,
        child: dict[str, Any],
        target_l2_key: str,
    ) -> list[str]:
        l2_name = str(child.get("name") or child.get("id") or "")
        queries: list[str] = []
        if target_l2_key in health.empty_l2_nodes:
            queries.append(f"{l2_name} core mechanisms formulas failure modes")
        dist = health.kind_distribution or {}
        total = max(int(health.total_leaves or 0), 1)
        if dist.get("formula", 0) < 0.05 * total:
            queries.append(f"{l2_name} occupancy offset bank transaction formula")
        if dist.get("failure", 0) < 0.05 * total:
            queries.append(f"{l2_name} deadlock race UB mbarrier wait")
        if health.leaves_with_high_reject_rate:
            queries.append(f"{l2_name} answerable numeric constraints")
        queries.append(f"{l2_name} sm_90 sm_100 TMA alignment")
        return queries[:10]

    def persist_health(self, payload: dict, path: Path) -> None:
        """Write a JSON health snapshot (typically ``taxonomy_health.json``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    def expand_strategic(
        self,
        *,
        diversity_report: DiversityReport | None = None,
        max_leaves_to_add: int = 100,
        facets: list[str] | None = None,
        domain_ids: list[str] | None = None,
        accepted: Sequence[Question] | None = None,
        rejected: Sequence[Question] | None = None,
        max_llm_calls: int | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Expand high-priority L2 nodes using facet prompts; dump when ``apply``."""
        if diversity_report:
            targets = self._prioritize_from_diversity(diversity_report)
        else:
            health = self.analyze_health(
                accepted_questions=accepted, rejected_questions=rejected
            )
            targets = health.expansion_priority

        facet_ids = [str(x) for x in (facets or self.facet_ids)] or list(self.facet_ids)
        added_total = 0
        expansion_log: list[dict[str, Any]] = []
        call_cap = int(max_llm_calls) if max_llm_calls is not None else 3
        calls = 0
        zeros = 0

        for i, l2_key in enumerate(targets):
            if added_total >= max_leaves_to_add:
                break
            if calls >= max(0, call_cap):
                break
            pair = self._lookup_l2(l2_key)
            if not pair:
                continue
            domain, child = pair
            if domain_ids and str(domain.get("id")) not in domain_ids:
                continue
            current = _active_leaf_count(child)
            need = min(
                self.target_leaves_per_l2 - current,
                max_leaves_to_add - added_total,
                20,
            )
            if need <= 0:
                continue
            facet_id = facet_ids[i % len(facet_ids)]
            facet = self._facet_dict(facet_id)
            calls += 1
            accepted_leaves = self._expand_l2(
                domain=domain, child=child, facet=facet, max_new=need
            )
            n = self._commit_leaves(accepted_leaves, apply=apply)
            added_total += n
            expansion_log.append(
                {"l2_key": l2_key, "added": n, "facet": facet_id}
            )
            logger.info(
                "Expanded %s: +%s leaves (total_added=%s)",
                l2_key,
                n,
                added_total,
            )
            if n == 0:
                zeros += 1
                if zeros >= 3:
                    break
            else:
                zeros = 0

        return {
            "total_added": added_total,
            "expansions": expansion_log,
            "total_leaves": len(self.taxonomy.leaves),
        }

    def _prioritize_from_diversity(self, report: Any) -> list[str]:
        """Convert diversity report underrepresented list to L2 expansion targets."""
        targets: list[str] = []
        for key in self._underrep_keys(report):
            if key.count(".") == 1:
                targets.append(key)
            elif key.count(".") == 0:
                for domain in self.taxonomy.domains:
                    if domain["id"] == key:
                        for child in domain.get("children", []):
                            targets.append(f"{key}.{child['id']}")
        return targets

    def refine_leaf_quality(
        self,
        leaf_ids: list[str],
        *,
        rewrite_claims: bool = True,
        add_hints: bool = True,
    ) -> dict[str, Any]:
        """Suggest claim/hint rewrites for short or hint-less leaves."""
        suggestions: list[dict[str, Any]] = []
        payload_leaves: list[dict[str, Any]] = []
        for l3_id in leaf_ids:
            leaf = next((lf for lf in self.taxonomy.leaves if lf.l3_id == l3_id), None)
            if not leaf:
                continue
            issues: list[str] = []
            if len(leaf.claim) < 30:
                issues.append("claim_too_short")
            if add_hints and not leaf.hints:
                issues.append("missing_hints")
            if leaf.kind == "formula" and not re.search(r"[=→]|=>|->", leaf.claim):
                issues.append("formula_missing_equation")
            rec = {
                "l3_id": l3_id,
                "current_claim": leaf.claim,
                "issues": issues,
                "suggested_claim": None,
                "suggested_hints": None,
            }
            suggestions.append(rec)
            payload_leaves.append(
                {
                    "l3_id": l3_id,
                    "claim": leaf.claim,
                    "kind": leaf.kind,
                    "hints": list(leaf.hints),
                    "issues": issues,
                }
            )

        if payload_leaves and rewrite_claims:
            try:
                curator = load_prompt("curator")
                template = load_prompt("refine")
                user = _render(template, {"leaves_json": _dumps(payload_leaves)})
                obj = self._complete_json(
                    [
                        {"role": "system", "content": curator},
                        {"role": "user", "content": user},
                    ]
                )
                by_id: dict[str, dict[str, Any]] = {}
                for item in obj.get("suggestions") or []:
                    if isinstance(item, dict) and item.get("l3_id"):
                        by_id[str(item["l3_id"])] = item
                for rec in suggestions:
                    hit = by_id.get(rec["l3_id"])
                    if not hit:
                        continue
                    rec["suggested_claim"] = hit.get("suggested_claim")
                    rec["suggested_hints"] = hit.get("suggested_hints")
                    if isinstance(hit.get("issues"), list):
                        rec["issues"] = hit.get("issues")
            except FileNotFoundError:
                logger.warning("taxonomy refine prompt missing")
            except Exception as exc:  # noqa: BLE001
                logger.warning("taxonomy refine failed: %s", exc)

        return {
            "leaves_analyzed": len(leaf_ids),
            "suggestions": suggestions,
        }

    def apply_refine(self, result: dict[str, Any], *, apply: bool = True) -> int:
        """Patch suggested claims/hints; dump taxonomy when ``apply`` and any patched."""
        patched = 0
        for rec in result.get("suggestions") or []:
            if not isinstance(rec, dict):
                continue
            l3 = str(rec.get("l3_id") or "")
            claim = rec.get("suggested_claim")
            if not l3 or not claim:
                continue
            hints = rec.get("suggested_hints")
            hint_list = [str(h) for h in hints] if isinstance(hints, list) else None
            if patch_leaf(self.taxonomy, l3, claim=str(claim), hints=hint_list):
                patched += 1
        if apply and patched:
            self.taxonomy.version = int(self.taxonomy.version) + 1
            try:
                dump_taxonomy(self.taxonomy)
            except Exception as exc:  # noqa: BLE001
                logger.warning("taxonomy refine dump failed: %s", exc)
        return patched

    def _has_issues(self, health: TaxonomyHealthReport) -> bool:
        return bool(
            health.empty_l2_nodes
            or health.sparse_l2_nodes
            or health.leaves_with_short_claims
            or health.leaves_with_generic_claims
            or health.kind_imbalance
            or health.leaves_with_high_reject_rate
            or health.refinement_needed
            or health.leaves_with_no_questions
        )

    def _pick_expand_l2(self, health: TaxonomyHealthReport) -> str | None:
        for key in health.expansion_priority:
            pair = self._lookup_l2(key)
            if not pair:
                continue
            _, child = pair
            if _active_leaf_count(child) < self.target_leaves_per_l2:
                return key
        return None

    def _pick_gap_l2(
        self,
        health: TaxonomyHealthReport,
        diversity_report: Any,
    ) -> str | None:
        for key in health.expansion_priority:
            pair = self._lookup_l2(key)
            if pair and _active_leaf_count(pair[1]) < self.target_leaves_per_l2:
                return key
        for key in self._underrep_keys(diversity_report):
            pair = self._lookup_l2(key)
            if key.count(".") == 1 and pair and _active_leaf_count(pair[1]) < self.target_leaves_per_l2:
                return key
        for key in self._prioritize_from_diversity(diversity_report):
            pair = self._lookup_l2(key)
            if pair and _active_leaf_count(pair[1]) < self.target_leaves_per_l2:
                return key
        return None

    def _tick_result(
        self,
        *,
        health: TaxonomyHealthReport,
        total_added: int,
        facet: str | None,
        mission: str,
        expansions: list[dict[str, Any]],
        gray_pairs: list[dict[str, Any]],
        dropped_dups: int,
    ) -> dict[str, Any]:
        return {
            "status": "taxonomy_health_checked",
            "health_score": float(health.health_score),
            "total_added": int(total_added),
            "facet": facet,
            "mission": mission,
            "llm_calls": int(self._llm_calls),
            "expansions": expansions,
            "gray_pairs": gray_pairs,
            "dropped_dups": int(dropped_dups),
        }

    def run_tick(
        self,
        *,
        accepted: Sequence[Question] | None = None,
        rejected: Sequence[Question] | None = None,
        diversity_report: DiversityReport | None = None,
        apply: bool = True,
        last_facet: str | None = None,
        data_dir: Path | None = None,
        fake_llm: bool = False,
        max_leaves: int | None = None,
    ) -> dict:
        """One manager tick: analyze, pick a mission (expand/gap/analyze), optionally apply."""
        self._in_tick = True
        self._llm_calls = 0
        self._last_gray_pairs = []
        self._last_dropped = 0
        try:
            health = self.analyze_health(accepted, rejected)
            facet = self.choose_facet(health, last_facet)
            underrep = self._underrep_keys(diversity_report)
            empty_or_sparse = bool(health.empty_l2_nodes or health.sparse_l2_nodes)
            if health.leaves_with_high_reject_rate or (underrep and empty_or_sparse):
                mission = "gap_search"
            elif self._pick_expand_l2(health):
                mission = "expand"
            else:
                mission = "analyze"
            payload = self._health_summary(health)
            payload["facet"] = facet
            payload["mission"] = mission
            if data_dir is not None:
                self.persist_health(payload, Path(data_dir) / "taxonomy_health.json")

            if fake_llm or not self.enabled or health.health_score >= self.health_expand_threshold:
                return self._tick_result(
                    health=health,
                    total_added=0,
                    facet=facet,
                    mission=mission,
                    expansions=[],
                    gray_pairs=[],
                    dropped_dups=0,
                )

            total_added = 0
            expansions: list[dict[str, Any]] = []
            gray_pairs: list[dict[str, Any]] = []
            dropped_dups = 0
            llm_notes: dict[str, Any] | None = None
            leaf_cap = (
                int(max_leaves)
                if max_leaves is not None
                else self.max_leaves_per_tick
            )
            leaf_cap = max(0, min(leaf_cap, 20))

            if mission == "expand":
                l2_key = self._pick_expand_l2(health)
                if l2_key:
                    pair = self._lookup_l2(l2_key)
                    if pair:
                        domain, child = pair
                        current = _active_leaf_count(child)
                        need = min(
                            self.target_leaves_per_l2 - current,
                            leaf_cap,
                            20,
                        )
                        if need > 0:
                            accepted_leaves = self._expand_l2(
                                domain=domain,
                                child=child,
                                facet=self._facet_dict(facet),
                                max_new=need,
                            )
                            gray_pairs = list(self._last_gray_pairs)
                            dropped_dups = int(self._last_dropped)
                            n = self._commit_leaves(accepted_leaves, apply=apply)
                            total_added += n
                            expansions.append(
                                {"l2_key": l2_key, "added": n, "facet": facet}
                            )
            elif mission == "gap_search":
                l2_key = self._pick_gap_l2(health, diversity_report)
                if l2_key:
                    pair = self._lookup_l2(l2_key)
                    need = min(leaf_cap, 20, self.target_leaves_per_l2 - _active_leaf_count(pair[1])) if pair else 0
                    accepted_leaves = self.gap_search(
                        target_l2_key=l2_key,
                        health=health,
                        rejected=rejected,
                        max_new=need,
                        facet=facet,
                    )
                    gray_pairs = list(self._last_gray_pairs)
                    dropped_dups = int(self._last_dropped)
                    n = self._commit_leaves(accepted_leaves, apply=apply)
                    total_added += n
                    expansions.append(
                        {"l2_key": l2_key, "added": n, "facet": facet}
                    )
            elif self._has_issues(health):
                notes = self.llm_health_review(
                    health, accepted=accepted, rejected=rejected
                )
                if notes:
                    llm_notes = notes
                    extra = [
                        str(k)
                        for k in (notes.get("expansion_priority") or [])
                        if self._lookup_l2(str(k))
                    ]
                    if extra and self._llm_calls < self.max_llm_calls_per_tick:
                        l2_key = extra[0]
                        pair = self._lookup_l2(l2_key)
                        if pair:
                            domain, child = pair
                            need = min(
                                self.target_leaves_per_l2 - _active_leaf_count(child),
                                leaf_cap,
                                20,
                            )
                            if need > 0:
                                accepted_leaves = self._expand_l2(
                                    domain=domain,
                                    child=child,
                                    facet=self._facet_dict(facet),
                                    max_new=need,
                                )
                                gray_pairs = list(self._last_gray_pairs)
                                dropped_dups = int(self._last_dropped)
                                n = self._commit_leaves(accepted_leaves, apply=apply)
                                total_added += n
                                expansions.append(
                                    {
                                        "l2_key": l2_key,
                                        "added": n,
                                        "facet": facet,
                                        "from": "health_notes",
                                    }
                                )

            if data_dir is not None:
                payload = self._health_summary(health)
                payload["facet"] = facet
                payload["mission"] = mission
                payload["total_added"] = total_added
                if llm_notes:
                    payload["llm_notes"] = llm_notes
                self.persist_health(payload, Path(data_dir) / "taxonomy_health.json")

            return self._tick_result(
                health=health,
                total_added=total_added,
                facet=facet,
                mission=mission,
                expansions=expansions,
                gray_pairs=gray_pairs,
                dropped_dups=dropped_dups,
            )
        finally:
            self._in_tick = False
