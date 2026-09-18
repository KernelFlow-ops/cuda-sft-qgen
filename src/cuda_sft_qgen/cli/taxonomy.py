"""Taxonomy health, expand, manage, and rebuild commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cuda_sft_qgen.cli.common import _json_print, load_settings
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime

def cmd_expand_taxonomy(args: argparse.Namespace) -> int:
    """One-shot L2 leaf expansion (legacy CLI; prefer ``manage-taxonomy``)."""
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker

    settings = load_settings(fake_llm=args.fake)
    if not settings.fake_llm and not settings.nvidia_api_keys:
        print(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3.",
            file=sys.stderr,
        )
        return 2
    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        tax_cfg = settings.default_yaml.get("taxonomy") or {}
        per_l2 = args.per_l2 or int(tax_cfg.get("target_leaves_per_l2", 30))
        worker = KnowledgeWorker(rt.knowledge, rt.llm, embedder=rt.embedder, settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
        result = worker.expand_once(apply=bool(args.apply), mode="strategic", target_per_l2=per_l2,
                                    max_leaves=int(tax_cfg.get("expand_max_per_call", 40)), domain_ids=args.domain)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        rt.close()
        reset_runtime(token)
    return 0


def _reindex_taxonomy_embeds(rt, *, save: bool) -> dict:
    from cuda_sft_qgen.taxonomy_embed import TaxonomyEmbedIndex

    mgr = rt.settings.taxonomy_manager_cfg
    index = TaxonomyEmbedIndex(
        rt.settings.data_dir / "taxonomy_embed.npz",
        rt.embedder,
        cosine_dup=float(mgr.get("leaf_cosine_dup", 0.92)),
        cosine_gray=float(mgr.get("leaf_cosine_gray", 0.85)),
    )
    return {"rebuilt": index.rebuild(rt.taxonomy.leaves, save=save)}


def cmd_manage_taxonomy(args: argparse.Namespace) -> int:
    """Analyze, expand, or tick the taxonomy manager against the live store."""
    from cuda_sft_qgen.taxonomy_manager import manager_from_runtime
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker

    actions = [
        args.analyze,
        args.expand,
        args.gap_search,
        args.dedup,
        args.reindex,
        args.refine,
        args.tick,
    ]
    if not any(actions):
        args.analyze = True

    settings = load_settings(fake_llm=True if args.fake else None)
    if settings.fake_llm:
        settings.embedding_provider = "fake"

    llm_action = args.expand or args.gap_search or args.refine or args.tick
    reindex_action = args.dedup or args.reindex
    if not settings.fake_llm and not settings.nvidia_api_keys:
        if llm_action or reindex_action:
            print(
                "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3, or pass --fake.",
                file=sys.stderr,
            )
            return 2
        settings.fake_llm = True
        settings.embedding_provider = "fake"

    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        manager = manager_from_runtime(rt)
        worker = KnowledgeWorker(rt.knowledge, rt.llm, embedder=rt.embedder, settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
        accepted = rt.store.load_accepted()
        rejected = rt.store.load_rejected()
        max_leaves = 40 if args.max_leaves is None else args.max_leaves
        max_calls = 3 if args.max_calls is None else args.max_calls
        facets = [args.facet] if args.facet else None

        if args.dedup or args.reindex:
            _json_print(
                _reindex_taxonomy_embeds(rt, save=not rt.settings.fake_llm)
            )

        if args.tick:
            result = worker.expand_once(apply=bool(args.apply), mode="tick", facet=args.facet,
                                        max_calls=args.max_calls, max_leaves=args.max_leaves)
            _json_print(result)

        if args.analyze:
            health = manager.analyze_health(
                accepted_questions=accepted,
                rejected_questions=rejected,
            )
            _json_print(manager._health_summary(health))

        if args.expand:
            result = worker.expand_once(apply=bool(args.apply), mode="strategic", max_leaves=max_leaves,
                                        max_calls=max_calls, facet=args.facet, domain_ids=args.domain)
            _json_print(result)

        if args.gap_search:
            health = manager.analyze_health(
                accepted_questions=accepted,
                rejected_questions=rejected,
            )
            target = None
            if health.leaves_with_high_reject_rate:
                l3_id = health.leaves_with_high_reject_rate[0][0]
                leaf = rt.taxonomy.leaf_by_key(l3_id)
                if leaf:
                    target = f"{leaf.domain_id}.{leaf.l2_id}"
            if not target and health.expansion_priority:
                target = health.expansion_priority[0]
            elif not target and health.empty_l2_nodes:
                target = health.empty_l2_nodes[0]
            elif not target and health.sparse_l2_nodes:
                target = health.sparse_l2_nodes[0][0]
            if not target:
                _json_print({"total_added": 0, "reason": "no_gap_target"})
            else:
                facet = args.facet or manager.choose_facet(health)
                _json_print(worker.expand_once(apply=bool(args.apply), mode="gap", l2_key=target,
                                               max_leaves=max_leaves, max_calls=max_calls, facet=facet))

        if args.refine:
            _json_print(worker.refine_once(apply=bool(args.apply)))
    finally:
        closer = getattr(rt, "close", None)
        if callable(closer):
            closer()
        reset_runtime(token)
    return 0


def cmd_rebuild_taxonomy(args: argparse.Namespace) -> int:
    """Force Architect to expand sparse L2 leaves before planning."""
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker

    settings = load_settings(fake_llm=args.fake)
    if not settings.fake_llm and not settings.nvidia_api_keys:
        print(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3.",
            file=sys.stderr,
        )
        return 2
    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        worker = KnowledgeWorker(rt.knowledge, rt.llm, embedder=rt.embedder, settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
        out = worker.expand_once(apply=True)
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    finally:
        rt.close()
        reset_runtime(token)
    return 0


def cmd_taxonomy_health(args: argparse.Namespace) -> int:
    """Print L2 coverage/health scores from the current archive."""
    from cuda_sft_qgen.taxonomy_health import HealthAnalyzer

    settings = load_settings(fake_llm=True if args.fake else None)
    if not args.fake and args.with_diversity:
        if not settings.nvidia_api_keys:
            print(
                "No NVIDIA API keys for embeddings. Set keys or pass --fake.",
                file=sys.stderr,
            )
            return 2
        settings.embedding_provider = "nvidia"
    else:
        settings.embedding_provider = "fake"

    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        embedder = rt.embedder if args.with_diversity else None
        analyzer = HealthAnalyzer(
            rt.taxonomy,
            target_per_l2=args.target_per_l2 or 30,
            embedder=embedder,
        )
        report = analyzer.analyze(rt.store)

        output = {
            "overall_health": report.overall_health,
            "overall_coverage": report.overall_coverage,
            "overall_accept_rate": report.overall_accept_rate,
            "overall_diversity": report.overall_diversity,
            "underperforming": report.underperforming[:10],
            "expansion_targets": report.expansion_targets[:10],
        }

        if args.verbose or args.detail:
            output["l2_metrics"] = {
                k: {
                    "leaf_count": v.leaf_count,
                    "question_count": v.question_count,
                    "coverage": v.coverage,
                    "accept_rate": v.accept_rate,
                    "diversity": v.avg_diversity,
                    "health_score": v.health_score,
                }
                for k, v in report.l2_metrics.items()
            }

        _json_print(output)
    finally:
        reset_runtime(token)
    return 0


def cmd_taxonomy_expand(args: argparse.Namespace) -> int:
    """Health-driven leaf expansion with optional ``--apply``."""
    from cuda_sft_qgen.taxonomy_health.expander import LeafExpander
    from cuda_sft_qgen.taxonomy_health import HealthAnalyzer
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker

    settings = load_settings(fake_llm=args.fake)
    if not settings.fake_llm and not settings.nvidia_api_keys:
        print(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3.",
            file=sys.stderr,
        )
        return 2

    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        analyzer = HealthAnalyzer(rt.taxonomy, target_per_l2=args.target_per_l2 or 30)
        report = analyzer.analyze(rt.store)

        expander = LeafExpander(rt.taxonomy, rt.llm, embedder=rt.embedder)
        worker = KnowledgeWorker(rt.knowledge, rt.llm, embedder=rt.embedder, settings=settings, taxonomy=rt.taxonomy, archive=rt.store)

        if args.dry_run:
            preview = expander.preview_expansion(report, limit=args.limit or 10)
            _json_print({"dry_run": True, "suggestions": preview})
        else:
            targets = expander.identify_expansion_targets(report)
            if args.l2_key:
                targets = [(args.l2_key, {})]

            expansions = []
            total_added = 0
            for l2_key, metadata in targets[:args.limit or 5]:
                metric = report.l2_metrics.get(l2_key)
                if not metric:
                    continue

                need = min(args.max_new or 10, 20)
                result = worker.expand_once(apply=bool(args.apply), mode="l2", l2_key=l2_key,
                                            max_leaves=need, facet=args.facet, target_per_l2=args.target_per_l2)
                total_added += result["total_added"]
                expansions.append({"l2_key": l2_key, **result})

            _json_print({
                "expansions": expansions,
                "total_added": total_added,
            })
    finally:
        rt.close()
        reset_runtime(token)
    return 0


