"""Command-line entry point for generate, review, knowledge, and taxonomy tools."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime
from cuda_sft_qgen.store.archive import Archive


def _setup_logging(verbose: bool) -> None:
    """Configure root logging; ``--verbose`` enables DEBUG."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_generate(args: argparse.Namespace) -> int:
    """Run the generate+review pipeline until target or iteration limit."""
    data_dir = Path(args.data_dir) if getattr(args, "data_dir", None) else None
    settings = load_settings(
        fake_llm=args.fake,
        target=args.target,
        review_batch_size=args.batch,
        data_dir=data_dir,
    )
    if args.fake:
        settings.fake_llm = True
        settings.embedding_provider = "fake"
    if not settings.fake_llm and not settings.nvidia_api_keys:
        print(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3, or pass --fake.",
            file=sys.stderr,
        )
        return 2
    result = run_pipeline(settings, rebuild_taxonomy=args.rebuild_taxonomy)
    print(
        json.dumps(
            {
                "accepted_count": result.get("accepted_count"),
                "rejected_count": result.get("rejected_count"),
                "iteration": result.get("iteration"),
                "status": result.get("status"),
                "done": result.get("done"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Continue a previous run from the archive in ``--data-dir``."""
    args.rebuild_taxonomy = False
    if getattr(args, "retry_reviews", False):
        settings = load_settings(fake_llm=args.fake,
                                 data_dir=Path(args.data_dir) if args.data_dir else None)
        Archive(settings.data_dir).requeue_reviews(include_exhausted=True)
    return cmd_generate(args)


def cmd_knowledge(args: argparse.Namespace) -> int:
    """Inspect, ingest, audit, or export the knowledge graph."""
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker

    settings = load_settings(fake_llm=True if args.fake else None,
                             data_dir=Path(args.data_dir) if args.data_dir else None)
    rt = build_runtime(settings)
    try:
        if args.action == "stats":
            _json_print(rt.knowledge.stats())
        elif args.action == "query":
            if not args.claim:
                raise ValueError("query requires --claim")
            hops = max(1, int(getattr(args, "hops", 1) or 1))
            if hops <= 1:
                _json_print(rt.knowledge.context_for_claim(args.claim))
            else:
                _json_print(rt.knowledge.subgraph(args.claim, hops=hops, max_nodes=50))
        elif args.action == "graph":
            if not args.claim:
                raise ValueError("graph requires --claim")
            hops = max(1, int(getattr(args, "hops", 2) or 2))
            graph = rt.knowledge.subgraph(args.claim, hops=hops, max_nodes=50)
            print(rt.knowledge.to_mermaid(graph), end="")
        elif args.action == "viz":
            if not args.output:
                raise ValueError("viz requires --output")
            from cuda_sft_qgen.viz import render_graph_html
            if args.claim:
                hops = max(1, int(getattr(args, "hops", 2) or 2))
                graph = rt.knowledge.subgraph(args.claim, hops=hops, max_nodes=200)
            else:
                graph = rt.knowledge.export_graph()
            path = render_graph_html(graph, Path(args.output), title=f"knowledge {args.claim or 'all'}")
            _json_print({"path": str(path), "claims": len(graph.get("claims") or graph.get("nodes") or [])})
        elif args.action == "impact":
            _json_print({"events": rt.knowledge.impact_events(), "corrections": rt.store.corrections()})
        elif args.action == "export":
            if not args.output:
                raise ValueError("export requires --output")
            rt.knowledge.export_taxonomy(Path(args.output), rt.taxonomy)
            _json_print({"path": args.output})
        elif args.action == "audit":
            rt.knowledge.enqueue_daily_audits(args.limit)
            worker = KnowledgeWorker(rt.knowledge, rt.review_llm or rt.llm, embedder=rt.embedder,
                                     settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
            processed = 0
            while processed < args.limit and worker.run_once():
                processed += 1
            rt.refresh_knowledge()
            _json_print({"processed": processed, "stats": rt.knowledge.stats()})
        elif args.action == "ingest-pack":
            from cuda_sft_qgen.pack import ingest_pack, load_pack
            if not args.pack:
                raise ValueError("ingest-pack requires --pack")
            result = ingest_pack(rt.knowledge, load_pack(args.pack), apply=bool(args.apply))
            if args.apply:
                rt.refresh_knowledge(force=True)
            _json_print(result)
        elif args.action == "expand":
            worker = KnowledgeWorker(rt.knowledge, rt.llm, embedder=rt.embedder,
                                     settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
            if not args.pack:
                raise ValueError("expand requires --pack id")
            _json_print(worker.expand_pack(args.pack, apply=bool(args.apply), max_leaves=args.limit))
        elif args.action == "relate":
            worker = KnowledgeWorker(rt.knowledge, rt.review_llm or rt.llm, embedder=rt.embedder,
                                     settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
            queued = 0
            for claim in rt.knowledge.list_claims()[: args.limit]:
                if rt.knowledge.enqueue("relate", {"claim_id": claim["id"]}, f"relate:{claim['id']}"):
                    queued += 1
            processed = 0
            while processed < args.limit and worker.run_once():
                processed += 1
            _json_print({"queued": queued, "processed": processed, "stats": rt.knowledge.stats()})
    finally:
        rt.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export independently reviewed accepted questions for SFT training."""
    settings = load_settings(fake_llm=True, data_dir=Path(args.data_dir) if args.data_dir else None)
    rt = build_runtime(settings)
    try:
        rt.refresh_knowledge(force=True)
        _json_print(rt.store.export_training(Path(args.output)))
    finally:
        rt.close()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Isolated generation benchmark with taxonomy/knowledge sidecars disabled."""
    from cuda_sft_qgen.benchmark import run_benchmark
    settings = load_settings(fake_llm=args.fake)
    report = run_benchmark(settings, Path(args.output), target=args.target,
                           max_iterations=args.max_iterations, timeout=args.timeout)
    _json_print(report)
    return 0 if report.get("done") else 1


def cmd_gate(args: argparse.Namespace) -> int:
    """Throughput/quality gate: compare a short real run against a baseline."""
    from cuda_sft_qgen.gate import run_gate

    data_dir = Path(args.data_dir)
    if args.fake:
        print("gate refuses --fake; use a real model so knowledge-on throughput is measured.", file=sys.stderr)
        return 2
    settings = load_settings(
        fake_llm=False,
        target=args.target,
        review_batch_size=args.target,
        data_dir=data_dir,
        profile=args.profile,
    )
    if not settings.nvidia_api_keys:
        print(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3.",
            file=sys.stderr,
        )
        return 2
    report = run_gate(
        settings,
        data_dir,
        phase=args.phase,
        target=args.target,
        timeout=args.timeout,
        max_iterations=args.max_iterations,
        baseline_path=Path(args.baseline) if args.baseline else None,
        ingest_packs=[Path(p) for p in (args.pack or [])],
    )
    _json_print(report)
    return 1 if report.get("verdict") == "red" else 0


def cmd_review_benchmark(args: argparse.Namespace) -> int:
    """Evaluate the chair contract on frozen labeled quality cases."""
    from cuda_sft_qgen.review_benchmark import run_review_benchmark
    report = run_review_benchmark(load_settings(), Path(args.output))
    _json_print(report.get("summary", report))
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    """Print archive histograms from ``data/stats.json``."""
    settings = load_settings(fake_llm=True, data_dir=Path(_args.data_dir) if getattr(_args, "data_dir", None) else None)
    archive = Archive(settings.data_dir)
    stats = archive.compute_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_probe_concurrency(args: argparse.Namespace) -> int:
    """Hit the live API pool to confirm LLM/embed overlap across keys."""
    from cuda_sft_qgen.probe import probe_concurrency, probe_pipeline

    conc = probe_concurrency()
    print(json.dumps({"concurrency": conc}, ensure_ascii=False, indent=2))
    if not args.skip_pipeline:
        tmp = Path(args.data_dir) if args.data_dir else None
        if tmp is None:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix="qgen-probe-") as td:
                pipe = probe_pipeline(Path(td), target=args.target, batch=args.batch)
        else:
            Path(tmp).mkdir(parents=True, exist_ok=True)
            pipe = probe_pipeline(Path(tmp), target=args.target, batch=args.batch)
        print(json.dumps({"mini_pipeline": pipe}, ensure_ascii=False, indent=2))
        if not conc.get("llm_overlap") or not conc.get("embed_overlap"):
            return 1
        if not pipe.get("generate_overlap"):
            return 1
    elif not conc.get("llm_overlap") or not conc.get("embed_overlap"):
        return 1
    return 0


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


def _json_print(payload) -> None:
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


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


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Load and print the resolved settings for a profile."""
    """Validate configuration completeness and correctness."""
    try:
        from cuda_sft_qgen.config_v2 import load_config_v2

        profile = args.profile or os.getenv("CUDA_SFT_PROFILE", "default")
        config = load_config_v2(profile=profile)

        issues = []
        warnings = []

        # Check API keys
        if not config.fake_llm and not config.nvidia_api_keys:
            issues.append("No NVIDIA API keys configured")

        # Check file paths
        paths_to_check = [
            ("taxonomy", config.resolve_path(config.taxonomy_path)),
            ("quotas", config.resolve_path(config.quotas_path)),
            ("personas", config.resolve_path(config.personas_path)),
        ]

        for name, path in paths_to_check:
            if not path.exists():
                warnings.append(f"{name} file not found: {path}")

        # Check threshold ranges
        if not (0.0 <= config.review.semantic_dup_threshold <= 1.0):
            issues.append(f"semantic_dup_threshold out of range: {config.review.semantic_dup_threshold}")

        if config.review.semantic_gray_threshold >= config.review.semantic_dup_threshold:
            warnings.append("semantic_gray_threshold should be < semantic_dup_threshold")

        # Check concurrency settings
        if config.planner.writer_concurrency < 1:
            issues.append(f"writer_concurrency must be >= 1: {config.planner.writer_concurrency}")

        # Print results
        result = {
            "profile": profile,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

        if args.verbose:
            result["config"] = {
                "llm": config.llm.model,
                "embedding": config.embedding.model,
                "writer_concurrency": config.planner.writer_concurrency,
                "review_batch_size": config.review.batch_size,
                "data_dir": str(config.resolve_path(config.data_dir)),
            }

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if len(issues) == 0 else 1

    except Exception as e:
        print(json.dumps({"valid": False, "error": str(e)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


def cmd_estimate_cost(args: argparse.Namespace) -> int:
    """Rough token/USD estimate for a target question count."""
    """Estimate API costs for target question count."""
    try:
        from cuda_sft_qgen.config_v2 import load_config_v2

        profile = args.profile or os.getenv("CUDA_SFT_PROFILE", "default")
        target = args.target
        config = load_config_v2(profile=profile, target=target)

        # Token cost estimates (USD per 1M tokens)
        # Based on NVIDIA API pricing (approximate)
        llm_cost_per_1m_input = 2.00
        llm_cost_per_1m_output = 6.00
        embed_cost_per_1m = 0.20

        # Estimate tokens per question
        avg_prompt_tokens = 8000  # System + taxonomy + fewshot + task
        avg_completion_tokens = 400  # Generated question JSON
        cache_factor = 0.4 if config.llm.prompt_cache_enabled else 1.0  # 60% savings

        # Writer estimates
        writer_questions = int(target * (1 - config.planner.evolver_fraction))
        writer_prompt_tokens = writer_questions * avg_prompt_tokens * cache_factor
        writer_completion_tokens = writer_questions * avg_completion_tokens

        # Evolver estimates (shorter context)
        evolver_questions = target - writer_questions
        evolver_prompt_tokens = evolver_questions * (avg_prompt_tokens * 0.6) * cache_factor
        evolver_completion_tokens = evolver_questions * avg_completion_tokens

        # Every candidate receives a complete independent review; rejections add cost.
        chair_reviews = target
        chair_prompt_tokens = chair_reviews * 4000
        chair_completion_tokens = chair_reviews * 1200

        # Taxonomy manager estimates
        taxonomy_calls = int(target / max(1, config.review.batch_size * config.taxonomy.every_k_reviews))
        taxonomy_llm_calls = taxonomy_calls * config.taxonomy.max_llm_calls_per_tick
        taxonomy_prompt_tokens = taxonomy_llm_calls * 3000
        taxonomy_completion_tokens = taxonomy_llm_calls * 800

        # Total LLM tokens
        total_prompt_tokens = (
            writer_prompt_tokens + evolver_prompt_tokens +
            chair_prompt_tokens + taxonomy_prompt_tokens
        )
        total_completion_tokens = (
            writer_completion_tokens + evolver_completion_tokens +
            chair_completion_tokens + taxonomy_completion_tokens
        )

        # Embedding estimates
        embed_cache_factor = 0.2 if config.embedding.cache_enabled else 1.0  # 80% hit rate
        embedding_calls = int(target * config.review.batch_size * embed_cache_factor / config.embedding.chunk_size)
        embedding_tokens = embedding_calls * config.embedding.chunk_size * 100  # ~100 tokens per question

        # Cost calculations
        llm_input_cost = (total_prompt_tokens / 1_000_000) * llm_cost_per_1m_input
        llm_output_cost = (total_completion_tokens / 1_000_000) * llm_cost_per_1m_output
        embedding_cost = (embedding_tokens / 1_000_000) * embed_cost_per_1m

        result = {
            "profile": profile,
            "target_questions": target,
            "estimates": {
                "llm_tokens": {
                    "prompt": int(total_prompt_tokens),
                    "completion": int(total_completion_tokens),
                    "total": int(total_prompt_tokens + total_completion_tokens),
                },
                "embedding_tokens": int(embedding_tokens),
                "embedding_calls": embedding_calls,
            },
            "breakdown": {
                "writer": {
                    "questions": writer_questions,
                    "prompt_tokens": int(writer_prompt_tokens),
                    "completion_tokens": int(writer_completion_tokens),
                },
                "evolver": {
                    "questions": evolver_questions,
                    "prompt_tokens": int(evolver_prompt_tokens),
                    "completion_tokens": int(evolver_completion_tokens),
                },
                "chair": {
                    "reviews": chair_reviews,
                    "prompt_tokens": int(chair_prompt_tokens),
                    "completion_tokens": int(chair_completion_tokens),
                },
                "taxonomy": {
                    "calls": taxonomy_llm_calls,
                    "prompt_tokens": int(taxonomy_prompt_tokens),
                    "completion_tokens": int(taxonomy_completion_tokens),
                },
            },
            "cost_usd": {
                "llm_input": round(llm_input_cost, 2),
                "llm_output": round(llm_output_cost, 2),
                "llm_total": round(llm_input_cost + llm_output_cost, 2),
                "embeddings": round(embedding_cost, 2),
                "total": round(llm_input_cost + llm_output_cost + embedding_cost, 2),
            },
            "assumptions": {
                "llm_cost_per_1m_input_tokens": llm_cost_per_1m_input,
                "llm_cost_per_1m_output_tokens": llm_cost_per_1m_output,
                "embed_cost_per_1m_tokens": embed_cost_per_1m,
                "prompt_cache_enabled": config.llm.prompt_cache_enabled,
                "embedding_cache_enabled": config.embedding.cache_enabled,
            },
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


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


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``cuda-sft-qgen`` argument parser and subcommands."""
    p = argparse.ArgumentParser(
        prog="cuda-sft-qgen",
        description="Generate a CUDA-knowledge SFT question set with LangGraph agents.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run the generation pipeline")
    g.add_argument("-v", "--verbose", action="store_true")
    g.add_argument("--target", type=int, default=None)
    g.add_argument("--batch", type=int, default=None, help="REVIEW_BATCH_SIZE")
    g.add_argument("--fake", action="store_true", help="use FakeLLM, no API key")
    g.add_argument("--rebuild-taxonomy", action="store_true")
    g.add_argument(
        "--data-dir",
        default=None,
        help="output directory (overrides CUDA_SFT_DATA_DIR / paths.data_dir)",
    )
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("resume", help="continue from data/ accepted+pending")
    r.add_argument("-v", "--verbose", action="store_true")
    r.add_argument("--target", type=int, default=None)
    r.add_argument("--batch", type=int, default=None)
    r.add_argument("--fake", action="store_true")
    r.add_argument("--data-dir", default=None, help="output directory to resume")
    r.add_argument("--retry-reviews", action="store_true", help="explicitly requeue delayed and exhausted reviews")
    r.set_defaults(func=cmd_resume, rebuild_taxonomy=False)

    s = sub.add_parser("stats", help="print coverage histograms")
    s.add_argument("--data-dir", default=None)
    s.set_defaults(func=cmd_stats)

    kb = sub.add_parser("knowledge", help="query, audit, and export the versioned knowledge graph")
    kb.add_argument("action", choices=["stats", "query", "audit", "impact", "export", "graph", "viz", "ingest-pack", "expand", "relate"])
    kb.add_argument("--data-dir", default=None)
    kb.add_argument("--claim", default=None)
    kb.add_argument("--output", default=None)
    kb.add_argument("--limit", type=int, default=100)
    kb.add_argument("--hops", type=int, default=None, help="graph/query BFS depth (query default 1, graph default 2)")
    kb.add_argument("--pack", default=None, help="pack yaml path (ingest-pack) or pack id (expand)")
    kb.add_argument("--apply", action="store_true", help="write pack/expand results to the knowledge store")
    kb.add_argument("--fake", action="store_true")
    kb.set_defaults(func=cmd_knowledge)

    export = sub.add_parser("export", help="export only independently reviewed, currently valid questions")
    export.add_argument("--data-dir", default=None)
    export.add_argument("--output", required=True)
    export.set_defaults(func=cmd_export)

    bench = sub.add_parser("benchmark", help="run an isolated fixed-snapshot generation benchmark")
    bench.add_argument("--output", required=True)
    bench.add_argument("--target", type=int, default=100)
    bench.add_argument("--max-iterations", type=int, default=40)
    bench.add_argument("--timeout", type=float, default=3600)
    bench.add_argument("--fake", action="store_true")
    bench.set_defaults(func=cmd_benchmark)

    gate = sub.add_parser(
        "gate",
        help="knowledge-on 10-question generate+review throughput gate (real model only)",
    )
    gate.add_argument("--data-dir", required=True, help="empty output directory for this gate run")
    gate.add_argument("--phase", required=True, help="baseline | 0 | 1 | ...")
    gate.add_argument("--target", type=int, default=10)
    gate.add_argument("--baseline", default=None, help="path to previous gate_report.json")
    gate.add_argument("--timeout", type=float, default=2700)
    gate.add_argument("--max-iterations", type=int, default=40)
    gate.add_argument("--profile", default="dev")
    gate.add_argument("--fake", action="store_true", help="rejected; real gates cannot use FakeLLM")
    gate.add_argument("--pack", action="append", default=None, help="ingest this pack yaml before generate; repeatable")
    gate.set_defaults(func=cmd_gate)

    review_bench = sub.add_parser("review-benchmark", help="evaluate real reviewer on fixed defects and controls")
    review_bench.add_argument("--output", required=True)
    review_bench.set_defaults(func=cmd_review_benchmark)

    t = sub.add_parser("rebuild-taxonomy", help="run Taxonomy Architect once")
    t.add_argument("--fake", action="store_true")
    t.set_defaults(func=cmd_rebuild_taxonomy)

    ex = sub.add_parser("expand-taxonomy", help="densify each L2 toward a leaf target")
    ex.add_argument("--fake", action="store_true")
    ex.add_argument("--per-l2", type=int, default=None)
    ex.add_argument("--apply", action="store_true")
    ex.add_argument("--domain", action="append", default=None, help="limit to L1 id, repeatable")
    ex.set_defaults(func=cmd_expand_taxonomy)

    mt = sub.add_parser(
        "manage-taxonomy",
        help="taxonomy health sidecar (analyze/expand/gap-search/reindex)",
    )
    mt.add_argument("--fake", action="store_true")
    mt.add_argument("--analyze", action="store_true", help="health report (default if no action)")
    mt.add_argument(
        "--tick",
        action="store_true",
        help="one full manager tick (analyze + expand or gap-search), same as the pipeline sidecar",
    )
    mt.add_argument("--expand", action="store_true")
    mt.add_argument("--gap-search", action="store_true")
    mt.add_argument(
        "--dedup",
        action="store_true",
        help="rebuild leaf embedding index (does not retire leaves)",
    )
    mt.add_argument(
        "--reindex",
        action="store_true",
        help="same as --dedup: rebuild data/taxonomy_embed.npz",
    )
    mt.add_argument("--refine", action="store_true")
    mt.add_argument("--apply", action="store_true")
    mt.add_argument("--facet", default=None)
    mt.add_argument("--max-leaves", type=int, default=None)
    mt.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="max LLM expand calls for --expand (default 3)",
    )
    mt.add_argument("--domain", action="append", default=None, help="limit to L1 id, repeatable")
    mt.set_defaults(func=cmd_manage_taxonomy)

    pbe = sub.add_parser(
        "probe-concurrency",
        help="live test: parallel LLM + embedding + a tiny generate/review batch",
    )
    pbe.add_argument("--skip-pipeline", action="store_true")
    pbe.add_argument("--target", type=int, default=6)
    pbe.add_argument("--batch", type=int, default=6)
    pbe.add_argument("--data-dir", default=None)
    pbe.set_defaults(func=cmd_probe_concurrency)

    vc = sub.add_parser("validate-config", help="validate configuration completeness")
    vc.add_argument("-v", "--verbose", action="store_true", help="show full config details")
    vc.add_argument("--profile", default=None, help="profile to validate (default: CUDA_SFT_PROFILE or 'default')")
    vc.set_defaults(func=cmd_validate_config)

    ec = sub.add_parser("estimate-cost", help="estimate API costs for target question count")
    ec.add_argument("--target", type=int, required=True, help="target question count")
    ec.add_argument("--profile", default=None, help="profile to use (default: CUDA_SFT_PROFILE or 'default')")
    ec.set_defaults(func=cmd_estimate_cost)

    tax_sub = sub.add_parser("taxonomy", help="taxonomy health and expansion tools")
    tax_cmds = tax_sub.add_subparsers(dest="taxonomy_cmd", required=True)

    th = tax_cmds.add_parser("health", help="display taxonomy health report")
    th.add_argument("--fake", action="store_true")
    th.add_argument("--target-per-l2", type=int, default=None)
    th.add_argument("--with-diversity", action="store_true", help="compute diversity metrics (requires embeddings)")
    th.add_argument("--detail", action="store_true", help="show per-L2 metrics")
    th.set_defaults(func=cmd_taxonomy_health)

    te = tax_cmds.add_parser("expand", help="expand taxonomy leaves")
    te.add_argument("--fake", action="store_true")
    te.add_argument("--target-per-l2", type=int, default=None)
    te.add_argument("--dry-run", action="store_true", help="preview expansion suggestions")
    te.add_argument("--apply", action="store_true", help="apply changes to taxonomy")
    te.add_argument("--l2-key", default=None, help="target specific L2 node (e.g., mem.shared)")
    te.add_argument("--facet", default=None, help="facet id for expansion")
    te.add_argument("--max-new", type=int, default=None, help="max new leaves per L2")
    te.add_argument("--limit", type=int, default=None, help="max L2 nodes to expand")
    te.set_defaults(func=cmd_taxonomy_expand)

    return p


def main(argv: list[str] | None = None) -> None:
    """Parse argv, run the selected subcommand, and exit with its status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
