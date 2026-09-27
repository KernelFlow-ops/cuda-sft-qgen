"""Knowledge-graph CLI actions."""
from __future__ import annotations

import argparse
from pathlib import Path

from cuda_sft_qgen.cli.common import _json_print, _require_keys, load_settings
from cuda_sft_qgen.runtime import build_runtime

def cmd_knowledge(args: argparse.Namespace) -> int:
    """Inspect, ingest, audit, or export the knowledge graph."""
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker

    settings = load_settings(fake_llm=True if args.fake else None,
                             data_dir=Path(args.data_dir) if args.data_dir else None)
    if args.action == "compute-verify":
        if settings.fake_llm:
            print("compute-verify requires a real LLM; omit --fake and configure an API key")
            return 2
        missing_keys = _require_keys(settings)
        if missing_keys is not None:
            return missing_keys
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
            worker = KnowledgeWorker(rt.knowledge, rt.knowledge_llm or rt.llm, embedder=rt.embedder,
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
            worker = KnowledgeWorker(rt.knowledge, rt.knowledge_llm or rt.llm, embedder=rt.embedder,
                                     settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
            if not args.pack:
                raise ValueError("expand requires --pack id")
            _json_print(worker.expand_pack(args.pack, apply=bool(args.apply), max_leaves=args.limit))
        elif args.action == "relate":
            worker = KnowledgeWorker(rt.knowledge, rt.knowledge_llm or rt.llm, embedder=rt.embedder,
                                     settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
            queued = 0
            for claim in rt.knowledge.list_claims()[: args.limit]:
                if rt.knowledge.enqueue("relate", {"claim_id": claim["id"]}, f"relate:{claim['id']}"):
                    queued += 1
            processed = 0
            while processed < args.limit and worker.run_once():
                processed += 1
            _json_print({"queued": queued, "processed": processed, "stats": rt.knowledge.stats()})
        elif args.action == "inventory":
            # Inventory refresh is deliberately explicit: it performs network
            # retrieval and then runs the taxonomy critic only on cached source
            # concepts.  ``--apply`` publishes provisional claims and queues
            # their official evidence audits; without it this is a preview.
            worker = KnowledgeWorker(rt.knowledge, rt.knowledge_llm or rt.llm,
                                     embedder=rt.embedder, settings=settings,
                                     taxonomy=rt.taxonomy, archive=rt.store)
            _json_print(worker.external_inventory_once(force=True, apply=bool(args.apply)))
        elif args.action == "verify":
            if not args.claim or not args.reviewer or not args.note:
                raise ValueError("verify requires --claim, --reviewer, and --note")
            claim = rt.knowledge.get_claim(args.claim)
            if claim is None:
                raise ValueError(f"unknown claim: {args.claim}")
            applied = rt.knowledge.apply_expert_verification(
                args.claim, claim["revision"], reviewer=args.reviewer, note=args.note,
            )
            _json_print({"claim_id": args.claim, "revision": claim["revision"], "applied": applied})
        elif args.action == "compute-verify":
            if not args.claim:
                raise ValueError("compute-verify requires --claim")
            claim = rt.knowledge.get_claim(args.claim)
            if claim is None:
                raise ValueError(f"unknown claim: {args.claim}")
            if claim.get("kind") not in {"formula", "numeric"}:
                raise ValueError("compute-verify only supports formula/numeric claims")
            job_id = rt.knowledge.enqueue(
                "compute_verify", {"claim_id": args.claim, "revision": claim["revision"]},
                f"compute:{args.claim}:{claim['revision']}",
            )
            worker = KnowledgeWorker(rt.knowledge, rt.knowledge_llm or rt.llm, embedder=rt.embedder,
                                     settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
            processed = worker.run_once()
            _json_print({"claim_id": args.claim, "revision": claim["revision"],
                         "job_id": job_id, "processed": processed,
                         "claim": rt.knowledge.get_claim(args.claim)})
        elif args.action == "backlog":
            _json_print({"enqueued": rt.knowledge.schedule_legacy_backlog()})
        elif args.action == "refine-legacy":
            if not args.l2:
                raise ValueError("refine-legacy requires --l2")
            worker = KnowledgeWorker(rt.knowledge, rt.knowledge_llm or rt.llm, embedder=rt.embedder,
                                     settings=settings, taxonomy=rt.taxonomy, archive=rt.store)
            _json_print(worker.refine_legacy_batch(args.l2, limit=args.limit, apply=bool(args.apply)))
    finally:
        rt.close()
    return 0
