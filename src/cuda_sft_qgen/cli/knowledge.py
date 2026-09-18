"""Knowledge-graph CLI actions."""
from __future__ import annotations

import argparse
from pathlib import Path

from cuda_sft_qgen.cli.common import _json_print, load_settings
from cuda_sft_qgen.runtime import build_runtime

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


