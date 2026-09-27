"""Argparse subcommands for cuda-sft-qgen."""
from __future__ import annotations

import argparse

from cuda_sft_qgen.cli.config_cmds import cmd_estimate_cost, cmd_gate, cmd_review_benchmark, cmd_validate_config
from cuda_sft_qgen.cli.generate import cmd_benchmark, cmd_export, cmd_generate, cmd_probe_concurrency, cmd_resume, cmd_stats
from cuda_sft_qgen.cli.generate import cmd_agent_pilot
from cuda_sft_qgen.cli.knowledge import cmd_knowledge
from cuda_sft_qgen.cli.taxonomy import (
    cmd_expand_taxonomy,
    cmd_manage_taxonomy,
    cmd_rebuild_taxonomy,
    cmd_taxonomy_expand,
    cmd_taxonomy_health,
)

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
    r.add_argument("--data-dir", default=None, help="output directory to resume")
    r.add_argument("--retry-reviews", action="store_true", help="explicitly requeue delayed and exhausted reviews")
    r.set_defaults(func=cmd_resume, rebuild_taxonomy=False)

    s = sub.add_parser("stats", help="print coverage histograms")
    s.add_argument("--data-dir", default=None)
    s.set_defaults(func=cmd_stats)

    kb = sub.add_parser("knowledge", help="query, audit, and export the versioned knowledge graph")
    kb.add_argument("action", choices=["stats", "query", "audit", "impact", "export", "graph", "viz", "ingest-pack", "expand", "relate", "inventory", "verify", "compute-verify", "backlog", "refine-legacy"])
    kb.add_argument("--data-dir", default=None)
    kb.add_argument("--claim", default=None)
    kb.add_argument("--output", default=None)
    kb.add_argument("--limit", type=int, default=100)
    kb.add_argument("--hops", type=int, default=None, help="graph/query BFS depth (query default 1, graph default 2)")
    kb.add_argument("--pack", default=None, help="pack yaml path (ingest-pack) or pack id (expand)")
    kb.add_argument("--reviewer", default=None, help="reviewer name for expert verification")
    kb.add_argument("--l2", default=None, help="L2 key for legacy refinement")
    kb.add_argument("--note", default=None, help="expert verification note")
    kb.add_argument("--apply", action="store_true", help="write pack/expand results to the knowledge store")
    kb.add_argument("--fake", action="store_true")
    kb.set_defaults(func=cmd_knowledge)

    export = sub.add_parser("export", help="export only independently reviewed, currently valid questions")
    export.add_argument("--data-dir", default=None)
    export.add_argument("--output", required=True)
    export.set_defaults(func=cmd_export)

    pilot = sub.add_parser("agent-pilot", help="bounded real API synthesis/review ablation in isolated archives")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--samples", type=int, default=2)
    pilot.add_argument("--max-calls", type=int, default=120)
    pilot.add_argument("--timeout", type=float, default=1800)
    pilot.add_argument("--groups", nargs="+", choices=["A", "B", "C", "D"], default=["A", "B", "C", "D"])
    pilot.set_defaults(func=cmd_agent_pilot)

    bench = sub.add_parser("benchmark", help="run an isolated fixed-snapshot generation benchmark")
    bench.add_argument("--output", required=True)
    bench.add_argument("--target", type=int, default=100)
    bench.add_argument("--max-iterations", type=int, default=40)
    bench.add_argument("--timeout", type=float, default=3600)
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
