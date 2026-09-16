from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime
from cuda_sft_qgen.store.archive import Archive


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_generate(args: argparse.Namespace) -> int:
    settings = load_settings(
        fake_llm=args.fake,
        target=args.target,
        review_batch_size=args.batch,
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
    args.rebuild_taxonomy = False
    return cmd_generate(args)


def cmd_stats(_args: argparse.Namespace) -> int:
    settings = load_settings(fake_llm=True)
    archive = Archive(settings.data_dir)
    stats = archive.compute_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_probe_concurrency(args: argparse.Namespace) -> int:
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
    from cuda_sft_qgen.prompts import load_prompt
    from cuda_sft_qgen.taxonomy_expand import expand_taxonomy

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
        result = expand_taxonomy(
            rt.taxonomy,
            rt.llm,
            per_l2=per_l2,
            max_per_call=int(tax_cfg.get("expand_max_per_call", 40)),
            domain_ids=args.domain,
            system=load_prompt("architect"),
            apply=bool(args.apply),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
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
    from cuda_sft_qgen.taxonomy_manager import manager_from_runtime

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
            result = manager.run_tick(
                accepted=accepted,
                rejected=rejected,
                apply=bool(args.apply),
                last_facet=args.facet,
                data_dir=rt.settings.data_dir,
                fake_llm=rt.settings.fake_llm,
                max_leaves=None if args.max_leaves is None else max_leaves,
            )
            _json_print(result)

        if args.analyze:
            health = manager.analyze_health(
                accepted_questions=accepted,
                rejected_questions=rejected,
            )
            _json_print(manager._health_summary(health))
            manager.persist_health(
                manager._health_summary(health),
                rt.settings.data_dir / "taxonomy_health.json",
            )
            if not settings.fake_llm:
                extra = manager.llm_health_review(
                    health, accepted=accepted, rejected=rejected
                )
                if extra:
                    _json_print(extra)

        if args.expand:
            result = manager.expand_strategic(
                max_leaves_to_add=max_leaves,
                facets=facets,
                domain_ids=args.domain,
                accepted=accepted,
                rejected=rejected,
                max_llm_calls=max_calls,
                apply=bool(args.apply),
            )
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
                new_leaves = manager.gap_search(
                    target_l2_key=target,
                    health=health,
                    rejected=rejected,
                    max_new=max_leaves,
                    facet=facet,
                )
                added = manager._commit_leaves(new_leaves, apply=bool(args.apply))
                _json_print(
                    {
                        "target": target,
                        "facet": facet,
                        "total_added": added,
                    }
                )

        if args.refine:
            health = manager.analyze_health(
                accepted_questions=accepted,
                rejected_questions=rejected,
            )
            result = manager.refine_leaf_quality(
                list(health.refinement_needed),
                rewrite_claims=True,
                add_hints=True,
            )
            _json_print(result)
            if args.apply:
                patched = manager.apply_refine(result, apply=True)
                _json_print({"patched": patched})
    finally:
        closer = getattr(rt, "close", None)
        if callable(closer):
            closer()
        reset_runtime(token)
    return 0


def cmd_rebuild_taxonomy(args: argparse.Namespace) -> int:
    from cuda_sft_qgen.graph.nodes.architect import architect_node

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
        out = architect_node({"rebuild_taxonomy": True})
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    finally:
        reset_runtime(token)
    return 0


def build_parser() -> argparse.ArgumentParser:
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
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("resume", help="continue from data/ accepted+pending")
    r.add_argument("-v", "--verbose", action="store_true")
    r.add_argument("--target", type=int, default=None)
    r.add_argument("--batch", type=int, default=None)
    r.add_argument("--fake", action="store_true")
    r.set_defaults(func=cmd_resume, rebuild_taxonomy=False)

    s = sub.add_parser("stats", help="print coverage histograms")
    s.set_defaults(func=cmd_stats)

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
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
