from __future__ import annotations

import argparse
import json
import logging
import sys

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
