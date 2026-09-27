"""Generate, resume, stats, export, benchmark, and probe commands."""

from __future__ import annotations

import json
from pathlib import Path

from cuda_sft_qgen.cli.common import _json_print, _require_keys, load_settings
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.runtime import build_runtime
from cuda_sft_qgen.store.archive import Archive


def cmd_generate(args) -> int:
    """Run the generate+review pipeline until target or iteration limit."""
    if getattr(args, "fake", False):
        raise ValueError("--fake cannot generate questions; configure a real LLM API")
    data_dir = Path(args.data_dir) if getattr(args, "data_dir", None) else None
    settings = load_settings(
        fake_llm=False,
        target=args.target,
        review_batch_size=args.batch,
        data_dir=data_dir,
    )
    if (code := _require_keys(settings)) is not None:
        return code
    result = run_pipeline(settings, rebuild_taxonomy=args.rebuild_taxonomy,
                          retry_reviews=bool(getattr(args, "retry_reviews", False)))
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
    return 0 if result.get("done") else 1


def cmd_resume(args) -> int:
    """Continue a previous run from the archive in ``--data-dir``."""
    if getattr(args, "fake", False):
        raise ValueError("--fake cannot resume question generation")
    args.rebuild_taxonomy = False
    return cmd_generate(args)


def cmd_export(args) -> int:
    """Export independently reviewed accepted questions for SFT training."""
    settings = load_settings(fake_llm=True, data_dir=Path(args.data_dir) if args.data_dir else None)
    rt = build_runtime(settings)
    try:
        rt.refresh_knowledge(force=True)
        _json_print(rt.store.export_training(Path(args.output)))
    finally:
        rt.close()
    return 0


def cmd_benchmark(args) -> int:
    """Isolated generation benchmark with taxonomy/knowledge sidecars disabled."""
    if getattr(args, "fake", False):
        raise ValueError("--fake cannot benchmark question generation")
    from cuda_sft_qgen.benchmark import run_benchmark

    settings = load_settings(fake_llm=False)
    report = run_benchmark(
        settings,
        Path(args.output),
        target=args.target,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
    )
    _json_print(report)
    return 0 if report.get("done") else 1


def cmd_stats(_args) -> int:
    """Print archive histograms from ``data/stats.json``."""
    settings = load_settings(
        fake_llm=True,
        data_dir=Path(_args.data_dir) if getattr(_args, "data_dir", None) else None,
    )
    archive = Archive(settings.data_dir)
    stats = archive.compute_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_agent_pilot(args) -> int:
    """Run reproducible ablations; FakeLLM cannot establish Agent quality."""
    from cuda_sft_qgen.agent_pilot import run_agent_pilot
    settings = load_settings(fake_llm=False)
    if (code := _require_keys(settings)) is not None:
        return code
    report = run_agent_pilot(settings, Path(args.output), samples=args.samples,
                             max_calls=args.max_calls, groups=args.groups,
                             deadline_seconds=args.timeout)
    _json_print(report.get("summary", report))
    return 0 if report.get("status") == "completed" else 1


def cmd_probe_concurrency(args) -> int:
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
