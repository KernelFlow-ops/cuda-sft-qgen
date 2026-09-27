"""Reproducible isolated generation benchmark with measured usage and coverage."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from cuda_sft_qgen.graph.app import pipeline_lock, run_pipeline
from cuda_sft_qgen.observability import get_metrics
from cuda_sft_qgen.runtime import build_runtime
from cuda_sft_qgen.taxonomy import dump_taxonomy, load_taxonomy


def run_benchmark(settings, output: Path, *, target: int = 100,
                  max_iterations: int = 40, timeout: float = 3600) -> dict:
    """Run an isolated generate+review into ``output`` with knowledge disabled."""
    if settings.fake_llm:
        raise RuntimeError("Generation benchmarks require a real LLM API")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    taxonomy.domain_files = {}
    snapshot = output / "taxonomy.yaml"
    dump_taxonomy(taxonomy, snapshot)
    config = json.loads(json.dumps(settings.default_yaml))
    config.setdefault("paths", {})["taxonomy"] = str(snapshot)
    config.setdefault("taxonomy", {}).setdefault("manager", {})["enabled"] = False
    config.setdefault("knowledge", {})["enabled"] = False
    config.setdefault("generation", {})["max_runtime_seconds"] = timeout
    config["generation"]["concurrent_pipeline"] = True
    fixed = replace(settings, default_yaml=config, data_dir_override=output,
                    target_question_count=target, max_iterations=max_iterations)
    get_metrics().reset()
    started = time.time()
    manifest = {
        "started_at": started, "target": target, "fake": fixed.fake_llm,
        "model": fixed.llm_model, "review_model": fixed.review_model,
        "embedding_model": fixed.embedding_model,
        "writer_concurrency": fixed.writer_concurrency,
        "review_workers": fixed.review.get("chair_max_workers", 4),
        "taxonomy_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "config": config,
    }
    (output / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    error = None
    finally_error = None
    with pipeline_lock(output):
        rt = build_runtime(fixed)
        try:
            try:
                result = run_pipeline(fixed, runtime=rt, lock_held=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = {"status": "failed", "done": False}
            accepted = rt.store.load_accepted()
            rejected = rt.store.load_rejected()
            pending = rt.store.load_pending()
            training_manifest = rt.store.export_training(output / "training.jsonl")
        finally:
            try:
                rt.close()
            except Exception as exc:
                finally_error = f"{type(exc).__name__}: {exc}"
    metrics = get_metrics().snapshot()
    llm_pool = rt.pool.stats() if rt.pool is not None else None
    review_pool_ref = rt.role_pools.get("review")
    review_pool = review_pool_ref.stats() if review_pool_ref is not None else None
    elapsed = time.time() - started
    counters = metrics["counters"]
    candidate_latency = metrics["histograms"].get("generate_brief_seconds{status=ok}") or {}
    candidate_count = int(candidate_latency.get("count", 0))
    slow_candidates = int(counters.get("generate_over_30s", 0))
    prompt_tokens = sum(v for k, v in counters.items() if k.startswith("llm_prompt_tokens"))
    completion_tokens = sum(v for k, v in counters.items() if k.startswith("llm_completion_tokens"))
    usage_reports = sum(v for k, v in counters.items() if k.startswith("llm_usage_reports"))
    calls = sum(v for k, v in counters.items() if k.startswith("llm_requests"))
    pricing = config.get("pricing") or {}
    priced = (usage_reports > 0 and "llm_input_per_million" in pricing
              and "llm_output_per_million" in pricing)
    estimated_cost = ((prompt_tokens * float(pricing["llm_input_per_million"])
                       + completion_tokens * float(pricing["llm_output_per_million"])) / 1_000_000
                      if priced else None)
    report = {
        "status": result.get("status"), "done": result.get("done", False),
        "error": error or finally_error,
        "fake": fixed.fake_llm, "target": target, "accepted": len(accepted),
        "rejected": len(rejected), "pending": len(pending), "duration_seconds": elapsed,
        "accepted_per_hour": len(accepted) * 3600 / max(elapsed, 0.001),
        "covered_leaves": len({q.l3_id for q in accepted}),
        "leaf_coverage": len({q.l3_id for q in accepted}) / max(len(taxonomy.leaves), 1),
        "reviewed_accepted": sum(q.review_details.get("action") == "accept" for q in accepted),
        "candidate_latency_p95_seconds": candidate_latency.get("p95"),
        "candidate_under_30_rate": ((candidate_count - slow_candidates) / candidate_count
                                    if candidate_count else None),
        "llm_pool_peak": llm_pool.get("peak_llm") if llm_pool else None,
        "llm_pool_capacity": llm_pool.get("llm_capacity") if llm_pool else None,
        "review_pool_peak": review_pool.get("peak_llm") if review_pool else None,
        "review_pool_capacity": review_pool.get("llm_capacity") if review_pool else None,
        "llm_calls": calls, "calls_per_accepted": calls / max(len(accepted), 1),
        "reported_prompt_tokens": prompt_tokens, "reported_completion_tokens": completion_tokens,
        "usage_reports": usage_reports, "estimated_llm_cost_usd": estimated_cost,
        "cost_per_accepted_usd": estimated_cost / max(len(accepted), 1) if priced else None,
        "cost_note": "Provider billing is not inferred; cost requires configured rates and usage reports.",
        "human_defect_rate": None, "human_audit_note": "Requires independent inspection of sample.jsonl.",
        "metrics": metrics,
    }
    from cuda_sft_qgen.store.jsonl import write_jsonl
    write_jsonl(output / "sample.jsonl", [q.to_jsonl() for q in accepted[:20]])
    (output / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
