"""Gate, review-benchmark, validate-config, and estimate-cost commands."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cuda_sft_qgen.cli.common import _json_print, load_settings

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
    missing = settings.missing_api_keys + settings.missing_api_fields
    if missing:
        print(
            f"Missing API configuration: {', '.join(missing)}.",
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
        if config.missing_api_keys:
            issues.append(f"Missing API keys: {', '.join(config.missing_api_keys)}")
        if config.missing_api_fields:
            issues.append(f"Missing API fields: {', '.join(config.missing_api_fields)}")

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
