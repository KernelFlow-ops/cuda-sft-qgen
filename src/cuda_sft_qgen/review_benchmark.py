"""Isolated, reproducible quality-review evaluation on frozen labeled questions."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.graph.nodes.review import _parallel_chair_decision
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.observability import get_metrics
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_quality_cases(path: Path) -> list[dict[str, Any]]:
    """Load schema-v1 labeled cases; each must have a unique id and accept/reject label."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported quality-case schema version")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Quality benchmark requires at least one case")
    ids = set()
    for case in cases:
        if not case.get("case_id") or case["case_id"] in ids:
            raise ValueError("Quality case IDs must be present and unique")
        ids.add(case["case_id"])
        if case.get("expected_action") not in {"accept", "reject"}:
            raise ValueError("Expected action must be accept or reject")
        Question.model_validate(case["question"])
    return cases


def _usage(before: dict[str, Any], after: dict[str, Any], model: str) -> dict[str, Any]:
    def delta(name: str) -> int:
        key = f"llm_{name}{{model={model}}}"
        return after.get("counters", {}).get(key, 0) - before.get("counters", {}).get(key, 0)

    reports = delta("usage_reports")
    return {"available": reports > 0, "provider_reports": reports,
            **{name: delta(name) if reports else None
               for name in ("prompt_tokens", "completion_tokens", "total_tokens")}}


def _summary(results: list[dict[str, Any]], expected_cases: int) -> dict[str, Any]:
    defects = [item for item in results if item["expected_action"] == "reject"]
    controls = [item for item in results if item["expected_action"] == "accept"]
    misses = sum(item["decision"]["action"] == "accept" for item in defects)
    false_rejects = sum(item["decision"]["action"] == "reject" for item in controls)
    return {
        "case_count": expected_cases, "completed": len(results),
        "defect_count": len(defects), "control_count": len(controls),
        "detected_defects": sum(item["decision"]["action"] == "reject" for item in defects),
        "accepted_controls": sum(item["decision"]["action"] == "accept" for item in controls),
        "misses": misses, "false_rejections": false_rejects,
        "pending": sum(item["decision"]["action"] == "pending" for item in results),
        "pending_defects": sum(item["decision"]["action"] == "pending" for item in defects),
        "pending_controls": sum(item["decision"]["action"] == "pending" for item in controls),
        "miss_rate_all_defects": misses / len(defects) if defects else None,
        "false_rejection_rate_all_controls": false_rejects / len(controls) if controls else None,
    }


def _save_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


class _RecordedReviewer:
    """Wrap a reviewer LLM and record each complete_json call for the report."""
    def __init__(self, delegate: Any):
        """Wrap ``delegate.complete_json`` and record each call."""
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Forward to the delegate and store messages, options, and the response."""
        call = {"messages": deepcopy(messages), "options": deepcopy(kwargs)}
        self.calls.append(call)
        try:
            result = self.delegate.complete_json(messages, **kwargs)
        except Exception as exc:
            call["error_type"] = type(exc).__name__
            raise
        call["response"] = deepcopy(result)
        return result


def run_review_benchmark(settings: Settings, output: Path | str, *,
                         cases_path: Path | None = None) -> dict[str, Any]:
    """Review cases sequentially without reading or updating the production archive.

    This evaluates question quality only: every case has an empty neighbor set.
    Provider token counters are sampled around each isolated call; missing usage
    remains null. Run this benchmark without concurrent work in the same process.
    """
    if settings.fake_llm:
        raise ValueError("Quality benchmarking requires a real reviewer; fake outputs cannot establish quality")
    cases_path = cases_path or settings.root / "tests" / "fixtures" / "quality_cases.json"
    cases = load_quality_cases(cases_path)
    output = Path(output)
    if output.resolve() == cases_path.resolve():
        raise ValueError("Benchmark output must not overwrite its case fixture")
    from cuda_sft_qgen.graph.nodes.review_chair import render_chair_prompt
    prompt = render_chair_prompt(settings)
    model = settings.review_model
    report: dict[str, Any] = {
        "schema_version": 1, "benchmark": "question_quality", "mode": "real_llm",
        "started_at": _timestamp(), "finished_at": None,
        "scope": "Independent per-question quality; empty neighbors; no corpus deduplication evaluation",
        "fixture": {"path": str(cases_path),
                    "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()},
        "reviewer": {"model": model, "base_url": settings.agent_apis["review"].base_url,
                     "provider": settings.agent_apis["review"].provider,
                     "temperature": 0.0, "thinking_level": settings.review.get("thinking_level", "medium"),
                     "max_tokens": settings.review.get("max_tokens", 8192)},
        "system_prompt": prompt,
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "results": [], "summary": _summary([], len(cases)),
    }
    started = time.monotonic()
    before_run = get_metrics().snapshot()
    with tempfile.TemporaryDirectory(prefix="qgen-review-benchmark-") as directory:
        isolated = replace(settings, data_dir_override=Path(directory),
                           embedding_provider="fake", embedding_cache_enabled=False)
        runtime = build_runtime(isolated)
        recorder = _RecordedReviewer(runtime.review_llm or runtime.llm)
        runtime.review_llm = recorder
        token = set_runtime(runtime)
        try:
            _save_report(output, report)
            for case in cases:
                question = Question.model_validate(case["question"])
                item = {"question": question.model_dump(), "decision": {"verdict": "unique"},
                        "neighbors": []}
                before = get_metrics().snapshot()
                case_started = time.monotonic()
                case_timestamp = _timestamp()
                first_call = len(recorder.calls)
                decision = _parallel_chair_decision(item, prompt, allow_fixture_origin=True)
                if decision.get("action") not in {"accept", "reject", "pending"}:
                    decision = {"id": question.id, "action": "pending", "reason": "invalid_benchmark_decision"}
                report["results"].append({
                    "case_id": case["case_id"], "expected_action": case["expected_action"],
                    "source": case.get("source", {}), "annotation": case.get("annotation", ""),
                    "question_snapshot": deepcopy(case["question"]), "started_at": case_timestamp,
                    "latency_seconds": time.monotonic() - case_started, "decision": decision,
                    "reviewer_calls": deepcopy(recorder.calls[first_call:]),
                    "usage": _usage(before, get_metrics().snapshot(), model),
                })
                report["summary"] = _summary(report["results"], len(cases))
                _save_report(output, report)
        finally:
            reset_runtime(token)
            runtime.close()
    report["finished_at"] = _timestamp()
    report["duration_seconds"] = time.monotonic() - started
    report["usage"] = _usage(before_run, get_metrics().snapshot(), model)
    _save_report(output, report)
    return report
