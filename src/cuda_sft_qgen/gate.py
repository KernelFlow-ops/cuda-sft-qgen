"""Throughput and quality gate for a small generate+review run.

Compares wall time and accept rate against a baseline, and records quality
diagnostics without exporting private reference answers.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
from itertools import combinations
import json
import logging
import os
import re
from pathlib import Path
import time
from collections import Counter
from typing import Any

from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.observability import get_metrics
from cuda_sft_qgen.runtime import build_runtime
from cuda_sft_qgen.taxonomy import dump_taxonomy, load_taxonomy

logger = logging.getLogger(__name__)

WALL_RED = 1.25
WALL_YELLOW = 1.10
THROUGHPUT_RED = 0.80
NODE_P95_RED = 1.30
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL|Traceback)\b")
WARN_RE = re.compile(r"\bWARNING\b")
GATE_NODES = (
    "bootstrap", "architect", "taxonomy_health", "planner", "generate",
    "generate_batch", "salvage", "review_score", "review_chair", "archive",
    "taxonomy_manager",
)
LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
LOG_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3,6}")
METRIC_LABEL_RE = re.compile(r"^([^{]+)\{([^}]*)\}$")
STAGE_LOG_RE = {
    "generate": re.compile(r"generate_batch done .*?duration_s=(?P<duration>[0-9.]+)"),
    "review_start": re.compile(r"review_stage start\b"),
    "review_done": re.compile(r"review_stage done duration_s=(?P<duration>[0-9.]+)"),
    "repair_start": re.compile(r"repair_stage start\b"),
    "repair_done": re.compile(r"repair_batch items=\d+ ok=\d+ retry=\d+"),
}
CHECK_NAMES = (
    "correctness",
    "answerability",
    "claim_alignment",
    "hardware_compatibility",
    "answer_leakage",
    "metadata",
    "duplication",
)
# First matching prefix wins. Keep specific tokens ahead of generic ones.
REJECT_CLASSES = (
    ("retry_exhausted", "retry_exhausted"),
    ("review_exhausted", "review_exhausted"),
    ("review_error", "review_error"),
    ("json_parse", "json_parse"),
    ("generate_error", "generate_error"),
    ("sieve_", "sieve"),
    ("corpus_duplicate", "duplicate"),
    ("batch_neighbor", "duplicate"),
    ("knowledge_revision", "knowledge_invalidate"),
    ("429", "rate_limit"),
    ("rate limit", "rate_limit"),
)


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile; None for an empty series."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def _histogram(metrics: dict[str, Any], name: str) -> dict[str, float]:
    hist = (metrics.get("histograms") or {}).get(name) or {}
    return {
        "count": float(hist.get("count") or 0),
        "avg": float(hist.get("avg") or 0),
        "max": float(hist.get("max") or 0),
        "p50": float(hist.get("p50") or 0),
        "p95": float(hist.get("p95") or 0),
    }


def node_seconds(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract per-node duration histograms used by the gate report."""
    out: dict[str, dict[str, float]] = {}
    histograms = metrics.get("histograms") or {}
    for node in GATE_NODES:
        key = f"node_seconds{{node={node}}}"
        if key in histograms:
            out[node] = _histogram(metrics, key)
    return out


def _metric_labels(key: str) -> tuple[str, dict[str, str]]:
    """Split the collector's ``name{key=value,...}`` representation."""
    match = METRIC_LABEL_RE.match(key)
    if not match:
        return key, {}
    labels = dict(part.split("=", 1) for part in match.group(2).split(",") if "=" in part)
    return match.group(1), labels


def llm_activity(metrics: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Summarize actual requests by role, model, and scheduler priority."""
    counters = metrics.get("counters") or {}
    histograms = metrics.get("histograms") or {}
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, count in counters.items():
        metric, labels = _metric_labels(key)
        if metric not in {"llm_requests", "llm_failures"}:
            continue
        group_key = (labels.get("priority", "unknown"), labels.get("model", "unknown"),
                     labels.get("role", "unspecified"))
        group = groups.setdefault(group_key, {
            "priority": group_key[0], "model": group_key[1], "role": group_key[2],
            "calls": 0, "failures": 0,
        })
        group["calls" if metric == "llm_requests" else "failures"] += int(count or 0)

    for key, histogram in histograms.items():
        metric, labels = _metric_labels(key)
        if metric != "llm_seconds":
            continue
        group = groups.setdefault((labels.get("priority", "unknown"), labels.get("model", "unknown"),
                                   labels.get("role", "unspecified")), {
            "priority": labels.get("priority", "unknown"), "model": labels.get("model", "unknown"),
            "role": labels.get("role", "unspecified"), "calls": 0, "failures": 0,
        })
        group["duration_seconds"] = _histogram({"histograms": {"sample": histogram}}, "sample")

    configured = {
        role: str(getattr(api, "model", ""))
        for role, api in (getattr(settings, "agent_apis", {}) or {}).items()
    }
    priority_roles = {
        "0": ["writer", "evolver"],
        "1": ["writer", "salvage"],
        "2": ["writer", "review", "chair", "verifier"],
        "4": ["writer", "knowledge"],
    }
    observed = []
    for (priority, model, role), group in sorted(groups.items()):
        candidates = ([role] if role != "unspecified" else
                      [candidate for candidate in priority_roles.get(priority, [])
                       if candidate not in configured or configured[candidate] == model])
        observed.append({**group, "role_candidates": candidates})
    return {
        "attribution": (
            "Calls with a role label are attributed to that Agent; legacy calls without one list possible roles."
        ),
        "request_groups": observed,
        "configured_role_models": configured,
        "priority_roles": priority_roles,
    }


def stage_activity(path: Path, run_started: float) -> dict[str, Any]:
    """Recover concurrent generation/review/repair intervals from pipeline logs."""
    intervals: list[dict[str, Any]] = []
    open_starts: dict[str, list[datetime]] = {"review": [], "repair": []}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stamp = LOG_TIMESTAMP_RE.match(line)
            if stamp is None:
                continue
            try:
                timestamp = datetime.strptime(stamp.group(), LOG_TIMESTAMP_FORMAT)
            except ValueError:
                continue
            gen = STAGE_LOG_RE["generate"].search(line)
            if gen:
                end = (timestamp.timestamp() - run_started)
                duration = float(gen.group("duration"))
                intervals.append({"stage": "generation", "start_seconds": end - duration,
                                  "end_seconds": end, "duration_seconds": duration})
                continue
            for stage in ("review", "repair"):
                if STAGE_LOG_RE[f"{stage}_start"].search(line):
                    open_starts[stage].append(timestamp)
                    break
                done = STAGE_LOG_RE[f"{stage}_done"].search(line)
                if done and open_starts[stage]:
                    start = open_starts[stage].pop(0)
                    duration = max(0.0, (timestamp - start).total_seconds())
                    start_offset = start.timestamp() - run_started
                    intervals.append({"stage": stage, "start_seconds": start_offset,
                                      "end_seconds": start_offset + duration,
                                      "duration_seconds": duration})
                    break

    intervals.sort(key=lambda item: (item["start_seconds"], item["stage"]))
    overlaps: dict[str, dict[str, float | int]] = {}
    for left, right in combinations(intervals, 2):
        if left["stage"] == right["stage"]:
            continue
        seconds = min(left["end_seconds"], right["end_seconds"]) - max(
            left["start_seconds"], right["start_seconds"])
        if seconds <= 0:
            continue
        pair = "+".join(sorted((left["stage"], right["stage"])))
        summary = overlaps.setdefault(pair, {"count": 0, "overlap_seconds": 0.0})
        summary["count"] = int(summary["count"]) + 1
        summary["overlap_seconds"] = float(summary["overlap_seconds"]) + seconds
    return {
        "intervals": intervals,
        "overlaps": overlaps,
        "interval_counts": dict(Counter(item["stage"] for item in intervals)),
        "unmatched_starts": {stage: len(starts) for stage, starts in open_starts.items() if starts},
    }


def _secure_output_tree(output: Path) -> None:
    """Restrict gate artifact permissions after all runtime writers have closed."""
    paths = sorted(output.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        if path.is_symlink():
            continue
        path.chmod(0o700 if path.is_dir() else 0o600)
    output.chmod(0o700)


def scan_log(path: Path) -> dict[str, Any]:
    """Count ERROR/WARNING lines in a pipeline log for the gate verdict."""
    errors: list[str] = []
    warnings = 0
    if not path.exists():
        return {"log_errors": 0, "log_warnings": 0, "log_error_samples": []}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ERROR_RE.search(line):
            errors.append(line[:400])
        elif WARN_RE.search(line):
            warnings += 1
    return {
        "log_errors": len(errors),
        "log_warnings": warnings,
        "log_error_samples": errors[:12],
    }


def classify_reject_reason(reason: str | None) -> str:
    """Map a reject/defer reason onto a coarse gate error class."""
    text = (reason or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    for token, label in REJECT_CLASSES:
        if token in lowered:
            return label
    return "other"


def _classify_item(question: Question) -> str:
    """Prefer structured prefixes; fall back to the first failed chair check."""
    labeled = classify_reject_reason(question.reject_reason)
    if labeled not in {"other", "unknown"}:
        return labeled
    checks = (question.review_details or {}).get("checks") or {}
    if isinstance(checks, dict):
        for name in CHECK_NAMES:
            if str(checks.get(name) or "").lower() == "fail":
                return f"chair_{name}"
    action = str((question.review_details or {}).get("action") or "").lower()
    if action == "reject":
        return "chair_reject"
    return labeled


def _check_histogram(questions: list[Question]) -> dict[str, dict[str, int]]:
    """Count pass/fail/uncertain for each chair dimension across reviewed items."""
    hist = {name: Counter() for name in CHECK_NAMES}
    for question in questions:
        checks = (question.review_details or {}).get("checks") or {}
        if not isinstance(checks, dict):
            continue
        for name in CHECK_NAMES:
            value = str(checks.get(name) or "").strip().lower()
            if value in {"pass", "fail", "uncertain"}:
                hist[name][value] += 1
    return {name: dict(counts) for name, counts in hist.items()}


def build_quality_report(
    accepted: list[Question],
    rejected: list[Question],
    pending: list[Question],
) -> dict[str, Any]:
    """Summarize accept/reject/pending mix, error classes, and chair check fails.

    Args:
        accepted: Independently accepted questions.
        rejected: Hard rejects (sieve, dup, chair reject).
        pending: Mid-review or exhausted items.

    Returns:
        JSON-serializable quality block for ``gate_report.json``.
    """
    total = len(accepted) + len(rejected) + len(pending)
    reject_classes = Counter(_classify_item(q) for q in rejected)
    pending_classes = Counter(_classify_item(q) for q in pending)
    reviewed = [q for q in accepted + rejected + pending if q.review_details]
    answers = [q for q in accepted if (q.internal_reference_answer or "").strip()]
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "pending": len(pending),
        "total": total,
        "accept_rate": (len(accepted) / total) if total else 0.0,
        "reject_rate": (len(rejected) / total) if total else 0.0,
        "pending_rate": (len(pending) / total) if total else 0.0,
        "reject_classes": dict(reject_classes),
        "pending_classes": dict(pending_classes),
        "chair_checks": _check_histogram(reviewed),
        "reference_answer_rate": (len(answers) / len(accepted)) if accepted else 0.0,
        "reference_answer_n": len(answers),
        "salvage_or_json_rejects": int(reject_classes.get("json_parse", 0)),
        "sieve_rejects": int(reject_classes.get("sieve", 0)),
        "rate_limit_rejects": int(reject_classes.get("rate_limit", 0)),
    }


def write_quality_sample(
    accepted: list[Question],
    rejected: list[Question],
    pending: list[Question],
    path: Path,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Write up to ``limit`` accepted stems; pad with rejects and pending items.

    Returns:
        The records written to ``path``.
    """
    records: list[dict[str, Any]] = []
    for question in accepted[:limit]:
        details = question.review_details or {}
        records.append({
            "id": question.id,
            "status": "accepted",
            "question": question.question,
            "tests_claim": question.tests_claim,
            "l3_id": question.l3_id,
            "action": details.get("action") or "accept",
            "reason_code": classify_reject_reason(question.reject_reason or details.get("reason")),
            "checks": details.get("checks") or {},
            "language": question.language,
            "question_type": question.question_type,
        })
    if len(records) < limit:
        for question in list(rejected) + list(pending):
            if len(records) >= limit:
                break
            details = question.review_details or {}
            records.append({
                "id": question.id,
                "status": question.review_status,
                "question": question.question,
                "tests_claim": question.tests_claim,
                "l3_id": question.l3_id,
                "action": details.get("action") or question.review_status,
                "reason_code": classify_reject_reason(question.reject_reason or details.get("reason")),
                "checks": details.get("checks") or {},
                "language": question.language,
                "question_type": question.question_type,
            })
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    return records


def median_baseline(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Same-day API jitter: median of numeric gate fields, keep first report's labels."""
    if not reports:
        raise ValueError("median_baseline requires at least one report")
    merged = json.loads(json.dumps(reports[0]))
    keys = ["duration_seconds", "accepted_per_hour", "llm_calls", "knowledge_jobs_done"]
    for key in keys:
        values = sorted(float(r.get(key) or 0) for r in reports)
        n = len(values)
        merged[key] = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    node_names = set()
    for report in reports:
        node_names.update((report.get("node_seconds") or {}).keys())
    nodes: dict[str, dict[str, float]] = {}
    for node in node_names:
        series = [r.get("node_seconds", {}).get(node) or {} for r in reports]
        sample = next((s for s in series if s), {})
        out = dict(sample)
        for stat in ("p50", "p95", "avg", "max", "count"):
            values = sorted(float(s.get(stat) or 0) for s in series)
            n = len(values)
            out[stat] = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        nodes[node] = out
    merged["node_seconds"] = nodes
    merged["phase"] = "baseline-median"
    return merged


def compare_gate_reports(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    target: int | None = None,
) -> dict[str, Any]:
    """Return verdict payload. No baseline → recorded as the origin, not a pass/fail."""
    failures: list[str] = []
    warnings: list[str] = []
    want = int(target if target is not None else current.get("target") or 10)
    accepted = int(current.get("exportable", current.get("accepted")) or 0)
    exhausted = int(current.get("exhausted") or 0)
    duration = float(current.get("duration_seconds") or 0)
    rate = float(current.get("accepted_per_hour") or 0)

    if current.get("error"):
        failures.append(f"pipeline_error: {current['error']}")
    if accepted < want:
        label = "exportable" if "exportable" in current else "accepted"
        failures.append(f"{label} {accepted} < target {want}")
    limit = current.get("time_limit_seconds")
    if limit is not None and duration > float(limit):
        failures.append(f"duration {duration:.1f}s > limit {float(limit):.1f}s")
    if exhausted:
        failures.append(f"exhausted_reviews={exhausted}")
    if int(current.get("log_errors") or 0) > 0:
        failures.append(f"log_errors={current.get('log_errors')}")
    job_errors = int(current.get("knowledge_job_errors") or 0)
    if job_errors >= 3:
        failures.append(f"knowledge_job_errors={job_errors}")

    if baseline is None:
        verdict = "red" if failures else "baseline"
        return {
            "verdict": verdict,
            "failures": failures,
            "warnings": warnings,
            "ratios": {},
        }

    base_duration = max(float(baseline.get("duration_seconds") or 0), 0.001)
    base_rate = max(float(baseline.get("accepted_per_hour") or 0), 0.001)
    wall_ratio = duration / base_duration
    rate_ratio = rate / base_rate
    ratios = {"wall": wall_ratio, "accepted_per_hour": rate_ratio, "node_p95": {}}

    if wall_ratio > WALL_RED:
        failures.append(f"wall {wall_ratio:.2f}x > {WALL_RED}x baseline")
    elif wall_ratio > WALL_YELLOW:
        warnings.append(f"wall {wall_ratio:.2f}x yellow (>{WALL_YELLOW}x)")
    if rate_ratio < THROUGHPUT_RED:
        failures.append(f"accepted_per_hour {rate_ratio:.2f}x < {THROUGHPUT_RED}x baseline")

    cur_nodes = current.get("node_seconds") or {}
    base_nodes = baseline.get("node_seconds") or {}
    for node in ("generate_batch", "generate", "review_chair"):
        cur_node = cur_nodes.get(node) or {}
        base_node = base_nodes.get(node) or {}
        cur_p95 = float(cur_node.get("p95") or 0)
        base_p95 = float(base_node.get("p95") or 0)
        cur_n = float(cur_node.get("count") or 0)
        base_n = float(base_node.get("count") or 0)
        if base_p95 <= 0:
            continue
        ratio = cur_p95 / base_p95
        ratios["node_p95"][node] = ratio
        if min(cur_n, base_n) < 5:
            warnings.append(
                f"{node} p95 {ratio:.2f}x ignored as red (n={cur_n:.0f}/{base_n:.0f} < 5; 10-q batches are noisy)"
            )
            continue
        if ratio > NODE_P95_RED:
            failures.append(f"{node} p95 {ratio:.2f}x > {NODE_P95_RED}x baseline")

    if failures:
        verdict = "red"
    elif warnings:
        verdict = "yellow"
    else:
        verdict = "green"
    return {"verdict": verdict, "failures": failures, "warnings": warnings, "ratios": ratios}


def write_verdict_md(payload: dict[str, Any], path: Path) -> None:
    """Write a short Markdown verdict next to ``gate_report.json``."""
    report = payload.get("report") or payload
    compared = payload.get("compared") or {}
    lines = [
        f"# Gate {report.get('phase', '')}",
        "",
        f"- verdict: **{compared.get('verdict') or report.get('verdict')}**",
        f"- accepted: {report.get('accepted')}/{report.get('target')}",
        f"- exportable: {report.get('exportable')}/{report.get('target')}",
        f"- duration_seconds: {report.get('duration_seconds')}",
        f"- seconds_per_exportable: {report.get('seconds_per_exportable')}",
        f"- accepted_per_hour: {report.get('accepted_per_hour')}",
        f"- pending: {report.get('pending')} exhausted: {report.get('exhausted')}",
        f"- knowledge_jobs_done: {report.get('knowledge_jobs_done')} errors: {report.get('knowledge_job_errors')}",
        f"- knowledge_context_chars_p95: {(report.get('knowledge_context_chars') or {}).get('p95')}",
        f"- llm_request_groups: {len((report.get('agent_activity') or {}).get('request_groups') or [])}",
        f"- stage_intervals: {len((report.get('stage_activity') or {}).get('intervals') or [])}",
        f"- stage_overlaps: {len((report.get('stage_activity') or {}).get('overlaps') or {})}",
        "",
        "## Quality",
        f"- accept_rate: {(report.get('quality') or {}).get('accept_rate')}",
        f"- reject_classes: {(report.get('quality') or {}).get('reject_classes')}",
        f"- reference_answer_rate: {(report.get('quality') or {}).get('reference_answer_rate')}",
        f"- salvage_or_json_rejects: {(report.get('quality') or {}).get('salvage_or_json_rejects')}",
        f"- sieve_rejects: {(report.get('quality') or {}).get('sieve_rejects')}",
        "",
        "## Failures",
    ]
    failures = compared.get("failures") or report.get("failures") or []
    lines.extend(f"- {item}" for item in failures or ["(none)"])
    lines += ["", "## Warnings"]
    warnings = compared.get("warnings") or report.get("warnings") or []
    lines.extend(f"- {item}" for item in warnings or ["(none)"])
    if compared.get("ratios"):
        lines += ["", "## Ratios vs baseline", f"```json\n{json.dumps(compared['ratios'], indent=2)}\n```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_gate_impl(
    settings,
    output: Path,
    *,
    phase: str,
    target: int = 10,
    timeout: float = 2700,
    max_iterations: int = 40,
    baseline_path: Path | None = None,
    allow_fake: bool = False,
    ingest_packs: list[Path] | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    """Run a short generate+review in a fresh directory and compare to baseline.

    Raises:
        ValueError: If ``output`` is non-empty, or FakeLLM is used without
            ``allow_fake``.
    """
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Gate output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    if settings.fake_llm and not allow_fake:
        raise ValueError("real gate refuses FakeLLM; pass allow_fake only in tests")

    taxonomy = load_taxonomy(settings.taxonomy_path)
    taxonomy.domain_files = {}
    snapshot = output / "taxonomy.yaml"
    dump_taxonomy(taxonomy, snapshot)

    config = json.loads(json.dumps(settings.default_yaml))
    config.setdefault("paths", {})["taxonomy"] = str(snapshot)
    config.setdefault("knowledge", {})["enabled"] = True
    review_cfg = config.setdefault("review", {})
    review_cfg["batch_size"] = min(int(review_cfg.get("batch_size") or target), target)
    config.setdefault("generation", {})["max_runtime_seconds"] = timeout
    config["generation"]["concurrent_pipeline"] = True

    fixed = replace(
        settings,
        default_yaml=config,
        data_dir_override=output,
        target_question_count=target,
        max_iterations=max_iterations,
        review_batch_size=min(settings.review_batch_size, target),
    )
    get_metrics().reset()
    started = started_at if started_at is not None else time.time()
    log_path = output / "pipeline.log"
    run_meta = {
        "phase": phase,
        "pid": os.getpid(),
        "started_at": started,
        "log": str(log_path),
        "data_dir": str(output),
        "target": target,
        "fake": bool(fixed.fake_llm),
        "knowledge_enabled": True,
        "models": {role: api.model for role, api in fixed.agent_apis.items()},
        "writer_thinking_level": fixed.writer_thinking_level,
        "review_thinking_level": fixed.review.get("thinking_level"),
        "solver_thinking_level": fixed.review.get("solver_thinking_level"),
        "writer_concurrency": fixed.writer_concurrency,
        "llm_per_key_concurrency": fixed.llm_per_key_concurrency,
        "chair_max_workers": fixed.review.get("chair_max_workers"),
        "chair_trigger_count": fixed.review.get("chair_trigger_count"),
        "min_generate_batch": fixed.min_generate_batch,
    }
    (output / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)

    error = None
    export_manifest: dict[str, Any] = {}
    result: dict[str, Any] = {"status": "failed", "done": False}
    knowledge_stats: dict[str, Any] = {}
    rt = None
    try:
        from cuda_sft_qgen.graph.app import pipeline_lock

        with pipeline_lock(fixed.data_dir):
            try:
                rt = build_runtime(fixed)
                if ingest_packs:
                    from cuda_sft_qgen.pack import ingest_pack, load_pack
                    for pack_path in ingest_packs:
                        ingest_pack(rt.knowledge, load_pack(pack_path), apply=True, enqueue_jobs=False)
                    rt.refresh_knowledge(force=True)
                try:
                    result = run_pipeline(fixed, runtime=rt, lock_held=True)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.exception("gate pipeline failed")
                accepted = rt.store.load_accepted()
                rejected = rt.store.load_rejected()
                pending = rt.store.load_pending()
                if rt.knowledge is not None:
                    knowledge_stats = rt.knowledge.stats()
                rt.refresh_knowledge(force=True)
                export_manifest = rt.store.export_training(
                    output / "training.jsonl", require_independent_reviewer=True)
            finally:
                if rt is not None:
                    try:
                        rt.close()
                    except Exception as exc:
                        error = error or f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"
        accepted, rejected, pending = [], [], []
        logger.exception("gate runtime failed")
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    metrics = get_metrics().snapshot()
    elapsed = time.time() - started
    exportable = int(export_manifest.get("questions") or 0)
    counters = metrics.get("counters") or {}
    prompt_tokens = sum(v for k, v in counters.items() if k.startswith("llm_prompt_tokens"))
    completion_tokens = sum(v for k, v in counters.items() if k.startswith("llm_completion_tokens"))
    calls = sum(v for k, v in counters.items() if k.startswith("llm_requests"))
    jobs = knowledge_stats.get("jobs") or {}
    log_info = scan_log(log_path)
    exhausted = sum(1 for q in pending if getattr(q, "review_exhausted", False))
    quality = build_quality_report(accepted, rejected, pending)
    write_quality_sample(accepted, rejected, pending, output / "quality_sample.jsonl")
    report = {
        "phase": phase,
        "status": result.get("status"),
        "done": result.get("done", False),
        "error": error,
        "fake": bool(fixed.fake_llm),
        "knowledge_enabled": True,
        "target": target,
        "accepted": len(accepted),
        "exportable": exportable,
        "export_manifest": export_manifest,
        "rejected": len(rejected),
        "pending": len(pending),
        "exhausted": exhausted,
        "duration_seconds": elapsed,
        "time_limit_seconds": timeout,
        "seconds_per_exportable": elapsed / exportable if exportable else None,
        "speed_met": exportable >= target and elapsed <= timeout and error is None,
        "accepted_per_hour": exportable * 3600 / max(elapsed, 0.001),
        "covered_leaves": len({q.l3_id for q in accepted}),
        "llm_calls": calls,
        "reported_prompt_tokens": prompt_tokens,
        "reported_completion_tokens": completion_tokens,
        "knowledge_jobs": jobs,
        "knowledge_jobs_done": int(jobs.get("done") or 0),
        "knowledge_job_errors": int(jobs.get("failed") or 0),
        "knowledge_claims": knowledge_stats.get("claims") or {},
        "knowledge_context_chars": _histogram(metrics, "knowledge_context_chars"),
        "node_seconds": node_seconds(metrics),
        "agent_activity": llm_activity(metrics, fixed),
        "stage_activity": stage_activity(log_path, started),
        "taxonomy_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "model": fixed.llm_model,
        "review_model": fixed.review_model,
        "writer_concurrency": fixed.writer_concurrency,
        "metrics": metrics,
        "quality": quality,
        **log_info,
    }
    baseline = None
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    compared = compare_gate_reports(report, baseline, target=target)
    report["verdict"] = compared["verdict"]
    report["failures"] = compared["failures"]
    report["warnings"] = compared["warnings"]
    report["ratios"] = compared["ratios"]
    (output / "gate_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_verdict_md({"report": report, "compared": compared}, output / "verdict.md")
    return report


def run_gate(
    settings,
    output: Path,
    *,
    phase: str,
    target: int = 10,
    timeout: float = 2700,
    max_iterations: int = 40,
    baseline_path: Path | None = None,
    allow_fake: bool = False,
    ingest_packs: list[Path] | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    """Run the gate with owner-only permissions on its output tree."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Gate output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    try:
        return _run_gate_impl(
            settings,
            output,
            phase=phase,
            target=target,
            timeout=timeout,
            max_iterations=max_iterations,
            baseline_path=baseline_path,
            allow_fake=allow_fake,
            ingest_packs=ingest_packs,
            started_at=started_at,
        )
    finally:
        _secure_output_tree(output)
