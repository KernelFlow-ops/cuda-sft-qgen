#!/usr/bin/env python3
"""Run bounded, isolated real-API probes for pipeline Agent roles.

The probe reads one accepted Writer question and its original brief from a
completed gate directory, then writes only role outcomes and timings.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from urllib.parse import quote
import uuid

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.errors import JsonSalvageNeeded
from cuda_sft_qgen.graph.nodes.generate_core import GenerateOutcome
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.observability import current_agent_role, get_metrics
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime


class ProbeCallBudgetExceeded(RuntimeError):
    """Raised before a model request would exceed the probe-wide call budget."""


def validate_paths(gate_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Resolve inputs and reject output paths that can touch the gate archive."""
    gate = Path(gate_dir).resolve()
    output = Path(output_dir).resolve()
    if not gate.is_dir() or not (gate / "gate_report.json").is_file():
        raise ValueError("gate directory must contain gate_report.json")
    if gate == output or gate in output.parents or output in gate.parents:
        raise ValueError("probe output must be separate from the gate directory")
    if output.exists():
        raise ValueError("probe output directory must not already exist")
    return gate, output


def _generation_brief(connection: sqlite3.Connection, question: Question) -> QuestionBrief | None:
    task_id = str((question.generation_provenance or {}).get("task_id") or "")
    if not task_id:
        return None
    try:
        row = connection.execute(
            "SELECT payload FROM generation_tasks WHERE id=?", (task_id,)
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        brief = QuestionBrief.model_validate(json.loads(row[0]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if brief.role != "writer" or brief.l3_id != question.l3_id:
        return None
    return brief


def load_gate_seed(gate_dir: Path) -> tuple[Question, QuestionBrief]:
    """Select a real Writer accepted question with its original stored brief."""
    gate = Path(gate_dir).resolve()
    report_path = gate / "gate_report.json"
    database = gate / "questions.sqlite3"
    if not report_path.is_file() or not database.is_file():
        raise ValueError("completed gate directory requires gate_report.json and questions.sqlite3")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("gate report is unreadable") from exc
    if report.get("fake") is not False:
        raise ValueError("role probes require a completed real-API gate")
    if report.get("done") is not True or int(report.get("accepted") or 0) < 1:
        raise ValueError("gate report must be done with at least one accepted question")

    wal_file = Path(str(database) + "-wal")
    if wal_file.is_file() and wal_file.stat().st_size:
        raise ValueError("gate archive must be closed and checkpointed before probing")
    uri = f"file:{quote(str(database), safe='/')}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ValueError("gate archive database is unreadable") from exc
    try:
        try:
            rows = connection.execute(
                "SELECT payload FROM questions WHERE status='accepted' ORDER BY rowid"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ValueError("gate archive has no readable accepted question table") from exc
        candidates: list[Question] = []
        for row in rows:
            try:
                question = Question.model_validate_json(row[0])
            except Exception:
                continue
            if (question.review_status == "accepted" and question.source == "writer"
                    and question.generation_provenance.get("writer_real_api") is True):
                candidates.append(question)
        candidates.sort(key=lambda question: question.id)
        for question in candidates:
            brief = _generation_brief(connection, question)
            if brief is not None:
                return question, brief
    finally:
        connection.close()
    raise ValueError("gate archive has no accepted real Writer question with its original brief")


def make_salvage_raw(question: Question, minimum_chars: int) -> str:
    """Build malformed JSON containing only the public stem and claim fields."""
    fields = {
        "question": question.question,
        "tests_claim": question.tests_claim,
        "language": question.language,
    }
    prefix = json.dumps(fields, ensure_ascii=False)
    target = max(1200, int(minimum_chars) + 100)
    padding = "x" * max(1, target - len(prefix) - 32)
    return prefix[:-1] + ',"probe_padding":"' + padding + '"'


def summarize_outcome(outcome: GenerateOutcome) -> dict[str, object]:
    """Return candidate status without serializing question or private fields."""
    status = str(outcome.status)
    allowed = {"ok", "retry", "repair", "salvage", "hard_fail"}
    safe_status = status if status in allowed else "unknown"
    return {
        "candidate_status": "pending_review" if safe_status == "ok" and outcome.question else safe_status,
        "candidate_created": outcome.question is not None,
        "reason_code": _safe_reason_code(outcome.reason),
    }


def _safe_reason_code(reason: str | None) -> str | None:
    allowed = {
        "429", "json_parse", "salvage_changed_question", "salvage_changed_field",
        "salvage_question_unverifiable", "review_repair_api", "writer_gate",
    }
    text = str(reason or "")
    for item in allowed:
        if text == item or text.startswith(item + ":"):
            return item
    return None


def summarize_salvage_safety(
    outcome: GenerateOutcome,
    *,
    model_output_stem_matches: bool | None,
    expected_stem: str,
) -> dict[str, object]:
    """Report whether salvage preserved the source stem or failed closed."""
    candidate_matches = (
        None if outcome.question is None
        else outcome.question.question.strip() == expected_stem.strip()
    )
    changed_output_rejected = (
        model_output_stem_matches is False
        and outcome.question is None
        and outcome.status == "retry"
    )
    safe = candidate_matches is None or (
        candidate_matches is True and model_output_stem_matches is True
    )
    return {
        **summarize_outcome(outcome),
        "model_output_stem_matches": model_output_stem_matches,
        "candidate_stem_matches": candidate_matches,
        "changed_stem_rejected": changed_output_rejected,
        "fidelity_guard_safe": safe,
    }


_ACTIVE_STAGE: ContextVar[str] = ContextVar("probe_agent_role_stage", default="setup")


class CallBudget:
    """Count logical LLM completions by stage/role and cap the total requests."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.total = 0
        self.blocked = 0
        self.calls: Counter[tuple[str, str]] = Counter()
        self.failures: Counter[tuple[str, str]] = Counter()
        self._patched: list[tuple[object, bool, object]] = []
        self.salvage_expected_stem: str | None = None
        self.salvage_output_stem_matches: bool | None = None

    @contextmanager
    def stage(self, name: str):
        token = _ACTIVE_STAGE.set(name)
        try:
            yield
        finally:
            _ACTIVE_STAGE.reset(token)

    def install(self, runtime) -> None:
        clients = [
            runtime.llm,
            runtime.evolver_llm,
            runtime.salvage_llm,
            runtime.knowledge_llm,
            runtime.review_llm,
        ]
        seen: set[int] = set()
        for client in clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            namespace = getattr(client, "__dict__", {})
            had_instance_method = "complete" in namespace
            previous = namespace.get("complete")
            original = client.complete

            def tracked(messages, *args, _original=original, **kwargs):
                stage = _ACTIVE_STAGE.get()
                role = current_agent_role() or "unspecified"
                key = (stage, role)
                if self.total >= self.limit:
                    self.blocked += 1
                    raise ProbeCallBudgetExceeded("probe LLM call budget exhausted")
                self.total += 1
                self.calls[key] += 1
                try:
                    result = _original(messages, *args, **kwargs)
                except Exception:
                    self.failures[key] += 1
                    raise
                if stage == "salvage" and self.salvage_output_stem_matches is None:
                    self._record_salvage_stem(result)
                return result

            client.complete = tracked
            self._patched.append((client, had_instance_method, previous))

    def _record_salvage_stem(self, response: str) -> None:
        from cuda_sft_qgen.jsonutil import parse_question_payload

        try:
            obj = parse_question_payload(response)
        except Exception:
            obj = None
        question = str((obj or {}).get("question") or "").strip()
        if not question or self.salvage_expected_stem is None:
            return
        self.salvage_output_stem_matches = question == self.salvage_expected_stem.strip()

    def restore(self) -> None:
        for client, had_instance_method, previous in reversed(self._patched):
            if had_instance_method:
                client.complete = previous
            else:
                delattr(client, "complete")
        self._patched.clear()

    def stage_calls(self, stage: str) -> dict[str, int]:
        return {
            role: count
            for (call_stage, role), count in sorted(self.calls.items())
            if call_stage == stage
        }

    def stage_failures(self, stage: str) -> dict[str, int]:
        return {
            role: count
            for (call_stage, role), count in sorted(self.failures.items())
            if call_stage == stage
        }


def _timed_stage(name: str, call, budget: CallBudget) -> dict[str, object]:
    started = time.monotonic()
    try:
        with budget.stage(name):
            result = call()
    except ProbeCallBudgetExceeded:
        result = {"status": "call_budget_exhausted"}
    except Exception as exc:  # Keep exception messages out of the private report.
        result = {"status": "error", "error_type": type(exc).__name__}
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    result["calls_by_agent_role"] = budget.stage_calls(name)
    result["failed_calls_by_agent_role"] = budget.stage_failures(name)
    result["agent_called"] = sum(result["calls_by_agent_role"].values()) > 0
    return result


def _evolver_stage(brief: QuestionBrief, source: Question) -> dict[str, object]:
    from cuda_sft_qgen.graph.nodes.generate_core import generate_from_brief

    probe_brief = QuestionBrief.model_validate({
        **brief.model_dump(),
        "task_id": "probe-evolver-" + uuid.uuid4().hex,
        "role": "evolver",
        "seed_id": source.id,
        "seed_question": source.question,
        "operator": brief.operator or "deepen",
    })
    try:
        generate_from_brief(probe_brief, SteeringBrief())
    except JsonSalvageNeeded:
        return {"status": "json_salvage_needed", "candidate_created": False}
    return {
        "status": "draft_created",
        "candidate_created": True,
        "candidate_status": "pending_review",
    }


def _salvage_stage(
    source: Question,
    brief: QuestionBrief,
    runtime,
    budget: CallBudget,
) -> dict[str, object]:
    from cuda_sft_qgen.graph.nodes import generate_core

    raw = make_salvage_raw(
        source,
        max(800, runtime.settings.question_min_chars * 8),
    )
    original_parser = generate_core.parse_question_payload
    injected = {"done": False}

    def one_parser_miss(text, **kwargs):
        if text == raw and not injected["done"]:
            injected["done"] = True
            return None
        return original_parser(text, **kwargs)

    budget.salvage_expected_stem = source.question
    previous_setting = runtime.settings.salvage_use_llm
    runtime.settings.salvage_use_llm = True
    generate_core.parse_question_payload = one_parser_miss
    try:
        outcome = generate_core.salvage_item({
            "brief": brief.model_dump(),
            "raw": raw,
            "writer_real_api": True,
        })
    finally:
        generate_core.parse_question_payload = original_parser
        runtime.settings.salvage_use_llm = previous_setting
    summary = summarize_salvage_safety(
        outcome,
        model_output_stem_matches=budget.salvage_output_stem_matches,
        expected_stem=source.question,
    )
    summary["parser_miss_injected"] = injected["done"]
    summary["natural_trigger_rate_evidence"] = False
    return summary


def summarize_knowledge_job(
    *,
    worker_ran: bool,
    job_state: str,
    job_error: str | None,
    before_jobs: dict[str, int],
    after_jobs: dict[str, int],
) -> dict[str, object]:
    """Distinguish a leased job from an audit that completed successfully."""
    if job_state == "done":
        status = "audit_job_completed"
    elif job_state == "failed":
        status = "audit_job_failed"
    elif job_error:
        status = "audit_job_requeued_after_error"
    else:
        status = "audit_job_not_completed"
    return {
        "status": status,
        "worker_job_ran": bool(worker_ran),
        "audit_succeeded": job_state == "done",
        "job_state": job_state,
        "job_error_recorded": bool(job_error),
        "completed_jobs_delta": int(after_jobs.get("done") or 0) - int(before_jobs.get("done") or 0),
        "failed_jobs_delta": int(after_jobs.get("failed") or 0) - int(before_jobs.get("failed") or 0),
    }


def _knowledge_stage(runtime, source: Question) -> dict[str, object]:
    from cuda_sft_qgen.knowledge.doc_index import DocIndex
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker
    from cuda_sft_qgen.store.knowledge import official_url

    knowledge = runtime.knowledge
    claim = knowledge.get_claim(source.l3_id) if knowledge is not None else None
    if claim is None:
        return {"status": "skipped_source_claim_missing", "worker_job_ran": False}

    worker = KnowledgeWorker(
        knowledge,
        runtime.knowledge_llm or runtime.llm,
        settings=runtime.settings,
        taxonomy=runtime.taxonomy,
        archive=runtime.store,
        official_fetcher=_disable_remote_fetch,
    )
    worker.doc_index = DocIndex([
        chunk for chunk in worker.doc_index.chunks if official_url(chunk.url)
    ])
    query = str(claim.get("claim") or "") + " " + " ".join(claim.get("hints") or [])
    local_hits = worker.doc_index.search(query, k=6)
    if not local_hits:
        return {
            "status": "skipped_no_local_official_doc_match",
            "worker_job_ran": False,
            "local_doc_matches": 0,
        }

    revision = int(claim.get("revision") or 1)
    job_id = knowledge.enqueue(
        "audit",
        {"claim_id": source.l3_id, "revision": revision},
        "probe-audit:" + uuid.uuid4().hex,
    )
    if job_id is None:
        return {"status": "skipped_audit_enqueue_failed", "worker_job_ran": False}
    before_jobs = knowledge.stats().get("jobs") or {}
    started = time.monotonic()
    worker_ran = worker.run_once()
    after_jobs = knowledge.stats().get("jobs") or {}
    with knowledge.connect() as db:
        job = db.execute("SELECT status,error FROM jobs WHERE id=?", (job_id,)).fetchone()
    current = knowledge.get_claim(source.l3_id) or {}
    job_state = str(job["status"]) if job is not None else "missing"
    return {
        **summarize_knowledge_job(
            worker_ran=worker_ran,
            job_state=job_state,
            job_error=str(job["error"] or "") if job is not None else None,
            before_jobs=before_jobs,
            after_jobs=after_jobs,
        ),
        "claim_status_before": str(claim.get("status") or "unknown"),
        "claim_status_after": str(current.get("status") or "unknown"),
        "local_doc_matches": len(local_hits),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def _disable_remote_fetch(_url: str) -> str:
    raise RuntimeError("remote fetch disabled")


def _review_repair_stage(runtime, source: Question, brief: QuestionBrief) -> dict[str, object]:
    from cuda_sft_qgen.graph.nodes.review_archive import enqueue_review_repairs
    from cuda_sft_qgen.graph.nodes.generate_core import review_repair_item

    if not runtime.settings.generation.get("review_repair_enabled", True):
        return {"status": "skipped_disabled", "handoff_created": False, "candidate_created": False}

    repair_brief = QuestionBrief.model_validate({
        **brief.model_dump(),
        "task_id": "probe-repair-source-" + uuid.uuid4().hex,
        "role": "writer",
    })
    task_id = runtime.store.save_generation_task(repair_brief.model_dump())
    synthetic_reject = source.model_copy(deep=True)
    synthetic_reject.generation_provenance = {
        **synthetic_reject.generation_provenance,
        "task_id": task_id,
        "review_repair_attempt": 0,
    }
    synthetic_reject.defect_codes = ["answerability"]
    decision = {
        "action": "reject",
        "question": synthetic_reject.model_dump(),
        "reason": "probe-only: generate a fresh candidate for the same brief",
    }
    queued = enqueue_review_repairs(runtime, [decision], SteeringBrief())
    tasks = runtime.store.unfinished_review_repairs()
    task = next((item for item in tasks if item.get("source_id") == source.id), None)
    if queued != 1 or task is None:
        return {"status": "handoff_not_created", "handoff_created": False, "candidate_created": False}
    outcome = review_repair_item(task)
    return {
        "status": "handoff_processed",
        "handoff_created": True,
        "synthetic_repairable_reject": True,
        **summarize_outcome(outcome),
    }


def _new_report(gate_dir: Path, source: Question, max_calls: int, salvage_enabled: bool) -> dict:
    return {
        "status": "running",
        "source": {
            "gate_dir": str(gate_dir),
            "real_writer_accepted_seed_loaded": True,
            "question_id": source.id,
        },
        "call_budget": {"max_calls": max_calls, "calls": 0, "blocked_attempts": 0},
        "roles": {
            "evolver": {"status": "not_run"},
            "salvage_llm": {
                "status": "not_requested" if not salvage_enabled else "not_run",
                "parser_miss_injected": False,
                "natural_trigger_rate_evidence": False,
            },
            "knowledge_worker": {"status": "not_run", "local_docs_only": True},
            "review_repair": {"status": "not_run"},
        },
        "salvage_source_review": {
            "finding": (
                "The optional LLM branch appears naturally unreachable: local parsing must fail "
                "while the same complete question field remains extractable."
            ),
            "probe_method": "One parser miss is injected for the synthetic malformed input only.",
            "natural_trigger_rate_claimed": False,
        },
    }


def _secure_output(output: Path) -> None:
    for path in sorted(output.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)
    output.chmod(0o700)


def _write_report(output: Path, report: dict) -> None:
    path = output / "agent_role_probe.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def probe_passed(roles: dict, *, salvage_requested: bool, blocked_attempts: int) -> bool:
    evolver = roles["evolver"]
    knowledge = roles["knowledge_worker"]
    repair = roles["review_repair"]
    salvage = roles["salvage_llm"]
    return (
        blocked_attempts == 0
        and evolver.get("status") == "draft_created" and evolver.get("agent_called") is True
        and knowledge.get("audit_succeeded") is True and knowledge.get("agent_called") is True
        and repair.get("handoff_created") is True and repair.get("agent_called") is True
        and repair.get("candidate_status") == "pending_review"
        and (not salvage_requested or (salvage.get("agent_called") is True
                                       and salvage.get("parser_miss_injected") is True
                                       and salvage.get("fidelity_guard_safe") is True))
    )


def run_probe(
    gate_dir: Path,
    output_dir: Path,
    *,
    profile: str = "dev",
    max_calls: int = 8,
    include_salvage_llm: bool = False,
) -> dict:
    """Run the targeted roles in a disposable runtime archive."""
    if not 1 <= int(max_calls) <= 12:
        raise ValueError("max_calls must be between 1 and 12")
    gate, output = validate_paths(gate_dir, output_dir)
    source, writer_brief = load_gate_seed(gate)
    settings = load_settings(fake_llm=False, profile=profile)
    missing = settings.missing_api_keys + settings.missing_api_fields
    if missing:
        raise ValueError("missing API configuration fields: " + ", ".join(missing))

    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    report = _new_report(gate, source, max_calls, include_salvage_llm)
    runtime = None
    runtime_token = None
    runtime_temp = None
    budget = CallBudget(max_calls)
    try:
        runtime_temp = tempfile.TemporaryDirectory(prefix=".runtime-", dir=output)
        with nullcontext(runtime_temp.name) as runtime_path:
            config = json.loads(json.dumps(settings.default_yaml))
            config.setdefault("paths", {})["taxonomy"] = str(gate / "taxonomy.yaml")
            settings.default_yaml = config
            settings.data_dir_override = Path(runtime_path)
            settings.generation["request_max_attempts"] = 1
            settings.generation["request_timeout_seconds"] = min(
                60, int(settings.generation.get("request_timeout_seconds", 90))
            )
            settings.generation["stream_timeout_seconds"] = min(
                120, int(settings.generation.get("stream_timeout_seconds", 180))
            )
            settings.llm_rate_limit_retries_per_call = 1
            runtime = build_runtime(settings)
            runtime_token = set_runtime(runtime)
            budget.install(runtime)
            get_metrics().reset()

            report["roles"]["evolver"] = _timed_stage(
                "evolver", lambda: _evolver_stage(writer_brief, source), budget
            )

            if include_salvage_llm and budget.total < budget.limit:
                report["roles"]["salvage_llm"] = _timed_stage(
                    "salvage", lambda: _salvage_stage(source, writer_brief, runtime, budget), budget
                )
            elif include_salvage_llm:
                report["roles"]["salvage_llm"] = {
                    "status": "call_budget_exhausted_before_stage",
                    "parser_miss_injected": False,
                    "natural_trigger_rate_evidence": False,
                    "duration_seconds": 0.0,
                    "calls_by_agent_role": {},
                    "agent_called": False,
                }

            report["roles"]["knowledge_worker"] = _timed_stage(
                "knowledge_worker", lambda: _knowledge_stage(runtime, source), budget
            )
            report["roles"]["review_repair"] = _timed_stage(
                "review_repair", lambda: _review_repair_stage(runtime, source, writer_brief), budget
            )
            report["status"] = "completed" if probe_passed(
                report["roles"], salvage_requested=include_salvage_llm,
                blocked_attempts=budget.blocked,
            ) else "partial"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
    finally:
        budget.restore()
        if runtime_token is not None:
            reset_runtime(runtime_token)
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                report.setdefault("cleanup_error_type", type(exc).__name__)
        if runtime_temp is not None:
            runtime_temp.cleanup()

    report["call_budget"]["calls"] = budget.total
    report["call_budget"]["blocked_attempts"] = budget.blocked
    _write_report(output, report)
    _secure_output(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument(
        "--include-salvage-llm",
        action="store_true",
        help="run the optional Salvage LLM path using a one-shot parser-miss injection",
    )
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        report = run_probe(
            args.gate_dir,
            args.output,
            profile=args.profile,
            max_calls=args.max_calls,
            include_salvage_llm=args.include_salvage_llm,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}, indent=2))
        return 2
    print(json.dumps({
        "status": report["status"],
        "report": str(args.output.resolve() / "agent_role_probe.json"),
        "llm_calls": report["call_budget"]["calls"],
        "roles": {
            name: {
                "status": details.get("status"),
                "agent_called": details.get("agent_called", False),
                "duration_seconds": details.get("duration_seconds", 0),
            }
            for name, details in report["roles"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
