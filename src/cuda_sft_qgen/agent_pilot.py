"""Reproducible real-provider A/B/C/D generation ablation pilot.

The pilot is intentionally isolated from the production archive.  It freezes a
taxonomy snapshot, builds one deterministic brief set, runs every selected
group against that same set, and checkpoints a privacy-safe report after every
candidate.  It never exports private targets, answers, rubrics, or reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from cuda_sft_qgen.config import Settings, load_settings
from cuda_sft_qgen.embeddings import FakeEmbedder
from cuda_sft_qgen.graph.nodes.generate_core import generate_from_brief
from cuda_sft_qgen.graph.nodes.review_archive import archive_node
from cuda_sft_qgen.graph.nodes.review_chair import _parallel_chair_decision, review_chair_node
from cuda_sft_qgen.graph.nodes.review_score import review_score_node
from cuda_sft_qgen.llm import FakeLLM, LLMClient
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.observability import get_metrics
from cuda_sft_qgen.planner.pair_mining import mine_pairs, mixup_payload
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.review_benchmark import load_quality_cases
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy, dump_taxonomy, load_taxonomy


DEFAULT_GROUPS = ("A", "B", "C", "D")
DEFAULT_KINDS = (
    "numeric", "formula", "api", "failure", "layout", "feature", "mechanism", "profiling",
)
RANDOM_SEED = 20260921
PILOT_REQUEST_TIMEOUT_SECONDS = 30
PILOT_REQUEST_MAX_ATTEMPTS = 1
_QUESTION_TYPES = {
    "numeric": "conceptual", "formula": "conceptual", "api": "syntax",
    "failure": "debug", "layout": "layout", "feature": "compare",
    "mechanism": "conceptual", "profiling": "profiling",
}


class CallBudgetExceeded(RuntimeError):
    """Raised before a real logical LLM call would exceed the pilot budget."""


class PilotDeadlineExceeded(RuntimeError):
    """Raised between calls when the pilot wall-clock deadline has elapsed."""


class GroupCallBudgetExceeded(CallBudgetExceeded):
    """Raised when one ablation group has consumed its equal budget share."""


class _CallBudget:
    def __init__(self, maximum: int, deadline: float, groups: Sequence[str]) -> None:
        self.maximum = max(1, int(maximum))
        self.deadline = deadline
        self.count = 0
        self.by_channel: Counter[str] = Counter()
        self.by_group: Counter[str] = Counter()
        base, remainder = divmod(self.maximum, max(1, len(groups)))
        self.group_limits = {
            group: base + (1 if index < remainder else 0)
            for index, group in enumerate(groups)
        }
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def start(self, channel: str) -> tuple[int, float]:
        with self._lock:
            if time.monotonic() >= self.deadline:
                raise PilotDeadlineExceeded("pilot deadline reached")
            if self.count >= self.maximum:
                raise CallBudgetExceeded(f"logical LLM call budget exhausted ({self.maximum})")
            group = channel.split(":", 1)[0]
            if self.by_group[group] >= self.group_limits.get(group, self.maximum):
                raise GroupCallBudgetExceeded(
                    f"group {group} logical call budget exhausted ({self.group_limits.get(group)})"
                )
            self.count += 1
            self.by_channel[channel] += 1
            self.by_group[group] += 1
            return self.count, time.monotonic()

    def group_exhausted(self, group: str) -> bool:
        return self.by_group[group] >= self.group_limits.get(group, self.maximum)

    def finish(self, call_id: int, channel: str, started: float, error: BaseException | None) -> None:
        with self._lock:
            self.events.append({
                "call": call_id,
                "channel": channel,
                "latency_seconds": round(time.monotonic() - started, 6),
                "status": "error" if error else "ok",
                "error_type": type(error).__name__ if error else None,
            })


class _BudgetedLLM(LLMClient):
    """Count calls at the lowest logical ``complete`` boundary."""

    def __init__(self, delegate: Any, budget: _CallBudget, channel: str) -> None:
        self.delegate = delegate
        self.budget = budget
        self.channel = channel

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        call_id, started = self.budget.start(self.channel)
        error: BaseException | None = None
        try:
            return self.delegate.complete(messages, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self.budget.finish(call_id, self.channel, started, error)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_checkpoint(path: Path, report: dict[str, Any]) -> None:
    # Every incremental checkpoint carries a machine-readable privacy audit.
    # This matters for interrupted real-provider runs: a partial report should
    # have the same privacy guarantees as the final report, rather than relying
    # on a post-processing step that may never execute.
    if report.get("pilot") == "agent_ablation_abcd":
        report["privacy"] = _privacy_audit(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _privacy_audit(report: dict[str, Any]) -> dict[str, Any]:
    """Scan the public checkpoint without copying sensitive values into it.

    The audit checks both private internal field names and configured provider
    credentials.  It reports counts only: recording the matched value or field
    path would itself make the validation artifact sensitive.
    """
    public = {key: value for key, value in report.items() if key != "privacy"}
    forbidden_keys = {
        "internal_reference_answer", "draft_target", "verification_spec",
        "votes", "messages", "reasoning_content",
    }
    private_key_matches = 0

    def visit(value: Any) -> None:
        nonlocal private_key_matches
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    private_key_matches += 1
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(public)
    serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
    configured_secrets: list[str] = []
    for name, value in os.environ.items():
        if name.endswith(("_API_KEY", "_API_KEYS")) or "_API_KEY_" in name:
            configured_secrets.extend(part.strip() for part in value.replace(";", ",").split(",")
                                      if len(part.strip()) >= 8)
    secret_matches = sum(secret in serialized for secret in set(configured_secrets))
    passed = private_key_matches == 0 and secret_matches == 0
    return {
        "status": "pass" if passed else "fail",
        "policy_version": 1,
        "private_payload_key_matches": private_key_matches,
        "configured_secret_matches": secret_matches,
        "private_reasoning_persisted": False,
        "full_provider_requests_persisted": False,
    }


def _group_flags(group: str) -> dict[str, bool]:
    name = group.upper()
    if name not in DEFAULT_GROUPS:
        raise ValueError(f"unknown pilot group: {group}")
    return {
        "prompt_cot": name != "A",
        "answer_consistency": name in {"C", "D"},
        "mixup": name == "D",
        "legacy_output": name == "A",
    }


def _active_leaves(taxonomy: Taxonomy) -> list[Leaf]:
    return [
        leaf for leaf in taxonomy.leaves
        if not leaf.retired and str(getattr(leaf, "status", "") or "").lower()
        not in {"retired", "quarantined"}
    ]


def _leaf_for_kind(leaves: Sequence[Leaf], kind: str, rng: random.Random, used: set[str]) -> Leaf:
    matches = [
        leaf for leaf in leaves
        if leaf.l3_id not in used and (leaf.kind == kind or kind in leaf.types)
    ]
    if not matches:
        matches = [leaf for leaf in leaves if leaf.l3_id not in used]
    if not matches:
        matches = list(leaves)
    if not matches:
        raise ValueError("taxonomy has no active leaves")
    matches = sorted(matches, key=lambda leaf: (leaf.domain_id, leaf.l2_id, leaf.l3_id))
    return matches[rng.randrange(len(matches))]


def _build_briefs(
    taxonomy: Taxonomy,
    personas: dict[str, Any],
    *,
    samples: int,
    kinds: Sequence[str],
    random_seed: int,
) -> list[QuestionBrief]:
    """Build ``samples`` briefs per kind, shared byte-for-byte by all groups."""
    if samples < 1:
        raise ValueError("samples must be positive")
    requested = tuple(str(kind).strip().lower() for kind in kinds if str(kind).strip())
    if not requested:
        raise ValueError("at least one kind is required")
    unknown = sorted(set(requested) - set(DEFAULT_KINDS))
    if unknown:
        raise ValueError(f"unsupported kinds: {unknown}")
    rng = random.Random(random_seed)
    leaves = _active_leaves(taxonomy)
    persona = sorted(personas)[0] if personas else "cuda_interviewer"
    used: set[str] = set()
    briefs: list[QuestionBrief] = []
    for kind in requested:
        leaf: Leaf | None = None
        for sample_index in range(samples):
            if sample_index % 2 == 0 or leaf is None:
                leaf = _leaf_for_kind(leaves, kind, rng, used)
                used.add(leaf.l3_id)
            native_types = [value for value in leaf.types if value in {
                "conceptual", "syntax", "code_complete", "debug", "optimization",
                "layout", "compare", "counterfactual", "profiling",
            }]
            question_type = native_types[0] if native_types else _QUESTION_TYPES[kind]
            low = min(leaf.difficulty or (1,))
            high = max(leaf.difficulty or (3,))
            if high <= low:
                high = min(4, low + 2)
            if high <= low:
                low = max(1, high - 1)
            level = "low" if sample_index % 2 == 0 else "high"
            difficulty = low if level == "low" else high
            guidance = (
                "Use one direct CUDA constraint and one observable."
                if level == "low"
                else "Use multiple dependent CUDA constraints and require at least two reasoning steps."
            )
            family = f"pilot-{random_seed}-{kind}-{sample_index // 2:03d}-{leaf.l3_id}"
            task_id = hashlib.sha256(f"{family}:{level}".encode()).hexdigest()[:20]
            briefs.append(QuestionBrief(
                task_id=task_id,
                topic_path=list(leaf.path),
                l3_id=leaf.l3_id,
                l3_claim=leaf.claim,
                question_type=question_type,
                difficulty=int(difficulty),
                bloom="apply" if level == "low" else "analyze",
                persona=persona,
                hardware="agnostic",
                granularity="api" if kind == "api" else "kernel",
                language="zh",
                must_include=f"PILOT_FAMILY={family}; {guidance}",
                instance_hint=(leaf.hints[0] if leaf.hints else leaf.claim[:160]),
                angle=f"{kind} {level} difficulty controlled question",
                angle_id=f"pilot-{kind}-{level}",
                kind=kind,
                claim_revision=leaf.revision,
                source_ids=list(leaf.sources),
                breadth_or_depth="BREADTH" if level == "low" else "DEPTH",
            ))
    return briefs


def _group_settings(settings: Settings, root: Path, taxonomy_path: Path, group: str) -> Settings:
    flags = _group_flags(group)
    config = deepcopy(settings.default_yaml)
    config.setdefault("paths", {})["taxonomy"] = str(taxonomy_path)
    config.setdefault("generation", {})["prompt_cot"] = flags["prompt_cot"]
    config.setdefault("generation", {})["concurrent_pipeline"] = False
    # The real pilot is intentionally stricter than normal production runs.
    # Provider/socket timeouts prevent one stalled stream from consuming the
    # whole experiment, and a single transport attempt keeps groups comparable.
    config.setdefault("generation", {})["request_timeout_seconds"] = PILOT_REQUEST_TIMEOUT_SECONDS
    config.setdefault("generation", {})["request_max_attempts"] = PILOT_REQUEST_MAX_ATTEMPTS
    config.setdefault("review", {})["answer_consistency_enabled"] = flags["answer_consistency"]
    config.setdefault("review", {})["parallel_enabled"] = False
    config.setdefault("taxonomy", {}).setdefault("manager", {})["enabled"] = False
    config.setdefault("knowledge", {})["enabled"] = False
    return replace(
        settings,
        fake_llm=False,
        default_yaml=config,
        data_dir_override=root / f"group-{group.lower()}",
        embedding_cache_enabled=False,
    )


def _public_question(question: Question) -> dict[str, Any]:
    return {
        key: getattr(question, key)
        for key in (
            "id", "question", "tests_claim", "topic_path", "l3_id", "question_type",
            "difficulty", "bloom", "hardware", "granularity", "language", "kind",
            "mixup_mode", "parent_ids", "parent_difficulties", "difficulty_target",
        )
    }


def _public_reason(value: Any) -> str | None:
    """Keep a short debug reason while redacting answer-like payloads."""
    text = " ".join(str(value or "").split())
    if not text:
        return None
    for marker in ("internal_reference_answer", "draft_target", "reference answer"):
        text = text.replace(marker, "[private]")
        text = text.replace(marker.title(), "[private]")
    # Reviewer reasons should explain the defect, not reproduce an oracle.
    text = re.sub(r"\\boxed\s*\{[^{}]*\}", "[answer-redacted]", text)
    text = re.sub(
        r"(?i)\b(?:correct|expected)\s+answer\s*(?:is|=|:)\s*[^.;。；]{1,160}",
        "answer=[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:answer|答案)\s*[:：=]\s*[^.;。；]{1,160}",
        "answer=[redacted]",
        text,
    )
    text = re.sub(
        r"(?:正确|预期)答案\s*(?:是|为|=|:|：)\s*[^.;。；]{1,160}",
        "答案=[已脱敏]",
        text,
    )
    return text[:400]


def _reason_code(value: Any) -> str | None:
    """Convert verifier reasons to bounded codes, never prose or answers."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    tokens = re.findall(r"[a-z0-9_:-]+", text)
    return "_".join(tokens[:8])[:120] or None


def _public_decision(item: dict[str, Any]) -> dict[str, Any]:
    review = item.get("review") or {}
    verification = review.get("verification_evidence") or {}
    evidence = verification.get("evidence") or {}
    # Backward compatibility for checkpoints produced before the verifier
    # wrapped evidence under ``verification_evidence.evidence``.
    answer = evidence.get("answer_consistency") or verification.get("answer_consistency") or {}
    compile_check = evidence.get("compile") or verification.get("compile") or {}
    votes = evidence.get("votes") or answer.get("votes") or []
    valid_votes = evidence.get("valid_votes", answer.get("valid_votes"))
    if valid_votes is None and isinstance(votes, list):
        valid_votes = len(votes)
    tool = compile_check.get("tool") or evidence.get("tool")
    returncode = compile_check.get("returncode", evidence.get("returncode"))
    tool_reason = compile_check.get("reason") or evidence.get("tool_reason")
    action = str(item.get("action", "pending"))
    checks = dict(review.get("checks") or {})
    verifier_reason = _reason_code(verification.get("reason") or answer.get("reason"))
    if verifier_reason:
        decision_reason_code = verifier_reason
    elif action == "accept":
        decision_reason_code = "reviewer_checks_passed"
    elif action == "reject":
        decision_reason_code = "reviewer_checks_failed"
    elif action == "not_evaluated":
        decision_reason_code = "not_evaluated"
    else:
        decision_reason_code = "reviewer_pending"
    # Never persist a reviewer's free-form reason. It frequently restates the
    # private oracle (for example, "the correct answer is ...") even when the
    # verifier evidence itself has been carefully reduced to safe metadata.
    defect_tags = sorted(
        f"{name}:{status}" for name, status in checks.items()
        if str(status).lower() not in {"pass", "passed", "ok"}
    )
    return {
        "action": action,
        "decision_reason_code": decision_reason_code,
        "defect_tags": defect_tags,
        "checks": checks,
        "evidence_summary": {
            "status": verification.get("status") or answer.get("status"),
            "reason_code": _reason_code(verification.get("reason") or answer.get("reason")),
            "agreement": evidence.get("agreement", answer.get("agreement")),
            "valid_votes": valid_votes,
            "rip_score": evidence.get("rip_score"),
            "tool": tool,
            "returncode": returncode,
            "tool_status": compile_check.get("status"),
            "tool_reason_code": _reason_code(tool_reason),
            "review_evidence_count": len(review.get("evidence") or []),
        },
    }


def sanitize_pilot_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove legacy free-text review reasons from an in-memory checkpoint.

    Incremental reports from an interrupted process may have been created by a
    previously imported version of this module.  This idempotent sanitizer is
    therefore also useful when publishing an already completed real run.
    """
    for group in (report.get("groups") or {}).values():
        for collection in ("candidates", "quality_cases"):
            for row in group.get(collection) or []:
                row.pop("reason", None)
                checks = dict(row.get("checks") or {})
                row.setdefault("defect_tags", sorted(
                    f"{name}:{status}" for name, status in checks.items()
                    if str(status).lower() not in {"pass", "passed", "ok"}
                ))
                if not row.get("decision_reason_code"):
                    evidence_code = (row.get("evidence_summary") or {}).get("reason_code")
                    action = str(row.get("action", "pending"))
                    row["decision_reason_code"] = evidence_code or {
                        "accept": "reviewer_checks_passed",
                        "reject": "reviewer_checks_failed",
                        "not_evaluated": "not_evaluated",
                    }.get(action, "reviewer_pending")
    return report


def _metrics_usage(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    counters_before = before.get("counters") or {}
    counters_after = after.get("counters") or {}
    names = ("llm_prompt_tokens", "llm_completion_tokens", "llm_total_tokens", "llm_usage_reports")
    values: dict[str, int] = {}
    for name in names:
        values[name] = sum(
            int(value) for key, value in counters_after.items() if key.startswith(name)
        ) - sum(int(value) for key, value in counters_before.items() if key.startswith(name))
    reports = values.pop("llm_usage_reports")
    embedding_requests = sum(
        int(value) for key, value in counters_after.items() if key.startswith("embedding_requests")
    ) - sum(
        int(value) for key, value in counters_before.items() if key.startswith("embedding_requests")
    )
    embedding_tokens = sum(
        int(value) for key, value in counters_after.items() if key.startswith("embedding_tokens")
    ) - sum(
        int(value) for key, value in counters_before.items() if key.startswith("embedding_tokens")
    )
    return {
        "available": reports > 0,
        "provider_reports": reports,
        "prompt_tokens": values["llm_prompt_tokens"] if reports else None,
        "completion_tokens": values["llm_completion_tokens"] if reports else None,
        "total_tokens": values["llm_total_tokens"] if reports else None,
        "embedding_requests": embedding_requests,
        "embedding_tokens": embedding_tokens if embedding_tokens else None,
        "estimated_cost_usd": None,
        "cost_note": "No provider pricing is inferred by this pilot.",
    }


def _find_decision(chair_result: dict[str, Any], question_id: str) -> dict[str, Any]:
    payload = chair_result.get("last_diversity") or {}
    for decision in payload.get("final_decisions") or []:
        if decision.get("id") == question_id:
            return decision
    return {"id": question_id, "action": "pending", "reason": "missing_decision", "review": {}}


def _run_candidate(runtime: Any, brief: QuestionBrief, *, legacy_output: bool, iteration: int) -> tuple[dict[str, Any], Question]:
    started = time.monotonic()
    stage = "generate"
    try:
        question = generate_from_brief(brief, SteeringBrief())
        original_validation = list(question.draft_validation_errors)
        if legacy_output:
            # Group A intentionally measures the pre-structured writer contract.
            # The Chair must still inspect its public question instead of being
            # short-circuited by a gate that did not exist in that baseline.
            question.draft_validation_errors = []
        stage = "score"
        scored = review_score_node({"pending": [question.model_dump()]})
        stage = "chair"
        chaired = review_chair_node({
            "last_diversity": scored.get("last_diversity") or {},
            "review_rounds": iteration,
            "steering_brief": {},
        })
        prearchive = _find_decision(chaired, question.id)
        model_action = str((prearchive.get("question", {}).get("review_details") or {}).get("model_action") or prearchive.get("action", "pending"))
        consistency_enabled = bool(runtime.settings.review.get("answer_consistency_enabled", False))
        if not legacy_output and not consistency_enabled and model_action == "accept":
            # Group B intentionally isolates structured CoT from verification.
            # Its model verdict is useful ablation evidence, but it is not safe
            # to represent an unverified accept as production-admissible data.
            prearchive["action"] = "pending"
            prearchive["reason"] = "pilot_unverified_model_accept"
            payload = chaired.get("last_diversity") or {}
            payload["final_decisions"] = [
                prearchive if row.get("id") == question.id else row
                for row in payload.get("final_decisions") or []
            ]
        stage = "archive"
        archive_node({
            "last_diversity": chaired.get("last_diversity") or {},
            "steering_brief": chaired.get("steering_brief") or {},
            "iteration": iteration,
            "review_rounds": chaired.get("review_rounds", iteration + 1),
            "target": 10**9,
        })
        # Archive may downgrade an apparent accept when claim revisions or
        # verification evidence changed. Read the mutated authoritative batch.
        decision = _find_decision(chaired, question.id)
        decided_question = Question.model_validate(decision.get("question") or question.model_dump())
        record = {
            "brief_id": brief.task_id,
            "kind": brief.kind,
            "public_question": _public_question(decided_question),
            **_public_decision({**decision, "review": decided_question.review_details}),
            "model_action": model_action,
            "legacy_output_mode": legacy_output,
            "draft_contract": {
                "valid": not original_validation,
                "error_codes": original_validation,
            },
            "latency_seconds": round(time.monotonic() - started, 6),
            "failure": None,
        }
        return record, decided_question
    except Exception as exc:  # noqa: BLE001
        record = {
            "brief_id": brief.task_id,
            "kind": brief.kind,
            "public_question": None,
            "action": "not_evaluated",
            "checks": {},
            "evidence_summary": {},
            "legacy_output_mode": legacy_output,
            "draft_contract": None,
            "latency_seconds": round(time.monotonic() - started, 6),
            "failure": {"stage": stage, "error_type": type(exc).__name__},
        }
        return record, Question.model_construct(id="", question="", tests_claim="", topic_path=[])


def _mixup_briefs(runtime: Any, accepted: Sequence[Question], limit: int) -> list[QuestionBrief]:
    pairs = mine_pairs(
        accepted,
        embedder=runtime.embedder,
        min_similarity=0.70,
        max_similarity=1.0,
        top_k=8,
        max_pairs=max(1, limit),
        max_pairs_per_item=1,
        min_difficulty_gap=1,
        same_l2=True,
        same_language=True,
        same_granularity=True,
        active_claims={leaf.l3_id: leaf for leaf in _active_leaves(runtime.taxonomy)},
    )
    briefs: list[QuestionBrief] = []
    for pair in pairs:
        for mode in ("decomposed", "hybrid"):
            payload = mixup_payload(pair, mode)
            parent = pair.high
            leaf = runtime.taxonomy.leaf_by_key(parent.l3_id)
            if leaf is None:
                continue
            task_id = hashlib.sha256(f"pilot:{payload['pair_id']}:{mode}".encode()).hexdigest()[:20]
            briefs.append(QuestionBrief(
                task_id=task_id,
                topic_path=list(leaf.path),
                l3_id=leaf.l3_id,
                l3_claim=leaf.claim,
                question_type=parent.question_type,
                difficulty=int(payload["difficulty_target"]),
                bloom="analyze",
                persona=parent.persona,
                hardware=parent.hardware,
                granularity=parent.granularity,
                language=parent.language,
                must_include=payload["difficulty_policy"],
                angle=f"pilot {mode} from independently accepted parents",
                angle_id=f"pilot-mixup-{mode}",
                kind=parent.kind or leaf.kind or "mechanism",
                role="evolver",
                seed_id=payload["pair_id"],
                seed_question=(
                    f"PARENT A:\n{pair.low.question}\n\nPARENT B:\n{pair.high.question}"
                ),
                claim_revision=leaf.revision,
                source_ids=list(leaf.sources),
                knowledge_context={"mixup": payload},
                mixup_mode=mode,
                parent_ids=list(payload["parent_ids"]),
                parent_difficulties=list(payload["parent_difficulties"]),
                difficulty_target=int(payload["difficulty_target"]),
                difficulty_raw_score=float(payload["difficulty_raw_score"]),
            ))
    return briefs[: max(0, limit * 2)]


def _aggregate_group(group: dict[str, Any]) -> dict[str, Any]:
    candidates = group.get("candidates") or []
    actions = Counter(item.get("action", "not_evaluated") for item in candidates)
    completed = sum(item.get("public_question") is not None for item in candidates)
    leaves = {
        item["public_question"]["l3_id"]
        for item in candidates if item.get("public_question")
    }
    duplicate = sum(
        (item.get("checks") or {}).get("duplication") == "fail" for item in candidates
    )
    latencies = [float(item.get("latency_seconds") or 0) for item in candidates]
    answer_statuses = Counter(
        (item.get("evidence_summary") or {}).get("status") or "not_run"
        for item in candidates
    )
    compile_statuses = Counter(
        (item.get("evidence_summary") or {}).get("tool_status") or "not_run"
        for item in candidates
    )
    frozen = group.get("quality_cases") or []
    frozen_actions = Counter(item.get("action", "pending") for item in frozen)
    frozen_misses = sum(
        item.get("expected_action") == "reject" and item.get("action") == "accept"
        for item in frozen
    )
    frozen_false_rejects = sum(
        item.get("expected_action") == "accept" and item.get("action") == "reject"
        for item in frozen
    )
    quality_gate = len(frozen) >= 10 and completed >= 30
    return {
        "planned": len(candidates),
        "completed": completed,
        "actions": dict(actions),
        "operational_accept_rate": actions.get("accept", 0) / completed if completed else None,
        "accept_rate_quality_warning": "Acceptance rate is an operational outcome, not evidence of generated quality.",
        "covered_leaves": len(leaves),
        "duplicate_failures": duplicate,
        "failures": sum(item.get("failure") is not None for item in candidates),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "evidence_summary": {
            "answer_consistency": dict(answer_statuses),
            "compile": dict(compile_statuses),
        },
        "frozen_quality_summary": {
            "completed": len(frozen),
            "actions": dict(frozen_actions),
            "misses": frozen_misses if frozen else None,
            "false_rejections": frozen_false_rejects if frozen else None,
        },
        "quality_claim_status": "qualified" if quality_gate else "not_qualified",
        "quality_gates": {
            "qualified": quality_gate,
            "minimum_generated_completed": 30,
            "minimum_frozen_quality_cases": 10,
            "generated_completed": completed,
            "frozen_quality_cases": len(frozen),
        },
    }


def _run_quality_cases(
    runtime: Any,
    cases: Sequence[dict[str, Any]],
    deadline: float,
    budget: _CallBudget,
    group: str,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    prompt = load_prompt("chair")
    results: list[dict[str, Any]] = []
    for case in cases:
        if (
            time.monotonic() >= deadline
            or budget.count >= budget.maximum
            or budget.group_exhausted(group)
        ):
            break
        question = Question.model_validate(case["question"])
        question.draft_validation_errors = []
        started = time.monotonic()
        decision = _parallel_chair_decision(
            {"question": question.model_dump(), "decision": {"verdict": "unique"}, "neighbors": []},
            prompt,
        )
        results.append({
            "case_id": case["case_id"],
            "expected_action": case["expected_action"],
            "public_question": _public_question(question),
            **_public_decision(decision),
            "latency_seconds": round(time.monotonic() - started, 6),
        })
        if checkpoint is not None:
            checkpoint(results)
    return results


def run_agent_pilot(
    settings: Settings,
    output: Path | str,
    samples: int = 2,
    max_calls: int = 120,
    groups: Sequence[str] = DEFAULT_GROUPS,
    deadline_seconds: float = 1800,
    *,
    kinds: Sequence[str] = DEFAULT_KINDS,
    random_seed: int = RANDOM_SEED,
    quality_cases_path: Path | str | None = None,
    quality_case_limit: int | None = None,
) -> dict[str, Any]:
    """Run an isolated real-provider A/B/C/D ablation and checkpoint results."""
    if settings.fake_llm:
        raise ValueError("agent pilot requires a real LLM; fake_llm must be false")
    if str(settings.embedding_provider).lower() == "fake":
        raise ValueError("agent pilot requires a real embedding provider")
    selected_groups = tuple(str(group).upper() for group in groups)
    if not selected_groups or len(set(selected_groups)) != len(selected_groups):
        raise ValueError("groups must be nonempty and unique")
    for group in selected_groups:
        _group_flags(group)
    output = Path(output).resolve()
    started = time.monotonic()
    deadline = started + max(1.0, float(deadline_seconds))
    budget = _CallBudget(max_calls, deadline, selected_groups)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    briefs = _build_briefs(
        taxonomy, settings.personas, samples=samples, kinds=kinds, random_seed=random_seed
    )
    public_briefs = [{
        "task_id": brief.task_id, "topic_path": brief.topic_path, "l3_id": brief.l3_id,
        "question_type": brief.question_type, "difficulty": brief.difficulty,
        "bloom": brief.bloom, "hardware": brief.hardware, "granularity": brief.granularity,
        "language": brief.language, "kind": brief.kind,
    } for brief in briefs]
    report: dict[str, Any] = {
        "schema_version": 1,
        "pilot": "agent_ablation_abcd",
        "mode": "real_llm_real_embedding",
        "started_at": _utcnow(),
        "finished_at": None,
        "status": "running",
        "random_seed": random_seed,
        "deadline_seconds": deadline_seconds,
        "max_calls": max_calls,
        "samples": samples,
        "kinds": list(kinds),
        "groups_requested": list(selected_groups),
        "briefs": public_briefs,
        "brief_set_sha256": _stable_hash(public_briefs),
        "taxonomy": {},
        "config": {
            "writer_model": settings.llm_model,
            "review_model": settings.review_model,
            "embedding_model": settings.embedding_model,
            "request_timeout_seconds": PILOT_REQUEST_TIMEOUT_SECONDS,
            "request_max_attempts": PILOT_REQUEST_MAX_ATTEMPTS,
            "group_flags": {group: _group_flags(group) for group in selected_groups},
        },
        "config_sha256": "",
        "groups": {},
        "calls": {},
        "usage": {},
        "cost_usd": None,
        "cost_note": "No rate table supplied; cost remains null.",
    }
    report["config_sha256"] = _stable_hash(report["config"])
    _atomic_checkpoint(output, report)
    before_usage = get_metrics().snapshot()
    quality_cases: list[dict[str, Any]] = []
    if quality_cases_path is not None:
        quality_cases = load_quality_cases(Path(quality_cases_path))
        if quality_case_limit is not None:
            quality_cases = quality_cases[: max(0, int(quality_case_limit))]

    stop_reason: str | None = None
    partial_reasons: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cuda-sft-agent-pilot-", dir=output.parent) as directory:
        isolated_root = Path(directory)
        snapshot = isolated_root / "taxonomy.yaml"
        frozen = deepcopy(taxonomy)
        frozen.domain_files = {}
        dump_taxonomy(frozen, snapshot)
        report["taxonomy"] = {
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "active_leaves": len(_active_leaves(frozen)),
        }
        _atomic_checkpoint(output, report)
        for group_index, group_name in enumerate(selected_groups):
            if time.monotonic() >= deadline or budget.count >= budget.maximum:
                stop_reason = "deadline" if time.monotonic() >= deadline else "call_budget"
                break
            fixed = _group_settings(settings, isolated_root, snapshot, group_name)
            group_usage_before = get_metrics().snapshot()
            runtime = build_runtime(fixed)
            if isinstance(runtime.llm, FakeLLM) or isinstance(runtime.embedder, FakeEmbedder):
                runtime.close()
                raise ValueError("agent pilot runtime resolved to a fake provider")
            writer_delegate = runtime.llm
            review_delegate = runtime.review_llm or writer_delegate
            runtime.llm = _BudgetedLLM(writer_delegate, budget, f"{group_name}:writer")
            runtime.review_llm = _BudgetedLLM(review_delegate, budget, f"{group_name}:review")
            if getattr(runtime, "evolver_llm", None) is not None:
                runtime.evolver_llm = _BudgetedLLM(runtime.evolver_llm, budget, f"{group_name}:evolver")
            if getattr(runtime, "salvage_llm", None) is not None:
                runtime.salvage_llm = _BudgetedLLM(runtime.salvage_llm, budget, f"{group_name}:salvage")
            token = set_runtime(runtime)
            group_report: dict[str, Any] = {
                "group": group_name,
                "flags": _group_flags(group_name),
                "stage_status": "running",
                "candidates": [],
                "bootstrap": {
                    "scope": "isolated_per_group_same_frozen_briefs",
                    "cross_group_parent_import": False,
                    "parent_pool": "this_group_independently_reviewed_accepted_only",
                },
                "mixup": {"status": "not_applicable" if group_name != "D" else "bootstrap_pending"},
                "quality_cases": [],
                "summary": {},
            }
            report["groups"][group_name] = group_report
            group_partial_reason: str | None = None
            try:
                for index, source_brief in enumerate(briefs):
                    if time.monotonic() >= deadline or budget.count >= budget.maximum:
                        stop_reason = "deadline" if time.monotonic() >= deadline else "call_budget"
                        break
                    if budget.group_exhausted(group_name):
                        group_partial_reason = "group_call_budget"
                        break
                    record, _ = _run_candidate(
                        runtime,
                        source_brief.model_copy(deep=True),
                        legacy_output=group_name == "A",
                        iteration=index,
                    )
                    group_report["candidates"].append(record)
                    group_report["summary"] = _aggregate_group(group_report)
                    if record.get("failure"):
                        group_partial_reason = group_partial_reason or "candidate_failure"
                    report["calls"] = {
                        "logical_total": budget.count,
                        "by_channel": dict(budget.by_channel),
                        "by_group": dict(budget.by_group),
                        "group_limits": dict(budget.group_limits),
                        "maximum": budget.maximum,
                    }
                    _atomic_checkpoint(output, report)
                if group_name == "D" and stop_reason is None and not budget.group_exhausted(group_name):
                    accepted = runtime.store.load_accepted()
                    pair_briefs = _mixup_briefs(runtime, accepted, max(1, samples))
                    if not pair_briefs:
                        group_report["mixup"] = {
                            "status": "not_evaluated",
                            "reason": "insufficient_independently_accepted_compatible_parents",
                            "accepted_bootstrap_parents": len(accepted),
                        }
                        group_partial_reason = group_partial_reason or "mixup_not_evaluated"
                    else:
                        group_report["mixup"] = {
                            "status": "evaluated", "briefs": len(pair_briefs),
                            "accepted_bootstrap_parents": len(accepted),
                        }
                        for offset, brief in enumerate(pair_briefs, start=len(briefs)):
                            if time.monotonic() >= deadline or budget.count >= budget.maximum:
                                stop_reason = "deadline" if time.monotonic() >= deadline else "call_budget"
                                break
                            if budget.group_exhausted(group_name):
                                group_partial_reason = "group_call_budget"
                                break
                            record, _ = _run_candidate(
                                runtime, brief, legacy_output=False, iteration=offset
                            )
                            group_report["candidates"].append(record)
                            if record.get("failure"):
                                group_partial_reason = group_partial_reason or "candidate_failure"
                            group_report["summary"] = _aggregate_group(group_report)
                            _atomic_checkpoint(output, report)
                if quality_cases and stop_reason is None and not budget.group_exhausted(group_name):
                    def _quality_checkpoint(rows: list[dict[str, Any]]) -> None:
                        group_report["quality_cases"] = list(rows)
                        group_report["summary"] = _aggregate_group(group_report)
                        _atomic_checkpoint(output, report)

                    group_report["quality_cases"] = _run_quality_cases(
                        runtime, quality_cases, deadline, budget, group_name,
                        checkpoint=_quality_checkpoint,
                    )
                    if budget.count >= budget.maximum:
                        stop_reason = "call_budget"
                    elif budget.group_exhausted(group_name) and len(group_report["quality_cases"]) < len(quality_cases):
                        group_partial_reason = "group_call_budget"
                    elif time.monotonic() >= deadline:
                        stop_reason = "deadline"
                group_report["summary"] = _aggregate_group(group_report)
                base_ids = {brief.task_id for brief in briefs}
                completed_base = sum(
                    item.get("brief_id") in base_ids for item in group_report["candidates"]
                )
                if completed_base < len(briefs):
                    group_partial_reason = group_partial_reason or "incomplete_base_briefs"
                if group_name == "D" and group_report["mixup"].get("status") != "evaluated":
                    group_partial_reason = group_partial_reason or "mixup_not_evaluated"
                if group_partial_reason:
                    group_report["stage_status"] = "partial"
                    group_report["partial_reason"] = group_partial_reason
                    partial_reasons.append(f"{group_name}:{group_partial_reason}")
                else:
                    group_report["stage_status"] = "completed"
                    group_report["partial_reason"] = None
                _atomic_checkpoint(output, report)
            except GroupCallBudgetExceeded:
                group_report["stage_status"] = "partial"
                group_report["partial_reason"] = "group_call_budget"
                partial_reasons.append(f"{group_name}:group_call_budget")
            except (CallBudgetExceeded, PilotDeadlineExceeded) as exc:
                stop_reason = "call_budget" if isinstance(exc, CallBudgetExceeded) else "deadline"
            finally:
                # Provider-reported token counts are captured independently per
                # group. Logical call counts alone cannot establish equal spend
                # because retries and response lengths differ substantially.
                group_report["usage"] = _metrics_usage(
                    group_usage_before, get_metrics().snapshot()
                )
                reset_runtime(token)
                runtime.close()
            if stop_reason:
                break

    report["finished_at"] = _utcnow()
    report["duration_seconds"] = round(time.monotonic() - started, 6)
    if stop_reason or partial_reasons or len(report["groups"]) < len(selected_groups):
        report["status"] = "partial"
    else:
        report["status"] = "completed"
    report["stop_reason"] = stop_reason
    report["partial_reasons"] = partial_reasons
    report["calls"] = {
        "logical_total": budget.count,
        "by_channel": dict(budget.by_channel),
        "by_group": dict(budget.by_group),
        "group_limits": dict(budget.group_limits),
        "maximum": budget.maximum,
        "events": budget.events,
    }
    report["usage"] = _metrics_usage(before_usage, get_metrics().snapshot())
    report["cost_usd"] = None
    sanitize_pilot_report(report)
    _atomic_checkpoint(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated real A/B/C/D agent ablation pilot")
    parser.add_argument("output", type=Path, help="Checkpoint/report JSON path")
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--samples", type=int, default=2, help="Number of frozen briefs per selected kind")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS), help="Comma-separated subset of A,B,C,D")
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    parser.add_argument("--kind", action="append", choices=DEFAULT_KINDS, help="Repeat to select mechanisms")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--quality-cases", type=Path)
    parser.add_argument("--quality-case-limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_agent_pilot(
        load_settings(),
        args.output,
        max_calls=args.max_calls,
        samples=args.samples,
        groups=tuple(part.strip() for part in args.groups.split(",") if part.strip()),
        deadline_seconds=args.deadline_seconds,
        kinds=tuple(args.kind or DEFAULT_KINDS),
        random_seed=args.random_seed,
        quality_cases_path=args.quality_cases,
        quality_case_limit=args.quality_case_limit,
    )
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "logical_calls": report["calls"]["logical_total"],
    }, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
