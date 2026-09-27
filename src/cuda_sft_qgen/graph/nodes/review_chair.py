"""Independent quality review for scored candidate questions.

Accept requires every check to pass, a private reference answer, and
assessed metadata. Uncertainty and call failures stay pending and never
default to accept.

This module does not plan briefs, write questions, or commit the archive.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ValidationError

from cuda_sft_qgen.errors import ReviewError
from cuda_sft_qgen.enums import ANSWER_FORMATS, ARCH_TARGETS, CHECK_TO_DEFECT, DEFECT_CODES
from cuda_sft_qgen.graph.nodes.review_score import _question_fingerprint
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.graph.handoff import brief_snapshot_from_question
from cuda_sft_qgen.keys import PRI_CHAIR
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.models import DiversityReport, QualityReview, Question, SteeringBrief
from cuda_sft_qgen.observability import agent_role, trace_node
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.prompts.fewshot import is_quality_reject
from cuda_sft_qgen.review.consistency import consistency_payload, consistency_reason
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.diversity import build_diversity_report, steering_from_report
from cuda_sft_qgen.planner.pair_mining import cuda_difficulty_features
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)
_METADATA_KEYS = ("question_type", "difficulty", "bloom", "hardware", "granularity", "language")
_SECOND_PASS_TYPES = frozenset({"formula", "code_complete", "debug", "layout"})


def render_chair_prompt(settings: Any) -> str:
    """Render verified hardware facts into the chair prompt."""
    from cuda_sft_qgen import hardware_facts

    try:
        sheet = hardware_facts.render_all_verified(hardware_facts.load(str(settings.hardware_facts_path)))
    except Exception:  # noqa: BLE001
        logger.warning("hardware_facts_unavailable for chair")
        sheet = "  (no verified hardware facts)"
    return load_prompt("chair").replace("{{hardware_fact_sheet}}", sheet)


def _normalize_review_obj(obj: dict, *, legacy_assessment: bool = False) -> dict:
    """Normalize integral value scores and discard unknown defect codes."""
    normalized = dict(obj)
    values = normalized.get("value_scores")
    if isinstance(values, dict):
        normalized["value_scores"] = {
            key: int(value) if isinstance(value, float) and value.is_integer() else value
            for key, value in values.items()
        }
    codes = normalized.get("defect_codes")
    if isinstance(codes, list):
        unknown = [code for code in codes if code not in DEFECT_CODES]
        if unknown:
            logger.warning("chair_unknown_defect_code codes=%s", unknown)
        normalized["defect_codes"] = [code for code in codes if code in DEFECT_CODES]
    assessed = normalized.get("assessed_metadata")
    if legacy_assessment and isinstance(assessed, dict):
        fmt = assessed.get("answer_format")
        arch = assessed.get("arch_target")
        if ((fmt is not None and fmt not in ANSWER_FORMATS)
                or (arch is not None and arch not in ARCH_TARGETS)):
            assessed = dict(assessed)
            if fmt is not None and fmt not in ANSWER_FORMATS:
                assessed["answer_format"] = None
            if arch is not None and arch not in ARCH_TARGETS:
                assessed["arch_target"] = None
            normalized["assessed_metadata"] = assessed
    return normalized


@dataclass(frozen=True)
class ChairOutcome:
    action: str
    reason: str
    defect_codes: list[str]


def apply_decision_rules(review: QualityReview, *, verification_status: str, verification_reason: str,
                         structured: bool, is_fake: bool, construction_error: str | None,
                         exploration: bool, min_value_score: int,
                         verification_evidence: dict[str, Any] | None = None) -> ChairOutcome:
    """Apply deterministic value, verification, and exploration gates."""
    codes = list(review.defect_codes)
    if review.action == "reject" and not codes:
        codes = [CHECK_TO_DEFECT[key] for key, value in review.checks.model_dump().items() if value == "fail"]
    action, reason = review.action, review.reason
    nested_evidence = (verification_evidence or {}).get("evidence") or {}
    cute_shape_result = nested_evidence.get("cute_shapes") if isinstance(nested_evidence, dict) else None
    cute_shape_contradiction = (
        verification_status == "fail"
        and verification_reason == "cute_shape_contradiction"
        and isinstance(cute_shape_result, dict)
        and cute_shape_result.get("status") == "contradiction"
    )
    if cute_shape_contradiction:
        action = "reject"
        reason = "cute_shape_guard:" + str(cute_shape_result.get("reason") or "contradiction")
        codes = ["correctness"]
    if action == "accept" and review.value_scores is not None and min(review.value_scores.values()) < min_value_score:
        action, reason, codes = "reject", f"low_value:{min(review.value_scores.values())}", ["low_value"]
    if action == "accept":
        allowed = {"pass"} if structured and not is_fake else {"pass", "disabled"}
        if verification_status not in allowed:
            action, reason = "pending", "verification:" + (verification_reason or "unconfirmed")
    if action == "accept" and construction_error and not is_fake:
        action, reason = "pending", construction_error
    if action == "accept" and exploration and verification_status != "pass":
        action, reason = "pending", "exploration_requires_verification"
    if action == "reject":
        reason = "chair:" + ",".join(codes or ["correctness"]) + ":" + reason[:300]
    return ChairOutcome(action, reason, codes)


def _chair_priority(settings: Any) -> int:
    """Chair waits behind Writer/Salvage unless YAML ``review.priority`` overrides."""
    return int(settings.review.get("priority", PRI_CHAIR))


def _needs_second_pass(question: Question, result: dict[str, Any], settings: Any) -> bool:
    """True when the first pass is uncertain (no fail) on a high-stakes type."""
    if not bool(settings.review.get("second_pass_on_uncertain", True)):
        return False
    if result.get("action") != "pending":
        return False
    qtype = (question.requested_metadata or {}).get("question_type") or question.question_type
    if qtype not in _SECOND_PASS_TYPES:
        return False
    checks = (result.get("review") or {}).get("checks") or {}
    if not isinstance(checks, dict) or not checks:
        return False
    values = [str(v).lower() for v in checks.values()]
    if "fail" in values:
        return False
    return "uncertain" in values


def _compact_neighbors(neighbors: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep neighbor stems short so the chair JSON stays bounded."""
    out: list[dict[str, Any]] = []
    for neighbor in (neighbors or [])[:6]:
        stem = str(neighbor.get("question") or "")
        if len(stem) > 400:
            stem = stem[:397] + "..."
        row = {"id": neighbor.get("id"), "question": stem}
        # Do not add null optional keys: legacy handoff tests and downstream
        # prompts rely on the compact shape when score metadata is absent.
        if neighbor.get("similarity") is not None:
            row["similarity"] = neighbor.get("similarity")
        if neighbor.get("l3_id") is not None:
            row["l3_id"] = neighbor.get("l3_id")
        out.append(row)
    return out


def _compact_knowledge(context: dict[str, Any] | None) -> dict[str, Any]:
    """Send claim status + short evidence, not the full subgraph."""
    if not context:
        return {"status": "missing", "claims": [], "evidence": []}
    # Small legacy contexts are already bounded and are passed through so the
    # handoff remains backward compatible with callers that provide revision
    # metadata or string evidence directly.
    try:
        if context.get("claims") or context.get("evidence"):
            # Preserve small legacy payloads exactly; larger knowledge graphs
            # are compacted below to keep the reviewer prompt bounded.
            if len(json.dumps(context, ensure_ascii=False)) <= 4000:
                return context
    except (TypeError, ValueError):
        pass
    claims = list(context.get("claims") or [])
    status = str((claims[0] or {}).get("status") or "unverified") if claims else "missing"
    slim_claims = []
    for claim in claims[:3]:
        text = str(claim.get("claim") or "")
        slim_claims.append({
            "id": claim.get("id"),
            "status": claim.get("status"),
            "claim": text[:220],
        })
    evidence = []
    for row in list(context.get("evidence") or [])[:3]:
        # Legacy knowledge rows were stored as plain strings.  Preserve them
        # as evidence instead of failing the whole independent review call.
        if isinstance(row, dict):
            quote = str(row.get("quote") or row.get("text") or "")
            url = row.get("url")
        else:
            quote, url = str(row or ""), None
        evidence.append({"url": url, "quote": quote[:240]})
    return {"status": status, "claims": slim_claims, "evidence": evidence}


def _construction_gate(q: Question, review: QualityReview) -> str | None:
    """Validate the requested synthesis against the independent Chair verdict.

    An integer difficulty label cannot represent level 5. Saturated level-4
    hybrid questions therefore also need a positive complexity-change check.
    Static token counts are a diagnostic prior, never a calibrated truth label.
    """
    required = set()
    if q.generation_provenance.get("structured_cot") or q.verification_spec:
        required.add("oracle_alignment")
    if q.mixup_mode in {"decomposed", "hybrid"}:
        required.update({"parent_claim_alignment", "mode_validity", "difficulty_order", "complexity_change"})
    checks = review.construction_checks or {}
    if any(checks.get(key) != "pass" for key in required):
        return "construction_checks_unconfirmed"
    if q.parent_difficulties and review.assessed_metadata:
        assessed = review.assessed_metadata.difficulty
        low, high = min(q.parent_difficulties), max(q.parent_difficulties)
        if q.mixup_mode == "decomposed" and not low <= assessed <= high:
            return "decomposed_difficulty_outside_parents"
        if q.mixup_mode == "hybrid" and assessed < min(4, high + 1):
            return "hybrid_difficulty_not_increased"
    return None


def _parallel_chair_decision(item: dict, prompt_system: str, verified_evidence: dict | None = None,
                             *, allow_fixture_origin: bool = False) -> dict[str, Any]:
    """Run one independent quality-review call; errors become pending, not accept."""
    rt = get_runtime()
    q = Question.model_validate(item["question"])
    try:
        if not rt.settings.fake_llm and not allow_fixture_origin and not rt.store.has_real_writer_origin(q):
            return {"id": q.id, "action": "pending", "reason": "unverified_writer_origin"}
        if q.draft_validation_errors and not isinstance(getattr(rt, "review_llm", None) or rt.llm, FakeLLM):
            # A Writer candidate without a complete private target and
            # verification contract has no auditable answer path.  Preserve it
            # for retry instead of allowing a reviewer to accept an underspecified
            # draft merely because the public stem looks plausible.
            return {
                "id": q.id,
                "action": "pending",
                "reason": "draft_validation:" + ",".join(q.draft_validation_errors[:4]),
                "review": {"action": "pending", "reason": "draft_validation", "draft_validation_errors": q.draft_validation_errors},
            }
        knowledge = getattr(rt, "knowledge", None)
        context = item.get("knowledge_context")
        if context is None and knowledge is not None:
            context = knowledge.context_for_claim(q.l3_id, revision=q.claim_revision)
        snapshot = item.get("brief_snapshot") or brief_snapshot_from_question(q)
        diagnostics = consistency_payload(q)
        diagnostics.pop("question", None)
        features = cuda_difficulty_features(q).as_dict()
        reviewer = getattr(rt, "review_llm", None) or rt.llm
        # Verification is part of the existing Chair stage, so both sequential
        # and concurrent graph paths retain the same backpressure and retries.
        from cuda_sft_qgen.review.verifier import evaluate_candidate
        # Evidence reuse is an explicit in-process argument, never a field
        # supplied by Writer or read from a possibly stale pending record.
        class BudgetedVerifier:
            """Count independent solver requests without changing Chair's API."""
            def complete_json(self, *args, **kwargs):
                reserve = getattr(rt, "reserve_verification_call", None)
                if callable(reserve) and reserve() is False:
                    raise RuntimeError("verification_run_budget_exhausted")
                with agent_role("verifier"):
                    return reviewer.complete_json(*args, **kwargs)
        solver = reviewer if isinstance(reviewer, FakeLLM) else BudgetedVerifier()
        verification = dict(verified_evidence) if verified_evidence is not None else evaluate_candidate(q, solver, rt.settings)
        if (verification.get("status") != "pass"
                and verification.get("reason") != "cute_shape_contradiction"
                and getattr(rt, "verification_budget_exhausted", False) is True):
            return {"id": q.id, "action": "pending", "reason": "verification_run_budget_exhausted"}
        verification["difficulty_features"] = features
        verification["difficulty_raw_score"] = features["score"]
        payload = {
            "id": q.id, "question": q.question, "l3_id": q.l3_id,
            "l3_claim": q.l3_claim, "tests_claim": q.tests_claim,
            "claim_revision": q.claim_revision, "source_ids": q.source_ids,
            "requested_metadata": q.requested_metadata or {k: getattr(q, k) for k in _METADATA_KEYS},
            "brief_snapshot": snapshot,
            "seed_id": q.seed_id, "operator": q.operator, "angle_id": q.angle_id,
            "rule_diagnostics": diagnostics,
            "deduplication": item.get("decision", {}),
            "accepted_neighbors": _compact_neighbors(item.get("neighbors") or []),
            "knowledge_context": _compact_knowledge(context if isinstance(context, dict) else None),
            "verification_evidence": verification,
            "verification_spec": q.verification_spec,
            "mixup_context": q.mixup_context,
            "required_construction_checks": (["oracle_alignment"] if q.verification_spec else []) + (
                ["parent_claim_alignment", "mode_validity", "difficulty_order", "complexity_change"]
                if q.mixup_mode in {"decomposed", "hybrid"} else []),
        }
        reviewer = getattr(rt, "review_llm", None) or rt.llm
        with agent_role("chair"):
            obj = reviewer.complete_json(
                [{"role": "system", "content": prompt_system},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                retries=1, priority=_chair_priority(rt.settings), temperature=0.0,
                thinking_level=str(rt.settings.review.get("thinking_level", "medium")),
                max_tokens=int(rt.settings.review.get("max_tokens", 2048)),
            )
        legacy_assessment = not bool(
            q.answer_format or q.verification_spec or q.draft_plan or q.generation_provenance.get("structured_cot")
        )
        review = QualityReview.model_validate(_normalize_review_obj(obj, legacy_assessment=legacy_assessment))
        if review.id != q.id:
            raise ValueError("review question id does not match the requested question")
        if review.action == "accept" and review.value_scores is None:
            raise ValueError("accepted review requires value_scores")
        details = review.model_dump()
        details["reviewer_model"] = "fake" if isinstance(reviewer, FakeLLM) else str(
            rt.settings.review.get("model") or getattr(rt.settings, "review_model", None) or rt.settings.llm_model
        )
        # Independent means a separate reviewer call with a complete audit
        # contract; it does not require a different model deployment.
        details["reviewer_independent"] = True
        structured = bool(q.verification_spec or q.draft_plan or q.generation_provenance.get("structured_cot"))
        outcome = apply_decision_rules(
            review,
            verification_status=str(verification.get("status") or ""),
            verification_reason=str(verification.get("reason") or ""),
            structured=structured,
            is_fake=isinstance(reviewer, FakeLLM),
            construction_error=_construction_gate(q, review),
            exploration=q.exploration,
            min_value_score=int(rt.settings.review.get("min_value_score", 3)),
            verification_evidence=verification,
        )
        action, reason = outcome.action, outcome.reason
        details["defect_codes"] = outcome.defect_codes
        details["value_scores"] = review.value_scores
        details["verification_evidence"] = verification
        if action != review.action:
            details["model_action"] = review.action
            details["action"], details["reason"] = action, reason
            # Preserve the strict persisted review contract: a deferred accept
            # must not masquerade as a passed review during resume/export.
            if action == "pending":
                details["checks"]["answerability"] = "uncertain"
        return {"id": q.id, "action": action, "reason": reason, "review": details}
    except Exception as exc:
        # Pydantic validation messages can embed private reference answers or
        # Writer artifacts. Log only a typed error; keep raw responses private.
        if isinstance(exc, ValidationError):
            fields = [(error["loc"], error["type"])
                      for error in exc.errors(include_input=False, include_context=False, include_url=False)[:12]]
            logger.warning("Quality review deferred for %s: ValidationError fields=%s", q.id, fields)
        else:
            logger.warning("Quality review deferred for %s: %s", q.id, type(exc).__name__)
        return {"id": q.id, "action": "pending", "reason": f"review_error:{type(exc).__name__}"}


def _parallel_chair_decisions(items: list[dict], max_workers: int = 4) -> dict[str, dict[str, Any]]:
    """Chair each scored item on a worker thread; inherit the runtime context."""
    if not items:
        return {}
    prompt = render_chair_prompt(get_runtime().settings)
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(copy_context().run, _parallel_chair_decision, item, prompt): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            qid = item["question"]["id"]
            try:
                results[qid] = future.result()
            except Exception as exc:
                results[qid] = {"id": qid, "action": "pending", "reason": f"review_worker_error:{type(exc).__name__}"}
                logger.warning("Quality review worker failed for %s: %s", qid, exc)
    rt = get_runtime()
    second = [
        item for item in items
        if _needs_second_pass(Question.model_validate(item["question"]), results.get(item["question"]["id"], {}), rt.settings)
    ]
    if second:
        logger.info("chair second_pass n=%s", len(second))
        extra = {}
        with ThreadPoolExecutor(max_workers=max(1, min(len(second), max_workers))) as executor:
            futures = {
                executor.submit(copy_context().run, _parallel_chair_decision, item, prompt,
                                (results.get(item["question"]["id"], {}).get("review") or {}).get("verification_evidence")): item
                for item in second
            }
            for future in as_completed(futures):
                item = futures[future]
                qid = item["question"]["id"]
                try:
                    extra[qid] = future.result()
                except Exception as exc:
                    extra[qid] = {"id": qid, "action": "pending", "reason": f"review_worker_error:{type(exc).__name__}"}
        for qid, second_result in extra.items():
            first = results.get(qid) or {}
            if second_result.get("action") == "accept" and first.get("action") == "pending":
                details = dict(second_result.get("review") or {})
                details["second_pass"] = True
                second_result = {**second_result, "review": details}
                results[qid] = second_result
            else:
                results[qid] = {
                    "id": qid,
                    "action": "pending",
                    "reason": "second_pass_unconfirmed:" + str(second_result.get("reason") or first.get("reason") or ""),
                    "review": second_result.get("review") or first.get("review"),
                }
    return results


def _enhanced_steering(report: DiversityReport, quotas: dict[str, Any], forbidden_templates: list[str],
                       anti_cap: int = 15) -> SteeringBrief:
    """Augment quota steering with high-KL and low-entropy notes."""
    rt = get_runtime()
    steering = steering_from_report(report, quotas, forbidden_templates, anti_cap=anti_cap)
    if report.kl_l1 > 0.20:
        scarcest = sorted(report.l1_hist, key=report.l1_hist.get)[:3]
        steering.underrepresented.extend(k for k in scarcest if k not in steering.underrepresented)
        steering.style_notes += f"; High KL divergence ({report.kl_l1:.3f}) - prioritize {scarcest}"
    if report.entropy_ratio_l1 < 0.80:
        for skeleton in rt.store.recent_rejected_skeletons(30):
            if skeleton not in steering.forbidden_templates and len(steering.forbidden_templates) < 30:
                steering.forbidden_templates.append(skeleton)
        steering.style_notes += "; Low entropy - broaden coverage"
    notes: list[str] = []
    for question in rt.store.load_rejected()[-8:]:
        reason = (question.reject_reason or "").strip()
        if is_quality_reject(reason) and reason not in notes:
            notes.append(reason[:80])
        if len(notes) >= 3:
            break
    if notes:
        steering.style_notes += "; Recent rejects (do not repeat): " + " | ".join(notes)
    if len(steering.style_notes) > 400:
        steering.style_notes = steering.style_notes[:397] + "..."
    return steering


def _defer_review(q: Question, reason: str, settings: Any) -> None:
    """Keep ``q`` pending with exponential backoff; mark exhausted after max attempts."""
    q.review_status = "pending"
    q.reject_reason = reason
    q.review_exhausted = q.review_attempts >= max(1, int(settings.review.get("max_attempts", 3)))
    delay = max(0.0, float(settings.review.get("retry_base_seconds", 30))) * 2 ** min(q.review_attempts - 1, 6)
    q.review_next_at = 0.0 if q.review_exhausted else time.time() + min(delay, 1800.0)


def _curate_select(scored: list[dict], results: dict, settings: Any,
                   vectors: np.ndarray | None) -> tuple[set[str], set[str]]:
    """Select diverse accepted questions per bucket with value-weighted MMR."""
    if not settings.review.get("rip_enabled", True):
        return set(), set()
    buckets: dict[tuple, list[tuple[Question, float, float | None, int | None]]] = defaultdict(list)
    below_min: set[str] = set()
    rip_min = float(settings.review.get("rip_min", 0.34))
    for item in scored:
        q = Question.model_validate(item["question"])
        outcome = results.get(q.id) or {}
        if outcome.get("action") != "accept":
            continue
        review = outcome.get("review") or {}
        evidence = review.get("verification_evidence") or {}
        rip = evidence.get("rip_score", (evidence.get("evidence") or {}).get("rip_score"))
        rip = float(rip) if isinstance(rip, (float, int)) else None
        if rip is not None and rip < rip_min:
            below_min.add(q.id)
            continue
        scores = review.get("value_scores")
        relevance = sum(scores.values()) / (5 * len(scores)) if isinstance(scores, dict) and scores else 0.6
        bucket = (tuple(q.topic_path[:2]), q.kind, q.hardware, q.language)
        buckets[bucket].append((q, relevance, rip, item.get("vector_row")))

    def similarity(a: tuple, b: tuple) -> float:
        row_a, row_b = a[3], b[3]
        if vectors is None or not vectors.size or row_a is None or row_b is None:
            return 0.0
        if int(row_a) >= len(vectors) or int(row_b) >= len(vectors):
            return 0.0
        return float(vectors[int(row_a)] @ vectors[int(row_b)])

    not_selected: set[str] = set()
    keep = max(1, int(settings.review.get("rip_keep_per_bucket", 3)))
    lam = float(settings.review.get("mmr_lambda", 0.5))
    for candidates in buckets.values():
        remaining = list(candidates)
        selected: list[tuple] = []
        while remaining and len(selected) < keep:
            if not selected:
                choice = min(remaining, key=lambda row: (-row[1],
                    -row[2] if row[2] is not None else 1.0, row[0].id))
            else:
                choice = min(remaining, key=lambda row: (
                    -(lam * row[1] - (1 - lam) * max(similarity(row, other) for other in selected)),
                    -row[2] if row[2] is not None else 1.0, row[0].id,
                ))
            remaining.remove(choice)
            selected.append(choice)
        not_selected.update(row[0].id for row in remaining)
    return not_selected, below_min


def _rip_select(scored: list[dict], results: dict, settings: Any) -> set[str]:
    """Preserve the old selector name for callers using bucket curation."""
    return _curate_select(scored, results, settings, None)[0]


@trace_node("review_chair")
def review_chair_node(state: PipelineState) -> dict:
    """Independently audit all due candidates, then deduplicate surviving representatives."""
    try:
        rt = get_runtime()
        payload = dict(state.get("last_diversity") or {})
        scored = payload.get("scored") or []
        report = DiversityReport.model_validate(payload.get("report") or {"accepted_count": 0, "pending_count": 0})
        parallel = bool(rt.settings.review.get("parallel_enabled", True))
        results = _parallel_chair_decisions(scored, int(rt.settings.review.get("chair_max_workers", 4)) if parallel else 1)
        raw_vectors = payload.get("vectors")
        vectors = np.asarray(raw_vectors if raw_vectors is not None else [], dtype=np.float32)
        not_selected, below_min = _curate_select(scored, results, rt.settings, vectors)
        batch_deduper = Deduper(rt.settings)
        surviving_rows, surviving_ids = [], []
        decisions = []
        for item in scored:
            q = Question.model_validate(item["question"])
            fp = _question_fingerprint(q)
            result = results.get(q.id, {"action": "pending", "reason": "missing_review_result"})
            action, reason = result["action"], result["reason"]
            if q.id in below_min:
                action, reason = "pending", "rip_below_min"
            elif q.id in not_selected:
                action, reason = "reject", "curation:mmr_not_selected"
            budget_deferred = reason == "verification_run_budget_exhausted"
            if not budget_deferred:
                q.review_attempts += 1
            cons = consistency_reason(q)
            if cons and cons not in q.warnings:
                q.warnings.append(cons)
            q.review_details = result.get("review") or {"action": "pending", "reason": reason}
            q.defect_codes = list(q.review_details.get("defect_codes") or [])
            q.value_scores = dict(q.review_details.get("value_scores") or {})
            q.internal_reference_answer = q.review_details.get("internal_reference_answer")
            q.verification_evidence = q.review_details.get("verification_evidence") or {}
            q.difficulty_raw_score = cuda_difficulty_features(q).score
            if action == "accept":
                from cuda_sft_qgen.models import AssessedMetadata

                q.assessed_metadata = AssessedMetadata.model_validate(q.review_details["assessed_metadata"])
                if not q.requested_metadata:
                    q.requested_metadata = {k: getattr(q, k) for k in _METADATA_KEYS}
                for key in _METADATA_KEYS:
                    setattr(q, key, getattr(q.assessed_metadata, key))
                corpus_decision = item.get("decision") or {}
                if corpus_decision.get("verdict") == "dup":
                    action, reason = "reject", corpus_decision.get("reason") or "corpus_duplicate"
                else:
                    nearest_id, cosine = None, None
                    row = item.get("vector_row")
                    if surviving_rows and row is not None and vectors.size:
                        sims = vectors[int(row)] @ vectors[np.asarray(surviving_rows)].T
                        best = int(np.argmax(sims))
                        nearest_id, cosine = surviving_ids[best], float(sims[best])
                    local = batch_deduper.decide(fp, nearest_id, cosine)
                    if local.verdict == "dup":
                        action, reason = "reject", local.reason
                        q.dup_verdict, q.nearest_id = local.verdict, local.nearest_id
                        q.nearest_cosine = local.nearest_cosine
                    elif local.verdict == "gray":
                        action, reason = "pending", "batch_neighbor_requires_review"
                        q.nearest_id, q.nearest_cosine = local.nearest_id, local.nearest_cosine
                    else:
                        batch_deduper.add_question(q)
                        if row is not None and vectors.size:
                            surviving_rows.append(int(row))
                            surviving_ids.append(q.id)
            if budget_deferred:
                q.review_status, q.reject_reason = "pending", reason
                q.review_next_at, q.review_exhausted = 0.0, False
            elif action == "pending":
                _defer_review(q, reason, rt.settings)
            else:
                q.review_next_at = 0.0
                q.review_exhausted = False
                q.review_status = "accepted" if action == "accept" else "rejected"
                q.reject_reason = reason if action == "reject" else None
            decisions.append({"id": q.id, "action": action, "reason": reason,
                              "question": q.model_dump(), "vector_row": item.get("vector_row")})
        for item in payload.get("deferred") or []:
            decisions.append({"id": item["id"], "action": "pending",
                              "reason": item.get("reject_reason") or "review_deferred",
                              "question": item, "vector_row": None})
        anti_cap = int(rt.settings.generation.get("anti_templates_max", 15))
        # The feedback loop must measure survivors, not rejected candidate labels.
        accepted = rt.store.load_accepted()
        survivors = [Question.model_validate(d["question"]) for d in decisions if d["action"] == "accept"]
        if getattr(rt, "taxonomy", None) is not None:
            report = build_diversity_report(accepted, survivors, rt.taxonomy, rt.settings.quotas)
        steering = _enhanced_steering(report, rt.settings.quotas, rt.store.recent_rejected_skeletons(anti_cap), anti_cap)
        payload.update(final_decisions=decisions, steering_brief=steering.model_dump())
        reviewed_this_wave = any(result.get("reason") != "verification_run_budget_exhausted"
                                 for result in results.values())
        return {"last_diversity": payload, "steering_brief": steering.model_dump(),
                "review_rounds": int(state.get("review_rounds") or 0) + int(reviewed_this_wave),
                "status": "review_chaired",
                "log": [f"chair decisions={len(decisions)} pending={sum(d['action'] == 'pending' for d in decisions)} parallel={parallel}"]}
    except Exception as exc:
        raise ReviewError(f"Review chair failed: {exc}", stage="chair") from exc
