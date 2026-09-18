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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any

import numpy as np

from cuda_sft_qgen.errors import ReviewError
from cuda_sft_qgen.graph.nodes.review_score import _question_fingerprint
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.graph.handoff import brief_snapshot_from_question
from cuda_sft_qgen.keys import PRI_CHAIR
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.models import DiversityReport, QualityReview, Question, SteeringBrief
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.review.consistency import consistency_payload, consistency_reason
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.diversity import build_diversity_report, steering_from_report
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)
_METADATA_KEYS = ("question_type", "difficulty", "bloom", "hardware", "granularity", "language")
_SECOND_PASS_TYPES = frozenset({"formula", "code_complete", "debug", "layout"})


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
        out.append({
            "id": neighbor.get("id"),
            "similarity": neighbor.get("similarity"),
            "l3_id": neighbor.get("l3_id"),
            "question": stem,
        })
    return out


def _compact_knowledge(context: dict[str, Any] | None) -> dict[str, Any]:
    """Send claim status + short evidence, not the full subgraph."""
    if not context:
        return {"status": "missing", "claims": [], "evidence": []}
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
        quote = str(row.get("quote") or row.get("text") or "")
        evidence.append({"url": row.get("url"), "quote": quote[:240]})
    return {"status": status, "claims": slim_claims, "evidence": evidence}


def _parallel_chair_decision(item: dict, prompt_system: str) -> dict[str, Any]:
    """Run one independent quality-review call; errors become pending, not accept."""
    rt = get_runtime()
    q = Question.model_validate(item["question"])
    try:
        knowledge = getattr(rt, "knowledge", None)
        context = item.get("knowledge_context")
        if context is None and knowledge is not None:
            context = knowledge.context_for_claim(q.l3_id, revision=q.claim_revision)
        snapshot = item.get("brief_snapshot") or brief_snapshot_from_question(q)
        diagnostics = consistency_payload(q)
        diagnostics.pop("question", None)
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
        }
        reviewer = getattr(rt, "review_llm", None) or rt.llm
        obj = reviewer.complete_json(
            [{"role": "system", "content": prompt_system},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            retries=1, priority=_chair_priority(rt.settings), temperature=0.0,
            thinking_level=str(rt.settings.review.get("thinking_level", "medium")),
            max_tokens=int(rt.settings.review.get("max_tokens", 2048)),
        )
        review = QualityReview.model_validate(obj)
        if review.id != q.id:
            raise ValueError("review question id does not match the requested question")
        details = review.model_dump()
        details["reviewer_model"] = "fake" if isinstance(reviewer, FakeLLM) else str(
            rt.settings.review.get("model") or getattr(rt.settings, "review_model", None) or rt.settings.nvidia_model
        )
        # Independent means a separate reviewer call with a complete audit
        # contract; it does not require a different model deployment.
        details["reviewer_independent"] = True
        return {"id": q.id, "action": review.action, "reason": review.reason, "review": details}
    except Exception as exc:
        logger.warning("Quality review deferred for %s: %s", q.id, exc)
        return {"id": q.id, "action": "pending", "reason": f"review_error:{type(exc).__name__}:{exc}"}


def _parallel_chair_decisions(items: list[dict], max_workers: int = 4) -> dict[str, dict[str, Any]]:
    """Chair each scored item on a worker thread; inherit the runtime context."""
    if not items:
        return {}
    prompt = load_prompt("chair")
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
                executor.submit(copy_context().run, _parallel_chair_decision, item, prompt): item
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
        if reason and reason not in notes:
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
        batch_deduper = Deduper(rt.settings)
        vectors = np.asarray(payload.get("vectors") or [], dtype=np.float32)
        surviving_rows, surviving_ids = [], []
        decisions = []
        for item in scored:
            q = Question.model_validate(item["question"])
            fp = _question_fingerprint(q)
            result = results.get(q.id, {"action": "pending", "reason": "missing_review_result"})
            action, reason = result["action"], result["reason"]
            q.review_attempts += 1
            cons = consistency_reason(q)
            if cons and cons not in q.warnings:
                q.warnings.append(cons)
            q.review_details = result.get("review") or {"action": "pending", "reason": reason}
            q.internal_reference_answer = q.review_details.get("internal_reference_answer")
            if action == "accept":
                from cuda_sft_qgen.models import AssessedMetadata

                q.assessed_metadata = AssessedMetadata.model_validate(q.review_details["assessed_metadata"])
                if not q.requested_metadata:
                    q.requested_metadata = {k: getattr(q, k) for k in _METADATA_KEYS}
                for key, value in q.assessed_metadata.model_dump().items():
                    setattr(q, key, value)
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
            if action == "pending":
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
        return {"last_diversity": payload, "steering_brief": steering.model_dump(),
                "review_rounds": int(state.get("review_rounds") or 0) + 1, "status": "review_chaired",
                "log": [f"chair decisions={len(decisions)} pending={sum(d['action'] == 'pending' for d in decisions)} parallel={parallel}"]}
    except Exception as exc:
        raise ReviewError(f"Review chair failed: {exc}", stage="chair") from exc
