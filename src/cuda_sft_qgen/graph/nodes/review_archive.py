"""Transactional archive node: commit chair decisions and derived index updates.

Knowledge dependencies are recorded before accept is committed so a concurrent
correction can invalidate the question. The transactional store is authoritative.

This module does not score, chair, or generate questions.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

from cuda_sft_qgen.errors import ReviewError
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)
_REPAIRABLE_DEFECTS = frozenset({
    "correctness", "answerability", "multiple_correct", "insufficient_premises",
    "claim_alignment", "hardware", "leakage", "metadata", "ub_misstated", "invented_api",
})


def enqueue_review_repairs(rt, decisions: list[dict], steering: SteeringBrief) -> int:
    """Queue one new Writer attempt for an API-authored, repairable Chair reject."""
    cfg = rt.settings.generation
    if rt.settings.fake_llm or not cfg.get("review_repair_enabled", True):
        return 0
    maximum = int(cfg.get("review_repair_max_attempts", 1))
    queued = 0
    for item in decisions:
        if item["action"] != "reject":
            continue
        q = Question.model_validate(item["question"])
        codes = set(q.defect_codes)
        if (not codes or not codes.issubset(_REPAIRABLE_DEFECTS)
                or not rt.store.has_real_writer_origin(q)):
            continue
        attempts = int(q.generation_provenance.get("review_repair_attempt") or 0)
        if attempts >= maximum:
            continue
        original_task_id = str(q.generation_provenance.get("task_id") or "")
        brief = rt.store.generation_task_payload(original_task_id) if original_task_id else None
        if brief is None:
            continue
        brief = {**brief, "task_id": uuid.uuid4().hex}
        created = rt.store.enqueue_review_repair(q.id, {
            "brief": brief, "reason": str(item.get("reason") or "quality review")[:400],
            "source_id": q.id, "attempts": attempts + 1,
            "steering": steering.model_dump(),
        })
        queued += int(created)
    return queued


@trace_node("archive")
def archive_node(state: PipelineState) -> dict:
    """Commit chair decisions transactionally; record knowledge dependencies first."""
    try:
        rt = get_runtime()
        payload = state.get("last_diversity") or {}
        decisions = payload.get("final_decisions") or []
        knowledge = getattr(rt, "knowledge", None)
        if knowledge is not None:
            for item in decisions:
                q = Question.model_validate(item["question"])
                # Register dependencies before checking, so a concurrent correction
                # can invalidate this question through the impact event consumer.
                knowledge.record_question(q)
                if item["action"] != "accept":
                    continue
                dependencies = [{"claim_id": q.l3_id, "revision": q.claim_revision}] + q.knowledge_dependencies
                for dependency in dependencies:
                    claim = knowledge.get_claim(dependency["claim_id"])
                    revision = dependency.get("revision")
                    missing_parent = claim is None and dependency["claim_id"] in {
                        p.get("l3_id") for p in q.mixup_context.get("parents", []) if isinstance(p, dict)
                    }
                    if missing_parent or (claim is not None and (claim.get("status") in {"quarantined", "retired"}
                                             or (revision is not None and claim["revision"] != revision))):
                        item["action"], item["reason"] = "pending", "knowledge_revision_invalidated"
                        q.review_status, q.reject_reason = "pending", item["reason"]
                        q.review_exhausted, q.review_next_at = False, 0.0
                        q.review_details, q.internal_reference_answer = {}, None
                        q.verification_evidence = {}
                        item["question"] = q.model_dump()
                        break
        steering = SteeringBrief.model_validate(payload.get("steering_brief") or state.get("steering_brief") or {})
        run_state = {"iteration": int(state.get("iteration") or 0),
                     "review_rounds": int(state.get("review_rounds") or 0)}
        committed = rt.store.commit_review(decisions, steering, run_state)
        vectors = np.asarray(payload.get("vectors") or [], dtype=np.float32)
        gist_vectors = np.asarray(payload.get("gist_vectors") or [], dtype=np.float32)
        gist_index = getattr(rt, "gist_index", None)
        for item in decisions:
            if item["action"] != "accept":
                continue
            q = Question.model_validate(item["question"])
            if q.exact_hash not in rt.deduper.by_exact:
                rt.deduper.add_question(q)
            row = item.get("vector_row")
            if row is not None and vectors.size and q.id not in rt.index.ids:
                rt.index.add([q.id], vectors[[int(row)]])
            if gist_index is not None and row is not None and gist_vectors.size and q.id not in gist_index.ids:
                gist_index.add([q.id], gist_vectors[[int(row)]])
        # The transactional store is authoritative; derived index files can be rebuilt.
        try:
            if vectors.size:
                rt.index.save(rt.settings.data_dir / "embeddings.npz")
            if gist_index is not None:
                gist_index.save(rt.settings.data_dir / "gist_embeddings.npz")
            cache = getattr(rt, "embedding_cache", None)
            if cache is not None:
                cache.save()
        except Exception as exc:
            logger.warning("Committed review; derived index/cache save failed: %s", exc)
        accepted_count = rt.store.status_counts()["verified_accepted"]
        if accepted_count < int(state.get("target") or rt.settings.target_question_count):
            enqueued = enqueue_review_repairs(rt, decisions, steering)
            if enqueued:
                logger.info("chair repair queued=%s", enqueued)
        rt.store.compute_stats()
        rejected_count = committed["rejected_count"]
        return {"pending": committed["pending"], "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "done": accepted_count >= int(state.get("target") or rt.settings.target_question_count),
                "status": "archived", "log": [f"archive accepted={accepted_count} rejected={rejected_count} pending={len(committed['pending'])}"]}
    except Exception as exc:
        raise ReviewError(f"Archive failed: {exc}", stage="archive") from exc
