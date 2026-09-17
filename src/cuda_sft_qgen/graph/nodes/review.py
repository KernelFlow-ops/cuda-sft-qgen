"""Review nodes: embedding score, independent chair audit, transactional archive.

Chair failures and uncertain checks stay pending. Dedup among accepted
survivors runs after quality review so a gray neighbor cannot skip the chair.
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
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.keys import PRI_WRITER
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.models import DiversityReport, QualityReview, Question, SteeringBrief
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.review.consistency import consistency_payload, consistency_reason
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.diversity import build_diversity_report, steering_from_report
from cuda_sft_qgen.review.fingerprint import Fingerprint, fingerprint
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)
_METADATA_KEYS = ("question_type", "difficulty", "bloom", "hardware", "granularity", "language")


def _question_fingerprint(q: Question) -> Fingerprint:
    """Return (and lazily fill) the three-layer fingerprint on ``q``."""
    if not q.exact_hash:
        fp = fingerprint(q.question)
        q.exact_hash, q.template_fingerprint, q.simhash = fp.exact, fp.skeleton, fp.simhash
    return Fingerprint(exact=q.exact_hash, skeleton=q.template_fingerprint, simhash=q.simhash)


@trace_node("review_score")
def review_score_node(state: PipelineState) -> dict:
    """Retrieve accepted neighbors; batch dedup runs after independent quality review."""
    try:
        rt = get_runtime()
        now = time.time()
        pending, deferred = [], []
        for item in state.get("pending") or []:
            q = Question.model_validate(item)
            if q.review_exhausted or q.review_next_at > now:
                deferred.append(q.model_dump())
            else:
                pending.append(q)
        accepted = rt.store.load_accepted()
        by_id = {q.id: q for q in accepted}
        scored, vectors, nn_values = [], [], []
        if pending:
            drain = getattr(rt, "drain_prefetch", None)
            if callable(drain):
                drain()
            vecs = rt.embedder.embed([q.question for q in pending])
            indices, scores = rt.index.search(vecs, top_k=int(rt.settings.review.get("top_k", 8)))
            vectors = vecs.tolist()
            for i, q in enumerate(pending):
                neighbors = []
                for idx, score in zip(indices[i].tolist(), scores[i].tolist()):
                    if 0 <= idx < len(rt.index.ids):
                        neighbor = by_id.get(rt.index.ids[idx])
                        if neighbor is not None and neighbor.id != q.id:
                            neighbors.append({
                                "id": neighbor.id, "question": neighbor.question,
                                "l3_claim": neighbor.l3_claim, "similarity": float(score),
                            })
                nearest = neighbors[0] if neighbors else {}
                decision = rt.deduper.decide(
                    _question_fingerprint(q), nearest.get("id"), nearest.get("similarity"), skip_id=q.id
                )
                # Exact/template matches can differ from the semantic nearest neighbor.
                if decision.nearest_id in by_id and not any(n["id"] == decision.nearest_id for n in neighbors):
                    neighbor = by_id[decision.nearest_id]
                    neighbors.append({"id": neighbor.id, "question": neighbor.question, "l3_claim": neighbor.l3_claim})
                q.dup_verdict = decision.verdict
                q.nearest_id, q.nearest_cosine = decision.nearest_id, decision.nearest_cosine
                q.template_jaccard, q.hamming = decision.template_jaccard, decision.hamming
                if nearest:
                    nn_values.append(nearest["similarity"])
                scored.append({"question": q.model_dump(), "decision": decision.model_dump(),
                               "neighbors": neighbors, "vector_row": i})
        if not pending and not deferred:
            return {"status": "review_empty", "last_diversity": {}}
        report = build_diversity_report(
            accepted, pending, rt.taxonomy, rt.settings.quotas,
            mean_nn_cosine=float(np.mean(nn_values)) if nn_values else None,
            leaf_min=int(rt.settings.review.get("leaf_min_questions", 3)),
            kl_alert=float(rt.settings.review.get("kl_alert", 0.15)),
            entropy_alert=float(rt.settings.review.get("entropy_ratio_alert", 0.85)),
            overrep_factor=float(rt.settings.review.get("overrep_factor", 2.0)),
        )
        return {
            "status": "review_scored",
            "last_diversity": {"report": report.model_dump(), "scored": scored,
                               "vectors": vectors, "deferred": deferred},
            "log": [f"review_score n={len(pending)} deferred={len(deferred)}"],
        }
    except Exception as exc:
        raise ReviewError(f"Review scoring failed: {exc}", stage="score") from exc


def _parallel_chair_decision(item: dict, prompt_system: str) -> dict[str, Any]:
    """Run one independent quality-review call; errors become pending, not accept."""
    rt = get_runtime()
    q = Question.model_validate(item["question"])
    try:
        knowledge = getattr(rt, "knowledge", None)
        context = item.get("knowledge_context")
        if context is None and knowledge is not None:
            context = knowledge.context_for_claim(q.l3_id, revision=q.claim_revision)
        payload = {
            "id": q.id, "question": q.question, "l3_id": q.l3_id,
            "l3_claim": q.l3_claim, "tests_claim": q.tests_claim,
            "claim_revision": q.claim_revision, "source_ids": q.source_ids,
            "requested_metadata": q.requested_metadata or {k: getattr(q, k) for k in _METADATA_KEYS},
            "seed_id": q.seed_id, "operator": q.operator, "angle_id": q.angle_id,
            "rule_diagnostics": consistency_payload(q),
            "deduplication": item.get("decision", {}),
            "accepted_neighbors": item.get("neighbors", []),
            "knowledge_context": context or {},
        }
        reviewer = getattr(rt, "review_llm", None) or rt.llm
        obj = reviewer.complete_json(
            [{"role": "system", "content": prompt_system},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            retries=0, priority=int(rt.settings.review.get("priority", PRI_WRITER)), temperature=0.0,
            thinking_level=str(rt.settings.review.get("thinking_level", "medium")),
            max_tokens=int(rt.settings.review.get("max_tokens", 8192)),
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
                    if claim is not None and (claim.get("status") in {"quarantined", "retired"}
                                             or (revision is not None and claim["revision"] != revision)):
                        item["action"], item["reason"] = "pending", "knowledge_revision_invalidated"
                        q.review_status, q.reject_reason = "pending", item["reason"]
                        q.review_exhausted, q.review_next_at = False, 0.0
                        q.review_details, q.internal_reference_answer = {}, None
                        item["question"] = q.model_dump()
                        break
        steering = SteeringBrief.model_validate(payload.get("steering_brief") or state.get("steering_brief") or {})
        run_state = {"iteration": int(state.get("iteration") or 0),
                     "review_rounds": int(state.get("review_rounds") or 0)}
        committed = rt.store.commit_review(decisions, steering, run_state)
        vectors = np.asarray(payload.get("vectors") or [], dtype=np.float32)
        for item in decisions:
            if item["action"] != "accept":
                continue
            q = Question.model_validate(item["question"])
            if q.exact_hash not in rt.deduper.by_exact:
                rt.deduper.add_question(q)
            row = item.get("vector_row")
            if row is not None and vectors.size and q.id not in rt.index.ids:
                rt.index.add([q.id], vectors[[int(row)]])
        # The transactional store is authoritative; derived index files can be rebuilt.
        try:
            if vectors.size:
                rt.index.save(rt.settings.data_dir / "embeddings.npz")
            cache = getattr(rt, "embedding_cache", None)
            if cache is not None:
                cache.save()
        except Exception as exc:
            logger.warning("Committed review; derived index/cache save failed: %s", exc)
        rt.store.compute_stats()
        accepted_count, rejected_count = committed["accepted_count"], committed["rejected_count"]
        return {"pending": committed["pending"], "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "done": accepted_count >= int(state.get("target") or rt.settings.target_question_count),
                "status": "archived", "log": [f"archive accepted={accepted_count} rejected={rejected_count} pending={len(committed['pending'])}"]}
    except Exception as exc:
        raise ReviewError(f"Archive failed: {exc}", stage="archive") from exc
