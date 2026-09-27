"""Embedding score node: neighbor retrieval and fingerprint fill for pending questions.

Dedup among accepted survivors runs after independent quality review so a
gray neighbor cannot skip the chair.

This module does not chair, archive, or write questions.
"""

from __future__ import annotations

import time

import numpy as np

from cuda_sft_qgen.errors import ReviewError
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.review.dedup import gist_verdict
from cuda_sft_qgen.review.diversity import build_diversity_report
from cuda_sft_qgen.review.fingerprint import Fingerprint, fingerprint
from cuda_sft_qgen.runtime import get_runtime


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
        scored, vectors, gist_vectors, nn_values = [], [], [], []
        if pending:
            wait_prefetch = getattr(rt, "wait_prefetch", None)
            if callable(wait_prefetch):
                wait_prefetch([q.question for q in pending])
            vecs = rt.embedder.embed([q.question for q in pending])
            gist_vecs = rt.embedder.embed([q.tests_claim or q.question for q in pending])
            gist_index = getattr(rt, "gist_index", None)
            indices, scores = rt.index.search(vecs, top_k=int(rt.settings.review.get("top_k", 8)))
            vectors = vecs.tolist()
            gist_vectors = gist_vecs.tolist()
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
                if gist_index is not None and len(gist_index):
                    gist_ids, gist_scores = gist_index.search(gist_vecs[i:i + 1], top_k=5)
                    best_id, best_cos = None, None
                    for index, score in zip(gist_ids[0].tolist(), gist_scores[0].tolist()):
                        if 0 <= index < len(gist_index.ids):
                            other = by_id.get(gist_index.ids[index])
                            if other is not None and other.id != q.id and other.language != q.language:
                                best_id, best_cos = other.id, float(score)
                                break
                    decision = gist_verdict(
                        decision, best_cos,
                        dup=float(rt.settings.review.get("gist_dup_threshold", 0.92)),
                        gray=float(rt.settings.review.get("gist_gray_threshold", 0.86)),
                        nearest_id=best_id,
                    )
                    if best_id and best_id in by_id and not any(n["id"] == best_id for n in neighbors):
                        other = by_id[best_id]
                        neighbors = neighbors[:5] + [{"id": other.id, "question": other.question,
                                                       "l3_claim": other.l3_claim, "similarity": best_cos}]
                neighbors = neighbors[:6]
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
                               "vectors": vectors, "gist_vectors": gist_vectors, "deferred": deferred},
            "log": [f"review_score n={len(pending)} deferred={len(deferred)}"],
        }
    except Exception as exc:
        raise ReviewError(f"Review scoring failed: {exc}", stage="score") from exc
