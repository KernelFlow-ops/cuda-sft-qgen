from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import (
    DiversityReport,
    DupDecision,
    Question,
    SteeringBrief,
)
from cuda_sft_qgen.keys import PRI_CHAIR
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.review.consistency import (
    chair_may_reject_unique,
    consistency_payload,
    consistency_reason,
    leaf_soft_suspect,
)
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.diversity import build_diversity_report, steering_from_report
from cuda_sft_qgen.review.fingerprint import Fingerprint
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)


def _nearest(index_row: np.ndarray, score_row: np.ndarray, ids: list[str]) -> tuple[str | None, float | None]:
    for idx, score in zip(index_row.tolist(), score_row.tolist()):
        if idx < 0:
            continue
        if idx >= len(ids):
            continue
        return ids[idx], float(score)
    return None, None


def review_score_node(state: PipelineState) -> dict:
    """B1 embedding + B2 sentinel + B3 steward. Leaf/hw mismatch is deferred to chair rules."""
    rt = get_runtime()
    pending = [Question.model_validate(x) for x in (state.get("pending") or [])]
    if not pending:
        return {
            "status": "review_empty",
            "last_diversity": {},
        }
    texts = [q.question for q in pending]
    drain = getattr(rt, "drain_prefetch", None)
    if callable(drain):
        drain()
    logger.info(
        "review_score pending=%s embed_capacity=%s cache=%s",
        len(pending),
        None if rt.pool is None else rt.pool.embed_capacity,
        None if rt.embedding_cache is None else rt.embedding_cache.stats(),
    )
    vecs = rt.embedder.embed(texts)
    top_k = int(rt.settings.review.get("top_k", 8))
    nn_idx, nn_scores = rt.index.search(vecs, top_k=top_k)
    scored: list[dict[str, Any]] = []
    nn_vals: list[float] = []
    batch_deduper = Deduper(rt.settings)

    for i, q in enumerate(pending):
        nearest_id, nearest_cos = _nearest(nn_idx[i], nn_scores[i], rt.index.ids)
        if i > 0:
            local_sims = vecs[i] @ vecs[:i].T
            local_best = float(np.max(local_sims))
            local_j = int(np.argmax(local_sims))
            if nearest_cos is None or local_best > nearest_cos:
                nearest_cos = local_best
                nearest_id = pending[local_j].id
        fp = Fingerprint(
            exact=q.exact_hash,
            skeleton=q.template_fingerprint,
            simhash=int(q.simhash),
        )
        corpus = rt.deduper.decide(fp, nearest_id, nearest_cos)
        intra = batch_deduper.decide(fp, nearest_id, nearest_cos)
        decision = corpus
        if intra.verdict == "dup" and corpus.verdict != "dup":
            decision = intra
        elif intra.verdict == "gray" and corpus.verdict == "unique":
            decision = intra
        q.dup_verdict = decision.verdict
        q.nearest_id = decision.nearest_id
        q.nearest_cosine = decision.nearest_cosine
        q.template_jaccard = decision.template_jaccard
        q.hamming = decision.hamming
        if decision.verdict != "dup":
            batch_deduper.add_question(q)
        if decision.verdict == "gray":
            q.warnings.append("gray_zone")
        if nearest_cos is not None and nearest_cos > -1:
            nn_vals.append(float(nearest_cos))
        scored.append(
            {
                "question": q.to_jsonl(),
                "decision": decision.model_dump(),
                "vector_row": i,
            }
        )

    accepted_all = rt.store.load_accepted()
    report = build_diversity_report(
        accepted_all,
        pending,
        rt.taxonomy,
        rt.settings.quotas,
        mean_nn_cosine=float(np.mean(nn_vals)) if nn_vals else None,
        leaf_min=int(rt.settings.review.get("leaf_min_questions", 3)),
        kl_alert=float(rt.settings.review.get("kl_alert", 0.15)),
        entropy_alert=float(rt.settings.review.get("entropy_ratio_alert", 0.85)),
        overrep_factor=float(rt.settings.review.get("overrep_factor", 2.0)),
    )
    # Persist vectors on the side via last_diversity payload; archive_node writes index.
    return {
        "status": "review_scored",
        "last_diversity": {
            "report": report.model_dump(),
            "scored": scored,
            "vectors": vecs.tolist(),
        },
        "log": [
            f"review_score n={len(pending)} unique="
            f"{sum(1 for s in scored if s['decision']['verdict']=='unique')} "
            f"dup={sum(1 for s in scored if s['decision']['verdict']=='dup')} "
            f"gray={sum(1 for s in scored if s['decision']['verdict']=='gray')}"
        ],
    }


def _rule_action(decision: DupDecision) -> tuple[str, str | None]:
    if decision.verdict == "dup":
        return "reject", decision.reason
    return "accept", None


def review_chair_node(state: PipelineState) -> dict:
    rt = get_runtime()
    payload = state.get("last_diversity") or {}
    scored = payload.get("scored") or []
    report = DiversityReport.model_validate(payload.get("report") or {})
    review_rounds = int(state.get("review_rounds") or 0) + 1
    k = max(1, rt.settings.review_llm_every_k_batches)
    use_llm = review_rounds % k == 0

    gray_items = [
        s for s in scored if (s.get("decision") or {}).get("verdict") == "gray"
    ]
    suspects = []
    for s in scored:
        q = Question.model_validate(s["question"])
        payload = consistency_payload(q)
        if payload.get("rule_reason") or leaf_soft_suspect(q):
            suspects.append(payload)
    chair_map: dict[str, str] = {}
    chair_reason: dict[str, str] = {}
    steering: SteeringBrief | None = None
    if use_llm and (gray_items or suspects or True):
        try:
            user = (
                "Diversity report:\n"
                + report.model_dump_json(indent=2)
                + "\n\nGray items (reject only if obvious duplicates):\n"
                + str(
                    [
                        {
                            "id": s["question"]["id"],
                            "question": s["question"]["question"][:500],
                            "decision": s["decision"],
                        }
                        for s in gray_items
                    ]
                )
                + "\n\nConsistency suspects (reject only for leaf_mismatch or hw_mismatch):\n"
                + str(suspects)
            )
            obj = rt.llm.complete_json(
                [
                    {"role": "system", "content": load_prompt("chair")},
                    {"role": "user", "content": user},
                ],
                retries=rt.settings.max_parse_retries,
                priority=PRI_CHAIR,
            )
            for d in obj.get("decisions") or []:
                if d.get("id") and d.get("action") in {"accept", "reject"}:
                    qid = str(d["id"])
                    chair_map[qid] = str(d["action"])
                    if d.get("reason"):
                        chair_reason[qid] = str(d["reason"])
            if obj.get("steering_brief"):
                steering = SteeringBrief.model_validate(obj["steering_brief"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("chair LLM failed, using rule steering: %s", exc)

    if steering is None:
        anti_cap = int(rt.settings.generation.get("anti_templates_max", 15))
        forbidden = rt.store.recent_rejected_skeletons(anti_cap)
        steering = steering_from_report(
            report, rt.settings.quotas, forbidden, anti_cap=anti_cap
        )

    decisions: list[dict[str, Any]] = []
    for s in scored:
        q = Question.model_validate(s["question"])
        dec = DupDecision.model_validate(s["decision"])
        action, reason = _rule_action(dec)
        cons = consistency_reason(q)
        if cons:
            action, reason = "reject", cons
        elif dec.verdict == "gray":
            chair_action = chair_map.get(q.id, "accept")
            if chair_action == "reject":
                why = chair_reason.get(q.id) or "gray_chair_duplicate"
                action, reason = "reject", why
            else:
                action, reason = "accept", None
        elif dec.verdict == "unique":
            chair_action = chair_map.get(q.id, "accept")
            why = chair_reason.get(q.id)
            if chair_action == "reject" and chair_may_reject_unique(why):
                action, reason = "reject", why
            else:
                action, reason = "accept", None
        decisions.append(
            {
                "id": q.id,
                "action": action,
                "reason": reason,
                "question": q.to_jsonl(),
                "vector_row": s.get("vector_row"),
            }
        )

    payload = dict(payload)
    payload["final_decisions"] = decisions
    payload["steering_brief"] = steering.model_dump()
    return {
        "last_diversity": payload,
        "steering_brief": steering.model_dump(),
        "review_rounds": review_rounds,
        "status": "review_chaired",
        "log": [f"chair decisions={len(decisions)} gray={len(gray_items)}"],
    }


def archive_node(state: PipelineState) -> dict:
    rt = get_runtime()
    payload = state.get("last_diversity") or {}
    decisions = payload.get("final_decisions") or []
    vectors = np.asarray(payload.get("vectors") or [], dtype=np.float32)
    accepted_ids: list[str] = []
    accepted_rows: list[int] = []
    n_acc = 0
    n_rej = 0
    for item in decisions:
        q = Question.model_validate(item["question"])
        if item["action"] == "accept":
            q.review_status = "accepted"
            q.reject_reason = None
            rt.store.append_accepted(q)
            # Deduper may already contain this from scoring uniques/grays.
            if q.exact_hash not in rt.deduper.by_exact:
                rt.deduper.add_question(q)
            accepted_ids.append(q.id)
            if item.get("vector_row") is not None:
                accepted_rows.append(int(item["vector_row"]))
            n_acc += 1
        else:
            q.review_status = "rejected"
            q.reject_reason = item.get("reason") or "dup"
            rt.store.append_rejected(q)
            n_rej += 1
    if accepted_ids and vectors.size and accepted_rows:
        take = vectors[np.array(accepted_rows, dtype=np.int64)]
        rt.index.add(accepted_ids, take)
        rt.index.save(rt.settings.data_dir / "embeddings.npz")
    cache = getattr(rt, "embedding_cache", None)
    if cache is not None:
        cache.save()
    rt.store.save_pending([])
    steering = SteeringBrief.model_validate(
        payload.get("steering_brief") or state.get("steering_brief") or {}
    )
    rt.store.save_steering(steering)
    accepted_count = int(state.get("accepted_count") or 0) + n_acc
    rejected_count = int(state.get("rejected_count") or 0) + n_rej
    rt.store.save_run_state(
        {
            "iteration": int(state.get("iteration") or 0),
            "review_rounds": int(state.get("review_rounds") or 0),
            "accepted_count": accepted_count,
        }
    )
    rt.store.compute_stats()
    done = accepted_count >= int(state.get("target") or 0)
    return {
        "pending": [],
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "done": done,
        "status": "archived",
        "log": [f"archive accepted+={n_acc} rejected+={n_rej} total_accepted={accepted_count}"],
    }
