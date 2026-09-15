from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from cuda_sft_qgen.errors import RateLimitError
from cuda_sft_qgen.graph.nodes.generate import (
    apply_outcome,
    enqueue_retry,
    GenerateOutcome,
    question_from_obj,
    _safe_generate,
)
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.jsonutil import parse_question_payload
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.sieve import sieve_question

logger = logging.getLogger(__name__)


def _local_or_llm_salvage(item: dict[str, Any]) -> GenerateOutcome:
    rt = get_runtime()
    brief = QuestionBrief.model_validate(item["brief"])
    raw = str(item.get("raw") or "")
    obj = parse_question_payload(
        raw,
        fallback_claim=brief.l3_claim,
        min_question_chars=rt.settings.sieve_min_chars,
        keep_truncated=rt.settings.json_keep_truncated,
    )
    if obj and str(obj.get("question") or "").strip():
        q = question_from_obj(brief, obj, source=brief.role)
        reason = sieve_question(q, rt.settings, rt.llm)
        if reason:
            q.reject_reason = reason
            rt.store.append_rejected(q)
            return GenerateOutcome(status="hard_fail", reason=reason)
        return GenerateOutcome(status="ok", question=q)

    if not rt.settings.salvage_use_llm:
        return GenerateOutcome(
            status="retry",
            brief=brief.model_dump(),
            reason="json_parse",
            attempts=1,
        )

    prompt = (
        load_prompt("salvage")
        .replace("{{l3_claim}}", brief.l3_claim)
        .replace("{{raw}}", raw[:12000])
        .replace("{{min_chars}}", str(rt.settings.question_min_chars))
        .replace("{{max_chars}}", str(rt.settings.question_max_chars))
    )
    last: Exception | None = None
    for _ in range(max(1, rt.settings.salvage_llm_attempts)):
        try:
            text = rt.llm.complete(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Return the repaired JSON object now."},
                ],
                thinking_level=rt.settings.salvage_thinking_level,
                max_tokens=rt.settings.salvage_max_tokens,
            )
        except RateLimitError as exc:
            return GenerateOutcome(
                status="retry",
                brief=brief.model_dump(),
                reason="429",
                attempts=1,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
            continue
        obj = parse_question_payload(
            text,
            fallback_claim=brief.l3_claim,
            min_question_chars=rt.settings.sieve_min_chars,
            keep_truncated=rt.settings.json_keep_truncated,
        )
        if obj and str(obj.get("question") or "").strip():
            q = question_from_obj(brief, obj, source=brief.role)
            reason = sieve_question(q, rt.settings, rt.llm)
            if reason:
                q.reject_reason = reason
                rt.store.append_rejected(q)
                return GenerateOutcome(status="hard_fail", reason=reason)
            return GenerateOutcome(status="ok", question=q)
    logger.warning("salvage failed: %s", last)
    return GenerateOutcome(
        status="retry",
        brief=brief.model_dump(),
        reason="json_parse",
        attempts=1,
    )


def salvage_node(state: PipelineState) -> dict:
    rt = get_runtime()
    items = list(state.get("salvage_items") or [])
    pending = list(state.get("pending") or [])
    if not items or not rt.settings.salvage_enabled:
        for item in items:
            enqueue_retry(item.get("brief") or {}, reason="json_parse", attempts=1)
        return {"salvage_items": [], "pending": pending, "status": "salvage_skipped"}

    lock = threading.Lock()
    leftover: list[dict] = []
    ok = 0
    workers = max(1, min(len(items), rt.settings.writer_concurrency))

    def _one(item: dict[str, Any]) -> None:
        nonlocal ok
        outcome = _local_or_llm_salvage(item)
        if outcome.status == "ok":
            apply_outcome(
                outcome, pending=pending, salvage_items=leftover, lock=lock
            )
            with lock:
                ok += 1
            return
        if outcome.status == "retry" and outcome.brief is not None:
            enqueue_retry(
                outcome.brief,
                reason=outcome.reason or "json_parse",
                attempts=outcome.attempts,
            )
            return
        apply_outcome(outcome, pending=pending, salvage_items=leftover, lock=lock)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, it) for it in items]
        for fut in as_completed(futs):
            fut.result()
    return {
        "pending": pending,
        "salvage_items": [],
        "status": f"salvaged_{ok}",
        "log": [f"salvage recovered={ok} of {len(items)} pending={len(pending)}"],
    }


def retry_drain_node(state: PipelineState) -> dict:
    rt = get_runtime()
    pending = list(state.get("pending") or [])
    if not rt.settings.retry_drain_enabled:
        return {"pending": pending, "status": "retry_drain_skipped", "salvage_items": []}
    batch = int(state.get("review_batch_size") or rt.settings.review_batch_size)
    room = max(batch - len(pending), 0)
    limit = rt.settings.retry_drain_batch or rt.settings.writer_concurrency
    limit = min(max(1, limit), max(room, 1) if room else max(1, limit))
    if room == 0:
        return {"pending": pending, "status": "retry_drain_full", "salvage_items": []}
    items = rt.store.due_retry_items(limit=min(limit, room))
    if not items:
        return {"pending": pending, "status": "retry_drain_empty", "salvage_items": []}
    steering = SteeringBrief.model_validate(state.get("steering_brief") or {})
    salvage_items: list[dict] = []
    lock = threading.Lock()
    workers = max(1, min(len(items), rt.settings.writer_concurrency))
    logger.info("retry_drain n=%s workers=%s", len(items), workers)

    def _one(item: dict[str, Any]) -> None:
        brief = item.get("brief") or {}
        attempts = int(item.get("attempts") or 1) + 1
        outcome = _safe_generate(brief, steering, attempts=attempts)
        apply_outcome(
            outcome, pending=pending, salvage_items=salvage_items, lock=lock
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, it) for it in items]
        for fut in as_completed(futs):
            fut.result()
    return {
        "pending": pending,
        "salvage_items": salvage_items,
        "status": "retry_drained",
        "log": [f"retry_drain items={len(items)} pending={len(pending)}"],
    }
