"""Salvage broken writer JSON and drain due retry-queue briefs."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any

from cuda_sft_qgen.graph.nodes.generate_core import (
    apply_outcome,
    enqueue_retry,
    generate_or_salvage,
    review_repair_item,
    rewrite_item,
    salvage_item,
)
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)

_local_or_llm_salvage = salvage_item


def repair_batch_node(items: list[dict[str, Any]]) -> dict:
    """Run JSON salvage and Writer rewrites off the generation/review workers."""
    rt = get_runtime()
    logger.info("repair_stage start items=%s", len(items))
    pending: list[dict] = []
    lock = threading.Lock()
    counts = {"ok": 0, "retry": 0, "hard_fail": 0}
    workers = max(1, min(len(items), int(rt.settings.generation.get("repair_concurrency", 2))))

    def _one(item: dict[str, Any]) -> None:
        try:
            if item["kind"] == "review_repair":
                outcome = review_repair_item(item["payload"])
            elif item["kind"] == "rewrite":
                outcome = rewrite_item(item["payload"])
            else:
                outcome = _local_or_llm_salvage(item["payload"])
            if item["kind"] == "review_repair" and outcome.status == "retry":
                # The durable repair task remains pending for a later resume.
                kind = "retry"
            else:
                kind = apply_outcome(outcome, pending=pending, salvage_items=[], lock=lock)
                if item["kind"] == "review_repair":
                    rt.store.complete_review_repair(item["payload"]["repair_task_id"])
            if kind == "ok" and outcome.question is not None:
                rt.prefetch_embed(outcome.question.question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("repair item failed kind=%s error=%s", item.get("kind"), type(exc).__name__)
            payload = item.get("payload") or {}
            if item.get("kind") != "review_repair":
                enqueue_retry(payload.get("brief") or {}, reason=f"repair_error:{type(exc).__name__}",
                              attempts=int(payload.get("attempts") or 1))
            kind = "retry"
        retry_task_id = (item.get("payload") or {}).get("retry_task_id")
        if retry_task_id:
            rt.store.complete_retry(str(retry_task_id))
        with lock:
            counts[kind] = counts.get(kind, 0) + 1

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="repair-item") as pool:
        futures = [pool.submit(copy_context().run, _one, item) for item in items]
        for future in as_completed(futures):
            future.result()
    logger.info("repair_batch items=%s ok=%s retry=%s", len(items), counts["ok"], counts["retry"])
    return {"status": "repaired", "counts": counts}


def salvage_node(state: PipelineState) -> dict:
    """Repair ``salvage_items`` in parallel; unrepaired briefs go to the retry queue."""
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
            if outcome.question is not None:
                prefetch = getattr(rt, "prefetch_embed", None)
                if callable(prefetch):
                    prefetch(outcome.question.question)
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
        futs = [pool.submit(copy_context().run, _one, it) for it in items]
        for fut in as_completed(futs):
            fut.result()
    logger.info(
        "salvage recovered=%s of %s pending=%s leftover=%s",
        ok,
        len(items),
        len(pending),
        len(leftover),
    )
    return {
        "pending": pending,
        "salvage_items": [],
        "status": f"salvaged_{ok}",
        "log": [f"salvage recovered={ok} of {len(items)} pending={len(pending)}"],
    }


def retry_drain_node(state: PipelineState) -> dict:
    """Re-run due retry-queue briefs into pending without starving review capacity."""
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
    repair_items: list[dict] = []
    repair_queue = state.get("_repair_queue")
    lock = threading.Lock()
    workers = max(1, min(len(items), rt.settings.writer_concurrency))
    anti_k = int(rt.settings.generation.get("anti_templates_max", 15))
    corpus = FewshotCorpus.from_store(rt.store, anti_k=anti_k)
    logger.info("retry_drain n=%s workers=%s", len(items), workers)

    def _one(item: dict[str, Any]) -> None:
        brief = item.get("brief") or {}
        attempts = int(item.get("attempts") or 1) + 1
        outcome = generate_or_salvage(
            brief, steering, attempts=attempts, corpus=corpus,
            defer_rewrite=repair_queue is not None,
        )
        kind = apply_outcome(
            outcome, pending=pending, salvage_items=salvage_items, lock=lock,
            repair_items=repair_items, steering=steering,
        )
        if repair_queue is not None and kind == "salvage":
            repair_queue.put({"kind": "salvage", "payload": {
                "brief": outcome.brief, "raw": outcome.raw or "",
                "writer_real_api": outcome.writer_real_api,
                "retry_task_id": item.get("task_id"),
            }})
        elif repair_queue is not None and kind == "repair":
            repair_queue.put({"kind": "rewrite", "payload": {
                "brief": outcome.brief, "question": outcome.question.model_dump(),
                "reason": outcome.reason, "attempts": outcome.attempts,
                "steering": steering.model_dump(),
                "retry_task_id": item.get("task_id"),
            }})
        if item.get("task_id") and (repair_queue is None or kind not in {"salvage", "repair"}):
            rt.store.complete_retry(item["task_id"])
        if kind == "ok" and outcome.question is not None:
            prefetch = getattr(rt, "prefetch_embed", None)
            if callable(prefetch):
                prefetch(outcome.question.question)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(copy_context().run, _one, it) for it in items]
        for fut in as_completed(futs):
            fut.result()
    return {
        "pending": pending,
        "salvage_items": [] if repair_queue is not None else salvage_items,
        "repair_items": [] if repair_queue is not None else repair_items,
        "status": "retry_drained",
        "log": [f"retry_drain items={len(items)} pending={len(pending)}"],
    }
