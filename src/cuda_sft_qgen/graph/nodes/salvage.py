"""Salvage broken writer JSON and drain due retry-queue briefs."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any

from cuda_sft_qgen.graph.nodes.generate import (
    apply_outcome,
    enqueue_retry,
    generate_or_salvage,
    salvage_item,
)
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)

_local_or_llm_salvage = salvage_item


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
    lock = threading.Lock()
    workers = max(1, min(len(items), rt.settings.writer_concurrency))
    anti_k = int(rt.settings.generation.get("anti_templates_max", 15))
    corpus = FewshotCorpus.from_store(rt.store, anti_k=anti_k)
    logger.info("retry_drain n=%s workers=%s", len(items), workers)

    def _one(item: dict[str, Any]) -> None:
        brief = item.get("brief") or {}
        attempts = int(item.get("attempts") or 1) + 1
        outcome = generate_or_salvage(
            brief, steering, attempts=attempts, corpus=corpus
        )
        kind = apply_outcome(
            outcome, pending=pending, salvage_items=salvage_items, lock=lock
        )
        if item.get("task_id"):
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
        "salvage_items": salvage_items,
        "status": "retry_drained",
        "log": [f"retry_drain items={len(items)} pending={len(pending)}"],
    }
