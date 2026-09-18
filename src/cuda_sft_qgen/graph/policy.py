"""Shared generate/review/drain policy used by sequential and concurrent loops.

Workers produce data; this module decides when to review, salvage, drain
retries, or tick taxonomy. It does not call the LLM.

Sequential LangGraph edges live in ``routing.py``. The concurrent coordinator
in ``pipeline.py`` must use the same predicates so the two paths do not drift.
"""

from __future__ import annotations

import os
from typing import Any

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.runtime import get_runtime


def review_flush_min(batch: int) -> int:
    """Minimum pending to flush a nearly-full review batch.

    Tiny leftover generate batches (1-2 briefs) cannot use writer concurrency
    and stall pending for minutes. Tests without a runtime keep the old
    pending>=batch behaviour (flush_min=0).
    """
    try:
        rt = get_runtime()
    except Exception:
        return 0
    raw = rt.settings.generation.get("review_flush_min", None)
    if raw is not None and raw != "":
        return max(0, int(raw))
    env = os.getenv("REVIEW_FLUSH_MIN")
    if env and env.strip():
        return max(0, int(env))
    return max(0, min(batch, rt.settings.writer_concurrency))


def should_review(state: PipelineState) -> bool:
    """True when pending should flush to chair instead of another generate wave."""
    pending = state.get("pending") or []
    accepted = int(state.get("accepted_count") or 0)
    target = int(state.get("target") or 0)
    batch = int(state.get("review_batch_size") or 20)
    iteration = int(state.get("iteration") or 0)
    max_iter = int(state.get("max_iterations") or 200)
    if not pending:
        return False
    if len(pending) >= batch:
        return True
    if accepted + len(pending) >= target:
        return True
    if iteration >= max_iter:
        return True
    remaining = max(target - accepted - len(pending), 0)
    min_batch = 0
    try:
        min_batch = max(1, get_runtime().settings.min_generate_batch)
    except Exception:
        min_batch = review_flush_min(batch)
    # Remaining hole is smaller than a concurrent generate wave: flush pending
    # instead of a serial n=1/n=2 writer batch.
    if pending and min_batch and remaining < min_batch:
        return True
    flush_min = review_flush_min(batch)
    if flush_min and len(pending) >= flush_min:
        leftover = batch - len(pending)
        if leftover < flush_min:
            return True
    return False


def should_salvage(state: PipelineState) -> bool:
    """True when salvage items exist and salvage is enabled (default True)."""
    items = state.get("salvage_items") or []
    if not items:
        return False
    try:
        return bool(get_runtime().settings.salvage_enabled)
    except Exception:
        return True


def should_drain(state: PipelineState) -> bool:
    """True only when there is no pending work and retry items are due."""
    try:
        rt = get_runtime()
    except Exception:
        return False
    if not rt.settings.retry_drain_enabled:
        return False
    if should_review(state):
        return False
    pending = len(state.get("pending") or [])
    # Fresh briefs beat JSON retries. Drain only when the pipeline is idle
    # (no pending to grow or review); otherwise drain stalls writers for minutes.
    if pending > 0:
        return False
    return rt.store.due_retry_count() > 0


def should_taxonomy_tick(state: PipelineState) -> bool:
    """True every ``every_k_reviews`` chair rounds while below target."""
    try:
        rt = get_runtime()
    except Exception:
        return False
    if rt.settings.fake_llm:
        return False
    mgr = rt.settings.taxonomy_manager_cfg
    if not bool(mgr.get("enabled", True)):
        return False
    accepted = int(state.get("accepted_count") or 0)
    target = int(state.get("target") or 0)
    if accepted >= target:
        return False
    review_rounds = int(state.get("review_rounds") or 0)
    if review_rounds <= 0:
        return False
    every_k = int(mgr.get("every_k_reviews", 5) or 0)
    if every_k <= 0:
        return False
    return review_rounds >= every_k and review_rounds % every_k == 0


def chair_review_trigger(settings: Any) -> int:
    """Concurrent path: start chair when this many due items are waiting.

    Sequential LangGraph still uses ``review_batch_size`` via :func:`should_review`.
    """
    batch = max(1, int(getattr(settings, "review_batch_size", 20) or 20))
    review = getattr(settings, "review", {}) or {}
    workers = max(1, int(review.get("chair_max_workers", 4)))
    return min(batch, workers)


def chair_flush_size(settings: Any) -> int:
    """How many due items the concurrent chair wave should take.

    Flushing the full review batch (20) on the main thread blocks the next
    generate wave. Cap to a small multiple of chair workers so generate and
    review stay overlapped.
    """
    batch = max(1, int(getattr(settings, "review_batch_size", 20) or 20))
    review = getattr(settings, "review", {}) or {}
    workers = max(1, int(review.get("chair_max_workers", 4)))
    raw = review.get("chair_flush_max")
    if raw is not None and str(raw).strip() != "":
        return max(1, min(batch, int(raw)))
    return max(1, min(batch, max(workers, workers * 2)))


def pending_room(
    *,
    pending_n: int,
    active_n: int,
    capacity: int,
    total_capacity: int,
) -> tuple[int, int]:
    """Return ``(retry_room, new_room)`` for the concurrent producer.

    ``retry_room`` is space under the review-capacity cap for due/retry work.
    ``new_room`` is additional space under the hard pending cap for fresh briefs.
    """
    retry_room = max(0, capacity - active_n)
    new_room = min(retry_room, max(0, total_capacity - pending_n))
    return retry_room, max(0, new_room)
