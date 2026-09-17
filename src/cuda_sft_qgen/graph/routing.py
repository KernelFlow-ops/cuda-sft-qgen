"""Conditional edges after bootstrap, generate, salvage, archive, and taxonomy.

Review is triggered when pending fills a batch, the remaining hole is smaller
than a concurrent generate wave, or the run is finishing. JSON salvage and
retry drain yield to fresh writer work so 429 retries do not stall throughput.
"""

from __future__ import annotations

import os

from langgraph.graph import END

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.runtime import get_runtime


def route_after_bootstrap(state: PipelineState) -> str:
    """Architect if rebuilding taxonomy; else review leftover pending or plan."""
    if state.get("rebuild_taxonomy"):
        return "architect"
    if state.get("done") and not (state.get("pending") or []):
        return END
    if state.get("done") and (state.get("pending") or []):
        return "review_score"
    if should_review(state):
        return "review_score"
    return "planner"


def _review_flush_min(batch: int) -> int:
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
        min_batch = _review_flush_min(batch)
    # Remaining hole is smaller than a concurrent generate wave: flush pending
    # instead of a serial n=1/n=2 writer batch.
    if pending and min_batch and remaining < min_batch:
        return True
    flush_min = _review_flush_min(batch)
    if flush_min and len(pending) >= flush_min:
        leftover = batch - len(pending)
        if leftover < flush_min:
            return True
    return False


def _should_salvage(state: PipelineState) -> bool:
    """True when salvage items exist and salvage is enabled (default True)."""
    items = state.get("salvage_items") or []
    if not items:
        return False
    try:
        return bool(get_runtime().settings.salvage_enabled)
    except Exception:
        return True


def _should_drain(state: PipelineState) -> bool:
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


def route_after_generate(state: PipelineState) -> str:
    """Salvage broken JSON first, else drain retries, else review/plan/end."""
    if _should_salvage(state):
        return "salvage"
    if _should_drain(state):
        return "retry_drain"
    return _route_continue(state)


def route_after_salvage(state: PipelineState) -> str:
    """Drain due retries if idle; otherwise continue to review or planner."""
    if _should_drain(state):
        return "retry_drain"
    return _route_continue(state)


def route_after_drain(state: PipelineState) -> str:
    """Salvage any drain-produced JSON, else taxonomy tick, else continue."""
    if _should_salvage(state):
        return "salvage"
    if should_taxonomy_tick(state):
        return "taxonomy_manager"
    return _route_continue(state)


def _route_continue(state: PipelineState) -> str:
    """Shared continue: review if due, else END on target/iter, else planner."""
    if should_review(state):
        return "review_score"
    accepted = int(state.get("accepted_count") or 0)
    target = int(state.get("target") or 0)
    iteration = int(state.get("iteration") or 0)
    max_iter = int(state.get("max_iterations") or 200)
    pending = state.get("pending") or []
    if state.get("done") and not pending:
        return END
    if accepted >= target and not pending:
        return END
    if iteration >= max_iter:
        return END if not pending else "review_score"
    return "planner"


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


def route_after_archive(state: PipelineState) -> str:
    """END on target or max iterations; else taxonomy tick or next planner batch."""
    accepted = int(state.get("accepted_count") or 0)
    target = int(state.get("target") or 0)
    iteration = int(state.get("iteration") or 0)
    max_iter = int(state.get("max_iterations") or 200)
    if accepted >= target:
        return END
    if iteration >= max_iter:
        return END
    # Prefer the next generate batch over draining JSON retries. Drain still
    # runs after generate when pending is nearly empty.
    if should_taxonomy_tick(state):
        return "taxonomy_manager"
    return "planner"


def route_after_taxonomy(state: PipelineState) -> str:
    """END on target/iter; drain retries if idle, otherwise plan the next batch."""
    accepted = int(state.get("accepted_count") or 0)
    target = int(state.get("target") or 0)
    iteration = int(state.get("iteration") or 0)
    max_iter = int(state.get("max_iterations") or 200)
    if accepted >= target:
        return END
    if iteration >= max_iter:
        return END
    if _should_drain(state):
        return "retry_drain"
    return "planner"
