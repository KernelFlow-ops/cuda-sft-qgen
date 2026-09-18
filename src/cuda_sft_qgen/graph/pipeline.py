"""Bounded producer/reviewer coordinator using the existing graph nodes."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from cuda_sft_qgen.graph.nodes.bootstrap import bootstrap_node
from cuda_sft_qgen.graph.nodes.generate import generate_batch_node
from cuda_sft_qgen.graph.nodes.planner import planner_node
from cuda_sft_qgen.graph.nodes.review import archive_node, review_chair_node, review_score_node
from cuda_sft_qgen.graph.nodes.salvage import retry_drain_node, salvage_node
from cuda_sft_qgen.graph.policy import chair_flush_size, chair_review_trigger, pending_room
from cuda_sft_qgen.observability import get_metrics

logger = logging.getLogger(__name__)


def run_concurrent(rt, initial: dict) -> dict:
    """Run generate on a background worker while the main thread chairs pending.

    Backpressure: new briefs are planned only when pending is below review
    capacity. Durable generation tasks and retry-queue items resume before
    fresh planner draws.

    Args:
        rt: Initialized :class:`~cuda_sft_qgen.runtime.Runtime`.
        initial: Bootstrap state from :func:`cuda_sft_qgen.graph.app.initial_state`.

    Returns:
        Final state dict with status ``archived``, ``review_pending``, or
        ``iteration_limit``.
    """
    state = {**initial, **bootstrap_node(initial)}
    if initial.get("rebuild_taxonomy"):
        from cuda_sft_qgen.graph.nodes.architect import architect_node
        state.update(architect_node(state))
    settings = rt.settings
    batch = max(1, settings.review_batch_size)
    review_trigger = chair_review_trigger(settings)
    flush_n = chair_flush_size(settings)
    capacity = max(batch, int(settings.review.get("queue_capacity", batch * 2)))
    total_capacity = max(capacity, int(settings.review.get("max_pending_total", capacity * 2)))
    generation = None
    last_tick = -1
    review_error_streak = 0
    start = time.monotonic()
    deadline = float(settings.generation.get("max_runtime_seconds", 0) or 0)
    resumed_this_run: set[str] = set()

    def refresh() -> None:
        rt.refresh_knowledge()
        state.update(
            accepted_count=len(rt.store.load_accepted()),
            rejected_count=len(rt.store.load_rejected()),
            pending=[q.model_dump() for q in rt.store.load_pending()],
            steering_brief=rt.store.load_steering().model_dump(),
        )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation-stage") as producer:
        while True:
            refresh()
            if generation is not None and generation.done():
                generated = generation.result()
                generation = None
                if generated.get("salvage_items"):
                    state.update(salvage_node({**state, "salvage_items": generated["salvage_items"]}))
                refresh()

            target_hit = state["accepted_count"] >= settings.target_question_count
            expired = bool(deadline and time.monotonic() - start >= deadline)
            active = [q for q in state["pending"] if not q.get("review_exhausted")]
            retry_room, room = pending_room(
                pending_n=len(state["pending"]),
                active_n=len(active),
                capacity=capacity,
                total_capacity=total_capacity,
            )
            can_produce = not target_hit and not expired and room > 0
            can_generate = can_produce and state["iteration"] < settings.max_iterations
            if generation is None and retry_room > 0:
                resumed = [task for task in rt.store.unfinished_generation_tasks()
                           if task.get("task_id") not in resumed_this_run][:min(batch, retry_room)]
                briefs = []
                if resumed:
                    briefs = resumed
                    resumed_this_run.update(task["task_id"] for task in resumed)
                elif settings.retry_drain_enabled and rt.store.due_retry_count():
                    # Durable retries must not starve behind fresh work, including
                    # when the limit for new planning iterations has been reached.
                    snapshot = {**state, "pending": [], "review_batch_size": min(batch, retry_room)}
                    generation = producer.submit(copy_context().run, retry_drain_node, snapshot)
                elif can_generate:
                    planned = planner_node({
                        **state,
                        "pending": active,
                        "review_batch_size": batch,
                        "generate_room": room,
                    })
                    state.update(planned)
                    briefs = state.get("batch_briefs") or []
                if briefs:
                    for brief in briefs[:max(room, retry_room)]:
                        rt.store.save_generation_task(brief)
                        resumed_this_run.add(brief["task_id"])
                    snapshot = {**state, "batch_briefs": briefs[:max(room, retry_room)], "pending": []}
                    generation = producer.submit(copy_context().run, generate_batch_node, snapshot)
                rt.store.save_run_state({"iteration": state["iteration"],
                                         "review_rounds": state["review_rounds"],
                                         "accepted_count": state["accepted_count"]})

            now = time.time()
            ready = [q for q in state["pending"] if not q.get("review_exhausted")
                     and float(q.get("review_next_at") or 0) <= now]
            if ready and (len(ready) >= review_trigger or generation is None):
                rt.reconcile_index()
                reviewing = {**state, "pending": ready[:flush_n]}
                reviewing.update(review_score_node(reviewing))
                reviewing.update(review_chair_node(reviewing))
                state.update(archive_node(reviewing))
                state["review_rounds"] = reviewing["review_rounds"]
                refresh()
                error_n = sum(
                    1 for q in (reviewing.get("last_diversity") or {}).get("final_decisions") or []
                    if str(q.get("reason") or "").startswith("review_error")
                    or str(q.get("reason") or "").startswith("review_worker_error")
                )
                if error_n:
                    review_error_streak += error_n
                else:
                    review_error_streak = 0
                max_streak = int(settings.review.get("circuit_breaker_error_streak", 8) or 0)
                max_exhausted = int(settings.review.get("circuit_breaker_exhausted", 20) or 0)
                exhausted_n = sum(1 for q in state["pending"] if q.get("review_exhausted"))
                if max_streak and review_error_streak >= max_streak:
                    logger.warning("circuit breaker: review_error_streak=%s", review_error_streak)
                    state["status"] = "review_pending"
                    break
                if max_exhausted and exhausted_n >= max_exhausted:
                    logger.warning("circuit breaker: exhausted=%s", exhausted_n)
                    state["status"] = "review_pending"
                    break
                rounds = state["review_rounds"]
                every = max(1, int(settings.taxonomy_manager_cfg.get("every_k_reviews", 5)))
                if rt.knowledge_worker is not None and rounds != last_tick and rounds % every == 0:
                    rt.knowledge_worker.enqueue_expansion(
                        last_facet=state.get("last_taxonomy_facet"), review_round=rounds)
                    last_tick = rounds
                continue

            if generation is None:
                if ready:
                    continue
                # No producer or immediately reviewable work remains. Persist
                # delayed/exhausted work for resume rather than polling indefinitely.
                break
            time.sleep(0.05)

    refresh()
    state["done"] = state["accepted_count"] >= settings.target_question_count
    state["status"] = "archived" if state["done"] else (
        "review_pending" if state["pending"] else "iteration_limit")
    elapsed = time.monotonic() - start
    get_metrics().observe("pipeline_seconds", elapsed)
    get_metrics().increment("accepted_total", state["accepted_count"])
    logger.info("pipeline_complete accepted=%s pending=%s duration_s=%.2f status=%s",
                state["accepted_count"], len(state["pending"]), elapsed, state["status"])
    return state
