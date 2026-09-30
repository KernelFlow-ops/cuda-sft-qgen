"""Bounded producer/reviewer coordinator using the existing graph nodes."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from queue import Empty, Queue

from cuda_sft_qgen.graph.nodes.bootstrap import bootstrap_node
from cuda_sft_qgen.graph.nodes.generate import generate_batch_node
from cuda_sft_qgen.graph.nodes.generate_core import enqueue_retry
from cuda_sft_qgen.graph.nodes.planner import planner_node
from cuda_sft_qgen.graph.nodes.review import archive_node, review_chair_node, review_score_node
from cuda_sft_qgen.graph.nodes.salvage import repair_batch_node, retry_drain_node
from cuda_sft_qgen.graph.policy import chair_flush_size, chair_review_trigger, pending_room
from cuda_sft_qgen.observability import get_metrics

logger = logging.getLogger(__name__)


def _review_stage(snapshot: dict) -> tuple[dict, dict]:
    """Score, chair, and archive one wave on the single review worker."""
    from cuda_sft_qgen.runtime import get_runtime

    rt = get_runtime()
    started = time.monotonic()
    logger.info("review_stage start pending=%s", len(snapshot.get("pending") or []))
    rt.reconcile_index()
    reviewing = dict(snapshot)
    reviewing.update(review_score_node(reviewing))
    reviewing.update(review_chair_node(reviewing))
    archived = archive_node(reviewing)
    logger.info("review_stage done duration_s=%.2f", time.monotonic() - started)
    return reviewing, archived


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
    generation_slots = 0
    generation_task_ids: set[str] = set()
    review = None
    repair = None
    repair_slots = 0
    repair_queue: Queue[dict] = Queue()
    last_tick = -1
    review_error_streak = 0
    circuit_open = False
    start = time.monotonic()
    deadline = float(settings.generation.get("max_runtime_seconds", 0) or 0)
    resumed_this_run: set[str] = set()
    resumed_repairs_this_run: set[str] = set()

    def refresh() -> None:
        rt.refresh_knowledge()
        counts = rt.store.status_counts()
        state.update(
            accepted_count=counts.get("verified_accepted", 0),
            rejected_count=counts.get("rejected", 0),
            pending=[q.model_dump() for q in rt.store.load_pending()],
            steering_brief=rt.store.load_steering().model_dump(),
        )

    with (ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation-stage") as producer,
          ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-stage") as reviewer,
          ThreadPoolExecutor(max_workers=1, thread_name_prefix="repair-stage") as repairer):
        while True:
            refresh()
            if generation is not None and generation.done():
                generated = generation.result()
                generation = None
                generation_slots = 0
                generation_task_ids.clear()
                for item in generated.get("salvage_items") or []:
                    repair_queue.put({"kind": "salvage", "payload": item})
                for item in generated.get("repair_items") or []:
                    repair_queue.put({"kind": "rewrite", "payload": item})
                refresh()

            if repair is not None and repair.done():
                repair.result()
                repair = None
                repair_slots = 0
                refresh()

            if review is not None and review.done():
                reviewing, archived = review.result()
                review = None
                state.update(archived)
                state["review_rounds"] = reviewing["review_rounds"]
                refresh()
                error_n = sum(
                    1 for q in (reviewing.get("last_diversity") or {}).get("final_decisions") or []
                    if str(q.get("reason") or "").startswith("review_error")
                    or str(q.get("reason") or "").startswith("review_worker_error")
                )
                review_error_streak = review_error_streak + error_n if error_n else 0
                max_streak = int(settings.review.get("circuit_breaker_error_streak", 8) or 0)
                max_exhausted = int(settings.review.get("circuit_breaker_exhausted", 20) or 0)
                exhausted_n = sum(1 for q in state["pending"] if q.get("review_exhausted"))
                if ((max_streak and review_error_streak >= max_streak)
                        or (max_exhausted and exhausted_n >= max_exhausted)):
                    logger.warning("review circuit breaker: errors=%s exhausted=%s", review_error_streak, exhausted_n)
                    circuit_open = True
                rounds = state["review_rounds"]
                every = max(1, int(settings.taxonomy_manager_cfg.get("every_k_reviews", 5)))
                if (rt.knowledge_worker is not None and not rt.verification_budget_exhausted
                        and rounds != last_tick and rounds % every == 0):
                    rt.knowledge_worker.enqueue_expansion(
                        last_facet=state.get("last_taxonomy_facet"), review_round=rounds)
                    last_tick = rounds

            target_hit = state["accepted_count"] >= settings.target_question_count
            budget_exhausted = rt.verification_budget_exhausted
            expired = (circuit_open or budget_exhausted
                       or bool(deadline and time.monotonic() - start >= deadline))
            trusted_pending = [q for q in state["pending"] if rt.store.has_real_writer_origin(q)]
            active = [q for q in trusted_pending if not q.get("review_exhausted")]
            # An admitted task keeps its review slot through generation and
            # repair. A saved pending question takes over that reservation.
            pending_ids = {q["id"] for q in trusted_pending}
            in_generation = (sum(1 for task_id in generation_task_ids
                                 if "q_" + task_id not in pending_ids)
                             if generation is not None and generation_task_ids
                             else generation_slots if generation is not None else 0)
            reserved = in_generation + repair_queue.qsize() + (repair_slots if repair is not None else 0)
            retry_room, room = pending_room(
                pending_n=len(trusted_pending) + reserved,
                active_n=len(active) + reserved,
                capacity=capacity,
                total_capacity=total_capacity,
            )
            can_produce = not target_hit and not expired and room > 0
            can_generate = (can_produce and state["iteration"] - state["run_start_iteration"]
                            < settings.max_iterations)
            if generation is None and not target_hit and not expired and retry_room > 0:
                resumed = [task for task in rt.store.unfinished_generation_tasks()
                           if task.get("task_id") not in resumed_this_run][:min(batch, retry_room)]
                briefs = []
                if resumed:
                    briefs = resumed
                    resumed_this_run.update(task["task_id"] for task in resumed)
                elif settings.retry_drain_enabled and rt.store.due_retry_count():
                    # Durable retries must not starve behind fresh work, including
                    # when the limit for new planning iterations has been reached.
                    snapshot = {**state, "pending": [], "review_batch_size": min(batch, retry_room),
                                "_repair_queue": repair_queue}
                    generation = producer.submit(copy_context().run, retry_drain_node, snapshot)
                    generation_slots = min(batch, retry_room)
                    generation_task_ids = set()
                elif can_generate and state["accepted_count"] + len(active) < settings.target_question_count:
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
                    snapshot = {**state, "batch_briefs": briefs[:max(room, retry_room)],
                                "pending": [], "_repair_queue": repair_queue}
                    generation = producer.submit(copy_context().run, generate_batch_node, snapshot)
                    generation_slots = len(briefs[:max(room, retry_room)])
                    generation_task_ids = {str(b["task_id"]) for b in briefs[:max(room, retry_room)]}
                rt.store.save_run_state({"iteration": state["iteration"],
                                         "review_rounds": state["review_rounds"],
                                         "accepted_count": state["accepted_count"]})

            now = time.time()
            ready = [q for q in trusted_pending if not q.get("review_exhausted")
                     and float(q.get("review_next_at") or 0) <= now]
            if (review is None and not target_hit and not expired and ready
                    and (len(ready) >= review_trigger or generation is None)):
                snapshot = {**state, "pending": ready[:flush_n]}
                review = reviewer.submit(copy_context().run, _review_stage, snapshot)

            if repair is None and not target_hit and not expired:
                items = []
                limit = max(1, int(settings.generation.get("repair_concurrency", 2)))
                for _ in range(limit):
                    try:
                        items.append(repair_queue.get_nowait())
                    except Empty:
                        break
                if len(items) < limit:
                    current_pending = [q for q in rt.store.load_pending()
                                       if rt.store.has_real_writer_origin(q)]
                    current_ids = {q.id for q in current_pending}
                    current_generation = (sum(1 for task_id in generation_task_ids
                                              if "q_" + task_id not in current_ids)
                                          if generation is not None and generation_task_ids
                                          else generation_slots if generation is not None else 0)
                    occupied = (len(current_pending) + current_generation
                                + repair_queue.qsize() + len(items))
                    durable_room = max(0, total_capacity - occupied)
                    for task in rt.store.unfinished_review_repairs():
                        if durable_room <= 0:
                            break
                        task_id = task["repair_task_id"]
                        if task_id in resumed_repairs_this_run:
                            continue
                        items.append({"kind": "review_repair", "payload": task})
                        resumed_repairs_this_run.add(task_id)
                        durable_room -= 1
                        if len(items) >= limit:
                            break
                if items:
                    repair = repairer.submit(copy_context().run, repair_batch_node, items)
                    repair_slots = len(items)

            if generation is None and review is None and repair is None:
                if target_hit or expired:
                    while True:
                        try:
                            item = repair_queue.get_nowait()
                        except Empty:
                            break
                        enqueue_retry(item["payload"].get("brief") or {},
                                      reason="repair_deferred", attempts=1)
                    break
                if not repair_queue.empty():
                    continue
                if ready:
                    continue
                delayed = [float(q.get("review_next_at") or 0) for q in active
                           if float(q.get("review_next_at") or 0) > now]
                if settings.retry_drain_enabled and retry_room > 0:
                    delayed.extend(float(item.get("next_ok_at") or 0)
                                   for item in rt.store.load_retry_queue()
                                   if float(item.get("next_ok_at") or 0) > now)
                if not delayed:
                    break
                # Delayed work remains runnable in this invocation. Keep the
                # coordinator alive until it is due or a run limit is hit.
                wake_in = min(delayed) - now
                if deadline:
                    wake_in = min(wake_in, max(0.0, deadline - (time.monotonic() - start)))
                time.sleep(min(0.25, max(0.0, wake_in)))
                continue
            time.sleep(0.25)

    refresh()
    state["done"] = state["accepted_count"] >= settings.target_question_count
    state["status"] = "archived" if state["done"] else (
        "verification_budget_exhausted" if rt.verification_budget_exhausted else
        "review_pending" if (state["pending"] or rt.store.unfinished_review_repairs()
                             or (settings.retry_drain_enabled and rt.store.retry_queue_size()))
        else "iteration_limit")
    elapsed = time.monotonic() - start
    get_metrics().observe("pipeline_seconds", elapsed)
    get_metrics().increment("accepted_total", state["accepted_count"])
    logger.info("pipeline_complete accepted=%s pending=%s duration_s=%.2f status=%s",
                state["accepted_count"], len(state["pending"]), elapsed, state["status"])
    return state
