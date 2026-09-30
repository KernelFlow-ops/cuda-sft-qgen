"""Writer/evolver batch node: parallel generation over the current brief batch.

Helpers live in generate_core. This module is the graph entry for generate_batch
and re-exports those helpers so existing import paths keep working.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextvars import copy_context

from cuda_sft_qgen.errors import GeneratorError
from cuda_sft_qgen.graph.nodes.generate_core import (
    GenerateOutcome,
    _safe_generate,
    apply_outcome,
    enqueue_retry,
    generate_from_brief,
    generate_or_salvage,
    question_from_obj,
    retry_wait_sec,
    salvage_item,
)
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.observability import get_metrics, trace_node
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)


@trace_node("generate_batch")
def generate_batch_node(state: PipelineState) -> dict:
    """Run the current brief batch on bounded writer workers; collect salvage items."""
    try:
        rt = get_runtime()
        briefs = state.get("batch_briefs") or []
        steering = SteeringBrief.model_validate(state.get("steering_brief") or {})
        pending = list(state.get("pending") or [])
        if not briefs:
            return {"status": "generated_0", "pending": pending, "salvage_items": []}

        workers = max(1, min(len(briefs), rt.settings.writer_concurrency))
        queue_size = int(rt.settings.generation.get("priority_queue_size", 200))
        backpressure_threshold = float(rt.settings.generation.get("backpressure_threshold", 0.20))
        backpressure_sleep = float(rt.settings.generation.get("backpressure_sleep", 0.5))
        stagger_sec = float(rt.settings.generation.get("submit_stagger_sec", 0.8))
        if rt.settings.fake_llm:
            stagger_sec = 0.0

        logger.info(
            "generate_batch briefs=%s workers=%s keys=%s llm_per_key=%s queue_size=%s stagger=%.2fs",
            len(briefs),
            workers,
            len(rt.settings.llm_api_keys),
            rt.settings.llm_per_key_concurrency,
            queue_size,
            stagger_sec,
        )

        salvage_items: list[dict] = []
        repair_items: list[dict] = []
        repair_queue = state.get("_repair_queue")
        counts = {"ok": 0, "salvage": 0, "repair": 0, "retry": 0, "hard_fail": 0}
        lock = threading.Lock()
        anti_k = int(rt.settings.generation.get("anti_templates_max", 15))
        corpus = FewshotCorpus.from_store(rt.store, anti_k=anti_k)
        batch_t0 = time.monotonic()

        if queue_size < 1:
            raise ValueError("priority_queue_size must be positive")

        # Optional tqdm progress bar
        pbar = None
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(briefs), desc="Generating", unit="brief")
        except ImportError:
            pass

        def _one(brief_data: dict) -> None:
            t0 = time.monotonic()
            rt.store.save_generation_task(brief_data)
            outcome = generate_or_salvage(
                brief_data, steering, attempts=1, corpus=corpus,
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
                }})
            elif repair_queue is not None and kind == "repair":
                repair_queue.put({"kind": "rewrite", "payload": {
                    "brief": outcome.brief, "question": outcome.question.model_dump(),
                    "reason": outcome.reason, "attempts": outcome.attempts,
                    "steering": steering.model_dump(),
                }})
            if kind == "ok" and outcome.question is not None:
                prefetch = getattr(rt, "prefetch_embed", None)
                if callable(prefetch):
                    prefetch(outcome.question.question)
            with lock:
                counts[kind] = counts.get(kind, 0) + 1
            duration = time.monotonic() - t0
            get_metrics().observe("generate_brief_seconds", duration, {"status": kind})
            if kind == "ok" and duration >= 30:
                get_metrics().increment("generate_over_30s")
            logger.info(
                "generate_brief status=%s duration_s=%.2f l3=%s",
                kind,
                duration,
                (brief_data or {}).get("l3_id") or "",
            )
            if pbar is not None:
                pbar.update(1)

        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="writer") as executor:
                inflight = set()
                for i, brief in enumerate(briefs):
                    if len(inflight) >= workers + queue_size:
                        done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                        for fut in done:
                            try:
                                fut.result()
                            except Exception as exc:  # noqa: BLE001
                                logger.error("Task failed: %s", exc)

                    if rt.pool is not None:
                        capacity_frac = rt.pool.remaining_capacity_fraction("llm")
                        if capacity_frac < backpressure_threshold:
                            logger.debug(
                                "Backpressure triggered: llm capacity=%.1f%% < %.1f%%, sleeping %.1fs",
                                capacity_frac * 100,
                                backpressure_threshold * 100,
                                backpressure_sleep,
                            )
                            time.sleep(backpressure_sleep)
                    if i and stagger_sec > 0:
                        time.sleep(stagger_sec)

                    inflight.add(executor.submit(copy_context().run, _one, brief))

                for fut in as_completed(inflight):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Task failed: %s", exc)
        finally:
            if pbar is not None:
                pbar.close()

        elapsed = time.monotonic() - batch_t0
        rate = (counts["ok"] / elapsed * 60.0) if elapsed > 0 else 0.0
        logger.info(
            "generate_batch done ok=%s salvage=%s retry=%s fail=%s pending=%s "
            "duration_s=%.1f ok_per_min=%.2f",
            counts["ok"],
            counts["salvage"],
            counts["retry"],
            counts["hard_fail"],
            len(pending),
            elapsed,
            rate,
        )
        return {
            "pending": pending,
            "batch_briefs": [],
            "salvage_items": [] if repair_queue is not None else salvage_items,
            "repair_items": [] if repair_queue is not None else repair_items,
            "rejected_count": int(state.get("rejected_count") or 0) + counts["hard_fail"],
            "status": f"generated_{counts['ok']}",
            "log": [
                f"generate ok={counts['ok']} salvage={counts['salvage']} "
                f"retry={counts['retry']} fail={counts['hard_fail']} pending={len(pending)}"
            ],
        }
    except Exception as exc:
        logger.error("Critical error in generate_batch_node: %s", exc)
        raise GeneratorError(
            f"Generation batch failed: {exc}",
            context={"brief_count": len(state.get("batch_briefs") or [])},
            recoverable=True,
        ) from exc
