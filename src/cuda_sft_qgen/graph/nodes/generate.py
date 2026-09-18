"""Writer/evolver batch node: parallel generation over the current brief batch.

Helpers live in generate_core. This module is the graph entry for generate_batch
and re-exports those helpers so existing import paths keep working.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import as_completed

from cuda_sft_qgen.errors import GeneratorError
from cuda_sft_qgen.executor import PriorityExecutor
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
from cuda_sft_qgen.keys import PRI_WRITER
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)


@trace_node("generate_batch")
def generate_batch_node(state: PipelineState) -> dict:
    """Run the current brief batch in a priority executor; collect salvage items."""
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
            len(rt.settings.nvidia_api_keys),
            rt.settings.llm_per_key_concurrency,
            queue_size,
            stagger_sec,
        )

        salvage_items: list[dict] = []
        counts = {"ok": 0, "salvage": 0, "retry": 0, "hard_fail": 0}
        lock = threading.Lock()
        anti_k = int(rt.settings.generation.get("anti_templates_max", 15))
        corpus = FewshotCorpus.from_store(rt.store, anti_k=anti_k)
        batch_t0 = time.monotonic()

        # Use PriorityExecutor instead of ThreadPoolExecutor
        executor = PriorityExecutor(max_workers=workers, max_queue_size=queue_size)

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
            outcome = generate_or_salvage(brief_data, steering, attempts=1, corpus=corpus)
            kind = apply_outcome(
                outcome, pending=pending, salvage_items=salvage_items, lock=lock
            )
            if kind == "ok" and outcome.question is not None:
                prefetch = getattr(rt, "prefetch_embed", None)
                if callable(prefetch):
                    prefetch(outcome.question.question)
            with lock:
                counts[kind] = counts.get(kind, 0) + 1
            logger.info(
                "generate_brief status=%s duration_s=%.2f l3=%s",
                kind,
                time.monotonic() - t0,
                (brief_data or {}).get("l3_id") or "",
            )
            if pbar is not None:
                pbar.update(1)

        try:
            futs = []
            for i, brief in enumerate(briefs):
                # Backpressure control: check remaining capacity before submitting
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

                # Submit with Writer priority (0)
                fut = executor.submit(PRI_WRITER, _one, brief)
                futs.append(fut)

            # Wait for all tasks to complete
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Task failed: %s", exc)
        finally:
            executor.shutdown(wait=True)
            if pbar is not None:
                pbar.close()
            rt.store.save_pending([Question.model_validate(x) for x in pending])

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
            "salvage_items": salvage_items,
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
