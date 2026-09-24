"""Planner node: sample the next writer/evolver brief batch from quotas + steering."""

from __future__ import annotations

import logging

from cuda_sft_qgen.errors import PlannerError
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.planner import plan_briefs
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)


@trace_node("planner")
def planner_node(state: PipelineState) -> dict:
    """Draw up to ``review_batch_size`` briefs, keeping the writer pool busy.

    Near the accepted target, the batch is padded to ``min_generate_batch`` so
    leftover holes of 1–2 items do not serialize the writer pool. Returns
    ``skip_plan`` when pending already fills the review batch.
    """
    try:
        rt = get_runtime()
        accepted_n = int(state.get("accepted_count") or 0)
        pending = [Question.model_validate(x) for x in (state.get("pending") or [])]
        target = int(state.get("target") or rt.settings.target_question_count)
        batch = int(state.get("review_batch_size") or rt.settings.review_batch_size)
        if accepted_n >= target:
            logger.info("planner target_reached accepted=%s target=%s", accepted_n, target)
            return {
                "done": True,
                "batch_briefs": [],
                "status": "target_reached",
                "iteration": int(state.get("iteration") or 0) + 1,
            }
        remaining_target = max(target - accepted_n - len(pending), 0)
        min_batch = max(1, rt.settings.min_generate_batch)
        generate_room = state.get("generate_room")
        if generate_room is not None:
            # Concurrent coordinator: keep the writer pool busy up to pending room,
            # not fill-to-review_batch_size (that 6/14 oscillation idles writers).
            room = max(0, int(generate_room))
            pool_n = max(min_batch, int(rt.settings.writer_concurrency))
            n = min(pool_n, remaining_target, room)
        else:
            fill = max(batch - len(pending), 0)
            n = min(batch, remaining_target, fill)
            # Sequential path: pad leftover holes so the writer pool stays busy.
            if 0 < n < min_batch:
                n = min_batch
        if 0 < n < min_batch and remaining_target > 0 and (
            generate_room is None or int(generate_room) >= min_batch
        ):
            n = min_batch
        fill = n
        if n <= 0:
            logger.info(
                "planner skip_plan accepted=%s pending=%s target=%s fill=%s",
                accepted_n,
                len(pending),
                target,
                fill,
            )
            return {
                "batch_briefs": [],
                "status": "skip_plan",
                "done": accepted_n >= target,
                "iteration": int(state.get("iteration") or 0) + 1,
            }

        steering = SteeringBrief.model_validate(state.get("steering_brief") or {})
        accepted = rt.store.load_accepted()
        seed_pool = accepted

        # Load adaptive planner config if enabled
        adaptive_config = rt.settings.raw_config.get("adaptive_sampling")
        rejected = None
        if adaptive_config and adaptive_config.get("enabled", False):
            rejected = rt.store.load_rejected()

        try:
            briefs = plan_briefs(
                n=n,
                taxonomy=rt.taxonomy,
                quotas=rt.settings.quotas,
                personas=rt.settings.personas,
                accepted=accepted,
                pending=pending,
                steering=steering,
                evolver_fraction=rt.settings.evolver_fraction,
                seed_pool=seed_pool,
                anti_cap=int(rt.settings.generation.get("anti_templates_max", 15)),
                adaptive_config=adaptive_config,
                rejected=rejected,
                knowledge=getattr(rt, "knowledge", None),
                embedder=getattr(rt, "embedder", None),
                mixup_config=(getattr(rt.settings, "default_yaml", {}) or {}).get("mixup"),
            )
        except Exception as exc:
            raise PlannerError(
                f"Failed to generate briefs: {exc}",
                context={
                    "n": n,
                    "accepted_count": len(accepted),
                    "pending_count": len(pending),
                },
            ) from exc

        iteration = int(state.get("iteration") or 0) + 1
        logger.info(
            "planner iteration=%s briefs=%s pending=%s accepted=%s remaining=%s",
            iteration,
            len(briefs),
            len(pending),
            accepted_n,
            remaining_target,
        )
        return {
            "batch_briefs": [b.model_dump() for b in briefs],
            "iteration": iteration,
            "status": f"planned_{len(briefs)}",
            "done": False,
            "log": [f"planner iteration={iteration} briefs={len(briefs)} pending={len(pending)}"],
        }
    except PlannerError:
        raise
    except Exception as exc:
        logger.error("Unexpected error in planner_node: %s", exc)
        raise PlannerError(
            f"Unexpected planner error: {exc}",
            recoverable=False,
        ) from exc
