"""Bootstrap node: hydrate pipeline state from the on-disk archive."""

from __future__ import annotations

import logging

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)


@trace_node("bootstrap")
def bootstrap_node(state: PipelineState) -> dict:
    """Load accepted/pending/rejected counts, steering, and resume counters."""
    rt = get_runtime()
    accepted = rt.store.load_accepted()
    pending = rt.store.load_pending()
    rejected = rt.store.load_rejected()
    steering = rt.store.load_steering()
    run = rt.store.load_run_state()
    iteration = int(run.get("iteration") or 0)
    review_rounds = int(run.get("review_rounds") or 0)
    logger.info(
        "bootstrap accepted=%s pending=%s rejected=%s target=%s iteration=%s",
        len(accepted),
        len(pending),
        len(rejected),
        rt.settings.target_question_count,
        iteration,
    )
    return {
        "target": rt.settings.target_question_count,
        "review_batch_size": rt.settings.review_batch_size,
        "max_iterations": rt.settings.max_iterations,
        "iteration": iteration,
        "review_rounds": review_rounds,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "pending": [q.model_dump() for q in pending],
        "steering_brief": steering.model_dump(),
        "batch_briefs": [],
        "done": len(accepted) >= rt.settings.target_question_count,
        "status": "bootstrapped",
        "log": [
            f"bootstrap accepted={len(accepted)} pending={len(pending)} "
            f"rejected={len(rejected)} target={rt.settings.target_question_count}"
        ],
    }
