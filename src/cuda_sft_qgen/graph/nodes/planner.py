from __future__ import annotations

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.planner import plan_briefs
from cuda_sft_qgen.runtime import get_runtime


def planner_node(state: PipelineState) -> dict:
    rt = get_runtime()
    accepted_n = int(state.get("accepted_count") or 0)
    pending = [Question.model_validate(x) for x in (state.get("pending") or [])]
    target = int(state.get("target") or rt.settings.target_question_count)
    batch = int(state.get("review_batch_size") or rt.settings.review_batch_size)
    if accepted_n >= target:
        return {
            "done": True,
            "batch_briefs": [],
            "status": "target_reached",
            "iteration": int(state.get("iteration") or 0) + 1,
        }
    remaining_target = max(target - accepted_n - len(pending), 0)
    fill = max(batch - len(pending), 0)
    n = min(batch, remaining_target, fill)
    if n <= 0:
        return {
            "batch_briefs": [],
            "status": "skip_plan",
            "done": accepted_n >= target,
            "iteration": int(state.get("iteration") or 0) + 1,
        }

    steering = SteeringBrief.model_validate(state.get("steering_brief") or {})
    accepted = rt.store.load_accepted()
    seed_pool = accepted[-50:] if accepted else []
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
    )
    iteration = int(state.get("iteration") or 0) + 1
    return {
        "batch_briefs": [b.model_dump() for b in briefs],
        "iteration": iteration,
        "status": f"planned_{len(briefs)}",
        "done": False,
        "log": [f"planner iteration={iteration} briefs={len(briefs)} pending={len(pending)}"],
    }
