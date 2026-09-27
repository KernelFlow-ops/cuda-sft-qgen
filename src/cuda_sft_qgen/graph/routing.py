"""Conditional edges after bootstrap, generate, salvage, archive, and taxonomy.

Review is triggered when pending fills a batch, the remaining hole is smaller
than a concurrent generate wave, or the run is finishing. JSON salvage and
retry drain yield to fresh writer work so 429 retries do not stall throughput.

Predicates live in ``policy.py`` so the concurrent coordinator cannot drift.
"""

from __future__ import annotations

from langgraph.graph import END

from cuda_sft_qgen.graph.policy import (
    cannot_plan_more,
    iteration_limit_reached,
    should_drain,
    should_review,
    should_salvage,
    should_taxonomy_tick,
)
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.runtime import get_runtime

# Re-export for tests that imported these names from routing.
__all__ = [
    "route_after_archive",
    "route_after_bootstrap",
    "route_after_drain",
    "route_after_generate",
    "route_after_salvage",
    "route_after_taxonomy",
    "should_review",
    "should_taxonomy_tick",
]


def route_after_bootstrap(state: PipelineState) -> str:
    """Architect if rebuilding taxonomy; else review leftover pending or plan."""
    if state.get("done") or int(state.get("accepted_count") or 0) >= int(state.get("target") or 0):
        return END
    if state.get("rebuild_taxonomy"):
        return "architect"
    if should_review(state):
        return "review_score"
    if cannot_plan_more(state):
        return END
    return "planner"


def route_after_generate(state: PipelineState) -> str:
    """Salvage broken JSON first, else drain retries, else review/plan/end."""
    if should_salvage(state):
        return "salvage"
    if should_drain(state):
        return "retry_drain"
    return _route_continue(state)


def route_after_salvage(state: PipelineState) -> str:
    """Drain due retries if idle; otherwise continue to review or planner."""
    if should_drain(state):
        return "retry_drain"
    return _route_continue(state)


def route_after_drain(state: PipelineState) -> str:
    """Salvage any drain-produced JSON, else taxonomy tick, else continue."""
    if should_salvage(state):
        return "salvage"
    if should_taxonomy_tick(state):
        return "taxonomy_manager"
    return _route_continue(state)


def _route_continue(state: PipelineState) -> str:
    """Shared continue: review if due, else END on target/iter, else planner."""
    accepted = int(state.get("accepted_count") or 0)
    target = int(state.get("target") or 0)
    pending = state.get("pending") or []
    if state.get("done") and not pending:
        return END
    if accepted >= target and not pending:
        return END
    if accepted >= target:
        return END
    if should_review(state):
        return "review_score"
    if cannot_plan_more(state):
        return END
    return "planner"


def _verification_budget_exhausted() -> bool:
    try:
        return get_runtime().verification_budget_exhausted is True
    except RuntimeError:
        return False


def route_after_archive(state: PipelineState) -> str:
    """END on target or max iterations; else taxonomy tick or next planner batch."""
    accepted = int(state.get("accepted_count") or 0)
    target = int(state.get("target") or 0)
    if accepted >= target:
        return END
    if _verification_budget_exhausted():
        return END
    if should_review(state):
        return "review_score"
    if iteration_limit_reached(state) or cannot_plan_more(state):
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
    if accepted >= target:
        return END
    if _verification_budget_exhausted():
        return END
    if should_review(state):
        return "review_score"
    if iteration_limit_reached(state) or cannot_plan_more(state):
        return END
    if should_drain(state):
        return "retry_drain"
    return "planner"
