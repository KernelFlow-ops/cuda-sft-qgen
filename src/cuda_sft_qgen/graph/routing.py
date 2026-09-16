from __future__ import annotations

from langgraph.graph import END

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.runtime import get_runtime


def route_after_bootstrap(state: PipelineState) -> str:
    if state.get("rebuild_taxonomy"):
        return "architect"
    if state.get("done") and not (state.get("pending") or []):
        return END
    if state.get("done") and (state.get("pending") or []):
        return "review_score"
    if should_review(state):
        return "review_score"
    return "planner"


def should_review(state: PipelineState) -> bool:
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
    return False


def _should_salvage(state: PipelineState) -> bool:
    items = state.get("salvage_items") or []
    if not items:
        return False
    try:
        return bool(get_runtime().settings.salvage_enabled)
    except Exception:
        return True


def _should_drain(state: PipelineState) -> bool:
    try:
        rt = get_runtime()
    except Exception:
        return False
    if not rt.settings.retry_drain_enabled:
        return False
    if should_review(state):
        return False
    return rt.store.due_retry_count() > 0


def route_after_generate(state: PipelineState) -> str:
    if _should_salvage(state):
        return "salvage"
    if _should_drain(state):
        return "retry_drain"
    return _route_continue(state)


def route_after_salvage(state: PipelineState) -> str:
    if _should_drain(state):
        return "retry_drain"
    return _route_continue(state)


def route_after_drain(state: PipelineState) -> str:
    if _should_salvage(state):
        return "salvage"
    if should_taxonomy_tick(state):
        return "taxonomy_manager"
    return _route_continue(state)


def _route_continue(state: PipelineState) -> str:
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
    if should_taxonomy_tick(state):
        return "taxonomy_manager"
    return "planner"


def route_after_taxonomy(state: PipelineState) -> str:
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
