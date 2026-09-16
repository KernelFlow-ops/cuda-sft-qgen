from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.graph.nodes.architect import architect_node
from cuda_sft_qgen.graph.nodes.bootstrap import bootstrap_node
from cuda_sft_qgen.graph.nodes.generate import generate_batch_node
from cuda_sft_qgen.graph.nodes.planner import planner_node
from cuda_sft_qgen.graph.nodes.review import archive_node, review_chair_node, review_score_node
from cuda_sft_qgen.graph.nodes.salvage import retry_drain_node, salvage_node
from cuda_sft_qgen.graph.routing import (
    route_after_archive,
    route_after_bootstrap,
    route_after_drain,
    route_after_generate,
    route_after_salvage,
)
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.runtime import Runtime, build_runtime, reset_runtime, set_runtime

logger = logging.getLogger(__name__)


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("bootstrap", bootstrap_node)
    g.add_node("architect", architect_node)
    g.add_node("planner", planner_node)
    g.add_node("generate_batch", generate_batch_node)
    g.add_node("salvage", salvage_node)
    g.add_node("retry_drain", retry_drain_node)
    g.add_node("review_score", review_score_node)
    g.add_node("review_chair", review_chair_node)
    g.add_node("archive", archive_node)

    g.add_edge(START, "bootstrap")
    g.add_conditional_edges("bootstrap", route_after_bootstrap)
    g.add_edge("architect", "planner")
    g.add_edge("planner", "generate_batch")
    g.add_conditional_edges("generate_batch", route_after_generate)
    g.add_conditional_edges("salvage", route_after_salvage)
    g.add_conditional_edges("retry_drain", route_after_drain)
    g.add_edge("review_score", "review_chair")
    g.add_edge("review_chair", "archive")
    g.add_conditional_edges("archive", route_after_archive)
    return g.compile()


def initial_state(*, rebuild_taxonomy: bool = False) -> PipelineState:
    return {
        "rebuild_taxonomy": rebuild_taxonomy,
        "pending": [],
        "salvage_items": [],
        "batch_briefs": [],
        "steering_brief": {},
        "last_diversity": {},
        "log": [],
        "done": False,
        "status": "init",
    }


def run_pipeline(
    settings: Settings,
    *,
    rebuild_taxonomy: bool = False,
    runtime: Runtime | None = None,
) -> dict[str, Any]:
    own_runtime = runtime is None
    rt = runtime or build_runtime(settings)
    token = set_runtime(rt)
    app = build_graph()
    recursion_limit = max(200, settings.max_iterations * 8)
    try:
        result = app.invoke(
            initial_state(rebuild_taxonomy=rebuild_taxonomy),
            {"recursion_limit": recursion_limit},
        )
        rt.store.compute_stats()
        return result
    finally:
        reset_runtime(token)
        if own_runtime:
            closer = getattr(rt, "close", None)
            if callable(closer):
                closer()
            logger.info("pipeline finished")
