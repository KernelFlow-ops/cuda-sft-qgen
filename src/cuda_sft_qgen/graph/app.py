"""Compile the LangGraph pipeline and run it under a data-dir lock.

Prefers the concurrent producer/reviewer coordinator when
``generation.concurrent_pipeline`` is true (the default). Sequential
``StateGraph.invoke`` remains available for debugging.
"""

from __future__ import annotations

import logging
import fcntl
from typing import Any

from langgraph.graph import END, START, StateGraph

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.graph.nodes.architect import architect_node, taxonomy_manager_node
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
    route_after_taxonomy,
)
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.runtime import Runtime, build_runtime, reset_runtime, set_runtime

logger = logging.getLogger(__name__)


def build_graph():
    """Compile the sequential LangGraph: bootstrap → plan → generate → review."""
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
    g.add_node("taxonomy_manager", taxonomy_manager_node)

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
    g.add_conditional_edges("taxonomy_manager", route_after_taxonomy)
    return g.compile()


def initial_state(*, rebuild_taxonomy: bool = False) -> PipelineState:
    """Empty run state; counters and paths are filled by bootstrap."""
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
    """Run generate+review until target, iteration limit, or pending-only stop.

    Acquires ``data_dir/pipeline.lock`` so two processes cannot share one
    archive. Starts the knowledge worker when knowledge and taxonomy manager
    are enabled.

    Args:
        settings: Loaded runtime settings.
        rebuild_taxonomy: If True, Architect expands sparse L2 leaves first.
        runtime: Optional pre-built runtime (tests); otherwise constructed here.

    Returns:
        Final pipeline state dict with accepted/rejected/pending counts.

    Raises:
        RuntimeError: If another pipeline already holds the data-dir lock.
    """
    own_runtime = runtime is None
    rt = runtime or build_runtime(settings)
    try:
        run_lock = (settings.data_dir / "pipeline.lock").open("a")
    except Exception:
        if own_runtime:
            rt.close()
        raise
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        if own_runtime:
            rt.close()
        raise RuntimeError(f"Another generation pipeline is using {settings.data_dir}") from None
    token = set_runtime(rt)
    # Each planner loop can take planner + generate + salvage + drain +
    # review_score + chair + archive + taxonomy_manager (~8-10 nodes).
    recursion_limit = max(200, settings.max_iterations * 12)
    try:
        rt.refresh_knowledge(force=True)
        rt.reconcile_index()
        if (not settings.fake_llm and settings.knowledge_cfg.get("enabled", True)
                and settings.taxonomy_manager_cfg.get("enabled", True)):
            from cuda_sft_qgen.knowledge_worker import KnowledgeWorker
            rt.knowledge_worker = KnowledgeWorker(
                rt.knowledge, rt.llm, embedder=rt.embedder, settings=settings,
                taxonomy=rt.taxonomy, archive=rt.store,
            )
            rt.knowledge_worker.start()
        if settings.generation.get("concurrent_pipeline", True):
            from cuda_sft_qgen.graph.pipeline import run_concurrent
            result = run_concurrent(rt, initial_state(rebuild_taxonomy=rebuild_taxonomy))
        else:
            result = build_graph().invoke(
                initial_state(rebuild_taxonomy=rebuild_taxonomy),
                {"recursion_limit": recursion_limit},
            )
        rt.refresh_knowledge(force=True)
        result["accepted_count"] = len(rt.store.load_accepted())
        result["rejected_count"] = len(rt.store.load_rejected())
        result["pending"] = [q.model_dump() for q in rt.store.load_pending()]
        result["done"] = result["accepted_count"] >= settings.target_question_count
        if not result["done"] and result["pending"]:
            result["status"] = "review_pending"
        rt.store.compute_stats()
        return result
    finally:
        if rt.knowledge_worker is not None:
            rt.knowledge_worker.stop()
        reset_runtime(token)
        fcntl.flock(run_lock, fcntl.LOCK_UN)
        run_lock.close()
        if own_runtime:
            closer = getattr(rt, "close", None)
            if callable(closer):
                closer()
            logger.info("pipeline finished")
