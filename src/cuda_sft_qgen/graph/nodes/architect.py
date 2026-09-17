"""Architect / taxonomy-health nodes: enqueue background leaf expansion."""

from __future__ import annotations

import logging

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.observability import trace_node
from cuda_sft_qgen.runtime import get_runtime

logger = logging.getLogger(__name__)


def _sparsest_l2(taxonomy) -> tuple[dict, dict] | None:
    """Return the (domain, L2 child) pair with the fewest active leaves."""
    pairs = [(domain, child) for domain in taxonomy.domains for child in domain.get("children", [])]
    return min(pairs, key=lambda pair: sum(not item.get("retired") for item in pair[1].get("leaves", []))) if pairs else None


@trace_node("architect")
def architect_node(state: PipelineState) -> dict:
    """Optional first-run expansion: queue taxonomy work, then mark taxonomy ready."""
    rt = get_runtime()
    if not rt.taxonomy.domains:
        return {"status": "taxonomy_ready", "log": ["architect skipped: empty taxonomy"]}
    result = taxonomy_health_node(state)
    return {**result, "status": "taxonomy_ready"}


@trace_node("taxonomy_health")
def taxonomy_health_node(state: PipelineState) -> dict:
    """Enqueue durable background work without blocking question generation."""
    ticks = int(state.get("taxonomy_ticks") or 0) + 1
    try:
        rt = get_runtime()
        worker = getattr(rt, "knowledge_worker", None)
        if worker is not None:
            job_id = worker.enqueue_expansion(
                last_facet=state.get("last_taxonomy_facet"),
                review_round=int(state.get("review_rounds") or 0),
            )
            return {
                "status": "taxonomy_queued",
                "taxonomy_ticks": ticks,
                "last_taxonomy_health": {"job_id": job_id, "status": "queued"},
                "log": [f"taxonomy expansion queued job={job_id}"],
            }
        return {
            "status": "taxonomy_health_checked",
            "taxonomy_ticks": ticks,
            "log": ["taxonomy background worker unavailable"],
        }
    except Exception as exc:
        logger.warning("taxonomy enqueue failed: %s", exc)
        return {"status": "taxonomy_unavailable", "taxonomy_ticks": ticks, "log": [str(exc)]}


taxonomy_manager_node = taxonomy_health_node
