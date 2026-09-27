"""LangGraph TypedDict state shared by sequential pipeline nodes.

``log`` uses ``operator.add`` so node updates append rather than replace.
All other fields overwrite. This is the graph's runtime state passed between
pipeline nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    """Partial pipeline state; missing keys are treated as unset."""

    rebuild_taxonomy: bool  # If True, Architect expands sparse L2 leaves first.
    target: int  # Accepted-question target for this run.
    review_batch_size: int  # Chair batch size; also drives review flush.
    max_iterations: int  # Planner loops before forced stop.
    iteration: int  # Planner loops completed.
    run_start_iteration: int  # Lifetime iteration count at invocation start.
    review_rounds: int  # Chair batches completed (taxonomy tick cadence).
    accepted_count: int
    rejected_count: int
    pending: list[dict[str, Any]]  # Candidates waiting for or mid review.
    salvage_items: list[dict[str, Any]]  # Raw JSON that failed local parse.
    batch_briefs: list[dict[str, Any]]  # Current writer/evolver task batch.
    steering_brief: dict[str, Any]  # Chair feedback into the next planner draw.
    last_diversity: dict[str, Any]
    taxonomy_ticks: int
    last_taxonomy_health: dict
    last_taxonomy_facet: str
    done: bool
    status: str
    log: Annotated[list[str], operator.add]
