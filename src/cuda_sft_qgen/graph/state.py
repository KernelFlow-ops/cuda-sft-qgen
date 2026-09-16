from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    rebuild_taxonomy: bool
    target: int
    review_batch_size: int
    max_iterations: int
    iteration: int
    review_rounds: int
    accepted_count: int
    rejected_count: int
    pending: list[dict[str, Any]]
    salvage_items: list[dict[str, Any]]
    batch_briefs: list[dict[str, Any]]
    steering_brief: dict[str, Any]
    last_diversity: dict[str, Any]
    taxonomy_ticks: int
    last_taxonomy_health: dict
    last_taxonomy_facet: str
    done: bool
    status: str
    log: Annotated[list[str], operator.add]
