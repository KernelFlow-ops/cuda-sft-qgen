"""Validated handoff payloads between planner, writer, chair, and archive.

Each boundary is a Pydantic model so a downstream agent cannot silently drop
brief fields (MAST inter-agent misalignment). Nodes still persist SQLite
records; these objects are the in-memory contract.

This module does not call the LLM.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from cuda_sft_qgen.models import DupDecision, QualityReview, Question, QuestionBrief, SteeringBrief


class ScoredCandidate(BaseModel):
    """Review-score output for one pending question, including the originating brief."""

    question: Question
    decision: DupDecision
    neighbors: list[dict[str, Any]] = Field(default_factory=list)
    brief_snapshot: QuestionBrief | None = None
    knowledge_context: dict[str, Any] = Field(default_factory=dict)
    vector_row: int | None = None


class ChairDecisionRecord(BaseModel):
    """One chair outcome ready for the archivist.

    ``action`` pending means backoff/exhaust; it is never treated as accept.
    """

    id: str
    action: Literal["accept", "reject", "pending"]
    reason: str
    question: Question
    review: QualityReview | None = None
    vector_row: int | None = None


class ArchiveBatch(BaseModel):
    """Transactional review commit input: decisions plus planner steering."""

    decisions: list[ChairDecisionRecord] = Field(default_factory=list)
    steering_brief: SteeringBrief = Field(default_factory=SteeringBrief)


def brief_snapshot_from_question(question: Question) -> dict[str, Any]:
    """Rebuild the planner-facing brief fields stored on a generated question.

    Used when the original QuestionBrief is not on the scored item (resume path).
    """
    requested = question.requested_metadata or {}
    return {
        "task_id": question.id.removeprefix("q_"),
        "topic_path": list(question.topic_path),
        "l3_id": question.l3_id,
        "l3_claim": question.l3_claim,
        "question_type": requested.get("question_type") or question.question_type,
        "difficulty": requested.get("difficulty") or question.difficulty,
        "bloom": requested.get("bloom") or question.bloom,
        "persona": question.persona,
        "hardware": requested.get("hardware") or question.hardware,
        "granularity": requested.get("granularity") or question.granularity,
        "language": requested.get("language") or question.language,
        "kind": question.kind,
        "role": "evolver" if question.source == "evolver" else "writer",
        "seed_id": question.seed_id,
        "angle_id": question.angle_id,
        "claim_revision": question.claim_revision,
        "source_ids": list(question.source_ids),
    }
