"""Validated handoff payloads between planner, writer, chair, and archive.

Each boundary is a Pydantic model so a downstream agent cannot silently drop
brief fields (MAST inter-agent misalignment). Nodes still persist SQLite
records; these objects are the in-memory contract.

This module does not call the LLM.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from cuda_sft_qgen.models import (
    DupDecision,
    PairPlan,
    QualityReview,
    Question,
    QuestionBrief,
    SeedAnalysis,
    SteeringBrief,
    VerificationEvidence,
    VerificationSpec,
)


class GenerationDraft(BaseModel):
    """Private Writer self-check carried alongside a candidate.

    ``plan`` contains the structured Analyze/Plan/Construct/Self-check notes;
    ``target`` is the Writer's provisional answer and is never an export field.
    """

    plan: dict[str, Any] = Field(default_factory=dict)
    target: str | None = None

    def seed_analysis(self) -> SeedAnalysis | None:
        value = self.plan.get("seed_analysis")
        return SeedAnalysis.model_validate(value) if isinstance(value, dict) else None

    def pair_plan(self) -> PairPlan | None:
        value = self.plan.get("pair_plan")
        return PairPlan.model_validate(value) if isinstance(value, dict) else None


class ScoredCandidate(BaseModel):
    """Review-score output for one pending question, including the originating brief."""

    question: Question
    decision: DupDecision
    neighbors: list[dict[str, Any]] = Field(default_factory=list)
    brief_snapshot: QuestionBrief | None = None
    generation_draft: GenerationDraft | None = None
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
    if question.brief_snapshot:
        # Preserve the exact planner assignment on resume, while excluding
        # writer answers/spec source that could contaminate the Chair prompt.
        snapshot = dict(question.brief_snapshot)
        snapshot["draft_validation_errors"] = list(question.draft_validation_errors)
        return snapshot
    snapshot = {
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
        "mixup_mode": question.mixup_mode,
        "parent_ids": list(question.parent_ids),
        "parent_difficulties": list(question.parent_difficulties),
        "difficulty_target": question.difficulty_target,
        "mixup_context": dict(question.mixup_context or {}),
        "generation_provenance": {
            key: value for key, value in (question.generation_provenance or {}).items()
            if key not in {"draft_target", "verification_spec", "verification_evidence", "rubric", "answer"}
        },
        "draft_validation_errors": list(question.draft_validation_errors),
    }
    return snapshot


def generation_draft_from_question(question: Question) -> GenerationDraft | None:
    """Validate the Writer's private draft before a downstream handoff."""
    if not question.draft_plan and not question.draft_target:
        return None
    return GenerationDraft(plan=dict(question.draft_plan or {}), target=question.draft_target)


def verification_from_question(question: Question) -> tuple[VerificationSpec | None, VerificationEvidence | None]:
    """Validate private verification artifacts at the handoff boundary."""
    spec = VerificationSpec.model_validate(question.verification_spec) if question.verification_spec else None
    evidence = VerificationEvidence.model_validate(question.verification_evidence) if question.verification_evidence else None
    return spec, evidence
