"""Handoff schemas reject incomplete chair/archive payloads."""

from __future__ import annotations

import pytest

from cuda_sft_qgen.graph.handoff import (
    ArchiveBatch,
    ChairDecisionRecord,
    GenerationDraft,
    brief_snapshot_from_question,
    generation_draft_from_question,
)
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.models import VerificationSpec


def _q(**overrides) -> Question:
    data = {
        "id": "q_abc123",
        "question": "What is CUDA occupancy?",
        "tests_claim": "Occupancy is active warps over max warps.",
        "topic_path": ["compute", "occupancy"],
        "l3_id": "cuda_occupancy_001",
        "question_type": "conceptual",
        "difficulty": 2,
        "bloom": "understand",
        "persona": "cuda_interviewer",
        "hardware": "hopper",
        "granularity": "kernel",
        "language": "zh",
        "source": "writer",
    }
    data.update(overrides)
    return Question.model_validate(data)


def test_brief_snapshot_round_trips_requested_metadata():
    q = _q(
        source="evolver",
        seed_id="q_seed",
        claim_revision=2,
        requested_metadata={
            "question_type": "debug",
            "difficulty": 3,
            "bloom": "analyze",
            "hardware": "hopper",
            "granularity": "kernel",
            "language": "zh",
        },
    )
    snap = brief_snapshot_from_question(q)
    assert snap["task_id"] == "abc123"
    assert snap["role"] == "evolver"
    assert snap["question_type"] == "debug"
    assert snap["claim_revision"] == 2


def test_archive_batch_accepts_pending_without_review_model():
    q = _q()
    batch = ArchiveBatch(
        decisions=[
            ChairDecisionRecord(
                id=q.id, action="pending", reason="review_error:Timeout", question=q
            )
        ],
        steering_brief=SteeringBrief(),
    )
    assert batch.decisions[0].action == "pending"
    assert batch.decisions[0].review is None


def test_generation_draft_is_validated_and_private():
    q = _q(draft_plan={"analyze": "claim", "plan": "numeric scene"}, draft_target="128B")
    draft = generation_draft_from_question(q)
    assert isinstance(draft, GenerationDraft)
    assert draft.plan["plan"] == "numeric scene"
    assert draft.target == "128B"
    assert "draft_plan" not in q.to_jsonl()
    assert "draft_target" not in q.to_jsonl()


def test_verification_spec_requires_compile_contract():
    with pytest.raises(Exception):
        VerificationSpec.model_validate({
            "mode": "compile", "answer_type": "enum", "rubric": ["legal"],
            "code": "int main() {}",
        })


def test_private_verification_fields_are_not_exported():
    q = _q(
        verification_spec={
            "mode": "scalar", "answer_type": "number",
            "expected_behavior": "exact scalar", "rubric": ["exact"],
        },
        verification_evidence={"status": "not_run"},
        generation_provenance={"seed_domain_id": "T08"},
        draft_validation_errors=["missing_draft_target"],
    )
    exported = q.to_jsonl()
    assert "verification_spec" not in exported
    assert "verification_evidence" not in exported
    assert "generation_provenance" not in exported
    assert "draft_validation_errors" not in exported
