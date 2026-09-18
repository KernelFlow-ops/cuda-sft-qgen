"""Handoff schemas reject incomplete chair/archive payloads."""

from __future__ import annotations

from cuda_sft_qgen.graph.handoff import (
    ArchiveBatch,
    ChairDecisionRecord,
    brief_snapshot_from_question,
)
from cuda_sft_qgen.models import Question, SteeringBrief


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
