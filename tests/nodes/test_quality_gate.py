"""Unit tests for chair quality-gate contract and pending-on-error behavior."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from cuda_sft_qgen.graph.nodes.generate import question_from_obj
from cuda_sft_qgen.graph.nodes.review import (
    _parallel_chair_decision, archive_node, review_chair_node, review_score_node,
)
from cuda_sft_qgen.models import AssessedMetadata, QualityChecks, QualityReview, Question, SteeringBrief
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
from cuda_sft_qgen.review.consistency import leaf_mismatch
from cuda_sft_qgen.runtime import reset_runtime, set_runtime


def review_result(question, action="accept"):
    """Minimal valid QualityReview payload for ``question``."""
    return {
        "id": question.id, "action": action, "reason": "Concrete reviewed evidence",
        "checks": {key: "pass" for key in QualityChecks.model_fields},
        "evidence": ["PRIVATE expected answer is 17"],
        "internal_reference_answer": "PRIVATE 17",
        "value_scores": {"pedagogical_value": 4, "discriminativeness": 4, "realism": 4},
        "assessed_metadata": {
            **{key: getattr(question, key) for key in AssessedMetadata.model_fields},
            "answer_format": question.answer_format or "short_answer",
            "arch_target": question.arch_target or "any",
        },
    }


@pytest.fixture
def active_runtime(fake_runtime):
    """Bind ``fake_runtime`` to the ContextVar for the duration of a test."""
    token = set_runtime(fake_runtime)
    try:
        yield fake_runtime
    finally:
        reset_runtime(token)


@pytest.mark.parametrize("malform", ["batch_schema", "unknown_field", "wrong_id", "uncertain_accept", "missing_answer"])
def test_malformed_review_never_accepts(active_runtime, sample_question, malform):
    q = sample_question()
    result = review_result(q)
    if malform == "batch_schema":
        result = {"decisions": [result]}
    elif malform == "unknown_field":
        result["extra"] = True
    elif malform == "wrong_id":
        result["id"] = "another-question"
    elif malform == "uncertain_accept":
        result["checks"]["correctness"] = "uncertain"
    else:
        result["internal_reference_answer"] = ""
    active_runtime.review_llm = MagicMock()
    active_runtime.review_llm.complete_json.return_value = result
    outcome = _parallel_chair_decision({"question": q.model_dump()}, load_prompt("chair"))
    assert outcome["action"] == "pending"


def test_full_question_claim_neighbor_and_context_reach_reviewer(active_runtime, sample_question):
    q = sample_question(question="CUDA " + "long statement " * 160 + "FINAL QUESTION?", claim_revision=2)
    neighbor = {"id": "neighbor", "question": "Complete neighboring CUDA question?"}
    active_runtime.review_llm = MagicMock()
    active_runtime.review_llm.complete_json.return_value = review_result(q)
    context = {"claims": [{"id": q.l3_id, "revision": 2}], "evidence": ["source"]}
    outcome = _parallel_chair_decision(
        {"question": q.model_dump(), "neighbors": [neighbor], "knowledge_context": context}, load_prompt("chair")
    )
    assert outcome["action"] == "accept"
    payload = json.loads(active_runtime.review_llm.complete_json.call_args.args[0][1]["content"])
    assert payload["question"] == q.question
    assert payload["l3_claim"] == q.l3_claim
    assert payload["accepted_neighbors"] == [neighbor]
    assert payload["knowledge_context"] == context


def test_unique_candidates_all_reviewed_and_retry_is_bounded(active_runtime, sample_question):
    rt = active_runtime
    rt.review_llm = MagicMock()
    rt.review_llm.complete_json.side_effect = TimeoutError("offline")
    rt.settings.review["max_attempts"] = 2
    rt.settings.review_llm_every_k_batches = 100
    questions = [sample_question(id=f"q{i}", question=f"CUDA independent question {i}?") for i in range(2)]
    scored = review_score_node({"pending": [q.model_dump() for q in questions]})
    reviewed = review_chair_node(scored)
    assert rt.review_llm.complete_json.call_count == 2
    decisions = reviewed["last_diversity"]["final_decisions"]
    assert all(d["action"] == "pending" and d["question"]["review_attempts"] == 1 for d in decisions)
    assert all(d["question"]["review_next_at"] > time.time() for d in decisions)
    committed = archive_node({**reviewed, "target": 10})
    assert committed["accepted_count"] == 0
    assert len(rt.store.load_pending()) == 2
    deferred = review_score_node(committed)
    assert deferred["last_diversity"]["scored"] == []
    due = [q.model_dump() for q in rt.store.load_pending()]
    for item in due:
        item["review_next_at"] = 0
    next_review = review_chair_node(review_score_node({"pending": due}))
    assert all(d["question"]["review_exhausted"] for d in next_review["last_diversity"]["final_decisions"])


def test_rejected_duplicate_does_not_suppress_next_candidate(active_runtime, sample_question):
    rt = active_runtime
    original = sample_question(id="original", question="An accepted baseline CUDA question?")
    rt.store.append_accepted(original)
    rt.index.add([original.id], np.array([[1.0, 0.0]], dtype=np.float32))
    rt.embedder = MagicMock()
    rt.embedder.embed.return_value = np.array([[.95, np.sqrt(1 - .95 ** 2)], [.8, .6]], dtype=np.float32)
    questions = [sample_question(id="bad", question="A semantically repeated CUDA candidate?"),
                 sample_question(id="good", question="Why do independent CUDA processes have distinct address mappings?")]
    rt.review_llm = MagicMock()
    by_id = {q.id: q for q in questions}
    rt.review_llm.complete_json.side_effect = lambda messages, **kw: review_result(by_id[json.loads(messages[1]["content"])["id"]])
    scored = review_score_node({"pending": [q.model_dump() for q in questions]})
    results = review_chair_node(scored)["last_diversity"]["final_decisions"]
    assert [(d["id"], d["action"]) for d in results] == [("bad", "reject"), ("good", "accept")]


def test_private_reference_and_evidence_stay_internal(sample_question):
    q = sample_question()
    q.review_details = QualityReview.model_validate(review_result(q)).model_dump()
    q.internal_reference_answer = "PRIVATE 17"
    assert "PRIVATE" in json.dumps(q.model_dump())
    exported = q.to_jsonl()
    assert "PRIVATE" not in json.dumps(exported)
    assert "internal_reference_answer" not in exported
    assert "checks" in exported["review_details"]


def test_quality_rejection_is_enforced_for_unique_candidate(active_runtime, sample_question):
    rt = active_runtime
    q = sample_question(question="CUDA answer is 17. What is the answer?", difficulty=4)
    response = review_result(q, "reject")
    response["checks"]["answer_leakage"] = "fail"
    rt.review_llm = MagicMock()
    rt.review_llm.complete_json.return_value = response
    result = review_chair_node(review_score_node({"pending": [q.model_dump()]}))
    decision = result["last_diversity"]["final_decisions"][0]
    assert decision["question"]["dup_verdict"] == "unique"
    assert decision["action"] == "reject"
    archive_node(result)
    assert len(rt.store.load_rejected()) == 1
    assert rt.store.load_accepted() == []


def test_assessed_labels_replace_requested_labels(active_runtime, sample_question):
    rt = active_runtime
    q = sample_question(difficulty=4, bloom="create")
    response = review_result(q)
    response["assessed_metadata"].update(difficulty=1, bloom="remember")
    rt.review_llm = MagicMock()
    rt.review_llm.complete_json.return_value = response
    result = review_chair_node(review_score_node({"pending": [q.model_dump()]}))
    revised = result["last_diversity"]["final_decisions"][0]["question"]
    assert revised["difficulty"] == 1
    assert revised["requested_metadata"]["difficulty"] == 4
    assert revised["assessed_metadata"]["bloom"] == "remember"


def test_knowledge_correction_before_commit_revokes_acceptance(active_runtime, sample_question):
    rt = active_runtime
    q = sample_question(claim_revision=1)
    q.review_details = review_result(q)
    q.internal_reference_answer = "PRIVATE"
    rt.knowledge = MagicMock()
    rt.knowledge.get_claim.return_value = {"id": q.l3_id, "revision": 2, "status": "quarantined"}
    result = archive_node({"last_diversity": {"final_decisions": [
        {"action": "accept", "reason": "passed", "question": q.model_dump(), "vector_row": None}
    ]}})
    assert result["accepted_count"] == 0
    assert result["pending"][0]["review_exhausted"] is False
    assert result["pending"][0]["review_next_at"] == 0.0
    assert result["pending"][0]["review_details"] == {}
    assert result["pending"][0]["internal_reference_answer"] is None


def test_self_declared_claim_cannot_hide_off_leaf_stem(sample_question):
    q = sample_question(question="How many lanes in this CUDA warp are active?",
                        l3_id="async_copy", l3_claim="cudaMemcpyAsync transfer semantics")
    assert leaf_mismatch(q) == "sieve_off_leaf"


def test_generation_keeps_lineage_and_passes_graph_to_writer(fake_settings, sample_brief):
    brief = sample_brief(task_id="stable-task", seed_id="seed", angle_id="angle", claim_revision=3,
                         source_ids=["source-1"], knowledge_context={"claims": [{"id": "related", "revision": 2}]})
    q = question_from_obj(
        brief,
        {
            "question": "CUDA question?",
            "tests_claim": "claim",
            "operator": "deepen",
            "draft_plan": {"analyze": "claim", "plan": "scene", "self_check": "pass"},
            "draft_target": "128B",
        },
        source="evolver",
    )
    assert q.id == "q_stable-task"
    assert (q.seed_id, q.angle_id, q.operator, q.claim_revision) == ("seed", "angle", "deepen", 3)
    assert q.knowledge_dependencies == [{"claim_id": "related", "revision": 2}]
    assert q.draft_plan["plan"] == "scene"
    assert q.draft_target == "128B"
    assert "draft_target" not in q.to_jsonl()
    assert q.requested_metadata["difficulty"] == brief.difficulty
    prompt = StructuredPrompt(fake_settings).build(brief, SteeringBrief())
    assert '"id": "related"' in prompt.user


def test_generation_missing_private_contract_is_recorded(sample_brief):
    q = question_from_obj(
        sample_brief(knowledge_context={"claims": [{"id": "claim", "revision": 3}]}),
        {"question": "CUDA question?", "tests_claim": "claim", "language": "en"},
        source="writer",
    )
    assert "missing_draft_plan" in q.draft_validation_errors
    assert "missing_draft_target" in q.draft_validation_errors
    assert "missing_verification_spec" in q.draft_validation_errors
    assert q.verification_evidence["status"] == "not_run"


def test_generation_records_mixup_parent_revisions(sample_brief):
    context = {
        "claims": [{"id": "anchor", "revision": 4}],
        "mixup": {
            "mode": "hybrid",
            "parent_ids": ["p1", "p2"],
            "parents": [
                {"id": "p1", "l3_id": "parent_a", "claim_revision": 2},
                {"id": "p2", "l3_id": "parent_b", "claim_revision": 7},
            ],
        },
    }
    q = question_from_obj(
        sample_brief(knowledge_context=context),
        {"question": "CUDA question?", "tests_claim": "claim", "language": "en"},
        source="evolver",
    )
    assert {tuple(sorted(d.items())) for d in q.knowledge_dependencies} == {
        (("claim_id", "anchor"), ("revision", 4)),
        (("claim_id", "parent_a"), ("revision", 2)),
        (("claim_id", "parent_b"), ("revision", 7)),
    }
