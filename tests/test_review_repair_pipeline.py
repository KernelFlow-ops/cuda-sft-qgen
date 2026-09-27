"""Post-Chair repair tasks are bounded, durable, and return to pending review."""

from __future__ import annotations

import threading

from cuda_sft_qgen.graph.nodes.generate_core import GenerateOutcome
from cuda_sft_qgen.graph.nodes.review_archive import archive_node, enqueue_review_repairs
from cuda_sft_qgen.graph.nodes.salvage import repair_batch_node
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.runtime import reset_runtime, set_runtime


REAL_ORIGIN = {"writer_real_api": True, "writer_provider": "openai", "writer_model": "test-model"}


def _decision(question, code="hardware"):
    question.defect_codes = [code]
    return {"action": "reject", "reason": "chair:" + code, "question": question.model_dump()}


def test_review_repair_is_bounded_and_idempotent(fake_runtime, sample_brief, sample_question):
    fake_runtime.settings.fake_llm = False
    brief = sample_brief().model_dump()
    task_id = fake_runtime.store.save_generation_task(brief)
    question = sample_question(generation_provenance={**REAL_ORIGIN, "task_id": task_id})
    decision = _decision(question)

    assert enqueue_review_repairs(fake_runtime, [decision], SteeringBrief()) == 1
    assert enqueue_review_repairs(fake_runtime, [decision], SteeringBrief()) == 0
    tasks = fake_runtime.store.unfinished_review_repairs()
    assert len(tasks) == 1
    assert tasks[0]["source_id"] == question.id
    assert tasks[0]["brief"]["task_id"] != task_id
    assert tasks[0]["attempts"] == 1

    already_repaired = sample_question(id="repaired", generation_provenance={
        **REAL_ORIGIN, "task_id": task_id, "review_repair_attempt": 1,
    })
    duplicate = sample_question(id="duplicate", generation_provenance={**REAL_ORIGIN, "task_id": task_id})
    assert enqueue_review_repairs(fake_runtime, [
        _decision(already_repaired), _decision(duplicate, "duplicate")
    ], SteeringBrief()) == 0
    assert len(fake_runtime.store.unfinished_review_repairs()) == 1


def test_archive_hands_repairable_reject_to_repair_stage(fake_runtime, sample_brief, sample_question):
    fake_runtime.settings.fake_llm = False
    brief = sample_brief().model_dump()
    task_id = fake_runtime.store.save_generation_task(brief)
    question = sample_question(generation_provenance={**REAL_ORIGIN, "task_id": task_id})
    token = set_runtime(fake_runtime)
    try:
        archive_node({
            "last_diversity": {"final_decisions": [_decision(question)]},
            "iteration": 1, "review_rounds": 1, "target": 1,
        })
    finally:
        reset_runtime(token)
    assert [q.id for q in fake_runtime.store.load_rejected()] == [question.id]
    assert len(fake_runtime.store.unfinished_review_repairs()) == 1


def test_review_repair_failure_remains_durable(fake_runtime, sample_brief, sample_question, monkeypatch):
    from cuda_sft_qgen.graph.nodes import salvage

    fake_runtime.settings.fake_llm = False
    brief = sample_brief().model_dump()
    task_id = fake_runtime.store.save_generation_task(brief)
    question = sample_question(generation_provenance={**REAL_ORIGIN, "task_id": task_id})
    enqueue_review_repairs(fake_runtime, [_decision(question)], SteeringBrief())
    task = fake_runtime.store.unfinished_review_repairs()[0]
    monkeypatch.setattr(salvage, "review_repair_item", lambda payload: GenerateOutcome(
        status="retry", brief=payload["brief"], reason="429"
    ))
    token = set_runtime(fake_runtime)
    try:
        repair_batch_node([{"kind": "review_repair", "payload": task}])
    finally:
        reset_runtime(token)
    assert len(fake_runtime.store.unfinished_review_repairs()) == 1


def test_review_repair_success_returns_to_pending(fake_runtime, sample_brief, sample_question, monkeypatch):
    from cuda_sft_qgen.graph.nodes import salvage

    fake_runtime.settings.fake_llm = False
    brief = sample_brief().model_dump()
    task_id = fake_runtime.store.save_generation_task(brief)
    source = sample_question(generation_provenance={**REAL_ORIGIN, "task_id": task_id})
    enqueue_review_repairs(fake_runtime, [_decision(source)], SteeringBrief())
    task = fake_runtime.store.unfinished_review_repairs()[0]
    repaired = sample_question(id="new_api_draft", generation_provenance={
        **REAL_ORIGIN, "review_repair_of": source.id, "review_repair_attempt": 1,
    })
    monkeypatch.setattr(salvage, "review_repair_item", lambda payload: GenerateOutcome(
        status="ok", question=repaired
    ))
    monkeypatch.setattr(fake_runtime, "prefetch_embed", lambda text: None)
    token = set_runtime(fake_runtime)
    try:
        repair_batch_node([{"kind": "review_repair", "payload": task}])
    finally:
        reset_runtime(token)
    assert fake_runtime.store.unfinished_review_repairs() == []
    assert [q.id for q in fake_runtime.store.load_pending()] == ["new_api_draft"]


def test_coordinator_resumes_durable_review_repair(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    fake_runtime.settings.max_iterations = 0
    fake_runtime.store.enqueue_review_repair("old", {
        "brief": sample_brief().model_dump(), "source_id": "old",
        "reason": "chair:hardware", "attempts": 1,
    })
    resumed = threading.Event()

    def repair(items):
        assert items[0]["kind"] == "review_repair"
        fake_runtime.store.complete_review_repair(items[0]["payload"]["repair_task_id"])
        resumed.set()
        return {"status": "repaired"}

    monkeypatch.setattr(pipeline, "repair_batch_node", repair)
    token = set_runtime(fake_runtime)
    try:
        pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert resumed.is_set()
    assert fake_runtime.store.unfinished_review_repairs() == []
