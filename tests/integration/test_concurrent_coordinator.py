"""Integration tests for the concurrent producer/reviewer coordinator."""

from __future__ import annotations

import threading

from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.graph.nodes.generate import GenerateOutcome, apply_outcome
from cuda_sft_qgen.runtime import reset_runtime, set_runtime


def configure(runtime):
    """Shrink the fake runtime to a 1-question concurrent pipeline."""
    runtime.settings.target_question_count = 1
    runtime.settings.review_batch_size = 1
    runtime.settings.generation["min_generate_batch"] = 1
    runtime.settings.review.update(queue_capacity=1, max_pending_total=2)
    runtime.settings.generation["concurrent_pipeline"] = True


def test_due_retry_runs_after_new_generation_iteration_limit(fake_runtime, sample_brief):
    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 0
    fake_runtime.store.enqueue_retry({"brief": sample_brief().model_dump(),
                                     "attempts": 1, "reason": "429", "next_ok_at": 0})
    result = run_pipeline(fake_runtime.settings, runtime=fake_runtime)
    assert result["accepted_count"] == 1
    assert fake_runtime.store.due_retry_count() == 0


def test_unfinished_generation_restores_after_iteration_limit(fake_runtime, sample_brief):
    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 0
    brief = sample_brief().model_dump()
    task_id = fake_runtime.store.save_generation_task(brief)
    result = run_pipeline(fake_runtime.settings, runtime=fake_runtime)
    assert result["accepted_count"] == 1
    assert fake_runtime.store.load_accepted()[0].id == "q_" + task_id
    assert fake_runtime.store.unfinished_generation_tasks() == []


def test_exhausted_review_does_not_consume_active_queue(fake_runtime, sample_question):
    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 2
    fake_runtime.store.save_pending([sample_question(id="exhausted", review_exhausted=True)])
    result = run_pipeline(fake_runtime.settings, runtime=fake_runtime)
    assert result["accepted_count"] == 1
    assert [q.id for q in fake_runtime.store.load_pending()] == ["exhausted"]


def test_total_pending_bound_stops_new_generation(fake_runtime, sample_question):
    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 5
    fake_runtime.store.save_pending([sample_question(id=f"exhausted{i}", review_exhausted=True) for i in range(2)])
    result = run_pipeline(fake_runtime.settings, runtime=fake_runtime)
    assert result["status"] == "review_pending"
    assert result["iteration"] == 0
    assert fake_runtime.store.unfinished_generation_tasks() == []


def test_salvage_task_remains_recoverable_until_durable_outcome(fake_runtime, sample_brief, sample_question):
    brief = sample_brief().model_dump()
    task_id = fake_runtime.store.save_generation_task(brief)
    token = set_runtime(fake_runtime)
    try:
        pending, salvage = [], []
        apply_outcome(GenerateOutcome(status="salvage", brief=brief, raw="broken JSON"),
                      pending=pending, salvage_items=salvage, lock=threading.Lock())
        assert fake_runtime.store.unfinished_generation_tasks()[0]["task_id"] == task_id
        q = sample_question(id="q_" + task_id)
        apply_outcome(GenerateOutcome(status="ok", question=q),
                      pending=pending, salvage_items=salvage, lock=threading.Lock())
        assert fake_runtime.store.unfinished_generation_tasks() == []
        assert fake_runtime.store.load_pending()[0].id == q.id
    finally:
        reset_runtime(token)


def test_review_starts_before_writer_batch_finishes(fake_runtime, sample_question, monkeypatch):
    from cuda_sft_qgen.graph import pipeline

    configure(fake_runtime)
    fake_runtime.settings.review_batch_size = 8
    fake_runtime.settings.review["chair_max_workers"] = 1
    fake_runtime.settings.max_iterations = 1
    reviewed = threading.Event()
    generated = threading.Event()
    real_review_score = pipeline.review_score_node

    def generate(state):
        fake_runtime.store.save_pending([sample_question(id="overlap")])
        assert reviewed.wait(5), "review waited for the whole writer batch"
        generated.set()
        return {"salvage_items": []}

    def score(state):
        assert not generated.is_set()
        reviewed.set()
        return real_review_score(state)

    monkeypatch.setattr(pipeline, "generate_batch_node", generate)
    monkeypatch.setattr(pipeline, "review_score_node", score)
    result = run_pipeline(fake_runtime.settings, runtime=fake_runtime)
    assert reviewed.is_set() and generated.is_set()
    assert result["accepted_count"] == 1
