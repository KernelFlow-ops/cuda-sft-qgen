"""Integration tests for the concurrent producer/reviewer coordinator."""

from __future__ import annotations

import importlib
import threading
import time

import pytest

from cuda_sft_qgen.graph.nodes.generate import GenerateOutcome, apply_outcome
from cuda_sft_qgen.runtime import reset_runtime, set_runtime

REAL_ORIGIN = {"writer_real_api": True, "writer_provider": "openai", "writer_model": "test-model"}


def configure(runtime):
    """Shrink the fake runtime to a 1-question concurrent pipeline."""
    runtime.settings.target_question_count = 1
    runtime.settings.review_batch_size = 1
    runtime.settings.generation["min_generate_batch"] = 1
    runtime.settings.review.update(queue_capacity=1, max_pending_total=2)
    runtime.settings.generation["concurrent_pipeline"] = True


def test_total_pending_bound_stops_new_generation(fake_runtime, sample_question):
    from cuda_sft_qgen.graph.app import initial_state
    from cuda_sft_qgen.graph.pipeline import run_concurrent

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 5
    fake_runtime.store.save_pending([sample_question(id=f"exhausted{i}", review_exhausted=True,
                                                     generation_provenance=REAL_ORIGIN) for i in range(2)])
    token = set_runtime(fake_runtime)
    try:
        result = run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert result["status"] == "review_pending"
    assert result["iteration"] == 0
    assert fake_runtime.store.unfinished_generation_tasks() == []


def test_unverified_pending_does_not_block_real_writer_slot(fake_runtime, sample_question, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 1
    fake_runtime.settings.review.update(queue_capacity=1, max_pending_total=1)
    fake_runtime.store.save_pending([sample_question(id="legacy-no-origin")])
    started = threading.Event()
    monkeypatch.setattr(pipeline, "generate_batch_node", lambda state: (started.set() or {"salvage_items": []}))
    token = set_runtime(fake_runtime)
    try:
        pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert started.is_set()


def test_pending_covering_target_does_not_consume_planner_iterations(
    fake_runtime, sample_question, monkeypatch
):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 3
    fake_runtime.store.save_pending([
        sample_question(id="awaiting-review", generation_provenance=REAL_ORIGIN)
    ])

    def review(_state):
        time.sleep(0.4)
        raise RuntimeError("review probe finished")

    monkeypatch.setattr(pipeline, "_review_stage", review)
    monkeypatch.setattr(pipeline, "planner_node", lambda _state: pytest.fail(
        "planner ran while pending questions covered the target"
    ))
    token = set_runtime(fake_runtime)
    try:
        with pytest.raises(RuntimeError, match="review probe finished"):
            pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert fake_runtime.store.load_run_state().get("iteration", 0) == 0


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
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.review_batch_size = 20
    fake_runtime.settings.writer_concurrency = 4
    fake_runtime.settings.review["chair_max_workers"] = 4
    fake_runtime.settings.max_iterations = 1
    reviewed = threading.Event()
    generated = threading.Event()
    real_review_score = pipeline.review_score_node

    def generate(state):
        fake_runtime.store.save_pending([sample_question(id="overlap", generation_provenance=REAL_ORIGIN)])
        assert reviewed.wait(5), "review waited for the whole writer batch"
        generated.set()
        return {"salvage_items": []}

    def score(state):
        assert not generated.is_set()
        reviewed.set()
        return real_review_score(state)

    monkeypatch.setattr(pipeline, "generate_batch_node", generate)
    monkeypatch.setattr(pipeline, "review_score_node", score)
    token = set_runtime(fake_runtime)
    try:
        result = pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert reviewed.is_set() and generated.is_set()
    assert result["iteration"] == 1


def test_next_writer_wave_starts_during_review(fake_runtime, sample_question, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.target_question_count = 10
    fake_runtime.settings.max_iterations = 2
    fake_runtime.settings.review_batch_size = 20
    fake_runtime.settings.review["chair_max_workers"] = 4
    fake_runtime.settings.writer_concurrency = 4
    review_started = threading.Event()
    second_wave = threading.Event()
    calls = 0
    real_review = pipeline._review_stage

    def generate(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            fake_runtime.store.save_pending([sample_question(id="first-wave", generation_provenance=REAL_ORIGIN)])
            assert review_started.wait(5)
        else:
            second_wave.set()
        return {"salvage_items": []}

    def review(state):
        review_started.set()
        assert second_wave.wait(5), "next Writer wave waited for Chair"
        return real_review(state)

    monkeypatch.setattr(pipeline, "generate_batch_node", generate)
    monkeypatch.setattr(pipeline, "_review_stage", review)
    token = set_runtime(fake_runtime)
    try:
        pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert calls == 2 and second_wave.is_set()


def test_repair_starts_before_writer_batch_finishes(fake_runtime, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 1
    repair_started = threading.Event()
    generation_done = threading.Event()

    def generate(state):
        state["_repair_queue"].put({"kind": "salvage", "payload": {"brief": {}, "raw": ""}})
        assert repair_started.wait(5), "repair waited for the Writer batch"
        generation_done.set()
        return {"salvage_items": []}

    def repair(items):
        assert not generation_done.is_set()
        assert len(items) == 1
        repair_started.set()
        return {"status": "repaired"}

    monkeypatch.setattr(pipeline, "generate_batch_node", generate)
    monkeypatch.setattr(pipeline, "repair_batch_node", repair)
    token = set_runtime(fake_runtime)
    try:
        pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert repair_started.is_set() and generation_done.is_set()


def test_inflight_repair_reserves_pending_capacity(fake_runtime, sample_question, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 2
    fake_runtime.settings.review.update(queue_capacity=1, max_pending_total=1)
    calls = 0

    def generate(state):
        nonlocal calls
        calls += 1
        if calls > 1:
            pytest.fail("Writer was scheduled while repair held the only pending slot")
        state["_repair_queue"].put({"kind": "salvage", "payload": {
            "brief": state["batch_briefs"][0], "raw": "broken",
        }})
        return {"salvage_items": []}

    def repair(_items):
        time.sleep(0.4)
        assert calls == 1
        fake_runtime.store.save_pending([sample_question(id="repaired", generation_provenance=REAL_ORIGIN)])
        return {"status": "repaired"}

    monkeypatch.setattr(pipeline, "generate_batch_node", generate)
    monkeypatch.setattr(pipeline, "repair_batch_node", repair)
    token = set_runtime(fake_runtime)
    try:
        pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert calls == 1


def test_durable_repair_waits_for_new_generation_reservations(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.target_question_count = 2
    fake_runtime.settings.review_batch_size = 2
    fake_runtime.settings.max_iterations = 1
    fake_runtime.settings.review.update(queue_capacity=2, max_pending_total=2)
    fake_runtime.store.enqueue_review_repair("old-reject", {
        "brief": sample_brief().model_dump(), "source_id": "old-reject", "attempts": 1,
    })
    generation_finished = threading.Event()
    repair_started = threading.Event()

    def generate(_state):
        time.sleep(0.5)
        generation_finished.set()
        return {"salvage_items": []}

    def repair(_items):
        assert generation_finished.is_set(), "Durable repair exceeded the pending capacity"
        repair_started.set()
        return {"status": "repaired"}

    monkeypatch.setattr(pipeline, "generate_batch_node", generate)
    monkeypatch.setattr(pipeline, "repair_batch_node", repair)
    token = set_runtime(fake_runtime)
    try:
        pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert repair_started.is_set()


def test_immediately_due_pending_review_is_bounded_by_attempt_limit(
    fake_runtime, sample_question, monkeypatch
):
    """A zero-backoff Agent failure must exhaust instead of polling forever.

    Production normally uses a positive retry delay.  A zero value is useful
    in tests and is also a valid operator override, so the coordinator must
    still terminate via ``review.max_attempts`` when every Chair call returns
    pending (for example, an incomplete private draft contract).
    """
    chair = importlib.import_module("cuda_sft_qgen.graph.nodes.review_chair")

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 0
    fake_runtime.settings.review.update(max_attempts=3, retry_base_seconds=0)
    fake_runtime.store.save_pending([sample_question(id="always-pending", generation_provenance=REAL_ORIGIN)])

    def pending_decisions(items, max_workers=4):
        return {
            item["question"]["id"]: {
                "id": item["question"]["id"],
                "action": "pending",
                "reason": "draft_validation:missing_draft_target",
                "review": {
                    "action": "pending",
                    "reason": "draft_validation",
                },
            }
            for item in items
        }

    monkeypatch.setattr(chair, "_parallel_chair_decisions", pending_decisions)

    from cuda_sft_qgen.graph.app import initial_state
    from cuda_sft_qgen.graph.pipeline import run_concurrent

    token = set_runtime(fake_runtime)
    try:
        result = run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)

    pending = fake_runtime.store.load_pending()
    assert result["status"] == "review_pending"
    assert len(pending) == 1
    assert pending[0].review_attempts == 3
    assert pending[0].review_exhausted is True


def test_delayed_review_runs_after_planning_limit(fake_runtime, sample_question, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 0
    fake_runtime.settings.generation["max_runtime_seconds"] = 2
    next_at = time.time() + 0.15
    fake_runtime.store.save_pending([sample_question(
        id="delayed-review", review_next_at=next_at, generation_provenance=REAL_ORIGIN,
    )])
    monkeypatch.setattr(pipeline, "planner_node", lambda _: pytest.fail("planner exceeded iteration limit"))
    monkeypatch.setattr(pipeline, "_review_stage", lambda _: (_ for _ in ()).throw(
        RuntimeError("delayed review ran")
    ))

    token = set_runtime(fake_runtime)
    try:
        with pytest.raises(RuntimeError, match="delayed review ran"):
            pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)


def test_delayed_retry_runs_after_planning_limit(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 0
    fake_runtime.settings.generation["max_runtime_seconds"] = 2
    fake_runtime.store.enqueue_retry({
        "brief": sample_brief().model_dump(), "attempts": 1,
        "next_ok_at": time.time() + 0.15,
    })
    drained = threading.Event()

    def drain(_state):
        items = fake_runtime.store.due_retry_items(limit=1)
        assert len(items) == 1
        fake_runtime.store.complete_retry(items[0]["task_id"])
        drained.set()
        return {"salvage_items": []}

    monkeypatch.setattr(pipeline, "retry_drain_node", drain)
    monkeypatch.setattr(pipeline, "planner_node", lambda _: pytest.fail("planner exceeded iteration limit"))
    token = set_runtime(fake_runtime)
    try:
        result = pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert drained.is_set()
    assert result["iteration"] == 0


def test_delayed_review_stops_at_deadline(fake_runtime, sample_question, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 0
    fake_runtime.settings.generation["max_runtime_seconds"] = 0.05
    fake_runtime.store.save_pending([sample_question(
        id="after-deadline", review_next_at=time.time() + 1,
        generation_provenance=REAL_ORIGIN,
    )])
    monkeypatch.setattr(pipeline, "_review_stage", lambda _: pytest.fail("review ran after deadline"))
    token = set_runtime(fake_runtime)
    try:
        result = pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert result["status"] == "review_pending"
    assert result["iteration"] == 0


def test_delayed_review_stops_when_budget_is_exhausted(fake_runtime, sample_question, monkeypatch):
    from cuda_sft_qgen.graph import pipeline
    from cuda_sft_qgen.graph.app import initial_state

    configure(fake_runtime)
    fake_runtime.settings.max_iterations = 0
    fake_runtime._verification_exhausted = True
    fake_runtime.store.save_pending([sample_question(
        id="budget-exhausted", review_next_at=time.time() + 1,
        generation_provenance=REAL_ORIGIN,
    )])
    monkeypatch.setattr(pipeline, "_review_stage", lambda _: pytest.fail("review exceeded budget"))
    token = set_runtime(fake_runtime)
    try:
        result = pipeline.run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert result["status"] == "verification_budget_exhausted"
    assert result["iteration"] == 0
