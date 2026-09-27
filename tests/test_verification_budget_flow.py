"""A run budget pauses verification without exhausting stored questions."""

from __future__ import annotations

import importlib

from cuda_sft_qgen.graph.app import initial_state
from cuda_sft_qgen.graph.nodes.bootstrap import bootstrap_node
from cuda_sft_qgen.graph.nodes.planner import planner_node
from cuda_sft_qgen.graph.pipeline import run_concurrent
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime


def test_verification_budget_flag_is_atomic(fake_runtime):
    fake_runtime.settings.review["verification_run_max_calls"] = 1
    assert fake_runtime.reserve_verification_call() is True
    assert fake_runtime.verification_budget_exhausted is True
    assert fake_runtime.reserve_verification_call() is False
    assert fake_runtime.verification_budget_exhausted is True


def test_budget_exhaustion_skips_chair_call(fake_runtime, sample_question, monkeypatch):
    chair = importlib.import_module("cuda_sft_qgen.graph.nodes.review_chair")
    q = sample_question(generation_provenance={
        "writer_real_api": True, "writer_provider": "openai", "writer_model": "test-model",
    })
    fake_runtime.settings.fake_llm = False
    fake_runtime._verification_exhausted = True
    monkeypatch.setattr(fake_runtime.llm, "complete_json", lambda *args, **kwargs: (
        (_ for _ in ()).throw(AssertionError("Chair must not run after budget exhaustion"))
    ))
    token = set_runtime(fake_runtime)
    try:
        result = chair._parallel_chair_decision({"question": q.model_dump()}, "unused")
    finally:
        reset_runtime(token)
    assert result["action"] == "pending"
    assert result["reason"] == "verification_run_budget_exhausted"


def test_budget_deferred_candidate_keeps_review_attempts(fake_runtime, sample_question, monkeypatch):
    chair = importlib.import_module("cuda_sft_qgen.graph.nodes.review_chair")
    q = sample_question(review_attempts=2)
    fake_runtime.settings.review["max_attempts"] = 2
    monkeypatch.setattr(chair, "_parallel_chair_decisions", lambda items, max_workers=4: {
        q.id: {"id": q.id, "action": "pending", "reason": "verification_run_budget_exhausted"}
    })
    token = set_runtime(fake_runtime)
    try:
        result = chair.review_chair_node({
            "last_diversity": {
                "report": {"accepted_count": 0, "pending_count": 1},
                "scored": [{"question": q.model_dump(), "decision": {}, "vector_row": None}],
            },
            "review_rounds": 3,
        })
    finally:
        reset_runtime(token)
    decision = result["last_diversity"]["final_decisions"][0]
    assert decision["action"] == "pending"
    assert decision["question"]["review_attempts"] == 2
    assert decision["question"]["review_exhausted"] is False
    assert decision["question"]["review_next_at"] == 0
    assert result["review_rounds"] == 3


def test_budget_exhaustion_stops_coordinator_with_explicit_status(fake_runtime, sample_question):
    fake_runtime.settings.target_question_count = 1
    fake_runtime.settings.review_batch_size = 1
    fake_runtime._verification_exhausted = True
    fake_runtime.store.save_pending([sample_question(generation_provenance={
        "writer_real_api": True, "writer_provider": "openai", "writer_model": "test-model",
    })])
    token = set_runtime(fake_runtime)
    try:
        result = run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
    assert result["status"] == "verification_budget_exhausted"
    assert fake_runtime.store.load_pending()[0].review_attempts == 0


def test_empty_chair_wave_does_not_increment_rounds(fake_runtime):
    chair = importlib.import_module("cuda_sft_qgen.graph.nodes.review_chair")
    token = set_runtime(fake_runtime)
    try:
        result = chair.review_chair_node({"last_diversity": {}, "review_rounds": 3})
    finally:
        reset_runtime(token)
    assert result["review_rounds"] == 3


def test_new_runtime_resumes_pending_with_fresh_budget(fake_runtime, sample_question):
    settings = fake_runtime.settings
    settings.target_question_count = 1
    settings.review_batch_size = 1
    settings.max_iterations = 1
    fake_runtime.store.save_run_state({"iteration": 40, "review_rounds": 2})
    fake_runtime.store.save_pending([sample_question(generation_provenance={
        "writer_real_api": True, "writer_provider": "openai", "writer_model": "test-model",
    })])
    fake_runtime._verification_exhausted = True
    token = set_runtime(fake_runtime)
    try:
        paused = run_concurrent(fake_runtime, initial_state())
    finally:
        reset_runtime(token)
        fake_runtime.close()
    assert paused["status"] == "verification_budget_exhausted"

    resumed_runtime = build_runtime(settings)
    token = set_runtime(resumed_runtime)
    try:
        assert resumed_runtime.verification_budget_exhausted is False
        resumed = run_concurrent(resumed_runtime, initial_state())
        assert resumed["accepted_count"] == 1
        assert resumed["iteration"] == 40
        assert resumed["run_start_iteration"] == 40
        settings.target_question_count = 2
        fresh = bootstrap_node(initial_state())
        planned = planner_node(fresh)
        assert planned["iteration"] == 41
        assert len(planned["batch_briefs"]) == 1
    finally:
        reset_runtime(token)
        resumed_runtime.close()
