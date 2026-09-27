"""Tests for review-flush routing when pending is nearly a full batch."""

from __future__ import annotations

from langgraph.graph import END

from cuda_sft_qgen.graph.routing import (
    route_after_archive, route_after_bootstrap, route_after_generate, should_review,
)


def test_should_review_when_remaining_below_min_batch(fake_runtime):
    from cuda_sft_qgen.runtime import set_runtime

    set_runtime(fake_runtime)
    fake_runtime.settings.writer_concurrency = 6
    state = {
        "pending": [{}] * 2,
        "accepted_count": 0,
        "target": 4,
        "review_batch_size": 20,
        "iteration": 1,
        "max_iterations": 200,
    }
    assert should_review(state) is True
    assert route_after_generate(state) == "review_score"


def test_should_review_nearly_full_pending_with_runtime(fake_runtime):
    from cuda_sft_qgen.runtime import set_runtime

    set_runtime(fake_runtime)
    fake_runtime.settings.writer_concurrency = 6
    state = {
        "pending": [{}] * 16,
        "accepted_count": 0,
        "target": 1000,
        "review_batch_size": 20,
        "iteration": 4,
        "max_iterations": 200,
    }
    assert should_review(state) is True
    assert route_after_generate(state) == "review_score"


def test_should_not_review_below_batch_size():
    state = {
        "pending": [{}] * 5,
        "accepted_count": 0,
        "target": 1000,
        "review_batch_size": 20,
        "iteration": 1,
        "max_iterations": 200,
    }
    assert should_review(state) is False
    assert route_after_generate(state) == "planner"


def test_should_review_when_pending_reaches_n():
    state = {
        "pending": [{}] * 20,
        "accepted_count": 0,
        "target": 1000,
        "review_batch_size": 20,
        "iteration": 1,
        "max_iterations": 200,
    }
    assert should_review(state) is True
    assert route_after_generate(state) == "review_score"


def test_tail_batch_reviews_even_if_smaller_than_n():
    state = {
        "pending": [{}] * 3,
        "accepted_count": 18,
        "target": 20,
        "review_batch_size": 20,
        "iteration": 3,
        "max_iterations": 200,
    }
    assert should_review(state) is True


def test_done_without_pending_ends():
    state = {
        "pending": [],
        "accepted_count": 20,
        "target": 20,
        "review_batch_size": 20,
        "iteration": 4,
        "max_iterations": 200,
        "done": True,
    }
    assert route_after_generate(state) == END


def test_deferred_pending_stops_sequential_run_without_empty_review():
    state = {
        "pending": [{"review_next_at": 10**12, "review_exhausted": False}],
        "accepted_count": 0, "target": 1, "review_batch_size": 1,
        "iteration": 3, "run_start_iteration": 0, "max_iterations": 40,
    }
    assert should_review(state) is False
    assert route_after_generate(state) == END
    assert route_after_archive(state) == END


def test_iteration_limit_is_per_invocation():
    state = {
        "pending": [], "accepted_count": 0, "target": 10,
        "review_batch_size": 2, "iteration": 41,
        "run_start_iteration": 40, "max_iterations": 2,
    }
    assert route_after_generate(state) == "planner"
    state["iteration"] = 42
    assert route_after_generate(state) == END


def test_one_due_question_reviews_when_other_pending_fills_batch():
    state = {
        "pending": [{"review_next_at": 10**12}] * 19 + [{"review_next_at": 0}],
        "accepted_count": 0, "target": 1000, "review_batch_size": 20,
        "iteration": 1, "max_iterations": 40,
    }
    assert should_review(state) is True
    assert route_after_generate(state) == "review_score"


def test_completed_target_does_not_review_leftover_pending():
    state = {
        "pending": [{"review_next_at": 0}], "accepted_count": 5,
        "target": 5, "done": True, "rebuild_taxonomy": True,
    }
    assert route_after_bootstrap(state) == END
    assert route_after_generate(state) == END
