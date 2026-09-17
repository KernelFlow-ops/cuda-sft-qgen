"""Tests for review-flush routing when pending is nearly a full batch."""

from __future__ import annotations

from langgraph.graph import END

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.graph.routing import route_after_generate, should_review


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


def test_fake_pipeline_reaches_target(tmp_path):
    settings = load_settings(
        fake_llm=True,
        target=4,
        review_batch_size=2,
        data_dir=tmp_path,
    )
    settings.max_iterations = 20
    result = run_pipeline(settings)
    assert result["accepted_count"] >= 4
    accepted = (tmp_path / "accepted.jsonl").read_text(encoding="utf-8").strip()
    assert accepted
    assert (tmp_path / "stats.json").exists()
