from __future__ import annotations

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.graph.nodes.architect import taxonomy_manager_node
from cuda_sft_qgen.graph.routing import (
    route_after_archive,
    route_after_generate,
    should_review,
    should_taxonomy_tick,
)
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime


def test_should_taxonomy_tick_false_without_runtime():
    assert (
        should_taxonomy_tick(
            {"review_rounds": 5, "accepted_count": 1, "target": 100}
        )
        is False
    )


def test_should_taxonomy_tick_respects_fake_and_every_k(tmp_path):
    settings = load_settings(fake_llm=True, data_dir=tmp_path, target=100)
    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        base = {"accepted_count": 1, "target": 100, "review_rounds": 5}
        assert should_taxonomy_tick(base) is False
        rt.settings.fake_llm = False
        assert should_taxonomy_tick(base) is True
        assert should_taxonomy_tick({**base, "review_rounds": 4}) is False
        assert should_taxonomy_tick({**base, "review_rounds": 10}) is True
        assert should_taxonomy_tick({**base, "review_rounds": 0}) is False
        assert should_taxonomy_tick({**base, "accepted_count": 100}) is False
        rt.settings.taxonomy_manager_cfg["enabled"] = False
        assert should_taxonomy_tick(base) is False
    finally:
        reset_runtime(token)


def test_route_after_archive_skips_taxonomy_when_fake():
    state = {
        "accepted_count": 1,
        "target": 100,
        "iteration": 1,
        "max_iterations": 200,
        "review_rounds": 5,
        "pending": [],
    }
    assert route_after_archive(state) == "planner"


def test_taxonomy_node_does_not_rewrite_pending_or_steering(tmp_path):
    settings = load_settings(fake_llm=True, data_dir=tmp_path, target=8)
    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        pending = [{"id": "q1"}]
        steering = {"underrepresented": ["T07.layout_algebra"]}
        out = taxonomy_manager_node(
            {
                "pending": pending,
                "steering_brief": steering,
                "last_diversity": {},
                "taxonomy_ticks": 0,
            }
        )
        assert "pending" not in out
        assert "steering_brief" not in out
        assert out["status"] == "taxonomy_health_checked"
    finally:
        reset_runtime(token)


def test_generate_routing_unchanged():
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


def test_manage_taxonomy_parser_has_tick():
    from cuda_sft_qgen.cli import build_parser

    args = build_parser().parse_args(["manage-taxonomy", "--tick", "--fake", "--max-calls", "2"])
    assert args.tick is True
    assert args.fake is True
    assert args.max_calls == 2


def test_fake_pipeline_still_reaches_target(tmp_path):
    settings = load_settings(
        fake_llm=True,
        target=4,
        review_batch_size=2,
        data_dir=tmp_path,
    )
    settings.max_iterations = 20
    result = run_pipeline(settings)
    assert result["accepted_count"] >= 4
    assert int(result.get("taxonomy_ticks") or 0) == 0
