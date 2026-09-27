"""Unit tests for writer/evolver generate_batch and JSON salvage hooks."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from cuda_sft_qgen.graph.nodes.generate import (
    generate_batch_node,
    generate_or_salvage,
    question_from_obj,
    generate_from_brief,
    salvage_item,
)
from cuda_sft_qgen.graph.nodes.generate_core import GenerateOutcome
from cuda_sft_qgen.models import QuestionBrief, SteeringBrief
from cuda_sft_qgen.runtime import set_runtime


def test_question_from_obj_basic(sample_brief):
    """Test converting dict to Question object."""
    brief = sample_brief()
    obj = {
        "question": "What is CUDA occupancy?",
        "tests_claim": "CUDA occupancy determines SM utilization",
        "language": "zh",
    }

    question = question_from_obj(brief, obj, source="writer")

    assert question.question == "What is CUDA occupancy?"
    assert question.tests_claim == "CUDA occupancy determines SM utilization"
    assert question.language == "zh"
    assert question.l3_id == brief.l3_id
    assert question.question_type == brief.question_type
    assert question.source == "writer"
    assert question.template_fingerprint
    assert question.simhash
    assert question.exact_hash


def test_generate_from_brief_fake_llm(fake_runtime, sample_brief, sample_steering_brief):
    """FakeLLM cannot synthesize a question even through the node helper."""
    set_runtime(fake_runtime)
    with pytest.raises(RuntimeError, match="real LLM API"):
        generate_from_brief(sample_brief(), sample_steering_brief())


def test_evolver_brief_uses_evolver_client(fake_runtime, sample_brief, sample_steering_brief, monkeypatch):
    monkeypatch.setenv("WRITER_THINKING_LEVEL", "high")
    monkeypatch.setenv("EVOLVER_THINKING_LEVEL", "medium")
    fake_runtime.llm = MagicMock()
    fake_runtime.evolver_llm = MagicMock()
    fake_runtime.evolver_llm.complete.return_value = (
        '{"question":"How does CUDA occupancy constrain a kernel launch?",'
        '"tests_claim":"CUDA occupancy determines SM utilization"}'
    )
    fake_runtime.prompt_builder = SimpleNamespace(build=lambda *args, **kwargs: SimpleNamespace(
        system="system", user="user", cache_key="", generation_provenance={}
    ))
    set_runtime(fake_runtime)
    question = generate_from_brief(
        sample_brief(role="evolver", seed_question="Earlier occupancy question"),
        sample_steering_brief(),
    )
    assert question.source == "evolver"
    fake_runtime.evolver_llm.complete.assert_called_once()
    assert fake_runtime.evolver_llm.complete.call_args.kwargs["thinking_level"] == "medium"
    fake_runtime.llm.complete.assert_not_called()


def test_generate_batch_node_empty(fake_runtime):
    """Test generate batch node with no briefs."""
    set_runtime(fake_runtime)

    state = {
        "batch_briefs": [],
        "steering_brief": SteeringBrief().model_dump(),
        "pending": [],
    }

    result = generate_batch_node(state)

    assert result["status"] == "generated_0"
    assert result["pending"] == []
    assert result["salvage_items"] == []


def test_generate_batch_node_with_existing_pending(
    fake_runtime, sample_brief, sample_steering_brief, sample_question
):
    """Test generate batch node preserves existing pending."""
    set_runtime(fake_runtime)

    existing_pending = [sample_question(id="q_existing").to_jsonl()]
    briefs = [sample_brief().model_dump() for _ in range(3)]

    state = {
        "batch_briefs": briefs,
        "steering_brief": sample_steering_brief().model_dump(),
        "pending": existing_pending,
    }

    result = generate_batch_node(state)

    # Should have at least the original pending question
    assert len(result["pending"]) >= 1
    assert any(q["id"] == "q_existing" for q in result["pending"])


def test_generate_or_salvage_does_not_inline_by_default(
    fake_runtime, sample_brief, sample_steering_brief, monkeypatch
):
    set_runtime(fake_runtime)
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("salvage must not run on the writer thread")

    monkeypatch.setattr(
        "cuda_sft_qgen.graph.nodes.generate.salvage_item", boom
    )

    class BadLLM:
        def complete(self, *args, **kwargs):
            return "not a json object"

    fake_runtime.llm = BadLLM()
    out = generate_or_salvage(
        sample_brief().model_dump(),
        sample_steering_brief(),
    )
    assert out.status == "salvage"
    assert called["n"] == 0


def test_inline_salvage_keeps_real_writer_origin(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph.nodes import generate_core

    set_runtime(fake_runtime)
    fake_runtime.settings.default_yaml["generation"]["salvage_inline"] = True
    monkeypatch.setenv("SALVAGE_INLINE", "1")
    seen = []
    monkeypatch.setattr(generate_core, "_safe_generate", lambda *a, **k: GenerateOutcome(
        status="salvage", brief=sample_brief().model_dump(), raw="broken", writer_real_api=True
    ))
    monkeypatch.setattr(generate_core, "salvage_item", lambda item: (seen.append(item) or GenerateOutcome(
        status="retry", brief=item["brief"]
    )))
    generate_or_salvage(sample_brief().model_dump(), SteeringBrief())
    assert seen[0]["writer_real_api"] is True


def test_salvage_item_skips_llm_when_raw_has_no_question(
    fake_runtime, sample_brief
):
    set_runtime(fake_runtime)
    fake_runtime.settings.salvage_use_llm = True
    out = salvage_item(
        {
            "brief": sample_brief().model_dump(),
            "raw": "long reasoning dump without a payload",
        }
    )
    assert out.status == "retry"
    assert out.reason == "salvage_question_unverifiable"
