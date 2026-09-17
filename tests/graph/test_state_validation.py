"""Tests for Pydantic state validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from cuda_sft_qgen.graph.state_v2 import PipelineState


def test_pipeline_state_valid():
    """Test valid PipelineState creation."""
    state = PipelineState(
        target=100,
        review_batch_size=10,
        max_iterations=50,
        iteration=5,
        review_rounds=3,
        accepted_count=25,
        rejected_count=5,
        pending=[],
        steering_brief={},
        batch_briefs=[],
        done=False,
        status="running",
        log=["test log"],
    )
    assert state.target == 100
    assert state.accepted_count == 25
    assert state.iteration == 5
    assert not state.done


def test_pipeline_state_defaults():
    """Test PipelineState with default values."""
    state = PipelineState()
    assert state.iteration == 0
    assert state.review_rounds == 0
    assert state.accepted_count == 0
    assert state.rejected_count == 0
    assert state.pending == []
    assert state.batch_briefs == []
    assert not state.done
    assert state.log == []


def test_pipeline_state_negative_counts():
    """Test that negative counts are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PipelineState(accepted_count=-1)
    assert "accepted_count" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        PipelineState(rejected_count=-5)
    assert "rejected_count" in str(exc_info.value)


def test_pipeline_state_negative_iteration():
    """Test that negative iteration is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PipelineState(iteration=-1)
    assert "iteration" in str(exc_info.value)


def test_pipeline_state_type_coercion():
    """Test that types are coerced properly."""
    state = PipelineState(
        target="100",  # str -> int
        iteration="5",  # str -> int
        accepted_count="25",  # str -> int
    )
    assert isinstance(state.target, int)
    assert state.target == 100
    assert isinstance(state.iteration, int)
    assert state.iteration == 5


def test_pipeline_state_invalid_types():
    """Test that invalid types raise ValidationError."""
    with pytest.raises(ValidationError):
        PipelineState(pending="not_a_list")  # should be list

    with pytest.raises(ValidationError):
        PipelineState(iteration="not_a_number")  # should be int


def test_pipeline_state_extra_fields():
    """Test that extra fields are allowed (for forward compatibility)."""
    state = PipelineState(
        target=100,
        extra_field="should_be_allowed",  # type: ignore
    )
    # Pydantic will allow extra fields if configured
    assert state.target == 100


def test_pipeline_state_model_dump():
    """Test serialization via model_dump."""
    state = PipelineState(
        target=100,
        iteration=5,
        accepted_count=25,
        pending=[{"id": "q1", "question": "test"}],
        log=["log1", "log2"],
    )
    dumped = state.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["target"] == 100
    assert dumped["iteration"] == 5
    assert dumped["accepted_count"] == 25
    assert len(dumped["pending"]) == 1
    assert len(dumped["log"]) == 2


def test_pipeline_state_model_validate():
    """Test deserialization via model_validate."""
    data = {
        "target": 100,
        "iteration": 5,
        "accepted_count": 25,
        "rejected_count": 5,
        "pending": [],
        "batch_briefs": [],
        "done": False,
        "status": "running",
        "log": ["test"],
    }
    state = PipelineState.model_validate(data)
    assert state.target == 100
    assert state.iteration == 5
    assert state.accepted_count == 25


def test_pipeline_state_partial_data():
    """Test that partial data uses defaults."""
    data = {"target": 200, "iteration": 10}
    state = PipelineState.model_validate(data)
    assert state.target == 200
    assert state.iteration == 10
    assert state.accepted_count == 0  # default
    assert state.pending == []  # default
