"""Tests for unified error handling."""

import pytest

from cuda_sft_qgen.errors import (
    GeneratorError,
    NodeError,
    PlannerError,
    ReviewError,
    TaxonomyError,
)


def test_node_error_base():
    """Test NodeError base class."""
    error = NodeError("Test error", node_name="test_node")
    assert "[test_node]" in str(error)
    assert "Test error" in str(error)
    assert error.node_name == "test_node"
    assert error.context == {}
    assert error.recoverable is True  # default is True


def test_node_error_with_context():
    """Test NodeError with context."""
    error = NodeError("Test error", node_name="test_node", context={"key": "value"})
    assert error.context == {"key": "value"}


def test_node_error_not_recoverable():
    """Test NodeError with recoverable flag."""
    error = NodeError("Test error", node_name="test_node", recoverable=False)
    assert error.recoverable is False


def test_planner_error():
    """Test PlannerError."""
    error = PlannerError("Planning failed")
    assert isinstance(error, NodeError)
    assert "[planner]" in str(error)
    assert "Planning failed" in str(error)


def test_planner_error_with_context():
    """Test PlannerError with context."""
    error = PlannerError(
        "Planning failed",
        context={"iteration": 5, "target": 100},
    )
    assert error.context["iteration"] == 5
    assert error.context["target"] == 100


def test_generator_error():
    """Test GeneratorError."""
    error = GeneratorError("Generation failed")
    assert isinstance(error, NodeError)
    assert "[generator]" in str(error)


def test_generator_error_recoverable():
    """Test GeneratorError with recoverable flag."""
    error = GeneratorError(
        "Temporary failure",
        recoverable=True,
    )
    assert error.recoverable is True


def test_review_error():
    """Test ReviewError."""
    error = ReviewError("Review failed")
    assert isinstance(error, NodeError)
    assert "[review]" in str(error)


def test_review_error_with_context():
    """Test ReviewError with context."""
    error = ReviewError(
        "Scoring failed",
        context={"batch_size": 32, "pending_count": 100},
    )
    assert error.context["batch_size"] == 32
    assert error.context["pending_count"] == 100


def test_taxonomy_error():
    """Test TaxonomyError."""
    error = TaxonomyError("Taxonomy operation failed")
    assert isinstance(error, NodeError)
    assert "[taxonomy]" in str(error)


def test_taxonomy_error_with_context():
    """Test TaxonomyError with context."""
    error = TaxonomyError(
        "Expansion failed",
        context={"node_id": "cuda.memory", "depth": 3},
    )
    assert error.context["node_id"] == "cuda.memory"
    assert error.context["depth"] == 3


def test_error_chaining():
    """Test that errors can be chained properly."""
    original = ValueError("Original error")
    try:
        raise GeneratorError("Wrapped error") from original
    except GeneratorError as error:
        assert error.__cause__ is original


def test_error_str_representation():
    """Test string representation of errors."""
    error = PlannerError("Test", context={"key": "value"})
    error_str = str(error)
    assert "Test" in error_str
    assert "[planner]" in error_str


def test_error_repr():
    """Test repr of errors."""
    error = NodeError("Test", node_name="test_node")
    repr_str = repr(error)
    assert "NodeError" in repr_str


def test_multiple_context_keys():
    """Test error with multiple context keys."""
    error = GeneratorError(
        "Complex failure",
        context={
            "batch_id": "batch_001",
            "question_count": 10,
            "retry_count": 3,
            "last_error": "timeout",
        },
    )
    assert len(error.context) == 4
    assert error.context["batch_id"] == "batch_001"
    assert error.context["retry_count"] == 3
