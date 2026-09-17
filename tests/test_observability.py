"""Tests for observability tracing."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from cuda_sft_qgen.observability import trace_node


def test_trace_node_decorator_basic():
    """Test that @trace_node decorator works on a basic function."""

    @trace_node("test_function")
    def sample_func(x: int, y: int) -> int:
        return x + y

    result = sample_func(3, 5)
    assert result == 8


def test_trace_node_with_dict_return():
    """Test trace_node with function returning dict."""

    @trace_node("dict_function")
    def dict_func(state: dict) -> dict:
        return {"result": state.get("value", 0) + 10, "status": "ok"}

    result = dict_func({"value": 5})
    assert result["result"] == 15
    assert result["status"] == "ok"


def test_trace_node_timing():
    """Test that trace_node can track execution time."""

    @trace_node("slow_function")
    def slow_func() -> str:
        time.sleep(0.01)  # 10ms delay
        return "done"

    start = time.time()
    result = slow_func()
    elapsed = time.time() - start

    assert result == "done"
    assert elapsed >= 0.01


def test_trace_node_with_exception():
    """Test that trace_node propagates exceptions."""

    @trace_node("error_function")
    def error_func() -> None:
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        error_func()


def test_trace_node_preserves_function_name():
    """Test that decorator preserves function metadata."""

    @trace_node("my_node")
    def my_function(x: int) -> int:
        """Docstring."""
        return x * 2

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "Docstring."


@patch("cuda_sft_qgen.observability.logger")
def test_trace_node_logs_execution(mock_logger):
    """Test that trace_node logs execution details."""

    @trace_node("logged_function")
    def logged_func(state: dict) -> dict:
        return {"status": "completed", "count": len(state.get("items", []))}

    result = logged_func({"items": [1, 2, 3]})

    assert result["status"] == "completed"
    assert result["count"] == 3
    # Check that logger was called (implementation-dependent)
    assert mock_logger.info.called or mock_logger.debug.called


def test_trace_node_with_state_dict():
    """Test trace_node with typical pipeline state dict."""

    @trace_node("pipeline_node")
    def pipeline_func(state: dict) -> dict:
        iteration = state.get("iteration", 0)
        return {
            "iteration": iteration + 1,
            "status": "processed",
            "log": [f"iteration {iteration + 1}"],
        }

    state = {"iteration": 5, "accepted_count": 10}
    result = pipeline_func(state)

    assert result["iteration"] == 6
    assert result["status"] == "processed"
    assert len(result["log"]) == 1


def test_trace_node_multiple_decorators():
    """Test applying trace_node to multiple functions."""

    @trace_node("func_a")
    def func_a(x: int) -> int:
        return x + 1

    @trace_node("func_b")
    def func_b(x: int) -> int:
        return x * 2

    assert func_a(5) == 6
    assert func_b(5) == 10


def test_trace_node_with_no_args():
    """Test trace_node with function that takes no args."""

    @trace_node("no_args")
    def no_args_func() -> dict:
        return {"result": "success"}

    result = no_args_func()
    assert result["result"] == "success"


def test_trace_node_with_kwargs():
    """Test trace_node with function using kwargs."""

    @trace_node("kwargs_function")
    def kwargs_func(state: dict, *, priority: int = 0) -> dict:
        return {"priority": priority, "data": state.get("data")}

    result = kwargs_func({"data": "test"}, priority=5)
    assert result["priority"] == 5
    assert result["data"] == "test"


def test_trace_node_nested_calls():
    """Test trace_node with nested function calls."""

    @trace_node("inner")
    def inner_func(x: int) -> int:
        return x * 2

    @trace_node("outer")
    def outer_func(x: int) -> int:
        return inner_func(x) + 10

    result = outer_func(5)
    assert result == 20  # (5 * 2) + 10
