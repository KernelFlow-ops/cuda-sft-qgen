"""Typed exceptions used by graph nodes, LLM clients, and retry paths."""

from __future__ import annotations

from typing import Any


class RateLimitError(RuntimeError):
    """All keys were rate-limited for this call; requeue the brief."""


class JsonSalvageNeeded(Exception):
    """Model returned question-like text that local JSON parse could not fully accept."""

    def __init__(self, raw: str, brief: dict) -> None:
        """Store the unparsed model text and the originating generation brief.

        Args:
            raw: Model output that failed local JSON parse.
            brief: Planner brief that produced ``raw``.
        """
        super().__init__("json salvage needed")
        self.raw = raw
        self.brief = brief


# Node-level errors for unified error handling
class NodeError(Exception):
    """Base class for all graph node errors."""

    def __init__(
        self,
        message: str,
        node_name: str | None = None,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Attach graph-node metadata used by retry and logging.

        Args:
            message: Human-readable failure.
            node_name: Graph node that raised the error.
            recoverable: Whether the pipeline may retry the work.
            context: Extra structured fields for logs and salvage.
        """
        super().__init__(message)
        self.node_name = node_name
        self.recoverable = recoverable
        self.context = context or {}

    def __str__(self) -> str:
        prefix = f"[{self.node_name}] " if self.node_name else ""
        return f"{prefix}{super().__str__()}"


class PlannerError(NodeError):
    """Error in planner node - brief generation failed."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
    ) -> None:
        """Tag the error as originating from the planner node."""
        super().__init__(message, node_name="planner", recoverable=recoverable, context=context)


class GeneratorError(NodeError):
    """Error in generator node - question generation failed."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
    ) -> None:
        """Tag the error as originating from the generator node."""
        super().__init__(message, node_name="generator", recoverable=recoverable, context=context)


class ReviewError(NodeError):
    """Error in review node - scoring/chair/archive failed."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
    ) -> None:
        """Tag the error as review; ``stage`` is score/chair/archive."""
        context = context or {}
        if stage:
            context["stage"] = stage
        super().__init__(message, node_name="review", recoverable=recoverable, context=context)


class TaxonomyError(NodeError):
    """Error in taxonomy management - architect/health check failed."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
    ) -> None:
        """Tag the error as originating from taxonomy management."""
        super().__init__(message, node_name="taxonomy", recoverable=recoverable, context=context)
