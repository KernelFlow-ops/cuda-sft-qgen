"""Pydantic view of pipeline state for validation and TypedDict interop."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PipelineState(BaseModel):
    """Pipeline state using Pydantic for validation and type safety.

    Backward compatible with TypedDict-based state while adding validation.
    """

    rebuild_taxonomy: bool = False
    target: int = Field(default=100, ge=1)
    review_batch_size: int = Field(default=10, ge=1)
    max_iterations: int = Field(default=100, ge=1)
    iteration: int = Field(default=0, ge=0)
    review_rounds: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    pending: list[dict[str, Any]] = Field(default_factory=list)
    salvage_items: list[dict[str, Any]] = Field(default_factory=list)
    batch_briefs: list[dict[str, Any]] = Field(default_factory=list)
    steering_brief: dict[str, Any] = Field(default_factory=dict)
    last_diversity: dict[str, Any] = Field(default_factory=dict)
    taxonomy_ticks: int = Field(default=0, ge=0)
    last_taxonomy_health: dict[str, Any] = Field(default_factory=dict)
    last_taxonomy_facet: str = ""
    done: bool = False
    status: str = "init"
    log: list[str] = Field(default_factory=list)

    model_config = {
        "extra": "allow",  # Allow extra fields for backward compatibility
        "validate_assignment": True,
    }

    @field_validator("pending", "salvage_items", "batch_briefs")
    @classmethod
    def validate_list_fields(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure list fields contain dicts."""
        if not isinstance(v, list):
            return []
        return [item if isinstance(item, dict) else {} for item in v]

    @field_validator("steering_brief", "last_diversity", "last_taxonomy_health")
    @classmethod
    def validate_dict_fields(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure dict fields are actually dicts."""
        if not isinstance(v, dict):
            return {}
        return v

    @model_validator(mode="after")
    def validate_counters(self) -> "PipelineState":
        """Ensure counters are consistent with data."""
        if self.iteration > self.max_iterations:
            raise ValueError(
                f"iteration ({self.iteration}) cannot exceed max_iterations ({self.max_iterations})"
            )
        if self.accepted_count + len(self.pending) > self.target * 2:
            # Soft warning - allow some overshoot but flag excessive divergence
            pass
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backward compatibility with TypedDict consumers."""
        return self.model_dump(exclude_none=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineState":
        """Create from dict for backward compatibility with TypedDict producers."""
        return cls.model_validate(data)

    def append_log(self, message: str) -> None:
        """Append to log (emulates operator.add behavior)."""
        self.log.append(message)

    def merge_update(self, update: dict[str, Any]) -> "PipelineState":
        """Merge update dict into state, handling log append specially."""
        data = self.model_dump()
        for key, value in update.items():
            if key == "log" and isinstance(value, list):
                # Append to log instead of replacing
                data["log"] = data.get("log", []) + value
            else:
                data[key] = value
        return PipelineState.model_validate(data)
