from __future__ import annotations


class RateLimitError(RuntimeError):
    """All keys were rate-limited for this call; requeue the brief."""


class JsonSalvageNeeded(Exception):
    """Model returned question-like text that local JSON parse could not fully accept."""

    def __init__(self, raw: str, brief: dict) -> None:
        super().__init__("json salvage needed")
        self.raw = raw
        self.brief = brief
