"""Shared CLI helpers: logging, JSON printing, and settings loading."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from cuda_sft_qgen.config import load_settings

__all__ = ["load_settings", "_setup_logging", "_json_print", "_require_keys"]


def _setup_logging(verbose: bool) -> None:
    """Configure root logging; ``--verbose`` enables DEBUG."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _json_print(payload) -> None:
    """Print a JSON object to stdout."""
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _require_keys(settings) -> int | None:
    """Return 2 if real generation is requested without API keys."""
    if not settings.fake_llm and not settings.nvidia_api_keys:
        print(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3, or pass --fake.",
            file=sys.stderr,
        )
        return 2
    return None
