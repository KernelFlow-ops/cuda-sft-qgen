"""Load Markdown/YAML prompt templates shipped with the package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DIR = Path(__file__).parent


def _first_existing(paths: list[Path]) -> Path:
    """Return the first path that exists; raise using the preferred name."""
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def load_prompt(name: str) -> str:
    """Read a prompt Markdown file from prompts/, layers/, or taxonomy/.

    Args:
        name: Stem without extension, e.g. ``writer`` or ``expand``.

    Returns:
        File contents as UTF-8 text.
    """
    path = _first_existing(
        [
            _DIR / f"{name}.md",
            _DIR / "layers" / f"{name}.md",
            _DIR / "taxonomy" / f"{name}.md",
        ]
    )
    return path.read_text(encoding="utf-8")


def load_prompt_yaml(name: str) -> Any:
    """Load a YAML prompt sidecar from prompts/ or prompts/taxonomy/.

    Args:
        name: Stem without extension, e.g. ``facets`` or ``anti_patterns``.
    """
    path = _first_existing(
        [
            _DIR / f"{name}.yaml",
            _DIR / "taxonomy" / f"{name}.yaml",
        ]
    )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
