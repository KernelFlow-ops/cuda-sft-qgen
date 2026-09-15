from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    path = _DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")
