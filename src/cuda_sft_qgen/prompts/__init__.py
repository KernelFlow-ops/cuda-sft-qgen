from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    path = _DIR / f"{name}.md"
    if not path.exists():
        path = _DIR / "layers" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def load_prompt_yaml(name: str) -> Any:
    path = _DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
