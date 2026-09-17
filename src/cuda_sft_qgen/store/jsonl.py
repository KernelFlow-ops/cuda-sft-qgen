"""Append-only and atomic JSONL helpers for archive sidecars."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as a single line; creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSONL file; missing files yield nothing."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace ``path`` via temp file + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
