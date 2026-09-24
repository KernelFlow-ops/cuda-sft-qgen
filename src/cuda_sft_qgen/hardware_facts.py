"""Verified per-architecture limits rendered into writer and chair prompts. Does not call LLMs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=4)
def load(path: str) -> dict[str, Any]:
    """Load hardware_facts.yaml; missing files yield empty common and arch maps."""
    source = Path(path)
    if not source.is_file():
        return {"common": {}, "arch": {}}
    with source.open("r", encoding="utf-8") as handle:
        facts = yaml.safe_load(handle) or {}
    return {"common": facts.get("common") or {}, "arch": facts.get("arch") or {}}


def verified_arches(facts: dict[str, Any]) -> list[str]:
    """Return verified architecture keys in file order."""
    return [key for key, entry in (facts.get("arch") or {}).items()
            if not entry.get("verify_before_use", False)]


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.rstrip("\n").splitlines())


def render(arch_target: str, facts: dict[str, Any] | None = None) -> str:
    """Render common facts and verified limits for the requested architecture."""
    facts = facts or {"common": {}, "arch": {}}
    parts = [yaml.safe_dump({"common": facts.get("common") or {}}, allow_unicode=True, sort_keys=False).rstrip()]
    if arch_target != "any":
        arches = facts.get("arch") or {}
        lookup = [arch_target]
        if arch_target in {"sm_90a", "sm_100a"}:
            lookup.append(arch_target[:-1])
        key = next((candidate for candidate in lookup if candidate in arches), None)
        entry = arches.get(key) if key else None
        if isinstance(entry, dict) and not entry.get("verify_before_use", False):
            parts.append(yaml.safe_dump({key: {k: v for k, v in entry.items() if k != "verify_before_use"}},
                                        allow_unicode=True, sort_keys=False).rstrip())
        else:
            parts.append(f"# limits for {arch_target} not verified; do not state numeric limits for it")
    return _indent("\n".join(parts))


def render_all_verified(facts: dict[str, Any] | None = None) -> str:
    """Render common facts and every verified architecture for the chair."""
    facts = facts or {"common": {}, "arch": {}}
    payload = {"common": facts.get("common") or {}}
    payload.update({key: {k: v for k, v in facts["arch"][key].items() if k != "verify_before_use"}
                    for key in verified_arches(facts)})
    return _indent(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
