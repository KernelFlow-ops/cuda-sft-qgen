from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


@dataclass(frozen=True)
class Leaf:
    domain_id: str
    domain_name: str
    l2_id: str
    l2_name: str
    l3_id: str
    claim: str
    types: tuple[str, ...]
    difficulty: tuple[int, ...]
    hints: tuple[str, ...] = ()

    @property
    def path(self) -> list[str]:
        return [self.domain_id, self.l2_id, self.l3_id]

    @property
    def path_key(self) -> str:
        return ".".join(self.path)


@dataclass
class Taxonomy:
    version: int
    name: str
    leaves: list[Leaf]
    domains: list[dict[str, Any]]

    def leaf_by_key(self, key: str) -> Leaf | None:
        for leaf in self.leaves:
            if leaf.path_key == key or leaf.l3_id == key:
                return leaf
        return None

    def l1_ids(self) -> list[str]:
        return [d["id"] for d in self.domains]

    def l2_keys(self) -> list[str]:
        keys: list[str] = []
        for d in self.domains:
            for c in d.get("children", []):
                keys.append(f"{d['id']}.{c['id']}")
        return keys

    def iter_l2(self) -> Iterator[tuple[str, str, dict[str, Any]]]:
        for d in self.domains:
            for c in d.get("children", []):
                yield d["id"], d["name"], c


def load_taxonomy(path: Path) -> Taxonomy:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    leaves: list[Leaf] = []
    for domain in raw.get("domains", []):
        for child in domain.get("children", []):
            for item in child.get("leaves", []):
                leaves.append(
                    Leaf(
                        domain_id=domain["id"],
                        domain_name=domain["name"],
                        l2_id=child["id"],
                        l2_name=child["name"],
                        l3_id=item["id"],
                        claim=item["claim"],
                        types=tuple(item.get("types") or ["conceptual"]),
                        difficulty=tuple(item.get("difficulty") or [2]),
                        hints=tuple(item.get("hints") or []),
                    )
                )
    if not leaves:
        raise ValueError(f"taxonomy has no leaves: {path}")
    return Taxonomy(
        version=int(raw.get("version", 1)),
        name=str(raw.get("name", "cuda")),
        leaves=leaves,
        domains=list(raw.get("domains", [])),
    )


def dump_taxonomy(taxonomy: Taxonomy, path: Path) -> None:
    payload = {
        "version": taxonomy.version,
        "name": taxonomy.name,
        "domains": taxonomy.domains,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def merge_leaves(taxonomy: Taxonomy, new_leaves: list[dict[str, Any]]) -> int:
    """Merge architect-produced leaves into the in-memory taxonomy. Returns added count."""
    existing = {leaf.l3_id for leaf in taxonomy.leaves}
    added = 0
    domain_index = {d["id"]: d for d in taxonomy.domains}
    for item in new_leaves:
        l3_id = item.get("l3_id") or item.get("id")
        domain_id = item.get("domain_id")
        l2_id = item.get("l2_id")
        claim = item.get("claim")
        if not (l3_id and domain_id and l2_id and claim):
            continue
        if l3_id in existing:
            continue
        domain = domain_index.get(domain_id)
        if domain is None:
            continue
        child = None
        for c in domain.get("children", []):
            if c["id"] == l2_id:
                child = c
                break
        if child is None:
            child = {"id": l2_id, "name": item.get("l2_name", l2_id), "leaves": []}
            domain.setdefault("children", []).append(child)
        child.setdefault("leaves", []).append(
            {
                "id": l3_id,
                "claim": claim,
                "types": item.get("types") or ["conceptual"],
                "difficulty": item.get("difficulty") or [2],
            }
        )
        taxonomy.leaves.append(
            Leaf(
                domain_id=domain_id,
                domain_name=domain.get("name", domain_id),
                l2_id=l2_id,
                l2_name=child.get("name", l2_id),
                l3_id=l3_id,
                claim=claim,
                types=tuple(item.get("types") or ["conceptual"]),
                difficulty=tuple(item.get("difficulty") or [2]),
            )
        )
        existing.add(l3_id)
        added += 1
    return added
