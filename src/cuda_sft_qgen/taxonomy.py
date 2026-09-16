from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import yaml

LEAF_KINDS = (
    "mechanism",
    "api",
    "formula",
    "feature",
    "failure",
    "numeric",
)


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
    kind: str = ""
    retired: bool = False

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
    index_path: Path | None = None
    domain_files: dict[str, Path] = field(default_factory=dict)

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


def _leaf_from_item(
    *,
    domain_id: str,
    domain_name: str,
    l2_id: str,
    l2_name: str,
    item: dict[str, Any],
) -> Leaf:
    kind = str(item.get("kind") or "").strip()
    if kind and kind not in LEAF_KINDS:
        kind = ""
    return Leaf(
        domain_id=domain_id,
        domain_name=domain_name,
        l2_id=l2_id,
        l2_name=l2_name,
        l3_id=str(item["id"]),
        claim=str(item["claim"]),
        types=tuple(item.get("types") or ["conceptual"]),
        difficulty=tuple(item.get("difficulty") or [2]),
        hints=tuple(item.get("hints") or []),
        kind=kind,
        retired=bool(item.get("retired") or False),
    )


def _leaves_from_domains(domains: list[dict[str, Any]]) -> list[Leaf]:
    leaves: list[Leaf] = []
    for domain in domains:
        for child in domain.get("children", []):
            for item in child.get("leaves", []):
                leaves.append(
                    _leaf_from_item(
                        domain_id=str(domain["id"]),
                        domain_name=str(domain.get("name", domain["id"])),
                        l2_id=str(child["id"]),
                        l2_name=str(child.get("name", child["id"])),
                        item=item,
                    )
                )
    return leaves


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def load_taxonomy(path: Path) -> Taxonomy:
    raw = _load_yaml(path)
    domain_files: dict[str, Path] = {}
    includes = raw.get("include")
    if includes:
        domains: list[dict[str, Any]] = []
        for rel in includes:
            dpath = (path.parent / str(rel)).resolve()
            if not dpath.exists():
                continue
            domain = _load_yaml(dpath)
            if "id" not in domain:
                raise ValueError(f"domain file missing id: {dpath}")
            domains.append(domain)
            domain_files[str(domain["id"])] = dpath
    else:
        domains = list(raw.get("domains") or [])
    leaves = _leaves_from_domains(domains)
    if not leaves:
        raise ValueError(f"taxonomy has no leaves: {path}")
    return Taxonomy(
        version=int(raw.get("version", 1)),
        name=str(raw.get("name", "cuda")),
        leaves=leaves,
        domains=domains,
        index_path=path,
        domain_files=domain_files,
    )


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )


def dump_taxonomy(taxonomy: Taxonomy, path: Path | None = None) -> None:
    path = path or taxonomy.index_path
    if path is None:
        raise ValueError("dump_taxonomy requires a path")
    if taxonomy.domain_files:
        include: list[str] = []
        root = path.parent
        for domain in taxonomy.domains:
            did = str(domain["id"])
            dpath = taxonomy.domain_files.get(did)
            if dpath is None:
                dpath = root / "domains" / f"{did}.yaml"
                taxonomy.domain_files[did] = dpath
            payload = {
                "id": domain["id"],
                "name": domain.get("name", did),
                "children": domain.get("children") or [],
            }
            _dump_yaml(dpath, payload)
            include.append(str(dpath.relative_to(root)))
        _dump_yaml(
            path,
            {
                "version": taxonomy.version,
                "name": taxonomy.name,
                "include": include,
            },
        )
        return
    _dump_yaml(
        path,
        {
            "version": taxonomy.version,
            "name": taxonomy.name,
            "domains": taxonomy.domains,
        },
    )


def _leaf_record(item: dict[str, Any], l3_id: str, claim: str) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": l3_id,
        "claim": claim,
        "types": item.get("types") or ["conceptual"],
        "difficulty": item.get("difficulty") or [2],
    }
    kind = str(item.get("kind") or "").strip()
    if kind:
        rec["kind"] = kind
    hints = item.get("hints") or []
    if hints:
        rec["hints"] = list(hints)
    if item.get("retired"):
        rec["retired"] = True
    return rec


def merge_leaves(
    taxonomy: Taxonomy,
    new_leaves: list[dict[str, Any]],
    *,
    allow_new_domains: bool = False,
) -> int:
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
            if not allow_new_domains:
                continue
            domain = {
                "id": domain_id,
                "name": item.get("domain_name") or domain_id,
                "children": [],
            }
            taxonomy.domains.append(domain)
            domain_index[domain_id] = domain
        child = None
        for c in domain.get("children", []):
            if c["id"] == l2_id:
                child = c
                break
        if child is None:
            if not allow_new_domains:
                continue
            child = {"id": l2_id, "name": item.get("l2_name", l2_id), "leaves": []}
            domain.setdefault("children", []).append(child)
        rec = _leaf_record(item, str(l3_id), str(claim))
        child.setdefault("leaves", []).append(rec)
        taxonomy.leaves.append(
            _leaf_from_item(
                domain_id=str(domain_id),
                domain_name=str(domain.get("name", domain_id)),
                l2_id=str(l2_id),
                l2_name=str(child.get("name", l2_id)),
                item=rec,
            )
        )
        existing.add(str(l3_id))
        added += 1
    return added


def patch_leaf(
    taxonomy: Taxonomy,
    l3_id: str,
    *,
    claim: str | None = None,
    hints: list[str] | None = None,
) -> bool:
    """Update claim/hints on an existing leaf. Return True if found."""
    found = False
    new_claim = str(claim).strip() if claim else None
    new_hints = [str(h) for h in hints] if hints is not None else None
    for i, leaf in enumerate(taxonomy.leaves):
        if leaf.l3_id != l3_id:
            continue
        taxonomy.leaves[i] = replace(
            leaf,
            claim=new_claim if new_claim else leaf.claim,
            hints=tuple(new_hints) if new_hints is not None else leaf.hints,
        )
        found = True
        break
    for domain in taxonomy.domains:
        for child in domain.get("children") or []:
            for item in child.get("leaves") or []:
                if str(item.get("id")) != l3_id:
                    continue
                if new_claim:
                    item["claim"] = new_claim
                if new_hints is not None:
                    if new_hints:
                        item["hints"] = list(new_hints)
                    else:
                        item.pop("hints", None)
                found = True
    return found


def set_leaf_retired(taxonomy: Taxonomy, l3_id: str, retired: bool = True) -> bool:
    """Set retired on in-memory domains + Leaf list. Return True if found."""
    found = False
    for i, leaf in enumerate(taxonomy.leaves):
        if leaf.l3_id == l3_id:
            taxonomy.leaves[i] = replace(leaf, retired=retired)
            found = True
            break
    for domain in taxonomy.domains:
        for child in domain.get("children") or []:
            for item in child.get("leaves") or []:
                if str(item.get("id")) != l3_id:
                    continue
                if retired:
                    item["retired"] = True
                else:
                    item.pop("retired", None)
                found = True
    return found
