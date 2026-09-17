"""Curriculum packs: load, ingest, and frontier against the knowledge store."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cuda_sft_qgen.store.knowledge import ENTITY_KINDS, RELATIONS, KnowledgeStore, official_url

REQUIRED_CLAIM = ("l3_id", "claim", "domain_id", "l2_id")


def load_pack(path: Path | str) -> dict[str, Any]:
    """Load a curriculum pack YAML; requires a mapping with an ``id`` field."""
    path = Path(path)
    pack = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(pack, dict) or not pack.get("id"):
        raise ValueError(f"pack must be a mapping with id: {path}")
    pack["_path"] = str(path)
    return pack


def list_packs(root: Path | str) -> list[Path]:
    """Sorted ``*.yaml`` paths in ``root``, or ``[]`` if it is not a directory."""
    directory = Path(root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.yaml"))


def _claim_payload(raw: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_CLAIM if not raw.get(key)]
    if missing:
        raise ValueError(f"seed claim missing {missing}")
    payload = {
        "id": raw["l3_id"],
        "l3_id": raw["l3_id"],
        "claim": str(raw["claim"]).strip(),
        "domain_id": raw["domain_id"],
        "domain_name": raw.get("domain_name") or raw["domain_id"],
        "l2_id": raw["l2_id"],
        "l2_name": raw.get("l2_name") or raw["l2_id"],
        "types": list(raw.get("types") or ["conceptual"]),
        "difficulty": list(raw.get("difficulty") or [2]),
        "kind": raw.get("kind") or "mechanism",
        "hints": list(raw.get("hints") or []),
        "sources": [url for url in (raw.get("sources") or pack.get("official_sources") or []) if official_url(str(url))],
        "applicability": dict(raw.get("applicability") or {}),
    }
    return payload


def ingest_pack(store: KnowledgeStore, pack: dict[str, Any], *, apply: bool = False,
                enqueue_jobs: bool = True) -> dict[str, Any]:
    """Add pack seeds as provisional claims/relations. Never overwrites verified claims."""
    claims = [_claim_payload(raw, pack) for raw in pack.get("seed_claims") or []]
    relations = list(pack.get("relations") or [])
    entities = list(pack.get("entities") or [])
    added_claims: list[str] = []
    skipped: list[str] = []
    added_relations = 0
    added_entities = 0
    if not apply:
        return {
            "pack": pack["id"],
            "applied": False,
            "would_add_claims": [c["l3_id"] for c in claims],
            "would_add_relations": len(relations),
            "would_add_entities": len(entities),
        }
    for entity in entities:
        kind = str(entity.get("kind") or "")
        if kind not in ENTITY_KINDS:
            continue
        store.upsert_entity(str(entity["id"]), kind, str(entity.get("label") or entity["id"]), entity.get("attrs") or {})
        added_entities += 1
    for payload in claims:
        if store.add_claim(payload, status="provisional", reason=f"pack:{pack['id']}"):
            added_claims.append(payload["l3_id"])
            if enqueue_jobs:
                store.enqueue("audit", {"claim_id": payload["l3_id"], "revision": 1}, f"audit:{payload['l3_id']}:1")
                store.enqueue("relate", {"claim_id": payload["l3_id"]}, f"relate:{payload['l3_id']}")
        else:
            skipped.append(payload["l3_id"])
    for edge in relations:
        kind = str(edge.get("kind") or "")
        if kind not in RELATIONS:
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        try:
            store.add_relation(source, target, kind, status="provisional", reason=str(edge.get("reason") or f"pack:{pack['id']}"))
            added_relations += 1
        except (KeyError, ValueError):
            continue
    store.set_metadata(f"pack:{pack['id']}:ingested", True)
    return {
        "pack": pack["id"],
        "applied": True,
        "added_claims": added_claims,
        "skipped_claims": skipped,
        "added_relations": added_relations,
        "added_entities": added_entities,
    }


def curriculum_frontier(store: KnowledgeStore, pack: dict[str, Any]) -> dict[str, Any]:
    """Compare pack L2 leaf targets against claims currently in the store."""
    by_l2: dict[str, int] = {}
    for claim in store.list_claims():
        key = f"{claim.get('domain_id')}.{claim.get('l2_id')}"
        by_l2[key] = by_l2.get(key, 0) + 1
    gaps = []
    for target in pack.get("l2_targets") or []:
        domain = target.get("domain")
        l2 = target.get("l2")
        want = int(target.get("target_leaves") or 16)
        have = by_l2.get(f"{domain}.{l2}", 0)
        gaps.append({"domain": domain, "l2": l2, "have": have, "want": want, "missing": max(0, want - have)})
    return {"pack": pack["id"], "l2": gaps, "gap_prompts": list(pack.get("gap_prompts") or [])}
