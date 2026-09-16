#!/usr/bin/env python3
"""Split the seed cuda.yaml into per-domain files and graft extra L2/L3 seeds.

Keeps every original L3 id. New leaves use ``{l2_id}_{snake}`` ids.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAX_DIR = ROOT / "configs" / "taxonomy"
DOMAIN_DIR = TAX_DIR / "domains"
SRC = TAX_DIR / "cuda.yaml"

sys.path.insert(0, str(ROOT / "scripts"))
from taxonomy_legacy_extra import EXTRA_LEAVES, KIND_OVERRIDE, NEW_L2  # noqa: E402

VALID_TYPES = {
    "conceptual",
    "syntax",
    "code_complete",
    "debug",
    "optimization",
    "layout",
    "compare",
    "counterfactual",
    "profiling",
}
TYPE_FIX = {
    "feature": "conceptual",
    "formula": "conceptual",
    "failure": "debug",
    "api": "syntax",
    "numeric": "optimization",
    "evaluate": "compare",
}


def sanitize_leaf(item: dict) -> dict:
    out = dict(item)
    types = []
    for t in out.get("types") or []:
        t = TYPE_FIX.get(str(t), str(t))
        if t in VALID_TYPES and t not in types:
            types.append(t)
    out["types"] = types or ["conceptual"]
    diffs = []
    for d in out.get("difficulty") or [2]:
        try:
            di = int(d)
        except (TypeError, ValueError):
            continue
        if 1 <= di <= 4 and di not in diffs:
            diffs.append(di)
    out["difficulty"] = diffs or [2]
    return out


FILES = {
    "T01": "T01_programming_model.yaml",
    "T02": "T02_memory.yaml",
    "T03": "T03_warp_occupancy.yaml",
    "T04": "T04_sync.yaml",
    "T05": "T05_optimization.yaml",
    "T06": "T06_tensor_core.yaml",
    "T07": "T07_cutlass_cute.yaml",
    "T08": "T08_hopper_blackwell.yaml",
    "T09": "T09_toolchain.yaml",
    "T10": "T10_multi_gpu.yaml",
    "T11": "T11_libraries.yaml",
    "T12": "T12_profiling.yaml",
    "T13": "T13_algorithms.yaml",
    "T14": "T14_runtime_graphs.yaml",
    "T15": "T15_numeric.yaml",
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )


def _annotate(item: dict) -> dict:
    out = dict(item)
    l3 = str(out.get("id") or "")
    if l3 in KIND_OVERRIDE and "kind" not in out:
        out["kind"] = KIND_OVERRIDE[l3]
    return out


def graft_legacy(domains: list[dict]) -> list[dict]:
    out = copy.deepcopy(domains)
    by_id = {d["id"]: d for d in out}
    for domain in out:
        did = domain["id"]
        children = domain.setdefault("children", [])
        child_ix = {c["id"]: c for c in children}
        for child in children:
            leaves = child.setdefault("leaves", [])
            child["leaves"] = [sanitize_leaf(_annotate(x)) for x in leaves]
            extra = EXTRA_LEAVES.get((did, child["id"]), [])
            have = {str(x["id"]) for x in child["leaves"]}
            for rec in extra:
                if rec["id"] not in have:
                    child["leaves"].append(sanitize_leaf(dict(rec)))
                    have.add(rec["id"])
        for spec in NEW_L2.get(did, []):
            if spec["id"] in child_ix:
                # Merge extra leaves into existing L2 of the same id.
                dest = child_ix[spec["id"]]
                have = {str(x["id"]) for x in dest.get("leaves") or []}
                for rec in spec.get("leaves") or []:
                    if rec["id"] not in have:
                        dest.setdefault("leaves", []).append(sanitize_leaf(dict(rec)))
                        have.add(rec["id"])
                continue
            spec_copy = copy.deepcopy(spec)
            spec_copy["leaves"] = [sanitize_leaf(x) for x in spec_copy.get("leaves") or []]
            children.append(spec_copy)
            child_ix[spec["id"]] = children[-1]
        for child in children:
            child["leaves"] = [sanitize_leaf(x) for x in child.get("leaves") or []]
    return out


def write_index(domain_ids: list[str], version: int, name: str) -> None:
    include = []
    for did in domain_ids:
        if did in FILES:
            include.append(f"domains/{FILES[did]}")
        else:
            matches = sorted(DOMAIN_DIR.glob(f"{did}_*.yaml"))
            if not matches:
                raise FileNotFoundError(f"no domain file for {did}")
            include.append(f"domains/{matches[0].name}")
    _dump(SRC, {"version": version, "name": name, "include": include})


def main() -> None:
    raw = _load(SRC)
    if "include" in raw:
        # Already split; still re-graft legacy extras onto T01–T15 files.
        domains = []
        for rel in raw["include"]:
            domains.append(_load(TAX_DIR / rel))
        grafted = graft_legacy([d for d in domains if d["id"] in FILES])
        grafted_ids = {d["id"] for d in grafted}
        DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
        for d in grafted:
            _dump(DOMAIN_DIR / FILES[d["id"]], d)
        keep = [d for d in domains if d["id"] not in grafted_ids]
        ids = [d["id"] for d in grafted] + [d["id"] for d in keep]
        write_index(ids, int(raw.get("version", 2)), str(raw.get("name", "nvidia_cuda_knowledge")))
        print(f"re-grafted legacy domains; index ids={len(ids)}")
        return

    domains = graft_legacy(list(raw.get("domains") or []))
    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    ids = []
    for d in domains:
        did = d["id"]
        ids.append(did)
        _dump(DOMAIN_DIR / FILES[did], {"id": did, "name": d.get("name", did), "children": d.get("children") or []})
    write_index(ids, int(raw.get("version", 1)) + 1, str(raw.get("name", "nvidia_cuda_knowledge")))
    n_leaf = sum(len(c.get("leaves") or []) for d in domains for c in d.get("children") or [])
    print(f"wrote {len(ids)} legacy domain files, {n_leaf} leaves (T16+ not in this pass)")


if __name__ == "__main__":
    main()
