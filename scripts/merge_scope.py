"""Merge configs/scope.yaml L1/L2 skeleton into the taxonomy — ONLY ADD, never delete (fix/tasks/T32).

SKELETON placed by T16. Task T32 implements `merge()` exactly as specified in fix/tasks/T32_scope_merge.md.
Usage: python scripts/merge_scope.py --scope configs/scope.yaml --taxonomy configs/taxonomy/cuda.yaml [--apply]
stdout: one JSON line {"added_l1": [], "added_l2": ["T06.block_scaled_fp4", ...], "patched_l2": ["T08.tma", ...],
        "weighted_l1": ["T01", ...]}; logs go to stderr. Scope v3 semantics: see fix/tasks/T32_scope_merge.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge(scope: dict, taxonomy) -> dict:
    """Mutate taxonomy.domains in place and return the summary dict (T32 step 1)."""
    summary = {"added_l1": [], "added_l2": [], "patched_l2": [], "weighted_l1": []}
    domains = {str(domain["id"]): domain for domain in taxonomy.domains}
    for scoped in scope.get("domains") or []:
        domain_id = str(scoped["id"])
        domain = domains.get(domain_id)
        if domain is None:
            domain = {"id": domain_id, "name": scoped.get("name", domain_id),
                      "weight": scoped["weight"], "children": []}
            taxonomy.domains.append(domain)
            domains[domain_id] = domain
            summary["added_l1"].append(domain_id)
        elif "weight" not in domain:
            domain["weight"] = scoped["weight"]
            summary["weighted_l1"].append(domain_id)
        existing = {str(child["id"]): child for child in domain.get("children") or []}
        overrides = scoped.get("l2_overrides") or {}
        for child_id, child in existing.items():
            changed = False
            for key, value in (overrides.get(child_id) or {}).items():
                if key in {"arch", "target_leaves"} and key not in child:
                    child[key] = value
                    changed = True
            if "target_leaves" not in child and scoped.get("default_target") is not None:
                child["target_leaves"] = scoped["default_target"]
                changed = True
            if changed:
                summary["patched_l2"].append(f"{domain_id}.{child_id}")
        for item in scoped.get("new_l2") or []:
            child_id = str(item["id"])
            if child_id in existing:
                continue
            domain.setdefault("children", []).append({
                "id": child_id, "name": item["name"], "arch": item["arch"],
                "target_leaves": item["target_leaves"], "leaves": [],
            })
            existing[child_id] = domain["children"][-1]
            summary["added_l2"].append(f"{domain_id}.{child_id}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True); ap.add_argument("--taxonomy", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import yaml
    from cuda_sft_qgen.taxonomy import dump_taxonomy, load_taxonomy
    tax = load_taxonomy(Path(a.taxonomy).resolve())
    summary = merge(yaml.safe_load(Path(a.scope).resolve().read_text(encoding="utf-8")), tax)
    if a.apply:
        dump_taxonomy(tax)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
