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
    raise NotImplementedError("implemented by task T32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True); ap.add_argument("--taxonomy", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import yaml
    from cuda_sft_qgen.taxonomy import dump_taxonomy, load_taxonomy
    tax = load_taxonomy(Path(a.taxonomy))
    summary = merge(yaml.safe_load(Path(a.scope).read_text(encoding="utf-8")), tax)
    if a.apply:
        dump_taxonomy(tax)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
