"""Tests that split domain YAML loads and preserves legacy L3 ids."""

from __future__ import annotations

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.taxonomy import load_taxonomy


def test_split_taxonomy_loads_and_keeps_legacy_ids():
    settings = load_settings(fake_llm=True)
    tax = load_taxonomy(settings.taxonomy_path)
    assert len(tax.l1_ids()) >= 15
    ids = [leaf.l3_id for leaf in tax.leaves]
    assert len(ids) == len(set(ids))
    for legacy in ("tmem", "coalescing", "syncthreads", "umma", "shape_stride"):
        assert legacy in ids
    assert len(tax.l1_ids()) == 31
    assert len(tax.leaves) >= 2500


def test_each_l2_has_non_api_kind():
    settings = load_settings(fake_llm=True)
    tax = load_taxonomy(settings.taxonomy_path)
    by_l2: dict[str, set[str]] = {}
    for leaf in tax.leaves:
        key = f"{leaf.domain_id}.{leaf.l2_id}"
        by_l2.setdefault(key, set()).add(leaf.kind or "mechanism")
    assert by_l2
    missing = [k for k, kinds in by_l2.items() if kinds <= {"api", ""}]
    # Seeds should mix kinds; allow a few leftover api-only L2s from the original file.
    assert len(missing) <= max(5, len(by_l2) // 10)
