"""Planner agent: hardware labels must be compatible with the leaf claim."""
from __future__ import annotations

import random

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.planner import leaf_hardware_allowed, plan_briefs
from cuda_sft_qgen.taxonomy import Leaf, load_taxonomy
from cuda_sft_qgen.models import SteeringBrief


def test_ampere_leaf_does_not_allow_hopper():
    leaf = Leaf(
        domain_id="T30",
        domain_name="arch",
        l2_id="sm80_cp_async",
        l2_name="sm80",
        l3_id="sm80_cp_async_l2_persist",
        claim="Ampere sm_80 起 L2 persisting window 通过 cudaStreamSetAttribute 设置",
        types=("optimization",),
        difficulty=(3,),
    )
    allowed = leaf_hardware_allowed(leaf)
    assert "ampere" in allowed
    assert "agnostic" in allowed
    assert "hopper" not in allowed


def test_tma_leaf_does_not_allow_ampere():
    leaf = Leaf(
        domain_id="T08",
        domain_name="hopper",
        l2_id="tma",
        l2_name="tma",
        l3_id="tma_align_128",
        claim="Hopper TMA 要求 128B 对齐",
        types=("debug",),
        difficulty=(3,),
    )
    allowed = leaf_hardware_allowed(leaf)
    assert "hopper" in allowed
    assert "blackwell" in allowed
    assert "ampere" not in allowed


def test_plan_briefs_hardware_matches_leaf_claim():
    settings = load_settings(fake_llm=True)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    briefs = plan_briefs(
        n=40,
        taxonomy=taxonomy,
        quotas=settings.quotas,
        personas=settings.personas,
        accepted=[],
        pending=[],
        steering=SteeringBrief(),
        evolver_fraction=0.0,
        seed_pool=[],
        rng=random.Random(0),
    )
    assert len(briefs) == 40
    for b in briefs:
        leaf = taxonomy.leaf_by_key(b.l3_id)
        assert leaf is not None
        allowed = leaf_hardware_allowed(leaf)
        assert b.hardware in allowed, (b.l3_id, b.hardware, allowed, leaf.claim[:80])
