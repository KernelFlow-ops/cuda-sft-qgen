"""Joint planner (task T26; invariants I3, I4)."""
import random
import re
from collections import Counter

import pytest
import yaml

from fixv2_helpers import make_question, tiny_taxonomy


@pytest.fixture(scope="module")
def cfgs(repo_root):
    load = lambda n: yaml.safe_load((repo_root / "configs" / n).read_text(encoding="utf-8"))
    from cuda_sft_qgen.planner.sampling import load_angles
    return {"compat": load("compat_matrix.yaml"), "scope": load("scope.yaml"), "quotas": load("quotas.yaml"),
            "personas": load("personas.yaml")["personas"], "angles": load_angles(repo_root / "configs" / "angles.yaml")}


def _plan(cfgs, n=2000, seed=7, taxonomy=None, accepted=(), pending=(), coverage=None, **kw):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.planner.sampling import plan_briefs
    return plan_briefs(n=n, taxonomy=taxonomy or tiny_taxonomy(), quotas=cfgs["quotas"], personas=cfgs["personas"],
                       accepted=list(accepted), pending=list(pending), steering=SteeringBrief(), evolver_fraction=0.0,
                       seed_pool=[], rng=random.Random(seed), angles=cfgs["angles"], compat=cfgs["compat"],
                       scope=cfgs["scope"], coverage=coverage, joint=True, **kw)


def test_compatibility_invariant(cfgs):
    from cuda_sft_qgen.enums import ARCH_TO_HARDWARE
    from cuda_sft_qgen.planner.joint import allowed_arch_targets, l2_meta
    compat, tax = cfgs["compat"], tiny_taxonomy()
    meta, leaves = l2_meta(tax), {lf.l3_id: lf for lf in tax.leaves}
    for b in _plan(cfgs):
        assert b.question_type in compat["kind_to_types"][b.kind]
        assert b.answer_format in compat["type_to_formats"][b.question_type]
        assert b.bloom in compat["difficulty_to_bloom"][str(b.difficulty)]
        leaf = leaves[b.l3_id]
        allowed = allowed_arch_targets(leaf, meta[(leaf.domain_id, leaf.l2_id)].get("arch") or ["any"], compat["arch_claim_rules"])
        assert b.arch_target in allowed, (b.l3_id, b.arch_target, allowed)
        assert b.hardware == ARCH_TO_HARDWARE[b.arch_target]
        assert b.scenario in (compat["domain_to_scenarios"].get(leaf.domain_id) or compat["domain_to_scenarios"]["default"])
        assert b.difficulty in leaf.difficulty
        combo = {"kind": b.kind, "question_type": b.question_type, "answer_format": b.answer_format,
                 "difficulty": b.difficulty, "bloom": b.bloom, "granularity": b.granularity}
        for rule in compat["forbidden_combinations"]:
            assert not all(combo.get(k) == v for k, v in rule.items() if k != "note")


def test_arch_rules_first_match(cfgs):
    from cuda_sft_qgen.planner.joint import allowed_arch_targets
    tax = tiny_taxonomy()
    tma = next(lf for lf in tax.leaves if lf.l3_id == "tma_encode")
    assert set(allowed_arch_targets(tma, ["sm_90", "sm_90a", "sm_100a"], cfgs["compat"]["arch_claim_rules"])) <= {"sm_90", "sm_90a", "sm_100a"}
    cp = next(lf for lf in tax.leaves if lf.l3_id == "cp_async_sizes")
    assert "sm_90a" not in allowed_arch_targets(cp, ["sm_80", "sm_86", "sm_89"], cfgs["compat"]["arch_claim_rules"])


def test_instance_hint_only_from_hints(cfgs):
    tax = tiny_taxonomy()
    hints = {lf.l3_id: set(lf.hints) for lf in tax.leaves}
    for b in _plan(cfgs, n=400):
        assert b.instance_hint == "" or b.instance_hint in hints[b.l3_id]
        if not hints[b.l3_id]:
            assert b.instance_hint == ""


def test_exploration_share(cfgs):
    briefs = _plan(cfgs, n=2000)
    share = sum(b.exploration for b in briefs) / len(briefs)
    assert 0.06 <= share <= 0.14
    assert all(b.l3_id == "tma_expect_tx" for b in briefs if b.exploration)
    assert all(not b.exploration for b in briefs if b.l3_id != "tma_expect_tx")


def test_hard_max_blocks_leaf(cfgs):
    acc = [make_question(id=f"a{i}", l3_id="smem_bank_gcd", topic_path=["T02", "smem", "smem_bank_gcd"]) for i in range(12)]
    assert all(b.l3_id != "smem_bank_gcd" for b in _plan(cfgs, n=40, accepted=acc))


def test_stems_for_leaf_full_text():
    from cuda_sft_qgen.planner.sampling import _stems_for_leaf
    long_q = "x" * 700
    acc = [make_question(id="a", l3_id="L", question=long_q)]
    pend = [make_question(id="p", l3_id="L", question="pending stem about __shared__ banks")]
    stems = _stems_for_leaf(acc, pend, "L")
    assert len(stems) == 2 and len(stems[0]) == 600 and stems[1].startswith("pending")


def test_mixup_anchor_is_low_parent(cfgs, monkeypatch):
    from types import SimpleNamespace
    from cuda_sft_qgen.planner import sampling
    low = make_question(id="low", l3_id="smem_bank_gcd", topic_path=["T02", "smem", "smem_bank_gcd"], difficulty=2)
    high = make_question(id="high", l3_id="smem_pad", topic_path=["T02", "smem", "smem_pad"], difficulty=4)
    pair = SimpleNamespace(low=low, high=high, pair_id="pair:high:low")
    monkeypatch.setattr(sampling, "_mixup_specs", lambda *a, **k: [(pair, "hybrid")])
    monkeypatch.setattr(sampling, "mixup_payload", lambda p, m: {
        "mode": m, "pair_id": "pair:high:low", "parent_ids": ["low", "high"], "parent_difficulties": [2, 4],
        "difficulty_target": 4, "difficulty_raw_score": 3.0, "difficulty_policy": "x", "parents": []})
    briefs = _plan(cfgs, n=1, seed=1)
    assert briefs[0].l3_id == "smem_bank_gcd" and briefs[0].role == "evolver"


def test_joint_disabled_uses_legacy_path(cfgs):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.planner.sampling import plan_briefs
    legacy = plan_briefs(n=50, taxonomy=tiny_taxonomy(), quotas=cfgs["quotas"], personas=cfgs["personas"], accepted=[],
                         pending=[], steering=SteeringBrief(), evolver_fraction=0.0, seed_pool=[], rng=random.Random(3),
                         angles=cfgs["angles"], compat=cfgs["compat"], scope=cfgs["scope"], joint=False)
    assert len(legacy) == 50 and all(b.arch_target == "any" and b.scenario == "generic" for b in legacy)


def test_persona_matches_domain(cfgs):
    for b in _plan(cfgs, n=500):
        domains = cfgs["personas"][b.persona]["domains"]
        assert b.topic_path[0] in domains, (b.persona, b.topic_path[0])


def test_distribution_moves_toward_quota(cfgs):
    fmts = Counter(b.answer_format for b in _plan(cfgs, n=2000))
    assert len(fmts) >= 7
