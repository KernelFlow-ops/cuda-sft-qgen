"""P0 planner regressions: legacy expiry cliff, forbidden-combination fallback,
saturated leaf cap, mixup operator, and brief-level reject coverage."""
from __future__ import annotations

import logging
import random
from collections import Counter

import pytest

from p0_helpers import leaf_item, make_question, planner_cfgs, taxonomy


@pytest.fixture(scope="module")
def cfgs():
    return planner_cfgs()


def _plan(cfgs, tax, *, n, seed=3, compat=None, accepted=(), **kw):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.planner.sampling import plan_briefs

    return plan_briefs(n=n, taxonomy=tax, quotas=cfgs["quotas"], personas=cfgs["personas"], accepted=list(accepted),
                       pending=[], steering=SteeringBrief(), evolver_fraction=0.0, seed_pool=[],
                       rng=random.Random(seed), angles=cfgs["angles"], compat=compat or cfgs["compat"],
                       scope=cfgs["scope"], joint=True, **kw)


# --- item 1: audit_due_at never removes a legacy leaf from the main pool -------------------------

def test_overdue_legacy_leaf_is_planned_in_main_pool(cfgs):
    tax = taxonomy(
        [leaf_item("smem_gcd", "formula", ["conceptual", "layout"], "ways = gcd(S,32) for 4B words"),
         leaf_item("smem_overdue", "formula", ["conceptual"], "legacy ways = gcd(S,32)",
                   status="legacy_unreviewed", audit_due_at=1.0)],
        [leaf_item("tma_tx", "numeric", ["conceptual"], "expect_tx counts full box bytes incl OOB; mbarrier",
                   status="provisional")],
    )
    briefs = _plan(cfgs, tax, n=300)
    overdue = [b for b in briefs if b.l3_id == "smem_overdue"]
    assert overdue and not any(b.exploration for b in overdue)
    # Provisional leaves are reachable only through exploration, and exploration is provisional only.
    assert all(b.exploration for b in briefs if b.l3_id == "tma_tx")
    assert {b.l3_id for b in briefs if b.exploration} <= {"tma_tx"}


def test_all_overdue_legacy_taxonomy_stays_plannable(cfgs):
    """14 days after the first run every legacy claim is overdue; planning must not collapse."""
    tax = taxonomy([leaf_item(f"legacy_{i}", "formula", ["conceptual"], f"legacy claim {i} ways = gcd(S,32)",
                              status="legacy_unreviewed", audit_due_at=1.0) for i in range(3)])
    briefs = _plan(cfgs, tax, n=12)
    assert len(briefs) == 12
    assert not any(b.exploration for b in briefs)


def test_retired_and_quarantined_leaves_are_never_planned(cfgs):
    tax = taxonomy([leaf_item("live", "formula", ["conceptual"], "ways = gcd(S,32) for 4B words"),
                    leaf_item("gone", "formula", ["conceptual"], "retired gcd claim", status="retired"),
                    leaf_item("bad", "formula", ["conceptual"], "quarantined gcd claim", status="quarantined")])
    assert {b.l3_id for b in _plan(cfgs, tax, n=20)} == {"live"}


# --- item 7a: forbidden combinations are never returned when a valid one exists -------------------

def _axes(cfgs, leaf, *, compat=None, type_weights=None, n=200, seed=11):
    from cuda_sft_qgen.planner import joint

    quotas = cfgs["quotas"]
    weights = joint.AxisWeights(
        type=type_weights or dict(quotas["question_types"]),
        difficulty={str(k): v for k, v in quotas["difficulty"].items()},
        bloom=dict(quotas["bloom"]), granularity=dict(quotas["granularity"]), language=dict(quotas["language"]),
        arch=dict(quotas["arch_target"]), format=dict(quotas["answer_format"]))
    rng = random.Random(seed)
    return [joint.sample_axes(leaf, rng=rng, weights=weights, compat=compat or cfgs["compat"],
                              angles=cfgs["angles"], personas=cfgs["personas"], l2_arch=["sm_90", "sm_90a"],
                              coverage=joint.LeafCoverage(), role="writer", generic_share=0.0, generic_cap=1.0)
            for _ in range(n)]


def _combo(leaf, axes):
    return {"kind": leaf.kind, "question_type": axes.question_type, "answer_format": axes.answer_format,
            "difficulty": axes.difficulty, "bloom": axes.bloom, "granularity": axes.granularity}


def _interleave_leaf():
    # Mirrors tma_encode_interleave: an api leaf whose only native api type is syntax, at difficulty 4.
    tax = taxonomy([], [leaf_item("tma_interleave", "api", ["layout", "syntax"],
                                  "TMA interleave/swizzle modes are a hardware enum", difficulty=(4,))])
    return tax.leaves[0]


def test_always_forbidden_leaf_gets_valid_axes(cfgs):
    from cuda_sft_qgen.planner.joint import violates

    leaf = _interleave_leaf()
    draws = _axes(cfgs, leaf)
    assert not any(violates(_combo(leaf, a), cfgs["compat"]["forbidden_combinations"]) for a in draws)
    assert all(a.difficulty == 4 for a in draws)
    assert all(a.question_type in cfgs["compat"]["kind_to_types"]["api"] for a in draws)
    assert all(a.answer_format in cfgs["compat"]["type_to_formats"][a.question_type] for a in draws)


def test_exact_fallback_uses_the_sampling_weights(cfgs):
    leaf = _interleave_leaf()
    weights = {name: 0.0 for name in cfgs["quotas"]["question_types"]}
    weights.update(syntax=1.0, debug=1.0)          # syntax@4 is forbidden, so only debug remains valid
    assert {a.question_type for a in _axes(cfgs, leaf, type_weights=weights, n=60)} == {"debug"}


def test_no_valid_combination_logs_and_returns_fewest_violations(cfgs, caplog):
    leaf = _interleave_leaf()
    compat = dict(cfgs["compat"], forbidden_combinations=[{"kind": "api"}, {"question_type": "syntax", "difficulty": 4}])
    with caplog.at_level(logging.WARNING, logger="cuda_sft_qgen.planner.joint"):
        draws = _axes(cfgs, leaf, compat=compat, n=3)
    assert len(draws) == 3 and all(a.difficulty == 4 for a in draws)
    assert "planner_forbidden_fallback" in caplog.text
    # Only the unavoidable {"kind": "api"} rule is violated when a non-syntax type exists.
    assert all(a.question_type != "syntax" for a in draws)


# --- item 7b: saturated pool falls back to the least-saturated leaves -----------------------------

def test_all_saturated_falls_back_to_least_saturated_leaf(cfgs, caplog):
    tax = taxonomy([leaf_item("full_a", "formula", ["conceptual"], "gcd claim a ways = gcd(S,32)"),
                    leaf_item("full_b", "formula", ["conceptual"], "gcd claim b ways = gcd(S,32)"),
                    leaf_item("least", "formula", ["conceptual"], "gcd claim c ways = gcd(S,32)")])
    accepted = [make_question(id=f"{leaf}_{i}", l3_id=leaf, topic_path=["T02", "smem", leaf])
                for leaf, count in (("full_a", 14), ("full_b", 13), ("least", 12)) for i in range(count)]
    compat = dict(cfgs["compat"], provisional_exploration_share=0.0)
    with caplog.at_level(logging.WARNING, logger="cuda_sft_qgen.planner.sampling"):
        for seed in range(5):
            assert _plan(cfgs, tax, n=1, seed=seed, compat=compat, accepted=accepted)[0].l3_id == "least"
    assert "planner_all_saturated" in caplog.text
    # The overflow is spread: once "least" ties with full_b, both share the next briefs.
    spread = Counter(b.l3_id for b in _plan(cfgs, tax, n=3, compat=compat, accepted=accepted))
    assert spread["full_a"] == 0 and spread["least"] == 2 and spread["full_b"] == 1


# --- item 9: mixup briefs carry parent ids and no evolver operator --------------------------------

def test_mixup_brief_has_no_operator_and_keeps_parent_ids(cfgs, monkeypatch):
    from types import SimpleNamespace

    from cuda_sft_qgen.planner import sampling

    tax = taxonomy([leaf_item("smem_gcd", "formula", ["conceptual", "layout"], "ways = gcd(S,32) for 4B words"),
                    leaf_item("smem_pad", "failure", ["debug"], "tile[tid][k] on 32x32 float is 32-way conflicted")])
    low = make_question(id="low", l3_id="smem_gcd", topic_path=["T02", "smem", "smem_gcd"], difficulty=2)
    high = make_question(id="high", l3_id="smem_pad", topic_path=["T02", "smem", "smem_pad"], difficulty=4)
    pair = SimpleNamespace(low=low, high=high, pair_id="pair:high:low")
    monkeypatch.setattr(sampling, "_mixup_specs", lambda *a, **k: [(pair, "hybrid")])
    monkeypatch.setattr(sampling, "mixup_payload", lambda p, m: {
        "mode": m, "pair_id": "pair:high:low", "parent_ids": ["low", "high"], "parent_difficulties": [2, 4],
        "difficulty_target": 4, "difficulty_raw_score": 3.0, "difficulty_policy": "x", "parents": []})
    brief = _plan(cfgs, tax, n=1)[0]
    assert brief.role == "evolver" and brief.mixup_mode == "hybrid"
    assert brief.operator is None
    assert brief.parent_ids == ["low", "high"]


# --- item 20a: brief-level Chair rejects steer the joint sampler ----------------------------------

def test_coverage_history_counts_brief_level_rejects(tmp_path):
    from cuda_sft_qgen.store.archive import Archive

    store = Archive(tmp_path)
    common = dict(l3_id="smem_gcd", topic_path=["T02", "smem", "smem_gcd"], arch_target="sm_80", hardware="ampere")
    store.append_accepted(make_question(id="ok", angle_id="con_why_constraint", **common))
    for qid, codes, angle in (("brief_a", ["claim_alignment"], "con_plug_numbers"),
                              ("brief_b", ["hardware", "metadata"], "con_plug_numbers"),
                              ("writer", ["correctness"], "lay_offset"),
                              ("mixed", ["claim_alignment", "leakage"], "lay_offset"),
                              ("sieve", [], "lay_offset")):
        rejected = make_question(id=qid, angle_id=angle, defect_codes=codes, **common)
        store.append_rejected(rejected)
    cov = store.coverage_history()["smem_gcd"]
    assert cov.count == 1                                     # rejects never count as coverage
    assert cov.angles["con_plug_numbers"] == 2 and cov.angles["lay_offset"] == 0
    assert cov.archs["sm_80"] == 3                            # 1 accepted + 2 brief-level rejects
    assert cov.formats == Counter() and cov.scenarios["generic"] == 1


def test_brief_level_rejects_lower_the_angle_weight(cfgs, tmp_path):
    from cuda_sft_qgen.planner.joint import LeafCoverage

    leaf = taxonomy([leaf_item("smem_gcd", "formula", ["conceptual"], "ways = gcd(S,32) for 4B words")]).leaves[0]
    failed = LeafCoverage()
    failed.angles["con_plug_numbers"] = 5

    def rate(coverage):
        from cuda_sft_qgen.planner import joint

        quotas = cfgs["quotas"]
        weights = joint.AxisWeights(type={"conceptual": 1.0}, difficulty={"2": 1.0, "3": 1.0},
                                    bloom=dict(quotas["bloom"]), granularity=dict(quotas["granularity"]),
                                    language=dict(quotas["language"]), arch=dict(quotas["arch_target"]),
                                    format=dict(quotas["answer_format"]))
        rng = random.Random(4)
        cards = [joint.sample_axes(leaf, rng=rng, weights=weights, compat=cfgs["compat"], angles=cfgs["angles"],
                                   personas=cfgs["personas"], l2_arch=["any"], coverage=coverage, role="writer",
                                   generic_share=0.0, generic_cap=1.0).card["id"] for _ in range(300)]
        return cards.count("con_plug_numbers") / len(cards)

    assert rate(failed) < rate(LeafCoverage())
