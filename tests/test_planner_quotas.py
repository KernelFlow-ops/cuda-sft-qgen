"""Tests for quota-weighted brief sampling and steering residuals."""

from __future__ import annotations

import random

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.planner import plan_briefs
from cuda_sft_qgen.taxonomy import load_taxonomy


def _dummy(qid: str, l1: str, l2: str, l3: str) -> Question:
    return Question(
        id=qid,
        question=f"dummy question about {l3} with CUDA kernel occupancy?",
        tests_claim=l3,
        topic_path=[l1, l2, l3],
        question_type="conceptual",
        difficulty=1,
        bloom="remember",
        persona="cuda_interviewer",
        hardware="agnostic",
        granularity="api",
        language="zh",
    )


def test_planner_downweights_overrepresented_l1():
    settings = load_settings(fake_llm=True)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    accepted = [_dummy(f"q{i}", "T01", "qualifiers", "glob_dev_host") for i in range(40)]
    steering = SteeringBrief(
        overrepresented=["T01", "T01.qualifiers"],
        underrepresented=["T07", "T08"],
        preferred_types=["layout", "debug"],
        difficulty_nudge="up",
    )
    briefs = plan_briefs(
        n=40,
        taxonomy=taxonomy,
        quotas=settings.quotas,
        personas=settings.personas,
        accepted=accepted,
        pending=[],
        steering=steering,
        evolver_fraction=0.0,
        seed_pool=[],
        rng=random.Random(0),
    )
    t01 = sum(1 for b in briefs if b.topic_path[0] == "T01")
    boosted = sum(1 for b in briefs if b.topic_path[0] in {"T07", "T08"})
    assert t01 < 12
    # The joint planner balances the 30-domain scope before applying steering.
    assert boosted >= 3


def test_planner_does_not_emit_evolver_without_seeds():
    settings = load_settings(fake_llm=True)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    briefs = plan_briefs(
        n=10,
        taxonomy=taxonomy,
        quotas=settings.quotas,
        personas=settings.personas,
        accepted=[],
        pending=[],
        steering=SteeringBrief(),
        evolver_fraction=0.3,
        seed_pool=[],
        rng=random.Random(1),
    )
    assert all(b.role == "writer" for b in briefs)
    assert len(briefs) == 10


def test_planner_unique_l3_and_angles():
    settings = load_settings(fake_llm=True)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    briefs = plan_briefs(
        n=20,
        taxonomy=taxonomy,
        quotas=settings.quotas,
        personas=settings.personas,
        accepted=[],
        pending=[],
        steering=SteeringBrief(),
        evolver_fraction=0.0,
        seed_pool=[],
        rng=random.Random(2),
    )
    ids = [b.l3_id for b in briefs]
    assert len(ids) == len(set(ids))
    assert all(b.angle for b in briefs)
    assert all(not b.instance_hint or b.instance_hint in next(
        leaf.hints for leaf in taxonomy.leaves if leaf.l3_id == b.l3_id
    ) for b in briefs)


def test_planner_must_avoid_is_not_style_notes():
    settings = load_settings(fake_llm=True)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    notes = "Focus on qualifier-related and T10 concepts with simpler questions."
    briefs = plan_briefs(
        n=5,
        taxonomy=taxonomy,
        quotas=settings.quotas,
        personas=settings.personas,
        accepted=[],
        pending=[],
        steering=SteeringBrief(style_notes=notes),
        evolver_fraction=0.0,
        seed_pool=[],
        rng=random.Random(3),
    )
    assert briefs
    assert all(notes not in (b.must_avoid or "") for b in briefs)


def test_instance_hint_keeps_slash_fractions():
    from cuda_sft_qgen.planner import _instance_hint
    from cuda_sft_qgen.taxonomy import Leaf

    leaf = Leaf(
        domain_id="T29",
        domain_name="math",
        l2_id="mathdx_fft",
        l2_name="fft",
        l3_id="mathdx_fft_size_pow2",
        claim="FFTDx Size<> 必须是该 SM 目标支持的长度（常见 Size=2^k 及有限 3/5 因子）",
        types=("syntax",),
        difficulty=(2,),
        hints=("Size=2^k and finite 3/5 factors",),
        kind="formula",
    )
    hint = _instance_hint(leaf, random.Random(0))
    assert "3/5" in hint


def test_pick_angle_does_not_mismatch_kind():
    from cuda_sft_qgen.planner import _pick_angle, load_angles

    table = load_angles()
    rng = random.Random(0)
    for _ in range(40):
        card = _pick_angle(rng, "syntax", table, kind="formula")
        ask = str(card.get("ask") or "")
        kinds = card.get("kinds") or []
        assert "缺限定符" not in ask
        assert not kinds or "formula" in kinds
