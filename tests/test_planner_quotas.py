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
    assert boosted >= 8


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
    assert all(b.instance_hint for b in briefs)

