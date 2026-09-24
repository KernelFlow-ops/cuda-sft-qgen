"""Pair mining and MathMixup-style planner provenance tests."""

from __future__ import annotations

import numpy as np

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.planner import plan_briefs
from cuda_sft_qgen.planner.adaptive import mixup_difficulty_report
from cuda_sft_qgen.planner.pair_mining import (
    cuda_difficulty_features,
    mine_pairs,
    monotonicity_ok,
)
from cuda_sft_qgen.taxonomy import load_taxonomy


def _question(qid: str, difficulty: int, *, text: str = "shared memory coalescing") -> Question:
    return Question(
        id=qid,
        question=text,
        tests_claim="shared memory coalescing",
        topic_path=["T01", "memory", "shared"],
        l3_id="shared",
        question_type="conceptual",
        difficulty=difficulty,
        bloom="understand",
        persona="cuda_interviewer",
        hardware="agnostic",
        granularity="kernel",
        language="en",
    )


def test_pair_mining_uses_topk_cosine_and_difficulty_gap():
    low = _question("low", 1)
    high = _question("high", 4)
    unrelated = _question("unrelated", 2, text="completely different launch API")

    class Embedder:
        def embed(self, texts):
            return np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]][: len(texts)])

    pairs = mine_pairs(
        [low, high, unrelated],
        embedder=Embedder(),
        min_similarity=0.8,
        max_similarity=1.0,
        top_k=2,
        max_pairs=2,
    )
    assert len(pairs) == 1
    assert pairs[0].low.id == "low"
    assert pairs[0].high.id == "high"
    assert pairs[0].similarity > 0.99


def test_cuda_difficulty_features_are_explainable():
    easy = cuda_difficulty_features("Which API launches this kernel?")
    hard = cuda_difficulty_features(
        "If a Hopper kernel uses TMA and mbarrier across two streams, prove the ordering "
        "under these constraints and compile the __global__ code."
    )
    assert hard.score > easy.score
    assert hard.dependency_count >= 2
    assert hard.architecture_count >= 1
    assert hard.code_signal == 1


def test_mixup_briefs_encode_two_modes_and_parent_provenance():
    settings = load_settings(fake_llm=True)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    leaf = next(item for item in taxonomy.leaves if not item.retired)
    low = _question("low", 1)
    high = _question("high", 4)
    low = low.model_copy(update={"l3_id": leaf.l3_id, "topic_path": list(leaf.path)})
    high = high.model_copy(update={"l3_id": leaf.l3_id, "topic_path": list(leaf.path)})
    briefs = plan_briefs(
        n=2,
        taxonomy=taxonomy,
        quotas=settings.quotas,
        personas=settings.personas,
        accepted=[low, high],
        pending=[],
        steering=SteeringBrief(),
        evolver_fraction=0.0,
        seed_pool=[low, high],
        mixup_config={
            "enabled": True,
            "fraction": 1.0,
            "modes": ["decomposed", "hybrid"],
            "pair_similarity_min": 0.8,
            "pair_similarity_max": 1.0,
            "max_pairs": 1,
            "max_pairs_per_item": 1,
        },
    )
    assert len(briefs) == 2
    assert {b.role for b in briefs} == {"evolver"}
    assert all(str(b.seed_id).startswith("pair:") for b in briefs)
    modes = {b.knowledge_context["mixup"]["mode"] for b in briefs}
    assert modes == {"decomposed", "hybrid"}
    assert all(len(b.knowledge_context["mixup"]["parent_ids"]) == 2 for b in briefs)
    assert all(len(b.knowledge_context["mixup"]["parents"]) == 2 for b in briefs)
    assert all(b.parent_ids == b.knowledge_context["mixup"]["parent_ids"] for b in briefs)
    assert all("MIXUP_MODE=" in b.must_include for b in briefs)


def test_mixup_difficulty_report_and_monotonicity():
    samples = [
        _question("d", 2).model_copy(update={
            "mixup_mode": "decomposed", "parent_ids": ["p1", "p2"],
            "parent_difficulties": [1, 3],
        }),
        _question("h", 4).model_copy(update={
            "mixup_mode": "hybrid", "parent_ids": ["p1", "p2"],
            "parent_difficulties": [1, 3],
        }),
    ]
    report = mixup_difficulty_report(samples)
    assert report["complete_groups"] == 1
    assert report["monotonicity_rate"] == 1.0
    assert monotonicity_ok(2, 3, 4)
    assert not monotonicity_ok(3, 2, 4)


def test_pair_filters_language_granularity_revision_and_reuse():
    low = _question("low", 1).model_copy(update={"claim_revision": 2})
    high = _question("high", 4).model_copy(update={"claim_revision": 2})
    different_language = high.model_copy(update={"id": "en", "language": "zh"})
    stale = high.model_copy(update={"id": "stale", "claim_revision": 1})

    class Embedder:
        def embed(self, texts):
            return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    active = {"shared": type("Claim", (), {"revision": 2, "retired": False, "status": "verified"})()}
    assert mine_pairs(
        [low, different_language], embedder=Embedder(), min_similarity=0.8,
        max_similarity=1.0, active_claims=active, same_language=True,
    ) == []
    pairs = mine_pairs(
        [low, high, stale], embedder=Embedder(), min_similarity=0.8,
        max_similarity=1.0, active_claims=active, parent_reuse_counts={"high": 3},
        max_parent_reuse=3,
    )
    assert len(pairs) == 0
