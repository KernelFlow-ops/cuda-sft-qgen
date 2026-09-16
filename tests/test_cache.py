from __future__ import annotations

import numpy as np

from cuda_sft_qgen.cache import EmbeddingCache, PromptCache
from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.embeddings import FakeEmbedder, _chunks
from cuda_sft_qgen.models import QuestionBrief, SteeringBrief
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
from cuda_sft_qgen.review.normalize import exact_hash


def _brief(**kwargs) -> QuestionBrief:
    data = dict(
        topic_path=["T25", "occupancy", "occupancy_reg"],
        l3_id="occupancy_reg",
        l3_claim="occupancy = min(reg, smem, warp) limits",
        question_type="conceptual",
        difficulty=2,
        bloom="apply",
        persona="cuda_interviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
        angle="公式代入：给出参数，求出占用/事务数/offset/bank/字节",
        kind="formula",
        instance_hint="regs=64 block=256",
        must_include="regs=64",
    )
    data.update(kwargs)
    return QuestionBrief.model_validate(data)


def test_prompt_static_prefix_stable_across_claims():
    settings = load_settings(fake_llm=True)
    cache = PromptCache()
    sp = StructuredPrompt(settings, FewshotCorpus(), cache=cache)
    a = sp.build(_brief(), SteeringBrief())
    b = sp.build(
        _brief(l3_claim="different claim about occupancy vs registers", instance_hint="regs=128"),
        SteeringBrief(),
    )
    assert a.static_prefix
    assert a.static_prefix == b.static_prefix
    assert a.static_prefix in a.system
    assert "occupancy = min(reg, smem, warp) limits" not in a.static_prefix
    assert "different claim" not in b.static_prefix
    assert cache.static_hits >= 1


def test_embedding_cache_hit_and_model_isolation(tmp_path):
    path = tmp_path / "embed_cache.npz"
    cache = EmbeddingCache("model-a", path)
    fake = FakeEmbedder()
    fake.cache = cache
    v1 = fake.embed(["hello CUDA coalescing"])
    calls = fake.uncached_calls
    v2 = fake.embed(["hello CUDA coalescing"])
    assert fake.uncached_calls == calls
    assert np.allclose(v1, v2)
    assert cache.hits >= 1
    cache.save()

    warmed = EmbeddingCache("model-a", path)
    fake2 = FakeEmbedder()
    fake2.cache = warmed
    v3 = fake2.embed(["hello CUDA coalescing"])
    assert fake2.uncached_calls == 0
    assert np.allclose(v1, v3)

    other = EmbeddingCache("model-b")
    fake3 = FakeEmbedder()
    fake3.cache = other
    fake3.embed(["hello CUDA coalescing"])
    assert fake3.uncached_calls == 1
    assert exact_hash("hello CUDA coalescing")


def test_embed_chunks_by_size():
    parts = _chunks([str(i) for i in range(20)], 8)
    assert [len(p) for p in parts] == [8, 8, 4]
    assert sum(len(p) for p in parts) == 20
