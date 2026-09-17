"""Tests for TaxonomyManager health analysis, expansion, and ticks."""

from __future__ import annotations

import random

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.embeddings import FakeEmbedder
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.prompts import load_prompt, load_prompt_yaml
from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.planner import plan_briefs
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy, load_taxonomy, merge_leaves, set_leaf_retired
from cuda_sft_qgen.taxonomy_embed import TaxonomyEmbedIndex
from cuda_sft_qgen.taxonomy_expand import validate_leaf
from cuda_sft_qgen.taxonomy_manager import TaxonomyManager, load_facets


def _leaf(**kwargs) -> Leaf:
    data = dict(
        domain_id="T01",
        domain_name="prog",
        l2_id="sparse",
        l2_name="Sparse",
        l3_id="sparse_a",
        claim="CUDA provides several memory types",
        types=("conceptual",),
        difficulty=(1,),
        hints=(),
        kind="api",
        retired=False,
    )
    data.update(kwargs)
    return Leaf(**data)


def _tiny_taxonomy() -> Taxonomy:
    domains = [
        {
            "id": "T01",
            "name": "prog",
            "children": [
                {"id": "empty_l2", "name": "Empty", "leaves": []},
                {
                    "id": "sparse",
                    "name": "Sparse",
                    "leaves": [
                        {
                            "id": "sparse_a",
                            "claim": "CUDA provides several memory types",
                            "types": ["conceptual"],
                            "difficulty": [1],
                            "kind": "api",
                        }
                    ],
                },
            ],
        }
    ]
    leaves = [_leaf()]
    return Taxonomy(version=1, name="tiny", leaves=leaves, domains=domains)


def test_taxonomy_prompts_and_facets_load():
    curator = load_prompt("curator")
    assert "NEVER output `question`" in curator or "NEVER output" in curator
    assert load_prompt("expand")
    assert load_prompt("health")
    assert load_prompt("gap_search")
    assert load_prompt("refine")
    assert load_prompt("merge")
    facets = load_facets()
    ids = [str(x.get("id")) for x in facets]
    assert ids == [
        "formula",
        "failure",
        "feature",
        "mechanism",
        "api",
        "numeric",
        "layout",
        "profiling",
        "arch_migrate",
        "cross_factor",
    ]
    raw = load_prompt_yaml("facets")
    assert raw.get("facets")


def test_analyze_health_prioritizes_empty_l2():
    mgr = TaxonomyManager(_tiny_taxonomy(), FakeLLM())
    health = mgr.analyze_health()
    assert "T01.empty_l2" in health.empty_l2_nodes
    assert health.expansion_priority[0] == "T01.empty_l2"
    assert "sparse_a" in health.leaves_with_generic_claims
    assert health.total_leaves == 1


def test_retired_leaves_ignored_for_coverage():
    tax = _tiny_taxonomy()
    assert set_leaf_retired(tax, "sparse_a", True)
    mgr = TaxonomyManager(tax, FakeLLM())
    health = mgr.analyze_health()
    assert "T01.sparse" in health.empty_l2_nodes
    assert any(lf.retired for lf in tax.leaves if lf.l3_id == "sparse_a")


def test_planner_skips_retired_leaves():
    settings = load_settings(fake_llm=True)
    tax = load_taxonomy(settings.taxonomy_path)
    live = next(lf for lf in tax.leaves if not lf.retired)
    assert set_leaf_retired(tax, live.l3_id, True)
    briefs = plan_briefs(
        n=24,
        taxonomy=tax,
        quotas=settings.quotas,
        personas=settings.personas,
        accepted=[],
        pending=[],
        steering=SteeringBrief(),
        evolver_fraction=0.0,
        seed_pool=[],
        rng=random.Random(0),
    )
    assert briefs
    assert all(b.l3_id != live.l3_id for b in briefs)


def test_validate_leaf_requires_l2_prefix():
    reason = validate_leaf(
        {"l3_id": "tmem", "claim": "TMEM is a Hopper tensor memory structure with a size formula = 512KB"},
        l2_id="layout_algebra",
        existing_ids=set(),
        existing_claims=[],
    )
    assert reason == "bad_id"


def test_merge_leaves_does_not_create_l2_by_default():
    tax = _tiny_taxonomy()
    n = merge_leaves(
        tax,
        [
            {
                "domain_id": "T01",
                "l2_id": "brand_new",
                "l3_id": "brand_new_foo",
                "claim": "offset = inner_product(c, stride) for a CuTe layout",
                "kind": "formula",
            }
        ],
        allow_new_domains=False,
    )
    assert n == 0
    assert all(c["id"] != "brand_new" for d in tax.domains for c in d["children"])


def test_embed_filter_drops_identical_claim(tmp_path):
    existing = _leaf(
        l3_id="sparse_tma",
        l2_id="tma",
        claim="TMA box stride must be 128B aligned on sm_90",
        kind="feature",
    )
    idx = TaxonomyEmbedIndex(tmp_path / "taxonomy_embed.npz", FakeEmbedder())
    n = idx.rebuild([existing])
    assert n == 1
    keep, dropped, gray = idx.filter_candidates(
        [
            {
                "l3_id": "tma_dup",
                "l2_id": "tma",
                "kind": "feature",
                "claim": "TMA box stride must be 128B aligned on sm_90",
            }
        ]
    )
    assert dropped
    assert not keep


def test_embed_filter_intra_batch_without_index(tmp_path):
    idx = TaxonomyEmbedIndex(tmp_path / "taxonomy_embed.npz", FakeEmbedder())
    claim = "TMA box stride must be 128B aligned on sm_90"
    keep, dropped, _gray = idx.filter_candidates(
        [
            {
                "l3_id": "tma_a",
                "l2_id": "tma",
                "kind": "feature",
                "claim": claim,
            },
            {
                "l3_id": "tma_b",
                "l2_id": "tma",
                "kind": "feature",
                "claim": claim,
            },
        ]
    )
    assert len(keep) == 1
    assert len(dropped) == 1


def test_embed_filter_keeps_distinct_claim(tmp_path):
    existing = _leaf(
        l3_id="sparse_tma",
        l2_id="tma",
        claim="TMA box stride must be 128B aligned on sm_90",
        kind="feature",
    )
    idx = TaxonomyEmbedIndex(tmp_path / "taxonomy_embed.npz", FakeEmbedder())
    idx.rebuild([existing])
    keep, dropped, _gray = idx.filter_candidates(
        [
            {
                "l3_id": "tma_expect_tx",
                "l2_id": "tma",
                "kind": "mechanism",
                "claim": "mbarrier expect_tx counts TMA bytes; wait before that count hangs",
            }
        ]
    )
    assert keep
    assert not dropped


def test_expand_strategic_caps_llm_calls():
    tax = _tiny_taxonomy()
    mgr = TaxonomyManager(tax, FakeLLM(), target_leaves_per_l2=30)
    result = mgr.expand_strategic(
        max_leaves_to_add=40, max_llm_calls=1, apply=False
    )
    assert len(result["expansions"]) <= 1


def test_validate_and_filter_coerces_kind_used_as_type():
    tax = _tiny_taxonomy()
    mgr = TaxonomyManager(tax, FakeLLM())
    pair = mgr._lookup_l2("T01.sparse")
    assert pair
    domain, child = pair
    out = mgr._validate_and_filter(
        [
            {
                "l3_id": "memcpy_overlap_race",
                "claim": "cudaMemcpyAsync on two streams without event wait races with the consumer kernel",
                "kind": "failure",
                "types": ["failure", "debug"],
            }
        ],
        domain=domain,
        child=child,
        max_new=5,
    )
    assert out
    assert out[0]["types"] == ["debug"]
    assert out[0]["l3_id"].startswith("sparse_")


def test_validate_and_filter_prefixes_l3_id():
    tax = _tiny_taxonomy()
    mgr = TaxonomyManager(tax, FakeLLM())
    pair = mgr._lookup_l2("T01.sparse")
    assert pair
    domain, child = pair
    out = mgr._validate_and_filter(
        [
            {
                "l3_id": "foo_offset",
                "claim": "offset = inner_product(c, stride) for a CuTe layout",
                "kind": "formula",
                "types": ["layout"],
            }
        ],
        domain=domain,
        child=child,
        max_new=5,
    )
    assert out
    assert out[0]["l3_id"] == "sparse_foo_offset"


def test_apply_refine_patches_claim():
    tax = _tiny_taxonomy()
    mgr = TaxonomyManager(tax, FakeLLM())
    n = mgr.apply_refine(
        {
            "suggestions": [
                {
                    "l3_id": "sparse_a",
                    "suggested_claim": (
                        "cudaMemcpy kind must match src/dst; HostToDevice on a "
                        "device pointer is illegal"
                    ),
                    "suggested_hints": ["cudaMemcpyHostToDevice"],
                }
            ]
        },
        apply=False,
    )
    assert n == 1
    leaf = tax.leaf_by_key("sparse_a")
    assert leaf is not None
    assert "cudaMemcpy" in leaf.claim
    assert "cudaMemcpyHostToDevice" in leaf.hints


def test_rebuild_save_false_does_not_write(tmp_path):
    idx = TaxonomyEmbedIndex(tmp_path / "taxonomy_embed.npz", FakeEmbedder())
    n = idx.rebuild([_leaf()], save=False)
    assert n == 1
    assert not (tmp_path / "taxonomy_embed.npz").exists()


def test_health_score_not_zero_on_full_tree():
    settings = load_settings(fake_llm=True)
    tax = load_taxonomy(settings.taxonomy_path)
    mgr = TaxonomyManager(tax, FakeLLM())
    health = mgr.analyze_health()
    assert health.health_score > 0
    assert health.kind_imbalance is False


def test_run_tick_max_leaves_zero_skips_llm_add():
    tax = _tiny_taxonomy()
    mgr = TaxonomyManager(tax, FakeLLM(), target_leaves_per_l2=30)
    before = len(tax.leaves)
    result = mgr.run_tick(apply=False, fake_llm=False, max_leaves=0)
    assert result["total_added"] == 0
    assert result["llm_calls"] == 0
    assert len(tax.leaves) == before


def test_run_tick_fake_llm_does_not_add_leaves(tmp_path):
    tax = _tiny_taxonomy()
    mgr = TaxonomyManager(tax, FakeLLM())
    before = len(tax.leaves)
    result = mgr.run_tick(
        accepted=[],
        rejected=[],
        apply=True,
        data_dir=tmp_path,
        fake_llm=True,
    )
    assert result["total_added"] == 0
    assert result["llm_calls"] == 0
    assert result["mission"] == "expand"
    assert len(tax.leaves) == before
    assert (tmp_path / "taxonomy_health.json").exists()
