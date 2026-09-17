"""Tests for curriculum pack load, ingest, and L2 frontier gaps."""

from pathlib import Path

from cuda_sft_qgen.pack import curriculum_frontier, ingest_pack, list_packs, load_pack
from cuda_sft_qgen.store.knowledge import KnowledgeStore
from cuda_sft_qgen.taxonomy import load_taxonomy


ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "configs" / "knowledge" / "packs"


def test_all_packs_load_and_dry_run(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite3")
    paths = list_packs(PACKS)
    assert {p.stem for p in paths} >= {
        "arch_compare", "hopper_blackwell", "formula_chains", "flash_attention",
        "op_memory", "ncu_workflow", "nsys_intro", "fused_ops",
    }
    for path in paths:
        pack = load_pack(path)
        dry = ingest_pack(store, pack, apply=False)
        assert dry["applied"] is False
        assert dry["would_add_claims"]
        assert store.list_claims() == []


def test_ingest_is_idempotent_and_keeps_verified(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite3")
    pack = load_pack(PACKS / "arch_compare.yaml")
    first = ingest_pack(store, pack, apply=True)
    assert first["added_claims"]
    assert first["added_relations"] >= 1
    n = len(store.list_claims())
    second = ingest_pack(store, pack, apply=True)
    assert second["added_claims"] == []
    assert len(store.list_claims()) == n
    cid = first["added_claims"][0]
    claim = store.get_claim(cid)
    url = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"
    store.apply_audit(cid, 1, verdict="supported", url=url, quote=claim["claim"], document=claim["claim"])
    assert store.get_claim(cid)["status"] == "verified"
    ingest_pack(store, pack, apply=True)
    assert store.get_claim(cid)["status"] == "verified"
    overlay = store.overlay_taxonomy(load_taxonomy(ROOT / "configs/taxonomy/cuda.yaml"))
    assert any(leaf.l3_id == cid for leaf in overlay.leaves)


def test_fused_ops_creates_t31_domain(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite3")
    ingest_pack(store, load_pack(PACKS / "fused_ops.yaml"), apply=True)
    tax = load_taxonomy(ROOT / "configs/taxonomy/cuda.yaml")
    overlay = store.overlay_taxonomy(tax)
    assert any(d["id"] == "T31" for d in overlay.domains)
    frontier = curriculum_frontier(store, load_pack(PACKS / "fused_ops.yaml"))
    assert frontier["l2"]


def test_ingest_can_skip_background_jobs(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite3")
    ingest_pack(store, load_pack(PACKS / "nsys_intro.yaml"), apply=True, enqueue_jobs=False)
    assert store.list_claims()
    assert store.stats()["jobs"] == {}


def test_expand_pack_unknown_and_relate_enqueue(tmp_path, fake_settings):
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker
    store = KnowledgeStore(tmp_path / "k.sqlite3")
    worker = KnowledgeWorker(store, llm=None, settings=fake_settings, taxonomy=None)
    result = worker.expand_pack("no_such_pack", apply=False)
    assert result["total_added"] == 0
    ingest_pack(store, load_pack(PACKS / "nsys_intro.yaml"), apply=True)
    queued = store.enqueue("relate", {"claim_id": store.list_claims()[0]["id"]}, "relate:test")
    assert queued


def test_taxonomy_yaml_loads_with_t31():
    tax = load_taxonomy(ROOT / "configs/taxonomy/cuda.yaml")
    assert any(leaf.domain_id == "T31" for leaf in tax.leaves)
    assert any(leaf.l2_id == "arch_compare" for leaf in tax.leaves)
    assert any(leaf.l2_id == "flash_fwd_fa2" for leaf in tax.leaves)
    assert any(leaf.l2_id == "ncu_workflow" for leaf in tax.leaves)
    assert any(leaf.l2_id == "nsys_intro" for leaf in tax.leaves)
    assert any(leaf.l2_id == "op_smem_budget" for leaf in tax.leaves)
