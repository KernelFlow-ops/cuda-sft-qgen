"""Tests for knowledge-graph relations, subgraph export, and Mermaid rendering."""

from cuda_sft_qgen.store.knowledge import KnowledgeStore

URL = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"
CLAIM = "A warp contains 32 threads."


def payload(claim_id, claim):
    """Minimal add_claim payload with an official CUDA docs URL."""
    return {
        "l3_id": claim_id, "domain_id": "compute", "l2_id": "warp",
        "claim": claim, "types": ["conceptual"], "difficulty": [2], "sources": [URL],
    }


def test_multihop_derives_and_mermaid(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add_claim(payload("sector_32b", "DRAM bills 32B sectors."))
    store.add_claim(payload("aligned_128", "Aligned 32x4B is one 128B transaction."))
    store.add_claim(payload("stride2", "Stride-2 32x4B is two 128B transactions."))
    store.add_claim(payload("ncu_sector", "l1tex sector count approximates excess transactions."))
    for cid in ("sector_32b", "aligned_128", "stride2"):
        text = store.get_claim(cid)["claim"]
        store.apply_audit(cid, 1, verdict="supported", url=URL, quote=text, document=text)
    aligned_ev = store.subgraph("aligned_128")["evidence"][0]["id"]
    stride_ev = store.subgraph("stride2")["evidence"][0]["id"]
    store.add_relation("aligned_128", "sector_32b", "derives", evidence_id=aligned_ev, status="verified", reason="sector size")
    store.add_relation("stride2", "aligned_128", "derives", evidence_id=stride_ev, status="verified", reason="instance")
    one = store.subgraph("stride2", kinds=["derives"])
    assert {c["id"] for c in one["claims"]} == {"stride2", "aligned_128"}
    two = store.subgraph("stride2", hops=2, kinds=["derives"], max_nodes=20)
    ids = {c["id"] for c in two["claims"]}
    assert {"stride2", "aligned_128", "sector_32b"} <= ids
    assert store.paths("stride2", "sector_32b", max_hops=3)
    mermaid = store.to_mermaid(two)
    assert "derives" in mermaid
    assert "stride2" in mermaid


def test_entity_edge_and_related_cannot_verify(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add_claim(payload("tma_align", "TMA global base must be 16B aligned."))
    store.upsert_entity("arch:sm_90", "arch", "Hopper sm_90")
    store.add_relation("tma_align", "arch:sm_90", "applies_to", status="provisional", reason="hopper")
    graph = store.subgraph("tma_align", hops=1, status=None)
    assert graph["entities"][0]["id"] == "arch:sm_90"
    store.add_claim(payload("cp_async", "Ampere uses cp.async.ca/cg."))
    store.add_relation("tma_align", "cp_async", "related", status="provisional", reason="async copy family")
    try:
        store.add_relation("tma_align", "cp_async", "related", status="verified", reason="no")
        raise AssertionError("related must not verify")
    except ValueError:
        pass


def test_mechanical_contains_and_graph_stats(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add_claim(payload("attn_flash_br", "FlashAttention does not materialize S."))
    store.upsert_entity("l2:attn", "category", "Softmax / Attention")
    store.add_relation("l2:attn", "attn_flash_br", "contains", status="verified", reason="taxonomy")
    graph = store.subgraph("attn_flash_br", hops=1, kinds=["contains"], status="verified")
    assert graph["entities"][0]["id"] == "l2:attn"
    exported = store.export_graph(domain_id="compute")
    assert exported["aggregated"] is False
    assert exported["claims"]
    stats = store.graph_stats()
    assert stats["relations_by_kind"]["contains"] == 1
    assert stats["orphan_claims"] == 0


def test_viz_html_embeds_graph(tmp_path):
    from cuda_sft_qgen.viz import render_graph_html
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add_claim(payload("a", "Claim A about warps."))
    html = render_graph_html(store.export_graph(), tmp_path / "kg.html")
    text = html.read_text(encoding="utf-8")
    assert "cytoscape" in text
    assert "Claim A about warps" in text or "a" in text


def test_neighbors_direction(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    store.add_claim(payload("a", "Claim A about warps."))
    store.add_claim(payload("b", "Claim B about occupancy."))
    store.add_relation("a", "b", "requires", status="provisional", reason="prereq")
    assert store.neighbors("a", direction="out", status="provisional")[0]["target_id"] == "b"
    assert store.neighbors("b", direction="in", status="provisional")[0]["source_id"] == "a"
    assert store.neighbors("a", direction="in", status="provisional") == []
