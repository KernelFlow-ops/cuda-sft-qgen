from __future__ import annotations

from unittest.mock import Mock

from cuda_sft_qgen.embeddings import FakeEmbedder
from cuda_sft_qgen.knowledge_worker import KnowledgeWorker, _DocumentText
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy
from cuda_sft_qgen.taxonomy_manager import TaxonomyManager
from cuda_sft_qgen.store.knowledge import KnowledgeStore


def _taxonomy() -> Taxonomy:
    leaf = Leaf(
        "T01", "Programming", "api", "API", "api_known",
        "cudaLaunchKernel accepts a valid launch configuration",
        ("conceptual",), (2,), kind="api",
    )
    domains = [{"id": "T01", "name": "Programming", "children": [{
        "id": "api", "name": "API", "leaves": [{"id": "api_known", "claim": leaf.claim, "kind": "api"}]
    }]}]
    return Taxonomy(1, "tiny", [leaf], domains)


def test_inventory_preserves_headings_and_cuda_symbols():
    rows = KnowledgeWorker._extract_inventory(
        "# Tensor Memory Accelerator\nUse mbarrier and TMA on sm_90.",
        "https://docs.nvidia.com/x",
        source_kind="guide",
    )
    assert any(row["section"] == "# Tensor Memory Accelerator" for row in rows)
    assert any("mbarrier" in row["tokens"] and "TMA" in row["tokens"] for row in rows)
    parser = _DocumentText()
    parser.feed("<h2>Tensor Memory Accelerator</h2>")
    parser.close()
    assert parser.parts == ["# Tensor Memory Accelerator", "Tensor Memory Accelerator"]
    parser = _DocumentText()
    parser.feed("<p>The 16-byte alignment <em>requirement was dropped</em> for TMA.</p>")
    parser.close()
    assert parser.parts == ["The 16-byte alignment requirement was dropped for TMA."]


def test_external_coverage_uses_candidates_and_hard_trigger():
    manager = TaxonomyManager(
        _taxonomy(), FakeLLM(), embedder=FakeEmbedder(),
        external_inventory=[{"id": "x", "concept": "mbarrier expect_tx TMA"}],
    )
    report = manager.external_coverage_report()
    assert report["hard_trigger"]
    assert report["uncovered"][0]["top_candidates"]
    assert report["uncovered"][0]["top_candidates"][0]["l3_id"] == "api_known"


def test_critic_budget_zero_does_not_call_llm():
    llm = Mock()
    manager = TaxonomyManager(
        _taxonomy(), llm,
        external_inventory=[{"id": "x", "concept": "brand new TMA mechanism"}],
    )
    manager._in_tick = True
    manager.max_llm_calls_per_tick = 0
    result = manager.external_critic(max_new=2)
    assert result["proposals"] == []
    llm.complete_json.assert_not_called()


def test_invalid_external_proposal_is_not_routed():
    manager = TaxonomyManager(
        _taxonomy(), FakeLLM(),
        external_inventory=[{"id": "x", "concept": "API", "url": "https://docs.nvidia.com/x"}],
    )
    result = manager._validate_external_proposals([
        {"l2_key": "T01.missing", "l3_id": "missing_bad", "claim": "x"},
        {"l2_key": "T01.api", "l3_id": "api_bad", "claim": "x", "source_url": "https://evil.example/x"},
    ], max_new=5)
    assert result == []


def test_external_inventory_job_is_durable_and_ttl_dedup(temp_data_dir):
    store = KnowledgeStore(temp_data_dir / "knowledge.sqlite3")
    worker = KnowledgeWorker(
        store, FakeLLM(),
        official_fetcher=lambda url: "# CUDA API\nmbarrier on sm_90",
    )
    job_id = worker.enqueue_external_inventory(force=True)
    assert job_id is not None
    assert worker.enqueue_external_inventory(force=False) is None
    assert worker.run_once()
    inventory = store.get_metadata("external_concept_inventory")
    assert inventory and store.stats()["jobs"].get("done", 0) >= 1
    assert worker.enqueue_external_inventory(force=False) is None


def test_external_inventory_job_recovers_expired_lease(temp_data_dir):
    import time
    store = KnowledgeStore(temp_data_dir / "knowledge.sqlite3")
    worker = KnowledgeWorker(store, FakeLLM())
    job_id = worker.enqueue_external_inventory(force=True)
    leased = store.take_job()
    assert leased and leased["id"] == job_id
    with store.connect() as db:
        db.execute("UPDATE jobs SET leased_at=? WHERE id=?", (time.time() - 1000, job_id))
    resumed = store.take_job(lease_seconds=1)
    assert resumed and resumed["id"] == job_id


def test_external_proposals_are_provisional_and_queued_for_audit(temp_data_dir):
    store = KnowledgeStore(temp_data_dir / "knowledge.sqlite3")
    taxonomy = _taxonomy()
    llm = Mock()
    llm.complete_json.return_value = {
        "proposals": [{
            "inventory_id": "x",
            "l2_key": "T01.api",
            "l3_id": "api_launch_constraint",
            "claim": "cudaLaunchKernel returns an error when block dimensions exceed maxThreadsPerBlock = 1024",
            "kind": "formula",
            "types": ["conceptual"],
            "difficulty": [2, 3],
            "hints": ["maxThreadsPerBlock"],
            "source_url": "https://docs.nvidia.com/x",
        }],
        "new_axes": [{"concept": "new axis", "reason": "does not fit"}],
    }
    worker = KnowledgeWorker(store, llm, taxonomy=taxonomy)
    result = worker.process_external_inventory(
        [{"id": "x", "concept": "cudaLaunchKernel checks maxThreadsPerBlock = 1024", "url": "https://docs.nvidia.com/x"}], apply=True, max_leaves=2,
    )
    assert result["total_added"] == 1
    assert store.get_claim("api_launch_constraint")["status"] == "provisional"
    assert store.get_metadata("taxonomy_structure_suggestions")["new_axes"]
    assert store.stats()["jobs"].get("queued", 0) >= 1
    assert len(taxonomy.leaves) == 1


def test_critic_error_is_not_reported_as_success():
    llm = Mock()
    llm.complete_json.side_effect = RuntimeError("gateway error")
    manager = TaxonomyManager(
        _taxonomy(), llm,
        external_inventory=[{"id": "x", "concept": "new CUDA mechanism"}],
    )
    manager._in_tick = True
    try:
        import pytest
        with pytest.raises(RuntimeError, match="no valid result"):
            manager.external_critic(max_new=1)
    finally:
        manager._in_tick = False


def test_apply_false_does_not_persist_structure_suggestions(temp_data_dir):
    store = KnowledgeStore(temp_data_dir / "knowledge.sqlite3")
    llm = Mock()
    llm.complete_json.return_value = {"new_axes": [{"concept": "new axis"}], "proposals": []}
    worker = KnowledgeWorker(store, llm, taxonomy=_taxonomy())
    worker.process_external_inventory(
        [{"id": "x", "concept": "mbarrier TMA", "url": "https://docs.nvidia.com/x"}],
        apply=False,
        max_leaves=1,
    )
    assert store.get_metadata("taxonomy_structure_suggestions") is None


def test_provisional_leaf_does_not_count_as_active_coverage():
    taxonomy = _taxonomy()
    provisional = Leaf(
        "T01", "Programming", "api", "API", "api_tma",
        "mbarrier expect_tx tracks TMA completion bytes on sm_90",
        ("conceptual",), (3,), kind="mechanism", status="provisional",
    )
    taxonomy.leaves.append(provisional)
    manager = TaxonomyManager(
        taxonomy, FakeLLM(),
        external_inventory=[{"id": "x", "concept": "mbarrier expect_tx TMA sm_90"}],
    )
    assert manager.external_coverage_report()["hard_trigger"]


def test_external_proposal_cannot_add_source_identifiers():
    manager = TaxonomyManager(
        _taxonomy(), FakeLLM(),
        external_inventory=[{
            "id": "x",
            "concept": "Added matrix pointer alignment checks in cublasLtMatmul. [CUB-9577]",
            "url": "https://docs.nvidia.com/x",
        }],
    )
    result = manager._validate_external_proposals([{
        "inventory_id": "x",
        "l2_key": "T01.api",
        "l3_id": "api_alignment_status",
        "claim": "cublasLtMatmul returns CUBLAS_STATUS_INVALID_VALUE for misalignment [CUB-9577]",
        "kind": "api",
        "types": ["conceptual"],
        "difficulty": [2],
        "source_url": "https://docs.nvidia.com/x",
    }], max_new=1)
    assert result == []


def test_external_gap_bypasses_high_health_early_exit():
    llm = Mock()
    llm.complete_json.return_value = {
        "proposals": [], "new_axes": [], "splits": [], "merges": [],
    }
    manager = TaxonomyManager(
        _taxonomy(), llm,
        external_inventory=[{
            "id": "x", "concept": "mbarrier expect_tx TMA sm_90",
            "url": "https://docs.nvidia.com/x",
        }],
    )
    health = manager.analyze_health()
    assert health.health_score >= manager.health_expand_threshold
    result = manager.run_tick(apply=False, max_leaves=1)
    assert result["mission"] == "external_coverage"
    assert result["llm_calls"] == 1
