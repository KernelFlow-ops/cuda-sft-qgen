"""Tests for KnowledgeStore revisions, audits, jobs, and KnowledgeWorker."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cuda_sft_qgen.knowledge_worker import KnowledgeWorker
from cuda_sft_qgen.store.knowledge import KnowledgeStore
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy
from cuda_sft_qgen.taxonomy_health import TaxonomyHealth

URL = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"
CLAIM = "A warp contains 32 threads."


@pytest.fixture
def store(tmp_path):
    """Empty KnowledgeStore in a temp directory."""
    return KnowledgeStore(tmp_path / "knowledge.sqlite3")


def payload(claim_id="warp_width", claim=CLAIM):
    """Minimal add_claim payload used by revision/audit tests."""
    return {"l3_id": claim_id, "domain_id": "compute", "l2_id": "warp", "claim": claim, "types": ["conceptual"], "difficulty": [2], "sources": [URL]}


def test_provisional_duplicate_and_revision_dependencies(store):
    assert store.add_claim(payload())
    assert not store.add_claim(payload("other", "  A warp contains 32 threads.  "))
    assert store.get_claim("warp_width")["status"] == "provisional"
    store.record_question("q1", [("warp_width", 1)])
    assert store.revise("warp_width", {"claim": "Each CUDA warp has 32 threads."}, reason="clarify", expected_revision=1) == 2
    assert store.impact_events()[0]["question_ids"] == ["q1"]
    with pytest.raises(ValueError, match="stale"):
        store.revise("warp_width", {}, reason="stale", expected_revision=1)
    assert store.context_for_claim("warp_width", revision=1)["claims"][0]["claim"] == CLAIM


def test_audit_requires_retrieved_official_evidence_and_current_revision(store):
    store.add_claim(payload())
    with pytest.raises(ValueError, match="verbatim"):
        store.apply_audit("warp_width", 1, verdict="supported", url=URL, quote=CLAIM, document="unrelated")
    with pytest.raises(ValueError, match="verbatim"):
        store.apply_audit("warp_width", 1, verdict="supported", url="https://example.com", quote=CLAIM, document=CLAIM)
    assert not store.apply_audit("warp_width", 2, verdict="supported", url=URL, quote=CLAIM, document=CLAIM)
    assert store.apply_audit("warp_width", 1, verdict="supported", url=URL, quote=CLAIM, document=CLAIM)
    assert store.get_claim("warp_width")["status"] == "verified"
    graph = store.context_for_claim("warp_width")
    assert graph["evidence"][0]["quote"] == CLAIM
    assert graph["relations"][0]["kind"] == "supported_by"


def test_quarantine_overlay_and_idempotent_import(store):
    leaf = Leaf("compute", "Compute", "warp", "Warp", "warp_width", CLAIM, ("conceptual",), (2,))
    tax = Taxonomy(1, "cuda", [leaf], [{"id": "compute", "children": [{"id": "warp", "leaves": [{"id": "warp_width", "claim": CLAIM}]}]}])
    assert store.import_taxonomy(tax) == 1
    assert store.import_taxonomy(tax) == 0
    assert store.get_claim("warp_width")["status"] == "legacy_unreviewed"
    store.apply_audit("warp_width", 1, verdict="contradicted", url=URL, quote="contradiction", document="contradiction")
    overlay = store.overlay_taxonomy(tax)
    assert overlay.leaves[0].retired
    assert overlay.leaves[0].status == "quarantined"
    assert not tax.leaves[0].retired
    assert store.import_taxonomy(tax) == 0
    assert store.get_claim("warp_width")["status"] == "quarantined"


def test_relations_need_evidence_and_stale_after_revision(store):
    store.add_claim(payload())
    store.add_claim(payload("warp_sync", "Warp synchronization requires participating threads."))
    with pytest.raises(ValueError):
        store.add_relation("warp_width", "warp_sync", "requires", status="verified")
    store.apply_audit("warp_width", 1, verdict="supported", url=URL, quote=CLAIM, document=CLAIM)
    evidence_id = store.subgraph("warp_width")["evidence"][0]["id"]
    store.add_relation("warp_width", "warp_sync", "requires", status="verified", evidence_id=evidence_id, reason="explicitly established by source")
    assert store.related_claim_ids("warp_width") == {"warp_sync"}
    store.revise("warp_sync", {"claim": "A revised sync mechanism claim."}, reason="revision")
    assert store.related_claim_ids("warp_width") == set()


def test_queue_dedup_lease_and_failure_retry(store):
    job_id = store.enqueue("audit", {"claim_id": "x"}, "same")
    assert store.enqueue("audit", {}, "same") is None
    job = store.take_job()
    assert job["id"] == job_id
    assert store.take_job() is None
    store.finish_job(job_id, error="temporary")
    assert store.stats()["jobs"] == {"queued": 1}
    assert store.take_job() is None


def test_worker_hallucinated_quote_stays_provisional(store):
    store.add_claim(payload())
    llm = Mock()
    llm.complete_json.return_value = {"verdict": "supported", "url": URL, "quote": "invented quote"}
    worker = KnowledgeWorker(store, llm, official_fetcher=lambda url: CLAIM)
    store.enqueue_daily_audits()
    assert worker.run_once()
    assert store.get_claim("warp_width")["status"] == "provisional"
    assert store.stats()["evidence"] == 0


def test_worker_quarantine_invalidates_questions(store):
    store.add_claim(payload(claim="A warp contains 99 threads."))
    llm = Mock()
    llm.complete_json.return_value = {"verdict": "contradicted", "url": URL, "quote": CLAIM, "reason": "32, not 99"}
    archive = Mock()
    worker = KnowledgeWorker(store, llm, archive=archive, official_fetcher=lambda url: CLAIM)
    store.enqueue_daily_audits()
    worker.run_once()
    assert store.get_claim("warp_width")["status"] == "quarantined"
    archive.invalidate_claim.assert_called_once()


def test_background_node_enqueues_without_llm(store):
    from cuda_sft_qgen.graph.nodes.architect import taxonomy_health_node
    llm = Mock()
    worker = KnowledgeWorker(store, llm)
    with patch("cuda_sft_qgen.graph.nodes.architect.get_runtime", return_value=SimpleNamespace(knowledge_worker=worker)):
        result = taxonomy_health_node({"review_rounds": 5})
    assert result["status"] == "taxonomy_queued"
    assert not llm.complete_json.called


def test_leaf_coverage_is_not_question_volume():
    leaves = [Leaf("d", "D", "a", "A", f"a_{i}", CLAIM, ("conceptual",), (2,)) for i in range(100)]
    tax = Taxonomy(1, "x", leaves, [{"id": "d", "children": [{"id": "a", "leaves": [{"id": leaf.l3_id} for leaf in leaves]}]}])
    questions = [SimpleNamespace(topic_path=["d", "a", "a_0"], l3_id="a_0") for _ in range(30)]
    metric = TaxonomyHealth(tax).compute_l2_metrics(questions, [])["d.a"]
    assert metric.coverage == 0.01
    assert metric.question_quota_completion == 1.0
    assert metric.avg_diversity == 0.0
    assert not metric.knowledge_gap


def test_relation_evidence_does_not_verify_claim(store):
    store.add_claim(payload())
    store.add_evidence("warp_width", 1, url=URL, quote=CLAIM, document=CLAIM)
    assert store.get_claim("warp_width")["status"] == "provisional"


def test_daily_schedule_prioritizes_new_claims_and_respects_interval(store):
    store.add_claim(payload("old"), status="legacy_unreviewed")
    store.enqueue_daily_audits()
    store.add_claim(payload("new", "A new CUDA shared-memory mechanism claim."))
    store.enqueue_daily_audits()
    job = store.take_job()
    assert job["payload"]["claim_id"] == "new"
    store.finish_job(job["id"])
    store.apply_audit("new", 1, verdict="insufficient", reason="no matching evidence")
    assert store.enqueue_daily_audits(interval_seconds=86400) == 0
    assert store.enqueue_daily_audits(interval_seconds=0) == 1


def test_invalidation_includes_secondary_dependencies(store):
    store.add_claim(payload())
    store.record_question("secondary", [("unrelated_primary", 1), ("warp_width", 1)])
    llm = Mock()
    llm.complete_json.return_value = {"verdict": "contradicted", "url": URL, "quote": CLAIM, "reason": "contradiction"}
    archive = Mock()
    worker = KnowledgeWorker(store, llm, archive=archive, official_fetcher=lambda url: CLAIM)
    store.enqueue_daily_audits()
    worker.run_once()
    assert archive.invalidate_claim.call_args.kwargs["question_ids"] == ["secondary"]


def test_worker_replays_impact_after_audit_process_interruption(store):
    store.add_claim(payload())
    store.record_question("q", [("warp_width", 1)])
    store.apply_audit("warp_width", 1, verdict="contradicted", url=URL, quote=CLAIM, document=CLAIM)
    archive = Mock()
    worker = KnowledgeWorker(store, Mock(), archive=archive)
    assert not worker.run_once()
    archive.invalidate_claim.assert_called_once()
    assert not worker.run_once()
    archive.invalidate_claim.assert_called_once()


def test_expansion_preview_does_not_commit_and_apply_publishes_provisional(store):
    leaf = Leaf("compute", "Compute", "warp", "Warp", "warp_width", CLAIM, ("conceptual",), (2,))
    taxonomy = Taxonomy(1, "cuda", [leaf], [{"id": "compute", "name": "Compute", "children": [{"id": "warp", "name": "Warp", "leaves": [{"id": "warp_width", "claim": CLAIM}]}]}])
    store.import_taxonomy(taxonomy)
    llm = Mock()
    llm.complete_json.return_value = {"new_leaves": [payload("warp_sync", "__syncwarp requires all named participating lanes to execute the matching barrier.")]}
    worker = KnowledgeWorker(store, llm, taxonomy=taxonomy)
    preview = worker.expand_once(mode="l2", l2_key="compute.warp", max_leaves=1, apply=False)
    assert preview["proposed"] == 1
    assert preview["total_added"] == 0
    assert store.get_claim("warp_sync") is None
    result = worker.expand_once(mode="l2", l2_key="compute.warp", max_leaves=1, apply=True)
    assert result["total_added"] == 1
    assert store.get_claim("warp_sync")["status"] == "provisional"
    assert len(taxonomy.leaves) == 1


def test_audit_retry_exhaustion_defers_until_next_daily_cycle(store):
    store.add_claim(payload())
    store.enqueue_daily_audits()
    job = store.take_job()
    with store.connect() as db:
        db.execute("UPDATE jobs SET attempts=3 WHERE id=?", (job["id"],))
    store.finish_job(job["id"], error="source unavailable")
    assert store.enqueue_daily_audits() == 0
    assert store.get_claim("warp_width")["status"] == "provisional"


def test_late_question_registration_reemits_stale_dependency_event(store):
    store.add_claim(payload())
    store.apply_audit("warp_width", 1, verdict="contradicted", url=URL, quote=CLAIM, document=CLAIM)
    original_cursor = store.impact_events()[-1]["id"]
    store.record_question("late", [("warp_width", 1)])
    events = store.impact_events(original_cursor)
    assert len(events) == 1
    assert events[0]["question_ids"] == ["late"]
    store.record_question("late", [("warp_width", 1)])
    assert len(store.impact_events(original_cursor)) == 1


@pytest.mark.parametrize("difficulty", [[999], [True], "hard", [], [1, 5]])
def test_expansion_rejects_invalid_difficulty(difficulty):
    from cuda_sft_qgen.taxonomy_expand import validate_leaf
    item = payload()
    item["difficulty"] = difficulty
    assert validate_leaf(item, l2_id="warp", existing_ids=set(), existing_claims=[]) == "bad_difficulty"
