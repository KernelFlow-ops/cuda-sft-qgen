"""Tests that planner briefs attach knowledge context from the store."""

from __future__ import annotations

import random

from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.planner import plan_briefs
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy


class Knowledge:
    """Minimal KnowledgeStore stand-in for planner context tests."""

    def __init__(self, related=()):
        """Store related claim ids returned by :meth:`related_claim_ids`."""
        self.related = related

    def related_claim_ids(self, claim_id):
        """Return the canned related-id set (``claim_id`` unused)."""
        return set(self.related)

    def context_for_claim(self, claim_id, revision=None):
        """Return a one-claim context payload for the requested id."""
        return {"claims": [{"id": claim_id, "revision": revision}], "relations": [], "evidence": []}


def briefs(settings, seeds, knowledge=None):
    """Plan one brief against a single-leaf test taxonomy."""
    leaf = Leaf("T01", "Domain", "sync", "Sync", "warp_stall", "Warp stalls delay issue",
                ("conceptual",), (2,), revision=3, sources=("official-source",))
    taxonomy = Taxonomy(1, "test", [leaf], [{"id": "T01", "name": "Domain", "children": []}])
    return plan_briefs(n=1, taxonomy=taxonomy, quotas=settings.quotas, personas=settings.personas,
                       accepted=seeds, pending=[], steering=SteeringBrief(), evolver_fraction=1,
                       seed_pool=seeds, knowledge=knowledge, rng=random.Random(7))


def test_same_claim_seed_precedes_related_claim(fake_settings, sample_question):
    seeds = [sample_question(id="related", l3_id="register_pressure"),
             sample_question(id="same", l3_id="warp_stall")]
    brief = briefs(fake_settings, seeds, Knowledge(["register_pressure"]))[0]
    assert brief.role == "evolver"
    assert brief.seed_id == "same"
    assert brief.claim_revision == 3
    assert brief.source_ids == ["official-source"]
    assert brief.knowledge_context["claims"][0]["revision"] == 3


def test_unrelated_claim_does_not_become_seed(fake_settings, sample_question):
    seeds = [sample_question(id="unrelated", l3_id="texture_fetch")]
    brief = briefs(fake_settings, seeds, Knowledge())[0]
    assert brief.role == "writer"
    assert brief.seed_id is None
    assert briefs(fake_settings, seeds)[0].seed_id is None


def test_generation_context_caps_requires_chain(fake_settings, sample_question, tmp_path):
    from cuda_sft_qgen.store.knowledge import KnowledgeStore
    store = KnowledgeStore(tmp_path / "k.sqlite3")
    for i in range(8):
        store.add_claim({
            "l3_id": f"c{i}", "domain_id": "T01", "l2_id": "sync",
            "claim": f"Claim number {i} about warp stalls.",
            "types": ["conceptual"], "difficulty": [2],
        })
    for i in range(7):
        store.add_relation(f"c{i}", f"c{i+1}", "requires", status="provisional", reason="chain")
    ctx = store.context_for_generation("c0", max_claims=6)
    assert len(ctx["claims"]) <= 6
    assert all(len(c["claim"]) <= 220 for c in ctx["claims"])


def test_explicit_relation_allows_cross_claim_seed(fake_settings, sample_question):
    seeds = [sample_question(id="related", l3_id="register_pressure")]
    brief = briefs(fake_settings, seeds, Knowledge(["register_pressure"]))[0]
    assert brief.role == "evolver"
    assert brief.seed_id == "related"
