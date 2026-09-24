"""Legacy audit backlog and planner downgrade (task T36)."""
import random
import sqlite3

import yaml

from fixv2_helpers import tiny_taxonomy


def test_schedule_backlog(tmp_path):
    from cuda_sft_qgen.store.knowledge import KnowledgeStore
    ks = KnowledgeStore(tmp_path / "k.sqlite3")
    for i in range(25):
        ks.add_claim({"l3_id": f"a_{i:02d}", "claim": f"api claim number {i} about launch limits", "domain_id": "T01",
                      "l2_id": "a", "kind": "api"}, status="legacy_unreviewed")
    for i in range(3):
        ks.add_claim({"l3_id": f"b_{i}", "claim": f"formula claim {i}: x = {i} + 1", "domain_id": "T01",
                      "l2_id": "b", "kind": "formula"}, status="legacy_unreviewed")
    n = ks.schedule_legacy_backlog(batch_per_l2=20, due_days=14, now=1000.0)
    assert n == 23
    db = sqlite3.connect(tmp_path / "k.sqlite3")
    kinds = dict(db.execute("SELECT kind, COUNT(*) FROM jobs GROUP BY kind").fetchall())
    assert kinds == {"audit": 20, "compute_verify": 3}
    assert db.execute("SELECT COUNT(*) FROM claims WHERE audit_due_at=?", (1000.0 + 14 * 86400,)).fetchone()[0] == 23
    assert ks.schedule_legacy_backlog(batch_per_l2=20, due_days=14, now=2000.0) == 5     # remaining a_20..a_24
    assert ks.schedule_legacy_backlog(batch_per_l2=20, due_days=14, now=3000.0) == 0


def test_overlay_carries_due_date(tmp_path):
    from cuda_sft_qgen.store.knowledge import KnowledgeStore
    ks = KnowledgeStore(tmp_path / "k.sqlite3")
    tax = tiny_taxonomy()
    ks.add_claim({"l3_id": "smem_old", "claim": "legacy smem claim ways = gcd", "domain_id": "T02", "l2_id": "smem",
                  "kind": "formula"}, status="legacy_unreviewed")
    ks.schedule_legacy_backlog(batch_per_l2=5, due_days=1, now=10.0)
    leaf = next(lf for lf in ks.overlay_taxonomy(tax).leaves if lf.l3_id == "smem_old")
    assert leaf.audit_due_at == 10.0 + 86400


def test_expired_legacy_only_explored(repo_root):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.planner.sampling import load_angles, plan_briefs
    extra = [{"id": "smem_expired", "claim": "legacy claim ways = gcd(S,32)", "kind": "formula", "types": ["conceptual"],
              "difficulty": [2], "status": "legacy_unreviewed", "audit_due_at": 1.0}]
    load = lambda n: yaml.safe_load((repo_root / "configs" / n).read_text(encoding="utf-8"))
    briefs = plan_briefs(n=1500, taxonomy=tiny_taxonomy(extra), quotas=load("quotas.yaml"),
                         personas=load("personas.yaml")["personas"], accepted=[], pending=[], steering=SteeringBrief(),
                         evolver_fraction=0.0, seed_pool=[], rng=random.Random(2),
                         angles=load_angles(repo_root / "configs" / "angles.yaml"), compat=load("compat_matrix.yaml"),
                         scope=load("scope.yaml"), joint=True)
    expired = [b for b in briefs if b.l3_id == "smem_expired"]
    assert expired and all(b.exploration for b in expired)
