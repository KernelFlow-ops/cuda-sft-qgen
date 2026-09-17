"""Tests for atomic review commit, claim invalidation, and training export."""

import json

import pytest

from cuda_sft_qgen.models import SteeringBrief
from cuda_sft_qgen.models import AssessedMetadata, QualityChecks
from cuda_sft_qgen.store.archive import Archive


def decision(q, action="accept"):
    """Chair decision dict for ``commit_review`` tests."""
    return {"question": q.model_dump(), "action": action, "reason": "test"}


def test_commit_survives_projection_failure_without_conflicting_status(tmp_path, sample_question, monkeypatch):
    archive = Archive(tmp_path)
    q = sample_question()
    archive.save_pending([q])
    monkeypatch.setattr(archive, "export_compatibility", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        archive.commit_review([decision(q)], SteeringBrief(), {"iteration": 1})
    resumed = Archive(tmp_path)
    assert [x.id for x in resumed.load_accepted()] == [q.id]
    assert not resumed.load_pending()
    result = resumed.commit_review([decision(q, "reject")], SteeringBrief(), {"iteration": 1})
    assert result["accepted_count"] == 1
    assert result["rejected_count"] == 0
    assert resumed.load_run_state()["accepted_count"] == 1


def test_generation_snapshot_does_not_erase_review_or_other_pending(tmp_path, sample_question):
    archive = Archive(tmp_path)
    first, second = sample_question(id="first"), sample_question(id="second")
    archive.save_pending([first])
    archive.commit_review([decision(first)], {}, {})
    archive.save_pending([first, second])
    archive.save_pending([])
    assert [q.id for q in archive.load_pending()] == ["second"]
    assert [q.id for q in archive.load_accepted()] == ["first"]


def test_migration_preserves_original_and_terminal_wins(tmp_path, sample_question):
    q = sample_question()
    original = q.model_dump_json() + "\n"
    (tmp_path / "accepted.jsonl").write_text(original)
    (tmp_path / "pending.jsonl").write_text(original)
    archive = Archive(tmp_path)
    assert len(archive.load_accepted()) == 1
    assert not archive.load_pending()
    assert (tmp_path / "accepted.jsonl.legacy").read_text() == original
    assert (tmp_path / "accepted.jsonl").read_text() == original


def test_retry_lease_can_be_recovered_after_worker_crash(tmp_path):
    archive = Archive(tmp_path)
    archive.enqueue_retry({"brief": {"l3_id": "x"}, "next_ok_at": 0})
    taken = archive.due_retry_items(limit=1, now=10)
    assert len(taken) == 1
    assert not archive.due_retry_items(limit=1, now=11)
    recovered = Archive(tmp_path).due_retry_items(limit=1, now=1000)
    assert recovered[0]["task_id"] == taken[0]["task_id"]
    archive.complete_retry(taken[0]["task_id"])
    assert archive.retry_queue_size() == 0


def test_knowledge_correction_withdraws_export_and_preserves_audit(tmp_path, sample_question):
    archive = Archive(tmp_path)
    q = sample_question(claim_revision=1, review_details={
                        "id": "q_test_001", "action": "accept", "reason": "verified",
                        "checks": QualityChecks(**{k: "pass" for k in QualityChecks.model_fields}).model_dump(),
                        "evidence": ["private"],
                        "assessed_metadata": AssessedMetadata(question_type="conceptual", difficulty=2,
                            bloom="understand", hardware="hopper", granularity="kernel", language="zh").model_dump(),
                        "internal_reference_answer": "private answer",
                        "reviewer_model": "real", "reviewer_independent": True},
                        internal_reference_answer="private answer")
    archive.commit_review([decision(q)], {}, {})
    before = tmp_path / "before.jsonl"
    assert archive.export_training(before)["questions"] == 1
    assert "private" not in before.read_text()
    assert archive.load_accepted()[0].internal_reference_answer == "private answer"
    assert archive.invalidate_claim(q.l3_id, 1, "incorrect fact", 1) == 1
    assert archive.export_training(tmp_path / "after.jsonl")["questions"] == 0
    assert archive.corrections()[0]["question_id"] == q.id
    assert len(archive.load_pending()) == 1


def test_unfinished_generation_tasks_resume_idempotently(tmp_path):
    archive = Archive(tmp_path)
    brief = {"l3_id": "a"}
    task_id = archive.save_generation_task(brief)
    assert archive.save_generation_task(brief) == task_id
    resumed = Archive(tmp_path)
    assert len(resumed.unfinished_generation_tasks()) == 1
    resumed.complete_generation_task(task_id)
    assert not archive.unfinished_generation_tasks()


def test_training_export_revalidates_every_quality_check(tmp_path, sample_question):
    archive = Archive(tmp_path)
    q = sample_question(
        review_details={"id": "wrong", "action": "accept", "reviewer_model": "real"},
        internal_reference_answer=None,
    )
    archive.append_accepted(q)
    manifest = archive.export_training(tmp_path / "strict.jsonl")
    assert manifest["questions"] == 0
    assert manifest["skipped_unverified"] == 1
