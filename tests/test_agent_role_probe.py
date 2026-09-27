"""Offline safety tests for the bounded real-Agent role probe."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from cuda_sft_qgen.graph.nodes import generate_core
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.store.archive import Archive

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_agent_roles


def test_probe_result_requires_every_requested_role_to_pass():
    roles = {
        "evolver": {"status": "draft_created", "agent_called": True},
        "salvage_llm": {"agent_called": True, "parser_miss_injected": True,
                        "fidelity_guard_safe": True},
        "knowledge_worker": {"audit_succeeded": True, "agent_called": True},
        "review_repair": {"handoff_created": True, "agent_called": True,
                          "candidate_status": "pending_review"},
    }
    assert probe_agent_roles.probe_passed(roles, salvage_requested=True, blocked_attempts=0)
    roles["evolver"]["status"] = "error"
    assert not probe_agent_roles.probe_passed(roles, salvage_requested=True, blocked_attempts=0)
    roles["evolver"]["status"] = "draft_created"
    assert not probe_agent_roles.probe_passed(roles, salvage_requested=True, blocked_attempts=1)


def _gate_fixture(gate_dir: Path, sample_question, sample_brief) -> Question:
    gate_dir.mkdir()
    (gate_dir / "gate_report.json").write_text(
        json.dumps({"fake": False, "done": True, "accepted": 1}), encoding="utf-8"
    )
    brief = sample_brief(task_id="source-task", role="writer")
    archive = Archive(gate_dir)
    task_id = archive.save_generation_task(brief.model_dump())
    question = sample_question(
        id="source-question",
        internal_reference_answer="private source answer",
        generation_provenance={
            "task_id": task_id,
            "writer_real_api": True,
            "writer_provider": "openai",
            "writer_model": "test-model",
        },
        brief_snapshot=brief.model_dump(),
    )
    archive.append_accepted(question)
    archive.close()
    return question


def test_load_gate_seed_reads_original_brief_without_mutating_archive(tmp_path, sample_question, sample_brief):
    gate = tmp_path / "gate"
    expected = _gate_fixture(gate, sample_question, sample_brief)
    public_record = json.loads((gate / "accepted.jsonl").read_text(encoding="utf-8"))
    assert "generation_provenance" not in public_record
    assert "internal_reference_answer" not in public_record
    before = {path.name: path.read_bytes() for path in gate.iterdir()}

    question, brief = probe_agent_roles.load_gate_seed(gate)

    assert question.id == expected.id
    assert question.generation_provenance["writer_real_api"] is True
    assert question.internal_reference_answer == "private source answer"
    assert brief.task_id == "source-task"
    assert brief.role == "writer"
    assert {path.name: path.read_bytes() for path in gate.iterdir()} == before
    assert not (gate / "accepted.jsonl.legacy").exists()


@pytest.mark.parametrize(
    "report",
    [
        {"fake": False, "done": False, "accepted": 1},
        {"fake": False, "done": True, "accepted": 0},
    ],
)
def test_load_gate_seed_requires_completed_gate_with_accepts(
    tmp_path, sample_question, sample_brief, report
):
    gate = tmp_path / "gate"
    _gate_fixture(gate, sample_question, sample_brief)
    (gate / "gate_report.json").write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="must be done"):
        probe_agent_roles.load_gate_seed(gate)


def test_probe_output_must_be_separate_and_new(tmp_path):
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "gate_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="separate"):
        probe_agent_roles.validate_paths(gate, gate / "probe")
    with pytest.raises(ValueError, match="separate"):
        probe_agent_roles.validate_paths(gate, tmp_path)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "artifact.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        probe_agent_roles.validate_paths(gate, occupied)
    assert (occupied / "artifact.txt").read_text(encoding="utf-8") == "keep"


def test_outcome_summary_never_serializes_question_or_private_answer(sample_question):
    private = "private-reference-answer-must-not-appear"
    question = sample_question(internal_reference_answer=private)
    summary = probe_agent_roles.summarize_outcome(
        generate_core.GenerateOutcome(status="ok", question=question)
    )

    encoded = json.dumps(summary)
    assert "private-reference-answer" not in encoded
    assert question.question not in encoded
    assert summary["candidate_created"] is True


def test_salvage_role_rejects_changed_question_without_api(fake_runtime, sample_brief, monkeypatch):
    brief = sample_brief(kind="mechanism")
    original = "How does CUDA occupancy affect the number of active warps on an SM?"
    raw = json.dumps({
        "question": original,
        "tests_claim": brief.l3_claim,
        "language": brief.language,
    }, ensure_ascii=False)
    raw = raw[:-1] + ',"probe_padding":"' + ("x" * 1300) + '"'
    response = json.dumps({
        "question": "How does CUDA warp occupancy affect an SM?",
        "tests_claim": brief.l3_claim,
        "language": brief.language,
    }, ensure_ascii=False)
    fake_runtime.settings.salvage_use_llm = True
    monkeypatch.setattr(fake_runtime.llm, "complete", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(generate_core, "get_runtime", lambda: fake_runtime)
    parser = generate_core.parse_question_payload
    missed = {"done": False}

    def one_parser_miss(text, **kwargs):
        if text == raw and not missed["done"]:
            missed["done"] = True
            return None
        return parser(text, **kwargs)

    monkeypatch.setattr(generate_core, "parse_question_payload", one_parser_miss)
    outcome = generate_core.salvage_item({
        "brief": brief.model_dump(),
        "raw": raw,
        "writer_real_api": False,
    })

    assert missed["done"]
    assert outcome.status == "retry"
    assert outcome.question is None
    assert outcome.reason == "json_parse"


def test_salvage_safety_summary_marks_changed_output_as_blocked(sample_question):
    summary = probe_agent_roles.summarize_salvage_safety(
        generate_core.GenerateOutcome(status="retry", reason="json_parse"),
        model_output_stem_matches=False,
        expected_stem=sample_question().question,
    )

    assert summary["changed_stem_rejected"] is True
    assert summary["fidelity_guard_safe"] is True


@pytest.mark.parametrize(
    ("state", "error", "after", "expected_status", "expected_success"),
    [
        ("queued", "temporary failure", {"queued": 1}, "audit_job_requeued_after_error", False),
        ("failed", "permanent failure", {"failed": 1}, "audit_job_failed", False),
        ("done", None, {"done": 1}, "audit_job_completed", True),
    ],
)
def test_knowledge_probe_distinguishes_job_run_from_audit_success(
    state, error, after, expected_status, expected_success
):
    summary = probe_agent_roles.summarize_knowledge_job(
        worker_ran=True,
        job_state=state,
        job_error=error,
        before_jobs={},
        after_jobs=after,
    )

    assert summary["worker_job_ran"] is True
    assert summary["status"] == expected_status
    assert summary["audit_succeeded"] is expected_success
