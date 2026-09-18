"""Tests for throughput-gate comparison, median baseline, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuda_sft_qgen.cli import build_parser
from cuda_sft_qgen.gate import (
    build_quality_report,
    classify_reject_reason,
    compare_gate_reports,
    median_baseline,
    percentile,
    write_quality_sample,
    write_verdict_md,
)
from cuda_sft_qgen.models import Question


def _report(**overrides):
    base = {
        "phase": "0",
        "error": None,
        "fake": False,
        "target": 10,
        "accepted": 10,
        "rejected": 0,
        "pending": 0,
        "exhausted": 0,
        "duration_seconds": 100.0,
        "accepted_per_hour": 360.0,
        "log_errors": 0,
        "knowledge_job_errors": 0,
        "node_seconds": {
            "generate": {"p95": 8.0, "count": 8},
            "generate_batch": {"p95": 8.0, "count": 8},
            "review_chair": {"p95": 12.0, "count": 8},
        },
    }
    base.update(overrides)
    return base


def test_percentile_empty_and_linear():
    assert percentile([], 50) is None
    assert percentile([4], 95) == 4
    assert percentile([0, 10], 50) == 5


def test_baseline_without_compare_is_origin():
    result = compare_gate_reports(_report(), None, target=10)
    assert result["verdict"] == "baseline"
    assert result["failures"] == []


def test_baseline_missed_target_is_red():
    result = compare_gate_reports(_report(accepted=7), None, target=10)
    assert result["verdict"] == "red"
    assert any("accepted" in item for item in result["failures"])


def test_wall_and_throughput_thresholds():
    baseline = _report()
    green = compare_gate_reports(_report(duration_seconds=105, accepted_per_hour=350), baseline)
    assert green["verdict"] == "green"
    yellow = compare_gate_reports(_report(duration_seconds=115, accepted_per_hour=350), baseline)
    assert yellow["verdict"] == "yellow"
    red_wall = compare_gate_reports(_report(duration_seconds=130, accepted_per_hour=350), baseline)
    assert red_wall["verdict"] == "red"
    red_rate = compare_gate_reports(_report(duration_seconds=100, accepted_per_hour=200), baseline)
    assert red_rate["verdict"] == "red"


def test_node_p95_and_exhausted_and_log_errors():
    baseline = _report()
    slow_writer = compare_gate_reports(
        _report(node_seconds={"generate_batch": {"p95": 12.0, "count": 8}, "review_chair": {"p95": 12.0, "count": 8}}),
        baseline,
    )
    assert slow_writer["verdict"] == "red"
    assert any("generate_batch p95" in item for item in slow_writer["failures"])
    assert compare_gate_reports(_report(exhausted=1), baseline)["verdict"] == "red"
    assert compare_gate_reports(_report(log_errors=2), baseline)["verdict"] == "red"
    assert compare_gate_reports(_report(error="RuntimeError: boom"), baseline)["verdict"] == "red"
    assert compare_gate_reports(_report(knowledge_job_errors=3), baseline)["verdict"] == "red"


def test_verdict_markdown(tmp_path: Path):
    compared = compare_gate_reports(_report(duration_seconds=130), _report())
    path = tmp_path / "verdict.md"
    write_verdict_md({"report": _report(phase="1", duration_seconds=130), "compared": compared}, path)
    text = path.read_text(encoding="utf-8")
    assert "verdict: **red**" in text
    assert "wall" in text


def test_median_baseline_picks_middle_wall():
    slow = _report(duration_seconds=600, accepted_per_hour=60, node_seconds={"generate_batch": {"p95": 160}})
    mid = _report(duration_seconds=500, accepted_per_hour=72, node_seconds={"generate_batch": {"p95": 90}})
    fast = _report(duration_seconds=400, accepted_per_hour=90, node_seconds={"generate_batch": {"p95": 70}})
    merged = median_baseline([fast, slow, mid])
    assert merged["duration_seconds"] == 500
    assert merged["node_seconds"]["generate_batch"]["p95"] == 90
    even = median_baseline([fast, slow])
    assert even["duration_seconds"] == 500
    # 556s vs median 500 is yellow, not red
    verdict = compare_gate_reports(_report(duration_seconds=556, accepted_per_hour=71,
                                           node_seconds={"generate_batch": {"p95": 90}, "review_chair": {"p95": 12}}), merged)
    assert verdict["verdict"] in {"yellow", "green"}


def _question(**overrides) -> Question:
    data = {
        "id": "q_test",
        "question": "What is the TMA 128B alignment requirement on Hopper?",
        "tests_claim": "TMA box stride must be 128B-aligned.",
        "topic_path": ["T08", "tma"],
        "l3_id": "tma_align",
        "question_type": "conceptual",
        "difficulty": 2,
        "bloom": "understand",
        "persona": "exam_writer",
        "hardware": "hopper",
        "granularity": "kernel",
        "language": "en",
    }
    data.update(overrides)
    return Question.model_validate(data)


def test_classify_reject_reason_prefixes():
    assert classify_reject_reason("sieve_too_short") == "sieve"
    assert classify_reject_reason("retry_exhausted:429") == "retry_exhausted"
    assert classify_reject_reason("review_error:TimeoutError:x") == "review_error"
    assert classify_reject_reason("json_parse") == "json_parse"
    assert classify_reject_reason("HTTP 429 too many requests") == "rate_limit"
    assert classify_reject_reason("corpus_duplicate") == "duplicate"
    assert classify_reject_reason("") == "unknown"
    assert classify_reject_reason("weird_new_reason") == "other"


def test_quality_report_rates_and_chair_histogram():
    accepted = [
        _question(
            id="q_ok",
            internal_reference_answer="128B alignment",
            review_details={
                "action": "accept",
                "checks": {name: "pass" for name in (
                    "correctness", "answerability", "claim_alignment",
                    "hardware_compatibility", "answer_leakage", "metadata", "duplication",
                )},
            },
        )
    ]
    rejected = [_question(id="q_sieve", reject_reason="sieve_too_vague", review_status="rejected")]
    pending = [_question(
        id="q_pend",
        reject_reason="review_error:ValueError:bad json",
        review_status="pending",
        review_details={"action": "pending", "checks": {"correctness": "uncertain", "answerability": "pass"}},
    )]
    report = build_quality_report(accepted, rejected, pending)
    assert report["total"] == 3
    assert report["accept_rate"] == pytest.approx(1 / 3)
    assert report["reject_classes"] == {"sieve": 1}
    assert report["pending_classes"] == {"review_error": 1}
    assert report["reference_answer_rate"] == 1.0
    assert report["chair_checks"]["correctness"]["pass"] == 1
    assert report["chair_checks"]["correctness"]["uncertain"] == 1


def test_quality_report_classifies_chair_fail_checks():
    rejected = [_question(
        id="q_wrong",
        reject_reason="arithmetic disagrees with tests_claim",
        review_status="rejected",
        review_details={
            "action": "reject",
            "checks": {"correctness": "fail", "answerability": "pass"},
        },
    )]
    report = build_quality_report([], rejected, [])
    assert report["reject_classes"] == {"chair_correctness": 1}


def test_quality_sample_writes_jsonl(tmp_path: Path):
    accepted = [_question(id="q_ok", internal_reference_answer="42")]
    rejected = [_question(id="q_bad", question="", reject_reason="sieve_too_short", review_status="rejected")]
    path = tmp_path / "quality_sample.jsonl"
    records = write_quality_sample(accepted, rejected, [], path, limit=10)
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["id"] == "q_ok"
    assert lines[0]["internal_reference_answer"] == "42"
    assert lines[1]["id"] == "q_bad"
    assert len(records) == 2


def test_verdict_markdown_includes_quality(tmp_path: Path):
    report = _report()
    report["quality"] = {
        "accept_rate": 0.8,
        "reject_classes": {"sieve": 1},
        "reference_answer_rate": 1.0,
        "salvage_or_json_rejects": 0,
        "sieve_rejects": 1,
    }
    compared = compare_gate_reports(report, _report())
    path = tmp_path / "verdict.md"
    write_verdict_md({"report": report, "compared": compared}, path)
    text = path.read_text(encoding="utf-8")
    assert "## Quality" in text
    assert "accept_rate: 0.8" in text


def test_cli_rejects_fake(capsys):
    args = build_parser().parse_args(["gate", "--data-dir", "/tmp/gate-x", "--phase", "baseline", "--fake"])
    assert args.func(args) == 2
    assert "refuses --fake" in capsys.readouterr().err
