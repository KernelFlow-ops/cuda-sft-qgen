"""Tests for throughput-gate comparison, median baseline, and CLI wiring."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuda_sft_qgen.cli import build_parser
from cuda_sft_qgen import gate as gate_module
from cuda_sft_qgen.gate import (
    build_quality_report,
    classify_reject_reason,
    compare_gate_reports,
    llm_activity,
    median_baseline,
    percentile,
    stage_activity,
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


def test_exportable_count_and_deadline_are_hard_gates():
    unexportable = compare_gate_reports(_report(accepted=50, exportable=49,
                                                target=50, time_limit_seconds=1500), None)
    assert unexportable["verdict"] == "red"
    assert any("exportable 49 < target 50" in item for item in unexportable["failures"])
    slow = compare_gate_reports(_report(accepted=50, exportable=50, target=50,
                                        duration_seconds=1500.1, time_limit_seconds=1500), None)
    assert slow["verdict"] == "red"
    assert any("duration" in item for item in slow["failures"])


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
    pending = [_question(id="q_pending", internal_reference_answer="secret", review_status="pending")]
    path = tmp_path / "quality_sample.jsonl"
    records = write_quality_sample(accepted, rejected, pending, path, limit=10)
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["id"] == "q_ok"
    assert lines[1]["id"] == "q_bad"
    assert lines[2]["id"] == "q_pending"
    assert all("internal_reference_answer" not in row for row in lines)
    assert all("reason" not in row and "warnings" not in row for row in lines)
    assert lines[1]["reason_code"] == "sieve"
    assert len(records) == 3


def test_llm_activity_reports_observed_model_priority_groups():
    metrics = {
        "counters": {
            "llm_requests{model=writer-v1,priority=0}": 3,
            "llm_failures{model=writer-v1,priority=0}": 1,
            "llm_requests{model=chair-v2,priority=2}": 2,
        },
        "histograms": {
            "llm_seconds{model=writer-v1,priority=0}": {
                "count": 3, "avg": 4.0, "max": 8.0, "p50": 3.0, "p95": 7.0,
            },
        },
    }
    settings = SimpleNamespace(agent_apis={
        "writer": SimpleNamespace(model="writer-v1"),
        "evolver": SimpleNamespace(model="evolver-v1"),
        "review": SimpleNamespace(model="chair-v2"),
    })
    groups = llm_activity(metrics, settings)["request_groups"]
    assert groups[0]["calls"] == 3
    assert groups[0]["failures"] == 1
    assert groups[0]["role_candidates"] == ["writer"]
    assert groups[0]["duration_seconds"]["p95"] == 7.0
    assert groups[1]["calls"] == 2
    assert "verifier" in groups[1]["role_candidates"]


def test_llm_activity_attributes_explicit_agent_role():
    settings = SimpleNamespace(agent_apis={"review": SimpleNamespace(model="chair-v2")})
    metrics = {"counters": {"llm_requests{model=chair-v2,priority=2,role=verifier}": 5}}
    group = llm_activity(metrics, settings)["request_groups"][0]
    assert group["role"] == "verifier"
    assert group["role_candidates"] == ["verifier"]
    assert group["calls"] == 5


def test_stage_activity_recovers_overlapping_intervals(tmp_path: Path):
    from datetime import datetime

    started = 1_700_000_000.0
    stamp = lambda seconds: datetime.fromtimestamp(started + seconds).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    path = tmp_path / "pipeline.log"
    path.write_text("\n".join([
        f"{stamp(5)} INFO cuda_sft_qgen.graph.pipeline: review_stage start pending=2",
        f"{stamp(7)} INFO cuda_sft_qgen.graph.nodes.salvage: repair_stage start items=1",
        f"{stamp(8)} INFO cuda_sft_qgen.graph.pipeline: review_stage done duration_s=3.00",
        f"{stamp(9)} INFO cuda_sft_qgen.graph.nodes.salvage: repair_batch items=1 ok=1 retry=0",
        f"{stamp(10)} INFO cuda_sft_qgen.graph.nodes.generate: generate_batch done ok=1 salvage=0 retry=0 fail=0 pending=1 duration_s=10.0 ok_per_min=6",
    ]) + "\n", encoding="utf-8")
    activity = stage_activity(path, started)
    assert activity["interval_counts"] == {"review": 1, "repair": 1, "generation": 1}
    assert activity["overlaps"]["generation+review"]["overlap_seconds"] == pytest.approx(3.0)
    assert activity["overlaps"]["generation+repair"]["overlap_seconds"] == pytest.approx(2.0)
    assert activity["overlaps"]["repair+review"]["overlap_seconds"] == pytest.approx(1.0)


def test_run_gate_secures_output_tree_on_return(tmp_path: Path, monkeypatch):
    output = tmp_path / "private-gate"

    def write_artifacts(_settings, output, **_kwargs):
        nested = output / "nested"
        nested.mkdir()
        (output / "report.json").write_text("{}", encoding="utf-8")
        (nested / "sample.jsonl").write_text("{}\n", encoding="utf-8")
        return {"verdict": "baseline"}

    monkeypatch.setattr(gate_module, "_run_gate_impl", write_artifacts)
    report = gate_module.run_gate(SimpleNamespace(), output, phase="test")
    assert report["verdict"] == "baseline"
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "nested").stat().st_mode & 0o777 == 0o700
    assert (output / "report.json").stat().st_mode & 0o777 == 0o600
    assert (output / "nested" / "sample.jsonl").stat().st_mode & 0o777 == 0o600


def test_gate_holds_pipeline_lock_through_runtime_lifecycle(tmp_path: Path, fake_settings, monkeypatch):
    from cuda_sft_qgen.graph import app as graph_app
    from cuda_sft_qgen import pack as pack_module

    real_pipeline_lock = graph_app.pipeline_lock
    events = []
    lock_active = False

    @contextmanager
    def tracked_lock(data_dir):
        nonlocal lock_active
        with real_pipeline_lock(data_dir):
            lock_active = True
            events.append("lock_enter")
            try:
                yield
            finally:
                events.append("lock_exit")
                lock_active = False

    class Store:
        def load_accepted(self):
            return []

        def load_rejected(self):
            return []

        def load_pending(self):
            return []

        def export_training(self, path, *, require_independent_reviewer=False):
            assert lock_active
            assert path.name == "training.jsonl"
            assert require_independent_reviewer is True
            events.append("training_export")
            return {"questions": 0, "path": str(path)}

    class Runtime:
        store = Store()
        knowledge = None

        def refresh_knowledge(self, *, force=False):
            assert lock_active
            assert force is True
            events.append("knowledge_refresh")

        def close(self):
            assert lock_active
            events.append("runtime_close")

    def build_runtime(_settings):
        assert lock_active
        events.append("runtime_build")
        return Runtime()

    def run_pipeline(_settings, *, runtime, lock_held):
        assert lock_active
        assert lock_held is True
        assert runtime is not None
        events.append("pipeline")
        return {"status": "archived", "done": True}

    def load_pack(path):
        assert lock_active
        assert path.name == "pack.yaml"
        events.append("pack_load")
        return object()

    def ingest_pack(_knowledge, _pack, **kwargs):
        assert lock_active
        assert kwargs == {"apply": True, "enqueue_jobs": False}
        events.append("pack_ingest")
        return {}

    monkeypatch.setattr(graph_app, "pipeline_lock", tracked_lock)
    monkeypatch.setattr(pack_module, "load_pack", load_pack)
    monkeypatch.setattr(pack_module, "ingest_pack", ingest_pack)
    monkeypatch.setattr(gate_module, "build_runtime", build_runtime)
    monkeypatch.setattr(gate_module, "run_pipeline", run_pipeline)
    output = tmp_path / "gate-lock"
    report = gate_module.run_gate(fake_settings, output, phase="lock-test", target=1, allow_fake=True,
                                  ingest_packs=[tmp_path / "pack.yaml"])

    assert report["status"] == "archived"
    assert events == ["lock_enter", "runtime_build", "pack_load", "pack_ingest", "knowledge_refresh",
                      "pipeline", "knowledge_refresh", "training_export", "runtime_close", "lock_exit"]


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
