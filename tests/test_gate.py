"""Tests for throughput-gate comparison, median baseline, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuda_sft_qgen.cli import build_parser
from cuda_sft_qgen.gate import compare_gate_reports, median_baseline, percentile, write_verdict_md


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


def test_cli_rejects_fake(capsys):
    args = build_parser().parse_args(["gate", "--data-dir", "/tmp/gate-x", "--phase", "baseline", "--fake"])
    assert args.func(args) == 2
    assert "refuses --fake" in capsys.readouterr().err
