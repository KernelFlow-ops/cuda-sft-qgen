"""Tests for isolated chair evaluation on frozen labeled quality cases."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuda_sft_qgen import review_benchmark
from cuda_sft_qgen.models import QualityChecks
from cuda_sft_qgen.observability import get_metrics


FIXTURE = Path(__file__).parent / "fixtures" / "quality_cases.json"


def test_fixture_contains_complete_frozen_defects_and_controls():
    cases = review_benchmark.load_quality_cases(FIXTURE)
    defects = [case for case in cases if case["expected_action"] == "reject"]
    controls = [case for case in cases if case["expected_action"] == "accept"]
    assert [case["source"]["line"] for case in defects] == [5, 11, 18, 21, 25, 32, 42, 50]
    assert len(controls) == 4
    assert len({case["question"]["id"] for case in cases}) == 12
    assert all(case["annotation"] and case["question"]["tests_claim"] for case in cases)
    original = FIXTURE.parents[2] / "data" / "run50" / "accepted.jsonl"
    if original.exists():
        rows = [json.loads(line) for line in original.read_text(encoding="utf-8").splitlines()]
        assert all(case["question"] == rows[case["source"]["line"] - 1] for case in defects)


def test_benchmark_rejects_fake_quality_claims(fake_settings, tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="real reviewer"):
        review_benchmark.run_review_benchmark(fake_settings, output)
    assert not output.exists()


def test_benchmark_blind_review_snapshots_metrics_and_isolation(fake_settings, tmp_path, monkeypatch):
    cases = review_benchmark.load_quality_cases(FIXTURE)
    settings = replace(fake_settings, fake_llm=False)
    original = settings.data_dir / "accepted.jsonl"
    original.write_text("original corpus is not read or modified", encoding="utf-8")
    actions = ["accept", "timeout", *(["reject"] * 6), "reject", "accept", "accept", "accept"]
    calls = []
    runtimes = []

    class Reviewer:
        def complete_json(self, messages, **options):
            payload = json.loads(messages[1]["content"])
            assert not {"annotation", "expected_action", "case_id"} & payload.keys()
            assert payload["accepted_neighbors"] == []
            calls.append((payload, options))
            action = actions[len(calls) - 1]
            if action == "timeout":
                raise TimeoutError("mock reviewer timed out")
            metrics = get_metrics()
            labels = {"model": settings.review_model}
            for name, value in (("prompt_tokens", 10), ("completion_tokens", 5),
                                ("total_tokens", 15), ("usage_reports", 1)):
                metrics.increment("llm_" + name, value, labels)
            checks = {key: "pass" for key in QualityChecks.model_fields}
            if action == "reject":
                checks["correctness"] = "fail"
            return {"id": payload["id"], "action": action, "reason": "mock review outcome",
                    "checks": checks, "evidence": ["mock evidence, not a quality assessment"],
                    "internal_reference_answer": "mock answer" if action == "accept" else None,
                    "assessed_metadata": payload["requested_metadata"] if action == "accept" else None}

    def runtime_factory(isolated):
        assert isolated.data_dir != settings.data_dir
        assert isolated.embedding_provider == "fake"
        runtime = SimpleNamespace(settings=isolated, review_llm=Reviewer(), llm=None,
                                  knowledge=None, closed=False)
        runtime.close = lambda: setattr(runtime, "closed", True)
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(review_benchmark, "build_runtime", runtime_factory)
    output = tmp_path / "reports" / "benchmark.json"
    report = review_benchmark.run_review_benchmark(settings, output, cases_path=FIXTURE)
    assert report["summary"]["misses"] == 1
    assert report["summary"]["false_rejections"] == 1
    assert report["summary"]["pending"] == 1
    assert report["summary"]["detected_defects"] == 6
    assert report["summary"]["accepted_controls"] == 3
    assert report["usage"]["provider_reports"] == 11
    assert report["usage"]["total_tokens"] == 165
    assert report["results"][1]["usage"]["total_tokens"] is None
    assert report["results"][1]["reviewer_calls"][0]["error_type"] == "TimeoutError"
    assert report["results"][0]["question_snapshot"] == cases[0]["question"]
    assert report["reviewer"]["model"] == settings.review_model
    assert report["started_at"] and report["finished_at"] and report["duration_seconds"] >= 0
    assert all(result["latency_seconds"] >= 0 for result in report["results"])
    assert all(options["temperature"] == 0 for _, options in calls)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert output.stat().st_mode & 0o077 == 0
    assert original.read_text(encoding="utf-8") == "original corpus is not read or modified"
    assert runtimes[0].closed
    assert not runtimes[0].settings.data_dir.exists()


def test_fixture_rejects_duplicate_ids(tmp_path):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["cases"].append(document["cases"][0])
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        review_benchmark.load_quality_cases(path)
