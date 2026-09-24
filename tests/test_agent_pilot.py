"""Offline contract tests for the real-provider agent pilot harness."""

from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuda_sft_qgen import agent_pilot
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.store.archive import Archive
from cuda_sft_qgen.taxonomy import load_taxonomy


class _RealDelegate:
    def complete(self, messages, **kwargs):
        return "unused"


class _RealEmbedder:
    def embed(self, texts):
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _install_offline_pipeline(monkeypatch, runtimes):
    def runtime_factory(settings):
        runtime = SimpleNamespace(
            settings=settings,
            store=Archive(settings.data_dir),
            llm=_RealDelegate(),
            review_llm=None,
            embedder=_RealEmbedder(),
            taxonomy=load_taxonomy(settings.taxonomy_path),
            closed=False,
        )

        def close():
            runtime.store.close()
            runtime.closed = True

        runtime.close = close
        runtimes.append(runtime)
        return runtime

    def generate(brief, steering):
        # Exercise both budget channels without using a provider.
        get_runtime().llm.complete([{"role": "user", "content": "writer"}])
        return Question(
            id="q_" + brief.task_id,
            question=f"{brief.kind} CUDA question at difficulty {brief.difficulty}?",
            tests_claim=brief.l3_claim,
            topic_path=list(brief.topic_path),
            l3_id=brief.l3_id,
            l3_claim=brief.l3_claim,
            question_type=brief.question_type,
            difficulty=brief.difficulty,
            bloom=brief.bloom,
            persona=brief.persona,
            hardware=brief.hardware,
            granularity=brief.granularity,
            language=brief.language,
            kind=brief.kind,
            claim_revision=brief.claim_revision,
            source_ids=list(brief.source_ids),
            mixup_mode=brief.mixup_mode,
            parent_ids=list(brief.parent_ids),
            parent_difficulties=list(brief.parent_difficulties),
            difficulty_target=brief.difficulty_target,
            difficulty_raw_score=brief.difficulty_raw_score,
            draft_validation_errors=["legacy_missing_private_contract"] if not brief.mixup_mode else [],
        )

    def score(state):
        question = state["pending"][0]
        return {
            "last_diversity": {
                "scored": [{"question": question, "decision": {"verdict": "unique"},
                            "neighbors": [], "vector_row": 0}],
                "vectors": [[1.0, 0.0]],
                "report": {"accepted_count": 0, "pending_count": 1},
            }
        }

    def chair(state):
        get_runtime().review_llm.complete([{"role": "user", "content": "review"}])
        item = state["last_diversity"]["scored"][0]
        question = Question.model_validate(item["question"])
        question.review_details = {
            "action": "accept",
            "checks": {
                "correctness": "pass", "answerability": "pass", "claim_alignment": "pass",
                "hardware_compatibility": "pass", "answer_leakage": "pass",
                "metadata": "pass", "duplication": "pass",
            },
            "evidence": ["offline contract evidence"],
            "verification_evidence": {"answer_consistency": {"status": "pass", "votes": ["x"]}},
        }
        decision = {
            "id": question.id, "action": "accept", "reason": "offline",
            "question": question.model_dump(), "vector_row": 0,
        }
        payload = dict(state["last_diversity"])
        payload["final_decisions"] = [decision]
        return {"last_diversity": payload, "steering_brief": {}, "review_rounds": 1}

    def archive(state):
        for decision in state["last_diversity"]["final_decisions"]:
            question = Question.model_validate(decision["question"])
            if decision["action"] == "accept":
                get_runtime().store.append_accepted(question)
        return {"status": "archived"}

    monkeypatch.setattr(agent_pilot, "build_runtime", runtime_factory)
    monkeypatch.setattr(agent_pilot, "generate_from_brief", generate)
    monkeypatch.setattr(agent_pilot, "review_score_node", score)
    monkeypatch.setattr(agent_pilot, "review_chair_node", chair)
    monkeypatch.setattr(agent_pilot, "archive_node", archive)


def test_agent_pilot_rejects_fake_providers(fake_settings, tmp_path):
    with pytest.raises(ValueError, match="real LLM"):
        agent_pilot.run_agent_pilot(fake_settings, tmp_path / "pilot.json", groups=("A",))
    real_llm_fake_embedding = replace(fake_settings, fake_llm=False, embedding_provider="fake")
    with pytest.raises(ValueError, match="real embedding"):
        agent_pilot.run_agent_pilot(real_llm_fake_embedding, tmp_path / "pilot.json", groups=("A",))


def test_agent_pilot_isolated_checkpoint_budget_and_mixup(fake_settings, tmp_path, monkeypatch):
    settings = replace(fake_settings, fake_llm=False, embedding_provider="nvidia")
    runtimes = []
    _install_offline_pipeline(monkeypatch, runtimes)
    output = tmp_path / "reports" / "pilot.json"
    report = agent_pilot.run_agent_pilot(
        settings,
        output,
        max_calls=40,
        samples=2,
        groups=("A", "D"),
        deadline_seconds=60,
        kinds=("mechanism",),
        random_seed=7,
    )
    assert report["status"] == "completed"
    assert report["calls"]["logical_total"] == sum(report["calls"]["by_channel"].values())
    assert report["calls"]["logical_total"] <= 40
    assert report["groups"]["A"]["flags"] == {
        "prompt_cot": False, "answer_consistency": False,
        "mixup": False, "legacy_output": True,
    }
    assert report["groups"]["D"]["mixup"]["status"] == "evaluated"
    assert report["groups"]["D"]["bootstrap"]["cross_group_parent_import"] is False
    assert report["calls"]["group_limits"] == {"A": 20, "D": 20}
    assert report["config"]["request_timeout_seconds"] == 30
    assert report["config"]["request_max_attempts"] == 1
    assert report["privacy"] == {
        "status": "pass",
        "policy_version": 1,
        "private_payload_key_matches": 0,
        "configured_secret_matches": 0,
        "private_reasoning_persisted": False,
        "full_provider_requests_persisted": False,
    }
    assert report["groups"]["A"]["usage"]["available"] is False
    for runtime in runtimes:
        generation = runtime.settings.default_yaml["generation"]
        assert generation["request_timeout_seconds"] == 30
        assert generation["request_max_attempts"] == 1
    d_modes = {
        item["public_question"]["mixup_mode"]
        for item in report["groups"]["D"]["candidates"]
        if item.get("public_question") and item["public_question"]["mixup_mode"]
    }
    assert d_modes == {"decomposed", "hybrid"}
    assert report["groups"]["A"]["summary"]["quality_claim_status"] == "not_qualified"
    assert report["cost_usd"] is None
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    serialized = output.read_text(encoding="utf-8")
    assert "internal_reference_answer" not in serialized
    assert "draft_target" not in serialized
    assert "verification_spec" not in serialized
    assert '"x"' not in serialized
    assert all(runtime.closed for runtime in runtimes)
    assert not any(path.name.startswith("cuda-sft-agent-pilot-") for path in tmp_path.iterdir())


def test_agent_pilot_stops_at_logical_call_budget(fake_settings, tmp_path, monkeypatch):
    settings = replace(fake_settings, fake_llm=False, embedding_provider="nvidia")
    runtimes = []
    _install_offline_pipeline(monkeypatch, runtimes)
    report = agent_pilot.run_agent_pilot(
        settings,
        tmp_path / "pilot.json",
        max_calls=1,
        samples=1,
        groups=("B",),
        deadline_seconds=60,
        kinds=("api",),
    )
    assert report["status"] == "partial"
    assert report["calls"]["logical_total"] == 1
    assert report["groups"]["B"]["candidates"][0]["failure"]["stage"] == "chair"


def test_public_decision_uses_new_verifier_shape_without_answers():
    public = agent_pilot._public_decision({
        "action": "pending",
        "reason": "review uncertain; answer: secret-value",
        "review": {
            "checks": {"correctness": "uncertain"},
            "verification_evidence": {
                "status": "pending",
                "reason": "insufficient_votes",
                "difficulty_raw_score": 3.5,
                "evidence": {
                    "votes": ["secret-value", "other-secret"],
                    "agreement": 0.5,
                    "valid_votes": 2,
                    "rip_score": 0.7,
                    "compile": {
                        "status": "failed", "tool": "nvcc", "returncode": 1,
                        "reason": "compile_failed", "stderr": "private diagnostic",
                    },
                },
            },
        },
    })
    assert public["action"] == "pending"
    assert public["decision_reason_code"] == "insufficient_votes"
    assert public["defect_tags"] == ["correctness:uncertain"]
    assert public["evidence_summary"] == {
        "status": "pending",
        "reason_code": "insufficient_votes",
        "agreement": 0.5,
        "valid_votes": 2,
        "rip_score": 0.7,
        "tool": "nvcc",
        "returncode": 1,
        "tool_status": "failed",
        "tool_reason_code": "compile_failed",
        "review_evidence_count": 0,
    }
    assert "secret-value" not in json.dumps(public)


def test_public_reason_redacts_unlabelled_oracle_answers():
    reason = agent_pilot._public_reason(
        "The correct answer is 16; expected answer: 32. 正确答案为 8。"
    )
    assert reason == "The answer=[redacted]; answer=[redacted]. 答案=[已脱敏]。"
    assert not any(value in reason for value in ("16", "32", "8"))


def test_cot_only_group_preserves_model_action_but_cannot_accept(fake_settings, tmp_path, monkeypatch):
    settings = replace(fake_settings, fake_llm=False, embedding_provider="nvidia")
    runtimes = []
    _install_offline_pipeline(monkeypatch, runtimes)
    report = agent_pilot.run_agent_pilot(
        settings,
        tmp_path / "pilot.json",
        max_calls=8,
        samples=1,
        groups=("B",),
        deadline_seconds=60,
        kinds=("formula",),
    )
    candidate = report["groups"]["B"]["candidates"][0]
    assert candidate["model_action"] == "accept"
    assert candidate["action"] == "pending"
    assert report["groups"]["B"]["summary"]["actions"] == {"pending": 1}


def test_privacy_audit_detects_private_fields_and_configured_secret(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-secret-value")
    audit = agent_pilot._privacy_audit({
        "pilot": "agent_ablation_abcd",
        "nested": {"draft_target": "nvapi-test-secret-value"},
    })
    assert audit["status"] == "fail"
    assert audit["private_payload_key_matches"] == 1
    assert audit["configured_secret_matches"] == 1


def test_sanitize_pilot_report_removes_legacy_free_text_reason():
    report = {
        "groups": {"A": {"candidates": [{
            "action": "reject",
            "reason": "The correct answer is a private oracle value.",
            "checks": {"correctness": "fail", "metadata": "pass"},
            "evidence_summary": {},
        }]}}
    }
    sanitized = agent_pilot.sanitize_pilot_report(report)
    row = sanitized["groups"]["A"]["candidates"][0]
    assert "reason" not in row
    assert row["decision_reason_code"] == "reviewer_checks_failed"
    assert row["defect_tags"] == ["correctness:fail"]
    assert "private oracle" not in json.dumps(sanitized)
