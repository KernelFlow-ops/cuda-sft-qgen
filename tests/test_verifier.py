"""Deterministic verifier tests; real API coverage lives in the pilot command."""

import json

import pytest

from cuda_sft_qgen.models import Question
from cuda_sft_qgen.review.verifier import (
    answer_consistency,
    canonical_numeric,
    canonical_symbolic,
    compile_cuda,
    complexity_features,
    evaluate_candidate,
)


def _question(text: str, *, kind: str = "formula", **overrides) -> Question:
    data = dict(
        id="q_verify",
        question=text,
        tests_claim="The CUDA claim is testable.",
        topic_path=["T01", "test", "q_verify"],
        l3_id="q_verify",
        l3_claim="The CUDA claim is testable.",
        question_type="syntax" if "```" in text else "conceptual",
        difficulty=2,
        bloom="apply",
        persona="cuda_interviewer",
        hardware="agnostic",
        granularity="kernel",
        language="en",
        kind=kind,
    )
    data.update(overrides)
    return Question(**data)


def test_answer_consistency_rejects_target_mismatch_and_defers_ties():
    assert answer_consistency("32", ["32", "32", "16"])["status"] == "pass"
    assert answer_consistency("32", ["16", "16", "32"])["status"] == "pending"
    assert answer_consistency("32", ["16", "32"])["status"] == "pending"


def test_cuda_complexity_features_are_explainable():
    easy = complexity_features(_question("Which CUDA API launches the kernel?"))
    hard = complexity_features(_question(
        "If a Hopper kernel uses mbarrier across two streams, compile this code and "
        "identify the observable ordering.\n```cuda\n__global__ void k() {}\n```",
        kind="failure",
    ))
    assert hard["dependency_depth"] > easy["dependency_depth"]
    assert hard["architecture_specificity"] >= 1
    assert hard["code_size"] > 0


def test_compile_cuda_reports_real_tool_or_unavailable():
    result = compile_cuda("```cuda\n__global__ void k() {}\n```")
    assert result["status"] in {"pass", "fail", "pending", "unavailable"}
    if result["status"] == "pass":
        assert result["returncode"] == 0


def test_safe_numeric_and_symbolic_canonicalization_reject_code():
    assert canonical_numeric(0) == "0"
    assert canonical_numeric("0.5 + 1/2") == "1"
    assert canonical_numeric("128 bytes") == canonical_numeric("128B") == "128 B"
    assert canonical_numeric("128B") != canonical_numeric("128b")
    assert canonical_symbolic("x + x") == "2*x"
    with pytest.raises(ValueError):
        canonical_numeric("__import__('os').system('id')")
    with pytest.raises(ValueError):
        canonical_symbolic("f(x)")


class ScriptedReviewer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.payloads = []

    def complete_json(self, messages, **kwargs):
        self.payloads.append(json.loads(messages[1]["content"]))
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _scalar_question(target="1", **overrides):
    spec = {
        "mode": "scalar", "answer_type": "number", "expected_behavior": "result is one",
        "rubric": ["exact numeric answer"],
    }
    return _question(
        "Compute 1/2 + 1/2.", draft_target=target,
        draft_plan={"structured": True}, verification_spec=spec,
        **overrides,
    )


def test_blind_solver_never_receives_target_spec_rubric_or_parents():
    reviewer = ScriptedReviewer([{"answer": "1", "evidence": "fraction sum"}] * 3)
    result = evaluate_candidate(
        _scalar_question(mixup_context={"parents": [{"id": "secret"}]}), reviewer,
        {"answer_consistency_enabled": True, "verification_solver_k": 3},
    )
    assert result["status"] == "pass"
    assert len(reviewer.payloads) == 3
    encoded = json.dumps(reviewer.payloads)
    assert "draft_target" not in encoded
    assert "rubric" not in encoded
    assert "secret" not in encoded


def test_incomplete_votes_and_target_mismatch_are_pending():
    incomplete = ScriptedReviewer([
        {"answer": "1", "evidence": "ok"},
        {"answer": "1"},
        {"answer": "1", "evidence": "ok"},
    ])
    assert evaluate_candidate(
        _scalar_question(), incomplete,
        {"answer_consistency_enabled": True, "verification_solver_k": 3},
    )["status"] == "pending"
    mismatch = ScriptedReviewer([{"answer": "2", "evidence": "independent"}] * 3)
    result = evaluate_candidate(
        _scalar_question(), mismatch,
        {"answer_consistency_enabled": True, "verification_solver_k": 3},
    )
    assert result["status"] == "pending"
    assert result["reason"] == "majority_differs_from_writer_target"


def test_numeric_fraction_and_decimal_votes_are_equivalent():
    reviewer = ScriptedReviewer([
        {"answer": "1/2", "evidence": "fraction"},
        {"answer": "0.5", "evidence": "decimal"},
        {"answer": "2/4", "evidence": "reduced"},
    ])
    result = evaluate_candidate(
        _scalar_question(target="0.5"), reviewer,
        {"answer_consistency_enabled": True, "verification_solver_k": 3},
    )
    assert result["status"] == "pass"
    assert result["evidence"]["majority"] == "1/2"


def test_open_mode_uses_three_answers_and_one_rubric_call():
    outputs = [{"response": f"answer {i}"} for i in range(3)] + [{"scores": [
        {"score": .9, "evidence": "grounded"},
        {"score": .7, "evidence": "adequate"},
        {"score": .8, "evidence": "specific"},
    ]}]
    reviewer = ScriptedReviewer(outputs)
    q = _question(
        "Explain this CUDA tradeoff.", draft_target="private",
        draft_plan={"structured": True},
        verification_spec={"mode": "open", "answer_type": "text",
                           "expected_behavior": "explains tradeoff", "rubric": ["CUDA-correct"]},
    )
    result = evaluate_candidate(q, reviewer, {"answer_consistency_enabled": True})
    assert result["status"] == "pass"
    assert result["evidence"]["rip_score"] == .7
    assert len(reviewer.payloads) == 4
    assert "rubric" not in reviewer.payloads[0]
    assert reviewer.payloads[-1]["rubric"] == ["CUDA-correct"]


def test_open_low_score_is_valid_rip_evidence_not_global_reject():
    outputs = [{"response": "weak"}] * 3 + [{"scores": [
        {"score": .1, "evidence": "weak"}, {"score": .2, "evidence": "partial"},
        {"score": .3, "evidence": "some detail"},
    ]}]
    q = _question("Explain CUDA.", draft_target="private", draft_plan={"structured": True},
                  verification_spec={"mode": "open", "answer_type": "text",
                                     "expected_behavior": "explanation", "rubric": ["correct"]})
    result = evaluate_candidate(q, ScriptedReviewer(outputs), {"answer_consistency_enabled": True})
    assert result["status"] == "pass"
    assert result["evidence"]["rip_score"] == .1


def test_compile_snippet_is_not_accepted_as_error_oracle():
    q = _question(
        "Will it compile?", draft_target="failure", draft_plan={"structured": True},
        verification_spec={
            "mode": "compile", "answer_type": "enum", "expected_behavior": "compile failure",
            "rubric": ["compiler outcome"], "code": "__global__ void k() {}",
            "architecture": "sm_86", "expected_compile": "failure",
        },
    )
    result = evaluate_candidate(q, ScriptedReviewer([]), {"answer_consistency_enabled": True})
    assert result["status"] == "pending"
    assert result["reason"] == "incomplete_translation_unit"


def test_expected_compile_failure_can_pass_as_negative_oracle(monkeypatch):
    from cuda_sft_qgen.review import verifier

    class Temp:
        def cleanup(self):
            pass

    monkeypatch.setattr(verifier, "_compile", lambda spec, cfg, executable: ({
        "status": "completed", "returncode": 1, "stdout": "", "stderr": "error: unknown identifier",
    }, Temp()))
    q = _question(
        "Will this complete program compile?", draft_target="failure", draft_plan={"structured": True},
        verification_spec={
            "mode": "compile", "answer_type": "enum", "expected_behavior": "compile failure",
            "rubric": ["compiler rejects unknown identifier"],
            "code": "#include <cuda_runtime.h>\nint main(){ return definitely_unknown; }",
            "architecture": "sm_86", "expected_compile": "failure", "expected_diagnostic": "error",
        },
    )
    result = evaluate_candidate(q, ScriptedReviewer([]),
                                {"answer_consistency_enabled": True, "compile_timeout_seconds": 30})
    assert result["status"] == "pass"


def test_missing_tool_is_pending(monkeypatch):
    from cuda_sft_qgen.review import verifier

    real_which = verifier.shutil.which
    monkeypatch.setattr(verifier.shutil, "which", lambda name: None if name == "nvcc" else real_which(name))
    q = _question(
        "Compile this program.", draft_target="success", draft_plan={"structured": True},
        verification_spec={
            "mode": "compile", "answer_type": "enum", "expected_behavior": "compile success",
            "rubric": ["compiler accepts"],
            "code": "#include <cuda_runtime.h>\nint main(){ return 0; }",
            "architecture": "sm_86", "expected_compile": "success",
        },
    )
    result = evaluate_candidate(q, ScriptedReviewer([]), {"answer_consistency_enabled": True})
    assert result["status"] == "pending"
    assert result["reason"] == "nvcc_not_found"


def test_runtime_nonzero_exit_is_pending_not_claim_failure(monkeypatch, tmp_path):
    from cuda_sft_qgen.review import verifier

    class Temp:
        def cleanup(self):
            pass

    monkeypatch.setattr(verifier, "_gpu_arch", lambda: "sm_86")
    monkeypatch.setattr(verifier, "_compile", lambda spec, cfg, executable: ({
        "status": "completed", "returncode": 0, "workdir": str(tmp_path),
    }, Temp()))
    monkeypatch.setattr(verifier, "_sandbox", lambda command, root, gpu: command)
    monkeypatch.setattr(verifier, "_run", lambda command, root, timeout: {
        "status": "completed", "returncode": 10, "stdout": "", "stderr": "cudaMalloc failed",
    })
    q = _question("Run CUDA.", draft_target="42", draft_plan={"structured": True},
                  verification_spec={"mode": "runtime", "answer_type": "number",
                                     "expected_behavior": "prints 42", "rubric": ["stdout"],
                                     "code": "#include <cuda_runtime.h>\nint main(){return 0;}",
                                     "architecture": "sm_86", "expected_compile": "success",
                                     "expected_stdout": "42", "sanitizer": "none"})
    result = evaluate_candidate(q, ScriptedReviewer([]), {"answer_consistency_enabled": True})
    assert result["status"] == "pending"
    assert result["reason"] == "runtime_nonzero_exit_inconclusive"


def test_disabled_and_draft_errors_do_not_call_reviewer():
    reviewer = ScriptedReviewer([])
    legacy = _question("Legacy CUDA question")
    assert evaluate_candidate(legacy, reviewer, {"answer_consistency_enabled": True})["status"] == "disabled"
    bad = _scalar_question(draft_validation_errors=["missing_target"])
    assert evaluate_candidate(bad, reviewer, {"answer_consistency_enabled": True})["status"] == "pending"
    assert reviewer.payloads == []
