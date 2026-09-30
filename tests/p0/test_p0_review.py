"""P0 review regressions: second-pass trigger/finality, low-value rejects, open-mode
floor, pre-Chair duplicate and contract rejects, deferred checks, reviewer
independence, derived hardware buckets, metadata relabel, and repair routing."""
from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from p0_helpers import make_question

CHECKS = ("correctness", "answerability", "claim_alignment", "hardware_compatibility", "answer_leakage",
          "metadata", "duplication")
REAL_ORIGIN = {"writer_real_api": True, "writer_provider": "nvidia", "writer_model": "writer-model"}
GOOD_VALUES = {"pedagogical_value": 4, "discriminativeness": 4, "realism": 4}


def _review(qid, action="accept", fail=(), uncertain=(), **extra):
    checks = {k: ("fail" if k in fail else "uncertain" if k in uncertain else "pass") for k in CHECKS}
    obj = {"id": qid, "action": action, "reason": "reviewed", "checks": checks, "evidence": ["concrete evidence"],
           "internal_reference_answer": "4", "value_scores": dict(GOOD_VALUES),
           "assessed_metadata": {"question_type": "conceptual", "difficulty": 2, "bloom": "apply",
                                 "granularity": "kernel", "language": "en"}}
    obj.update(extra)
    return obj


def _rules(review_obj, **kw):
    from cuda_sft_qgen.graph.nodes.review_chair import apply_decision_rules
    from cuda_sft_qgen.models import QualityReview

    args = dict(verification_status="pass", verification_reason="ok", structured=True, is_fake=False,
                construction_error=None, exploration=False, min_value_score=3)
    args.update(kw)
    return apply_decision_rules(QualityReview.model_validate(review_obj), **args)


class _ScriptedReviewer:
    """Returns queued review objects per question id; counts every call."""

    def __init__(self, script):
        self.script = {qid: list(items) for qid, items in script.items()}
        self.calls = 0

    def complete_json(self, messages, **kwargs):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        return self.script[payload["id"]].pop(0)


def _chair_fake(review_for):
    """A FakeLLM whose Chair answer is ``review_for(payload)`` (verification stays synthetic)."""
    from cuda_sft_qgen.llm import FakeLLM

    class ChairFake(FakeLLM):
        calls = 0

        def complete(self, messages, **kwargs):
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            if "QUALITY_REVIEW_V" not in system:
                return super().complete(messages, **kwargs)
            type(self).calls += 1
            return json.dumps(review_for(json.loads(messages[-1]["content"])))

    return ChairFake()


@pytest.fixture
def mock_rt():
    rt = MagicMock()
    rt.knowledge = None
    rt.settings.review = {"second_pass_on_uncertain": True}
    return rt


@pytest.fixture
def active(fake_runtime):
    from cuda_sft_qgen.runtime import reset_runtime, set_runtime

    token = set_runtime(fake_runtime)
    yield fake_runtime
    reset_runtime(token)


# --- item 10: the second pass triggers on type OR kind ---------------------------------------------

def test_second_pass_triggers_for_formula_and_numeric_kinds():
    from types import SimpleNamespace

    from cuda_sft_qgen.graph.nodes.review_chair import _needs_second_pass

    settings = SimpleNamespace(review={"second_pass_on_uncertain": True})
    pending = {"action": "pending", "review": {"checks": {"correctness": "uncertain", "answerability": "pass"}}}
    assert _needs_second_pass(make_question(question_type="conceptual", kind="formula"), pending, settings)
    assert _needs_second_pass(make_question(question_type="conceptual", kind="numeric"), pending, settings)
    assert _needs_second_pass(make_question(question_type="code_complete", kind="mechanism"), pending, settings)
    assert not _needs_second_pass(make_question(question_type="conceptual", kind="mechanism"), pending, settings)


# --- item 11: a second-pass reject with a failed check is final ------------------------------------

def test_second_pass_reject_with_failed_check_is_final(mock_rt):
    from cuda_sft_qgen.graph.nodes.review_chair import _parallel_chair_decisions

    first = lambda qid: _review(qid, "pending", uncertain=("correctness",), assessed_metadata=None)
    reviewer = _ScriptedReviewer({
        "q_fail": [first("q_fail"), _review("q_fail", "reject", fail=("correctness",), assessed_metadata=None)],
        "q_value": [first("q_value"), _review("q_value", "reject", defect_codes=["low_value"],
                                              assessed_metadata=None)],
        "q_accept": [first("q_accept"), _review("q_accept")],
    })
    mock_rt.review_llm = reviewer
    items = [{"question": make_question(id=qid, question_type="debug", kind="failure").model_dump(),
              "decision": {"verdict": "unique"}} for qid in ("q_fail", "q_value", "q_accept")]
    with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_rt):
        results = _parallel_chair_decisions(items, max_workers=1)
    assert reviewer.calls == 6
    assert results["q_fail"]["action"] == "reject" and results["q_fail"]["review"]["second_pass"] is True
    assert results["q_fail"]["reason"].startswith("chair:correctness:")
    assert results["q_value"]["action"] == "pending"
    assert results["q_value"]["reason"].startswith("second_pass_unconfirmed:")
    assert results["q_accept"]["action"] == "accept" and results["q_accept"]["review"]["second_pass"] is True


# --- item 12: a Chair reject for low value alone is a valid reject --------------------------------

def test_quality_review_allows_non_check_defect_reject():
    from cuda_sft_qgen.models import QualityReview

    assert QualityReview.model_validate(_review("q", "reject", defect_codes=["low_value"])).action == "reject"
    with pytest.raises(ValueError, match="failed check or a non-check defect"):
        QualityReview.model_validate(_review("q", "reject"))
    with pytest.raises(ValueError):
        QualityReview.model_validate(_review("q", "reject", defect_codes=["metadata"]))


def test_low_value_chair_reject_through_full_chair_path(active, sample_question):
    from cuda_sft_qgen.graph.nodes.review import review_chair_node, review_score_node

    active.review_llm = _chair_fake(lambda p: _review(
        p["id"], "reject", defect_codes=["low_value"], assessed_metadata=None,
        value_scores={"pedagogical_value": 1, "discriminativeness": 2, "realism": 2}))
    q = sample_question(id="q_low", question="Why does a 32-thread warp reading stride-12 floats conflict?")
    decision = review_chair_node(review_score_node({"pending": [q.model_dump()]}))["last_diversity"]["final_decisions"][0]
    assert decision["action"] == "reject"
    assert decision["reason"].startswith("chair:low_value:")
    assert decision["question"]["defect_codes"] == ["low_value"]
    assert type(active.review_llm).calls == 1


# --- item 13: open-mode verification needs a minimum rubric score ----------------------------------

def _open_question():
    return make_question(question="Explain why this CUDA kernel's shared-memory tiling reduces DRAM traffic.",
                         draft_target="private", draft_plan={"structured": True},
                         verification_spec={"mode": "open", "answer_type": "text", "expected_behavior": "explain",
                                            "rubric": ["names data reuse", "quantifies traffic"]})


class _OpenReviewer:
    def __init__(self, scores):
        self.outputs = [{"response": f"answer {i}"} for i in range(3)] + [
            {"scores": [{"score": s, "evidence": "rubric"} for s in scores]}]

    def complete_json(self, messages, **kwargs):
        return self.outputs.pop(0)


@pytest.mark.parametrize("scores,floor,status,reason", [
    ([.9, .5, .8], None, "pending", "open_rubric_below_min"),
    ([.9, .6, .8], None, "pass", "open_rip_score_valid"),
    ([.9, .5, .8], .4, "pass", "open_rip_score_valid"),
])
def test_open_mode_requires_minimum_rubric_score(scores, floor, status, reason):
    from cuda_sft_qgen.review.verifier import evaluate_candidate

    cfg = {"answer_consistency_enabled": True, **({"open_min_score": floor} if floor is not None else {})}
    result = evaluate_candidate(_open_question(), _OpenReviewer(scores), cfg)
    assert (result["status"], result["reason"]) == (status, reason)
    assert result["evidence"]["rip_score"] == min(scores)


def test_open_min_score_default(fake_settings):
    assert fake_settings.review["open_min_score"] == 0.6


# --- item 14: hard duplicates are rejected before verification or the Chair ------------------------

@pytest.mark.parametrize("reason", ["L0_exact", "L1_template", "L2_semantic", "L2_gist_crosslingual"])
def test_hard_duplicate_rejected_without_llm_call(mock_rt, reason):
    from cuda_sft_qgen.graph.nodes.review_chair import _parallel_chair_decision

    mock_rt.review_llm = reviewer = _ScriptedReviewer({})
    q = make_question(id="dup")
    with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_rt), \
            patch("cuda_sft_qgen.review.verifier.evaluate_candidate") as verify:
        result = _parallel_chair_decision({"question": q.model_dump(),
                                           "decision": {"verdict": "dup", "reason": reason}}, "prompt")
    assert (result["action"], result["reason"]) == ("reject", reason)
    assert result["review"]["defect_codes"] == ["duplicate"]
    assert reviewer.calls == 0 and not verify.called


def test_exact_duplicate_in_chair_node_spends_no_call_but_gray_does(active, sample_question):
    from cuda_sft_qgen.graph.nodes.review import review_chair_node, review_score_node

    original = sample_question(id="orig", question="Why do 32 banks make stride-12 float reads 4-way conflicted?")
    active.store.append_accepted(original)
    active.deduper.add_question(make_question(id="orig", question=original.question,
                                              exact_hash=original.exact_hash or _exact(original.question)))
    active.review_llm = _chair_fake(lambda p: _review(p["id"], assessed_metadata=None, action="pending",
                                                      uncertain=("correctness",)))
    copy = sample_question(id="copy", question=original.question)
    decision = review_chair_node(review_score_node({"pending": [copy.model_dump()]}))["last_diversity"][
        "final_decisions"][0]
    assert (decision["action"], decision["reason"]) == ("reject", "L0_exact")
    assert type(active.review_llm).calls == 0
    fresh = sample_question(id="fresh", question="How does cudaMemcpyAsync overlap with a kernel on two streams?")
    review_chair_node(review_score_node({"pending": [fresh.model_dump()]}))
    assert type(active.review_llm).calls == 1


def _exact(text):
    from cuda_sft_qgen.review.fingerprint import fingerprint

    return fingerprint(text).exact


# --- item 15: a broken Writer contract is a writer-fixable reject ----------------------------------

def test_draft_validation_errors_reject_and_queue_one_rewrite(active, sample_brief, sample_question):
    from cuda_sft_qgen.graph.nodes.review import review_chair_node, review_score_node
    from cuda_sft_qgen.graph.nodes.review_archive import enqueue_review_repairs
    from cuda_sft_qgen.models import SteeringBrief

    active.settings.fake_llm = False
    active.review_llm = reviewer = _ScriptedReviewer({})
    task_id = active.store.save_generation_task(sample_brief().model_dump())
    q = sample_question(id="q_contract", generation_provenance={**REAL_ORIGIN, "task_id": task_id},
                        draft_validation_errors=["missing_draft_target", "missing_verification_spec"])
    decisions = review_chair_node(review_score_node({"pending": [q.model_dump()]}))["last_diversity"]["final_decisions"]
    decision = decisions[0]
    assert decision["action"] == "reject"
    assert decision["reason"] == "writer_contract:missing_draft_target,missing_verification_spec"
    assert decision["question"]["defect_codes"] == ["writer_contract"]
    assert reviewer.calls == 0
    assert enqueue_review_repairs(active, decisions, SteeringBrief()) == 1


# --- item 16: a deferred accept keeps the Chair's checks ------------------------------------------

def test_deferred_accept_keeps_checks_and_records_reason(active, sample_question):
    from cuda_sft_qgen.graph.nodes.review_chair import _parallel_chair_decision
    from cuda_sft_qgen.models import QualityReview

    q = sample_question(id="q_defer", verification_spec={"mode": "scalar"}, draft_target="4")
    active.review_llm = _ScriptedReviewer({"q_defer": [_review("q_defer")]})
    result = _parallel_chair_decision({"question": q.model_dump()}, "prompt",
                                      {"status": "pending", "reason": "low_solver_agreement"})
    details = result["review"]
    assert result["action"] == "pending" and result["reason"] == "verification:low_solver_agreement"
    assert details["checks"] == {k: "pass" for k in CHECKS}
    assert details["deferred_reason"] == "verification:low_solver_agreement"
    assert (details["model_action"], details["action"]) == ("accept", "pending")
    fields = set(QualityReview.model_fields)
    assert QualityReview.model_validate({k: v for k, v in details.items() if k in fields}).action == "pending"


# --- item 17: independence is computed, and only gates export when configured -----------------------

@pytest.mark.parametrize("review_change,expected", [
    ({}, False), ({"model": "other-model"}, True), ({"provider": "openai"}, True),
    ({"base_url": "https://review.example/v1"}, True),
])
def test_reviewer_independent_is_truthful(active, sample_question, review_change, expected):
    from cuda_sft_qgen.graph.nodes.review_chair import _parallel_chair_decision

    apis = dict(active.settings.agent_apis)
    writer = replace(apis["writer"], model="writer-model", provider="nvidia", base_url="https://writer.example/v1")
    apis.update(writer=writer, evolver=writer, review=replace(writer, **review_change))
    active.settings.agent_apis = apis
    q = sample_question(id="q_ind", generation_provenance=dict(REAL_ORIGIN))
    active.review_llm = _ScriptedReviewer({"q_ind": [_review("q_ind")]})
    result = _parallel_chair_decision({"question": q.model_dump()}, "prompt")
    assert result["action"] == "accept"
    assert result["review"]["reviewer_independent"] is expected


def _exportable(sample_question, qid, independent):
    from cuda_sft_qgen.models import QualityChecks

    return sample_question(id=qid, question=f"Exportable CUDA question {qid}?", generation_provenance=dict(REAL_ORIGIN),
                           internal_reference_answer="private", review_details={
                               "id": qid, "action": "accept", "reason": "verified",
                               "checks": QualityChecks(**{k: "pass" for k in CHECKS}).model_dump(),
                               "evidence": ["checked"], "internal_reference_answer": "private",
                               "assessed_metadata": {"question_type": "conceptual", "difficulty": 2,
                                                     "bloom": "understand", "granularity": "kernel", "language": "zh"},
                               "reviewer_model": "real-reviewer", "reviewer_independent": independent})


def test_export_counts_non_independent_and_gates_only_when_required(tmp_path, sample_question):
    from cuda_sft_qgen.store.archive import Archive

    archive = Archive(tmp_path / "data")
    for qid, independent in (("q_same", False), ("q_other", True)):
        archive.append_accepted(_exportable(sample_question, qid, independent))
    default = archive.export_training(tmp_path / "default.jsonl")
    assert (default["questions"], default["skipped_unverified"], default["non_independent_reviewer"]) == (2, 0, 1)
    curriculum = json.loads((tmp_path / "default.jsonl.curriculum.json").read_text(encoding="utf-8"))
    assert curriculum["non_independent_reviewer"] == 1
    strict = archive.export_training(tmp_path / "strict.jsonl", require_independent_reviewer=True)
    assert (strict["questions"], strict["skipped_unverified"], strict["independent_reviewer_required"]) == (1, 1, True)


def test_require_independent_reviewer_default_is_off(fake_settings):
    assert fake_settings.review["require_independent_reviewer_for_export"] is False


# --- item 18: hardware buckets are derived from arch_target ----------------------------------------

def test_hardware_bucket_mapping():
    from cuda_sft_qgen import enums

    assert [enums.hardware_bucket(a) for a in ("sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90a", "sm_120", "any")] \
        == ["volta", "turing", "ampere", "ampere", "ada", "hopper", "blackwell", "agnostic"]
    assert set(enums.ARCH_TO_HARDWARE.values()) == set(enums.HARDWARE_BUCKETS)
    with pytest.raises(ValueError):
        enums.hardware_bucket("sm_91")


def test_assessed_metadata_hardware_is_optional_and_accepts_new_buckets():
    from cuda_sft_qgen.models import AssessedMetadata

    base = {"question_type": "debug", "difficulty": 2, "bloom": "apply", "granularity": "kernel", "language": "en"}
    assert AssessedMetadata.model_validate(base).hardware is None
    assert AssessedMetadata.model_validate({**base, "hardware": "ada"}).hardware == "ada"


def test_question_hardware_derived_from_brief_arch_target():
    from cuda_sft_qgen.graph.nodes.generate_core import question_from_obj
    from cuda_sft_qgen.models import QuestionBrief

    brief = QuestionBrief(topic_path=["T08", "async_copy", "cp_async"], l3_id="cp_async", l3_claim="cp.async sizes",
                          question_type="syntax", difficulty=2, bloom="apply", persona="p", hardware="ampere",
                          granularity="kernel", language="en", arch_target="sm_89")
    obj = {"question": "On Ada sm_89, which cp.async sizes are legal?", "tests_claim": "cp.async.cg needs 16 bytes."}
    assert question_from_obj(brief, obj, source="writer").hardware == "ada"
    legacy = brief.model_copy(update={"arch_target": "any", "hardware": "hopper"})
    assert question_from_obj(legacy, obj, source="writer").hardware == "hopper"


def test_chair_prompt_assesses_arch_target_not_bucket(fake_settings):
    from cuda_sft_qgen.graph.nodes.review_chair import render_chair_prompt

    text = render_chair_prompt(fake_settings)
    assert '"arch_target": "sm_XX|any"' in text
    assert '"hardware": "agnostic|ampere|hopper|blackwell"' not in text
    assert "derived from arch_target" in text


# --- item 19: a metadata-only reject becomes a relabel ---------------------------------------------

def test_metadata_only_reject_is_relabeled_behind_accept_gates():
    reject = _review("q", "reject", fail=("metadata",), defect_codes=["metadata"])
    out = _rules(reject)
    assert (out.action, out.relabeled, out.defect_codes) == ("accept", True, [])
    assert out.reason.startswith("relabeled:")
    assert _rules(reject, verification_status="pending", verification_reason="low_solver_agreement").action == "pending"
    low = _rules({**reject, "value_scores": {"pedagogical_value": 1, "discriminativeness": 4, "realism": 4}})
    assert (low.action, low.defect_codes) == ("reject", ["low_value"])
    assert _rules(_review("q", "reject", fail=("metadata", "correctness"))).action == "reject"
    assert _rules(_review("q", "reject", fail=("metadata",), uncertain=("answerability",))).action == "reject"
    assert _rules({**reject, "assessed_metadata": None}).action == "reject"
    assert _rules({**reject, "internal_reference_answer": None}).action == "reject"
    assert _rules({**reject, "value_scores": None}).action == "reject"


def test_relabel_writes_chair_labels_and_derives_hardware(active, sample_question):
    from cuda_sft_qgen.graph.nodes.review import review_chair_node, review_score_node
    from cuda_sft_qgen.models import QualityReview

    assessed = {"question_type": "debug", "difficulty": 3, "bloom": "analyze", "hardware": "ampere",
                "arch_target": "sm_89", "granularity": "kernel", "language": "en", "answer_format": "short_answer"}
    active.review_llm = _chair_fake(lambda p: _review(p["id"], "reject", fail=("metadata",), defect_codes=["metadata"],
                                                      assessed_metadata=assessed))
    q = sample_question(id="q_relabel", question="On an Ada sm_89 GPU, why does this kernel spill registers?",
                        arch_target="sm_80", hardware="ampere", answer_format="single_choice")
    decision = review_chair_node(review_score_node({"pending": [q.model_dump()]}))["last_diversity"][
        "final_decisions"][0]
    stored = decision["question"]
    assert decision["action"] == "accept"
    assert stored["review_details"]["relabeled"] is True
    assert (stored["arch_target"], stored["hardware"], stored["answer_format"]) == ("sm_89", "ada", "short_answer")
    assert (stored["question_type"], stored["difficulty"]) == ("debug", 3)
    fields = set(QualityReview.model_fields)
    persisted = QualityReview.model_validate({k: v for k, v in stored["review_details"].items() if k in fields})
    assert persisted.action == "accept" and persisted.checks.metadata == "pass"


# --- item 20b: same-brief rewrites only for Writer-level defects -----------------------------------

@pytest.mark.parametrize("codes,queued", [
    (["correctness"], 1), (["writer_contract"], 1), (["leakage", "answerability"], 1),
    (["claim_alignment"], 0), (["hardware"], 0), (["metadata"], 0), (["correctness", "hardware"], 0),
    (["low_value"], 0), (["duplicate"], 0),
])
def test_brief_level_rejects_are_not_rewritten_from_same_brief(fake_runtime, sample_brief, sample_question,
                                                               codes, queued):
    from cuda_sft_qgen.graph.nodes.review_archive import enqueue_review_repairs
    from cuda_sft_qgen.models import SteeringBrief

    fake_runtime.settings.fake_llm = False
    task_id = fake_runtime.store.save_generation_task(sample_brief().model_dump())
    q = sample_question(id="q_rep", generation_provenance={**REAL_ORIGIN, "task_id": task_id}, defect_codes=codes)
    decision = {"action": "reject", "reason": "chair:" + ",".join(codes), "question": q.model_dump()}
    assert enqueue_review_repairs(fake_runtime, [decision], SteeringBrief()) == queued
