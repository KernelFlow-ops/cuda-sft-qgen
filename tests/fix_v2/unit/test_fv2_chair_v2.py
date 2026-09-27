"""Chair v2 post-processing rules as PURE functions (task T23; 01_ARCHITECTURE §7.1). No model is called here;
real reviewer behaviour is covered by llm/test_fv2_chair_calibration_llm.py."""
import pytest

CHECKS = ("correctness", "answerability", "claim_alignment", "hardware_compatibility", "answer_leakage", "metadata", "duplication")
META = {"question_type": "debug", "difficulty": 2, "bloom": "apply", "hardware": "agnostic", "granularity": "kernel", "language": "en"}


def _review(action="accept", fail=(), **extra):
    from cuda_sft_qgen.models import QualityReview
    obj = {"id": "q1", "action": action, "reason": "because",
           "checks": {k: ("fail" if k in fail else "pass") for k in CHECKS}, "evidence": ["e1"],
           "internal_reference_answer": "B" if action == "accept" else None,
           "assessed_metadata": META if action == "accept" else None}
    obj.update(extra)
    return QualityReview.model_validate(obj)


def _rules(review, **kw):
    from cuda_sft_qgen.graph.nodes.review_chair import apply_decision_rules
    args = dict(verification_status="pass", verification_reason="ok", structured=True, is_fake=False,
                construction_error=None, exploration=False, min_value_score=3)
    args.update(kw)
    return apply_decision_rules(review, **args)


def test_chair_prompt_rendered(settings):
    from cuda_sft_qgen.graph.nodes.review_chair import render_chair_prompt
    text = render_chair_prompt(settings)
    assert "{{" not in text and "QUALITY_REVIEW_V2" in text and "regs_per_sm_32bit" in text


def test_reject_derives_codes_and_prefix():
    out = _rules(_review("reject", fail=("correctness", "answer_leakage")))
    assert out.action == "reject" and out.defect_codes == ["correctness", "leakage"]
    assert out.reason.startswith("chair:correctness,leakage:")


def test_explicit_codes_kept():
    out = _rules(_review("reject", fail=("answerability",), defect_codes=["multiple_correct"]))
    assert out.reason.startswith("chair:multiple_correct:")


def test_low_value_rejected():
    out = _rules(_review("accept", value_scores={"pedagogical_value": 2, "discriminativeness": 4, "realism": 4}))
    assert out.action == "reject" and out.defect_codes == ["low_value"] and out.reason.startswith("chair:low_value:")


def test_accept_passes_with_good_scores():
    out = _rules(_review("accept", value_scores={"pedagogical_value": 4, "discriminativeness": 3, "realism": 5}))
    assert (out.action, out.defect_codes) == ("accept", [])


@pytest.mark.parametrize("status,structured,expected", [("pending", True, "pending"), ("disabled", True, "pending"),
                                                        ("disabled", False, "accept"), ("pass", True, "accept")])
def test_verification_gate(status, structured, expected):
    out = _rules(_review("accept"), verification_status=status, structured=structured)
    assert out.action == expected
    if expected == "pending":
        assert out.reason.startswith("verification:")


def test_construction_gate():
    out = _rules(_review("accept"), construction_error="construction_checks_unconfirmed")
    assert (out.action, out.reason) == ("pending", "construction_checks_unconfirmed")


def test_exploration_requires_pass():
    out = _rules(_review("accept"), verification_status="disabled", structured=False, exploration=True)
    assert (out.action, out.reason) == ("pending", "exploration_requires_verification")


def test_normalize_review_obj():
    from cuda_sft_qgen.graph.nodes.review_chair import _normalize_review_obj
    obj = _normalize_review_obj({"value_scores": {"pedagogical_value": 4.0, "realism": 3.5}, "defect_codes": ["leakage", "typo"]})
    assert obj["value_scores"] == {"pedagogical_value": 4, "realism": 3.5}
    assert obj["defect_codes"] == ["leakage"]


def test_legacy_assessed_format_normalization():
    from cuda_sft_qgen.graph.nodes.review_chair import _normalize_review_obj
    raw = {"assessed_metadata": {**META, "answer_format": "", "arch_target": "sm_unsupported"}}
    legacy = _normalize_review_obj(raw, legacy_assessment=True)["assessed_metadata"]
    assert legacy["answer_format"] is None and legacy["arch_target"] is None
    assert _normalize_review_obj(raw)["assessed_metadata"]["answer_format"] == ""
    assert _normalize_review_obj(raw)["assessed_metadata"]["arch_target"] == "sm_unsupported"


def test_node_wires_rules():
    import inspect
    from cuda_sft_qgen.graph.nodes import review_chair
    assert "apply_decision_rules" in inspect.getsource(review_chair._parallel_chair_decision)
    src = inspect.getsource(review_chair.review_chair_node)
    assert "q.defect_codes" in src and "q.value_scores" in src and "_curate_select" in src
