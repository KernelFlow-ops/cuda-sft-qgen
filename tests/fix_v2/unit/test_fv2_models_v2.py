"""Question / QuestionBrief / QualityReview v2 contracts (01_ARCHITECTURE §4, invariants I6, I12)."""
import pytest
from pydantic import ValidationError

from fixv2_helpers import make_brief, make_question

LEGACY = {"id": "old", "question": "q", "tests_claim": "t", "topic_path": ["a", "b", "c"], "question_type": "debug",
          "difficulty": 2, "bloom": "apply", "persona": "p", "hardware": "agnostic", "granularity": "kernel", "language": "zh"}


def test_defaults_and_legacy_readable():
    from cuda_sft_qgen.models import Question, QuestionBrief
    q = Question.model_validate(LEGACY)
    assert (q.answer_format, q.scenario, q.arch_target) == ("", "generic", "any")
    assert q.premise_check == [] and q.defect_codes == [] and q.value_scores == {} and q.exploration is False
    b = QuestionBrief.model_validate({"topic_path": ["a"], "l3_id": "x", "l3_claim": "c", "question_type": "debug",
                                      "difficulty": 2, "bloom": "apply", "persona": "p", "hardware": "agnostic",
                                      "granularity": "kernel", "language": "zh"})
    assert b.operator is None and b.used_angles == [] and b.exploration is False


@pytest.mark.parametrize("field,value", [("answer_format", "essay"), ("scenario", "space"),
                                         ("arch_target", "sm_91"), ("operator", "rewrite")])
def test_brief_rejects_unknown_enums(field, value):
    with pytest.raises(ValidationError):
        make_brief(**{field: value})


@pytest.mark.parametrize("field,value", [("answer_format", "essay"), ("scenario", "space"), ("arch_target", "sm_91")])
def test_question_rejects_unknown_enums(field, value):
    with pytest.raises(ValidationError):
        make_question(**{field: value})


def test_to_jsonl_strips_private_fields():
    q = make_question(draft_target="B", draft_rationale="r", premise_check=["p"], defect_codes=["leakage"],
                      value_scores={"pedagogical_value": 4, "discriminativeness": 4, "realism": 4}, exploration=True,
                      internal_reference_answer="B", answer_format="single_choice", scenario="reduction", arch_target="sm_80")
    rec = q.to_jsonl()
    for key in ("draft_target", "draft_rationale", "premise_check", "defect_codes", "value_scores", "exploration",
                "internal_reference_answer", "draft_plan", "verification_spec"):
        assert key not in rec, key
    assert rec["answer_format"] == "single_choice" and rec["scenario"] == "reduction" and rec["arch_target"] == "sm_80"


def _review(**over):
    base = {"id": "q", "action": "reject", "reason": "wrong",
            "checks": {k: "pass" for k in ("correctness", "answerability", "claim_alignment", "hardware_compatibility",
                                            "answer_leakage", "metadata", "duplication")},
            "evidence": ["e"], "internal_reference_answer": None, "assessed_metadata": None}
    base["checks"]["correctness"] = "fail"
    base.update(over)
    return base


def test_quality_review_v2_fields():
    from cuda_sft_qgen.models import QualityReview
    r = QualityReview.model_validate(_review(defect_codes=["correctness"],
                                             value_scores={"pedagogical_value": 3, "discriminativeness": 2, "realism": 5}))
    assert r.defect_codes == ["correctness"]
    with pytest.raises(ValidationError):
        QualityReview.model_validate(_review(defect_codes=["typo"]))
    with pytest.raises(ValidationError):
        QualityReview.model_validate(_review(value_scores={"pedagogical_value": 3}))
    with pytest.raises(ValidationError):
        QualityReview.model_validate(_review(value_scores={"pedagogical_value": 6, "discriminativeness": 2, "realism": 5}))
    with pytest.raises(ValidationError):   # strict: floats are not coerced (T23 normalizes before validation)
        QualityReview.model_validate(_review(value_scores={"pedagogical_value": 3.0, "discriminativeness": 2, "realism": 5}))
    assert QualityReview.model_validate(_review()).value_scores is None   # v1 review still valid


def test_assessed_metadata_optional_v2():
    from cuda_sft_qgen.models import AssessedMetadata
    base = {"question_type": "debug", "difficulty": 2, "bloom": "apply", "hardware": "agnostic",
            "granularity": "kernel", "language": "zh"}
    assert AssessedMetadata.model_validate(base).arch_target is None
    assert AssessedMetadata.model_validate({**base, "arch_target": "sm_90a", "answer_format": "fill_in"}).arch_target == "sm_90a"
    with pytest.raises(ValidationError):
        AssessedMetadata.model_validate({**base, "arch_target": "sm_91"})


def test_diversity_report_new_fields():
    from cuda_sft_qgen.models import DiversityReport
    r = DiversityReport(accepted_count=0, pending_count=0)
    assert r.format_hist == {} and r.joint_cells == 0 and r.max_cell_share == 0.0 and r.requested_vs_assessed_agreement is None
