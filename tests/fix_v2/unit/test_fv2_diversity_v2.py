"""Diversity report v2 (task T31)."""
from fixv2_helpers import make_question, tiny_taxonomy

QUOTAS = {"question_types": {"debug": 1.0}, "difficulty": {"2": 1.0},
          "answer_format": {"code_fix": 0.5, "multi_numeric": 0.5}}


def test_joint_cells_and_overrep():
    from cuda_sft_qgen.review.diversity import build_diversity_report
    acc = [make_question(id=f"a{i}", topic_path=["T02", "smem", "x"], answer_format="code_fix", arch_target="any",
                         requested_metadata={"difficulty": 2}) for i in range(10)]
    acc += [make_question(id="b", topic_path=["T08", "tma", "y"], answer_format="multi_numeric", arch_target="sm_90",
                          hardware="hopper", requested_metadata={"difficulty": 3})]
    r = build_diversity_report(acc, [], tiny_taxonomy(), QUOTAS)
    assert r.joint_cells == 2 and abs(r.max_cell_share - 10 / 11) < 1e-9
    assert "T02.smem" in r.overrepresented
    assert r.format_hist == {"code_fix": 10, "multi_numeric": 1} and r.arch_hist["sm_90"] == 1
    assert abs(r.requested_vs_assessed_agreement - 10 / 11) < 1e-9


def test_formats_under_target_in_steering():
    from cuda_sft_qgen.review.diversity import build_diversity_report, steering_from_report
    acc = [make_question(id=f"a{i}", answer_format="code_fix") for i in range(10)]
    r = build_diversity_report(acc, [], tiny_taxonomy(), QUOTAS)
    s = steering_from_report(r, QUOTAS)
    assert "formats_under_target: multi_numeric" in s.style_notes


def test_empty_pool():
    from cuda_sft_qgen.review.diversity import build_diversity_report
    r = build_diversity_report([], [], tiny_taxonomy(), QUOTAS)
    assert r.joint_cells == 0 and r.max_cell_share == 0.0 and r.requested_vs_assessed_agreement is None
