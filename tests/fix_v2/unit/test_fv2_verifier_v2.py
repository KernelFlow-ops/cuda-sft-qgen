"""Verifier v2 pure parts (task T24; invariant I9). Real solver behaviour: llm/test_fv2_solver_curve_llm.py (T41)."""
from fixv2_helpers import make_question

CFG = {"solver_thinking_level": "low", "solver_thinking_level_hard": "medium", "solver_max_tokens": 3072,
       "answer_consistency_high_risk_k": 3, "verification_max_calls": 6}


def test_solver_params_by_difficulty():
    from cuda_sft_qgen.review.verifier import solver_call_params
    for diff, level in ((1, "low"), (2, "low"), (3, "medium"), (4, "medium")):
        p = solver_call_params(diff, CFG)
        assert p == {"thinking_level": level, "max_tokens": 3072}
    assert solver_call_params(4, {**CFG, "solver_max_tokens": 99999})["max_tokens"] == 4096


def test_open_rubric_too_small_before_any_call():
    from cuda_sft_qgen.models import VerificationSpec
    from cuda_sft_qgen.review.verifier import _open_rip
    spec = VerificationSpec.model_validate({"mode": "open", "answer_type": "text", "expected_behavior": "x", "rubric": ["only one"]})
    # reviewer=None proves the check happens before any model call
    assert _open_rip(make_question(), None, spec, CFG) == {"status": "pending", "reason": "rubric_too_small"}


def test_blind_votes_uses_solver_params():
    import inspect
    from cuda_sft_qgen.review import verifier
    assert "solver_call_params" in inspect.getsource(verifier._blind_votes)
    assert "solver_call_params" in inspect.getsource(verifier._open_rip)


def test_cjk_features():
    from cuda_sft_qgen.planner.pair_mining import cuda_difficulty_features
    from cuda_sft_qgen.review.verifier import complexity_features
    zh = "若 blockDim 为 64，仅当线程数满足条件时必须先同步再等待，否则分支发散。"
    en = "If blockDim is 64, only when the count holds must you synchronize before waiting, otherwise the branch diverges."
    f = complexity_features(zh)
    assert f["num_constraints"] >= 2 and f["dependency_depth"] >= 1 and f["control_flow_depth"] >= 1
    assert abs(cuda_difficulty_features(zh).score - cuda_difficulty_features(en).score) < 0.6
