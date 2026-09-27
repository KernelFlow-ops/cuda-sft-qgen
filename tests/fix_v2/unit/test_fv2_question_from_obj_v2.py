"""question_from_obj v2 parsing (task T21)."""


def _obj(**over):
    obj = {
        "question": "blockDim.x = 32, `__shared__ float s[1024]`, 读取 s[threadIdx.x * 6]，冲突路数是多少？",
        "tests_claim": "Stride-6 float access over 32 banks is 2-way conflicted.",
        "language": "zh", "answer_format": "multi_numeric", "draft_target": "2",
        "draft_rationale": "gcd(6,32)=2.", "premise_check": ["32 banks -> stem"],
        "verification_spec": {"mode": "open", "answer_type": "text", "expected_behavior": "2", "rubric": ["2-way", "gcd"]},
        "draft_plan": {"seed_analysis": {"source_seed_ids": [], "domain_id": "T02", "l3_ids": ["smem_bank_stride_gcd"],
                                         "shared_mechanism": "bank mapping", "reusable_constraints": [], "differences": []},
                       "pair_plan": {"mode": "single", "parent_ids": [], "retained_claim": "gcd", "transferred_constraint": "",
                                     "construction": "stride instance", "expected_difficulty": 2},
                       "construct": "x", "self_check": ["premise_check"]},
    }
    obj.update(over)
    return obj


def _q(obj, **brief_over):
    from cuda_sft_qgen.graph.nodes.generate_core import question_from_obj
    from fixv2_helpers import make_brief
    return question_from_obj(make_brief(**brief_over), obj, source="writer", generation_provenance={"structured_cot": True})


def test_full_contract_no_errors():
    q = _q(_obj())
    assert q.draft_validation_errors == []
    assert q.answer_format == "multi_numeric" and q.draft_rationale and q.premise_check == ["32 banks -> stem"]
    assert q.brief_snapshot["answer_format"] == "multi_numeric" and q.brief_snapshot["arch_target"] == "any"


def test_missing_private_fields_flagged():
    q = _q(_obj(draft_rationale=None, premise_check=[]))
    assert "missing_draft_rationale" in q.draft_validation_errors and "missing_premise_check" in q.draft_validation_errors


def test_answer_format_mismatch_uses_brief():
    q = _q(_obj(answer_format="single_choice"))
    assert q.answer_format == "multi_numeric" and "answer_format_mismatch" in q.warnings


def test_non_english_tests_claim_warns():
    q = _q(_obj(tests_claim="跨度 6 的访问是 2 路冲突"))
    assert "tests_claim_not_english" in q.warnings


def test_full_topic_path_domain_is_canonicalized():
    obj = _obj()
    obj["draft_plan"]["seed_analysis"]["domain_id"] = "T02.smem.smem_bank_stride_gcd"
    q = _q(obj)
    assert q.draft_plan["seed_analysis"]["domain_id"] == "T02"
    assert "seed_domain_mismatch" not in q.draft_validation_errors


def test_different_seed_domain_remains_pending():
    obj = _obj()
    obj["draft_plan"]["seed_analysis"]["domain_id"] = "T08.tma.tma_encode"
    assert "seed_domain_mismatch" in _q(obj).draft_validation_errors


def test_brief_fields_copied():
    q = _q(_obj(), scenario="reduction", arch_target="sm_80", hardware="ampere", exploration=True)
    assert (q.scenario, q.arch_target, q.exploration) == ("reduction", "sm_80", True)
