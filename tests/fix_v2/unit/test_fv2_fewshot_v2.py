"""Few-shot v2 (task T22, invariant I5)."""
import pytest

from fixv2_helpers import make_brief, make_question


@pytest.mark.parametrize("reason,expected", [
    ("chair:correctness:bad math", True), ("chair:leakage,metadata:x", True), ("writer_leakage:claim_in_stem", True),
    ("sieve_too_vague", True), ("L0_exact", True), ("L1_template", True), ("L2_semantic", True),
    ("L2_gist_crosslingual", True), ("gray_chair", True),
    ("curation:mmr_not_selected", False), ("generate_error:Timeout", False), ("retry_exhausted:429", False),
    ("review_error:ValueError", False), ("rip_below_min", False), (None, False), ("", False)])
def test_is_quality_reject(reason, expected):
    from cuda_sft_qgen.prompts.fewshot import is_quality_reject
    assert is_quality_reject(reason) is expected


def test_negative_seeds_loaded():
    from cuda_sft_qgen.prompts.fewshot import load_negative_seeds
    negs = load_negative_seeds()
    assert len(negs) == 14
    assert all(q.reject_reason.startswith("chair:") for q in negs)
    assert all((q.review_details or {}).get("evidence") for q in negs)


def _corpus(accepted):
    from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
    return FewshotCorpus(accepted=accepted, rejected_quality=[], anti_skeletons=[])


def test_positive_slots():
    from cuda_sft_qgen.prompts.fewshot import FewshotCfg, select_positives
    brief = make_brief(l3_id="leaf_a", topic_path=["T02", "smem", "leaf_a"], kind="formula", answer_format="fill_in", language="en")
    acc = [make_question(id=f"same{i}", l3_id="leaf_a", topic_path=["T02", "smem", "leaf_a"], kind="formula", language="en") for i in range(3)]
    acc += [make_question(id=f"kind{i}", l3_id=f"leaf_k{i}", topic_path=["T02", "smem", f"leaf_k{i}"], kind="formula", language="en",
                          value_scores={"pedagogical_value": 5 - i, "discriminativeness": 4, "realism": 4}) for i in range(3)]
    acc += [make_question(id="fmt", l3_id="leaf_f", topic_path=["T02", "gmem", "leaf_f"], kind="api", answer_format="fill_in", language="en")]
    picks = select_positives(brief, _corpus(acc), FewshotCfg(positives=5))
    ids = [p.question.id for p in picks]
    assert sum(1 for i in ids if i.startswith("same")) <= 1
    assert ids.count("kind0") == 1 and "kind1" in ids            # highest value_scores first
    assert "fmt" in ids
    slot_s = [p for p in picks if p.question.id.startswith("same")]
    assert all("do NOT paraphrase" in p.note for p in slot_s)


def test_negative_format_has_reviewer_finding():
    from cuda_sft_qgen.prompts.fewshot import FewshotCfg, FewshotPick, format_negatives
    q = make_question(id="n1", reject_reason="chair:leakage:x", review_details={"evidence": ["first sentence states the rule"]})
    text = format_negatives([FewshotPick(question=q, slot="N")], FewshotCfg())
    assert "Reviewer finding: first sentence states the rule" in text


def test_code_snippets_longer():
    from cuda_sft_qgen.prompts.fewshot import FewshotCfg, FewshotPick, format_positives
    code = "```cuda\n" + ("int x = threadIdx.x;\n" * 60) + "```\nWhat is printed?"
    q = make_question(id="c", question=code)
    text = format_positives([FewshotPick(question=q, slot="K", note="n")], FewshotCfg())
    assert len(text) > 900


def test_build_accepts_corpus_argument(settings):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
    sp = StructuredPrompt(settings)
    marker = make_question(id="marker", l3_id="other_leaf", topic_path=["T02", "smem", "other_leaf"], kind="formula",
                           language="zh", question="__shared__ UNIQUE_MARKER_STEM 的 bank 冲突路数是多少？")
    b = sp.build(make_brief(), SteeringBrief(), corpus=FewshotCorpus(accepted=[marker]))
    assert "UNIQUE_MARKER_STEM" in b.system
    assert sp.corpus is not None and not sp.corpus.accepted          # shared builder state untouched
