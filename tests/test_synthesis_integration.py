"""Safety boundaries across synthesis, review, curation, and export."""
from copy import deepcopy
from types import SimpleNamespace

from cuda_sft_qgen.models import QualityReview
from cuda_sft_qgen.graph.nodes.review_chair import _construction_gate, _rip_select
from cuda_sft_qgen.store.archive import Archive


def _review(q):
    return dict(id=q.id, action='accept', reason='reviewed',
                checks={k: 'pass' for k in ('correctness', 'answerability', 'claim_alignment',
                    'hardware_compatibility', 'answer_leakage', 'metadata', 'duplication')},
                evidence=['fixture evidence'], internal_reference_answer='private target',
                assessed_metadata={k: getattr(q, k) for k in ('question_type', 'difficulty', 'bloom',
                    'hardware', 'granularity', 'language')},
                reviewer_model='real-fixture', reviewer_independent=True)


def test_curation_is_bucket_local(sample_question):
    questions = [sample_question(id='a'), sample_question(id='b'),
                 sample_question(id='c', topic_path=['other', 'topic', 'claim'])]
    results = {q.id: {'action': 'accept', 'review': {'verification_evidence': {
        'evidence': {'rip_score': score}}}} for q, score in zip(questions, [.8, .9, .1])}
    omitted = _rip_select([{'question': q.model_dump()} for q in questions], results,
                         SimpleNamespace(review={'rip_enabled': True, 'rip_keep_per_bucket': 1}))
    assert omitted == {'a'}  # Low-scoring topic c is not globally crowded out.


def test_decomposed_between_parents_and_hybrid_strict(sample_question):
    q = sample_question(mixup_mode='decomposed', parent_ids=['a', 'b'], parent_difficulties=[1, 3], difficulty=2)
    data = _review(q)
    data = {k: v for k, v in data.items() if k in QualityReview.model_fields}
    data['construction_checks'] = {k: 'pass' for k in ('parent_claim_alignment', 'mode_validity', 'difficulty_order', 'complexity_change')}
    assert _construction_gate(q, QualityReview.model_validate(data)) is None
    q.mixup_mode = 'hybrid'
    assert _construction_gate(q, QualityReview.model_validate(data)) == 'hybrid_difficulty_not_increased'
    data['construction_checks'].pop('complexity_change')
    assert _construction_gate(q, QualityReview.model_validate(data)) == 'construction_checks_unconfirmed'


def test_curriculum_revalidates_and_keeps_parent_family_in_one_split(tmp_path, sample_question):
    archive = Archive(tmp_path)
    try:
        parent = sample_question(id='parent')
        sibling = sample_question(id='sibling')
        child = sample_question(id='child', parent_ids=['parent', 'sibling'], mixup_mode='hybrid')
        for q in (parent, sibling, child):
            q.review_details = _review(q)
            archive.append_accepted(q)
        bad = sample_question(id='unverified')
        archive.append_accepted(bad)
        report = archive.export_training(tmp_path / 'training.jsonl')
        assert report['questions'] == 3
        manifest = archive.export_curriculum_manifest(tmp_path / 'curriculum.json')
        assert manifest['skipped_unverified'] == 1
        assert len({m['split'] for m in manifest['members']}) == 1
        assert len({m['family_id'] for m in manifest['members']}) == 1
        assert 'private target' not in (tmp_path / 'training.jsonl').read_text()
    finally:
        archive.close()


def test_verification_pending_cannot_be_exported(tmp_path, sample_question):
    archive = Archive(tmp_path)
    try:
        q = sample_question(verification_evidence={'status': 'pending', 'reason': 'unavailable'})
        q.review_details = _review(q)
        archive.append_accepted(q)
        assert archive.export_training(tmp_path / 'training.jsonl')['questions'] == 0
    finally:
        archive.close()


def test_nested_oracle_code_fences_are_restored():
    from cuda_sft_qgen.jsonutil import _restore_code_fences
    block = '```cuda\n__global__ void k(){}\n```'
    restored = _restore_code_fences({'verification_spec': {'code': '__CODE_0__'}, 'notes': ['__CODE_0__']}, [block])
    assert restored['verification_spec']['code'] == block
    assert restored['notes'] == [block]


def test_execution_disabled_is_pending_not_pass(sample_question):
    from cuda_sft_qgen.review.verifier import evaluate_candidate
    q = sample_question(draft_plan={'design': 'compile'}, verification_spec={
        'mode': 'compile', 'answer_type': 'enum', 'expected_behavior': 'compiles',
        'rubric': ['compile successfully'], 'code': '#include <cuda_runtime.h>\nint main(){return 0;}',
        'architecture': 'sm_86', 'expected_compile': 'success'})
    result = evaluate_candidate(q, object(), {'answer_consistency_enabled': True, 'execution_enabled': False})
    assert result['status'] == 'pending'
    assert result['reason'] == 'verification_execution_disabled'


def test_structured_disabled_evidence_is_not_exportable(tmp_path, sample_question):
    archive = Archive(tmp_path)
    try:
        q = sample_question(draft_plan={'design': 'structured'}, verification_evidence={'status': 'disabled'})
        q.review_details = _review(q)
        archive.append_accepted(q)
        assert archive.export_training(tmp_path / 'q.jsonl')['questions'] == 0
    finally:
        archive.close()


def test_handoff_accepts_actual_verifier_status(sample_question):
    from cuda_sft_qgen.graph.handoff import verification_from_question
    q = sample_question(verification_evidence={'status': 'pass', 'reason': 'verified', 'evidence': {'agreement': 1.0}})
    spec, evidence = verification_from_question(q)
    assert spec is None
    assert evidence.status == 'pass'


def test_runtime_negative_oracle_requires_diagnostic():
    import pytest
    from cuda_sft_qgen.models import VerificationSpec
    with pytest.raises(ValueError, match='negative runtime'):
        VerificationSpec(mode='runtime', answer_type='enum', expected_behavior='bad access',
                         rubric=['memory safety'], code='#include <cuda_runtime.h>\nint main(){}',
                         architecture='sm_86', expected_compile='success', expected_stdout='',
                         sanitizer='none', expected_runtime_outcome='sanitizer_error')


def test_run_verifier_budget_is_atomic(fake_runtime):
    from concurrent.futures import ThreadPoolExecutor
    fake_runtime.settings.review['verification_run_max_calls'] = 5
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: fake_runtime.reserve_verification_call(), range(50)))
    assert sum(results) == 5
    assert not fake_runtime.reserve_verification_call()


def test_symbolic_normalization_preserves_domain_and_bounds_work():
    import pytest
    from cuda_sft_qgen.review.verifier import canonical_numeric, canonical_symbolic
    assert canonical_symbolic('x/x') != canonical_symbolic('1')
    with pytest.raises(ValueError, match='budget'):
        canonical_numeric('((((99999999**16)**16)**16)**16)')


def test_supported_audit_quote_must_cover_numeric_claim(tmp_path):
    from unittest.mock import Mock
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker
    from cuda_sft_qgen.store.knowledge import KnowledgeStore
    url = 'https://docs.nvidia.com/cuda/cuda-programming-guide/index.html'
    store = KnowledgeStore(tmp_path / 'knowledge.sqlite3')
    store.add_claim({'l3_id': 'warp99', 'domain_id': 'T01', 'l2_id': 'warp',
                     'claim': 'A warp contains 99 threads.', 'types': ['conceptual'],
                     'difficulty': [1], 'sources': [url]})
    llm = Mock()
    llm.complete_json.return_value = {'verdict': 'supported', 'url': url, 'quote': 'A warp contains 32 threads.'}
    worker = KnowledgeWorker(store, llm, official_fetcher=lambda _: 'A warp contains 32 threads.')
    worker._audit({'claim_id': 'warp99', 'revision': 1})
    assert store.get_claim('warp99')['status'] == 'provisional'
    assert llm.complete_json.call_count == 1
