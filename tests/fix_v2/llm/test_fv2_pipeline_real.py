"""End-to-end REAL pipeline (task T40; invariants I3, I6, I7). Budget ≈ 80–250 requests, 20–40 min.
Pass criteria: finishes within QGEN_E2E_TIMEOUT; accepted ≥ max(3, target//5); every accepted item has assessed
metadata + value_scores + compat-valid snapshot + private-free to_jsonl; generate_error share of rejects ≤ 20%."""
import concurrent.futures
import json
import os

import pytest

pytestmark = [pytest.mark.llm, pytest.mark.slow]


def test_real_pipeline(tmp_path, eval_out):
    from cuda_sft_qgen.config import load_settings
    from cuda_sft_qgen.graph.app import run_pipeline
    from cuda_sft_qgen.store.archive import Archive
    target = int(os.getenv("QGEN_E2E_TARGET", "15"))
    timeout = float(os.getenv("QGEN_E2E_TIMEOUT", "2400"))
    settings = load_settings(target=target, data_dir=tmp_path)
    assert not settings.fake_llm
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        fut = ex.submit(run_pipeline, settings)
        result = fut.result(timeout=timeout)
    store = Archive(tmp_path)
    acc, rej, pen = store.load_accepted(), store.load_rejected(), store.load_pending()
    summary = {"result_status": (result or {}).get("status"), "accepted": len(acc), "rejected": len(rej), "pending": len(pen),
               "reject_reasons": {}}
    for q in rej:
        key = (q.reject_reason or "").split(":")[0]
        summary["reject_reasons"][key] = summary["reject_reasons"].get(key, 0) + 1
    (eval_out / "pipeline_real_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    assert len(acc) >= max(3, target // 5), summary
    compat = settings.compat
    private = ("draft_target", "draft_rationale", "premise_check", "internal_reference_answer", "draft_plan",
               "verification_spec", "defect_codes", "value_scores", "exploration")
    for q in acc:
        assert q.assessed_metadata is not None and q.value_scores, q.id
        snap = q.brief_snapshot
        assert snap.get("answer_format") in compat["type_to_formats"][snap["question_type"]], q.id
        assert not any(k in q.to_jsonl() for k in private)
    gen_err = summary["reject_reasons"].get("generate_error", 0)
    assert gen_err <= 0.2 * max(len(rej), 1), summary
    assert not any((q.reject_reason or "").startswith("curation_not_selected") for q in rej)
