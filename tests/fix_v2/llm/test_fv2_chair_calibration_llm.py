"""Real reviewer on tests/fixtures/quality_cases.json (tasks T23, T41).
Gate mode (default, 1 repeat, budget ≈ 50–60): FAR ≤ 0.10, FRR ≤ 0.15, pending(accept cases) ≤ 0.20, defect recall ≥ 0.70.
Strict mode (QGEN_CHAIR_STRICT=1, QGEN_CHAIR_REPEATS=3, used by T41): FAR ≤ 0.05, FRR ≤ 0.10, pending ≤ 0.15,
recall ≥ 0.80, stability ≥ 0.95."""
import json
import os

import pytest

pytestmark = [pytest.mark.llm, pytest.mark.slow]
EQUIV = {"answer_leakage+premise": "leakage", "answer_leakage": "leakage", "vague_glossary": "answerability",
         "unanswerable_open": "answerability", "insufficient_premises": "insufficient_premises",
         "spec_mismatch": "metadata", "hardware_compatibility": "hardware", "multiple_correct": "multiple_correct",
         "correctness": "correctness"}


def test_chair_calibration(real_runtime, repo_root, eval_out):
    from cuda_sft_qgen.graph.nodes.review_chair import _parallel_chair_decision, render_chair_prompt
    strict = os.getenv("QGEN_CHAIR_STRICT") == "1"
    repeats = int(os.getenv("QGEN_CHAIR_REPEATS", "3" if strict else "1"))
    cases = json.loads((repo_root / "tests" / "fixtures" / "quality_cases.json").read_text(encoding="utf-8"))["cases"]
    prompt = render_chair_prompt(real_runtime.settings)
    rows = []
    for repeat in range(repeats):
        for c in cases:
            out = _parallel_chair_decision({"question": c["question"], "neighbors": []}, prompt)
            rows.append({"case": c["case_id"], "repeat": repeat, "expected": c["expected_action"], "action": out["action"],
                         "reason": out["reason"][:200], "codes": (out.get("review") or {}).get("defect_codes") or [],
                         "expected_check": c.get("expected_failed_check")})
    (eval_out / "chair_calibration.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    first = [r for r in rows if r["repeat"] == 0]
    rej = [r for r in first if r["expected"] == "reject"]; acc = [r for r in first if r["expected"] == "accept"]
    far = sum(r["action"] == "accept" for r in rej) / len(rej)
    frr = sum(r["action"] == "reject" for r in acc) / len(acc)
    pend = sum(r["action"] == "pending" for r in acc) / len(acc)
    hit = [r for r in rej if r["action"] == "reject"]
    recall = sum(EQUIV.get(r["expected_check"], r["expected_check"]) in r["codes"] for r in hit) / max(len(hit), 1)
    by = {}
    for r in rows:
        by.setdefault(r["case"], set()).add(r["action"])
    stability = sum(len(v) == 1 for v in by.values()) / len(by)
    metrics = {"FAR": far, "FRR": frr, "pending": pend, "defect_recall": recall, "stability": stability,
               "repeats": repeats, "strict": strict, "independent_review": real_runtime.settings.review_model != real_runtime.settings.llm_model}
    (eval_out / "chair_calibration_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    if strict:
        assert far <= 0.05 and frr <= 0.10 and pend <= 0.15 and recall >= 0.80 and stability >= 0.95, metrics
    else:
        assert far <= 0.10 and frr <= 0.15 and pend <= 0.20 and recall >= 0.70, metrics
