"""Solver accuracy vs difficulty on golden seeds (task T41 only; marker eval). Budget ≈ 36 × 4 requests."""
import json
from collections import defaultdict

import pytest

pytestmark = [pytest.mark.llm, pytest.mark.eval, pytest.mark.slow]


def test_solver_difficulty_curve(real_runtime, repo_root, eval_out):
    from cuda_sft_qgen.models import Question, VerificationSpec
    from cuda_sft_qgen.review.verifier import _open_rip
    reviewer = real_runtime.review_llm or real_runtime.llm
    seeds = [json.loads(l) for l in (repo_root / "src" / "cuda_sft_qgen" / "prompts" / "fewshot_seed.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_diff = defaultdict(list)
    for s in seeds:
        spec = dict(s["verification_spec"])
        if spec["mode"] != "open" or len(spec.get("rubric") or []) < 2:
            spec = {"mode": "open", "answer_type": "text", "expected_behavior": s["internal_reference_answer"],
                    "rubric": [s["internal_reference_answer"], "reasoning consistent with the stated premises"]}
        out = _open_rip(Question.model_validate({**s, "dup_verdict": "unique"}), reviewer, VerificationSpec.model_validate(spec),
                        real_runtime.settings.review)
        if out.get("status") == "pass":
            by_diff[s["difficulty"]].append(float(out["rip_score"]))
    means = {d: sum(v) / len(v) for d, v in sorted(by_diff.items()) if v}
    (eval_out / "solver_curve.json").write_text(json.dumps(means), encoding="utf-8")
    assert means.get(4, 0) >= 0.5, means
    ds = sorted(means)
    assert all(means[ds[i]] + 0.1 >= means[ds[i + 1]] for i in range(len(ds) - 1)), means
