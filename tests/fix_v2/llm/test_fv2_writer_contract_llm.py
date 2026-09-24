"""Real writer on 8 diverse joint-planned briefs (tasks T20–T22, T25–T26). Budget ≈ 8–24 requests.
Thresholds (do not lower without A0 approval): parsed ≥ 7/8; contract complete ≥ 6/8; English tests_claim ≥ 6/8;
passes local writer gate ≥ 5/8; passes rule sieve ≥ 6/8; answer_format always equals the brief."""
import json
import random
import re

import pytest

pytestmark = [pytest.mark.llm, pytest.mark.slow]


def test_writer_contract_real(real_runtime, eval_out):
    from cuda_sft_qgen.errors import JsonSalvageNeeded
    from cuda_sft_qgen.graph.nodes.generate_core import generate_from_brief
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.planner.sampling import load_angles, plan_briefs
    from cuda_sft_qgen.review.writer_gate import local_writer_defects
    from cuda_sft_qgen.sieve import rule_sieve
    rt, s = real_runtime, real_runtime.settings
    briefs = plan_briefs(n=8, taxonomy=rt.taxonomy, quotas=s.quotas, personas=s.personas, accepted=[], pending=[],
                         steering=SteeringBrief(), evolver_fraction=0.0, seed_pool=[], rng=random.Random(20260924),
                         angles=load_angles(s.angles_path), compat=s.compat, scope=s.scope, joint=True,
                         knowledge=rt.knowledge)
    rows, stats = [], {"parsed": 0, "contract": 0, "english": 0, "gate_ok": 0, "sieve_ok": 0, "format_ok": 0}
    for b in briefs:
        row = {"brief": {k: getattr(b, k) for k in ("l3_id", "question_type", "answer_format", "arch_target", "scenario", "difficulty", "language")}}
        try:
            q = generate_from_brief(b, SteeringBrief())
        except JsonSalvageNeeded as exc:
            row["error"] = "json_salvage_needed"; rows.append(row); continue
        stats["parsed"] += 1
        stats["contract"] += not q.draft_validation_errors
        stats["english"] += not re.search(r"[\u4e00-\u9fff]", q.tests_claim or "")
        gate = local_writer_defects(q); sieve = rule_sieve(q, s)
        stats["gate_ok"] += gate is None; stats["sieve_ok"] += sieve is None
        stats["format_ok"] += q.answer_format == b.answer_format
        row.update(question=q.question, tests_claim=q.tests_claim, errors=q.draft_validation_errors, gate=gate, sieve=sieve,
                   warnings=q.warnings)
        rows.append(row)
    (eval_out / "writer_contract.json").write_text(json.dumps({"stats": stats, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    assert stats["parsed"] >= 7, stats
    assert stats["contract"] >= 6, stats
    assert stats["english"] >= 6, stats
    assert stats["gate_ok"] >= 5, stats
    assert stats["sieve_ok"] >= 6, stats
    assert stats["format_ok"] == stats["parsed"], stats
