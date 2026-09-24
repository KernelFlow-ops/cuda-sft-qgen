"""Prompt placeholder audit (fix/01_ARCHITECTURE.md §5.2).

--assets-only : compare placeholder sets in fix/assets/prompts with the expected table (no project import).
default       : also render writer / evolver / chair / salvage with the installed package and fail on any '{{'.
Run from the repo root. Prints OK or the list of problems; exit code 0/1.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXPECTED = {
    "writer.md": {"min_chars", "max_chars"},
    "evolver.md": {"min_chars", "max_chars"},
    "chair.md": {"hardware_fact_sheet"},
    "salvage.md": {"l3_claim", "raw", "min_chars", "max_chars"},
    "layers/constraints.md": {"answer_format", "arch_target", "bloom", "difficulty", "format_constraint", "granularity",
                              "hardware", "hardware_facts", "instance_hint", "kind", "kind_constraint", "language",
                              "must_avoid", "must_avoid_stems", "question_type", "scenario", "topic_path",
                              "used_angles", "used_formats", "used_scenarios"},
    "layers/context.md": {"anti_templates", "fewshot_neg", "fewshot_pos", "steering"},
    "layers/cot.md": {"answer_format", "cot_evolver", "cot_scene", "kind", "scenario"},
    "layers/evolver_policy.md": {"angle", "l3_claim", "operator", "seed_question"},
    "layers/quality.md": {"angle", "angle_guide", "answer_format", "bloom", "bloom_card", "diff_card", "difficulty",
                          "format_card", "kind", "kind_card"},
    "layers/writer_policy.md": {"angle", "angle_card", "breadth_or_depth_guidance", "breadth_or_depth_signal",
                                "l3_claim", "strong_token_examples"},
}
PH = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


def check_assets(base: Path) -> list[str]:
    problems = []
    for rel, want in EXPECTED.items():
        path = base / rel
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        got = set(PH.findall(path.read_text(encoding="utf-8")))
        if got != want:
            problems.append(f"{rel}: extra={sorted(got - want)} missing={sorted(want - got)}")
    return problems


def check_render() -> list[str]:
    from cuda_sft_qgen.config import load_settings
    from cuda_sft_qgen.graph.nodes.review_chair import render_chair_prompt
    from cuda_sft_qgen.models import QuestionBrief, SteeringBrief
    from cuda_sft_qgen.prompts import load_prompt
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
    s = load_settings()
    sp = StructuredPrompt(s)
    base = dict(topic_path=["T02", "smem", "x"], l3_id="x", l3_claim="ways = gcd(S,32)", question_type="conceptual",
                difficulty=2, bloom="apply", persona="perf_engineer", hardware="ampere", granularity="kernel",
                language="zh", kind="formula", answer_format="multi_numeric", scenario="reduction", arch_target="sm_80")
    problems = []
    for extra in ({}, {"role": "evolver", "seed_question": "seed", "operator": "deepen"}):
        b = sp.build(QuestionBrief(**base, **extra), SteeringBrief())
        left = PH.findall(b.system + b.user)
        if left:
            problems.append(f"writer/evolver render leftover {sorted(set(left))} ({extra.get('role', 'writer')})")
    if PH.findall(render_chair_prompt(s)):
        problems.append("chair render leftover")
    salvage = (load_prompt("salvage").replace("{{l3_claim}}", "c").replace("{{raw}}", "r")
               .replace("{{min_chars}}", "1").replace("{{max_chars}}", "2"))
    if PH.findall(salvage):
        problems.append("salvage render leftover")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-only", action="store_true")
    a = ap.parse_args()
    root = Path.cwd()
    base = root / "cuda_qgen_fix" / "assets" / "prompts" if a.assets_only else root / "src" / "cuda_sft_qgen" / "prompts"
    problems = check_assets(base)
    if not a.assets_only:
        try:
            problems += check_render()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"render failed: {type(exc).__name__}: {exc}")
    print("OK" if not problems else "\n".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
