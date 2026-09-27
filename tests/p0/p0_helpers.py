"""Shared builders for tests/p0 (imported as ``p0_helpers``; pytest puts this dir on sys.path)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_cfg(name: str) -> Any:
    """Parse one of the repository's configs/*.yaml files."""
    return yaml.safe_load((REPO_ROOT / "configs" / name).read_text(encoding="utf-8"))


def planner_cfgs() -> dict[str, Any]:
    """Real compat/scope/quotas/personas/angles used by joint planning."""
    from cuda_sft_qgen.planner.sampling import load_angles

    return {"compat": load_cfg("compat_matrix.yaml"), "scope": load_cfg("scope.yaml"),
            "quotas": load_cfg("quotas.yaml"), "personas": load_cfg("personas.yaml")["personas"],
            "angles": load_angles(REPO_ROOT / "configs" / "angles.yaml")}


def leaf_item(leaf_id: str, kind: str, types: list[str], claim: str, *, status: str = "verified",
              difficulty: tuple[int, ...] = (2, 3), **extra: Any) -> dict[str, Any]:
    return {"id": leaf_id, "claim": claim, "kind": kind, "types": list(types), "difficulty": list(difficulty),
            "hints": [], "status": status, **extra}


def taxonomy(t02_leaves: list[dict], t08_leaves: list[dict] | None = None):
    """Two-domain in-memory taxonomy (T02 smem with arch any, T08 tma on sm_90 family)."""
    from cuda_sft_qgen.taxonomy import Taxonomy, _leaves_from_domains

    domains = [{"id": "T02", "name": "mem", "weight": 1.0, "children": [
        {"id": "smem", "name": "smem", "arch": ["any"], "leaves": list(t02_leaves)}]}]
    if t08_leaves:
        domains.append({"id": "T08", "name": "async", "weight": 1.0, "children": [
            {"id": "tma", "name": "tma", "arch": ["sm_90", "sm_90a", "sm_100a"], "leaves": list(t08_leaves)}]})
    return Taxonomy(1, "p0", _leaves_from_domains(domains), domains)


def make_question(**over: Any):
    """A minimal valid Question on the p0 taxonomy."""
    from cuda_sft_qgen.models import Question

    base = dict(id="q_p0", question="Why does a stride-12 shared-memory read conflict 4 ways on 32 banks?",
                tests_claim="Stride 12 over 32 banks gives gcd(12, 32) = 4-way conflicts.",
                topic_path=["T02", "smem", "smem_gcd"], l3_id="smem_gcd",
                l3_claim="ways = gcd(S, 32) for 4-byte words", question_type="conceptual", difficulty=2,
                bloom="apply", persona="perf_engineer", hardware="agnostic", granularity="kernel", language="en",
                kind="formula")
    base.update(over)
    return Question.model_validate(base)
