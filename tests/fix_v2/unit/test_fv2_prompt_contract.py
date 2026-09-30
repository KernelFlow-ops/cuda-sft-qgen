"""Rendered prompts: invariants I1, I2, I4 (task T20)."""
import re

import yaml

CONFLICT = "Still return one object with"


def _build(settings, **over):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
    from fixv2_helpers import make_brief
    return StructuredPrompt(settings).build(make_brief(**over), SteeringBrief())


def test_writer_single_contract_no_placeholders(settings):
    b = _build(settings)
    assert b.system.count("JSON contract") == 1
    assert CONFLICT not in b.system
    assert not re.search(r"\{\{[a-zA-Z0-9_]+\}\}", b.system + b.user)


def test_evolver_renders_operator_and_persona(settings):
    b = _build(settings, role="evolver", seed_question="seed stem about shared memory banks", operator="profiler")
    assert "OPERATOR: profiler" in b.user
    assert "Persona you must inhabit" in b.system
    assert not re.search(r"\{\{[a-zA-Z0-9_]+\}\}", b.system + b.user)


def test_new_user_lines(settings):
    b = _build(settings, answer_format="code_fix", scenario="reduction", arch_target="sm_80")
    for line in ("ANSWER_FORMAT: code_fix", "SCENARIO: reduction", "ARCH_TARGET: sm_80"):
        assert line in b.user
    assert "ANSWER_FORMAT=code_fix" in b.system          # format_constraint + reminder


def test_instance_never_falls_back_to_claim(settings):
    b = _build(settings, instance_hint="", must_include="")
    assert "INSTANCE: (design your own scene)" in b.user
    claim = "For 4-byte words at stride S over 32 banks the conflict degree is gcd(S,32)."
    assert f"INSTANCE: {claim}" not in b.user and f"MUST_INCLUDE: {claim}" not in b.user


def test_hardware_facts_injected(settings):
    b = _build(settings, arch_target="sm_80", hardware="ampere")
    assert "max_smem_per_sm_kb: 164" in b.system


def test_quality_cards_and_taxonomy_prompt_markers(repo_root):
    prompts = repo_root / "src" / "cuda_sft_qgen" / "prompts"
    cards = yaml.safe_load((prompts / "quality_cards.yaml").read_text(encoding="utf-8"))
    assert len(cards["answer_format"]) == 9
    assert "COMPUTE_EXTRACT_V1" in (prompts / "taxonomy" / "compute_extract.md").read_text(encoding="utf-8")


def test_unrendered_placeholder_raises(settings, monkeypatch):
    import pytest
    from cuda_sft_qgen.prompts import structured_writer as sw
    from cuda_sft_qgen.models import SteeringBrief
    from fixv2_helpers import make_brief
    sp = sw.StructuredPrompt(settings)
    sp._constraints = sp._constraints + "\n{{definitely_missing_key}}"
    with pytest.raises(ValueError):
        sp.build(make_brief(), SteeringBrief())
