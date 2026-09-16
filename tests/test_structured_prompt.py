from __future__ import annotations

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.models import QuestionBrief, SteeringBrief
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt


def _brief(**kwargs) -> QuestionBrief:
    data = dict(
        topic_path=["T25", "occupancy", "occupancy_reg"],
        l3_id="occupancy_reg",
        l3_claim="occupancy = min(reg, smem, warp) limits",
        question_type="conceptual",
        difficulty=2,
        bloom="apply",
        persona="cuda_interviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
        angle="公式代入：给出参数，求出占用/事务数/offset/bank/字节",
        kind="formula",
        instance_hint="regs=64 block=256",
        must_include="regs=64",
    )
    data.update(kwargs)
    return QuestionBrief.model_validate(data)


def test_layers_and_json_schema_for_formula():
    settings = load_settings(fake_llm=True)
    sp = StructuredPrompt(settings, FewshotCorpus())
    bundle = sp.build(_brief(), SteeringBrief())
    assert "Hard constraints" in bundle.system or "Python len()" in bundle.system
    assert "公式" in bundle.system or "formula" in bundle.system.lower()
    assert "tests_claim" in bundle.system
    assert '"operator"' not in bundle.system
    assert "CLAIM:" in bundle.user
    assert "TYPE:" in bundle.user
    assert "HARDWARE:" in bundle.user
    assert "KIND: formula" in bundle.user
    assert "Write the JSON now." in bundle.user
    assert "reason privately" in bundle.system.lower() or "Do NOT write the reasoning" in bundle.system


def test_feature_kind_allows_arch_constraint_not_glossary_ban():
    settings = load_settings(fake_llm=True)
    sp = StructuredPrompt(settings, FewshotCorpus())
    bundle = sp.build(
        _brief(kind="feature", angle="特性边界：某代硬件该功能的合法范围与硬约束（禁止宣传腔）"),
        SteeringBrief(),
    )
    assert "KIND=feature" in bundle.system or "kind=feature" in bundle.system.lower() or "特性" in bundle.system
    assert "禁止任何介绍性内容" not in bundle.system


def test_prompt_cot_can_be_disabled():
    settings = load_settings(fake_llm=True)
    settings.default_yaml.setdefault("generation", {})["prompt_cot"] = False
    sp = StructuredPrompt(settings, FewshotCorpus())
    bundle = sp.build(_brief(), SteeringBrief())
    assert "reason privately" not in bundle.system.lower()


def test_empty_corpus_still_has_anti_patterns():
    settings = load_settings(fake_llm=True)
    sp = StructuredPrompt(settings, FewshotCorpus())
    bundle = sp.build(_brief(), SteeringBrief())
    assert "What is CUDA" in bundle.system or "名词解释" in bundle.system
    assert "(none yet)" in bundle.system


def test_angle_guide_is_not_duplicated_in_semi_prefix():
    settings = load_settings(fake_llm=True)
    sp = StructuredPrompt(settings, FewshotCorpus())
    guide = (
        "Other angles (contrast only, do not mix in):\n"
        "- 哪条能编译：给 2～3 个片段，指出合法/非法"
    )
    bundle = sp.build(_brief(angle_guide=guide), SteeringBrief())
    assert bundle.system.count("Other valid angles") == 1
    assert bundle.system.count("哪条能编译") == 1
