"""P0 prompt regressions: brace-safe single-pass rendering, anti-template order,
angle card, and the mixup parent-id contract."""
from __future__ import annotations

import json

import pytest

from p0_helpers import make_question

BRACES = ("{{", "}}", "{{N}, {N}}", "{{x}}", "{{kind}}")


@pytest.fixture
def settings(monkeypatch, tmp_path):
    from cuda_sft_qgen import config

    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    return config.load_settings(fake_llm=True, data_dir=tmp_path)


def _brief(**over):
    from cuda_sft_qgen.models import QuestionBrief

    base = dict(topic_path=["T02", "smem", "smem_gcd"], l3_id="smem_gcd", l3_claim="ways = gcd(S, 32) for 4B words",
                question_type="conceptual", difficulty=2, bloom="apply", persona="cuda_interviewer",
                hardware="agnostic", granularity="kernel", language="en", kind="formula",
                angle="plug in numbers", angle_id="con_plug_numbers", answer_format="multi_numeric")
    base.update(over)
    return QuestionBrief.model_validate(base)


# --- item 2: single-pass renderer --------------------------------------------------------------

def test_render_is_single_pass_and_ignores_value_braces():
    from cuda_sft_qgen.prompts.render import placeholders, render

    template = "A={{a}} B={{b}} literal {{N}, {N}}"
    assert placeholders(template) == {"a", "b"}
    assert render(template, {"a": "{{b}}", "b": "{{N}, {N}} }} {{"}) == \
        "A={{b}} B={{N}, {N}} }} {{ literal {{N}, {N}}"
    assert render("keep {{missing}}", {}) == "keep {{missing}}"


def test_build_survives_braces_in_every_injected_field(settings):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
    from cuda_sft_qgen.review.fingerprint import skeletonize

    code_skeleton = skeletonize("launch kernel<<<grid{128,1,1}, block{256,1,1}>>> and read tile{32,33}")
    assert "{{" in code_skeleton                        # the real skeleton shape that used to crash
    noisy = " ".join(BRACES)
    positive = make_question(id="pos", l3_id="smem_gcd", question=f"Positive stem with {noisy} braces?")
    negative = make_question(id="neg", l3_id="smem_gcd", question=f"Negative stem {noisy}?",
                             reject_reason="L1_template")
    corpus = FewshotCorpus(accepted=[positive], rejected_quality=[negative], anti_skeletons=[code_skeleton])
    brief = _brief(role="evolver", seed_question=f"Seed stem {noisy}", instance_hint=f"hint {noisy}",
                   must_include=f"must {noisy}", anti_templates=list(BRACES),
                   knowledge_context={"claims": [{"id": "smem_gcd", "claim": f"claim {noisy}"}],
                                      "evidence": [{"quote": f"quote {noisy}"}]})
    steering = SteeringBrief(forbidden_templates=list(BRACES) + [code_skeleton])
    bundle = StructuredPrompt(settings, corpus).build(brief, steering)
    assert f"hint {noisy}" in bundle.system and f"hint {noisy}" in bundle.user
    assert f"Seed stem {noisy}" in bundle.system and f"Seed stem {noisy}" in bundle.user
    assert f"Positive stem with {noisy} braces?" in bundle.system
    assert f"Negative stem {noisy}?" in bundle.system
    assert code_skeleton[:200] in bundle.system        # anti-template block and steering block
    assert "{{x}}" in bundle.system and "{{kind}}" in bundle.system
    assert json.dumps(f"quote {noisy}", ensure_ascii=False) in bundle.user


def test_undeclared_template_placeholder_still_raises(settings):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt

    prompt = StructuredPrompt(settings, FewshotCorpus())
    prompt._constraints += "\n{{not_a_mapped_key}}"
    with pytest.raises(ValueError, match="not_a_mapped_key"):
        prompt.build(_brief(), SteeringBrief())


# --- item 6: dynamic skeletons win the anti-template cap -----------------------------------------

def test_dynamic_skeletons_come_before_static_patterns(settings):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.prompts.fewshot import FewshotCfg, FewshotCorpus, select_anti
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt

    brief = _brief(anti_templates=[f"brief-skel-{i}" for i in range(3)])
    corpus = FewshotCorpus(anti_skeletons=[f"reject-skel-{i}" for i in range(5)])
    prompt = StructuredPrompt(settings, corpus)
    assert len(prompt.static_anti) > 15
    items = select_anti(brief, corpus, FewshotCfg(anti_max=15), prompt.static_anti)
    assert len(items) == 15
    assert items[:8] == [f"brief-skel-{i}" for i in range(3)] + [f"reject-skel-{i}" for i in range(5)]
    assert items[8:] == prompt.static_anti[:7]
    forbidden_block = prompt.build(brief, SteeringBrief()).system.split("## Forbidden skeletons")[1]
    assert all(f"reject-skel-{i}" in forbidden_block for i in range(5))


# --- item 8: the angle card is rendered from the brief's real fields ------------------------------

def test_angle_card_renders_must_have_and_must_not(settings):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt

    guide = ("Must have:\n- concrete substitutable numbers\nMust not:\n- a glossary definition\n"
             "Other angles (contrast only, do not mix in):\n- why the constraint exists")
    prompt = StructuredPrompt(settings, FewshotCorpus())
    brief = _brief(angle_guide=guide)
    card = prompt._angle_card_block(brief)
    assert card.startswith("Angle card: con_plug_numbers")
    assert "- concrete substitutable numbers" in card and "- a glossary definition" in card
    assert "Other angles" not in card and "why the constraint exists" not in card
    system = prompt.build(brief, SteeringBrief()).system
    angle_lock = system.split("## Angle lock")[1].split("## Leaf lock")[0]
    assert "- concrete substitutable numbers" in angle_lock and "- a glossary definition" in angle_lock
    assert system.count("Other angles (contrast only, do not mix in)") == 1


# --- item 9: mixup parent ids reach the Writer and close the contract ----------------------------

def test_mixup_prompt_carries_parent_ids_and_contract_passes(settings):
    from cuda_sft_qgen.graph.nodes.generate_core import question_from_obj
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
    from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt

    mixup = {"mode": "hybrid", "parent_ids": ["low", "high"], "parent_difficulties": [2, 4]}
    brief = _brief(role="evolver", seed_question="PARENT A:\nlow\n\nPARENT B:\nhigh", mixup_mode="hybrid",
                   parent_ids=["low", "high"], parent_difficulties=[2, 4], knowledge_context={"mixup": mixup})
    bundle = StructuredPrompt(settings, FewshotCorpus()).build(brief, SteeringBrief())
    assert "MIXUP_MODE: hybrid" in bundle.user
    assert 'PARENT_IDS: ["low", "high"]' in bundle.user
    assert "OPERATOR:" not in bundle.user
    assert "pair_plan.parent_ids to exactly PARENT_IDS" in bundle.system
    obj = {
        "question": "Combine both parents: which bank conflict degree results for stride 12 on 32 banks?",
        "tests_claim": "Stride 12 over 32 banks gives a 4-way conflict.", "draft_target": "4",
        "draft_rationale": "gcd", "premise_check": ["stride 12", "32 banks"],
        "draft_plan": {"seed_analysis": {"domain_id": "T02", "shared_mechanism": "bank mapping"},
                       "pair_plan": {"mode": "hybrid", "parent_ids": json.loads(bundle.user.split("PARENT_IDS: ")[1]
                                                                                .splitlines()[0]),
                                     "retained_claim": "gcd", "construction": "merge", "expected_difficulty": 4}},
        "verification_spec": {"mode": "scalar", "answer_type": "number", "expected_behavior": "4",
                              "rubric": ["gcd(12, 32) = 4"]},
    }
    question = question_from_obj(brief, obj, source="evolver")
    assert "mixup_plan_parent_mismatch" not in question.draft_validation_errors
    assert question.draft_validation_errors == []
