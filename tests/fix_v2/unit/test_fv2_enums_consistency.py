"""enums.py must agree with every v2 config file (both fix/assets and installed configs/)."""
import re

import pytest
import yaml

FILES = ("scope.yaml", "compat_matrix.yaml", "angles.yaml", "quotas.yaml", "personas.yaml")


def _load(base, name):
    alias = {"quotas.yaml": "quotas_v2.yaml", "personas.yaml": "personas_v2.yaml"}
    path = base / name
    if not path.exists():
        path = base / alias.get(name, name)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(params=["assets", "installed"])
def cfg(request, repo_root, fix_dir):
    base = fix_dir / "assets" / "configs" if request.param == "assets" else repo_root / "configs"
    return {n: _load(base, n) for n in FILES}


def test_enum_module_shape():
    from cuda_sft_qgen import enums
    assert set(enums.ARCH_TO_HARDWARE) == set(enums.ARCH_TARGETS)
    assert set(enums.ARCH_TO_HARDWARE.values()) <= {"agnostic", "ampere", "hopper", "blackwell"}
    assert set(enums.CHECK_TO_DEFECT.values()) <= set(enums.DEFECT_CODES)
    assert len(enums.ANSWER_FORMATS) == 9 and len(enums.OPERATORS) == 11


def test_configs_agree_with_enums(cfg):
    from cuda_sft_qgen import enums
    from cuda_sft_qgen.taxonomy import LEAF_KINDS
    scope, compat, angles, quotas, personas = (cfg[n] for n in FILES)
    assert tuple(scope["scenarios"]) == tuple(enums.SCENARIOS)
    l1_ids = {d["id"] for d in scope["domains"]}
    for d in scope["domains"]:
        for l2, ov in (d.get("l2_overrides") or {}).items():
            assert set(ov.get("arch") or []) <= set(enums.ARCH_TARGETS), (d["id"], l2)
        for n in d.get("new_l2") or []:
            assert set(n["arch"]) <= set(enums.ARCH_TARGETS), (d["id"], n["id"])
        assert d["weight"] > 0 and d["default_target"] > 0
    assert set(compat["kind_to_types"]) == set(LEAF_KINDS)
    for fmts in compat["type_to_formats"].values():
        assert set(fmts) <= set(enums.ANSWER_FORMATS)
    for scen in compat["domain_to_scenarios"].values():
        assert set(scen) <= set(enums.SCENARIOS)
    for ops in compat["kind_to_operators"].values():
        assert set(ops) <= set(enums.OPERATORS)
    for rule in compat["arch_claim_rules"]:
        re.compile(rule["match"])
        assert set(rule["allow"]) <= set(enums.ARCH_TARGETS)
    for qtype, cards in angles["angles"].items():
        assert qtype in compat["type_to_formats"]
        for card in cards:
            assert card["id"] and set(card["kinds"]) <= set(LEAF_KINDS)
            assert set(card["formats"]) <= set(enums.ANSWER_FORMATS)
            assert set(card["formats"]) & set(compat["type_to_formats"][qtype]), card["id"]
    assert set(quotas["arch_target"]) == set(enums.ARCH_TARGETS)
    assert set(quotas["answer_format"]) == set(enums.ANSWER_FORMATS)
    for pid, meta in personas["personas"].items():
        assert set(meta["domains"]) <= l1_ids, pid


def test_hardware_facts_keys(repo_root, fix_dir):
    for base in (fix_dir / "assets" / "configs", repo_root / "configs"):
        facts = yaml.safe_load((base / "hardware_facts.yaml").read_text(encoding="utf-8"))
        assert "common" in facts and "arch" in facts


def test_scope_matches_real_taxonomy(repo_root, fix_dir):
    """Every scope L1 / overridden L2 exists in the repo taxonomy; new_l2 ids do not collide."""
    from cuda_sft_qgen.taxonomy import load_taxonomy
    from cuda_sft_qgen.config import load_settings
    tax = load_taxonomy(load_settings().taxonomy_path)
    real = {d["id"]: {c["id"] for c in d.get("children", [])} for d in tax.domains}
    scope = yaml.safe_load((fix_dir / "assets" / "configs" / "scope.yaml").read_text(encoding="utf-8"))
    assert {d["id"] for d in scope["domains"]} == set(real)
    for d in scope["domains"]:
        assert set(d.get("l2_overrides") or {}) <= real[d["id"]], d["id"]
