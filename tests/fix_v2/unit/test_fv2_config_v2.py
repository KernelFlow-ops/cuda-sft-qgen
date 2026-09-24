"""config.py v2 defaults, env overrides, review-model warning (01_ARCHITECTURE §13)."""
import logging


def test_new_defaults():
    from cuda_sft_qgen.config import load_settings
    s = load_settings()
    assert s.generation["prompt_cot"] is True
    assert s.review["answer_consistency_enabled"] is True
    assert s.review["solver_thinking_level"] == "low"
    assert s.review["solver_thinking_level_hard"] == "medium"
    assert int(s.review["solver_max_tokens"]) == 3072
    assert s.planner_cfg.get("joint_sampling") is True
    assert s.compat.get("kind_to_types") and s.scope.get("domains")
    for attr in ("scope_path", "compat_matrix_path", "angles_path", "hardware_facts_path"):
        assert getattr(s, attr).exists(), attr
    k = s.knowledge_cfg
    assert k["docs_dir"] and "toolkit_version" in k and int(k["legacy_batch_per_l2"]) == 20


def test_env_overrides(monkeypatch):
    from cuda_sft_qgen.config import load_settings
    monkeypatch.setenv("STRUCTURED_COT", "0")
    monkeypatch.setenv("ANSWER_CONSISTENCY_ENABLED", "0")
    monkeypatch.setenv("PLANNER_JOINT_SAMPLING", "0")
    monkeypatch.setenv("SOLVER_THINKING_LEVEL_HARD", "high")
    s = load_settings()
    assert s.generation["prompt_cot"] is False
    assert s.review["answer_consistency_enabled"] is False
    assert s.planner_cfg["joint_sampling"] is False
    assert s.review["solver_thinking_level_hard"] == "high"


def test_review_model_equal_warns(monkeypatch, caplog):
    from cuda_sft_qgen.config import load_settings
    monkeypatch.delenv("REVIEW_MODEL", raising=False)
    monkeypatch.setenv("NVIDIA_MODEL", "same/model")
    with caplog.at_level(logging.WARNING):
        s = load_settings()
    if s.review_model == s.nvidia_model:
        assert any("review_model_equals_writer" in r.getMessage() for r in caplog.records)


def test_profiles_do_not_disable_v2(repo_root):
    import yaml
    for name in ("dev", "prod", "test"):
        prof = yaml.safe_load((repo_root / "configs" / "profiles" / f"{name}.yaml").read_text(encoding="utf-8")) or {}
        assert (prof.get("generation") or {}).get("prompt_cot") is not False, name
        assert (prof.get("review") or {}).get("answer_consistency_enabled") is not False, name
