"""hardware_facts module (01_ARCHITECTURE §9, task T12)."""


def _facts(repo_root):
    from cuda_sft_qgen import hardware_facts
    return hardware_facts, hardware_facts.load(str(repo_root / "configs" / "hardware_facts.yaml"))


def test_any_renders_common_only(repo_root):
    hf, facts = _facts(repo_root)
    text = hf.render("any", facts)
    assert "regs_per_sm_32bit" in text and "sm_80" not in text
    assert all(line.startswith("  ") for line in text.splitlines() if line)


def test_verified_arch_rendered(repo_root):
    hf, facts = _facts(repo_root)
    text = hf.render("sm_80", facts)
    assert "max_smem_per_sm_kb: 164" in text and "verify_before_use" not in text


def test_sm90a_falls_back_to_sm90(repo_root):
    hf, facts = _facts(repo_root)
    assert "max_smem_per_sm_kb: 228" in hf.render("sm_90a", facts)


def test_unverified_arch_not_rendered(repo_root):
    hf, facts = _facts(repo_root)
    assert "sm_100" not in hf.verified_arches(facts)
    text = hf.render("sm_100a", facts)
    assert "not verified" in text and "tmem" not in text


def test_render_all_verified(repo_root):
    hf, facts = _facts(repo_root)
    text = hf.render_all_verified(facts)
    for arch in hf.verified_arches(facts):
        assert arch in text
    assert "sm_120" not in text


def test_missing_file_is_empty(tmp_path):
    from cuda_sft_qgen import hardware_facts
    facts = hardware_facts.load(str(tmp_path / "nope.yaml"))
    assert facts == {"common": {}, "arch": {}}
