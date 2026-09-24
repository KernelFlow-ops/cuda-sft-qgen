"""Installed prompt files equal fix/assets/prompts byte-for-byte (task T14)."""
import hashlib

import pytest
import yaml

FILES = ["writer.md", "evolver.md", "chair.md", "salvage.md", "quality_cards.yaml", "anti_patterns.yaml",
         "layers/constraints.md", "layers/context.md", "layers/cot.md", "layers/evolver_policy.md",
         "layers/quality.md", "layers/writer_policy.md", "taxonomy/facets.yaml", "taxonomy/compute_extract.md"]


@pytest.mark.parametrize("rel", FILES)
def test_installed_equals_asset(rel, repo_root, fix_dir):
    a = (fix_dir / "assets" / "prompts" / rel).read_bytes()
    b = (repo_root / "src" / "cuda_sft_qgen" / "prompts" / rel).read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), rel


def test_prompt_markers(repo_root):
    p = repo_root / "src" / "cuda_sft_qgen" / "prompts"
    assert "QUALITY_REVIEW_V2" in (p / "chair.md").read_text(encoding="utf-8")
    assert (p / "writer.md").read_text(encoding="utf-8").count("JSON contract") == 1
    cards = yaml.safe_load((p / "quality_cards.yaml").read_text(encoding="utf-8"))
    assert len(cards["answer_format"]) == 9
    assert "COMPUTE_EXTRACT_V1" in (p / "taxonomy" / "compute_extract.md").read_text(encoding="utf-8")
