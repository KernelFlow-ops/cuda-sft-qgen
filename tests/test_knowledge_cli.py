"""Tests for ``cuda-sft-qgen knowledge`` CLI subcommands."""

import hashlib
from unittest.mock import patch

import pytest

from cuda_sft_qgen import cli
from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.store.knowledge import KnowledgeStore


@pytest.mark.parametrize("argv", [
    ["manage-taxonomy", "--analyze", "--fake"],
    ["manage-taxonomy", "--tick", "--fake", "--max-leaves", "0"],
    ["taxonomy", "expand", "--fake", "--dry-run"],
])
def test_local_knowledge_cli_preserves_yaml_and_claims(tmp_path, capsys, argv):
    settings = load_settings(fake_llm=True, data_dir=tmp_path)
    settings.embedding_provider = "fake"
    paths = list(settings.taxonomy_path.parent.rglob("*.yaml"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    args = cli.build_parser().parse_args(argv)
    with patch("cuda_sft_qgen.cli.load_settings", return_value=settings), patch(
        "cuda_sft_qgen.llm.FakeLLM.complete_json", side_effect=AssertionError("local inspection must not call LLM")
    ):
        assert args.func(args) == 0
    output = capsys.readouterr().out
    assert output
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    stats = KnowledgeStore(tmp_path / "knowledge.sqlite3").stats()
    assert set(stats["claims"]) == {"legacy_unreviewed"}
    assert stats["jobs"] == {}
    assert stats["evidence"] == 0
