"""Production generation and export never certify synthetic Writer output."""

from __future__ import annotations

import pytest

from cuda_sft_qgen.benchmark import run_benchmark
from cuda_sft_qgen.cli.parser import build_parser
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.graph.nodes.review_chair import _parallel_chair_decision
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.runtime import reset_runtime, set_runtime
from cuda_sft_qgen.store.archive import Archive


def test_pipeline_rejects_fake_before_creating_archive(fake_settings):
    with pytest.raises(RuntimeError, match="real LLM API"):
        run_pipeline(fake_settings)
    assert not (fake_settings.data_dir / "questions.sqlite3").exists()


def test_fake_llm_has_no_question_writer_response():
    with pytest.raises(RuntimeError, match="cannot generate questions"):
        FakeLLM().complete([
            {"role": "system", "content": "WRITER_PROMPT_V3"},
            {"role": "user", "content": "CLAIM: example"},
        ])


def test_injected_fake_client_cannot_bypass_real_settings(fake_runtime):
    fake_runtime.settings.fake_llm = False
    with pytest.raises(RuntimeError, match="real LLM API"):
        run_pipeline(fake_runtime.settings, runtime=fake_runtime)


def test_generation_benchmark_rejects_fake(fake_settings, tmp_path):
    output = tmp_path / "benchmark"
    with pytest.raises(RuntimeError, match="real LLM API"):
        run_benchmark(fake_settings, output, target=1)
    assert not output.exists()


@pytest.mark.parametrize("command", ["generate", "resume", "benchmark"])
def test_cli_has_no_fake_generation_switch(command):
    argv = [command, "--fake"]
    if command == "benchmark":
        argv += ["--output", "/tmp/unused-benchmark"]
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(argv)
    assert error.value.code == 2


def test_real_review_defers_unknown_writer_origin(fake_runtime, sample_question):
    fake_runtime.settings.fake_llm = False
    token = set_runtime(fake_runtime)
    try:
        result = _parallel_chair_decision({"question": sample_question().model_dump()}, "unused")
    finally:
        reset_runtime(token)
    assert result["action"] == "pending"
    assert result["reason"] == "unverified_writer_origin"


def test_legacy_accepted_does_not_satisfy_real_target(tmp_path, sample_question):
    archive = Archive(tmp_path)
    archive.append_accepted(sample_question(id="legacy"))
    assert archive.status_counts()["accepted"] == 1
    assert archive.status_counts()["verified_accepted"] == 0
    archive.append_accepted(sample_question(id="api", generation_provenance={
        "writer_real_api": True, "writer_provider": "openai", "writer_model": "test-model",
    }))
    assert archive.status_counts()["verified_accepted"] == 1
