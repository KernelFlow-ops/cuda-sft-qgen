"""CLI production question generation requires real API credentials."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("command", ["generate", "resume", "benchmark"])
def test_fake_generation_switch_is_rejected(command, temp_data_dir):
    args = [sys.executable, "-m", "cuda_sft_qgen", command, "--fake"]
    if command == "benchmark":
        args += ["--output", str(temp_data_dir / "bench")]
    result = subprocess.run(
        args,
        env={**os.environ, "CUDA_SFT_DATA_DIR": str(temp_data_dir)},
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert not (temp_data_dir / "questions.sqlite3").exists()


def test_stats_can_read_an_empty_archive_without_generating(temp_data_dir):
    result = subprocess.run(
        [sys.executable, "-m", "cuda_sft_qgen", "stats"],
        env={**os.environ, "CUDA_SFT_DATA_DIR": str(temp_data_dir)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["accepted_count"] == 0


@pytest.mark.parametrize("done, expected_code", [(True, 0), (False, 1)])
def test_generate_exit_code_reflects_target_completion(monkeypatch, capsys, done, expected_code):
    from cuda_sft_qgen.cli import generate as generate_cmd

    monkeypatch.setattr(generate_cmd, "load_settings", lambda **kwargs: object())
    monkeypatch.setattr(generate_cmd, "_require_keys", lambda settings: None)
    monkeypatch.setattr(generate_cmd, "run_pipeline", lambda *args, **kwargs: {
        "accepted_count": 20 if done else 3,
        "rejected_count": 0,
        "iteration": 1,
        "status": "archived" if done else "review_pending",
        "done": done,
    })
    args = SimpleNamespace(target=20, batch=4, data_dir=None, rebuild_taxonomy=False)

    assert generate_cmd.cmd_generate(args) == expected_code
    assert json.loads(capsys.readouterr().out)["done"] is done
