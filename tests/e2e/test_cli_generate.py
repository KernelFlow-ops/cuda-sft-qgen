"""End-to-end CLI generate smoke tests using FakeLLM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_generate_basic(temp_data_dir):
    """Test CLI generate command produces output files."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "generate",
            "--fake",
            "--target",
            "5",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Check output
    output = json.loads(result.stdout)
    assert "accepted_count" in output
    assert "status" in output
    assert output["done"] is True

    # Check files were created
    assert (temp_data_dir / "accepted.jsonl").exists()
    assert (temp_data_dir / "stats.json").exists()


def test_cli_generate_with_batch(temp_data_dir):
    """Test CLI generate with custom batch size."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "generate",
            "--fake",
            "--target",
            "10",
            "--batch",
            "5",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["accepted_count"] >= 0


def test_cli_stats(temp_data_dir):
    """Test CLI stats command."""
    # First generate some data
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "generate",
            "--fake",
            "--target",
            "3",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
    )

    # Then run stats
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "stats",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    stats = json.loads(result.stdout)
    assert "accepted_count" in stats


def test_cli_generate_output_structure(temp_data_dir):
    """Test CLI generate produces correctly structured output."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "generate",
            "--fake",
            "--target",
            "3",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    # Check accepted.jsonl structure
    accepted_file = temp_data_dir / "accepted.jsonl"
    assert accepted_file.exists()

    with open(accepted_file, "r") as f:
        lines = f.readlines()
        if lines:
            question = json.loads(lines[0])
            assert "id" in question
            assert "question" in question
            assert "l3_id" in question
            assert "question_type" in question


def test_cli_generate_verbose(temp_data_dir):
    """Test CLI generate with verbose flag."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "generate",
            "--fake",
            "--target",
            "3",
            "--verbose",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # Verbose should produce more stderr output
    assert len(result.stderr) > 0


def test_cli_generate_creates_directories(tmp_path):
    """Test CLI generate creates data directory if needed."""
    new_data_dir = tmp_path / "new_data_dir"
    assert not new_data_dir.exists()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "generate",
            "--fake",
            "--target",
            "2",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(new_data_dir),
        },
        capture_output=True,
        text=True,
    )

    # Should create directory and succeed
    assert result.returncode == 0
    assert new_data_dir.exists()
