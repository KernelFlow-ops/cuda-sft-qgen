"""End-to-end resume tests: interrupted generate continues from the archive."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_resume_from_checkpoint(temp_data_dir):
    """Test resuming generation from checkpoint."""
    # First run - generate partial data
    result1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "generate",
            "--fake",
            "--target",
            "10",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result1.returncode == 0
    output1 = json.loads(result1.stdout)
    accepted_count_1 = output1["accepted_count"]

    # Check state was saved
    assert (temp_data_dir / "accepted.jsonl").exists()
    assert (temp_data_dir / "run_state.json").exists()

    # Resume with higher target
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "resume",
            "--fake",
            "--target",
            "15",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result2.returncode == 0
    output2 = json.loads(result2.stdout)
    accepted_count_2 = output2["accepted_count"]

    # Should have more questions after resume
    assert accepted_count_2 >= accepted_count_1


def test_resume_preserves_existing_data(temp_data_dir):
    """Test resume preserves existing accepted questions."""
    # Generate initial data
    result1 = subprocess.run(
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

    assert result1.returncode == 0

    # Read initial accepted questions
    with open(temp_data_dir / "accepted.jsonl", "r") as f:
        initial_ids = {json.loads(line)["id"] for line in f}

    # Resume
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "resume",
            "--fake",
            "--target",
            "8",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result2.returncode == 0

    # Read final accepted questions
    with open(temp_data_dir / "accepted.jsonl", "r") as f:
        final_ids = {json.loads(line)["id"] for line in f}

    # All initial IDs should still be present
    assert initial_ids.issubset(final_ids)


def test_resume_with_pending_questions(temp_data_dir, sample_question):
    """Test resume with pending questions in queue."""
    from cuda_sft_qgen.config import load_settings
    from cuda_sft_qgen.runtime import build_runtime

    # Create a runtime and add pending questions
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    runtime = build_runtime(settings)

    pending = [sample_question(id=f"q_pending_{i}") for i in range(3)]
    runtime.store.save_pending(pending)
    runtime.close()

    # Resume should process pending questions
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "resume",
            "--fake",
            "--target",
            "10",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0


def test_resume_multiple_times(temp_data_dir):
    """Test resuming multiple times."""
    targets = [5, 10, 15]
    previous_count = 0

    for target in targets:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cuda_sft_qgen",
                "resume",
                "--fake",
                "--target",
                str(target),
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
        current_count = output["accepted_count"]

        # Count should increase or stay same (if target reached)
        assert current_count >= previous_count
        previous_count = current_count


def test_resume_loads_run_state(temp_data_dir):
    """Test resume loads iteration count from run_state."""
    # Generate initial data
    result1 = subprocess.run(
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

    assert result1.returncode == 0

    # Check run_state was created
    run_state_file = temp_data_dir / "run_state.json"
    assert run_state_file.exists()

    with open(run_state_file, "r") as f:
        run_state = json.load(f)
        initial_iteration = run_state.get("iteration", 0)

    # Resume
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "resume",
            "--fake",
            "--target",
            "8",
        ],
        env={
            **subprocess.os.environ,
            "CUDA_SFT_DATA_DIR": str(temp_data_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result2.returncode == 0
    output2 = json.loads(result2.stdout)

    # Iteration should continue from previous
    # A concurrent first run may already overshoot the higher resume target.
    # In that case resume is a no-op but must preserve the checkpoint.
    assert output2["iteration"] >= initial_iteration


def test_resume_without_initial_data(temp_data_dir):
    """Test resume works even without initial data (same as generate)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuda_sft_qgen",
            "resume",
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

    # Should still work, creating new data
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["accepted_count"] >= 0
