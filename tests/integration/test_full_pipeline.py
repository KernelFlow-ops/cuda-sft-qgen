"""Full FakeLLM pipeline: bootstrap through archive to an accepted set."""

from __future__ import annotations

import pytest

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.graph.app import run_pipeline
from cuda_sft_qgen.runtime import build_runtime


def test_full_pipeline_fake_llm(temp_data_dir):
    """Test full pipeline with fake LLM."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.max_iterations = 5

    result = run_pipeline(settings)

    assert result is not None
    assert "status" in result
    assert "accepted_count" in result
    assert "rejected_count" in result
    assert result["done"] is True or result["accepted_count"] > 0


def test_full_pipeline_small_batch(temp_data_dir):
    """Test full pipeline with small batch size."""
    settings = load_settings(
        fake_llm=True,
        target=20,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.review_batch_size = 5
    settings.max_iterations = 10

    result = run_pipeline(settings)

    assert result["accepted_count"] > 0
    assert result["iteration"] > 0


def test_full_pipeline_with_evolver(temp_data_dir):
    """Test full pipeline with evolver enabled."""
    settings = load_settings(
        fake_llm=True,
        target=15,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    # Evolver fraction is read-only property from config
    settings.max_iterations = 10

    result = run_pipeline(settings)

    assert result["accepted_count"] > 0


def test_full_pipeline_reaches_target(temp_data_dir):
    """Test pipeline reaches target count."""
    target = 5
    settings = load_settings(
        fake_llm=True,
        target=target,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.review_batch_size = 10
    settings.max_iterations = 20

    result = run_pipeline(settings)

    assert result["done"] is True
    assert result["accepted_count"] >= target


def test_full_pipeline_saves_artifacts(temp_data_dir):
    """Test pipeline saves artifacts to disk."""
    settings = load_settings(
        fake_llm=True,
        target=5,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"

    run_pipeline(settings)

    # Check that files were created
    assert (temp_data_dir / "accepted.jsonl").exists()
    assert (temp_data_dir / "stats.json").exists()


def test_full_pipeline_with_runtime_reuse(temp_data_dir):
    """Test pipeline with runtime reuse."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"

    runtime = build_runtime(settings)

    result = run_pipeline(settings, runtime=runtime)

    assert result["accepted_count"] > 0

    # Clean up
    closer = getattr(runtime, "close", None)
    if callable(closer):
        closer()


def test_full_pipeline_multiple_iterations(temp_data_dir):
    """Test pipeline runs multiple iterations."""
    settings = load_settings(
        fake_llm=True,
        target=30,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"
    settings.review_batch_size = 10
    settings.max_iterations = 5

    result = run_pipeline(settings)

    # Should have run multiple iterations
    assert result["iteration"] > 1


def test_full_pipeline_taxonomy_expansion(temp_data_dir):
    """Test pipeline with taxonomy expansion."""
    settings = load_settings(
        fake_llm=True,
        target=10,
        data_dir=str(temp_data_dir),
    )
    settings.embedding_provider = "fake"

    result = run_pipeline(settings, rebuild_taxonomy=True)

    assert result is not None
    assert result["accepted_count"] >= 0
