"""Shared pytest fixtures: temp data dirs, fake runtime, and sample questions."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from cuda_sft_qgen import config as qgen_config
from cuda_sft_qgen.config import Settings, load_settings
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.runtime import Runtime, build_runtime
from cuda_sft_qgen.store.archive import Archive
from cuda_sft_qgen.taxonomy import Taxonomy, Leaf


def _skip_dotenv(*args, **kwargs) -> bool:
    """Stand-in for ``load_dotenv`` in sessions without real-LLM tests."""
    return False


def pytest_collection_finish(session):
    """Read the developer's .env only when the session selected ``llm`` tests.

    Offline tests must behave the same in every checkout (worktrees have no
    .env), so load_settings() is kept from reading it; real-LLM sessions load
    it once here, before any test or setup hook runs, exactly as before.
    """
    if any(item.get_closest_marker("llm") for item in session.items):
        load_dotenv(ROOT / ".env", override=False)
    else:
        qgen_config.load_dotenv = _skip_dotenv


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fake_settings(temp_data_dir):
    """Create settings with fake LLM and temporary data directory."""
    return load_settings(
        fake_llm=True,
        target=100,
        data_dir=str(temp_data_dir),
    )


@pytest.fixture
def fake_runtime(fake_settings):
    """Create a runtime with fake LLM for testing."""
    return build_runtime(fake_settings)


@pytest.fixture
def mock_archive(temp_data_dir):
    """Create a mock Archive for testing."""
    return Archive(temp_data_dir)


def sample_question(
    id: str = "q_test_001",
    question: str = "Test question about CUDA occupancy",
    topic_path: list[str] | None = None,
    l3_id: str = "cuda_occupancy_001",
    l3_claim: str = "CUDA occupancy determines SM utilization",
    question_type: str = "conceptual",
    difficulty: int = 2,
    bloom: str = "understand",
    persona: str = "cuda_interviewer",
    hardware: str = "hopper",
    granularity: str = "kernel",
    language: str = "zh",
    source: str = "writer",
    **kwargs
) -> Question:
    """Create a sample Question for testing."""
    if topic_path is None:
        topic_path = ["compute", "occupancy", l3_id]
    return Question(
        id=id,
        question=question,
        tests_claim=l3_claim,
        topic_path=topic_path,
        l3_id=l3_id,
        l3_claim=l3_claim,
        question_type=question_type,
        difficulty=difficulty,
        bloom=bloom,
        persona=persona,
        hardware=hardware,
        granularity=granularity,
        language=language,
        source=source,
        **kwargs
    )


def sample_brief(
    l3_id: str = "cuda_occupancy_001",
    l3_claim: str = "CUDA occupancy determines SM utilization",
    topic_path: list[str] | None = None,
    question_type: str = "conceptual",
    difficulty: int = 2,
    bloom: str = "understand",
    **kwargs
) -> QuestionBrief:
    """Create a sample QuestionBrief for testing."""
    if topic_path is None:
        topic_path = ["compute", "occupancy", l3_id]
    return QuestionBrief(
        topic_path=topic_path,
        l3_id=l3_id,
        l3_claim=l3_claim,
        question_type=question_type,
        difficulty=difficulty,
        bloom=bloom,
        persona="cuda_interviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
        **kwargs
    )


def make_steering_brief(**kwargs) -> SteeringBrief:
    """Create a sample SteeringBrief for testing."""
    return SteeringBrief(
        underrepresented=[],
        overrepresented=[],
        forbidden_templates=[],
        preferred_types=[],
        difficulty_nudge="hold",
        style_notes="",
        **kwargs
    )


@pytest.fixture
def sample_steering_brief():
    """Fixture that returns the steering-brief factory."""
    return make_steering_brief


@pytest.fixture
def sample_question():
    """Fixture that returns the sample_question factory function."""
    def _make_question(
        id: str = "q_test_001",
        question: str = "Test question about CUDA occupancy",
        topic_path: list[str] | None = None,
        l3_id: str = "cuda_occupancy_001",
        l3_claim: str = "CUDA occupancy determines SM utilization",
        question_type: str = "conceptual",
        difficulty: int = 2,
        bloom: str = "understand",
        persona: str = "cuda_interviewer",
        hardware: str = "hopper",
        granularity: str = "kernel",
        language: str = "zh",
        source: str = "writer",
        **kwargs
    ) -> Question:
        if topic_path is None:
            topic_path = ["compute", "occupancy", l3_id]
        return Question(
            id=id,
            question=question,
            tests_claim=l3_claim,
            topic_path=topic_path,
            l3_id=l3_id,
            l3_claim=l3_claim,
            question_type=question_type,
            difficulty=difficulty,
            bloom=bloom,
            persona=persona,
            hardware=hardware,
            granularity=granularity,
            language=language,
            source=source,
            **kwargs
        )
    return _make_question


@pytest.fixture
def sample_brief():
    """Fixture that returns the sample_brief factory function."""
    def _make_brief(
        l3_id: str = "cuda_occupancy_001",
        l3_claim: str = "CUDA occupancy determines SM utilization",
        topic_path: list[str] | None = None,
        question_type: str = "conceptual",
        difficulty: int = 2,
        bloom: str = "understand",
        **kwargs
    ) -> QuestionBrief:
        if topic_path is None:
            topic_path = ["compute", "occupancy", l3_id]
        return QuestionBrief(
            topic_path=topic_path,
            l3_id=l3_id,
            l3_claim=l3_claim,
            question_type=question_type,
            difficulty=difficulty,
            bloom=bloom,
            persona="cuda_interviewer",
            hardware="hopper",
            granularity="kernel",
            language="zh",
            **kwargs
        )
    return _make_brief


def sample_taxonomy() -> Taxonomy:
    """Create a minimal sample taxonomy for testing."""
    domains = [
        {
            "id": "compute",
            "name": "Compute & Memory",
            "children": [
                {
                    "id": "occupancy",
                    "name": "Occupancy",
                    "leaves": [
                        {
                            "id": "cuda_occupancy_001",
                            "claim": "CUDA occupancy determines SM utilization",
                            "types": ["conceptual"],
                            "difficulty": [1, 2, 3],
                            "kind": "mechanism",
                        },
                        {
                            "id": "cuda_occupancy_002",
                            "claim": "Register pressure affects occupancy",
                            "types": ["debug", "code_complete"],
                            "difficulty": [2, 3],
                            "kind": "failure",
                        },
                    ],
                }
            ],
        }
    ]
    leaves = [
        Leaf(
            domain_id="compute",
            domain_name="Compute & Memory",
            l2_id="occupancy",
            l2_name="Occupancy",
            l3_id="cuda_occupancy_001",
            claim="CUDA occupancy determines SM utilization",
            types=("conceptual",),
            difficulty=(1, 2, 3),
            kind="mechanism",
        ),
        Leaf(
            domain_id="compute",
            domain_name="Compute & Memory",
            l2_id="occupancy",
            l2_name="Occupancy",
            l3_id="cuda_occupancy_002",
            claim="Register pressure affects occupancy",
            types=("debug", "code_complete"),
            difficulty=(2, 3),
            kind="failure",
        ),
    ]
    return Taxonomy(version=1, name="cuda_test", leaves=leaves, domains=domains)
