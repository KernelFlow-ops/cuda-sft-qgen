"""Integration tests for review_chair_node with parallel processing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cuda_sft_qgen.graph.nodes.review import review_chair_node
from cuda_sft_qgen.models import Question


@pytest.fixture
def mock_runtime_full():
    """Full mock runtime for integration tests."""
    rt = MagicMock()
    rt.knowledge = None
    rt.review_llm = None
    rt.taxonomy = None
    rt.store.load_accepted.return_value = []

    # LLM mock
    rt.llm.complete_json.return_value = {
        "decisions": [
            {"id": "q-001", "action": "accept", "reason": ""},
            {"id": "q-002", "action": "reject", "reason": "duplicate"},
        ],
        "steering_brief": None,
    }

    # Store mock
    rt.store.recent_rejected_skeletons.return_value = ["template1", "template2"]

    # Settings mock
    rt.settings.review_llm_every_k_batches = 1
    rt.settings.max_parse_retries = 1
    rt.settings.review = {
        "parallel_enabled": True,
        "chair_max_workers": 4,
        "chair_batch_size": 10,
    }
    rt.settings.generation = {
        "anti_templates_max": 15,
    }
    rt.settings.quotas = {
        "question_types": {"conceptual": 0.3, "debug": 0.2},
        "difficulty": {"1": 0.2, "2": 0.4, "3": 0.3, "4": 0.1},
        "l1_share": {"min": 0.02, "max": 0.15},
    }

    return rt


def create_test_question(qid: str, verdict: str = "unique") -> dict:
    """Create a test question dict."""
    q = Question(
        id=qid,
        l3_id="CUDA.Memory.L2",
        l3_claim="L2 cache",
        question=f"Test question {qid}",
        tests_claim=f"Test claim {qid}",
        hardware="hopper",
        topic_path=["CUDA", "Memory", "L2"],
        question_type="conceptual",
        kind="mechanism",
        difficulty=2,
        bloom="apply",
        persona="cuda_expert",
        language="en",
        granularity="api",
    )
    q.dup_verdict = verdict
    return {
        "question": q.model_dump(),
        "decision": {
            "verdict": verdict,
            "nearest_id": None,
            "nearest_cosine": None,
            "template_jaccard": None,
            "hamming": None,
            "reason": None,
        },
        "vector_row": 0,
    }


class TestReviewChairIntegration:
    """Integration tests for review_chair_node."""

    def test_parallel_mode_enabled(self, mock_runtime_full):
        """Test that parallel mode processes items correctly."""
        scored = [
            create_test_question("q-001", "gray"),
            create_test_question("q-002", "gray"),
            create_test_question("q-003", "unique"),
            create_test_question("q-004", "dup"),
        ]

        state = {
            "last_diversity": {
                "scored": scored,
                "report": {
                    "accepted_count": 100,
                    "pending_count": 4,
                    "l1_hist": {"CUDA": 60, "Memory": 40},
                    "type_hist": {"conceptual": 70, "debug": 30},
                    "difficulty_hist": {"1": 20, "2": 50, "3": 25, "4": 5},
                    "hardware_hist": {"hopper": 80, "ampere": 20},
                    "granularity_hist": {"api": 60, "kernel": 40},
                    "kind_hist": {"mechanism": 100},
                    "leaf_coverage": 0.85,
                    "kl_l1": 0.10,
                    "kl_type": 0.05,
                    "kl_difficulty": 0.05,
                    "entropy_ratio_l1": 0.90,
                    "mean_nn_cosine": 0.75,
                    "underrepresented": [],
                    "overrepresented": [],
                    "notes": [],
                },
            },
            "review_rounds": 0,
        }

        with patch("cuda_sft_qgen.graph.nodes.review.get_runtime", return_value=mock_runtime_full):
            with patch("cuda_sft_qgen.graph.nodes.review.load_prompt", return_value="Test prompt"):
                result = review_chair_node(state)

                # Verify result structure
                assert result["status"] == "review_chaired"
                assert "last_diversity" in result
                assert "steering_brief" in result
                assert result["review_rounds"] == 1

                # Verify decisions were made
                decisions = result["last_diversity"]["final_decisions"]
                assert len(decisions) == 4

                # Verify log mentions parallel mode
                assert any("parallel=True" in log for log in result.get("log", []))

    def test_serial_mode_fallback(self, mock_runtime_full):
        """Test that serial mode works when parallel is disabled."""
        mock_runtime_full.settings.review["parallel_enabled"] = False

        scored = [
            create_test_question("q-001", "gray"),
            create_test_question("q-002", "unique"),
        ]

        state = {
            "last_diversity": {
                "scored": scored,
                "report": {
                    "accepted_count": 100,
                    "pending_count": 2,
                    "l1_hist": {"CUDA": 60, "Memory": 40},
                    "type_hist": {"conceptual": 70, "debug": 30},
                    "difficulty_hist": {"1": 20, "2": 50, "3": 25, "4": 5},
                    "hardware_hist": {"hopper": 80, "ampere": 20},
                    "granularity_hist": {"api": 60, "kernel": 40},
                    "kind_hist": {"mechanism": 100},
                    "leaf_coverage": 0.85,
                    "kl_l1": 0.10,
                    "kl_type": 0.05,
                    "kl_difficulty": 0.05,
                    "entropy_ratio_l1": 0.90,
                    "mean_nn_cosine": 0.75,
                    "underrepresented": [],
                    "overrepresented": [],
                    "notes": [],
                },
            },
            "review_rounds": 0,
        }

        with patch("cuda_sft_qgen.graph.nodes.review.get_runtime", return_value=mock_runtime_full):
            with patch("cuda_sft_qgen.graph.nodes.review.load_prompt", return_value="Test prompt"):
                result = review_chair_node(state)

                # Verify result
                assert result["status"] == "review_chaired"
                decisions = result["last_diversity"]["final_decisions"]
                assert len(decisions) == 2

                # Verify log mentions serial mode
                assert any("parallel=False" in log for log in result.get("log", []))

    def test_enhanced_steering_high_imbalance(self, mock_runtime_full):
        """Test that enhanced steering activates for high imbalance."""
        scored = [create_test_question("q-001", "unique")]

        # High KL divergence and low entropy
        state = {
            "last_diversity": {
                "scored": scored,
                "report": {
                    "accepted_count": 100,
                    "pending_count": 1,
                    "l1_hist": {"CUDA": 90, "Memory": 8, "Kernel": 2},
                    "type_hist": {"conceptual": 70, "debug": 30},
                    "difficulty_hist": {"1": 20, "2": 50, "3": 25, "4": 5},
                    "hardware_hist": {"hopper": 80, "ampere": 20},
                    "granularity_hist": {"api": 60, "kernel": 40},
                    "kind_hist": {"mechanism": 100},
                    "leaf_coverage": 0.85,
                    "kl_l1": 0.30,  # High KL
                    "kl_type": 0.05,
                    "kl_difficulty": 0.05,
                    "entropy_ratio_l1": 0.70,  # Low entropy
                    "mean_nn_cosine": 0.75,
                    "underrepresented": [],
                    "overrepresented": ["CUDA"],
                    "notes": [],
                },
            },
            "review_rounds": 0,
        }

        mock_runtime_full.store.recent_rejected_skeletons.return_value = [f"skel-{i}" for i in range(30)]

        with patch("cuda_sft_qgen.graph.nodes.review.get_runtime", return_value=mock_runtime_full):
            with patch("cuda_sft_qgen.graph.nodes.review.load_prompt", return_value="Test prompt"):
                result = review_chair_node(state)

                # Verify enhanced steering was applied
                steering = result["steering_brief"]

                # Should contain scarcest L1s
                assert "Kernel" in steering["underrepresented"] or "Memory" in steering["underrepresented"]

                # Should have expanded forbidden templates
                assert len(steering["forbidden_templates"]) > 15

                # Should have notes about enhancements
                assert "High KL divergence" in steering["style_notes"] or "Low entropy" in steering["style_notes"]

    def test_consistency_suspects_processing(self, mock_runtime_full):
        """Test that consistency suspects are processed correctly."""
        # Create a question with leaf mismatch
        q_suspect = Question(
            id="q-suspect",
            l3_id="CUDA.Memory.L2",
            l3_claim="L2 cache",
            question="Test question with mismatch",
            tests_claim="Test claim about L1 cache",  # Mismatch with L2
            hardware="hopper",
            topic_path=["CUDA", "Memory", "L2"],
            question_type="conceptual",
            kind="mechanism",
            difficulty=2,
            bloom="apply",
            persona="cuda_expert",
            language="en",
            granularity="api",
        )
        q_suspect.dup_verdict = "unique"

        scored = [
            {
                "question": q_suspect.model_dump(),
                "decision": {
                    "verdict": "unique",
                    "nearest_id": None,
                    "nearest_cosine": None,
                    "template_jaccard": None,
                    "hamming": None,
                    "reason": None,
                },
                "vector_row": 0,
            }
        ]

        state = {
            "last_diversity": {
                "scored": scored,
                "report": {
                    "accepted_count": 100,
                    "pending_count": 1,
                    "l1_hist": {"CUDA": 60, "Memory": 40},
                    "type_hist": {"conceptual": 70, "debug": 30},
                    "difficulty_hist": {"1": 20, "2": 50, "3": 25, "4": 5},
                    "hardware_hist": {"hopper": 80, "ampere": 20},
                    "granularity_hist": {"api": 60, "kernel": 40},
                    "kind_hist": {"mechanism": 100},
                    "leaf_coverage": 0.85,
                    "kl_l1": 0.10,
                    "kl_type": 0.05,
                    "kl_difficulty": 0.05,
                    "entropy_ratio_l1": 0.90,
                    "mean_nn_cosine": 0.75,
                    "underrepresented": [],
                    "overrepresented": [],
                    "notes": [],
                },
            },
            "review_rounds": 0,
        }

        with patch("cuda_sft_qgen.graph.nodes.review.get_runtime", return_value=mock_runtime_full):
            with patch("cuda_sft_qgen.graph.nodes.review.load_prompt", return_value="Test prompt"):
                result = review_chair_node(state)

                # Verify suspects were identified and processed
                assert result["status"] == "review_chaired"
                assert "log" in result
                assert mock_runtime_full.llm.complete_json.call_count == 1
                assert result["last_diversity"]["final_decisions"][0]["action"] == "pending"
