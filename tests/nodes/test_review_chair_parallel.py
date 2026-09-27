"""Tests for parallel review chair processing (Agent D)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from cuda_sft_qgen.graph.nodes.review import (
    _enhanced_steering,
    _parallel_chair_decision,
    _parallel_chair_decisions,
)
from cuda_sft_qgen.graph.nodes.review_chair import (
    _chair_priority,
    _compact_knowledge,
    _compact_neighbors,
    _needs_second_pass,
)
from cuda_sft_qgen.keys import PRI_CHAIR, PRI_WRITER
from cuda_sft_qgen.models import DiversityReport, Question, SteeringBrief


@pytest.fixture
def mock_question():
    """Mock Question object."""
    return Question(
        id="test-q-001",
        l3_id="CUDA.Memory.L2",
        l3_claim="L2 cache behavior",
        question="How does L2 cache work in CUDA?",
        tests_claim="Test L2 cache behavior",
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


@pytest.fixture
def mock_runtime():
    """Mock runtime with LLM."""
    rt = MagicMock()
    rt.knowledge = None
    rt.review_llm = None
    rt.settings.review = {}
    rt.settings.review_model = "test-reviewer"
    rt.llm.complete_json.return_value = {
        "id": "test-q-001",
        "action": "accept",
        "reason": "Verified question",
        "checks": {k: "pass" for k in ("correctness", "answerability", "claim_alignment",
                    "hardware_compatibility", "answer_leakage", "metadata", "duplication")},
        "evidence": ["The claim and stem agree."],
        "internal_reference_answer": "Expected answer",
        "value_scores": {
            "pedagogical_value": 4,
            "discriminativeness": 4,
            "realism": 4,
        },
        "assessed_metadata": {"question_type": "conceptual", "difficulty": 2,
                              "bloom": "apply", "hardware": "hopper",
                              "granularity": "api", "language": "en"},
    }
    rt.store.recent_rejected_skeletons.return_value = ["template1", "template2"]
    rt.store.load_rejected.return_value = []
    rt.settings.quotas = {
        "question_types": {"conceptual": 0.3, "debug": 0.2},
        "difficulty": {"1": 0.2, "2": 0.4, "3": 0.3, "4": 0.1},
        "l1_share": {"min": 0.02, "max": 0.15},
    }
    return rt


def test_compact_neighbors_truncates_stems():
    neighbors = _compact_neighbors(
        [{"id": "n1", "question": "x" * 500, "similarity": 0.2, "l3_id": "leaf"}]
    )
    assert len(neighbors) == 1
    assert len(neighbors[0]["question"]) == 400
    assert neighbors[0]["id"] == "n1"


def test_compact_knowledge_empty_is_not_reject_signal():
    slim = _compact_knowledge({"claims": [], "evidence": []})
    assert slim["status"] == "missing"
    assert slim["evidence"] == []


def _cute_shape_chair_question(base: Question) -> Question:
    return base.model_copy(update={
        "question": (
            "What are the shapes from zipped_divide and tiled_divide?\n"
            "```cpp\n"
            "__global__ void build_attention_layouts() {\n"
            "using namespace cute;\n"
            "auto score = make_layout(make_shape(_12{}, _18{}));\n"
            "auto tile = make_tile(_3{}, _2{});\n"
            "auto z = zipped_divide(score, tile);\n"
            "auto t = tiled_divide(score, tile);\n"
            "(void)z; (void)t;\n"
            "}\n```\n"
            "A. Both calls are well-formed; z has shape ((_3,_2),(_4,_9)) and t has shape (_6,(_4,_9)).\n"
            "B. Both calls are well-formed; z has shape (_6,(_4,_9)) and t has shape ((_3,_2),(_4,_9)).\n"
            "C. Both calls are well-formed; z has shape ((_3,_2),(_4,_9)) and t has shape ((_3,_2),(_4,_9)).\n"
            "D. Both calls are well-formed; z has shape (_6,_4,_9) and t has shape (_6,_4,_9)."
        ),
        "answer_format": "single_choice",
        "draft_target": "A",
        "internal_reference_answer": "A",
    })


def test_cute_shape_guard_contradiction_overrides_chair_accept(mock_question, mock_runtime):
    from cuda_sft_qgen.review.cute_shapes import check_cute_shapes

    question = _cute_shape_chair_question(mock_question)
    shape_result = check_cute_shapes(question)
    assert shape_result["reason"] == "no_correct_option"
    verified = {
        "status": "fail",
        "reason": "cute_shape_contradiction",
        "evidence": {"cute_shapes": shape_result, "votes": [{"answer": "A"}] * 5},
    }
    item = {"question": question.model_dump(), "decision": {"verdict": "gray"}}

    with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
        with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
            result = _parallel_chair_decision(item, "Test prompt", verified)

    assert result["action"] == "reject"
    assert result["reason"].startswith("chair:correctness:cute_shape_guard:no_correct_option")
    assert result["review"]["model_action"] == "accept"


def test_unsupported_cute_shape_guard_keeps_chair_accept_pending(mock_question, mock_runtime):
    question = _cute_shape_chair_question(mock_question)
    unsupported = {
        "status": "pending",
        "reason": "cute_shape_unsupported",
        "evidence": {"cute_shapes": {"status": "unsupported", "reason": "source_dynamic"}},
    }
    item = {"question": question.model_dump(), "decision": {"verdict": "gray"}}

    with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
        with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
            result = _parallel_chair_decision(item, "Test prompt", unsupported)

    assert result["action"] == "pending"
    assert result["reason"] == "verification:cute_shape_unsupported"


def test_other_verifier_failure_does_not_use_cute_shape_force_reject(mock_question, mock_runtime):
    question = _cute_shape_chair_question(mock_question)
    ordinary_failure = {
        "status": "fail",
        "reason": "runtime_claim_contradicted",
        "evidence": {"stdout": "unexpected"},
    }
    item = {"question": question.model_dump(), "decision": {"verdict": "gray"}}

    with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
        with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
            result = _parallel_chair_decision(item, "Test prompt", ordinary_failure)

    assert result["action"] == "pending"
    assert result["reason"] == "verification:runtime_claim_contradicted"


class TestParallelChairDecision:
    """Test single chair decision processing."""

    def test_accept_decision(self, mock_question, mock_runtime):
        """Test that accept decision is returned correctly."""
        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
                item = {
                    "question": mock_question.model_dump(),
                    "decision": {"verdict": "gray", "nearest_cosine": 0.92},
                    "is_suspect": False,
                }

                result = _parallel_chair_decision(item, "Test prompt")

                assert result["id"] == "test-q-001"
                assert result["action"] == "accept"
                assert "reason" in result

    def test_reject_decision(self, mock_question, mock_runtime):
        """Test that reject decision is returned correctly."""
        mock_runtime.llm.complete_json.return_value.update(action="reject", reason="duplicate question")
        mock_runtime.llm.complete_json.return_value["checks"]["duplication"] = "fail"

        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
                item = {
                    "question": mock_question.model_dump(),
                    "decision": {"verdict": "gray"},
                    "is_suspect": False,
                }

                result = _parallel_chair_decision(item, "Test prompt")

                assert result["id"] == "test-q-001"
                assert result["action"] == "reject"
                assert result["reason"] == "chair:duplicate:duplicate question"

    def test_llm_failure_fallback(self, mock_question, mock_runtime):
        """Test that LLM failure remains pending."""
        mock_runtime.llm.complete_json.side_effect = Exception("LLM error")

        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
                item = {
                    "question": mock_question.model_dump(),
                    "decision": {"verdict": "gray"},
                    "is_suspect": False,
                }

                result = _parallel_chair_decision(item, "Test prompt")

                assert result["id"] == "test-q-001"
                assert result["action"] == "pending"


class TestParallelChairDecisions:
    """Test parallel processing of multiple chair decisions."""

    def test_empty_items(self):
        """Test that empty items return empty dict."""
        result = _parallel_chair_decisions([])
        assert result == {}

    def test_parallel_processing(self, mock_question, mock_runtime):
        """Test that multiple items are processed in parallel."""
        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
                items = []
                for i in range(5):
                    q = mock_question.model_copy()
                    q.id = f"test-q-{i:03d}"
                    items.append({
                        "question": q.model_dump(),
                        "decision": {"verdict": "gray"},
                        "is_suspect": False,
                    })

                start = time.time()
                results = _parallel_chair_decisions(items, max_workers=4)
                elapsed = time.time() - start

                assert len(results) == 5
                assert all(qid in results for qid in [f"test-q-{i:03d}" for i in range(5)])
                # Parallel should be faster than 5 * single_call_time (rough check)
                assert elapsed < 5.0  # Generous upper bound

    def test_deterministic_results(self, mock_question, mock_runtime):
        """Test that same input produces same output (determinism)."""
        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            with patch("cuda_sft_qgen.graph.nodes.review_chair.load_prompt", return_value="Test prompt"):
                items = []
                for i in range(3):
                    q = mock_question.model_copy()
                    q.id = f"test-q-{i:03d}"
                    items.append({
                        "question": q.model_dump(),
                        "decision": {"verdict": "gray"},
                        "is_suspect": False,
                    })

                results1 = _parallel_chair_decisions(items, max_workers=2)
                results2 = _parallel_chair_decisions(items, max_workers=2)

                # Results should have same keys
                assert set(results1.keys()) == set(results2.keys())
                # Results should have same actions (assuming mock returns consistent values)
                for qid in results1:
                    assert results1[qid]["action"] == results2[qid]["action"]


class TestEnhancedSteering:
    """Test enhanced steering generation with high-imbalance handling."""

    def test_base_steering(self, mock_runtime):
        """Test base steering without enhancements."""
        report = DiversityReport(
            accepted_count=100,
            pending_count=20,
            l1_hist={"CUDA": 50, "Memory": 30, "Kernel": 20},
            type_hist={"conceptual": 60, "debug": 40},
            difficulty_hist={"1": 20, "2": 40, "3": 30, "4": 10},
            hardware_hist={"hopper": 80, "ampere": 20},
            granularity_hist={"api": 60, "kernel": 40},
            kind_hist={"mechanism": 100},
            leaf_coverage=0.85,
            kl_l1=0.10,  # Low KL
            kl_type=0.05,
            kl_difficulty=0.05,
            entropy_ratio_l1=0.90,  # High entropy
            mean_nn_cosine=0.75,
            underrepresented=["CUDA"],
            overrepresented=["Memory"],
            notes=[],
        )

        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            steering = _enhanced_steering(report, mock_runtime.settings.quotas, ["t1", "t2"], anti_cap=15)

            assert isinstance(steering, SteeringBrief)
            assert "CUDA" in steering.underrepresented
            assert "Memory" in steering.overrepresented
            assert len(steering.forbidden_templates) >= 2

    def test_high_kl_enhancement(self, mock_runtime):
        """Test that high KL divergence forces scarcest L1s."""
        report = DiversityReport(
            accepted_count=100,
            pending_count=20,
            l1_hist={"CUDA": 70, "Memory": 20, "Kernel": 5, "Profiling": 3, "Debug": 2},
            type_hist={"conceptual": 60, "debug": 40},
            difficulty_hist={"1": 20, "2": 40, "3": 30, "4": 10},
            hardware_hist={"hopper": 80, "ampere": 20},
            granularity_hist={"api": 60, "kernel": 40},
            kind_hist={"mechanism": 100},
            leaf_coverage=0.85,
            kl_l1=0.25,  # High KL (> 0.20)
            kl_type=0.05,
            kl_difficulty=0.05,
            entropy_ratio_l1=0.90,
            mean_nn_cosine=0.75,
            underrepresented=[],
            overrepresented=["CUDA"],
            notes=[],
        )

        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            steering = _enhanced_steering(report, mock_runtime.settings.quotas, [], anti_cap=15)

            # Should contain top-3 scarcest: Debug (2), Profiling (3), Kernel (5)
            assert "Debug" in steering.underrepresented or "Profiling" in steering.underrepresented
            assert "High KL divergence" in steering.style_notes

    def test_low_entropy_enhancement(self, mock_runtime):
        """Test that low entropy expands forbidden templates."""
        report = DiversityReport(
            accepted_count=100,
            pending_count=20,
            l1_hist={"CUDA": 90, "Memory": 10},  # Very imbalanced
            type_hist={"conceptual": 60, "debug": 40},
            difficulty_hist={"1": 20, "2": 40, "3": 30, "4": 10},
            hardware_hist={"hopper": 80, "ampere": 20},
            granularity_hist={"api": 60, "kernel": 40},
            kind_hist={"mechanism": 100},
            leaf_coverage=0.85,
            kl_l1=0.10,
            kl_type=0.05,
            kl_difficulty=0.05,
            entropy_ratio_l1=0.70,  # Low entropy (< 0.80)
            mean_nn_cosine=0.75,
            underrepresented=[],
            overrepresented=["CUDA"],
            notes=[],
        )

        mock_runtime.store.recent_rejected_skeletons.return_value = [f"skel-{i}" for i in range(30)]

        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            steering = _enhanced_steering(report, mock_runtime.settings.quotas, [], anti_cap=15)

            # Should expand forbidden templates
            assert len(steering.forbidden_templates) > 15  # More than base anti_cap
            assert "Low entropy" in steering.style_notes

    def test_combined_enhancements(self, mock_runtime):
        """Test that both enhancements work together."""
        report = DiversityReport(
            accepted_count=100,
            pending_count=20,
            l1_hist={"CUDA": 90, "Memory": 8, "Kernel": 2},
            type_hist={"conceptual": 60, "debug": 40},
            difficulty_hist={"1": 20, "2": 40, "3": 30, "4": 10},
            hardware_hist={"hopper": 80, "ampere": 20},
            granularity_hist={"api": 60, "kernel": 40},
            kind_hist={"mechanism": 100},
            leaf_coverage=0.85,
            kl_l1=0.30,  # High KL
            kl_type=0.05,
            kl_difficulty=0.05,
            entropy_ratio_l1=0.70,  # Low entropy
            mean_nn_cosine=0.75,
            underrepresented=[],
            overrepresented=["CUDA"],
            notes=[],
        )

        mock_runtime.store.recent_rejected_skeletons.return_value = [f"skel-{i}" for i in range(30)]

        with patch("cuda_sft_qgen.graph.nodes.review_chair.get_runtime", return_value=mock_runtime):
            steering = _enhanced_steering(report, mock_runtime.settings.quotas, [], anti_cap=15)

            # Both enhancements should be present
            assert "Kernel" in steering.underrepresented  # Scarcest L1
            assert len(steering.forbidden_templates) > 15  # Expanded templates
            assert "High KL divergence" in steering.style_notes
            assert "Low entropy" in steering.style_notes


def test_chair_priority_defaults_behind_writer():
    from types import SimpleNamespace

    assert _chair_priority(SimpleNamespace(review={})) == PRI_CHAIR
    assert PRI_CHAIR > PRI_WRITER
    assert _chair_priority(SimpleNamespace(review={"priority": 0})) == 0


def test_second_pass_for_uncertain_debug_not_reject():
    from types import SimpleNamespace

    q = Question(
        id="q_debug",
        question="What race occurs if this kernel omits __syncthreads?",
        tests_claim="Missing barrier causes a shared-memory race.",
        topic_path=["compute"],
        l3_id="barrier",
        question_type="debug",
        difficulty=3,
        bloom="analyze",
        persona="x",
        hardware="hopper",
        granularity="kernel",
        language="en",
    )
    settings = SimpleNamespace(review={"second_pass_on_uncertain": True})
    pending_uncertain = {
        "action": "pending",
        "review": {"checks": {"correctness": "uncertain", "answerability": "pass"}},
    }
    assert _needs_second_pass(q, pending_uncertain, settings) is True
    q.question_type = "conceptual"
    assert _needs_second_pass(q, pending_uncertain, settings) is False
    q.question_type = "debug"
    assert _needs_second_pass(q, {"action": "accept", "review": {"checks": {"correctness": "pass"}}}, settings) is False
    assert _needs_second_pass(q, {"action": "pending", "review": {"checks": {"correctness": "fail"}}}, settings) is False
