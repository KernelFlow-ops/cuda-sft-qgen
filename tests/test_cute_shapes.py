"""Deterministic guard for the whitelisted CuTe divide shape item."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from cuda_sft_qgen.models import Question
from cuda_sft_qgen.review.cute_shapes import check_cute_shapes

_ARCHIVE = Path(__file__).resolve().parents[1] / "data/validation/after-s5pvGe/gate/questions.sqlite3"
_ARCHIVE_QUESTION_ID = "q_b74742ecf6b14700a5fd0f2ad0f1055d"


_CODE = """using namespace cute;
constexpr auto layout = make_layout(make_shape(_12{}, Int<18>{}));
constexpr auto tiler = make_tile(_3{}, _3{});
constexpr auto z = zipped_divide(layout, tiler);
constexpr auto t = tiled_divide(layout, tiler);"""
_Z = "((_3,_3),(_4,_6))"
_T = "((_3,_3),_4,_6)"


def _option(z: str, t: str) -> str:
    return f"Both calls are well-formed; `z` has shape `{z}` and `t` has shape `{t}`."


def _question(options: list[str], *, target: str = "A", code: str = _CODE) -> Question:
    option_text = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", options))
    text = (
        "What are the shapes produced by zipped_divide and tiled_divide?\n"
        f"```cpp\n{code}\n```\n{option_text}"
    )
    return Question(
        id="cute_shape_guard",
        question=text,
        tests_claim="CuTe divide operations preserve the documented shape hierarchy.",
        topic_path=["T20", "divide", "shapes"],
        question_type="syntax",
        difficulty=2,
        bloom="apply",
        persona="cuda_interviewer",
        hardware="agnostic",
        granularity="api",
        language="en",
        answer_format="single_choice",
        draft_target=target,
        internal_reference_answer=target,
    )


def _wrong_options() -> list[str]:
    return [
        _option("((_3,_3),_4,_6)", "((_3,_3),(_4,_6))"),
        _option("((_3,_3),(_4,_6))", "((_3,_3),(_4,_6))"),
        _option("((_4,_6),(_3,_3))", "((_4,_6),_3,_3)"),
        _option("((_3,_3),(_4,_5))", "((_3,_3),_4,_5)"),
    ]


def test_shape_guard_accepts_unique_correct_option_and_scalar_leaf():
    result = check_cute_shapes(_question([
        _option(_Z, _T), *_wrong_options()[1:],
    ]))

    assert result["status"] == "pass"
    assert result["correct_options"] == ["A"]
    assert result["expected_t"] == _T
    assert result["answer_key"] == "A"


def test_shape_guard_rejects_original_all_wrong_options():
    result = check_cute_shapes(_question(_wrong_options()))

    assert result["status"] == "contradiction", result.get("reason")
    assert result["reason"] == "no_correct_option"
    assert result["correct_options"] == []


def test_after_gate_attention_question_has_no_correct_option():
    code = """__global__ void build_attention_layouts() {
using namespace cute;
auto score = make_layout(make_shape(_12{}, _18{}));
auto tile = make_tile(_3{}, _2{});
auto z = zipped_divide(score, tile);
auto t = tiled_divide(score, tile);
(void)z; (void)t;
}"""
    options = [
        "Both calls are well-formed; z has shape ((_3,_2),(_4,_9)) "
        "and t has shape (_6,(_4,_9)).",
        "Both calls are well-formed; z has shape (_6,(_4,_9)) "
        "and t has shape ((_3,_2),(_4,_9)).",
        "Both calls are well-formed; z has shape ((_3,_2),(_4,_9)) "
        "and t has shape ((_3,_2),(_4,_9)).",
        "Both calls are well-formed; z has shape (_6,_4,_9) "
        "and t has shape (_6,_4,_9).",
    ]

    result = check_cute_shapes(_question(options, code=code))

    assert result["status"] == "contradiction"
    assert result["reason"] == "no_correct_option"
    assert result["expected_z"] == "((_3,_2),(_4,_9))"
    assert result["expected_t"] == "((_3,_2),_4,_9)"


def test_exact_archive_question_has_no_correct_option():
    if not _ARCHIVE.is_file():
        pytest.skip("local validation archive is unavailable")
    with sqlite3.connect(f"file:{_ARCHIVE.as_posix()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT payload FROM questions WHERE id=?", (_ARCHIVE_QUESTION_ID,)).fetchone()
    assert row is not None, "local validation archive question is unavailable"
    try:
        question = Question.model_validate(json.loads(row[0]))
    except Exception:
        pytest.fail("local validation archive question could not be loaded")

    result = check_cute_shapes(question)

    assert result["status"] == "contradiction", result.get("reason")
    assert result["reason"] == "no_correct_option"


def test_shape_guard_rejects_duplicate_correct_options():
    result = check_cute_shapes(_question([
        _option(_Z, _T), _option(_Z, _T), *_wrong_options()[2:],
    ]))

    assert result["status"] == "contradiction"
    assert result["reason"] == "multiple_correct_options"
    assert result["correct_options"] == ["A", "B"]


def test_shape_guard_rejects_answer_key_mismatch():
    result = check_cute_shapes(_question([
        _option(_Z, _T), *_wrong_options()[1:],
    ], target="B"))

    assert result["status"] == "contradiction"
    assert result["reason"] == "answer_key_mismatch"


@pytest.mark.parametrize(
    "code",
    [
        _CODE.replace("make_shape(_12{}, Int<18>{})", "make_shape(M, Int<18>{})"),
        _CODE.replace(
            "make_layout(make_shape(_12{}, Int<18>{}))",
            "make_layout(make_shape(_12{}, Int<18>{}), make_stride(_1{}, _12{}))",
        ),
        _CODE + "\nconstexpr auto extra = _1{};",
        _CODE + "\nz = zipped_divide(layout, tiler);",
    ],
    ids=["dynamic-dimension", "custom-stride", "extra-variable", "redefined-result"],
)
def test_shape_guard_fails_closed_outside_source_whitelist(code: str):
    result = check_cute_shapes(_question([
        _option(_Z, _T), *_wrong_options()[1:],
    ], code=code))

    assert result["status"] == "unsupported"
