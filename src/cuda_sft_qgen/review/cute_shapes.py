"""Strict static checks for two-dimensional CuTe divide shape questions."""

from __future__ import annotations

import ast
import re
from typing import TypeAlias

from cuda_sft_qgen.models import Question

Shape: TypeAlias = int | tuple["Shape", ...]

_SHAPE_ASK = re.compile(r"\bshapes?\b|形状", re.I)
_CODE_BLOCK = re.compile(r"```(?:cuda|cpp|c\+\+|c)?[ \t]*\n(.*?)```", re.I | re.S)
_OPTION = re.compile(r"^[ \t]*([A-D])[.):][ \t]*(.*?)(?=^[ \t]*[A-D][.):][ \t]*|\Z)", re.M | re.S)
_SHAPE_LABEL = re.compile(
    r"(?<![A-Za-z0-9_])(?P<label>`?[zt]`?)\s+has\s+shape\s+",
    re.I,
)
_SHARED_SHAPE_LABEL = re.compile(
    r"(?<![A-Za-z0-9_])both\s+(?P<first>`?[zt]`?)\s+and\s+"
    r"(?P<second>`?[zt]`?)\s+have\s+shape\s+",
    re.I,
)
_DIM = r"(?:(?:cute::)?Int\s*<\s*(\d+)\s*>\s*\{\s*\}|_(\d+)\s*\{\s*\})"
_DECL_PREFIX = r"(?:(?:constexpr|const)\s+)*auto\s+"
_AUTO_DECL = re.compile(r"\b(?:(?:constexpr|const)\s+)*auto\s+(\w+)\s*=")
_FUNCTION_OPEN = re.compile(
    r"(?:int\s+main|__global__\s+void\s+[A-Za-z_]\w*)\s*\(\s*(?:void\s*)?\)\s*\{"
)
_VOID_USE = re.compile(r"\(\s*void\s*\)\s*([zt])\s*;")


class _UnsupportedShapeSyntax(ValueError):
    """The question is in scope but exceeds the guard's explicit grammar."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _result(status: str, reason: str, **details) -> dict[str, object]:
    return {"status": status, "reason": reason, **details}


def _dim_value(match: re.Match[str], first: int, second: int) -> int:
    return int(match.group(first) or match.group(second))


def _source_dimensions(text: str) -> tuple[int, int, int, int]:
    """Read only the four literal declarations in the whitelist."""
    blocks = list(_CODE_BLOCK.finditer(text))
    if len(blocks) != 1:
        raise _UnsupportedShapeSyntax("source_code_block_count")
    code = blocks[0].group(1)
    if len(code) > 4000:
        raise _UnsupportedShapeSyntax("source_too_long")
    code = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", code)

    # Keep harmless CuTe setup explicit. Any other statement or directive has
    # to stay visible to the reader and is outside the static proof grammar.
    lines = code.splitlines()
    kept_lines = []
    include_seen = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
        elif re.fullmatch(r"#include\s*[<\"]cute/(?:layout|tensor)\.hpp[>\"]", stripped):
            if include_seen:
                raise _UnsupportedShapeSyntax("source_include_not_whitelisted")
            include_seen = True
            continue
        elif re.fullmatch(r"using\s+namespace\s+cute\s*;", stripped):
            continue
        else:
            kept_lines.append(line)
    code = "\n".join(kept_lines).strip()

    # Accept a single no-argument function wrapper, with braces on their own
    # lines so its body cannot conceal additional executable statements.
    code_lines = code.splitlines()
    has_function_wrapper = bool(code_lines and _FUNCTION_OPEN.fullmatch(code_lines[0].strip()))
    if has_function_wrapper:
        if not code_lines or code_lines[-1].strip() != "}":
            raise _UnsupportedShapeSyntax("source_wrapper_not_whitelisted")
        code = "\n".join(code_lines[1:-1])

    layout_pattern = re.compile(
        rf"\b{_DECL_PREFIX}(\w+)\s*=\s*(?:cute::)?make_layout\s*\(\s*"
        rf"(?:cute::)?make_shape\s*\(\s*"
        rf"{_DIM}\s*,\s*{_DIM}\s*\)\s*\)\s*;",
        re.S,
    )
    tile_pattern = re.compile(
        rf"\b{_DECL_PREFIX}(\w+)\s*=\s*(?:cute::)?make_tile\s*\(\s*"
        rf"{_DIM}\s*,\s*{_DIM}\s*\)\s*;",
        re.S,
    )
    layout_matches = list(layout_pattern.finditer(code))
    tile_matches = list(tile_pattern.finditer(code))
    auto_declarations = list(_AUTO_DECL.finditer(code))
    if (len(layout_matches) != 1 or len(tile_matches) != 1
            or len(auto_declarations) != 4
            or len(re.findall(r"\b(?:cute::)?make_layout\s*\(", code)) != 1
            or len(re.findall(r"\b(?:cute::)?make_shape\s*\(", code)) != 1
            or len(re.findall(r"\b(?:cute::)?make_tile\s*\(", code)) != 1):
        raise _UnsupportedShapeSyntax("source_layout_or_tile_not_whitelisted")
    if re.search(r"\b(?:Stride|make_stride)\b", code):
        raise _UnsupportedShapeSyntax("custom_stride_not_supported")

    layout = layout_matches[0]
    tile = tile_matches[0]
    layout_name, tile_name = layout.group(1), tile.group(1)
    z_pattern = re.compile(
        rf"\b{_DECL_PREFIX}z\s*=\s*(?:cute::)?zipped_divide\s*\(\s*{re.escape(layout_name)}"
        rf"\s*,\s*{re.escape(tile_name)}\s*\)\s*;",
        re.S,
    )
    t_pattern = re.compile(
        rf"\b{_DECL_PREFIX}t\s*=\s*(?:cute::)?tiled_divide\s*\(\s*{re.escape(layout_name)}"
        rf"\s*,\s*{re.escape(tile_name)}\s*\)\s*;",
        re.S,
    )
    if (len(list(z_pattern.finditer(code))) != 1
            or len(list(t_pattern.finditer(code))) != 1
            or len(re.findall(r"\b(?:cute::)?zipped_divide\s*\(", code)) != 1
            or len(re.findall(r"\b(?:cute::)?tiled_divide\s*\(", code)) != 1):
        raise _UnsupportedShapeSyntax("divide_calls_not_whitelisted")
    void_uses = list(_VOID_USE.finditer(code))
    if void_uses:
        if ([match.group(1) for match in void_uses] != ["z", "t"] or not has_function_wrapper
                or void_uses[0].start() < max(layout.end(), tile.end(),
                                              *[m.end() for m in z_pattern.finditer(code)],
                                              *[m.end() for m in t_pattern.finditer(code)])):
            raise _UnsupportedShapeSyntax("source_void_uses_not_whitelisted")

    declarations = [layout_name, tile_name, "z", "t"]
    if len(set(declarations)) != 4 or sorted(match.group(1) for match in auto_declarations) != sorted(declarations):
        raise _UnsupportedShapeSyntax("source_variables_not_whitelisted")

    # The four matched declarations must account for every executable token.
    spans = sorted(
        (match.start(), match.end())
        for match in (
            layout, tile, *list(z_pattern.finditer(code)), *list(t_pattern.finditer(code)), *void_uses
        )
    )
    cursor = 0
    for start, end in spans:
        if code[cursor:start].strip():
            raise _UnsupportedShapeSyntax("source_executable_statements_not_whitelisted")
        cursor = end
    if code[cursor:].strip():
        raise _UnsupportedShapeSyntax("source_executable_statements_not_whitelisted")

    values = (
        _dim_value(layout, 2, 3),
        _dim_value(layout, 4, 5),
        _dim_value(tile, 2, 3),
        _dim_value(tile, 4, 5),
    )
    if any(value < 1 or value > 2_147_483_647 for value in values):
        raise _UnsupportedShapeSyntax("dimension_outside_static_range")
    return values


def _shape_tree(node: ast.AST) -> Shape:
    if isinstance(node, ast.Name):
        match = re.fullmatch(r"_(\d+)", node.id)
        if match:
            return int(match.group(1))
    elif isinstance(node, ast.Tuple):
        return tuple(_shape_tree(item) for item in node.elts)
    raise _UnsupportedShapeSyntax("option_shape_outside_whitelist")


def _parse_option_shape(expression: str) -> Shape:
    if len(expression) > 240:
        raise _UnsupportedShapeSyntax("option_shape_too_long")
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise _UnsupportedShapeSyntax("option_shape_parse_error") from exc
    if sum(1 for _ in ast.walk(parsed)) > 32:
        raise _UnsupportedShapeSyntax("option_shape_too_complex")
    return _shape_tree(parsed.body)


def _consume_shape(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    quoted = start < len(text) and text[start] == "`"
    if quoted:
        start += 1
    if start >= len(text) or text[start] != "(":
        raise _UnsupportedShapeSyntax("option_shape_missing_tuple")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char not in "(),_0123456789 \t\r\n":
            raise _UnsupportedShapeSyntax("option_shape_outside_whitelist")
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                if quoted:
                    while end < len(text) and text[end].isspace():
                        end += 1
                    if end >= len(text) or text[end] != "`":
                        raise _UnsupportedShapeSyntax("option_shape_quote_unclosed")
                    end += 1
                return text[start : index + 1], end
            if depth < 0:
                break
    raise _UnsupportedShapeSyntax("option_shape_unbalanced_tuple")


def _option_shapes(text: str) -> dict[str, Shape]:
    labels = list(_SHAPE_LABEL.finditer(text))
    shared = list(_SHARED_SHAPE_LABEL.finditer(text))
    if shared:
        if labels or len(shared) != 1:
            raise _UnsupportedShapeSyntax("option_z_and_t_shapes_required")
        statement = shared[0]
        pair = [statement.group("first").strip("`").lower(),
                statement.group("second").strip("`").lower()]
        if pair != ["z", "t"]:
            raise _UnsupportedShapeSyntax("option_z_then_t_required")
        if not re.fullmatch(r"\s*Both calls are well-formed\s*;\s*", text[: statement.start()], re.I):
            raise _UnsupportedShapeSyntax("option_prefix_not_whitelisted")
        expression, end = _consume_shape(text, statement.end())
        if not re.fullmatch(r"\s*\.?\s*", text[end:]):
            raise _UnsupportedShapeSyntax("option_suffix_not_whitelisted")
        shape = _parse_option_shape(expression)
        return {"z": shape, "t": shape}
    if len(labels) != 2 or {match.group("label").strip("`").lower() for match in labels} != {"z", "t"}:
        raise _UnsupportedShapeSyntax("option_z_and_t_shapes_required")
    labels.sort(key=lambda match: match.start())
    normalized_labels = [match.group("label").strip("`").lower() for match in labels]
    if normalized_labels != ["z", "t"]:
        raise _UnsupportedShapeSyntax("option_z_then_t_required")
    if not re.fullmatch(r"\s*Both calls are well-formed\s*;\s*", text[: labels[0].start()], re.I):
        raise _UnsupportedShapeSyntax("option_prefix_not_whitelisted")
    parsed: dict[str, Shape] = {}
    ends: list[int] = []
    for label in labels:
        expression, end = _consume_shape(text, label.end())
        parsed[label.group("label").strip("`").lower()] = _parse_option_shape(expression)
        ends.append(end)
    if not re.fullmatch(r"\s+and\s+", text[ends[0] : labels[1].start()], re.I):
        raise _UnsupportedShapeSyntax("option_shape_separator_not_whitelisted")
    if not re.fullmatch(r"\s*\.?\s*", text[ends[1] :]):
        raise _UnsupportedShapeSyntax("option_suffix_not_whitelisted")
    return parsed


def _format_shape(shape: Shape) -> str:
    if isinstance(shape, tuple):
        return "(" + ",".join(_format_shape(item) for item in shape) + ")"
    return f"_{shape}"


def _answer_keys(question: Question) -> set[str]:
    values = [question.draft_target, question.internal_reference_answer]
    keys: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        match = re.match(r"^(?:option\s+|choice\s+|answer\s*(?:is|:|=)?\s*)?([A-D])(?=$|[.)\s:;])", text, re.I)
        if match is None:
            match = re.search(r"\b(?:option|choice|answer)\s*(?:is|:|=)?\s*([A-D])\b", text, re.I)
        if match:
            keys.add(match.group(1).upper())
    return keys


def check_cute_shapes(question: Question) -> dict[str, object]:
    """Check a whitelisted CuTe two-dimensional divide single-choice item."""
    text = question.question or ""
    in_scope = (
        question.answer_format == "single_choice"
        and re.search(r"\bzipped_divide\b", text)
        and re.search(r"\btiled_divide\b", text)
        and _SHAPE_ASK.search(text)
    )
    if not in_scope:
        return _result("not_applicable", "scope_not_matched")

    try:
        M, N, m, n = _source_dimensions(text)
    except _UnsupportedShapeSyntax as exc:
        return _result("unsupported", exc.reason)
    if M % m or N % n:
        return _result("contradiction", "shape_division_not_exact", M=M, N=N, m=m, n=n)

    options = list(_OPTION.finditer(text))
    if len(options) != 4 or [match.group(1) for match in options] != ["A", "B", "C", "D"]:
        return _result("unsupported", "options_A_through_D_required")
    parsed_options: dict[str, dict[str, Shape]] = {}
    try:
        for option in options:
            parsed_options[option.group(1)] = _option_shapes(option.group(2))
    except _UnsupportedShapeSyntax as exc:
        return _result("unsupported", exc.reason)

    expected_z = ((m, n), (M // m, N // n))
    expected_t = ((m, n), M // m, N // n)
    correct = [
        letter for letter, shapes in parsed_options.items()
        if shapes["z"] == expected_z and shapes["t"] == expected_t
    ]
    details = {
        "expected_z": _format_shape(expected_z),
        "expected_t": _format_shape(expected_t),
        "correct_options": correct,
    }
    if len(correct) > 1:
        return _result("contradiction", "multiple_correct_options", **details)
    if not correct:
        return _result("contradiction", "no_correct_option", **details)

    keys = _answer_keys(question)
    if not keys:
        return _result("unsupported", "answer_key_unparseable", **details)
    if len(keys) > 1:
        return _result("contradiction", "answer_key_fields_disagree", **details)
    key = next(iter(keys))
    details["answer_key"] = key
    if key != correct[0]:
        return _result("contradiction", "answer_key_mismatch", **details)
    return _result("pass", "unique_answer_key_matches", **details)
