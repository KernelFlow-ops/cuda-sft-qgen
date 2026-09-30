"""Bounded independent verification for generated CUDA questions.

Writer targets/specifications are hypotheses, never ground truth. Solver calls
see only the public stem. Deterministic tool failures reject only when they
prove a direct contradiction; uncertainty remains pending.
"""

from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import tempfile
from typing import Any, Iterable

from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.keys import PRI_CHAIR
from cuda_sft_qgen.models import Question, VerificationSpec
from cuda_sft_qgen.review.cute_shapes import check_cute_shapes

_ANSWER = re.compile(r"\\boxed\s*\{([^{}]*)\}|(?:answer|答案)\s*[:：=]\s*([^\n.;。；]+)", re.I)
_CODE = re.compile(r"```(?:cuda|cpp|c\+\+|c)?\s*\n?(.*?)```", re.I | re.S)
_MAX_OUTPUT = 4000


def _cfg(settings: Any) -> dict[str, Any]:
    value = getattr(settings, "review", settings if isinstance(settings, dict) else {}) or {}
    return value if isinstance(value, dict) else {}


def canonical_answer(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    match = _ANSWER.search(text)
    if match:
        text = (match.group(1) or match.group(2) or "").strip()
    text = re.sub(r"\\left|\\right|\\mathrm\s*", "", text)
    return re.sub(r"\s+", " ", text).strip("` \t\r\n.;,。；，")


def extract_answer(value: Any) -> str:
    return canonical_answer(value)


def _fraction_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return Fraction(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _fraction_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _fraction_node(node.left), _fraction_node(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right.numerator) <= 16:
            if max(left.numerator.bit_length(), left.denominator.bit_length()) * abs(right.numerator) > 4096:
                raise ValueError("numeric_power_budget_exceeded")
            return left ** right.numerator
    raise ValueError("unsafe_or_non_numeric_expression")


def canonical_numeric(value: Any) -> str:
    """Canonical arithmetic via safe AST and Fraction; never ``eval``."""
    text = canonical_answer(value).replace("^", "**")
    unit = ""
    unit_match = re.fullmatch(r"([0-9eE+*/().\-\s]+)\s*([A-Za-z][A-Za-z0-9_/*^-]{0,31})", text)
    if unit_match:
        text, unit = unit_match.group(1).strip(), unit_match.group(2)
        # SI prefix and B/b case carries meaning (MB != mB; byte != bit).
        # Preserve that distinction instead of creating a false majority.
        unit = {"byte": "B", "bytes": "B", "bit": "b", "bits": "b"}.get(unit, unit)
    if len(text) > 200:
        raise ValueError("numeric_answer_too_long")
    tree = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ValueError("numeric_expression_too_complex")
    result = _fraction_node(tree.body)
    number = str(result.numerator) if result.denominator == 1 else f"{result.numerator}/{result.denominator}"
    return f"{number} {unit}" if unit else number


def canonical_symbolic(value: Any) -> str:
    """Canonicalize only safe polynomial/rational AST nodes with SymPy."""
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sympy_unavailable") from exc
    text = canonical_answer(value).replace("^", "**")
    if len(text) > 400:
        raise ValueError("symbolic_answer_too_long")
    tree = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 45:
        raise ValueError("symbolic_expression_too_complex")
    if sum(isinstance(n, ast.Pow) for n in ast.walk(tree)) > 1:
        raise ValueError("nested_or_multiple_symbolic_powers_unsupported")
    denominators: set[str] = set()

    def build(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return sp.Integer(node.value)
        if isinstance(node, ast.Name) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", node.id):
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = build(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp):
            left, right = build(node.left), build(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div):
                if right.free_symbols:
                    denominators.add(str(right))
                if right == 0:
                    raise ValueError("zero_denominator")
                return left / right
            if isinstance(node.op, ast.Pow) and bool(right.is_Integer) and abs(int(right)) <= 6:
                if int(right) < 0 and left.free_symbols:
                    denominators.add(str(left))
                return left ** int(right)
        raise ValueError("unsafe_or_non_polynomial_expression")

    result = sp.cancel(build(tree.body))
    # x/x and 1 have different domains at x=0. Preserve denominators instead
    # of treating algebraic cancellation as proof of unconditional equivalence.
    domain = "|nonzero:" + ",".join(sorted(denominators)) if denominators else ""
    return str(result) + domain


def _canonical(value: Any, answer_type: str, mode: str) -> str:
    if answer_type == "number": return canonical_numeric(value)
    if answer_type == "expression" or mode == "symbolic": return canonical_symbolic(value)
    text = canonical_answer(value)
    return text.casefold() if answer_type == "enum" else text


def answer_consistency(target: Any, votes: Iterable[Any], *, minimum_agreement: float = 0.6) -> dict[str, Any]:
    normalized = [canonical_answer(v) for v in votes]
    normalized = [v for v in normalized if v]
    target_norm = canonical_answer(target)
    if not normalized or not target_norm:
        return {"status": "pending", "reason": "missing_target_or_votes", "votes": normalized}
    common = Counter(normalized).most_common(); majority, count = common[0]
    agreement = count / len(normalized)
    tie = len(common) > 1 and common[0][1] == common[1][1]
    status = "pending" if tie or agreement < minimum_agreement or majority != target_norm else "pass"
    return {"status": status, "target": target_norm, "majority": majority, "agreement": agreement,
            "votes": normalized, "reason": "majority_differs_from_target" if majority != target_norm else ""}


def _en(words: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b")


def _zh(words: list[str]) -> re.Pattern[str]:
    return re.compile("(?:" + "|".join(words) + ")")


_C_EN, _C_ZH = _en(["if", "unless", "only when", "must", "provided", "assuming"]), _zh(["否则", "若", "仅当", "不得", "必须", "如果"])
_D_EN, _D_ZH = _en(["after", "before", "then", "depends", "ordering", "synchronize", "wait"]), _zh(["先", "后", "依赖", "同步", "等待"])
_F_EN, _F_ZH = _en(["if", "else", "switch", "for", "while", "branch", "diverg"]), _zh(["条件", "分支", "循环", "发散"])


def complexity_features(question: Question | str) -> dict[str, int]:
    text = question.question if isinstance(question, Question) else str(question or "")
    lower = text.lower(); blocks = _CODE.findall(text)
    return {
        "num_constraints": len(_C_EN.findall(lower)) + len(_C_ZH.findall(text)),
        "dependency_depth": len(_D_EN.findall(lower)) + len(_D_ZH.findall(text)),
        "control_flow_depth": len(_F_EN.findall(lower)) + len(_F_ZH.findall(text)),
        "memory_scope_count": len({x for x in ("thread", "warp", "block", "cluster", "device", "stream", "host") if x in lower}),
        "architecture_specificity": len({x for x in ("ampere", "hopper", "blackwell", "sm80", "sm90", "sm100", "a100", "h100", "b200") if x in lower}),
        "code_size": len(blocks[0]) if blocks else 0,
        "required_validation": int(any(x in lower for x in ("compile", "run", "output", "racecheck", "nvcc", "编译", "运行", "输出"))),
    }


def difficulty_score(f: dict[str, int]) -> float:
    return float(f.get("num_constraints", 0) + 1.5 * f.get("dependency_depth", 0)
                 + f.get("control_flow_depth", 0) + f.get("memory_scope_count", 0)
                 + 2 * f.get("architecture_specificity", 0) + min(f.get("code_size", 0) / 120, 5)
                 + 2 * f.get("required_validation", 0))


def extract_cuda_code(text: str) -> str:
    match = _CODE.search(text or "")
    return match.group(1).strip() if match else ""


def _run(command: list[str], root: Path, timeout: float) -> dict[str, Any]:
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": "/tmp", "LANG": "C.UTF-8",
           "LC_ALL": "C.UTF-8", "CUDA_CACHE_DISABLE": "1"}

    # subprocess preexec_fn can deadlock when Chair calls this from its worker
    # pool. Apply limits via exec(prlimit), keeping the fork path Python-free.
    limiter = shutil.which("prlimit")
    if not limiter:
        return {"status": "pending", "reason": "resource_limiter_unavailable"}
    command = [limiter, "--fsize=16777216", "--nofile=128", f"--cpu={max(2, int(timeout) + 2)}", "--", *command]

    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, \
             tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            proc = subprocess.Popen(command, cwd=root, stdout=stdout_file, stderr=stderr_file,
                                    text=True, env=env, start_new_session=True)
            timed_out = False
            try:
                proc.wait(timeout=max(0.1, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True; os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            stdout_file.seek(0, os.SEEK_END); out_size = stdout_file.tell()
            stderr_file.seek(0, os.SEEK_END); err_size = stderr_file.tell()
            stdout_file.seek(max(0, out_size - _MAX_OUTPUT)); stdout = stdout_file.read(_MAX_OUTPUT)
            stderr_file.seek(max(0, err_size - _MAX_OUTPUT)); stderr = stderr_file.read(_MAX_OUTPUT)
            if out_size > _MAX_OUTPUT or err_size > _MAX_OUTPUT:
                return {"status": "pending", "reason": "tool_output_truncated", "returncode": proc.returncode}
            if timed_out:
                return {"status": "pending", "reason": "timeout", "returncode": None,
                        "stdout": stdout, "stderr": stderr}
    except OSError as exc:
        return {"status": "pending", "reason": f"tool_error:{type(exc).__name__}"}
    return {"status": "completed", "returncode": proc.returncode,
            "stdout": (stdout or "")[-_MAX_OUTPUT:], "stderr": (stderr or "")[-_MAX_OUTPUT:]}


def _sandbox(command: list[str], root: Path, gpu: bool) -> list[str] | None:
    bwrap = shutil.which("bwrap")
    if not bwrap: return None
    args = [bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--clearenv",
            "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "TMPDIR", "/tmp",
            "--tmpfs", "/tmp", "--bind", str(root), "/work", "--chdir", "/work"]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(path).exists(): args += ["--ro-bind", path, path]
    if Path("/etc/ld.so.cache").exists():
        args += ["--dir", "/etc", "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    args += ["--dev", "/dev"]
    if gpu:
        # Expose NVIDIA character devices only, not host disks or unrelated
        # devices. WSL exposes CUDA through /dev/dxg instead of /dev/nvidia*.
        # Bubblewrap's synthetic /dev already supplies null/zero/etc.
        devices = list(Path("/dev").glob("nvidia*"))
        if Path("/dev/dxg").exists(): devices.append(Path("/dev/dxg"))
        for device in sorted(devices):
            args += ["--dev-bind", str(device), str(device)]
    return args + ["--proc", "/proc"] + command


def _complete_tu(code: str) -> bool:
    return bool(re.search(r"^\s*#\s*include", code, re.M) and re.search(r"\b(?:int|auto)\s+main\s*\(", code))


def _arch(value: str) -> str | None:
    match = re.fullmatch(r"sm_?(\d{2,3}[a-z]?)", value.strip().lower())
    return f"sm_{match.group(1)}" if match else None


def compile_cuda(source: str, *, arch: str | None = None, timeout_seconds: float = 20,
                 nvcc: str | None = None) -> dict[str, Any]:
    """Legacy fenced-snippet compiler. It is never a negative oracle."""
    compiler = nvcc or shutil.which("nvcc")
    if not compiler: return {"status": "unavailable", "reason": "nvcc_not_found"}
    code = extract_cuda_code(source)
    if not code: return {"status": "unavailable", "reason": "no_cuda_code_block"}
    with tempfile.TemporaryDirectory(prefix="cuda-sft-compile-") as directory:
        root = Path(directory); (root / "candidate.cu").write_text(code, encoding="utf-8")
        cmd = [compiler, "-std=c++17", "-c", "candidate.cu", "-o", "candidate.o"]
        if arch: cmd.insert(1, f"-arch={arch}")
        result = _run(cmd, root, timeout_seconds)
        if result["status"] == "completed": result["status"] = "pass" if result["returncode"] == 0 else "fail"
        return result


def _compile(spec: VerificationSpec, cfg: dict[str, Any], executable: bool) -> tuple[dict[str, Any], Any]:
    arch = _arch(spec.architecture or "")
    compiler = shutil.which("nvcc")
    if not compiler: return {"status": "pending", "reason": "nvcc_not_found"}, None
    if not arch: return {"status": "pending", "reason": "invalid_architecture"}, None
    if not _complete_tu(spec.code or ""):
        return {"status": "pending", "reason": "incomplete_translation_unit"}, None
    if spec.expected_compile == "failure" and not (spec.expected_diagnostic or "").strip():
        return {"status": "pending", "reason": "negative_compile_requires_diagnostic"}, None
    if spec.toolkit_version:
        try:
            version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                                     timeout=3, check=False, env={"PATH": "/usr/bin:/bin"})
        except Exception:
            return {"status": "pending", "reason": "toolkit_version_unavailable"}, None
        if spec.toolkit_version not in (version.stdout + version.stderr):
            return {"status": "pending", "reason": "toolkit_version_mismatch"}, None
    temp = tempfile.TemporaryDirectory(prefix="cuda-sft-verify-"); root = Path(temp.name)
    (root / "candidate.cu").write_text(spec.code or "", encoding="utf-8")
    output = "candidate" if executable else "candidate.o"
    inner = [compiler, "-std=c++17", f"-arch={arch}"] + ([] if executable else ["-c"]) + ["candidate.cu", "-o", output]
    cmd = _sandbox(inner, root, False)
    if not cmd: temp.cleanup(); return {"status": "pending", "reason": "bwrap_not_found"}, None
    result = _run(cmd, root, float(cfg.get("compile_timeout_seconds", 20)))
    result.update({"architecture": arch, "workdir": str(root)})
    return result, temp


def _compile_oracle(spec: VerificationSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    result, temp = _compile(spec, cfg, False)
    try:
        if result.get("status") != "completed": return result
        stderr_lower = result.get("stderr", "").casefold()
        infrastructure = (
            "operation not permitted", "no such file or directory", "unsupported gpu architecture",
            "cannot find", "fatal error: cuda_runtime.h", "bwrap:", "internal compiler error",
        )
        if any(token in stderr_lower for token in infrastructure):
            return {**result, "status": "pending", "reason": "compile_environment_unsupported"}
        succeeded = result["returncode"] == 0; expected = spec.expected_compile == "success"
        if succeeded != expected: return {**result, "status": "fail", "reason": "compile_outcome_contradicts_spec"}
        if not succeeded and spec.expected_diagnostic and spec.expected_diagnostic.casefold() not in result.get("stderr", "").casefold():
            return {**result, "status": "pending", "reason": "expected_diagnostic_not_observed"}
        return {**result, "status": "pass", "reason": "compile_outcome_matches_spec"}
    finally:
        if temp: temp.cleanup()


def _gpu_arch() -> str | None:
    tool = shutil.which("nvidia-smi")
    if not tool: return None
    try:
        result = subprocess.run([tool, "--query-gpu=compute_cap", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=3, check=False,
                                env={"PATH": "/usr/bin:/bin"})
    except Exception: return None
    return "sm_" + result.stdout.splitlines()[0].strip().replace(".", "") if result.returncode == 0 and result.stdout.strip() else None


def _runtime_oracle(spec: VerificationSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    required, actual = _arch(spec.architecture or ""), _gpu_arch()
    if not actual or required != actual:
        return {"status": "pending", "reason": "gpu_architecture_unavailable", "required": required, "actual": actual}
    compiled, temp = _compile(spec, cfg, True)
    try:
        if compiled.get("status") != "completed": return compiled
        if compiled.get("returncode") != 0:
            return {**compiled, "status": "pending", "reason": "runtime_source_failed_to_compile"}
        root = Path(compiled["workdir"]); inner = ["./candidate"]
        if spec.sanitizer in {"memcheck", "racecheck"}:
            sanitizer = shutil.which("compute-sanitizer")
            if not sanitizer: return {"status": "pending", "reason": "compute_sanitizer_not_found"}
            inner = [sanitizer, "--tool", spec.sanitizer, "--log-file", "sanitizer.log", "./candidate"]
        cmd = _sandbox(inner, root, True)
        if not cmd: return {"status": "pending", "reason": "bwrap_not_found"}
        result = _run(cmd, root, float(cfg.get("execution_timeout_seconds", cfg.get("runtime_timeout_seconds", 15))))
        if result.get("status") != "completed":
            return {**result, "status": "pending", "reason": "runtime_timeout_not_proof"}
        if spec.expected_runtime_outcome == "sanitizer_error":
            log_path = root / "sanitizer.log"
            log = log_path.read_text(encoding="utf-8", errors="replace")[-_MAX_OUTPUT:] if log_path.exists() else ""
            # A named diagnostic is positive evidence of the specified defect.
            # No finding is inconclusive: testing cannot prove absence of races.
            diagnostic = (spec.expected_sanitizer_diagnostic or "").casefold()
            observed = diagnostic in log.casefold() and "error summary: 0 errors" not in log.casefold()
            return {**result, "status": "pass" if observed else "pending",
                    "reason": "expected_sanitizer_diagnostic_observed" if observed else "sanitizer_defect_not_observed",
                    "sanitizer_log": log}
        # A nonzero process exit can be CUDA driver initialization, allocation,
        # sandbox, or generated-code failure. Without a trusted harness those
        # causes cannot be distinguished, so it is never a question-level fail.
        if result.get("returncode") != 0:
            return {**result, "status": "pending", "reason": "runtime_nonzero_exit_inconclusive"}
        if result.get("stdout", "").strip() != (spec.expected_stdout or "").strip():
            return {**result, "status": "fail", "reason": "stdout_contradicts_spec"}
        if spec.sanitizer in {"memcheck", "racecheck"}:
            log_path = root / "sanitizer.log"
            sanitizer_log = log_path.read_text(encoding="utf-8", errors="replace")[-_MAX_OUTPUT:] if log_path.exists() else ""
            result["sanitizer_log"] = sanitizer_log
            if "error summary: 0 errors" not in sanitizer_log.casefold():
                return {**result, "status": "pending", "reason": "sanitizer_result_ambiguous"}
        return {**result, "status": "pass", "reason": "runtime_matches_spec"}
    finally:
        if temp: temp.cleanup()


def solver_call_params(difficulty: int, cfg: dict) -> dict:
    """Choose solver effort and a bounded output budget from difficulty."""
    level = cfg.get("solver_thinking_level_hard", "medium") if int(difficulty) >= 3 else cfg.get("solver_thinking_level", "low")
    return {"thinking_level": str(level), "max_tokens": min(int(cfg.get("solver_max_tokens", 3072)), 4096)}


def _independent_responses(count: int, call: Any, cfg: dict[str, Any]) -> list[Any]:
    """Run independent solver samples concurrently while preserving vote order."""
    if not bool(cfg.get("verification_parallel_enabled", False)) or count == 1:
        results = []
        for _ in range(count):
            try:
                results.append(call())
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results
    workers = max(1, min(count, int(cfg.get("verification_parallel_workers", count))))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="solver-vote") as executor:
        futures = [executor.submit(copy_context().run, call) for _ in range(count)]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results


def _blind_votes(q: Question, reviewer: Any, spec: VerificationSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    high_risk = spec.mode in {"scalar", "symbolic"}
    configured = cfg.get("answer_consistency_high_risk_k" if high_risk else "answer_consistency_k")
    configured = configured if configured is not None else cfg.get("verification_solver_k", 5 if high_risk else 3)
    budget = max(1, int(cfg.get("verification_max_calls", cfg.get("verification_max_llm_calls", 6))))
    k = max(3, min(int(configured), 5, budget))
    if budget < 3:
        return {"status": "pending", "reason": "verification_llm_budget_too_small", "budget": budget}
    system = ("Solve this CUDA question independently. Return exactly one JSON object "
              "with string fields answer and evidence, e.g. {\"answer\":\"1\",\"evidence\":\"reason\"}. "
              "Do not infer or request a writer answer. Put the final result only in answer; "
              "put the justification in evidence. For a single numeric question return only "
              "the number or reduced fraction, in the units requested by the question, "
              "not a sentence. For an expression use plain arithmetic syntax, not LaTeX. "
              "Do not invent missing premises or reduce a multi-part problem to one value.")
    # Deliberately excludes target, spec, rubric, parents, and provenance.
    public = {"question": q.question}
    def solve() -> dict[str, Any]:
        return reviewer.complete_json([{"role": "system", "content": system},
                                       {"role": "user", "content": json.dumps(public, ensure_ascii=False)}],
                                      retries=0, temperature=0.2,
                                      **solver_call_params(q.difficulty, cfg),
                                      priority=int(cfg.get("priority", cfg.get("verification_priority", PRI_CHAIR))))

    votes, proofs, errors = [], [], []
    for obj in _independent_responses(k, solve, cfg):
        try:
            if isinstance(obj, Exception):
                raise obj
            proof = str(obj.get("evidence") or "").strip()
            if obj.get("answer") is None or not proof: raise ValueError("incomplete_vote")
            votes.append(_canonical(obj["answer"], spec.answer_type, spec.mode)); proofs.append(proof[:400])
        except Exception as exc: errors.append(type(exc).__name__)
    if len(votes) != k:
        return {"status": "pending", "reason": "incomplete_valid_votes", "k": k,
                "valid_votes": len(votes), "errors": errors, "solver_evidence": proofs}
    common = Counter(votes).most_common(); majority, count = common[0]
    agreement = count / k
    if (len(common) > 1 and common[0][1] == common[1][1]) or agreement < float(cfg.get("answer_consistency_min_agreement", .6)):
        return {"status": "pending", "reason": "low_solver_agreement", "votes": votes,
                "agreement": agreement, "solver_evidence": proofs}
    try: target = _canonical(q.draft_target, spec.answer_type, spec.mode)
    except Exception as exc:
        return {"status": "pending", "reason": f"invalid_writer_target:{type(exc).__name__}",
                "majority": majority, "solver_evidence": proofs}
    if majority != target:
        return {"status": "pending", "reason": "majority_differs_from_writer_target",
                "majority": majority, "agreement": agreement, "votes": votes, "solver_evidence": proofs}
    return {"status": "pass", "reason": "independent_majority_matches_target", "majority": majority,
            "agreement": agreement, "votes": votes, "solver_evidence": proofs}


def _open_rip(q: Question, reviewer: Any, spec: VerificationSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    budget = max(1, int(cfg.get("verification_max_calls", cfg.get("verification_max_llm_calls", 6))))
    if budget < 4:
        return {"status": "pending", "reason": "verification_llm_budget_too_small", "budget": budget}
    if len([row for row in (spec.rubric or []) if str(row).strip()]) < 2:
        return {"status": "pending", "reason": "rubric_too_small"}
    public = {"question": q.question}
    def solve() -> dict[str, Any]:
        return reviewer.complete_json([{"role": "system", "content":
                                       'Answer the CUDA question independently. Return only {"response":"your answer"} as valid JSON.'},
                                       {"role": "user", "content": json.dumps(public, ensure_ascii=False)}],
                                      retries=0, temperature=.3,
                                      **solver_call_params(q.difficulty, cfg),
                                      priority=int(cfg.get("priority", cfg.get("verification_priority", PRI_CHAIR))))

    responses, errors = [], []
    for obj in _independent_responses(3, solve, cfg):
        try:
            if isinstance(obj, Exception):
                raise obj
            response = str(obj.get("response") or "").strip()
            if not response: raise ValueError("empty_response")
            responses.append(response)
        except Exception as exc: errors.append(type(exc).__name__)
    if len(responses) != 3:
        return {"status": "pending", "reason": "incomplete_open_responses", "errors": errors}
    payload = {**public, "rubric": spec.rubric, "responses": responses}
    try:
        obj = reviewer.complete_json([{"role": "system", "content":
                                      'Score each response from 0 to 1 against the rubric. Return only valid JSON: '
                                      '{"scores":[{"score":0.5,"evidence":"reason"},'
                                      '{"score":0.5,"evidence":"reason"},{"score":0.5,"evidence":"reason"}]}.'},
                                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                     retries=0, temperature=0,
                                     thinking_level=str(cfg.get("thinking_level", "medium")),
                                     max_tokens=min(int(cfg.get("solver_max_tokens", 3072)), 2048),
                                     priority=int(cfg.get("priority", cfg.get("verification_priority", PRI_CHAIR))))
        rows = obj.get("scores")
        if not isinstance(rows, list) or len(rows) != 3: raise ValueError("invalid_scores")
        scores, proofs = [], []
        for row in rows:
            score, proof = float(row["score"]), str(row.get("evidence") or "").strip()
            if not 0 <= score <= 1 or not proof: raise ValueError("invalid_score")
            scores.append(score); proofs.append(proof[:400])
    except Exception as exc:
        return {"status": "pending", "reason": f"rubric_judge_error:{type(exc).__name__}"}
    rip = min(scores)
    floor = float(cfg.get("open_min_score", 0.6))
    if rip < floor:
        return {"status": "pending", "reason": "open_rubric_below_min",
                "rip_score": rip, "scores": scores, "rubric_evidence": proofs}
    return {"status": "pass", "reason": "open_rip_score_valid",
            "rip_score": rip, "scores": scores, "rubric_evidence": proofs}


def evaluate_candidate(q: Question, reviewer: Any, settings: Any) -> dict[str, Any]:
    """Evaluate one candidate within K+1 LLM calls and bounded tool time."""
    cfg = _cfg(settings)
    shape_check = check_cute_shapes(q)
    shape_evidence = {"cute_shapes": shape_check} if shape_check["status"] == "pass" else {}
    if shape_check["status"] == "contradiction":
        return {"status": "fail", "reason": "cute_shape_contradiction",
                "evidence": {"cute_shapes": shape_check}}
    if shape_check["status"] == "unsupported":
        return {"status": "pending", "reason": "cute_shape_unsupported",
                "evidence": {"cute_shapes": shape_check}}
    enabled = bool(cfg.get("verification_enabled", cfg.get("answer_consistency_enabled", False)))
    if not enabled:
        return {"status": "disabled", "reason": "verification_feature_disabled", "evidence": shape_evidence}
    if isinstance(reviewer, FakeLLM):
        return {"status": "disabled", "reason": "synthetic_reviewer",
                "evidence": {"synthetic": True, **shape_evidence}}
    # Legacy candidates have no structured contract and must not add LLM calls.
    if not q.verification_spec and not q.draft_plan:
        return {"status": "disabled", "reason": "legacy_unstructured_candidate", "evidence": shape_evidence}
    if q.draft_validation_errors:
        return {"status": "pending", "reason": "draft_validation_errors",
                "evidence": {"errors": list(q.draft_validation_errors), **shape_evidence}}
    try: spec = VerificationSpec.model_validate(q.verification_spec)
    except Exception as exc:
        return {"status": "pending", "reason": f"invalid_verification_spec:{type(exc).__name__}",
                "evidence": shape_evidence}
    if spec.mode in {"compile", "runtime"} and not bool(cfg.get("execution_enabled", cfg.get("verification_execution_enabled", True))):
        # Disabling the runner cannot certify a question that explicitly
        # requires execution; it must remain queued for a capable environment.
        return {"status": "pending", "reason": "verification_execution_disabled", "evidence": shape_evidence}
    try:
        if spec.mode in {"scalar", "symbolic"}: result = _blind_votes(q, reviewer, spec, cfg)
        elif spec.mode == "open": result = _open_rip(q, reviewer, spec, cfg)
        elif spec.mode == "compile": result = _compile_oracle(spec, cfg)
        else: result = _runtime_oracle(spec, cfg)
    except Exception as exc:
        result = {"status": "pending", "reason": f"verifier_error:{type(exc).__name__}"}
    result.update(shape_evidence)
    status, reason = result.pop("status", "pending"), result.pop("reason", "inconclusive")
    return {"status": status, "reason": reason, "evidence": result}


def verify_candidate(question: Question, *, target: Any = None, votes: Iterable[Any] = ()) -> dict[str, Any]:
    f = complexity_features(question)
    evidence: dict[str, Any] = {"features": f, "difficulty_raw_score": difficulty_score(f)}
    vote_list = list(votes)
    if target is not None or vote_list: evidence["answer_consistency"] = answer_consistency(target, vote_list)
    return evidence
