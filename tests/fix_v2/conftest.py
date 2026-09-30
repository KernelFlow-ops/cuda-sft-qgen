"""tests/fix_v2 conftest (installed by T16 as tests/fix_v2/conftest.py).

Policy (fix/README.md): every model-dependent test uses the REAL LLM/embedding API configured in the repo .env.
No FakeLLM, no stub LLMs, no fake embedders. Tests that need a model carry @pytest.mark.llm; when .env is not usable
they FAIL (not skip). A session-wide request budget (QGEN_LLM_MAX_CALLS) guards against runaway cost.
The repo .env is loaded by tests/conftest.py only when the session selects llm tests; offline runs never read it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]      # tests/fix_v2/conftest.py -> repo root
FIX_DIR = REPO_ROOT / "cuda_qgen_fix"
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))                    # `import fixv2_helpers`

_USAGE: dict[str, dict] = {}
_TOTAL = {"requests": 0, "total_tokens": 0}


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def env_problems() -> list[str]:
    """Why the real-LLM environment is unusable (empty list = usable)."""
    problems = []
    if _truthy(os.getenv("FAKE_LLM")):
        problems.append("FAKE_LLM is enabled; real-LLM tests require FAKE_LLM=0")
    if os.getenv("EMBEDDING_PROVIDER", "nvidia").strip().lower() == "fake":
        problems.append("EMBEDDING_PROVIDER=fake is not allowed")
    from cuda_sft_qgen.config import load_settings
    try:
        settings = load_settings(fake_llm=False)
        problems.extend(f"missing {name}" for name in settings.missing_api_keys + settings.missing_api_fields)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def pytest_configure(config):
    for line in ("llm: calls the real LLM/embedding API configured in .env (fails if unusable)",
                 "eval: long evaluation runs used only by task T41",
                 "gpu: requires nvcc and an NVIDIA GPU"):
        config.addinivalue_line("markers", line)


def _llm_requests() -> tuple[int, int]:
    from cuda_sft_qgen.observability import get_metrics
    counters = get_metrics().snapshot().get("counters", {})
    req = sum(v for k, v in counters.items() if k.startswith("llm_requests"))
    tok = sum(v for k, v in counters.items() if k.startswith("llm_total_tokens"))
    return int(req), int(tok)


def pytest_runtest_setup(item):
    if "llm" not in item.keywords:
        return
    problems = env_problems()
    if problems:
        pytest.fail("REAL LLM REQUIRED — fix .env: " + "; ".join(problems), pytrace=False)
    cap = int(os.getenv("QGEN_LLM_MAX_CALLS", "600"))
    if _TOTAL["requests"] >= cap:
        pytest.fail(f"LLM request budget exhausted ({_TOTAL['requests']} >= QGEN_LLM_MAX_CALLS={cap})", pytrace=False)


@pytest.fixture(autouse=True)
def _track_llm_usage(request):
    if "llm" not in request.keywords:
        yield
        return
    before = _llm_requests()
    yield
    after = _llm_requests()
    used = {"requests": after[0] - before[0], "total_tokens": after[1] - before[1]}
    _USAGE[request.node.nodeid] = used
    _TOTAL["requests"] += used["requests"]
    _TOTAL["total_tokens"] += used["total_tokens"]


def pytest_sessionfinish(session, exitstatus):
    if not _USAGE:
        return
    out = eval_out_dir()
    (out / "llm_usage.json").write_text(json.dumps({"total": _TOTAL, "per_test": _USAGE}, indent=1), encoding="utf-8")


def eval_out_dir() -> Path:
    path = Path(os.getenv("QGEN_EVAL_OUT") or Path(tempfile.gettempdir()) / "qgen_eval")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fix_dir() -> Path:
    return FIX_DIR


@pytest.fixture()
def settings():
    """Settings from the environment and configs (.env only in llm sessions; no model call is made)."""
    from cuda_sft_qgen.config import load_settings
    return load_settings()


@pytest.fixture()
def real_runtime(tmp_path):
    """A real Runtime (real LLM pool + real embedder) on an isolated data_dir, installed in the context var."""
    from cuda_sft_qgen.config import load_settings
    from cuda_sft_qgen.llm import FakeLLM
    from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime
    s = load_settings(data_dir=tmp_path)
    assert not s.fake_llm, "settings.fake_llm must be False for real-LLM tests"
    rt = build_runtime(s)
    assert not isinstance(rt.llm, FakeLLM) and not isinstance(rt.review_llm or rt.llm, FakeLLM)
    token = set_runtime(rt)
    try:
        yield rt
    finally:
        reset_runtime(token)
        rt.close()


@pytest.fixture()
def eval_out() -> Path:
    return eval_out_dir()
