"""P0 LLM/config/test-hygiene regressions: opt-in sampling params, reasoning-budget
retry, review token budget, env-over-profile warning, and .env isolation."""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROLE_PREFIXES = ("WRITER_", "EVOLVER_", "REVIEW_", "SALVAGE_", "KNOWLEDGE_", "EMBEDDING_")
MESSAGES = [{"role": "user", "content": "Return JSON"}]


@pytest.fixture
def clean_env(monkeypatch):
    """No developer .env and no role/threshold variables leaking into load_settings()."""
    from cuda_sft_qgen import config

    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in tuple(config.os.environ):
        if name.startswith(ROLE_PREFIXES) or name in {"MAX_TOKENS", "TEMPERATURE", "TOP_P", "REVIEW_MAX_TOKENS",
                                                      "CUDA_SFT_PROFILE", "THINKING_LEVEL"}:
            monkeypatch.delenv(name)
    for env_name, _key in config._PROFILE_REVIEW_ENV:
        monkeypatch.delenv(env_name, raising=False)
    return config


def _openai_settings(config, monkeypatch, tmp_path, **env):
    monkeypatch.setenv("WRITER_PROVIDER", "openai")
    monkeypatch.setenv("WRITER_API_KEY", "test-key-not-real")
    monkeypatch.setenv("WRITER_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("WRITER_MODEL", "reasoning-model")
    monkeypatch.setenv("TEMPERATURE", "0.42")
    monkeypatch.setenv("TOP_P", "0.8")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return config.load_settings(fake_llm=True, data_dir=tmp_path)


# --- item 3: temperature/top_p for OpenAI-compatible providers are opt-in ------------------------

def test_openai_payload_omits_sampling_params_by_default(clean_env, monkeypatch, tmp_path):
    from cuda_sft_qgen.llm import OpenAILLM

    settings = _openai_settings(clean_env, monkeypatch, tmp_path)
    assert all(api.sampling_params is False for api in settings.agent_apis.values())
    payload = OpenAILLM(settings, None, settings.agent_apis["writer"])._payload(MESSAGES, "medium", None, None)
    assert "temperature" not in payload and "top_p" not in payload
    assert payload["reasoning_effort"] == "medium"


def test_openai_payload_sends_sampling_params_when_enabled(clean_env, monkeypatch, tmp_path):
    from cuda_sft_qgen.config_v2 import load_config_v2
    from cuda_sft_qgen.llm import NvidiaLLM, OpenAILLM

    settings = _openai_settings(clean_env, monkeypatch, tmp_path, WRITER_SAMPLING_PARAMS="1")
    llm = OpenAILLM(settings, None, settings.agent_apis["writer"])
    payload = llm._payload(MESSAGES, "low", None, 512)
    reference = NvidiaLLM(settings, None, settings.agent_apis["writer"])._payload(MESSAGES, "low", None, 512)
    assert (payload["temperature"], payload["top_p"]) == (reference["temperature"], reference["top_p"]) == (0.42, 0.8)
    assert llm._payload(MESSAGES, "low", 0.0, None)["temperature"] == 0.0
    assert load_config_v2(fake_llm=True, data_dir=tmp_path).agents["writer"].sampling_params is True


def test_sampling_params_inherit_from_writer_per_role(clean_env, monkeypatch, tmp_path):
    settings = _openai_settings(clean_env, monkeypatch, tmp_path, WRITER_SAMPLING_PARAMS="1",
                                REVIEW_SAMPLING_PARAMS="0", SALVAGE_SAMPLING_PARAMS="")
    apis = settings.agent_apis
    assert apis["writer"].sampling_params and apis["evolver"].sampling_params and apis["salvage"].sampling_params
    assert apis["review"].sampling_params is False
    assert "SAMPLING_PARAMS" in (ROOT / ".env.example").read_text(encoding="utf-8")


# --- item 4: reasoning-only replies are typed and retried once with a larger budget ---------------

class _Pool:
    n_keys = 1

    def __init__(self, client):
        self.client, self.acquired = client, 0

    def acquire(self, kind, priority=0):
        self.acquired += 1
        return SimpleNamespace(client=self.client, index=0, label="k0")

    def release(self, kind, slot):
        pass

    def note_success(self, kind):
        pass


def test_reasoning_only_stream_raises_typed_error_without_same_budget_retries(fake_settings):
    from cuda_sft_qgen.llm import NvidiaLLM, ReasoningBudgetExceeded

    chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason="length", delta=SimpleNamespace(content=None, reasoning_content="thinking..."))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter([chunk]))))
    pool = _Pool(client)
    with pytest.raises(ReasoningBudgetExceeded, match="reasoning without final content; increase token budget"):
        NvidiaLLM(fake_settings, pool).complete(MESSAGES)
    assert issubclass(ReasoningBudgetExceeded, ValueError)
    assert pool.acquired == 1


class _ScriptedWriter:
    """Raises ReasoningBudgetExceeded for the first ``failures`` calls, then answers."""

    def __init__(self, failures):
        self.failures, self.calls = failures, []

    def complete(self, messages, **kwargs):
        from cuda_sft_qgen.llm import ReasoningBudgetExceeded

        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise ReasoningBudgetExceeded("model returned reasoning without final content; increase token budget")
        return json.dumps({"question": "Why does a warp of 32 threads reading stride-12 floats from shared memory "
                                       "hit 4-way bank conflicts on 32 banks?",
                           "tests_claim": "Stride 12 over 32 banks gives gcd(12, 32) = 4-way conflicts."})


@pytest.fixture
def writer_runtime(fake_runtime, monkeypatch):
    from cuda_sft_qgen.runtime import reset_runtime, set_runtime

    monkeypatch.setenv("WRITER_CONSISTENCY_GATE", "0")
    fake_runtime.prompt_builder = SimpleNamespace(build=lambda *a, **k: SimpleNamespace(
        system="system", user="user", cache_key="", generation_provenance={}))
    token = set_runtime(fake_runtime)
    yield fake_runtime
    reset_runtime(token)


@pytest.mark.parametrize("max_tokens,level,expected_tokens,expected_level", [
    ("4096", "high", 8192, "medium"), ("20000", "medium", 32768, "low"), ("4096", "low", 8192, "low"),
])
def test_reasoning_budget_retry_doubles_tokens_and_steps_thinking_down(
        writer_runtime, sample_brief, monkeypatch, max_tokens, level, expected_tokens, expected_level):
    from cuda_sft_qgen.graph.nodes.generate_core import _safe_generate
    from cuda_sft_qgen.models import SteeringBrief

    monkeypatch.setenv("MAX_TOKENS", max_tokens)
    monkeypatch.setenv("WRITER_THINKING_LEVEL", level)
    writer = writer_runtime.llm = _ScriptedWriter(failures=1)
    outcome = _safe_generate(sample_brief().model_dump(), SteeringBrief())
    assert outcome.status == "ok" and outcome.question is not None
    assert [call.get("max_tokens") for call in writer.calls] == [None, expected_tokens]
    assert [call["thinking_level"] for call in writer.calls] == [level, expected_level]


def test_repeated_reasoning_budget_failure_goes_to_retry_queue(writer_runtime, sample_brief):
    import threading

    from cuda_sft_qgen.graph.nodes.generate_core import apply_outcome, generate_or_salvage
    from cuda_sft_qgen.models import SteeringBrief

    writer = writer_runtime.llm = _ScriptedWriter(failures=99)
    brief = sample_brief(task_id="reasoning-task").model_dump()
    outcome = generate_or_salvage(brief, SteeringBrief())
    assert (outcome.status, outcome.reason) == ("retry", "reasoning_budget")
    kind = apply_outcome(outcome, pending=[], salvage_items=[], lock=threading.Lock())
    assert kind == "retry" and len(writer.calls) == 2
    queued = writer_runtime.store.load_retry_queue()
    assert [item["reason"] for item in queued] == ["reasoning_budget"]
    assert writer_runtime.store.load_rejected() == []


# --- item 5: the Chair gets a budget that survives thinking ----------------------------------------

@pytest.mark.parametrize("profile", ["default", "dev", "prod", "test"])
def test_review_max_tokens_default_is_8192(clean_env, tmp_path, profile):
    assert clean_env.load_settings(fake_llm=True, data_dir=tmp_path, profile=profile).review["max_tokens"] == 8192


# --- item 25: env values that mask a profile's review threshold are logged ------------------------

def test_env_override_of_profile_threshold_warns(clean_env, monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("SEMANTIC_DUP_THRESHOLD", "0.5")      # prod sets 0.95
    monkeypatch.setenv("SIMHASH_GRAY_HAMMING", "7")          # prod does not set this key
    monkeypatch.setenv("SIMHASH_HAMMING_THRESHOLD", "2")     # equal to prod's value: nothing is masked
    with caplog.at_level(logging.WARNING, logger="cuda_sft_qgen.config"):
        settings = clean_env.load_settings(fake_llm=True, data_dir=tmp_path, profile="prod")
    assert settings.semantic_dup_threshold == 0.5            # precedence unchanged: environment wins
    messages = [r.getMessage() for r in caplog.records if "config_env_overrides_profile" in r.getMessage()]
    assert messages == ["config_env_overrides_profile key=review.semantic_dup_threshold "
                        "env=SEMANTIC_DUP_THRESHOLD profile=prod"]


def test_default_profile_env_threshold_is_silent(clean_env, monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("SEMANTIC_DUP_THRESHOLD", "0.5")
    with caplog.at_level(logging.WARNING, logger="cuda_sft_qgen.config"):
        clean_env.load_settings(fake_llm=True, data_dir=tmp_path)
    assert "config_env_overrides_profile" not in caplog.text


# --- item 26: offline sessions never read the developer .env ---------------------------------------

def _root_conftest(request):
    target = str(ROOT / "tests" / "conftest.py")
    return next(p for p in request.config.pluginmanager.get_plugins() if getattr(p, "__file__", None) == target)


def test_dotenv_policy_depends_on_selected_llm_tests(request, monkeypatch):
    from cuda_sft_qgen import config

    root_conftest = _root_conftest(request)
    loaded = []
    monkeypatch.setattr(root_conftest, "load_dotenv", lambda path, override=False: loaded.append((path, override)))
    monkeypatch.setattr(config, "load_dotenv", config.load_dotenv)       # restored after the test
    offline = SimpleNamespace(items=[SimpleNamespace(get_closest_marker=lambda name: None)])
    root_conftest.pytest_collection_finish(offline)
    assert loaded == [] and config.load_dotenv("ignored") is False
    real = SimpleNamespace(items=[SimpleNamespace(get_closest_marker=lambda name: object() if name == "llm" else None)])
    root_conftest.pytest_collection_finish(real)
    assert loaded == [(ROOT / ".env", False)]


def test_offline_session_does_not_read_dotenv(request):
    import dotenv

    from cuda_sft_qgen import config

    if any(item.get_closest_marker("llm") for item in request.session.items):
        pytest.skip("this session selected real-LLM tests, which load .env by design")
    assert config.load_dotenv is not dotenv.load_dotenv


def test_fix_v2_conftest_has_no_import_time_dotenv():
    tree = ast.parse((ROOT / "tests" / "fix_v2" / "conftest.py").read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    assert not any(isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_dotenv"
                   for stmt in top_level for node in ast.walk(stmt))
