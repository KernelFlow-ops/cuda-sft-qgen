"""Canonical settings loader: CLI args > environment > profile YAML > defaults."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from cuda_sft_qgen.keys import parse_api_keys

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Walk parents from this file until ``pyproject.toml`` is found."""
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse env-style booleans (1/true/yes/on); None keeps ``default``."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    """Parse an int env value; empty/None keeps ``default``."""
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_float(value: str | None, default: float) -> float:
    """Parse a float env value; empty/None keeps ``default``."""
    if value is None or value.strip() == "":
        return default
    return float(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a mapping YAML file; missing files yield ``{}``."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings; override scalars replace, nested dicts recurse."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@dataclass(frozen=True)
class AgentAPI:
    provider: str
    base_url: str
    model: str
    api_keys: list[str] = field(repr=False)
    # OpenAI-compatible reasoning models often reject temperature/top_p, so they
    # are sent only when <ROLE>_SAMPLING_PARAMS opts in (NVIDIA always sends them).
    sampling_params: bool = False


AGENT_ROLES = ("writer", "evolver", "review", "salvage", "knowledge")


def _agent_api(role: str, writer: AgentAPI | None = None, *, fake_llm: bool = False) -> AgentAPI:
    prefix = role.upper()
    provider_value = (os.getenv(f"{prefix}_PROVIDER") or "").strip().lower()
    provider = provider_value or (writer.provider if writer else "nvidia")
    if provider not in {"nvidia", "openai"}:
        raise ValueError(f"{prefix}_PROVIDER must be nvidia|openai")
    base_value = (os.getenv(f"{prefix}_BASE_URL") or "").strip().rstrip("/")
    base_url = base_value or (writer.base_url if writer else "")
    model = (os.getenv(f"{prefix}_MODEL") or "").strip() or (
        writer.model if writer else "nvidia/nemotron-3-ultra-550b-a55b"
    )
    keys = parse_api_keys(prefix=prefix)
    changed_connection = writer is not None and (
        provider != writer.provider or (base_value and base_value != writer.base_url)
    )
    if changed_connection and not fake_llm and (not base_value or not keys):
        raise ValueError(f"{prefix}_BASE_URL and {prefix}_API_KEY are required when changing provider or base URL")
    sampling_value = (os.getenv(f"{prefix}_SAMPLING_PARAMS") or "").strip()
    sampling_params = (_as_bool(sampling_value) if sampling_value
                       else (writer.sampling_params if writer else False))
    return AgentAPI(provider, base_url, model, keys or (writer.api_keys if writer else []), sampling_params)


@dataclass
class Settings:
    """Resolved runtime configuration for one pipeline process.

    Path properties join YAML-relative paths against :attr:`root`. Generation
    and review knobs also accept environment overrides at load time.
    """

    root: Path
    agent_apis: dict[str, AgentAPI] = field(repr=False)
    thinking_level: str
    embedding_provider: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_keys: list[str] = field(repr=False)
    review_batch_size: int
    target_question_count: int
    writer_concurrency: int
    llm_per_key_concurrency: int
    embed_per_key_concurrency: int
    embed_oversub: int
    embed_chunk_size: int
    prompt_cache_enabled: bool
    embedding_cache_enabled: bool
    max_iterations: int
    semantic_dup_threshold: float
    semantic_gray_threshold: float
    template_jaccard_threshold: float
    template_gray_jaccard: float
    simhash_hamming_threshold: int
    simhash_gray_hamming: int
    review_llm_every_k_batches: int
    sieve_use_llm: bool
    log_reasoning: bool
    fake_llm: bool
    salvage_enabled: bool
    salvage_llm_attempts: int
    salvage_thinking_level: str
    salvage_max_tokens: int
    salvage_use_llm: bool
    retry_drain_enabled: bool
    retry_max_attempts: int
    retry_base_cooldown_sec: float
    retry_max_cooldown_sec: float
    retry_drain_batch: int
    llm_rate_limit_retries_per_call: int
    json_keep_truncated: bool
    question_min_chars: int
    question_max_chars: int
    default_yaml: dict[str, Any] = field(default_factory=dict)
    quotas: dict[str, Any] = field(default_factory=dict)
    personas: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    data_dir_override: Path | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)
    profile: str = "default"

    @property
    def data_dir(self) -> Path:
        """Archive directory; CLI/env override wins over YAML ``paths.data_dir``."""
        if self.data_dir_override is not None:
            return self.data_dir_override
        rel = self.default_yaml.get("paths", {}).get("data_dir", "data")
        return (self.root / rel).resolve()

    @property
    def taxonomy_path(self) -> Path:
        """Resolved path to the CUDA taxonomy YAML."""
        rel = self.default_yaml.get("paths", {}).get(
            "taxonomy", "configs/taxonomy/cuda.yaml"
        )
        return (self.root / rel).resolve()

    @property
    def quotas_path(self) -> Path:
        """Resolved path to type/difficulty quota YAML."""
        return (self.root / self.default_yaml.get("paths", {}).get("quotas", "configs/quotas.yaml")).resolve()

    @property
    def personas_path(self) -> Path:
        """Resolved path to writer persona YAML."""
        return (self.root / self.default_yaml.get("paths", {}).get("personas", "configs/personas.yaml")).resolve()

    @property
    def scope_path(self) -> Path:
        """Resolved path to domain scope and priors."""
        return (self.root / self.default_yaml.get("paths", {}).get("scope", "configs/scope.yaml")).resolve()

    @property
    def compat_matrix_path(self) -> Path:
        """Resolved path to axis compatibility rules."""
        return (self.root / self.default_yaml.get("paths", {}).get("compat_matrix", "configs/compat_matrix.yaml")).resolve()

    @property
    def angles_path(self) -> Path:
        """Resolved path to angle cards."""
        return (self.root / self.default_yaml.get("paths", {}).get("angles", "configs/angles.yaml")).resolve()

    @property
    def hardware_facts_path(self) -> Path:
        """Resolved path to verified hardware limits."""
        return (self.root / self.default_yaml.get("paths", {}).get("hardware_facts", "configs/hardware_facts.yaml")).resolve()

    @property
    def planner_cfg(self) -> dict[str, Any]:
        """Joint planner switches from merged YAML."""
        return self.default_yaml.get("planner", {})

    @property
    def generation(self) -> dict[str, Any]:
        """Raw ``generation`` mapping from merged YAML."""
        return self.default_yaml.get("generation", {})

    @property
    def review(self) -> dict[str, Any]:
        """Raw ``review`` mapping from merged YAML."""
        return self.default_yaml.get("review", {})

    @property
    def review_model(self) -> str:
        """Chair model id; defaults to the writer model."""
        return self.agent_apis["review"].model

    @property
    def llm_provider(self) -> str:
        return self.agent_apis["writer"].provider

    @property
    def llm_api_keys(self) -> list[str]:
        return self.agent_apis["writer"].api_keys

    @property
    def llm_base_url(self) -> str:
        return self.agent_apis["writer"].base_url

    @property
    def llm_model(self) -> str:
        return self.agent_apis["writer"].model

    @property
    def missing_api_keys(self) -> list[str]:
        if self.fake_llm:
            return []
        missing = ["WRITER_API_KEY"] if not self.llm_api_keys else []
        writer = self.agent_apis["writer"]
        missing.extend(f"{role.upper()}_API_KEY" for role, api in self.agent_apis.items()
                       if role != "writer" and not api.api_keys
                       and (api.provider, api.base_url) != (writer.provider, writer.base_url))
        if self.embedding_provider == "nvidia" and not self.embedding_api_keys:
            missing.append("EMBEDDING_API_KEY")
        return missing

    @property
    def missing_api_fields(self) -> list[str]:
        if self.fake_llm:
            return []
        missing = [f"{role.upper()}_{name}" for role, api in self.agent_apis.items()
                   for name, value in (("BASE_URL", api.base_url), ("MODEL", api.model)) if not value]
        if self.embedding_provider == "nvidia":
            missing.extend(name for name, value in (("EMBEDDING_BASE_URL", self.embedding_base_url),
                                                    ("EMBEDDING_MODEL", self.embedding_model)) if not value)
        return missing

    @property
    def knowledge_cfg(self) -> dict[str, Any]:
        """Knowledge-store worker settings from YAML."""
        return self.default_yaml.get("knowledge", {})

    @property
    def taxonomy_manager_cfg(self) -> dict[str, Any]:
        """Taxonomy manager cadence and expansion caps."""
        return (self.default_yaml.get("taxonomy") or {}).get("manager") or {}

    @property
    def evolver_fraction(self) -> float:
        """Share of briefs assigned to Evolver rather than Writer."""
        return float(self.generation.get("evolver_fraction", 0.30))

    @property
    def temperature(self) -> float:
        """Writer sampling temperature from merged generation YAML."""
        return float(self.generation.get("temperature", 1.0))

    @property
    def top_p(self) -> float:
        """Writer nucleus sampling ``top_p``."""
        return float(self.generation.get("top_p", 0.95))

    @property
    def max_tokens(self) -> int:
        """Completion budget; ``MAX_TOKENS`` env overrides YAML."""
        return _as_int(os.getenv("MAX_TOKENS"), int(self.generation.get("max_tokens", 4096)))

    @property
    def writer_thinking_level(self) -> str:
        """Writer thinking level; ``WRITER_THINKING_LEVEL`` overrides YAML."""
        env = os.getenv("WRITER_THINKING_LEVEL")
        if env and env.strip():
            return env.strip().lower()
        raw = self.generation.get("writer_thinking_level")
        return str(raw or self.thinking_level).strip().lower()

    @property
    def evolver_thinking_level(self) -> str:
        """Evolver thinking level; defaults to the Writer level."""
        env = os.getenv("EVOLVER_THINKING_LEVEL")
        return env.strip().lower() if env and env.strip() else self.writer_thinking_level

    @property
    def min_generate_batch(self) -> int:
        """Smallest generate wave; leftover holes smaller than this flush review."""
        env = os.getenv("MIN_GENERATE_BATCH")
        if env and env.strip():
            return max(1, int(env))
        raw = self.generation.get("min_generate_batch")
        if raw is None or raw == "":
            return max(1, self.writer_concurrency)
        return max(1, int(raw))

    @property
    def salvage_inline(self) -> bool:
        """If True, salvage runs inside generate instead of a dedicated node."""
        return _as_bool(
            os.getenv("SALVAGE_INLINE"),
            bool(self.generation.get("salvage_inline", False)),
        )

    @property
    def sieve_min_chars(self) -> int:
        """Alias of :attr:`question_min_chars` for older call sites."""
        return int(self.question_min_chars)

    @property
    def max_parse_retries(self) -> int:
        """Extra writer attempts after a local JSON parse failure."""
        return int(self.generation.get("max_parse_retries", 1))


# Review thresholds a profile may tighten (prod) and a developer .env may set.
_PROFILE_REVIEW_ENV = (
    ("SEMANTIC_DUP_THRESHOLD", "semantic_dup_threshold"),
    ("SEMANTIC_GRAY_THRESHOLD", "semantic_gray_threshold"),
    ("TEMPLATE_JACCARD_THRESHOLD", "template_jaccard_threshold"),
    ("TEMPLATE_GRAY_JACCARD", "template_gray_jaccard"),
    ("SIMHASH_HAMMING_THRESHOLD", "simhash_hamming_threshold"),
    ("SIMHASH_GRAY_HAMMING", "simhash_gray_hamming"),
)


def _warn_env_overrides_profile(profile_name: str, profile_review: dict[str, Any]) -> None:
    """Warn when an environment variable replaces a review threshold the profile set.

    Precedence is unchanged (environment wins); the warning only makes a masked
    profile value, e.g. prod's stricter thresholds under a developer .env, visible.
    """
    for env_name, key in _PROFILE_REVIEW_ENV:
        raw = (os.getenv(env_name) or "").strip()
        if not raw or key not in profile_review:
            continue
        try:
            differs = float(raw) != float(profile_review[key])
        except (TypeError, ValueError):
            differs = True
        if differs:
            logger.warning("config_env_overrides_profile key=review.%s env=%s profile=%s",
                           key, env_name, profile_name)


def _resolve_data_dir(data_dir: Path | None) -> Path | None:
    """CLI data_dir wins, then CUDA_SFT_DATA_DIR, else None (use YAML default)."""
    if data_dir is not None:
        return Path(data_dir)
    env = os.getenv("CUDA_SFT_DATA_DIR")
    if env and env.strip():
        return Path(env.strip())
    return None


def load_settings(
    *,
    fake_llm: bool | None = None,
    target: int | None = None,
    review_batch_size: int | None = None,
    env_file: Path | None = None,
    data_dir: Path | None = None,
    profile: str | None = None,
) -> Settings:
    """Canonical configuration loader: CLI > environment > profile > defaults."""
    root = find_project_root()
    load_dotenv(env_file or (root / ".env"), override=False)

    default_yaml = _load_yaml(root / "configs" / "default.yaml")
    profile_name = profile or os.getenv("CUDA_SFT_PROFILE", "default")
    profile_path = root / "configs" / "profiles" / f"{profile_name}.yaml"
    if profile_name != "default" and not profile_path.is_file():
        raise ValueError(f"Unknown configuration profile: {profile_name}")
    profile_yaml = _load_yaml(profile_path)
    default_yaml = _merge_config(default_yaml, profile_yaml)
    paths = default_yaml.get("paths", {})
    quotas = _load_yaml(root / paths.get("quotas", "configs/quotas.yaml"))
    personas_raw = _load_yaml(root / paths.get("personas", "configs/personas.yaml"))
    scope = _load_yaml(root / paths.get("scope", "configs/scope.yaml"))
    compat = _load_yaml(root / paths.get("compat_matrix", "configs/compat_matrix.yaml"))
    planner_yaml = _merge_config(_load_yaml(root / "configs" / "planner.yaml"), {
        "adaptive_sampling": default_yaml.get("adaptive_sampling", {})
    })
    review_cfg = default_yaml.setdefault("review", {})
    review_cfg["thinking_level"] = os.getenv("REVIEW_THINKING_LEVEL", str(review_cfg.get("thinking_level", "medium"))).strip().lower()
    review_cfg["max_tokens"] = _as_int(os.getenv("REVIEW_MAX_TOKENS"), int(review_cfg.get("max_tokens", 8192)))
    gen_cfg = default_yaml.setdefault("generation", {})
    planner_cfg = default_yaml.setdefault("planner", {})
    for env_name, key in (("TEMPERATURE", "temperature"), ("TOP_P", "top_p")):
        if os.getenv(env_name):
            gen_cfg[key] = float(os.environ[env_name])
    gen_cfg["writer_gate_max_tokens"] = _as_int(
        os.getenv("WRITER_GATE_MAX_TOKENS"),
        int(gen_cfg.get("writer_gate_max_tokens", 512)),
    )
    if gen_cfg["writer_gate_max_tokens"] < 1:
        raise ValueError("writer_gate_max_tokens must be positive")
    gen_cfg["submit_stagger_sec"] = _as_float(
        os.getenv("WRITER_SUBMIT_STAGGER_SEC"),
        float(gen_cfg.get("submit_stagger_sec", 0.35)),
    )
    if gen_cfg["submit_stagger_sec"] < 0:
        raise ValueError("writer_submit_stagger_sec must be nonnegative")
    gen_cfg["repair_concurrency"] = _as_int(
        os.getenv("REPAIR_CONCURRENCY"), int(gen_cfg.get("repair_concurrency", 2))
    )
    if gen_cfg["repair_concurrency"] < 1:
        raise ValueError("repair_concurrency must be positive")
    gen_cfg["review_repair_enabled"] = _as_bool(
        os.getenv("REVIEW_REPAIR_ENABLED"), bool(gen_cfg.get("review_repair_enabled", True))
    )
    gen_cfg["review_repair_max_attempts"] = _as_int(
        os.getenv("REVIEW_REPAIR_MAX_ATTEMPTS"), int(gen_cfg.get("review_repair_max_attempts", 1))
    )
    if gen_cfg["review_repair_max_attempts"] < 0:
        raise ValueError("review_repair_max_attempts must be nonnegative")
    for env_name, key, fallback in (
        ("CHAIR_MAX_WORKERS", "chair_max_workers", 4),
        ("CHAIR_TRIGGER_COUNT", "chair_trigger_count", 1),
        ("REVIEW_LLM_PER_KEY_CONCURRENCY", "llm_per_key_concurrency", 1),
    ):
        review_cfg[key] = _as_int(os.getenv(env_name), int(review_cfg.get(key, fallback)))
        if review_cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if os.getenv("REVIEW_MODEL"):
        review_cfg["model"] = os.environ["REVIEW_MODEL"]
    knowledge_cfg = _merge_config({
        "enabled": True, "audit_interval_seconds": 86400,
        "audit_batch_size": 100, "worker_poll_seconds": 5,
        "official_sources": [],
        "docs_dir": "data/docs", "toolkit_version": "12.x",
        "legacy_batch_per_l2": 20, "legacy_due_days": 14,
    }, default_yaml.get("knowledge", {}))
    knowledge_cfg["enabled"] = _as_bool(os.getenv("KNOWLEDGE_ENABLED"), knowledge_cfg["enabled"])
    for key in ("audit_interval_seconds", "audit_batch_size", "worker_poll_seconds"):
        knowledge_cfg[key] = _as_int(os.getenv(f"KNOWLEDGE_{key.upper()}"), int(knowledge_cfg[key]))
        if knowledge_cfg[key] < 1:
            raise ValueError(f"knowledge.{key} must be positive")
    default_yaml["knowledge"] = knowledge_cfg

    # One canonical precedence path for new agent controls: explicit environment
    # overrides profile YAML which overrides repository defaults. Nodes should
    # consume resolved settings, not re-read environment in worker threads.
    mixup_cfg = default_yaml.setdefault("mixup", {})
    for env, mapping, key, fallback in (
        ("STRUCTURED_COT", gen_cfg, "prompt_cot", True),
        ("WRITER_GATE_LLM", gen_cfg, "writer_gate_llm", False),
        ("ANSWER_CONSISTENCY_ENABLED", review_cfg, "answer_consistency_enabled", True),
        ("PLANNER_JOINT_SAMPLING", planner_cfg, "joint_sampling", True),
        ("CUDA_EXECUTION_ENABLED", review_cfg, "execution_enabled", True),
        ("CUDA_MIXUP_ENABLED", mixup_cfg, "enabled", False),
        ("EXTERNAL_INVENTORY_ENABLED", knowledge_cfg, "external_inventory_enabled", False),
    ):
        mapping[key] = _as_bool(os.getenv(env), bool(mapping.get(key, fallback)))
    for env, mapping, key, fallback in (
        ("SOLVER_K", review_cfg, "answer_consistency_k", 3),
        ("SOLVER_HIGH_RISK_K", review_cfg, "answer_consistency_high_risk_k", 5),
        ("VERIFIER_MAX_CALLS", review_cfg, "verification_max_calls", 6),
        ("VERIFIER_RUN_MAX_CALLS", review_cfg, "verification_run_max_calls", 600),
        ("SOLVER_MAX_TOKENS", review_cfg, "solver_max_tokens", 3072),
        ("CUDA_VERIFY_TIMEOUT", review_cfg, "execution_timeout_seconds", 20),
        ("LLM_REQUEST_TIMEOUT", gen_cfg, "request_timeout_seconds", 90),
        ("LLM_MAX_RETRIES", gen_cfg, "request_max_attempts", 3),
    ):
        mapping[key] = _as_int(os.getenv(env), int(mapping.get(key, fallback)))
        if mapping[key] < 1:
            raise ValueError(f"{key} must be positive")
    review_cfg["solver_thinking_level"] = os.getenv("SOLVER_THINKING_LEVEL", str(review_cfg.get("solver_thinking_level", "low")))
    review_cfg["solver_thinking_level_hard"] = os.getenv("SOLVER_THINKING_LEVEL_HARD", str(review_cfg.get("solver_thinking_level_hard", "medium")))
    planner_yaml["mixup"] = dict(mixup_cfg)

    thinking = os.getenv("THINKING_LEVEL", "medium").strip().lower()
    if thinking not in {"none", "low", "low_effort", "medium", "high"}:
        raise ValueError(
            f"THINKING_LEVEL must be none|low|medium|high, got {thinking}"
        )

    provider = os.getenv("EMBEDDING_PROVIDER", "nvidia").strip().lower()
    if provider not in {"nvidia", "huggingface", "fake"}:
        raise ValueError("EMBEDDING_PROVIDER must be nvidia|huggingface|fake")

    is_fake = _as_bool(os.getenv("FAKE_LLM"), False) if fake_llm is None else fake_llm
    writer_api = _agent_api("writer", fake_llm=is_fake)
    agent_apis = {"writer": writer_api}
    for role in AGENT_ROLES[1:]:
        agent_apis[role] = _agent_api(role, writer_api, fake_llm=is_fake)
    review_cfg["model"] = agent_apis["review"].model
    embedding_keys = parse_api_keys(prefix="EMBEDDING")
    embedding_base_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip().rstrip("/")
    llm_per_key = _as_int(
        os.getenv("LLM_PER_KEY_CONCURRENCY"),
        int(gen_cfg.get("llm_per_key_concurrency", 2)),
    )
    embed_per_key = _as_int(
        os.getenv("EMBED_PER_KEY_CONCURRENCY"),
        int(gen_cfg.get("embed_per_key_concurrency", 2)),
    )
    embed_oversub = _as_int(
        os.getenv("EMBED_OVERSUB"),
        int(gen_cfg.get("embed_oversub", 2)),
    )
    n_keys = max(1, len(writer_api.api_keys))
    writer_concurrency = max(1, _as_int(os.getenv("WRITER_CONCURRENCY"),
        int(gen_cfg.get("writer_concurrency", n_keys * max(1, llm_per_key)))))
    _warn_env_overrides_profile(profile_name, profile_yaml.get("review") or {})

    settings = Settings(
        root=root,
        agent_apis=agent_apis,
        thinking_level=thinking,
        embedding_provider=provider,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b"
        ),
        embedding_base_url=embedding_base_url,
        embedding_api_keys=embedding_keys,
        review_batch_size=review_batch_size
        if review_batch_size is not None
        else _as_int(
            os.getenv("REVIEW_BATCH_SIZE"),
            int(review_cfg.get("batch_size", 20)),
        ),
        target_question_count=target
        if target is not None
        else _as_int(os.getenv("TARGET_QUESTION_COUNT"), 1000),
        writer_concurrency=writer_concurrency,
        llm_per_key_concurrency=max(1, llm_per_key),
        embed_per_key_concurrency=max(1, embed_per_key),
        embed_oversub=max(1, embed_oversub),
        embed_chunk_size=_as_int(
            os.getenv("EMBED_CHUNK_SIZE"),
            int(gen_cfg.get("embed_chunk_size", 8)),
        ),
        prompt_cache_enabled=_as_bool(
            os.getenv("PROMPT_CACHE"),
            bool(gen_cfg.get("prompt_cache", True)),
        ),
        embedding_cache_enabled=_as_bool(
            os.getenv("EMBEDDING_CACHE"),
            bool(gen_cfg.get("embedding_cache", True)),
        ),
        max_iterations=_as_int(os.getenv("MAX_ITERATIONS"), 200),
        semantic_dup_threshold=_as_float(
            os.getenv("SEMANTIC_DUP_THRESHOLD"),
            float(review_cfg.get("semantic_dup_threshold", 0.93)),
        ),
        semantic_gray_threshold=_as_float(
            os.getenv("SEMANTIC_GRAY_THRESHOLD"),
            float(review_cfg.get("semantic_gray_threshold", 0.88)),
        ),
        template_jaccard_threshold=_as_float(
            os.getenv("TEMPLATE_JACCARD_THRESHOLD"),
            float(review_cfg.get("template_jaccard_threshold", 0.88)),
        ),
        template_gray_jaccard=_as_float(
            os.getenv("TEMPLATE_GRAY_JACCARD"),
            float(review_cfg.get("template_gray_jaccard", 0.75)),
        ),
        simhash_hamming_threshold=_as_int(
            os.getenv("SIMHASH_HAMMING_THRESHOLD"),
            int(review_cfg.get("simhash_hamming_threshold", 3)),
        ),
        simhash_gray_hamming=_as_int(
            os.getenv("SIMHASH_GRAY_HAMMING"),
            int(review_cfg.get("simhash_gray_hamming", 8)),
        ),
        review_llm_every_k_batches=_as_int(
            os.getenv("REVIEW_LLM_EVERY_K_BATCHES"),
            int(review_cfg.get("llm_every_k_batches", 1)),
        ),
        sieve_use_llm=_as_bool(os.getenv("SIEVE_USE_LLM"), False),
        log_reasoning=_as_bool(os.getenv("LOG_REASONING"), False),
        fake_llm=_as_bool(os.getenv("FAKE_LLM"), False)
        if fake_llm is None
        else fake_llm,
        salvage_enabled=_as_bool(os.getenv("SALVAGE_ENABLED"), True),
        salvage_llm_attempts=_as_int(os.getenv("SALVAGE_LLM_ATTEMPTS"), 1),
        salvage_thinking_level=(os.getenv("SALVAGE_THINKING_LEVEL") or "none").strip().lower(),
        salvage_max_tokens=_as_int(os.getenv("SALVAGE_MAX_TOKENS"), 4096),
        salvage_use_llm=_as_bool(os.getenv("SALVAGE_USE_LLM"), False),
        retry_drain_enabled=_as_bool(os.getenv("RETRY_DRAIN_ENABLED"), True),
        retry_max_attempts=_as_int(os.getenv("RETRY_MAX_ATTEMPTS"), 5),
        retry_base_cooldown_sec=_as_float(os.getenv("RETRY_BASE_COOLDOWN_SEC"), 15.0),
        retry_max_cooldown_sec=_as_float(os.getenv("RETRY_MAX_COOLDOWN_SEC"), 120.0),
        retry_drain_batch=_as_int(os.getenv("RETRY_DRAIN_BATCH"), 0),
        llm_rate_limit_retries_per_call=_as_int(
            os.getenv("LLM_RATE_LIMIT_RETRIES_PER_CALL"), 2
        ),
        json_keep_truncated=_as_bool(os.getenv("JSON_KEEP_TRUNCATED"), True),
        question_min_chars=_as_int(
            os.getenv("QUESTION_MIN_CHARS"),
            int(gen_cfg.get("sieve_min_chars", 40)),
        ),
        question_max_chars=_as_int(
            os.getenv("QUESTION_MAX_CHARS"),
            int(gen_cfg.get("question_max_chars", 2500)),
        ),
        default_yaml=default_yaml,
        quotas=quotas,
        personas=personas_raw.get("personas", {}),
        scope=scope,
        compat=compat,
        data_dir_override=_resolve_data_dir(data_dir),
        raw_config=planner_yaml,
        profile=profile_name,
    )
    if not settings.fake_llm and settings.review_model == settings.llm_model:
        logger.warning("review_model_equals_writer model=%s set REVIEW_MODEL to a different family", settings.llm_model)
    return settings
