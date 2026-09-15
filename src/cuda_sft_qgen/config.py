from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from cuda_sft_qgen.keys import parse_api_keys


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


@dataclass
class Settings:
    root: Path
    nvidia_api_key: str
    nvidia_api_keys: list[str]
    nvidia_base_url: str
    nvidia_model: str
    thinking_level: str
    embedding_provider: str
    embedding_model: str
    review_batch_size: int
    target_question_count: int
    writer_concurrency: int
    llm_per_key_concurrency: int
    embed_per_key_concurrency: int
    embed_chunk_size: int
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
    data_dir_override: Path | None = None

    @property
    def data_dir(self) -> Path:
        if self.data_dir_override is not None:
            return self.data_dir_override
        rel = self.default_yaml.get("paths", {}).get("data_dir", "data")
        return (self.root / rel).resolve()

    @property
    def taxonomy_path(self) -> Path:
        rel = self.default_yaml.get("paths", {}).get(
            "taxonomy", "configs/taxonomy/cuda.yaml"
        )
        return (self.root / rel).resolve()

    @property
    def generation(self) -> dict[str, Any]:
        return self.default_yaml.get("generation", {})

    @property
    def review(self) -> dict[str, Any]:
        return self.default_yaml.get("review", {})

    @property
    def evolver_fraction(self) -> float:
        return float(self.generation.get("evolver_fraction", 0.30))

    @property
    def temperature(self) -> float:
        return float(self.generation.get("temperature", 1.0))

    @property
    def top_p(self) -> float:
        return float(self.generation.get("top_p", 0.95))

    @property
    def max_tokens(self) -> int:
        return int(self.generation.get("max_tokens", 4096))

    @property
    def sieve_min_chars(self) -> int:
        return int(self.question_min_chars)

    @property
    def max_parse_retries(self) -> int:
        return int(self.generation.get("max_parse_retries", 1))


def load_settings(
    *,
    fake_llm: bool | None = None,
    target: int | None = None,
    review_batch_size: int | None = None,
    env_file: Path | None = None,
    data_dir: Path | None = None,
) -> Settings:
    root = find_project_root()
    load_dotenv(env_file or (root / ".env"), override=False)

    default_yaml = _load_yaml(root / "configs" / "default.yaml")
    quotas = _load_yaml(root / "configs" / "quotas.yaml")
    personas_raw = _load_yaml(root / "configs" / "personas.yaml")
    review_cfg = default_yaml.get("review", {})
    gen_cfg = default_yaml.get("generation", {})

    thinking = os.getenv("THINKING_LEVEL", "medium").strip().lower()
    if thinking not in {"none", "medium", "high"}:
        raise ValueError(f"THINKING_LEVEL must be none|medium|high, got {thinking}")

    provider = os.getenv("EMBEDDING_PROVIDER", "nvidia").strip().lower()
    if provider not in {"nvidia", "huggingface", "fake"}:
        raise ValueError("EMBEDDING_PROVIDER must be nvidia|huggingface|fake")

    api_keys = parse_api_keys()
    llm_per_key = _as_int(
        os.getenv("LLM_PER_KEY_CONCURRENCY"),
        int(gen_cfg.get("llm_per_key_concurrency", 2)),
    )
    embed_per_key = _as_int(
        os.getenv("EMBED_PER_KEY_CONCURRENCY"),
        int(gen_cfg.get("embed_per_key_concurrency", 2)),
    )
    n_keys = max(1, len(api_keys))
    writer_concurrency = n_keys * max(1, llm_per_key)

    settings = Settings(
        root=root,
        nvidia_api_key=api_keys[0] if api_keys else "",
        nvidia_api_keys=api_keys,
        nvidia_base_url=os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).rstrip("/"),
        nvidia_model=os.getenv(
            "NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
        ),
        thinking_level=thinking,
        embedding_provider=provider,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b"
        ),
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
        embed_chunk_size=_as_int(
            os.getenv("EMBED_CHUNK_SIZE"),
            int(gen_cfg.get("embed_chunk_size", 8)),
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
        salvage_use_llm=_as_bool(os.getenv("SALVAGE_USE_LLM"), True),
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
        data_dir_override=Path(data_dir) if data_dir is not None else None,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
