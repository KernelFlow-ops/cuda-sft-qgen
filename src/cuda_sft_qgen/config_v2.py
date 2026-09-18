"""Typed Pydantic view of ``config.load_settings``; not a second source of truth.

``run_pipeline`` uses :class:`cuda_sft_qgen.config.Settings`. This module maps
that object for ``validate-config`` / ``estimate-cost`` and tests. Nested
defaults here must not be read as runtime knobs — YAML+env on Settings wins.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from cuda_sft_qgen.config import load_settings


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "nvidia"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    thinking_level: str = "medium"
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 16384
    per_key_concurrency: int = 2
    rate_limit_retries_per_call: int = 2
    prompt_cache_enabled: bool = True

    @field_validator("thinking_level")
    @classmethod
    def validate_thinking(cls, v: str) -> str:
        """Accept ``none|low|low_effort|medium|high``."""
        if v not in {"none", "low", "low_effort", "medium", "high"}:
            raise ValueError(f"thinking_level must be none|low|medium|high, got {v}")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""
    provider: str = "nvidia"
    model: str = "nvidia/nemotron-3-embed-1b"
    per_key_concurrency: int = 2
    oversub: int = 2
    chunk_size: int = 8
    cache_enabled: bool = True


class ReviewConfig(BaseModel):
    """Review and deduplication configuration."""
    batch_size: int = 20
    model: str = ""
    top_k: int = 8
    semantic_dup_threshold: float = 0.94
    semantic_gray_threshold: float = 0.90
    template_jaccard_threshold: float = 0.91
    template_gray_jaccard: float = 0.82
    simhash_hamming_threshold: int = 3
    simhash_gray_hamming: int = 8
    kl_alert: float = 0.15
    entropy_ratio_alert: float = 0.85
    nn_cosine_alert: float = 0.80
    max_cluster_frac_alert: float = 0.25
    leaf_min_questions: int = 3
    leaf_coverage_target: float = 0.80
    overrep_factor: float = 2.0
    llm_every_k_batches: int = 1
    chair_parallel_workers: int = 4
    gray_batch_size: int = 10


class PlannerConfig(BaseModel):
    """Planner and generation configuration."""
    writer_concurrency: int = 6
    evolver_fraction: float = 0.30
    fewshot_positives: int = 5
    fewshot_negatives: int = 2
    anti_templates_max: int = 15
    fewshot_snippet_chars: int = 500
    fewshot_max_nn_cosine: float = 0.50
    fewshot_nn_cosine_fallback: float = 0.80
    fewshot_simhash_hamming_max: int = 20
    prompt_cot: bool = True
    sieve_min_chars: int = 24
    max_parse_retries: int = 1
    priority_queue_size: int = 200
    backpressure_threshold: float = 0.20
    backpressure_sleep: float = 0.5
    adaptive_sampling_enabled: bool = True
    exploration_rate: float = 0.15
    health_window: int = 200


class TaxonomyConfig(BaseModel):
    """Taxonomy management configuration."""
    target_leaves_per_l2: int = 30
    target_leaf_count: int = 4500
    expand_max_per_call: int = 40
    manager_enabled: bool = True
    every_k_reviews: int = 5
    max_llm_calls_per_tick: int = 2
    max_leaves_per_tick: int = 12
    health_expand_threshold: int = 70
    leaf_cosine_dup: float = 0.92
    leaf_cosine_gray: float = 0.85
    facets: list[str] = Field(default_factory=lambda: [
        "formula", "failure", "feature", "mechanism", "api",
        "numeric", "layout", "profiling", "arch_migrate", "cross_factor"
    ])
    allow_refine_in_pipeline: bool = False
    allow_merge_in_pipeline: bool = False
    allow_new_domains: bool = False


class RetryConfig(BaseModel):
    """Retry and salvage configuration."""
    salvage_enabled: bool = True
    salvage_llm_attempts: int = 1
    salvage_thinking_level: str = "none"
    salvage_max_tokens: int = 4096
    salvage_use_llm: bool = False
    drain_enabled: bool = True
    max_attempts: int = 5
    base_cooldown_sec: float = 15.0
    max_cooldown_sec: float = 120.0
    drain_batch: int = 0


class AppConfig(BaseModel):
    """Main application configuration."""
    root: Path
    profile: str = "default"

    # API keys
    nvidia_api_keys: list[str] = Field(default_factory=list, repr=False)

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    knowledge: dict[str, Any] = Field(default_factory=dict)

    # Paths
    taxonomy_path: Path = Field(default=Path("configs/taxonomy/cuda.yaml"))
    quotas_path: Path = Field(default=Path("configs/quotas.yaml"))
    personas_path: Path = Field(default=Path("configs/personas.yaml"))
    data_dir: Path = Field(default=Path("data"))

    # Runtime
    target_question_count: int = 1000
    max_iterations: int = 200
    fake_llm: bool = False
    sieve_use_llm: bool = False
    log_reasoning: bool = False
    json_keep_truncated: bool = True
    question_min_chars: int = 24
    question_max_chars: int = 2500

    # Legacy support
    _default_yaml: dict[str, Any] = {}
    _quotas: dict[str, Any] = {}
    _personas: dict[str, Any] = {}

    class Config:
        """Allow Path fields; Pydantic v1-style inner config for compatibility."""
        arbitrary_types_allowed = True

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative path against project root."""
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    @property
    def nvidia_api_key(self) -> str:
        """Primary API key for backward compatibility."""
        return self.nvidia_api_keys[0] if self.nvidia_api_keys else ""

    @property
    def nvidia_base_url(self) -> str:
        """LLM base URL for backward compatibility."""
        return self.llm.base_url

    @property
    def nvidia_model(self) -> str:
        """LLM model for backward compatibility."""
        return self.llm.model

    @property
    def thinking_level(self) -> str:
        """Thinking level for backward compatibility."""
        return self.llm.thinking_level

    @property
    def embedding_provider(self) -> str:
        """Embedding provider for backward compatibility."""
        return self.embedding.provider

    @property
    def embedding_model(self) -> str:
        """Embedding model for backward compatibility."""
        return self.embedding.model

    @property
    def default_yaml(self) -> dict[str, Any]:
        """Legacy YAML access."""
        return self._default_yaml

    @property
    def quotas(self) -> dict[str, Any]:
        """Legacy quotas access."""
        return self._quotas

    @property
    def personas(self) -> dict[str, Any]:
        """Legacy personas access."""
        return self._personas

    @property
    def generation(self) -> dict[str, Any]:
        """Legacy generation config access."""
        return {
            "temperature": self.llm.temperature,
            "top_p": self.llm.top_p,
            "max_tokens": self.llm.max_tokens,
            "writer_concurrency": self.planner.writer_concurrency,
            "llm_per_key_concurrency": self.llm.per_key_concurrency,
            "embed_per_key_concurrency": self.embedding.per_key_concurrency,
            "embed_oversub": self.embedding.oversub,
            "embed_chunk_size": self.embedding.chunk_size,
            "prompt_cache": self.llm.prompt_cache_enabled,
            "embedding_cache": self.embedding.cache_enabled,
            "evolver_fraction": self.planner.evolver_fraction,
            "fewshot_positives": self.planner.fewshot_positives,
            "fewshot_negatives": self.planner.fewshot_negatives,
            "anti_templates_max": self.planner.anti_templates_max,
            "fewshot_snippet_chars": self.planner.fewshot_snippet_chars,
            "fewshot_max_nn_cosine": self.planner.fewshot_max_nn_cosine,
            "fewshot_nn_cosine_fallback": self.planner.fewshot_nn_cosine_fallback,
            "fewshot_simhash_hamming_max": self.planner.fewshot_simhash_hamming_max,
            "prompt_cot": self.planner.prompt_cot,
            "sieve_min_chars": self.planner.sieve_min_chars,
            "max_parse_retries": self.planner.max_parse_retries,
        }


def load_config_v2(
    *,
    profile: str | None = None,
    target: int | None = None,
    fake_llm: bool | None = None,
    data_dir: Path | None = None,
    review_batch_size: int | None = None,
    env_file: Path | None = None,
) -> AppConfig:
    """Load configuration with profile support.

    Priority: CLI args > ENV > profile YAML > default.yaml
    """
    settings = load_settings(profile=profile, target=target, fake_llm=fake_llm, data_dir=data_dir,
                             review_batch_size=review_batch_size, env_file=env_file)
    review = dict(settings.review)
    for name in ("semantic_dup_threshold", "semantic_gray_threshold", "template_jaccard_threshold",
                 "template_gray_jaccard", "simhash_hamming_threshold", "simhash_gray_hamming"):
        review[name] = getattr(settings, name)
    review.update(batch_size=settings.review_batch_size, model=settings.review_model,
                  llm_every_k_batches=settings.review_llm_every_k_batches,
                  chair_parallel_workers=review.get("chair_max_workers", 4),
                  gray_batch_size=review.get("chair_batch_size", 10))
    planner = dict(settings.generation)
    adaptive = settings.raw_config.get("adaptive_sampling", {})
    planner.update(writer_concurrency=settings.writer_concurrency,
                   sieve_min_chars=settings.question_min_chars,
                   adaptive_sampling_enabled=bool(adaptive.get("enabled", False)),
                   exploration_rate=adaptive.get("epsilon", 0.15))
    taxonomy = dict(settings.default_yaml.get("taxonomy", {}))
    manager = taxonomy.pop("manager", {})
    taxonomy.update(manager)
    taxonomy["manager_enabled"] = manager.get("enabled", True)
    retry = {name: getattr(settings, name if name.startswith("salvage_") else f"retry_{name}")
             for name in RetryConfig.model_fields}
    config = AppConfig(
        root=settings.root, profile=settings.profile, nvidia_api_keys=settings.nvidia_api_keys,
        llm=LLMConfig(base_url=settings.nvidia_base_url, model=settings.nvidia_model,
                      thinking_level=settings.thinking_level, temperature=settings.temperature,
                      top_p=settings.top_p, max_tokens=settings.max_tokens,
                      per_key_concurrency=settings.llm_per_key_concurrency,
                      rate_limit_retries_per_call=settings.llm_rate_limit_retries_per_call,
                      prompt_cache_enabled=settings.prompt_cache_enabled),
        embedding=EmbeddingConfig(provider=settings.embedding_provider, model=settings.embedding_model,
                                  per_key_concurrency=settings.embed_per_key_concurrency,
                                  oversub=settings.embed_oversub, chunk_size=settings.embed_chunk_size,
                                  cache_enabled=settings.embedding_cache_enabled),
        review=ReviewConfig.model_validate(review), planner=PlannerConfig.model_validate(planner),
        taxonomy=TaxonomyConfig.model_validate(taxonomy), retry=RetryConfig.model_validate(retry),
        knowledge=settings.knowledge_cfg,
        taxonomy_path=settings.taxonomy_path, quotas_path=settings.quotas_path,
        personas_path=settings.personas_path, data_dir=settings.data_dir,
        **{name: getattr(settings, name) for name in (
            "target_question_count", "max_iterations", "fake_llm", "sieve_use_llm",
            "log_reasoning", "json_keep_truncated", "question_min_chars", "question_max_chars")},
    )
    config._default_yaml = settings.default_yaml
    config._quotas = settings.quotas
    config._personas = settings.personas
    return config
