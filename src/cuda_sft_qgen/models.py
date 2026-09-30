"""Pydantic contracts for questions, reviews, briefs, and diversity reports.

Accepted reviews require every quality check to pass, a private reference
answer, and independently assessed metadata. Uncertainty never authorizes
export.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cuda_sft_qgen.enums import ANSWER_FORMATS, ARCH_TARGETS, CHECK_TO_DEFECT, SCENARIOS, DEFECT_CODES, OPERATORS


def _validate_enum(value: str, allowed: tuple[str, ...], name: str, allow_empty: bool = False) -> str:
    """Validate a string label against its canonical vocabulary."""
    if value not in allowed and not (allow_empty and value == ""):
        raise ValueError(f"unknown {name} {value}")
    return value


QuestionType = Literal[
    "conceptual",
    "syntax",
    "code_complete",
    "debug",
    "optimization",
    "layout",
    "compare",
    "counterfactual",
    "profiling",
]

Bloom = Literal[
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]

Hardware = Literal["agnostic", "volta", "turing", "ampere", "ada", "hopper", "blackwell"]
Granularity = Literal["api", "kernel", "pipeline", "system"]
Language = Literal["zh", "en"]
Source = Literal["writer", "evolver"]
ReviewStatus = Literal["pending", "accepted", "rejected"]
DupVerdict = Literal["dup", "unique", "gray"]


class AssessedMetadata(BaseModel):
    """Reviewer-assigned labels; never copied blindly from the writer brief."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question_type: QuestionType
    difficulty: int = Field(ge=1, le=4)
    bloom: Bloom
    hardware: Hardware | None = None
    granularity: Granularity
    language: Language
    arch_target: str | None = None
    answer_format: str | None = None

    @field_validator("arch_target")
    @classmethod
    def _check_arch_target(cls, value: str | None) -> str | None:
        return None if value is None else _validate_enum(value, ARCH_TARGETS, "arch_target")

    @field_validator("answer_format")
    @classmethod
    def _check_answer_format(cls, value: str | None) -> str | None:
        return None if value is None else _validate_enum(value, ANSWER_FORMATS, "answer_format")


CheckResult = Literal["pass", "fail", "uncertain"]


class SeedAnalysis(BaseModel):
    """Short, auditable seed summary; never a free-form reasoning trace."""

    model_config = ConfigDict(extra="forbid")

    source_seed_ids: list[str] = Field(default_factory=list)
    domain_id: str = ""
    l3_ids: list[str] = Field(default_factory=list)
    shared_mechanism: str = Field(min_length=1, max_length=400)
    reusable_constraints: list[str] = Field(default_factory=list, max_length=8)
    differences: list[str] = Field(default_factory=list, max_length=8)


class PairPlan(BaseModel):
    """Declarative construction plan for a single seed or MathMixup pair."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "evolve", "original", "decomposed", "hybrid"]
    parent_ids: list[str] = Field(default_factory=list, max_length=2)
    retained_claim: str = Field(min_length=1, max_length=400)
    transferred_constraint: str = Field(default="", max_length=400)
    construction: str = Field(min_length=1, max_length=500)
    expected_difficulty: int = Field(ge=1, le=4)


class VerificationSpec(BaseModel):
    """Writer-proposed oracle contract, validated before independent review."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["scalar", "symbolic", "open", "compile", "runtime"]
    answer_type: Literal["number", "enum", "text", "expression"]
    expected_behavior: str = Field(min_length=1, max_length=500)
    rubric: list[str] = Field(min_length=1, max_length=8)
    expression: str | None = None
    variables: dict[str, Any] | None = None
    code: str | None = None
    architecture: str | None = None
    expected_compile: Literal["success", "failure"] | None = None
    expected_diagnostic: str | None = None
    expected_stdout: str | None = None
    sanitizer: Literal["none", "memcheck", "racecheck"] | None = None
    expected_runtime_outcome: Literal["success", "sanitizer_error"] = "success"
    expected_sanitizer_diagnostic: str | None = None
    toolkit_version: str | None = None

    @model_validator(mode="after")
    def validate_mode_contract(self) -> "VerificationSpec":
        if not self.expected_behavior.strip() or any(not item.strip() for item in self.rubric):
            raise ValueError("verification requires nonempty behavior and rubric criteria")
        if self.mode == "symbolic" and not (self.expression or "").strip():
            raise ValueError("symbolic verification requires expression")
        if self.mode in {"compile", "runtime"}:
            if not (self.code or "").strip():
                raise ValueError(f"{self.mode} verification requires a complete translation unit")
            if not (self.architecture or "").strip():
                raise ValueError(f"{self.mode} verification requires architecture")
        if self.mode == "compile" and self.expected_compile is None:
            raise ValueError("compile verification requires expected_compile")
        if self.mode == "runtime":
            if self.expected_compile != "success":
                raise ValueError("runtime verification requires expected_compile=success")
            if self.expected_stdout is None:
                raise ValueError("runtime verification requires expected_stdout")
            if self.sanitizer is None:
                raise ValueError("runtime verification requires sanitizer")
            if self.expected_runtime_outcome == "sanitizer_error" and (
                self.sanitizer == "none" or not (self.expected_sanitizer_diagnostic or "").strip()
            ):
                raise ValueError("negative runtime oracle requires a sanitizer and specific diagnostic")
        return self


class VerificationEvidence(BaseModel):
    """Independent verifier results; Writer output may not claim these results."""

    model_config = ConfigDict(extra="allow")

    status: Literal["not_run", "pass", "fail", "pending", "disabled"] = "not_run"
    checks: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class QualityChecks(BaseModel):
    """Independent pass/fail/uncertain verdicts for a single question."""

    model_config = ConfigDict(extra="forbid", strict=True)

    correctness: CheckResult
    answerability: CheckResult
    claim_alignment: CheckResult
    hardware_compatibility: CheckResult
    answer_leakage: CheckResult
    metadata: CheckResult
    duplication: CheckResult


class QualityReview(BaseModel):
    """Single-question reviewer contract; uncertainty can never authorize export."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    action: Literal["accept", "reject", "pending"]
    reason: str = Field(min_length=1)
    checks: QualityChecks
    evidence: list[str] = Field(min_length=1)
    internal_reference_answer: str | None
    assessed_metadata: AssessedMetadata | None
    # Only structured-generation candidates require these additional checks.
    # Optional on legacy reviews so resuming an old archive remains possible.
    construction_checks: dict[str, CheckResult] | None = None
    defect_codes: list[str] = Field(default_factory=list)
    value_scores: dict[str, int] | None = None

    @model_validator(mode="after")
    def validate_v2_fields(self) -> "QualityReview":
        if any(code not in DEFECT_CODES for code in self.defect_codes):
            raise ValueError("unknown defect code")
        if self.value_scores is not None:
            required = {"pedagogical_value", "discriminativeness", "realism"}
            if set(self.value_scores) != required or any(
                type(value) is not int or not 1 <= value <= 5 for value in self.value_scores.values()
            ):
                raise ValueError("value_scores must contain three integer scores in 1..5")
        return self

    @model_validator(mode="after")
    def validate_decision(self) -> "QualityReview":
        """Enforce accept/reject invariants so uncertain items stay pending."""
        results = self.checks.model_dump().values()
        if not self.reason.strip() or not all(e.strip() for e in self.evidence):
            raise ValueError("review reason and evidence must be nonempty")
        if self.action == "accept":
            if any(value != "pass" for value in results):
                raise ValueError("accept requires every quality check to pass")
            if not (self.internal_reference_answer or "").strip():
                raise ValueError("accept requires a private reference answer")
            if self.assessed_metadata is None:
                raise ValueError("accept requires independently assessed metadata")
        if self.action == "reject" and "fail" not in results:
            non_check_defects = set(DEFECT_CODES) - set(CHECK_TO_DEFECT.values())
            if not set(self.defect_codes).intersection(non_check_defects):
                raise ValueError("reject requires a failed check or a non-check defect")
        return self


class QuestionBrief(BaseModel):
    """Planner task sent to Writer/Evolver; metadata here is a request, not a verdict."""

    task_id: str = ""
    topic_path: list[str]
    l3_id: str
    l3_claim: str
    question_type: str
    difficulty: int
    bloom: str
    persona: str
    hardware: str
    granularity: str
    language: str
    must_include: str = ""
    must_avoid: str = ""
    anti_templates: list[str] = Field(default_factory=list)
    angle: str = ""
    instance_hint: str = ""
    must_avoid_stems: list[str] = Field(default_factory=list)
    angle_guide: str = ""
    angle_id: str = ""
    kind: str = ""
    role: Literal["writer", "evolver"] = "writer"
    seed_question: str | None = None
    seed_id: str | None = None
    claim_revision: int | None = None
    source_ids: list[str] = Field(default_factory=list)
    knowledge_context: dict[str, Any] = Field(default_factory=dict)
    breadth_or_depth: str = "BREADTH"  # "BREADTH" | "DEPTH"
    # MathMixup provenance is orthogonal to the requested difficulty.  Keeping
    # it on the handoff makes parent usage auditable without parsing seed_id.
    mixup_mode: Literal["original", "decomposed", "hybrid"] | None = None
    parent_ids: list[str] = Field(default_factory=list)
    parent_difficulties: list[int] = Field(default_factory=list)
    difficulty_target: int | None = Field(default=None, ge=1, le=4)
    difficulty_raw_score: float | None = None
    answer_format: str = ""
    scenario: str = "generic"
    arch_target: str = "any"
    operator: str | None = None
    used_angles: list[str] = Field(default_factory=list)
    used_formats: list[str] = Field(default_factory=list)
    used_scenarios: list[str] = Field(default_factory=list)
    exploration: bool = False

    @field_validator("answer_format")
    @classmethod
    def _check_answer_format(cls, value: str) -> str:
        return _validate_enum(value, ANSWER_FORMATS, "answer_format", allow_empty=True)

    @field_validator("scenario")
    @classmethod
    def _check_scenario(cls, value: str) -> str:
        return _validate_enum(value, SCENARIOS, "scenario")

    @field_validator("arch_target")
    @classmethod
    def _check_arch_target(cls, value: str) -> str:
        return _validate_enum(value, ARCH_TARGETS, "arch_target")

    @field_validator("operator")
    @classmethod
    def _check_operator(cls, value: str | None) -> str | None:
        return None if value is None else _validate_enum(value, OPERATORS, "operator")


class Question(BaseModel):
    """Generated candidate plus review, fingerprint, and knowledge provenance.

    ``draft_plan``, ``draft_target``, and ``internal_reference_answer`` are
    review-only and stripped from JSONL export. ``review_status`` stays pending
    until an independent chair accept or a hard reject.
    """

    id: str
    question: str
    tests_claim: str
    topic_path: list[str]
    l3_id: str = ""
    l3_claim: str = ""
    question_type: str
    difficulty: int
    bloom: str
    persona: str
    hardware: str
    granularity: str
    language: str
    template_fingerprint: str = ""
    simhash: int = 0
    exact_hash: str = ""
    source: str = "writer"
    nearest_id: str | None = None
    nearest_cosine: float | None = None
    template_jaccard: float | None = None
    hamming: int | None = None
    dup_verdict: str | None = None
    review_status: ReviewStatus = "pending"
    reject_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    kind: str = ""
    requested_metadata: dict[str, Any] = Field(default_factory=dict)
    assessed_metadata: AssessedMetadata | None = None
    seed_id: str | None = None
    operator: str | None = None
    angle_id: str = ""
    claim_revision: int | None = None
    source_ids: list[str] = Field(default_factory=list)
    knowledge_dependencies: list[dict[str, Any]] = Field(default_factory=list)
    review_details: dict[str, Any] = Field(default_factory=dict)
    # Writer-only self-instruction artifacts.  They are carried through review
    # so the Chair can compare the generated target, but never exported.
    draft_plan: dict[str, Any] | None = None
    draft_target: str | None = None
    verification_spec: dict[str, Any] = Field(default_factory=dict)
    verification_evidence: dict[str, Any] = Field(default_factory=dict)
    draft_validation_errors: list[str] = Field(default_factory=list)
    generation_provenance: dict[str, Any] = Field(default_factory=dict)
    mixup_context: dict[str, Any] = Field(default_factory=dict)
    brief_snapshot: dict[str, Any] = Field(default_factory=dict)
    internal_reference_answer: str | None = None
    review_attempts: int = 0
    review_next_at: float = 0.0
    review_exhausted: bool = False
    mixup_mode: Literal["original", "decomposed", "hybrid"] | None = None
    parent_ids: list[str] = Field(default_factory=list)
    parent_difficulties: list[int] = Field(default_factory=list)
    difficulty_target: int | None = Field(default=None, ge=1, le=4)
    difficulty_raw_score: float | None = None
    generation_round: int | None = None
    answer_format: str = ""
    scenario: str = "generic"
    arch_target: str = "any"
    draft_rationale: str | None = None
    premise_check: list[str] = Field(default_factory=list)
    defect_codes: list[str] = Field(default_factory=list)
    value_scores: dict[str, int] = Field(default_factory=dict)
    exploration: bool = False

    @field_validator("answer_format")
    @classmethod
    def _check_answer_format(cls, value: str) -> str:
        return _validate_enum(value, ANSWER_FORMATS, "answer_format", allow_empty=True)

    @field_validator("scenario")
    @classmethod
    def _check_scenario(cls, value: str) -> str:
        return _validate_enum(value, SCENARIOS, "scenario")

    @field_validator("arch_target")
    @classmethod
    def _check_arch_target(cls, value: str) -> str:
        return _validate_enum(value, ARCH_TARGETS, "arch_target")

    @field_validator("language", mode="before")
    @classmethod
    def _coerce_language(cls, value: Any) -> str:
        """Map noisy model tokens (en/zh/cn/english/...) onto the Language literal."""
        text = str(value or "").strip().strip("\"'").lower()
        if text.startswith("en"):
            return "en"
        if text.startswith("zh") or text.startswith("cn"):
            return "zh"
        return "zh"

    def to_jsonl(self) -> dict[str, Any]:
        """Serialize public training data without private generation artifacts."""
        record = self.model_dump(exclude={
            "internal_reference_answer", "draft_plan", "draft_target",
            "verification_spec", "verification_evidence", "draft_validation_errors",
            "generation_provenance", "mixup_context", "brief_snapshot",
            # The existing question-only JSONL schema remains stable; synthesis
            # lineage and curriculum labels live in the export manifest.
            "mixup_mode", "parent_ids", "parent_difficulties", "difficulty_target",
            "difficulty_raw_score", "generation_round",
            "draft_rationale", "premise_check", "defect_codes", "value_scores", "exploration",
        })
        record["review_details"] = {
            key: value for key, value in self.review_details.items()
            if key in {"action", "checks", "reviewer_model"}
        }
        return record


class SteeringBrief(BaseModel):
    """Chair-to-planner distribution correction for the next generate batch."""

    underrepresented: list[str] = Field(default_factory=list)
    overrepresented: list[str] = Field(default_factory=list)
    forbidden_templates: list[str] = Field(default_factory=list)
    preferred_types: list[str] = Field(default_factory=list)
    difficulty_nudge: Literal["up", "down", "hold"] = "hold"
    style_notes: str = ""

    def to_prompt_block(self, *, anti_cap: int = 15) -> str:
        """Render a compact steering block for writer prompts.

        Args:
            anti_cap: Max forbidden-template stems to include.
        """
        if not any(
            [
                self.underrepresented,
                self.overrepresented,
                self.forbidden_templates,
                self.preferred_types,
                self.style_notes,
                self.difficulty_nudge != "hold",
            ]
        ):
            return "(empty — no steering yet)"
        cap = max(1, int(anti_cap))
        lines = [
            f"underrepresented: {self.underrepresented}",
            f"overrepresented: {self.overrepresented}",
            f"forbidden_templates: {self.forbidden_templates[:cap]}",
            f"preferred_types: {self.preferred_types}",
            f"difficulty_nudge: {self.difficulty_nudge}",
            f"style_notes: {self.style_notes}",
        ]
        return "\n".join(lines)


class DupDecision(BaseModel):
    """Three-layer dedup outcome: exact, template, or semantic nearest neighbor."""

    verdict: DupVerdict
    reason: str | None = None
    nearest_id: str | None = None
    nearest_cosine: float | None = None
    template_jaccard: float | None = None
    hamming: int | None = None


class DiversityReport(BaseModel):
    """Quota coverage and imbalance stats used to build the next SteeringBrief."""

    accepted_count: int
    pending_count: int
    l1_hist: dict[str, int] = Field(default_factory=dict)
    type_hist: dict[str, int] = Field(default_factory=dict)
    difficulty_hist: dict[str, int] = Field(default_factory=dict)
    hardware_hist: dict[str, int] = Field(default_factory=dict)
    granularity_hist: dict[str, int] = Field(default_factory=dict)
    leaf_coverage: float = 0.0
    kind_hist: dict[str, int] = Field(default_factory=dict)
    kl_l1: float = 0.0
    kl_type: float = 0.0
    kl_difficulty: float = 0.0
    entropy_ratio_l1: float = 1.0
    mean_nn_cosine: float | None = None
    underrepresented: list[str] = Field(default_factory=list)
    overrepresented: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    format_hist: dict[str, int] = Field(default_factory=dict)
    scenario_hist: dict[str, int] = Field(default_factory=dict)
    arch_hist: dict[str, int] = Field(default_factory=dict)
    joint_cells: int = 0
    max_cell_share: float = 0.0
    requested_vs_assessed_agreement: float | None = None
