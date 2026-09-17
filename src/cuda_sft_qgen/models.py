"""Pydantic contracts for questions, reviews, briefs, and diversity reports.

Accepted reviews require every quality check to pass, a private reference
answer, and independently assessed metadata. Uncertainty never authorizes
export.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

Hardware = Literal["agnostic", "ampere", "hopper", "blackwell"]
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
    hardware: Hardware
    granularity: Granularity
    language: Language


CheckResult = Literal["pass", "fail", "uncertain"]


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
            raise ValueError("reject requires a failed check")
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


class Question(BaseModel):
    """Generated candidate plus review, fingerprint, and knowledge provenance.

    ``internal_reference_answer`` is review-only and is stripped from JSONL
    export. ``review_status`` stays pending until an independent chair accept
    or a hard reject.
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
    internal_reference_answer: str | None = None
    review_attempts: int = 0
    review_next_at: float = 0.0
    review_exhausted: bool = False

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
        """Serialize for archive export; drop the private reference answer."""
        record = self.model_dump(exclude={"internal_reference_answer"})
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
