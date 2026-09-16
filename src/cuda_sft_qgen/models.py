from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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


class QuestionBrief(BaseModel):
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
    breadth_or_depth: str = "BREADTH"  # "BREADTH" | "DEPTH"


class Question(BaseModel):
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

    @field_validator("language", mode="before")
    @classmethod
    def _coerce_language(cls, value: Any) -> str:
        text = str(value or "").strip().strip("\"'").lower()
        if text.startswith("en"):
            return "en"
        if text.startswith("zh") or text.startswith("cn"):
            return "zh"
        return "zh"

    def to_jsonl(self) -> dict[str, Any]:
        return self.model_dump()


class SteeringBrief(BaseModel):
    underrepresented: list[str] = Field(default_factory=list)
    overrepresented: list[str] = Field(default_factory=list)
    forbidden_templates: list[str] = Field(default_factory=list)
    preferred_types: list[str] = Field(default_factory=list)
    difficulty_nudge: Literal["up", "down", "hold"] = "hold"
    style_notes: str = ""

    def to_prompt_block(self, *, anti_cap: int = 15) -> str:
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
    verdict: DupVerdict
    reason: str | None = None
    nearest_id: str | None = None
    nearest_cosine: float | None = None
    template_jaccard: float | None = None
    hamming: int | None = None


class DiversityReport(BaseModel):
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
