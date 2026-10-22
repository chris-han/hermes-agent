"""Tender-specific annotation values; task roles never identify a person."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA = r"^[0-9a-f]{64}$"
Role = Literal["R1", "R2", "R3"]
Result = Literal[
    "TRUE",
    "FALSE",
    "SEMANTIC_UNKNOWN",
    "STRUCTURAL_UNKNOWN",
    "CONFLICT",
    "NOT_APPLICABLE",
]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(min_length=1)
    raw_sha256: str = Field(pattern=SHA)
    locator: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    result: Result
    applicability: Literal["APPLICABLE", "NOT_APPLICABLE", "UNKNOWN"]
    material_context_complete: bool
    cited_spans: list[Citation] = Field(default_factory=list)
    operative_rules: list[str] = Field(default_factory=list)
    subject: str = ""
    stage: str = ""
    trigger: str = ""
    consequence: str = ""
    antecedents: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    precedence: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    unfulfilled_dependencies: list[str] = Field(default_factory=list)

    @field_validator(
        "operative_rules",
        "antecedents",
        "exceptions",
        "precedence",
        "unfulfilled_dependencies",
    )
    @classmethod
    def nonempty_entries(cls, values):
        return [v.strip() for v in values if v.strip()]

    @model_validator(mode="after")
    def validate_result_basis(self):
        if not self.rationale.strip():
            raise ValueError("RATIONALE_REQUIRED")
        if self.result == "SEMANTIC_UNKNOWN" and (
            not self.material_context_complete or not self.cited_spans
        ):
            raise ValueError("SEMANTIC_UNKNOWN_REQUIRES_COMPLETE_MATERIAL_CONTEXT")
        if self.result in {"TRUE", "FALSE"} and (
            not self.material_context_complete
            or self.applicability != "APPLICABLE"
            or not self.cited_spans
            or self.unfulfilled_dependencies
            or not self.operative_rules
        ):
            raise ValueError("DECIDABLE_RESULT_REQUIRES_COMPLETE_EVIDENCE")
        if self.result == "STRUCTURAL_UNKNOWN" and not self.unfulfilled_dependencies:
            raise ValueError("STRUCTURAL_UNKNOWN_REQUIRES_MISSING_DEPENDENCIES")
        if self.result == "NOT_APPLICABLE" and self.applicability != "NOT_APPLICABLE":
            raise ValueError("NOT_APPLICABLE_REQUIRES_EXPLICIT_APPLICABILITY")
        if self.result == "CONFLICT" and len(self.cited_spans) < 2:
            raise ValueError("CONFLICT_REQUIRES_DISTINCT_EVIDENCE")
        return self


class DraftAnnotation(Annotation):
    """An unfinished judgment can be saved without selecting a label."""

    result: Result | None = None
    applicability: Literal["APPLICABLE", "NOT_APPLICABLE", "UNKNOWN"] = "UNKNOWN"
    material_context_complete: bool = False
    rationale: str = ""

    @model_validator(mode="after")
    def validate_result_basis(self):
        return self
