"""Domain-neutral semantic constraint-frame contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def _validate_ascii_identifier(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be ASCII-stable") from exc
    return text


@dataclass(frozen=True)
class NormalizedQuantityV01:
    """A deterministic normalized quantity used by frame slot unification."""

    amount: str
    unit_ref: str
    normalized_amount: str
    normalized_unit_ref: str
    currency_ref: str | None = None
    schema_version: Literal["normalized_quantity.v0_1"] = "normalized_quantity.v0_1"

    def __post_init__(self) -> None:
        _validate_ascii_identifier(self.unit_ref, "unit_ref")
        _validate_ascii_identifier(self.normalized_unit_ref, "normalized_unit_ref")
        if self.currency_ref is not None:
            _validate_ascii_identifier(self.currency_ref, "currency_ref")


@dataclass(frozen=True)
class FrameValueConstraintV01:
    """Typed value carried by the value_constraint semantic slot."""

    value_kind: Literal["concept_ref", "context_ref", "string", "normalized_quantity"]
    concept_ref: str | None = None
    context_ref: str | None = None
    string_value: str | None = None
    normalized_quantity: NormalizedQuantityV01 | None = None
    schema_version: Literal["frame_value_constraint.v0_1"] = "frame_value_constraint.v0_1"

    def __post_init__(self) -> None:
        values = {
            "concept_ref": self.concept_ref,
            "context_ref": self.context_ref,
            "string": self.string_value,
            "normalized_quantity": self.normalized_quantity,
        }
        if values[self.value_kind] is None:
            raise ValueError(f"{self.value_kind} value is required")
        if sum(value is not None for value in values.values()) != 1:
            raise ValueError("value_constraint must set exactly one typed value")
        if self.concept_ref is not None:
            _validate_ascii_identifier(self.concept_ref, "concept_ref")
        if self.context_ref is not None:
            _validate_ascii_identifier(self.context_ref, "context_ref")


@dataclass(frozen=True)
class ConstraintFrameV01:
    """The bounded seven-slot structural semantic match unit."""

    target_type_ref: str
    attribute_ref: str
    operator_ref: str
    value_constraint: FrameValueConstraintV01
    modality_ref: str
    stage_ref: str
    consequence_ref: str
    slot_evidence_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    inferred_slots: tuple[str, ...] = ()
    schema_version: Literal["constraint_frame.v0_1"] = "constraint_frame.v0_1"

    def __post_init__(self) -> None:
        for name in (
            "target_type_ref",
            "attribute_ref",
            "operator_ref",
            "modality_ref",
            "stage_ref",
            "consequence_ref",
        ):
            _validate_ascii_identifier(getattr(self, name), name)
        allowed_slots = {
            "target_type_ref",
            "attribute_ref",
            "operator_ref",
            "value_constraint",
            "modality_ref",
            "stage_ref",
            "consequence_ref",
        }
        if not set(self.slot_evidence_refs).issubset(allowed_slots):
            raise ValueError("slot_evidence_refs contains an unknown semantic slot")
        if not set(self.inferred_slots).issubset(allowed_slots):
            raise ValueError("inferred_slots contains an unknown semantic slot")


@dataclass(frozen=True)
class FramePatternV01:
    pattern_id: str
    frame: ConstraintFrameV01
    graph_release_ref: str
    source_evidence_refs: tuple[str, ...]
    lifecycle_state: Literal["candidate", "accepted", "retired"]
    pattern_hash: str
    corpus_schema_candidate_ref: str
    schema_version: Literal["frame_pattern.v0_1"] = "frame_pattern.v0_1"

    def __post_init__(self) -> None:
        _validate_ascii_identifier(self.pattern_id, "pattern_id")
        _validate_ascii_identifier(self.graph_release_ref, "graph_release_ref")
        _validate_ascii_identifier(
            self.corpus_schema_candidate_ref, "corpus_schema_candidate_ref"
        )
        if not self.source_evidence_refs:
            raise ValueError("source_evidence_refs must not be empty")
        if not self.pattern_hash.startswith("sha256:"):
            raise ValueError("pattern_hash must be a sha256 ref")


@dataclass(frozen=True)
class ComposedFramePatternV01:
    pattern_id: str
    composition_operator: Literal["ALL_OF"]
    constituents: tuple[FramePatternV01, ...]
    graph_release_ref: str
    source_evidence_refs: tuple[str, ...]
    lifecycle_state: Literal["candidate", "accepted", "retired"]
    pattern_hash: str
    schema_version: Literal["composed_frame_pattern.v0_1"] = (
        "composed_frame_pattern.v0_1"
    )

    def __post_init__(self) -> None:
        _validate_ascii_identifier(self.pattern_id, "pattern_id")
        _validate_ascii_identifier(self.graph_release_ref, "graph_release_ref")
        if len(self.constituents) < 2:
            raise ValueError("ALL_OF composition requires at least two constituents")
        if not self.source_evidence_refs:
            raise ValueError("source_evidence_refs must not be empty")
        if not self.pattern_hash.startswith("sha256:"):
            raise ValueError("pattern_hash must be a sha256 ref")
