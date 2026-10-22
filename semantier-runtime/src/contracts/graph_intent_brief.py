"""Immutable source-grounded GraphIntentBrief compiler-input contract."""

from __future__ import annotations

from contracts.frame_mapping import FrameMapping

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from eos.artifact_hashing import content_hash


SCHEMA_VERSION = "graph_intent_brief.v1"
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_MACHINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


_GRAPH_INTENT_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "graph_intent_proposal": {
            "type": "object",
            "properties": {
                "document_function": {
                    "type": "object",
                    "properties": {"type_id": {"type": "string", "pattern": _MACHINE_ID.pattern}},
                },
                "represented_or_governed_subject": {
                    "type": "object",
                    "properties": {"type_id": {"type": "string", "pattern": _MACHINE_ID.pattern}},
                },
                "primary_relationship": {
                    "type": "object",
                    "properties": {
                        "type_id": {"type": "string", "pattern": _MACHINE_ID.pattern},
                        "source_types": {"type": "array", "items": {"type": "string", "pattern": _MACHINE_ID.pattern}},
                        "target_types": {"type": "array", "items": {"type": "string", "pattern": _MACHINE_ID.pattern}},
                        "direction": {"enum": ["source_to_target", "target_to_source"]},
                    },
                },
                "source_role_interpretation": {
                    "type": "object",
                    "properties": {
                        key: {"type": "array", "items": {"type": "string", "pattern": _MACHINE_ID.pattern}}
                        for key in (
                            "canonical_objects",
                            "lexical_evidence",
                            "reference_examples",
                            "classifications",
                            "source_assessments",
                            "remediation_fields",
                            "contextual_fields",
                        )
                    },
                },
            },
        }
    },
    "required": ["graph_intent_proposal"],
    "additionalProperties": False,
}


def graph_intent_proposal_output_schema(catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """Constrain proposal identifiers to the selected governed catalog, not labels.

    This transport schema does not replace source grounding or catalog admission.
    """
    schema = deepcopy(_GRAPH_INTENT_EVIDENCE_SCHEMA)
    if catalog is None:
        return schema
    type_id = {"type": "string", "enum": sorted(catalog["admitted_type_ids"])}
    relation_id = {"type": "string", "enum": sorted(catalog["admitted_relation_ids"])}
    fields = schema["properties"]["graph_intent_proposal"]["properties"]
    for key in ("document_function", "represented_or_governed_subject"):
        fields[key]["properties"]["type_id"] = deepcopy(type_id)
    relationship = fields["primary_relationship"]["properties"]
    relationship["type_id"] = relation_id
    for key in ("source_types", "target_types"):
        relationship[key]["items"] = deepcopy(type_id)
    for field in fields["source_role_interpretation"]["properties"].values():
        field["items"] = deepcopy(type_id)
    fields["preferred_representation"] = {
        "type": "object", "properties": {
            "kind": {"enum": sorted(catalog["admitted_representation_kinds"])},
            "type_id": {"anyOf": [deepcopy(type_id), {"type": "null"}]},
        },
    }
    fields["material_dimensions"] = {
        "type": "array", "items": {
            "type": "object", "properties": {"type_id": deepcopy(type_id)},
        },
    }
    return schema


def _required(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _machine_id(value: Any, field: str) -> str:
    text = _required(value, field)
    if not text.isascii() or _MACHINE_ID.fullmatch(text) is None:
        raise ValueError(f"{field} must be an ASCII-stable machine identifier")
    return text


def _optional_machine_id(value: Any, field: str) -> str | None:
    return None if value is None else _machine_id(value, field)


def _sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if _SHA256_REF.fullmatch(text) is None:
        raise ValueError(f"{field} must be a sha256 ref")
    return text


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _strings(value: Any, field: str, *, machine_ids: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    converter = _machine_id if machine_ids else _required
    return tuple(converter(item, f"{field}[]") for item in value)


def _canonical_machine_refs(value: Any, field: str) -> tuple[str, ...]:
    refs = _strings(value, field, machine_ids=True)
    if len(refs) != len(set(refs)):
        raise ValueError(f"{field} contains duplicate role refs")
    return tuple(sorted(refs, key=lambda item: item.encode("utf-8")))


@dataclass(frozen=True)
class GraphIntentSourceBinding:
    organization_id: str
    workspace_id: str
    lineage_id: str
    source_identity_ref: str
    source_hash: str
    source_version: str | None
    canonical_source_ir_ref: str
    canonical_source_ir_hash: str

    def __post_init__(self) -> None:
        for field in (
            "organization_id",
            "workspace_id",
            "lineage_id",
            "source_identity_ref",
            "canonical_source_ir_ref",
        ):
            _machine_id(getattr(self, field), f"source_binding.{field}")
        _optional_machine_id(self.source_version, "source_binding.source_version")
        _sha256(self.source_hash, "source_binding.source_hash")
        _sha256(
            self.canonical_source_ir_hash,
            "source_binding.canonical_source_ir_hash",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "lineage_id": self.lineage_id,
            "source_identity_ref": self.source_identity_ref,
            "source_hash": self.source_hash,
            "source_version": self.source_version,
            "canonical_source_ir_ref": self.canonical_source_ir_ref,
            "canonical_source_ir_hash": self.canonical_source_ir_hash,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GraphIntentSourceBinding":
        payload = _object(value, "source_binding")
        return cls(
            organization_id=_machine_id(
                payload.get("organization_id"), "source_binding.organization_id"
            ),
            workspace_id=_machine_id(
                payload.get("workspace_id"), "source_binding.workspace_id"
            ),
            lineage_id=_machine_id(
                payload.get("lineage_id"), "source_binding.lineage_id"
            ),
            source_identity_ref=_machine_id(
                payload.get("source_identity_ref"), "source_binding.source_identity_ref"
            ),
            source_hash=_sha256(
                payload.get("source_hash"), "source_binding.source_hash"
            ),
            source_version=_optional_machine_id(
                payload.get("source_version"), "source_binding.source_version"
            ),
            canonical_source_ir_ref=_machine_id(
                payload.get("canonical_source_ir_ref"),
                "source_binding.canonical_source_ir_ref",
            ),
            canonical_source_ir_hash=_sha256(
                payload.get("canonical_source_ir_hash"),
                "source_binding.canonical_source_ir_hash",
            ),
        )


@dataclass(frozen=True)
class IntentTypeDescription:
    type_id: str
    description: str

    def __post_init__(self) -> None:
        _machine_id(self.type_id, "type_id")
        _required(self.description, "description")

    def as_dict(self) -> dict[str, str]:
        return {"type_id": self.type_id, "description": self.description}

    @classmethod
    def from_dict(cls, value: Any, field: str) -> "IntentTypeDescription":
        payload = _object(value, field)
        return cls(
            type_id=_machine_id(payload.get("type_id"), f"{field}.type_id"),
            description=_required(payload.get("description"), f"{field}.description"),
        )


@dataclass(frozen=True)
class PrimaryRelationship:
    type_id: str
    source_types: tuple[str, ...]
    target_types: tuple[str, ...]
    direction: str
    semantic_description: str
    source_role_refs: tuple[str, ...] = ()
    target_role_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _machine_id(self.type_id, "primary_relationship.type_id")
        if not self.source_types or not self.target_types:
            raise ValueError(
                "primary_relationship source_types and target_types are required"
            )
        for field in ("source_types", "target_types"):
            for value in getattr(self, field):
                _machine_id(value, f"primary_relationship.{field}[]")
        _machine_id(self.direction, "primary_relationship.direction")
        _required(
            self.semantic_description, "primary_relationship.semantic_description"
        )
        if bool(self.source_role_refs) != bool(self.target_role_refs):
            raise ValueError(
                "primary_relationship endpoint role refs must be supplied as a pair"
            )
        for field in ("source_role_refs", "target_role_refs"):
            refs = getattr(self, field)
            canonical = _canonical_machine_refs(refs, f"primary_relationship.{field}")
            if refs != canonical:
                raise ValueError(
                    f"primary_relationship.{field} must be canonically ordered"
                )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "type_id": self.type_id,
            "source_types": list(self.source_types),
            "target_types": list(self.target_types),
            "direction": self.direction,
            "semantic_description": self.semantic_description,
        }
        if self.source_role_refs:
            payload["source_role_refs"] = list(self.source_role_refs)
            payload["target_role_refs"] = list(self.target_role_refs)
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> "PrimaryRelationship":
        payload = _object(value, "primary_relationship")
        role_fields = {"source_role_refs", "target_role_refs"}
        present_role_fields = role_fields.intersection(payload)
        if present_role_fields and present_role_fields != role_fields:
            raise ValueError(
                "primary_relationship endpoint role refs must be supplied as a pair"
            )
        return cls(
            type_id=_machine_id(payload.get("type_id"), "primary_relationship.type_id"),
            source_types=_strings(
                payload.get("source_types"),
                "primary_relationship.source_types",
                machine_ids=True,
            ),
            target_types=_strings(
                payload.get("target_types"),
                "primary_relationship.target_types",
                machine_ids=True,
            ),
            direction=_machine_id(
                payload.get("direction"), "primary_relationship.direction"
            ),
            semantic_description=_required(
                payload.get("semantic_description"),
                "primary_relationship.semantic_description",
            ),
            source_role_refs=_canonical_machine_refs(
                payload.get("source_role_refs", ()),
                "primary_relationship.source_role_refs",
            ),
            target_role_refs=_canonical_machine_refs(
                payload.get("target_role_refs", ()),
                "primary_relationship.target_role_refs",
            ),
        )


@dataclass(frozen=True)
class PreferredRepresentation:
    kind: str
    type_id: str | None
    rationale: str

    def __post_init__(self) -> None:
        if self.kind not in {"binary_relation", "event", "associative_class", "frame"}:
            raise ValueError("preferred_representation.kind is invalid")
        _optional_machine_id(self.type_id, "preferred_representation.type_id")
        _required(self.rationale, "preferred_representation.rationale")

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "type_id": self.type_id, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, value: Any) -> "PreferredRepresentation":
        payload = _object(value, "preferred_representation")
        return cls(
            kind=str(payload.get("kind") or ""),
            type_id=_optional_machine_id(
                payload.get("type_id"), "preferred_representation.type_id"
            ),
            rationale=_required(
                payload.get("rationale"), "preferred_representation.rationale"
            ),
        )


@dataclass(frozen=True)
class MaterialDimension:
    type_id: str
    description: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _machine_id(self.type_id, "material_dimensions[].type_id")
        _required(self.description, "material_dimensions[].description")
        for ref in self.evidence_refs:
            _machine_id(ref, "material_dimensions[].evidence_refs[]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "description": self.description,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MaterialDimension":
        payload = _object(value, "material_dimensions[]")
        return cls(
            type_id=_machine_id(
                payload.get("type_id"), "material_dimensions[].type_id"
            ),
            description=_required(
                payload.get("description"), "material_dimensions[].description"
            ),
            evidence_refs=_strings(
                payload.get("evidence_refs"),
                "material_dimensions[].evidence_refs",
                machine_ids=True,
            ),
        )


@dataclass(frozen=True)
class SourceRoleInterpretation:
    canonical_objects: tuple[str, ...]
    lexical_evidence: tuple[str, ...]
    reference_examples: tuple[str, ...]
    classifications: tuple[str, ...]
    source_assessments: tuple[str, ...]
    remediation_fields: tuple[str, ...]
    contextual_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            for value in getattr(self, field):
                _machine_id(value, f"source_role_interpretation.{field}[]")

    def as_dict(self) -> dict[str, Any]:
        return {
            field: list(getattr(self, field)) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SourceRoleInterpretation":
        payload = _object(value, "source_role_interpretation")
        return cls(
            **{
                field: _strings(
                    payload.get(field),
                    f"source_role_interpretation.{field}",
                    machine_ids=True,
                )
                for field in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class AuthorityBoundary:
    source_role: str
    may_support_screening_or_reasoning: bool
    establishes_binding_conclusion: bool
    rationale: str

    def __post_init__(self) -> None:
        if self.source_role not in {
            "binding",
            "interpretive",
            "empirical",
            "operational",
            "unknown",
        }:
            raise ValueError("authority_boundary.source_role is invalid")
        if type(self.may_support_screening_or_reasoning) is not bool:
            raise ValueError("may_support_screening_or_reasoning must be boolean")
        if type(self.establishes_binding_conclusion) is not bool:
            raise ValueError("establishes_binding_conclusion must be boolean")
        _required(self.rationale, "authority_boundary.rationale")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_role": self.source_role,
            "may_support_screening_or_reasoning": self.may_support_screening_or_reasoning,
            "establishes_binding_conclusion": self.establishes_binding_conclusion,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "AuthorityBoundary":
        payload = _object(value, "authority_boundary")
        return cls(
            source_role=str(payload.get("source_role") or ""),
            may_support_screening_or_reasoning=payload.get(
                "may_support_screening_or_reasoning"
            ),
            establishes_binding_conclusion=payload.get(
                "establishes_binding_conclusion"
            ),
            rationale=_required(
                payload.get("rationale"), "authority_boundary.rationale"
            ),
        )


@dataclass(frozen=True)
class CompetencyQuestion:
    question_id: str
    text: str
    success_condition: str

    def __post_init__(self) -> None:
        _machine_id(self.question_id, "competency_questions[].question_id")
        _required(self.text, "competency_questions[].text")
        _required(self.success_condition, "competency_questions[].success_condition")

    def as_dict(self) -> dict[str, str]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "success_condition": self.success_condition,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CompetencyQuestion":
        payload = _object(value, "competency_questions[]")
        return cls(
            question_id=_machine_id(
                payload.get("question_id"), "competency_questions[].question_id"
            ),
            text=_required(payload.get("text"), "competency_questions[].text"),
            success_condition=_required(
                payload.get("success_condition"),
                "competency_questions[].success_condition",
            ),
        )


@dataclass(frozen=True)
class HardNegativeQuestion:
    question_id: str
    text: str
    distinction_to_preserve: str

    def __post_init__(self) -> None:
        _machine_id(self.question_id, "hard_negative_questions[].question_id")
        _required(self.text, "hard_negative_questions[].text")
        _required(
            self.distinction_to_preserve,
            "hard_negative_questions[].distinction_to_preserve",
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "distinction_to_preserve": self.distinction_to_preserve,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "HardNegativeQuestion":
        payload = _object(value, "hard_negative_questions[]")
        return cls(
            question_id=_machine_id(
                payload.get("question_id"), "hard_negative_questions[].question_id"
            ),
            text=_required(payload.get("text"), "hard_negative_questions[].text"),
            distinction_to_preserve=_required(
                payload.get("distinction_to_preserve"),
                "hard_negative_questions[].distinction_to_preserve",
            ),
        )


@dataclass(frozen=True)
class AlternativeConsidered:
    alternative_id: str
    description: str
    material_difference: str
    disposition: str

    def __post_init__(self) -> None:
        _machine_id(self.alternative_id, "alternatives_considered[].alternative_id")
        _required(self.description, "alternatives_considered[].description")
        _required(
            self.material_difference, "alternatives_considered[].material_difference"
        )
        if self.disposition not in {"selected", "rejected", "unresolved"}:
            raise ValueError("alternatives_considered[].disposition is invalid")

    def as_dict(self) -> dict[str, str]:
        return {
            "alternative_id": self.alternative_id,
            "description": self.description,
            "material_difference": self.material_difference,
            "disposition": self.disposition,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "AlternativeConsidered":
        payload = _object(value, "alternatives_considered[]")
        return cls(
            alternative_id=_machine_id(
                payload.get("alternative_id"),
                "alternatives_considered[].alternative_id",
            ),
            description=_required(
                payload.get("description"), "alternatives_considered[].description"
            ),
            material_difference=_required(
                payload.get("material_difference"),
                "alternatives_considered[].material_difference",
            ),
            disposition=str(payload.get("disposition") or ""),
        )


@dataclass(frozen=True)
class UnresolvedQuestion:
    question_id: str
    question: str
    material: bool
    affected_semantics: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _machine_id(self.question_id, "unresolved_questions[].question_id")
        _required(self.question, "unresolved_questions[].question")
        if type(self.material) is not bool:
            raise ValueError("unresolved_questions[].material must be boolean")
        for value in self.affected_semantics:
            _machine_id(value, "unresolved_questions[].affected_semantics[]")
        for ref in self.evidence_refs:
            _machine_id(ref, "unresolved_questions[].evidence_refs[]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "material": self.material,
            "affected_semantics": list(self.affected_semantics),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "UnresolvedQuestion":
        payload = _object(value, "unresolved_questions[]")
        return cls(
            question_id=_machine_id(
                payload.get("question_id"), "unresolved_questions[].question_id"
            ),
            question=_required(
                payload.get("question"), "unresolved_questions[].question"
            ),
            material=payload.get("material"),
            affected_semantics=_strings(
                payload.get("affected_semantics"),
                "unresolved_questions[].affected_semantics",
                machine_ids=True,
            ),
            evidence_refs=_strings(
                payload.get("evidence_refs"),
                "unresolved_questions[].evidence_refs",
                machine_ids=True,
            ),
        )


@dataclass(frozen=True)
class GraphIntentReview:
    state: str
    reviewed_by: str
    actor_ref: str
    operating_mode: str
    policy_ref: str | None
    rationale_ref: str | None
    decision_evidence_refs: tuple[str, ...]
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if self.state not in {"proposed", "accepted", "edited", "rejected"}:
            raise ValueError("review.state is invalid")
        if self.reviewed_by not in {"user", "system"}:
            raise ValueError("review.reviewed_by is invalid")
        _machine_id(self.actor_ref, "review.actor_ref")
        if self.operating_mode not in {"assisted", "auto_pilot"}:
            raise ValueError("review.operating_mode is invalid")
        _optional_machine_id(self.policy_ref, "review.policy_ref")
        _optional_machine_id(self.rationale_ref, "review.rationale_ref")
        for ref in self.decision_evidence_refs:
            _machine_id(ref, "review.decision_evidence_refs[]")
        if self.operating_mode == "auto_pilot":
            if self.reviewed_by != "system":
                raise ValueError("auto_pilot review must be reviewed_by system")
            if self.state in {"accepted", "edited"}:
                for field in ("policy_ref", "rationale_ref"):
                    if not getattr(self, field):
                        raise ValueError(f"auto_pilot requires {field}")
                if not self.decision_evidence_refs:
                    raise ValueError("auto_pilot requires decision_evidence_refs")
        if (
            self.operating_mode == "assisted"
            and self.reviewed_by == "system"
            and self.state in {"accepted", "edited"}
            and self.fallback_reason is None
        ):
            raise ValueError("fallback requires fallback_reason")
        if self.fallback_reason is not None:
            if self.operating_mode != "assisted":
                raise ValueError("fallback_reason is valid only for assisted review")
            _required(self.fallback_reason, "review.fallback_reason")
            if not self.decision_evidence_refs:
                raise ValueError("fallback requires decision_evidence_refs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reviewed_by": self.reviewed_by,
            "actor_ref": self.actor_ref,
            "operating_mode": self.operating_mode,
            "policy_ref": self.policy_ref,
            "rationale_ref": self.rationale_ref,
            "decision_evidence_refs": list(self.decision_evidence_refs),
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GraphIntentReview":
        payload = _object(value, "review")
        return cls(
            state=str(payload.get("state") or ""),
            reviewed_by=str(payload.get("reviewed_by") or ""),
            actor_ref=_machine_id(payload.get("actor_ref"), "review.actor_ref"),
            operating_mode=str(payload.get("operating_mode") or ""),
            policy_ref=_optional_machine_id(
                payload.get("policy_ref"), "review.policy_ref"
            ),
            rationale_ref=_optional_machine_id(
                payload.get("rationale_ref"), "review.rationale_ref"
            ),
            decision_evidence_refs=_strings(
                payload.get("decision_evidence_refs"),
                "review.decision_evidence_refs",
                machine_ids=True,
            ),
            fallback_reason=(
                str(payload["fallback_reason"])
                if payload.get("fallback_reason") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class GraphIntentBrief:
    graph_intent_id: str
    graph_intent_hash: str
    source_binding: GraphIntentSourceBinding
    document_function: IntentTypeDescription
    represented_or_governed_subject: IntentTypeDescription
    primary_relationship: PrimaryRelationship
    preferred_representation: PreferredRepresentation
    material_dimensions: tuple[MaterialDimension, ...]
    source_role_interpretation: SourceRoleInterpretation
    authority_boundary: AuthorityBoundary
    competency_questions: tuple[CompetencyQuestion, ...]
    hard_negative_questions: tuple[HardNegativeQuestion, ...]
    alternatives_considered: tuple[AlternativeConsidered, ...]
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]
    unresolved_questions: tuple[UnresolvedQuestion, ...]
    clarification_refs: tuple[str, ...]
    review: GraphIntentReview
    predecessor_graph_intent_ref: str | None = None
    successor_graph_intent_refs: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    frame_mapping: FrameMapping | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        _machine_id(self.graph_intent_id, "graph_intent_id")
        _sha256(self.graph_intent_hash, "graph_intent_hash")
        _optional_machine_id(
            self.predecessor_graph_intent_ref, "predecessor_graph_intent_ref"
        )
        for ref in self.evidence_refs:
            _machine_id(ref, "evidence_refs[]")
        for ref in self.clarification_refs:
            _machine_id(ref, "clarification_refs[]")
        if self.successor_graph_intent_refs:
            raise ValueError(
                "successor_graph_intent_refs must be empty because successor links are derived"
            )
        expected = f"sha256:{content_hash(self._hash_payload())}"
        if self.graph_intent_hash != expected:
            raise ValueError("graph_intent_hash does not match graph intent content")
        if self.graph_intent_id != f"graph-intent-{expected[7:31]}":
            raise ValueError("graph_intent_id does not match graph intent content")

    @property
    def compiler_eligible(self) -> bool:
        return self.review.state in {"accepted", "edited"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_intent_id": self.graph_intent_id,
            "graph_intent_hash": self.graph_intent_hash,
            "source_binding": self.source_binding.as_dict(),
            "document_function": self.document_function.as_dict(),
            "represented_or_governed_subject": self.represented_or_governed_subject.as_dict(),
            "primary_relationship": self.primary_relationship.as_dict(),
            "preferred_representation": self.preferred_representation.as_dict(),
            "material_dimensions": [
                item.as_dict() for item in self.material_dimensions
            ],
            "source_role_interpretation": self.source_role_interpretation.as_dict(),
            "authority_boundary": self.authority_boundary.as_dict(),
            "competency_questions": [
                item.as_dict() for item in self.competency_questions
            ],
            "hard_negative_questions": [
                item.as_dict() for item in self.hard_negative_questions
            ],
            "alternatives_considered": [
                item.as_dict() for item in self.alternatives_considered
            ],
            "evidence_refs": list(self.evidence_refs),
            "assumptions": list(self.assumptions),
            "unresolved_questions": [
                item.as_dict() for item in self.unresolved_questions
            ],
            "clarification_refs": list(self.clarification_refs),
            "review": self.review.as_dict(),
            "predecessor_graph_intent_ref": self.predecessor_graph_intent_ref,
            "successor_graph_intent_refs": list(self.successor_graph_intent_refs),
            **({"frame_mapping": self.frame_mapping.as_dict()} if self.frame_mapping else {}),
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("graph_intent_id")
        payload.pop("graph_intent_hash")
        payload.pop("successor_graph_intent_refs")
        return payload

    @classmethod
    def create(cls, content: dict[str, Any]) -> "GraphIntentBrief":
        payload = dict(content)
        payload["schema_version"] = SCHEMA_VERSION
        payload.pop("graph_intent_id", None)
        payload.pop("graph_intent_hash", None)
        payload["primary_relationship"] = PrimaryRelationship.from_dict(
            payload.get("primary_relationship")
        ).as_dict()
        identity_payload = dict(payload)
        identity_payload.pop("successor_graph_intent_refs", None)
        digest = content_hash(identity_payload)
        return cls.from_dict(
            {
                **payload,
                "graph_intent_id": f"graph-intent-{digest[:24]}",
                "graph_intent_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def from_dict(cls, value: Any) -> "GraphIntentBrief":
        payload = _object(value, "graph_intent_brief")

        def list_value(field: str) -> list[Any] | tuple[Any, ...]:
            value = payload.get(field)
            return value if isinstance(value, (list, tuple)) else ()

        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            graph_intent_id=str(payload.get("graph_intent_id") or ""),
            graph_intent_hash=str(payload.get("graph_intent_hash") or ""),
            source_binding=GraphIntentSourceBinding.from_dict(
                payload.get("source_binding")
            ),
            document_function=IntentTypeDescription.from_dict(
                payload.get("document_function"), "document_function"
            ),
            represented_or_governed_subject=IntentTypeDescription.from_dict(
                payload.get("represented_or_governed_subject"),
                "represented_or_governed_subject",
            ),
            primary_relationship=PrimaryRelationship.from_dict(
                payload.get("primary_relationship")
            ),
            preferred_representation=PreferredRepresentation.from_dict(
                payload.get("preferred_representation")
            ),
            material_dimensions=tuple(
                MaterialDimension.from_dict(item)
                for item in list_value("material_dimensions")
            ),
            source_role_interpretation=SourceRoleInterpretation.from_dict(
                payload.get("source_role_interpretation")
            ),
            authority_boundary=AuthorityBoundary.from_dict(
                payload.get("authority_boundary")
            ),
            competency_questions=tuple(
                CompetencyQuestion.from_dict(item)
                for item in list_value("competency_questions")
            ),
            hard_negative_questions=tuple(
                HardNegativeQuestion.from_dict(item)
                for item in list_value("hard_negative_questions")
            ),
            alternatives_considered=tuple(
                AlternativeConsidered.from_dict(item)
                for item in list_value("alternatives_considered")
            ),
            evidence_refs=_strings(
                payload.get("evidence_refs"), "evidence_refs", machine_ids=True
            ),
            assumptions=_strings(payload.get("assumptions"), "assumptions"),
            unresolved_questions=tuple(
                UnresolvedQuestion.from_dict(item)
                for item in list_value("unresolved_questions")
            ),
            clarification_refs=_strings(
                payload.get("clarification_refs"),
                "clarification_refs",
                machine_ids=True,
            ),
            review=GraphIntentReview.from_dict(payload.get("review")),
            frame_mapping=FrameMapping.from_dict(payload["frame_mapping"]) if "frame_mapping" in payload else None,
            predecessor_graph_intent_ref=_optional_machine_id(
                payload.get("predecessor_graph_intent_ref"),
                "predecessor_graph_intent_ref",
            ),
            successor_graph_intent_refs=_strings(
                payload.get("successor_graph_intent_refs"),
                "successor_graph_intent_refs",
                machine_ids=True,
            ),
        )


__all__ = [
    "SCHEMA_VERSION",
    "AlternativeConsidered",
    "AuthorityBoundary",
    "CompetencyQuestion",
    "GraphIntentBrief",
    "GraphIntentReview",
    "GraphIntentSourceBinding",
    "HardNegativeQuestion",
    "IntentTypeDescription",
    "MaterialDimension",
    "PreferredRepresentation",
    "PrimaryRelationship",
    "SourceRoleInterpretation",
    "UnresolvedQuestion",
]
