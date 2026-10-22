"""Release-independent canonical restriction-pattern contract and legacy projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from contracts.semantic_constraint_frame import ComposedFramePatternV01, FramePatternV01
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_SCHEMA_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_DECIMAL = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_DECIMAL_KEYS = {"amount", "normalized_amount", "decimal", "quantity"}
_PATTERN_PARSE_TOKEN = object()


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _ascii_ref(value: str, field: str) -> str:
    if not isinstance(value, str) or _ASCII_REF.fullmatch(value) is None or not value.isascii():
        raise ValueError(f"{field} must be an ASCII-stable identifier/reference")
    return value


def _canonical_ref_set(value: Sequence[str], field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    items = tuple(_ascii_ref(item, f"{field}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} contains a duplicate identity")
    return tuple(sorted(items, key=lambda item: item.encode("utf-8")))


def _validate_json_value(value: Any, field: str, *, key: str | None = None) -> None:
    if isinstance(value, float):
        raise ValueError(f"{field} contains a forbidden JSON floating-point value")
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError(f"{field} contains an integer outside the JSON safe integer range")
        raise ValueError(f"{field} contains an integer outside a schema-declared counter")
    elif isinstance(value, str) and key in _DECIMAL_KEYS:
        if value == "-0" or _DECIMAL.fullmatch(value) is None:
            raise ValueError(f"{field}.{key} must be a canonical decimal string")
    elif isinstance(value, Mapping):
        for nested_key, nested in value.items():
            if not isinstance(nested_key, str) or _SCHEMA_KEY.fullmatch(nested_key) is None:
                raise ValueError(f"{field} contains a non-ASCII-stable schema key")
            _validate_json_value(nested, field, key=nested_key)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json_value(nested, field)


def _evidence_identity(item: EvidenceRef, source: "SourceMetadata") -> str:
    if item.organization_id != source._organization_id or item.workspace_id != source._workspace_id:
        raise ValueError("EvidenceRef organization/workspace scope mismatch")
    if item.source_identity_ref != source.source_identity_ref:
        raise ValueError("EvidenceRef SourceIdentity mismatch")
    if item.created_at.tzinfo is None:
        raise ValueError("EvidenceRef.createdAt must be timezone-aware")
    if item.selector_schema_version != item.selector.schema_version:
        raise ValueError("EvidenceRef selector schema mismatch")
    if item.selector_hash != item.selector.selector_hash():
        raise ValueError("EvidenceRef selector hash mismatch")
    expected = evidence_ref_identity(
        organization_id=item.organization_id,
        workspace_id=item.workspace_id,
        source_identity_ref=item.source_identity_ref,
        selector_hash=item.selector_hash,
        evidence_role=item.evidence_role,
    )
    if item.evidence_ref != expected:
        raise ValueError("EvidenceRef identity mismatch")
    provenance = item.provenance
    if provenance.get("source_hash") != source.source_hash:
        raise ValueError("EvidenceRef source provenance mismatch")
    if provenance.get("canonical_source_ir_ref") != source.canonical_source_ir_ref:
        raise ValueError("EvidenceRef CanonicalSourceIR provenance mismatch")
    if provenance.get("canonical_source_ir_hash") != source.canonical_source_ir_hash:
        raise ValueError("EvidenceRef CanonicalSourceIR provenance mismatch")
    return item.evidence_ref


class Composition(_Contract):
    operator: Literal["atomic", "all_of", "any_of", "ordered_all_of"]
    member_pattern_refs: tuple[str, ...]
    condition_pattern_refs: tuple[str, ...]
    exception_pattern_refs: tuple[str, ...]

    @field_validator("condition_pattern_refs", "exception_pattern_refs", mode="before")
    @classmethod
    def canonical_sets(cls, value: Sequence[str], info: ValidationInfo) -> tuple[str, ...]:
        return _canonical_ref_set(value, info.field_name)

    @field_validator("member_pattern_refs", mode="before")
    @classmethod
    def members(cls, value: Sequence[str], info: ValidationInfo) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("member_pattern_refs must be an array")
        items = tuple(_ascii_ref(item, "member_pattern_refs[]") for item in value)
        if len(items) != len(set(items)):
            raise ValueError("member_pattern_refs contains a duplicate identity")
        data = info.data
        if data.get("operator") in {"all_of", "any_of"}:
            return tuple(sorted(items, key=lambda item: item.encode("utf-8")))
        return items

    @model_validator(mode="after")
    def no_contradiction(self) -> "Composition":
        if set(self.condition_pattern_refs) & set(self.exception_pattern_refs):
            raise ValueError("a pattern cannot be both condition and exception")
        return self


class SemanticDimensions(_Contract):
    target: dict[str, Any] | None
    regulated_actor: dict[str, Any] | None
    constraint_dimension: str
    operator: str
    value: dict[str, Any] | None
    threshold: dict[str, Any] | None
    unit: str | None
    modality: str
    stage: str | None
    scope: dict[str, Any] | None
    consequence: dict[str, Any] | None

    @model_validator(mode="after")
    def canonical_values(self) -> "SemanticDimensions":
        for name in ("target", "regulated_actor", "value", "threshold", "scope", "consequence"):
            _validate_json_value(getattr(self, name), f"semantic_dimensions.{name}")
        return self


class SourceMetadata(_Contract):
    source_identity_ref: str
    source_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_source_ir_ref: str
    canonical_source_ir_hash: str = Field(pattern=SHA256_PATTERN)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)
    source_classification: str
    # Private validation scope, injected transiently by the parent contract.
    _organization_id: str = ""
    _workspace_id: str = ""

    @model_validator(mode="after")
    def identifiers(self) -> "SourceMetadata":
        _ascii_ref(self.source_identity_ref, "source_identity_ref")
        _ascii_ref(self.canonical_source_ir_ref, "canonical_source_ir_ref")
        _ascii_ref(self.source_classification, "source_classification")
        identities = [item.evidence_ref for item in self.evidence_refs]
        if len(identities) != len(set(identities)):
            raise ValueError("evidence_refs contains a duplicate evidenceRef identity")
        ordered = tuple(sorted(self.evidence_refs, key=lambda item: (
            item.evidence_ref.encode("utf-8"),
            canonical_json(item.model_dump(mode="json", by_alias=True)).encode("utf-8"),
        )))
        object.__setattr__(self, "evidence_refs", ordered)
        return self


class WitnessMetadata(_Contract):
    positive_exemplar_refs: tuple[str, ...]
    hard_negative_refs: tuple[str, ...]

    @field_validator("positive_exemplar_refs", "hard_negative_refs", mode="before")
    @classmethod
    def canonical_sets(cls, value: Sequence[str], info: ValidationInfo) -> tuple[str, ...]:
        return _canonical_ref_set(value, info.field_name)


class LineageMetadata(_Contract):
    graph_intent_brief_ref: str
    graph_intent_brief_hash: str = Field(pattern=SHA256_PATTERN)
    schema_candidate_ref: str
    schema_candidate_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def identifiers(self) -> "LineageMetadata":
        _ascii_ref(self.graph_intent_brief_ref, "graph_intent_brief_ref")
        _ascii_ref(self.schema_candidate_ref, "schema_candidate_ref")
        return self


class AuthorityMetadata(_Contract):
    authority_role: Literal["accepted_release_lineage_input"]
    governed_rule_projection_ref: None
    governed_rule_projection_hash: None

    @model_validator(mode="before")
    @classmethod
    def projections_are_null(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and (
            value.get("governed_rule_projection_ref") is not None
            or value.get("governed_rule_projection_hash") is not None
        ):
            raise ValueError("governed rule projection fields are permanently null")
        return value


class CanonicalRestrictionPatternV1(_Contract):
    schema_version: Literal["canonical_restriction_pattern.v1"] = "canonical_restriction_pattern.v1"
    pattern_id: str
    pattern_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    pattern_kind: Literal["atomic", "composed"]
    frames: tuple[dict[str, Any], ...]
    composition: Composition
    semantic_dimensions: SemanticDimensions
    source_metadata: SourceMetadata
    witness_metadata: WitnessMetadata
    lexical_evidence_refs: tuple[str, ...]
    lineage_metadata: LineageMetadata
    authority_metadata: AuthorityMetadata

    @model_validator(mode="before")
    @classmethod
    def explicitly_unverified_entrypoint_only(cls, value: Any, info: ValidationInfo) -> Any:
        if (info.context or {}).get("pattern_parse_token") is not _PATTERN_PARSE_TOKEN:
            raise TypeError(
                "canonical pattern parsing requires an explicitly named unverified entrypoint"
            )
        return value

    @property
    def pattern_ref(self) -> str:
        return self.pattern_id

    @field_validator("lexical_evidence_refs", mode="before")
    @classmethod
    def lexical_refs(cls, value: Sequence[str]) -> tuple[str, ...]:
        return _canonical_ref_set(value, "lexical_evidence_refs")

    @field_validator("frames", mode="before")
    @classmethod
    def canonical_frames(cls, value: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("frames must be an array")
        frames = tuple(dict(frame) for frame in value)
        hashes: list[str] = []
        for frame in frames:
            if set(frame).difference({key for key in frame if _SCHEMA_KEY.fullmatch(key)}):
                raise ValueError("frames contain a non-ASCII-stable schema key")
            frame_hash = frame.get("frame_hash")
            if not isinstance(frame_hash, str) or re.fullmatch(SHA256_PATTERN, frame_hash) is None:
                raise ValueError("each frame requires frame_hash")
            _validate_json_value(frame, "frames")
            hashes.append(frame_hash)
        if len(hashes) != len(set(hashes)):
            raise ValueError("frames contain a duplicate frame_hash")
        return tuple(sorted(frames, key=lambda frame: frame["frame_hash"].encode("utf-8")))

    @model_validator(mode="after")
    def validate_contract(self, info: ValidationInfo) -> "CanonicalRestrictionPatternV1":
        _ascii_ref(self.pattern_id, "pattern_id")
        _ascii_ref(self.organization_id, "organization_id")
        _ascii_ref(self.workspace_id, "workspace_id")
        source = self.source_metadata
        object.__setattr__(source, "_organization_id", self.organization_id)
        object.__setattr__(source, "_workspace_id", self.workspace_id)
        for evidence in source.evidence_refs:
            _evidence_identity(evidence, source)
        if self.pattern_kind == "atomic":
            if self.composition.operator != "atomic" or self.composition.member_pattern_refs:
                raise ValueError("atomic patterns require atomic composition with no members")
        else:
            if self.composition.operator == "atomic" or len(self.composition.member_pattern_refs) < 2:
                raise ValueError("composed patterns require a non-atomic operator and at least two members")
            if self.pattern_id in self.composition.member_pattern_refs:
                raise ValueError("composition cycle includes the pattern itself")
        if not (info.context or {}).get("skip_hash"):
            expected = "sha256:" + content_hash(_hash_payload(self))
            if self.pattern_hash != expected:
                raise ValueError("pattern_hash mismatch")
        return self

def _reject_composition_cycles(pattern_id: str, graph: Mapping[str, Sequence[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(ref: str) -> None:
        if ref in visiting:
            raise ValueError("composition cycle detected")
        if ref in visited:
            return
        visiting.add(ref)
        for nested in graph.get(ref, ()):
            visit(nested)
        visiting.remove(ref)
        visited.add(ref)

    visit(pattern_id)


def parse_canonical_restriction_pattern_unverified(
    payload: Mapping[str, Any],
    *,
    component_artifacts: Sequence[tuple[bytes, str]] = (),
) -> CanonicalRestrictionPatternV1:
    """Parse/hash canonical bytes only; never use this as an authority decision."""
    raw = dict(payload)
    supplied_hash = raw.get("pattern_hash")
    provisional = CanonicalRestrictionPatternV1.__pydantic_validator__.validate_python(
        {**raw, "pattern_hash": supplied_hash or "sha256:" + "0" * 64},
        context={"skip_hash": supplied_hash is None, "pattern_parse_token": _PATTERN_PARSE_TOKEN},
    )
    if supplied_hash is not None:
        pattern = provisional
    else:
        actual = "sha256:" + content_hash(_hash_payload(provisional))
        pattern = CanonicalRestrictionPatternV1.__pydantic_validator__.validate_python(
            {**provisional.model_dump(mode="json"), "pattern_hash": actual},
            context={"pattern_parse_token": _PATTERN_PARSE_TOKEN},
        )
    _validate_component_artifacts_unverified(pattern, component_artifacts)
    return pattern


def create_canonical_restriction_pattern_unverified(
    *,
    component_artifacts: Sequence[tuple[bytes, str]] = (),
    **payload: Any,
) -> CanonicalRestrictionPatternV1:
    """Create deterministic bytes only; this function confers no authority."""
    raw = dict(payload)
    raw.pop("pattern_hash", None)
    return parse_canonical_restriction_pattern_unverified(
        raw, component_artifacts=component_artifacts
    )


def _validate_component_artifacts_unverified(
    pattern: CanonicalRestrictionPatternV1,
    component_artifacts: Sequence[tuple[bytes, str]],
) -> None:
    if pattern.pattern_kind == "atomic":
        if component_artifacts:
            raise ValueError("atomic patterns do not accept component artifacts")
        return
    if not isinstance(component_artifacts, (list, tuple)):
        raise ValueError("component_artifacts must be an array")
    components: dict[str, CanonicalRestrictionPatternV1] = {}
    for item in component_artifacts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("component artifacts require exact bytes/hash pairs")
        artifact_bytes, artifact_hash = item
        if not isinstance(artifact_bytes, bytes):
            raise ValueError("component artifact bytes are required")
        if re.fullmatch(SHA256_PATTERN, artifact_hash) is None:
            raise ValueError("component artifact hash is invalid")
        try:
            decoded = json.loads(artifact_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("component artifact bytes are not UTF-8 JSON") from exc
        component = CanonicalRestrictionPatternV1.__pydantic_validator__.validate_python(
            {**decoded, "pattern_hash": artifact_hash},
            context={"pattern_parse_token": _PATTERN_PARSE_TOKEN},
        )
        if canonical_pattern_bytes_unverified(component) != artifact_bytes:
            raise ValueError("component artifact bytes are not exact canonical bytes")
        if component.pattern_ref in components:
            raise ValueError("duplicate component artifact identity")
        components[component.pattern_ref] = component
    graph: dict[str, Sequence[str]] = {
        pattern.pattern_id: pattern.composition.member_pattern_refs
    }
    pending = list(pattern.composition.member_pattern_refs)
    while pending:
        ref = pending.pop()
        component = components.get(ref)
        if component is None:
            raise ValueError("composed patterns require resolved component bytes and hashes")
        graph[ref] = component.composition.member_pattern_refs
        pending.extend(item for item in component.composition.member_pattern_refs if item not in graph)
    _reject_composition_cycles(pattern.pattern_id, graph)


def _hash_payload(pattern: CanonicalRestrictionPatternV1) -> dict[str, Any]:
    # Pattern-owned keys are already snake_case; EvidenceRef retains its frozen
    # cross-boundary camelCase aliases inside the nested source metadata.
    payload = pattern.model_dump(mode="json", by_alias=True)
    payload.pop("pattern_hash")
    return payload


def canonical_pattern_bytes_unverified(pattern: CanonicalRestrictionPatternV1) -> bytes:
    """Return exact canonical UTF-8 bytes of the Layer-1 hashed payload."""
    return canonical_json(_hash_payload(pattern)).encode("utf-8")


def canonical_pattern_hash_unverified(pattern: CanonicalRestrictionPatternV1) -> str:
    """Hash canonical bytes only; this function confers no authority."""
    return "sha256:" + sha256(canonical_pattern_bytes_unverified(pattern)).hexdigest()


@dataclass(frozen=True)
class LegacyPatternAdapterContextUnverified:
    organization_id: str
    workspace_id: str
    source_identity_ref: str
    source_hash: str
    canonical_source_ir_ref: str
    canonical_source_ir_hash: str
    evidence_refs: tuple[EvidenceRef, ...]
    source_classification: str
    positive_exemplar_refs: tuple[str, ...]
    hard_negative_refs: tuple[str, ...]
    lexical_evidence_refs: tuple[str, ...]
    graph_intent_brief_ref: str
    graph_intent_brief_hash: str
    schema_candidate_ref: str
    schema_candidate_hash: str


@dataclass(frozen=True)
class LegacyPatternProjectionUnverified:
    pattern: CanonicalRestrictionPatternV1
    original_artifact_bytes: bytes
    original_artifact_hash: str
    original_pattern_hash: str
    successor_projection_hash: str


def _frame_payload(frame: Any) -> dict[str, Any]:
    payload = asdict(frame)
    frame_hash = "sha256:" + content_hash(payload)
    return {"frame_hash": frame_hash, **payload}


def _base_projection_payload(*, pattern_id: str, frames: Sequence[dict[str, Any]], context: LegacyPatternAdapterContextUnverified) -> dict[str, Any]:
    return {
        "schema_version": "canonical_restriction_pattern.v1",
        "pattern_id": pattern_id,
        "organization_id": context.organization_id,
        "workspace_id": context.workspace_id,
        "frames": list(frames),
        "semantic_dimensions": {
            "target": None, "regulated_actor": None, "constraint_dimension": "legacy_frame",
            "operator": "legacy", "value": None, "threshold": None, "unit": None,
            "modality": "legacy", "stage": None, "scope": None, "consequence": None,
        },
        "source_metadata": {
            "source_identity_ref": context.source_identity_ref,
            "source_hash": context.source_hash,
            "canonical_source_ir_ref": context.canonical_source_ir_ref,
            "canonical_source_ir_hash": context.canonical_source_ir_hash,
            "evidence_refs": list(context.evidence_refs),
            "source_classification": context.source_classification,
        },
        "witness_metadata": {
            "positive_exemplar_refs": context.positive_exemplar_refs,
            "hard_negative_refs": context.hard_negative_refs,
        },
        "lexical_evidence_refs": context.lexical_evidence_refs,
        "lineage_metadata": {
            "graph_intent_brief_ref": context.graph_intent_brief_ref,
            "graph_intent_brief_hash": context.graph_intent_brief_hash,
            "schema_candidate_ref": context.schema_candidate_ref,
            "schema_candidate_hash": context.schema_candidate_hash,
        },
        "authority_metadata": {
            "authority_role": "accepted_release_lineage_input",
            "governed_rule_projection_ref": None,
            "governed_rule_projection_hash": None,
        },
    }


def _projection(
    pattern: CanonicalRestrictionPatternV1,
    original_hash: str,
    original_bytes: bytes,
) -> LegacyPatternProjectionUnverified:
    return LegacyPatternProjectionUnverified(
        pattern=pattern,
        original_artifact_bytes=original_bytes,
        original_artifact_hash="sha256:" + sha256(original_bytes).hexdigest(),
        original_pattern_hash=original_hash,
        successor_projection_hash=pattern.pattern_hash,
    )


def _validate_legacy_bytes(pattern: Any, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError("legacy artifact bytes are required")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("legacy artifact bytes are not valid UTF-8 JSON") from exc
    if decoded != json.loads(canonical_json(asdict(pattern))):
        raise ValueError("legacy artifact bytes do not correspond to the supplied artifact")


def project_frame_pattern_v01_unverified(
    pattern: FramePatternV01,
    *,
    context: LegacyPatternAdapterContextUnverified | None = None,
    legacy_artifact_bytes: bytes,
) -> LegacyPatternProjectionUnverified:
    """Project legacy history without treating absent source metadata as inferable."""
    if type(context) is not LegacyPatternAdapterContextUnverified:
        raise TypeError("an explicitly unverified adapter context is required")
    _validate_legacy_bytes(pattern, legacy_artifact_bytes)
    payload = _base_projection_payload(pattern_id=pattern.pattern_id, frames=[_frame_payload(pattern.frame)], context=context)
    payload.update({
        "pattern_kind": "atomic",
        "composition": {"operator": "atomic", "member_pattern_refs": [], "condition_pattern_refs": [], "exception_pattern_refs": []},
    })
    return _projection(
        create_canonical_restriction_pattern_unverified(**payload),
        pattern.pattern_hash,
        legacy_artifact_bytes,
    )


def project_composed_frame_pattern_v01_unverified(
    pattern: ComposedFramePatternV01,
    *,
    context: LegacyPatternAdapterContextUnverified | None = None,
    legacy_artifact_bytes: bytes,
) -> LegacyPatternProjectionUnverified:
    """Project legacy ALL_OF history while retaining historical bytes and hash."""
    if type(context) is not LegacyPatternAdapterContextUnverified:
        raise TypeError("an explicitly unverified adapter context is required")
    _validate_legacy_bytes(pattern, legacy_artifact_bytes)
    members = [item.pattern_id for item in pattern.constituents]
    unique_frames = {
        frame["frame_hash"]: frame
        for frame in (_frame_payload(item.frame) for item in pattern.constituents)
    }
    payload = _base_projection_payload(
        pattern_id=pattern.pattern_id,
        frames=list(unique_frames.values()),
        context=context,
    )
    payload.update({
        "pattern_kind": "composed",
        "composition": {"operator": "all_of", "member_pattern_refs": members, "condition_pattern_refs": [], "exception_pattern_refs": []},
    })
    component_artifacts: list[tuple[bytes, str]] = []
    for item in pattern.constituents:
        member_payload = _base_projection_payload(
            pattern_id=item.pattern_id,
            frames=[_frame_payload(item.frame)],
            context=context,
        )
        member_payload.update({
            "pattern_kind": "atomic",
            "composition": {"operator": "atomic", "member_pattern_refs": [], "condition_pattern_refs": [], "exception_pattern_refs": []},
        })
        member = create_canonical_restriction_pattern_unverified(**member_payload)
        component_artifacts.append((canonical_pattern_bytes_unverified(member), member.pattern_hash))
    canonical = create_canonical_restriction_pattern_unverified(
        **payload, component_artifacts=component_artifacts
    )
    return _projection(canonical, pattern.pattern_hash, legacy_artifact_bytes)
