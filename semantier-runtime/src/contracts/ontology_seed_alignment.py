"""Explicit reviewed seed alignment and provisional source-grounded deltas."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.canonical_source_ir import SourceLocator
from contracts.normative_semantics import NormativeFrameV1
from contracts.ontology_building import ExploratoryGraphNodeV1, ExploratoryGraphEdgeV1
from contracts.ontology_seeds import StructuralTripleV1
from eos.artifact_hashing import content_hash

HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
ID = r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$"
AlignmentRelation = Literal["EXACT", "BROADER", "NARROWER", "RELATED", "DISTINCT", "UNRESOLVED"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
                raise ValueError("alignment timestamps must be timezone-aware UTC")
        return self


class ExactAlignmentInputV1(_Contract):
    artifact_ref: str
    artifact_hash: str = Field(pattern=HASH_PATTERN)
    payload_hash: str = Field(pattern=HASH_PATTERN)

    @classmethod
    def of(cls, artifact: BaseModel, ref_field: str, hash_field: str):
        return cls(artifact_ref=getattr(artifact, ref_field), artifact_hash=getattr(artifact, hash_field),
                   payload_hash="sha256:" + content_hash(artifact.model_dump(mode="json")))


class AlignmentInputPinsV1(_Contract):
    baseline_manifest: ExactAlignmentInputV1
    baseline_snapshot: ExactAlignmentInputV1
    input_envelope: ExactAlignmentInputV1
    seed_snapshot: ExactAlignmentInputV1
    seed_profile: ExactAlignmentInputV1


INPUT_PIN_FIELDS = {
    "baseline_manifest": ("build_ref", "build_hash"),
    "baseline_snapshot": ("snapshot_ref", "snapshot_hash"),
    "input_envelope": ("envelope_ref", "envelope_hash"),
    "seed_snapshot": ("snapshot_ref", "snapshot_hash"),
    "seed_profile": ("profile_ref", "profile_hash"),
}


class CanonicalQuoteV1(_Contract):
    block_id: str
    evidence_ref: str
    source_locator: SourceLocator
    quote: str = Field(min_length=1)


class AlignmentDecisionV1(_Contract):
    alignment_id: str = Field(pattern=ID)
    source_node_id: str
    seed_iri: str
    relation: AlignmentRelation
    rationale: str = Field(min_length=1)
    assessment_method: Literal["human"] = "human"
    canonical_consistency: Literal["CONSISTENT", "CONFLICT", "UNRESOLVED"] = "UNRESOLVED"
    canonical_evidence: tuple[CanonicalQuoteV1, ...] = Field(min_length=1)


class SourceDefinitionExtensionV1(_Contract):
    extension_kind: Literal["definition"] = "definition"
    extension_id: str = Field(pattern=ID)
    alignment_id: str = Field(pattern=ID)
    definition_kind: Literal["entity", "concept", "relation"]
    display_name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    canonical_evidence: tuple[CanonicalQuoteV1, ...] = Field(min_length=1)


class SourceNormativeExtensionV1(_Contract):
    extension_kind: Literal["normative_frame"] = "normative_frame"
    extension_id: str = Field(pattern=ID)
    alignment_id: str = Field(pattern=ID)
    frame: NormativeFrameV1
    canonical_evidence: tuple[CanonicalQuoteV1, ...] = Field(min_length=1)


SourceExtensionV1 = Annotated[Union[SourceDefinitionExtensionV1, SourceNormativeExtensionV1], Field(discriminator="extension_kind")]


class _Addressed(_Contract):
    REF: ClassVar[str]
    HASH: ClassVar[str]
    PREFIX: ClassVar[str]

    @model_validator(mode="after")
    def bound(self):
        digest = content_hash(self.model_dump(mode="json", exclude={self.REF, self.HASH}))
        if getattr(self, self.REF) != f"{self.PREFIX}:{digest}" or getattr(self, self.HASH) != f"sha256:{digest}":
            raise ValueError("alignment artifact ref/hash does not bind exact content")
        return self

    @classmethod
    def create(cls, **values):
        provisional = cls.model_construct(**{**values, cls.REF: "pending", cls.HASH: "sha256:" + "0" * 64})
        material = provisional.model_dump(mode="json", exclude={cls.REF, cls.HASH})
        digest = content_hash(material)
        return cls(**{**values, cls.REF: f"{cls.PREFIX}:{digest}", cls.HASH: f"sha256:{digest}"})


class OntologySeedAlignmentCommandV1(_Addressed):
    """Server-authored reviewed command; route orchestration resolves its actor."""
    REF = "alignment_ref"
    HASH = "alignment_hash"
    PREFIX = "ontology-seed-alignment"
    schema_version: Literal["ontology_seed_alignment.v1"] = "ontology_seed_alignment.v1"
    alignment_ref: str
    alignment_hash: str = Field(pattern=HASH_PATTERN)
    organization_id: str
    workspace_id: str
    actor_ref: str = Field(min_length=1)
    reviewed_at: datetime
    inputs: AlignmentInputPinsV1
    decisions: tuple[AlignmentDecisionV1, ...] = Field(min_length=1)
    extensions: tuple[SourceExtensionV1, ...] = ()
    authority_state: Literal["provisional"] = "provisional"

    @model_validator(mode="after")
    def unique_decisions(self):
        ids = [decision.alignment_id for decision in self.decisions]
        pairs = [(decision.source_node_id, decision.seed_iri) for decision in self.decisions]
        extensions = [extension.extension_id for extension in self.extensions]
        if len(set(ids)) != len(ids) or len(set(pairs)) != len(pairs) or len(set(extensions)) != len(extensions):
            raise ValueError("alignment decisions and extensions must have unique identities")
        if any(extension.alignment_id not in ids for extension in self.extensions):
            raise ValueError("extension has no explicit alignment decision")
        return self

    @classmethod
    def for_inputs(cls, *, actor_ref, reviewed_at, decisions, extensions=(), **artifacts):
        pins = AlignmentInputPinsV1(**{name: ExactAlignmentInputV1.of(artifacts[name], *fields) for name, fields in INPUT_PIN_FIELDS.items()})
        return cls.create(organization_id=artifacts["baseline_manifest"].organization_id,
            workspace_id=artifacts["baseline_manifest"].workspace_id, actor_ref=actor_ref, reviewed_at=reviewed_at,
            inputs=pins, decisions=tuple(decisions), extensions=tuple(extensions))


class SeedDefinitionContributionV1(_Contract):
    alignment_id: str
    seed_iri: str
    source_node_id: str
    relation: AlignmentRelation
    disposition: Literal["reused", "specialized", "extended", "rejected", "unresolved"]
    seed_definition: tuple[StructuralTripleV1, ...]
    canonical_evidence_refs: tuple[str, ...]
    rationale: str


class SeedDerivationV1(_Contract):
    edge_kind: Literal["seed_derivation"] = "seed_derivation"
    seed_snapshot_ref: str
    seed_snapshot_hash: str = Field(pattern=HASH_PATTERN)
    seed_iri: str
    candidate_node_id: str
    alignment_id: str


class AlignmentConflictV1(_Contract):
    alignment_id: str
    conflict_kind: Literal["canonical_conflict", "unresolved_consistency", "distinct_definition", "no_semantic_extension"]
    message: str
    canonical_evidence_refs: tuple[str, ...]


class SourceGroundedSeedDeltaV1(_Addressed):
    REF = "delta_ref"
    HASH = "delta_hash"
    PREFIX = "source-grounded-seed-delta"
    schema_version: Literal["source_grounded_seed_delta.v1"] = "source_grounded_seed_delta.v1"
    delta_ref: str
    delta_hash: str = Field(pattern=HASH_PATTERN)
    organization_id: str
    workspace_id: str
    alignment_ref: str
    alignment_hash: str = Field(pattern=HASH_PATTERN)
    inputs: AlignmentInputPinsV1
    contributions: tuple[SeedDefinitionContributionV1, ...]
    added_nodes: tuple[ExploratoryGraphNodeV1, ...]
    added_edges: tuple[ExploratoryGraphEdgeV1, ...]
    normative_frames: tuple[NormativeFrameV1, ...]
    seed_derivations: tuple[SeedDerivationV1, ...]
    conflicts: tuple[AlignmentConflictV1, ...]
    applied_extension_ids: tuple[str, ...]
    structural_measurements: dict[str, int | bool | str]
    comparison_verdict: Literal["INCONCLUSIVE"] = "INCONCLUSIVE"
    authority_state: Literal["provisional"] = "provisional"
    created_at: datetime


class OntologySeedComparisonV1(_Addressed):
    REF = "comparison_ref"
    HASH = "comparison_hash"
    PREFIX = "ontology-seed-comparison"
    schema_version: Literal["ontology_seed_comparison.v1"] = "ontology_seed_comparison.v1"
    comparison_ref: str
    comparison_hash: str = Field(pattern=HASH_PATTERN)
    organization_id: str
    workspace_id: str
    mode: Literal["seed_to_candidate"] = "seed_to_candidate"
    seed_snapshot: ExactAlignmentInputV1
    candidate_snapshot: ExactAlignmentInputV1
    delta_ref: str
    delta_hash: str = Field(pattern=HASH_PATTERN)
    contributions: tuple[SeedDefinitionContributionV1, ...]
    verdict: Literal["INCONCLUSIVE"] = "INCONCLUSIVE"
    reason: str = "No independently evaluated paired competency obligations are supplied."
    created_at: datetime
