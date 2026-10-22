"""Generic corpus-to-graph-schema candidate contract."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from contracts.knowledge_relation_taxonomy import (
    DefinitionScope,
    ProjectionShape,
    RelationFamilyId,
    RepresentationKind,
)
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_identity,
)


SCHEMA_VERSION = "corpus_graph_schema_candidate.v1"
V2_SCHEMA_VERSION = "corpus_graph_schema_candidate.v2"
V3_SCHEMA_VERSION = "corpus_graph_schema_candidate.v3"
V4_SCHEMA_VERSION = "corpus_graph_schema_candidate.v4"
_V3_FIELDS = {
    "schema_version",
    "candidate_id",
    "candidate_hash",
    "source_structure_profile",
    "semantic_role_proposals",
    "proposed_node_types",
    "proposed_relation_types",
    "proposed_frame_types",
    "proposed_event_types",
    "competency_questions",
    "representation_decisions",
    "extraction_contract_hints",
    "unresolved_ambiguities",
    "source_evidence_refs",
    "source_binding",
    "semantica_schema_evidence",
    "synthesis_decisions",
    "document_understanding",
    "graph_layer_coverage",
    "extraction_plan",
    "assumptions",
    "clarification_refs",
    "predecessor_candidate_ref",
    "successor_candidate_refs",
    "graph_intent_binding",
    "compiler_provenance",
    "competency_evaluation",
    "hard_negative_evaluation",
}
_V4_FIELDS = frozenset(_V3_FIELDS)
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELATION_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_TAXONOMY_REF = re.compile(
    r"^knowledge-relation-taxonomy:[a-z][a-z0-9_.-]*@v[1-9][0-9]*$"
)


def _ascii(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be ASCII-stable") from exc
    return text


def _sha256_ref(value: Any, field_name: str) -> str:
    text = str(value or "")
    if _SHA256_REF.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a sha256 ref")
    return text


def _tuple_of_dicts(value: Any, field_name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must contain at least one item")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} items must be objects")
    return tuple(dict(item) for item in value)


def _optional_tuple_of_dicts(
    value: Any, field_name: str
) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    _validate_dict_collection(value, field_name)
    return tuple(dict(item) for item in value)


def _validate_dict_collection(value: Any, field_name: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of objects")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} items must be objects")


def _relation_identifier(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or _RELATION_IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} must be an ASCII-stable relation identifier")
    return value


def _canonical_relation_refs(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    refs = tuple(_relation_identifier(item, f"{field_name}[]") for item in value)
    if len(refs) != len(set(refs)):
        raise ValueError(f"{field_name} contains duplicate values")
    return tuple(sorted(refs, key=lambda item: item.encode("utf-8")))


class RelationTypeCandidate(BaseModel):
    """Exact taxonomy-bound relation definition proposed by Candidate v4."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_type_id: str
    family_id: RelationFamilyId
    definition_scope: DefinitionScope
    namespace: str
    representation_kind: RepresentationKind
    projection_shape: ProjectionShape
    source_type_refs: tuple[str, ...]
    target_type_refs: tuple[str, ...]
    source_role_refs: tuple[str, ...]
    target_role_refs: tuple[str, ...]
    directed: bool
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)
    taxonomy_ref: str
    taxonomy_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    profile_ref: str | None
    profile_hash: str | None = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    profile_relation_catalog_hash: str | None = Field(
        pattern=r"^sha256:[0-9a-f]{64}$"
    )
    profile_relation_type_proposal_ref: str | None = None
    profile_relation_type_proposal_hash: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )

    @field_validator("relation_type_id", "namespace")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _relation_identifier(value, info.field_name)

    @field_validator("taxonomy_ref")
    @classmethod
    def validate_taxonomy_ref(cls, value: str) -> str:
        if _TAXONOMY_REF.fullmatch(value) is None:
            raise ValueError("taxonomy_ref is invalid")
        return value

    @field_validator(
        "source_type_refs",
        "target_type_refs",
        "source_role_refs",
        "target_role_refs",
        mode="before",
    )
    @classmethod
    def canonicalize_refs(cls, value: Any, info) -> tuple[str, ...]:
        return _canonical_relation_refs(value, info.field_name)

    @field_validator("directed", mode="before")
    @classmethod
    def validate_directed(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("directed must be a boolean")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def canonicalize_evidence(
        cls, value: tuple[EvidenceRef, ...]
    ) -> tuple[EvidenceRef, ...]:
        identities = [item.evidence_ref for item in value]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "relation_type_candidate contains duplicate EvidenceRef identity"
            )
        return tuple(
            sorted(
                value,
                key=lambda item: (
                    item.evidence_ref.encode("utf-8"),
                    canonical_json(
                        item.model_dump(
                            mode="json", by_alias=True, exclude_none=False
                        )
                    ).encode("utf-8"),
                ),
            )
        )

    @model_validator(mode="after")
    def validate_definition(self) -> "RelationTypeCandidate":
        if not self.relation_type_id.startswith(f"{self.namespace}."):
            raise ValueError("relation_type_id must be qualified by namespace")
        if (
            self.representation_kind == "binary_edge"
            and self.projection_shape != "binary"
        ):
            raise ValueError("binary_edge requires projection_shape=binary")
        if (
            self.representation_kind == "frame_role"
            and self.projection_shape != "frame"
        ):
            raise ValueError("frame_role requires projection_shape=frame")
        profile_pins = (
            self.profile_ref,
            self.profile_hash,
            self.profile_relation_catalog_hash,
        )
        if self.definition_scope == "canonical":
            if self.namespace != "core" or any(pin is not None for pin in profile_pins):
                raise ValueError(
                    "canonical relation candidates require namespace=core and null profile pins"
                )
        elif self.namespace == "core" or any(pin is None for pin in profile_pins):
            raise ValueError(
                "profile-defined relation candidates require non-core namespace and all profile pins"
            )
        proposal_pins = (
            self.profile_relation_type_proposal_ref,
            self.profile_relation_type_proposal_hash,
        )
        if any(pin is None for pin in proposal_pins) and any(
            pin is not None for pin in proposal_pins
        ):
            raise ValueError("profile relation proposal pins must be both null or non-null")
        if self.profile_relation_type_proposal_ref is not None:
            if (
                re.fullmatch(
                    r"profile-relation-type-proposal:[0-9a-f]{24}",
                    self.profile_relation_type_proposal_ref,
                )
                is None
            ):
                raise ValueError("profile relation proposal ref is invalid")
            if self.definition_scope != "profile_defined":
                raise ValueError(
                    "profile relation proposal lineage requires profile-defined scope"
                )
        return self

    def validate_source_binding(self, source_binding: dict[str, Any]) -> None:
        expected_scope = {
            "organization_id": source_binding.get("organization_id"),
            "workspace_id": source_binding.get("workspace_id"),
            "source_identity_ref": source_binding.get("source_identity_ref"),
        }
        expected_provenance = {
            "source_hash": source_binding.get("source_hash"),
            "canonical_source_ir_ref": source_binding.get("canonical_source_ir_ref"),
            "canonical_source_ir_hash": source_binding.get("canonical_source_ir_hash"),
        }
        for evidence in self.evidence_refs:
            for field_name, expected in expected_scope.items():
                if getattr(evidence, field_name) != expected:
                    raise ValueError(f"relation EvidenceRef {field_name} mismatch")
            if evidence.selector_schema_version != evidence.selector.schema_version:
                raise ValueError("relation EvidenceRef selector schema mismatch")
            if evidence.selector_hash != evidence.selector.selector_hash():
                raise ValueError("relation EvidenceRef selector hash mismatch")
            expected_identity = evidence_ref_identity(
                organization_id=evidence.organization_id,
                workspace_id=evidence.workspace_id,
                source_identity_ref=evidence.source_identity_ref,
                selector_hash=evidence.selector_hash,
                evidence_role=evidence.evidence_role,
            )
            if evidence.evidence_ref != expected_identity:
                raise ValueError("relation EvidenceRef identity mismatch")
            for field_name, expected in expected_provenance.items():
                if evidence.provenance.get(field_name) != expected:
                    raise ValueError(
                        f"relation EvidenceRef {field_name} provenance mismatch"
                    )

    def validate_outer_topology(self, relation: dict[str, Any]) -> None:
        outer_source_types = _canonical_relation_refs(
            relation.get("source_types"), "proposed_relation_types.source_types"
        )
        outer_target_types = _canonical_relation_refs(
            relation.get("target_types"), "proposed_relation_types.target_types"
        )
        if self.source_type_refs and self.source_type_refs != outer_source_types:
            raise ValueError(
                "relation_type_candidate source constraints mismatch outer topology"
            )
        if self.target_type_refs and self.target_type_refs != outer_target_types:
            raise ValueError(
                "relation_type_candidate target constraints mismatch outer topology"
            )
        for candidate_field, outer_field in (
            ("source_role_refs", "source_role_refs"),
            ("target_role_refs", "target_role_refs"),
        ):
            if getattr(self, candidate_field) and outer_field not in relation:
                raise ValueError(
                    f"relation_type_candidate {candidate_field} requires "
                    "endpoint role observations in outer topology"
                )
            if outer_field in relation and getattr(self, candidate_field) != (
                _canonical_relation_refs(
                    relation[outer_field],
                    f"proposed_relation_types.{outer_field}",
                )
            ):
                raise ValueError(
                    f"relation_type_candidate {candidate_field} mismatch outer topology"
                )
        direction = relation.get("direction")
        if self.directed and direction not in {
            "source_to_target",
            "target_to_source",
        }:
            raise ValueError(
                "directed relation_type_candidate requires directed outer topology"
            )
        if not self.directed and direction != "undirected":
            raise ValueError(
                "undirected relation_type_candidate requires undirected outer topology"
            )

    def as_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True, exclude_none=False)
        if self.profile_relation_type_proposal_ref is None:
            payload.pop("profile_relation_type_proposal_ref")
            payload.pop("profile_relation_type_proposal_hash")
        return payload


def _canonical_v4_relation_types(
    value: Any, source_binding: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    _validate_dict_collection(value, "proposed_relation_types")
    canonical: list[dict[str, Any]] = []
    identities: list[str] = []
    for item in value:
        relation = dict(item)
        candidate = RelationTypeCandidate.model_validate(
            relation.get("relation_type_candidate")
        )
        candidate.validate_source_binding(source_binding)
        if relation.get("type_id") != candidate.relation_type_id:
            raise ValueError(
                "proposed relation type_id must match relation_type_candidate"
            )
        candidate.validate_outer_topology(relation)
        relation["relation_type_candidate"] = candidate.as_dict()
        canonical.append(relation)
        identities.append(candidate.relation_type_id)
    if len(identities) != len(set(identities)):
        raise ValueError("proposed_relation_types contains duplicate relation_type_id")
    return tuple(
        sorted(
            canonical,
            key=lambda item: item["relation_type_candidate"]["relation_type_id"].encode(
                "utf-8"
            ),
        )
    )


@dataclass(frozen=True)
class CorpusGraphSchemaCandidate:
    """Reviewable schema proposal induced from source structure and evidence."""

    candidate_id: str
    candidate_hash: str
    source_structure_profile: dict[str, Any]
    intended_use: str
    semantic_role_proposals: tuple[dict[str, Any], ...]
    proposed_node_types: tuple[dict[str, Any], ...]
    proposed_relation_types: tuple[dict[str, Any], ...]
    proposed_frame_types: tuple[dict[str, Any], ...]
    representation_decisions: tuple[dict[str, Any], ...]
    extraction_contract_hints: dict[str, Any]
    unresolved_ambiguities: tuple[dict[str, Any], ...]
    source_evidence_refs: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    proposed_event_types: tuple[dict[str, Any], ...] = ()
    competency_questions: tuple[dict[str, str], ...] = ()
    source_binding: dict[str, Any] | None = None
    semantica_schema_evidence: dict[str, Any] | None = None
    synthesis_decisions: tuple[dict[str, Any], ...] = ()
    document_understanding: dict[str, Any] | None = None
    graph_layer_coverage: dict[str, Any] | None = None
    extraction_plan: tuple[dict[str, Any], ...] = ()
    assumptions: tuple[dict[str, Any], ...] = ()
    clarification_refs: tuple[str, ...] = ()
    predecessor_candidate_ref: str | None = None
    successor_candidate_refs: tuple[str, ...] = ()
    graph_intent_binding: dict[str, Any] | None = None
    compiler_provenance: dict[str, Any] | None = None
    competency_evaluation: tuple[dict[str, Any], ...] = ()
    hard_negative_evaluation: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version not in {
            SCHEMA_VERSION,
            V2_SCHEMA_VERSION,
            V3_SCHEMA_VERSION,
            V4_SCHEMA_VERSION,
        }:
            raise ValueError(
                "schema_version must be a supported corpus graph schema candidate version"
            )
        _ascii(self.candidate_id, "candidate_id")
        if self.schema_version in {V3_SCHEMA_VERSION, V4_SCHEMA_VERSION}:
            _sha256_ref(self.candidate_hash, "candidate_hash")
        elif not self.candidate_hash.startswith("sha256:"):
            raise ValueError("candidate_hash must be a sha256 ref")
        if self.schema_version not in {
            V3_SCHEMA_VERSION,
            V4_SCHEMA_VERSION,
        } and not self.intended_use.strip():
            raise ValueError("intended_use is required")
        if not self.source_evidence_refs:
            raise ValueError("source_evidence_refs must not be empty")
        for collection_name, items in (
            ("semantic_role_proposals", self.semantic_role_proposals),
            ("proposed_node_types", self.proposed_node_types),
            ("representation_decisions", self.representation_decisions),
        ):
            if not items:
                raise ValueError(f"{collection_name} must not be empty")
        for item in self.proposed_node_types:
            _ascii(item.get("type_id", ""), "proposed_node_types.type_id")
        for item in self.proposed_relation_types:
            _ascii(item.get("type_id", ""), "proposed_relation_types.type_id")
        for item in self.proposed_frame_types:
            _ascii(item.get("type_id", ""), "proposed_frame_types.type_id")
        _validate_dict_collection(self.proposed_event_types, "proposed_event_types")
        _validate_dict_collection(self.competency_questions, "competency_questions")
        for item in self.proposed_event_types:
            _ascii(item.get("type_id", ""), "proposed_event_types.type_id")
        for item in self.competency_questions:
            _ascii(item.get("question_id", ""), "competency_questions.question_id")
            if not str(item.get("text") or "").strip():
                raise ValueError("competency_questions.text is required")
        if (
            self.schema_version not in {V3_SCHEMA_VERSION, V4_SCHEMA_VERSION}
            and any(
                decision.get("binary_triple_sufficient") is False
                for decision in self.representation_decisions
            )
            and not self.proposed_frame_types
        ):
            raise ValueError("a non-binary representation decision requires a proposed frame type")
        if self.schema_version in {
            V2_SCHEMA_VERSION,
            V3_SCHEMA_VERSION,
            V4_SCHEMA_VERSION,
        }:
            for field_name, value in (
                ("source_binding", self.source_binding),
                ("semantica_schema_evidence", self.semantica_schema_evidence),
                ("document_understanding", self.document_understanding),
                ("graph_layer_coverage", self.graph_layer_coverage),
            ):
                if not isinstance(value, dict) or not value:
                    raise ValueError(
                        f"{field_name} must be a nonempty object for governed candidates"
                    )
            if not self.synthesis_decisions:
                raise ValueError(
                    "synthesis_decisions must not be empty for governed candidates"
                )
            _validate_dict_collection(self.extraction_plan, "extraction_plan")
            _validate_dict_collection(self.assumptions, "assumptions")
        if self.schema_version == V4_SCHEMA_VERSION:
            canonical_relation_types = _canonical_v4_relation_types(
                self.proposed_relation_types, self.source_binding or {}
            )
            if self.proposed_relation_types != canonical_relation_types:
                raise ValueError("candidate v4 relation types are not canonical")
        if self.schema_version in {V3_SCHEMA_VERSION, V4_SCHEMA_VERSION}:
            self._validate_v3()
        if self.schema_version in {
            V2_SCHEMA_VERSION,
            V3_SCHEMA_VERSION,
            V4_SCHEMA_VERSION,
        }:
            expected_hash = f"sha256:{content_hash(self._hash_payload())}"
            if self.candidate_hash != expected_hash:
                raise ValueError("candidate_hash does not match candidate content")
            if self.candidate_id != f"cgsc_{expected_hash[7:31]}":
                raise ValueError("candidate_id does not match candidate content")

    def _validate_v3(self) -> None:
        binding = self.graph_intent_binding
        if not isinstance(binding, dict) or set(binding) != {
            "graph_intent_ref",
            "graph_intent_hash",
            "review_state",
            "source_identity_ref",
        }:
            raise ValueError("graph_intent_binding is invalid for v3")
        _ascii(binding["graph_intent_ref"], "graph_intent_binding.graph_intent_ref")
        _ascii(
            binding["source_identity_ref"],
            "graph_intent_binding.source_identity_ref",
        )
        _sha256_ref(
            binding["graph_intent_hash"],
            "graph_intent_binding.graph_intent_hash",
        )
        if binding["review_state"] not in {"accepted", "edited"}:
            raise ValueError("graph_intent_binding.review_state is invalid")

        source = self.source_binding
        required_source = {
            "organization_id",
            "workspace_id",
            "lineage_id",
            "source_identity_ref",
            "source_hash",
            "source_version",
            "canonical_source_ir_ref",
            "canonical_source_ir_hash",
        }
        if not isinstance(source, dict) or set(source) != required_source:
            raise ValueError("source_binding is invalid for v3")
        if source["source_identity_ref"] != binding["source_identity_ref"]:
            raise ValueError("GraphIntent and candidate source identity mismatch")
        for field in ("source_hash", "canonical_source_ir_hash"):
            _sha256_ref(source[field], f"source_binding.{field}")

        provenance = self.compiler_provenance
        required_provenance = {
            "orchestration_skill_ref",
            "compiler_capability_ref",
            "candidate_schema_version",
            "profile_id",
            "profile_hash",
            "catalog_hash",
            "semantica_commit",
            "semantica_input_hash",
        }
        if self.schema_version == V4_SCHEMA_VERSION:
            required_provenance.update(
                {
                    "taxonomy_ref",
                    "taxonomy_hash",
                    "profile_relation_catalog_hash",
                }
            )
        if not isinstance(provenance, dict) or set(provenance) != required_provenance:
            raise ValueError("compiler_provenance is invalid for v3")
        if provenance["orchestration_skill_ref"] != "knowledge-graph-building@1.0.0":
            raise ValueError("v3 orchestration skill provenance is invalid")
        if provenance["compiler_capability_ref"] != "corpus-graph-schema-induction":
            raise ValueError("v3 compiler capability provenance is invalid")
        if provenance["candidate_schema_version"] != self.schema_version:
            raise ValueError("v3 compiler candidate schema provenance is invalid")
        for field in ("profile_id", "semantica_commit"):
            _ascii(provenance[field], f"compiler_provenance.{field}")
        for field in ("profile_hash", "catalog_hash", "semantica_input_hash"):
            _sha256_ref(provenance[field], f"compiler_provenance.{field}")
        if self.schema_version == V4_SCHEMA_VERSION:
            if _TAXONOMY_REF.fullmatch(str(provenance["taxonomy_ref"])) is None:
                raise ValueError("v4 compiler taxonomy_ref is invalid")
            _sha256_ref(
                provenance["taxonomy_hash"],
                "compiler_provenance.taxonomy_hash",
            )
            if provenance["profile_relation_catalog_hash"] is not None:
                _sha256_ref(
                    provenance["profile_relation_catalog_hash"],
                    "compiler_provenance.profile_relation_catalog_hash",
                )
            for relation in self.proposed_relation_types:
                candidate = relation["relation_type_candidate"]
                if (
                    candidate["taxonomy_ref"] != provenance["taxonomy_ref"]
                    or candidate["taxonomy_hash"] != provenance["taxonomy_hash"]
                ):
                    raise ValueError("v4 relation taxonomy pins mismatch compiler")
        if provenance["profile_id"] != self.source_structure_profile.get("profile_id"):
            raise ValueError("v3 compiler profile does not match structural profile")
        if source["source_hash"] != self.source_structure_profile.get("document_hash"):
            raise ValueError("v3 source document hash does not match structural profile")
        semantica = self.semantica_schema_evidence or {}
        if (
            semantica.get("semantica_commit") != provenance["semantica_commit"]
            or semantica.get("input_hash") != provenance["semantica_input_hash"]
        ):
            raise ValueError("v3 Semantica provenance mismatch")

        understanding = self.document_understanding or {}
        if understanding.get("schema_version") != "graph_intent_semantic_projection.v1":
            raise ValueError("v3 document_understanding must be a GraphIntent projection")
        projected = understanding.get("graph_intent")
        if not isinstance(projected, dict):
            raise ValueError("v3 document_understanding graph_intent is required")
        if not isinstance(understanding.get("structural_evidence_summary"), dict):
            raise ValueError("v3 structural evidence summary is required")
        structural_summary = understanding["structural_evidence_summary"]
        if structural_summary.get("profile_id") != self.source_structure_profile.get(
            "profile_id"
        ) or (
            structural_summary.get("document_hash") is not None
            and structural_summary.get("document_hash")
            != self.source_structure_profile.get("document_hash")
        ):
            raise ValueError("v3 structural evidence summary mismatch")
        primary = projected.get("primary_relationship")
        representation = projected.get("preferred_representation")
        dimensions = projected.get("material_dimensions")
        if not isinstance(primary, dict) or not isinstance(representation, dict):
            raise ValueError("v3 primary relationship and representation are required")
        if not isinstance(dimensions, list):
            raise ValueError("v3 material dimensions are invalid")
        allowed_directions = {"source_to_target", "target_to_source"}
        if self.schema_version == V4_SCHEMA_VERSION:
            allowed_directions.add("undirected")
        if primary.get("direction") not in allowed_directions:
            raise ValueError("v3 primary relationship direction is invalid")
        relation = next(
            (
                item
                for item in self.proposed_relation_types
                if item.get("type_id") == primary.get("type_id")
            ),
            None,
        )
        if relation is None or any(
            relation.get(field) != primary.get(field)
            for field in (
                "source_types",
                "target_types",
                "direction",
                "semantic_description",
            )
        ):
            raise ValueError("v3 primary relationship topology mismatch")
        if self.schema_version == V4_SCHEMA_VERSION and any(
            tuple(relation.get(field) or ()) != tuple(primary.get(field) or ())
            for field in ("source_role_refs", "target_role_refs")
        ):
            raise ValueError(
                "v4 primary relationship endpoint role topology mismatch"
            )
        representation_kind = representation.get("kind")
        representation_type = representation.get("type_id")
        if representation_kind in {"frame", "associative_class", "event"} and not str(
            representation_type or ""
        ):
            raise ValueError("v3 preferred representation type_id is required")
        if not any(
            item.get("selected_representation") == representation_kind
            and item.get("type_id") == representation_type
            for item in self.representation_decisions
        ):
            raise ValueError("v3 preferred representation is not compiled")
        node_type_ids = {
            item.get("type_id") for item in self.proposed_node_types
        }
        topology_type_ids = {
            *primary.get("source_types", []),
            *primary.get("target_types", []),
            *(item.get("type_id") for item in dimensions),
        }
        if None in topology_type_ids or not topology_type_ids <= node_type_ids:
            raise ValueError("v3 topology type is missing from proposed node types")
        representation_items: tuple[dict[str, Any], ...] = ()
        slot_field = "slots"
        if representation_kind in {"frame", "associative_class"}:
            representation_items = self.proposed_frame_types
        elif representation_kind == "event":
            representation_items = self.proposed_event_types
            slot_field = "required_roles"
        if representation_kind in {"frame", "associative_class", "event"}:
            if not representation_items:
                raise ValueError("v3 preferred representation type is not proposed")
            compiled_representation = next(
                (
                    item
                    for item in representation_items
                    if item.get("type_id") == representation_type
                ),
                None,
            )
            if compiled_representation is None:
                raise ValueError("v3 preferred representation type is not proposed")
            material_dimension_ids = {
                item.get("type_id") for item in dimensions
            }
            if not material_dimension_ids <= set(
                compiled_representation.get(slot_field) or []
            ):
                raise ValueError(
                    "v3 material dimensions are missing from representation slots"
                )
        coverage = {
            (item.get("coverage_kind"), item.get("target_ref"))
            for item in self.extraction_plan
        }
        if ("primary_relationship", primary.get("type_id")) not in coverage:
            raise ValueError("v3 extraction plan lacks primary relationship coverage")
        if any(
            ("material_dimension", item.get("type_id")) not in coverage
            for item in dimensions
        ):
            raise ValueError("v3 extraction plan lacks material dimension coverage")
        relationship_plan = next(
            (
                item
                for item in self.extraction_plan
                if item.get("coverage_kind") == "primary_relationship"
                and item.get("target_ref") == primary.get("type_id")
            ),
            None,
        )
        if relationship_plan is None or any(
            relationship_plan.get(field) != primary.get(field)
            for field in ("source_types", "target_types", "direction")
        ) or relationship_plan.get("strategy") != "intent_relation":
            raise ValueError("v3 primary relationship extraction topology mismatch")

        structural_role_by_type = {
            str(item.get("type_id")): str(item.get("derived_from_role"))
            for item in self.proposed_node_types
            if item.get("derived_from_role")
        }
        for dimension in dimensions:
            type_id = str(dimension.get("type_id"))
            plan = next(
                (
                    item
                    for item in self.extraction_plan
                    if item.get("coverage_kind") == "material_dimension"
                    and item.get("target_ref") == type_id
                ),
                None,
            )
            if plan is None or plan.get("strategy") != "intent_slot":
                raise ValueError("v3 material dimension grounding plan is missing")
            if type_id in structural_role_by_type:
                valid_grounding = (
                    plan.get("grounding_kind") == "structural_role"
                    and plan.get("role_id") == structural_role_by_type[type_id]
                )
            else:
                valid_grounding = (
                    plan.get("grounding_kind") == "accepted_claim_evidence"
                    and bool(plan.get("accepted_evidence_refs"))
                    and bool(plan.get("source_block_refs"))
                )
            if not valid_grounding:
                raise ValueError("v3 material dimension grounding is invalid")

        dimension_type_ids = {
            str(item.get("type_id")) for item in dimensions
        }
        endpoint_types = {
            *map(str, primary.get("source_types") or []),
            *map(str, primary.get("target_types") or []),
        } - {
            *structural_role_by_type,
            *dimension_type_ids,
            str(representation_type),
        }
        for type_id in endpoint_types:
            plan = next(
                (
                    item
                    for item in self.extraction_plan
                    if item.get("strategy") == "intent_evidence_node"
                    and item.get("target_ref") == type_id
                ),
                None,
            )
            if (
                plan is None
                or plan.get("grounding_kind") != "accepted_claim_evidence"
                or not str(plan.get("claim_path") or "")
                or not plan.get("accepted_evidence_refs")
                or not plan.get("source_block_refs")
            ):
                raise ValueError("v3 endpoint grounding is invalid")

        if representation_kind in {"frame", "associative_class", "event"}:
            constructor_strategy = (
                "intent_event_constructor"
                if representation_kind == "event"
                else "intent_frame_constructor"
            )
            constructor = next(
                (
                    item
                    for item in self.extraction_plan
                    if item.get("strategy") == constructor_strategy
                    and item.get("target_ref") == representation_type
                ),
                None,
            )
            constructor_types = (
                constructor.get("required_participant_types")
                if constructor is not None and representation_kind == "event"
                else (
                    constructor.get("required_slot_types")
                    if constructor is not None
                    else None
                )
            )
            expected_constructor_types = list(
                dict.fromkeys(item.get("type_id") for item in dimensions)
            )
            if constructor is None or constructor_types != expected_constructor_types:
                raise ValueError("v3 representation constructor is invalid")
            if representation_kind == "event" and (
                not isinstance(constructor.get("event_extraction_rule"), dict)
                or not constructor["event_extraction_rule"]
                or not isinstance(constructor.get("role_type_map"), dict)
                or not dimension_type_ids
                <= set(map(str, constructor["role_type_map"].values()))
            ):
                raise ValueError("v3 event representation constructor is invalid")

        _validate_dict_collection(
            self.competency_evaluation, "competency_evaluation"
        )
        _validate_dict_collection(
            self.hard_negative_evaluation, "hard_negative_evaluation"
        )
        competency_ids = {
            item.get("question_id") for item in self.competency_questions
        }
        if competency_ids != {
            item.get("question_id") for item in self.competency_evaluation
        }:
            raise ValueError("v3 competency evaluation coverage mismatch")
        if projected.get("competency_questions") != [
            dict(item) for item in self.competency_questions
        ]:
            raise ValueError("v3 competency questions do not match GraphIntent")
        plan_by_ref = {
            str(item.get("container_ref") or ""): item
            for item in self.extraction_plan
        }
        executable_coverage_refs = {
            str(item.get("target_ref"))
            for item in self.extraction_plan
            if item.get("coverage_kind")
            in {"primary_relationship", "material_dimension"}
        }
        expected_requirements = [
            {
                "output_kind": "relation",
                "type_id": str(primary.get("type_id")),
            }
        ]
        if representation_kind in {"frame", "associative_class", "event"}:
            expected_requirements.append(
                {
                    "output_kind": (
                        "event" if representation_kind == "event" else "frame"
                    ),
                    "type_id": str(representation_type),
                }
            )
        expected_requirements.extend(
            {
                "output_kind": "node",
                "type_id": str(item.get("type_id")),
            }
            for item in dimensions
        )
        for evaluation in self.competency_evaluation:
            plan = plan_by_ref.get(str(evaluation.get("coverage_plan_ref") or ""))
            refs = [str(item) for item in evaluation.get("coverage_refs") or []]
            if (
                evaluation.get("status") != "pending_extraction"
                or plan is None
                or plan.get("coverage_kind") != "competency_question"
                or plan.get("target_ref") != evaluation.get("question_id")
                or plan.get("strategy") != "intent_competency_check"
                or plan.get("required_coverage_refs") != refs
                or plan.get("requirements") != expected_requirements
                or not refs
                or not set(refs) <= executable_coverage_refs
            ):
                raise ValueError(
                    "v3 competency evaluation lacks concrete extraction plan evidence"
                )
        hard_negative_ids = {
            item.get("question_id")
            for item in projected.get("hard_negative_questions", [])
        }
        if hard_negative_ids != {
            item.get("question_id") for item in self.hard_negative_evaluation
        }:
            raise ValueError("v3 hard-negative evaluation coverage mismatch")
        hard_negative_by_id = {
            item.get("question_id"): item
            for item in projected.get("hard_negative_questions", [])
        }
        for evaluation in self.hard_negative_evaluation:
            plan = plan_by_ref.get(str(evaluation.get("coverage_plan_ref") or ""))
            question = hard_negative_by_id.get(evaluation.get("question_id")) or {}
            if (
                evaluation.get("status") != "unresolved"
                or plan is None
                or plan.get("coverage_kind") != "hard_negative_guard"
                or plan.get("target_ref") != evaluation.get("question_id")
                or plan.get("strategy") != "intent_distinction_guard"
                or plan.get("verification_mode") != "assisted_only"
                or plan.get("constraint") != question.get("distinction_to_preserve")
                or evaluation.get("distinction_to_preserve")
                != question.get("distinction_to_preserve")
                or evaluation.get("coverage_refs") != []
            ):
                raise ValueError(
                    "v3 hard-negative evaluation lacks assisted-only guard evidence"
                )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "source_structure_profile": dict(self.source_structure_profile),
            "semantic_role_proposals": [dict(item) for item in self.semantic_role_proposals],
            "proposed_node_types": [dict(item) for item in self.proposed_node_types],
            "proposed_relation_types": [dict(item) for item in self.proposed_relation_types],
            "proposed_frame_types": [dict(item) for item in self.proposed_frame_types],
            "proposed_event_types": [dict(item) for item in self.proposed_event_types],
            "competency_questions": [dict(item) for item in self.competency_questions],
            "representation_decisions": [dict(item) for item in self.representation_decisions],
            "extraction_contract_hints": dict(self.extraction_contract_hints),
            "unresolved_ambiguities": [dict(item) for item in self.unresolved_ambiguities],
            "source_evidence_refs": list(self.source_evidence_refs),
        }
        if self.schema_version not in {V3_SCHEMA_VERSION, V4_SCHEMA_VERSION}:
            payload["intended_use"] = self.intended_use
        if self.schema_version in {
            V2_SCHEMA_VERSION,
            V3_SCHEMA_VERSION,
            V4_SCHEMA_VERSION,
        }:
            payload.update(
                {
                    "source_binding": dict(self.source_binding or {}),
                    "semantica_schema_evidence": dict(
                        self.semantica_schema_evidence or {}
                    ),
                    "synthesis_decisions": [
                        dict(item) for item in self.synthesis_decisions
                    ],
                    "document_understanding": dict(
                        self.document_understanding or {}
                    ),
                    "graph_layer_coverage": dict(self.graph_layer_coverage or {}),
                    "extraction_plan": [dict(item) for item in self.extraction_plan],
                    "assumptions": [dict(item) for item in self.assumptions],
                    "clarification_refs": list(self.clarification_refs),
                    "predecessor_candidate_ref": self.predecessor_candidate_ref,
                    "successor_candidate_refs": list(self.successor_candidate_refs),
                }
            )
        if self.schema_version in {V3_SCHEMA_VERSION, V4_SCHEMA_VERSION}:
            payload.update(
                {
                    "graph_intent_binding": dict(self.graph_intent_binding or {}),
                    "compiler_provenance": dict(self.compiler_provenance or {}),
                    "competency_evaluation": [
                        dict(item) for item in self.competency_evaluation
                    ],
                    "hard_negative_evaluation": [
                        dict(item) for item in self.hard_negative_evaluation
                    ],
                }
            )
        return payload

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("candidate_id", None)
        payload.pop("candidate_hash", None)
        return payload

    @classmethod
    def create_v2(cls, content: dict[str, Any]) -> "CorpusGraphSchemaCandidate":
        payload = dict(content)
        payload["schema_version"] = V2_SCHEMA_VERSION
        payload.pop("candidate_id", None)
        payload.pop("candidate_hash", None)
        digest = content_hash(payload)
        return cls.from_dict(
            {
                **payload,
                "candidate_id": f"cgsc_{digest[:24]}",
                "candidate_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def create_v3(cls, content: dict[str, Any]) -> "CorpusGraphSchemaCandidate":
        payload = dict(content)
        payload["schema_version"] = V3_SCHEMA_VERSION
        payload.pop("candidate_id", None)
        payload.pop("candidate_hash", None)
        payload.pop("intended_use", None)
        digest = content_hash(payload)
        return cls.from_dict(
            {
                **payload,
                "candidate_id": f"cgsc_{digest[:24]}",
                "candidate_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def create_v4(cls, content: dict[str, Any]) -> "CorpusGraphSchemaCandidate":
        payload = dict(content)
        payload["schema_version"] = V4_SCHEMA_VERSION
        payload.pop("candidate_id", None)
        payload.pop("candidate_hash", None)
        payload.pop("intended_use", None)
        source_binding = payload.get("source_binding")
        if not isinstance(source_binding, dict):
            raise ValueError("source_binding is required for candidate v4")
        payload["proposed_relation_types"] = [
            dict(item)
            for item in _canonical_v4_relation_types(
                payload.get("proposed_relation_types"), source_binding
            )
        ]
        digest = content_hash(payload)
        return cls.from_dict(
            {
                **payload,
                "candidate_id": f"cgsc_{digest[:24]}",
                "candidate_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusGraphSchemaCandidate":
        if payload.get("schema_version") == V3_SCHEMA_VERSION and set(payload) != (
            _V3_FIELDS
        ):
            raise ValueError("candidate payload must contain exact v3 fields")
        if payload.get("schema_version") == V4_SCHEMA_VERSION and set(payload) != (
            _V4_FIELDS
        ):
            raise ValueError("candidate payload must contain exact v4 fields")
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            candidate_id=str(payload.get("candidate_id") or ""),
            candidate_hash=str(payload.get("candidate_hash") or ""),
            source_structure_profile=dict(payload.get("source_structure_profile") or {}),
            intended_use=str(payload.get("intended_use") or ""),
            semantic_role_proposals=_tuple_of_dicts(
                payload.get("semantic_role_proposals"), "semantic_role_proposals"
            ),
            proposed_node_types=_tuple_of_dicts(
                payload.get("proposed_node_types"), "proposed_node_types"
            ),
            proposed_relation_types=_optional_tuple_of_dicts(
                payload.get("proposed_relation_types"), "proposed_relation_types"
            ),
            proposed_frame_types=tuple(
                dict(item) for item in (payload.get("proposed_frame_types") or [])
            ),
            proposed_event_types=_optional_tuple_of_dicts(
                payload.get("proposed_event_types"), "proposed_event_types"
            ),
            competency_questions=_optional_tuple_of_dicts(
                payload.get("competency_questions"), "competency_questions"
            ),
            representation_decisions=_tuple_of_dicts(
                payload.get("representation_decisions"), "representation_decisions"
            ),
            extraction_contract_hints=dict(payload.get("extraction_contract_hints") or {}),
            unresolved_ambiguities=tuple(
                dict(item) for item in (payload.get("unresolved_ambiguities") or [])
            ),
            source_evidence_refs=tuple(str(ref) for ref in (payload.get("source_evidence_refs") or [])),
            source_binding=(
                dict(payload["source_binding"])
                if isinstance(payload.get("source_binding"), dict)
                else None
            ),
            semantica_schema_evidence=(
                dict(payload["semantica_schema_evidence"])
                if isinstance(payload.get("semantica_schema_evidence"), dict)
                else None
            ),
            synthesis_decisions=_optional_tuple_of_dicts(
                payload.get("synthesis_decisions"), "synthesis_decisions"
            ),
            document_understanding=(
                dict(payload["document_understanding"])
                if isinstance(payload.get("document_understanding"), dict)
                else None
            ),
            graph_layer_coverage=(
                dict(payload["graph_layer_coverage"])
                if isinstance(payload.get("graph_layer_coverage"), dict)
                else None
            ),
            extraction_plan=_optional_tuple_of_dicts(
                payload.get("extraction_plan"), "extraction_plan"
            ),
            assumptions=_optional_tuple_of_dicts(
                payload.get("assumptions"), "assumptions"
            ),
            clarification_refs=tuple(
                str(ref) for ref in (payload.get("clarification_refs") or [])
            ),
            predecessor_candidate_ref=(
                str(payload["predecessor_candidate_ref"])
                if payload.get("predecessor_candidate_ref") is not None
                else None
            ),
            successor_candidate_refs=tuple(
                str(ref) for ref in (payload.get("successor_candidate_refs") or [])
            ),
            graph_intent_binding=(
                dict(payload["graph_intent_binding"])
                if isinstance(payload.get("graph_intent_binding"), dict)
                else None
            ),
            compiler_provenance=(
                dict(payload["compiler_provenance"])
                if isinstance(payload.get("compiler_provenance"), dict)
                else None
            ),
            competency_evaluation=_optional_tuple_of_dicts(
                payload.get("competency_evaluation"), "competency_evaluation"
            ),
            hard_negative_evaluation=_optional_tuple_of_dicts(
                payload.get("hard_negative_evaluation"),
                "hard_negative_evaluation",
            ),
        )
