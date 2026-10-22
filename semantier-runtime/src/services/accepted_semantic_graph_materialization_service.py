"""Sealed replay seam for accepted semantic-graph materialization inputs."""

from __future__ import annotations

from storage import hosted_postgres as hosted

from dataclasses import dataclass
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from contracts.canonical_source_ir import (
    CanonicalSourceDocument,
    content_hash as canonical_ir_content_hash,
)
from contracts.accepted_graph_release_lineage import AcceptedGraphReleaseLineageV2
from contracts.accepted_graph_release_v2 import (
    AcceptedGraphReleaseV2,
    CanonicalPatternManifestV1,
    OccurrenceGroundingManifestV1,
)
from contracts.canonical_restriction_pattern import (
    CanonicalRestrictionPatternV1,
    canonical_pattern_bytes_unverified,
    create_canonical_restriction_pattern_unverified,
)
from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate, V4_SCHEMA_VERSION
from contracts.evidence_location import SourceIdentity
from contracts.extracted_relation_occurrence import ExtractedRelationOccurrenceV1
from contracts.extracted_relation_occurrence import create_extracted_relation_occurrence
from contracts.governed_extract import ExtractResult
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.knowledge_relation_instance import (
    AcceptedRelationBindingV1,
    KnowledgeRelationInstanceV1,
    create_knowledge_relation_instance,
)
from contracts.schema_decision import SchemaDecision, V3_SCHEMA_VERSION
from contracts.semantic_constraint_frame import ComposedFramePatternV01, FramePatternV01
from contracts.source_semantic_witness import (
    SourceSemanticWitnessSetManifestV1,
    SourceSemanticWitnessV1,
)
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.falkordb_backend import (
    GraphBackend,
    InMemoryGraphBackend,
)
from semantier.semantica_context_graph.accepted_release_conformance import (
    validate_accepted_release_conformance,
)
from semantier.semantica_context_graph.native_builder import (
    native_graph_from_accepted_release_bundle,
)
from services.corpus_graph_schema_profile_registry import CorpusGraphSchemaProfileRegistry
from services.governed_extract_service import GovernedExtractService
from services.knowledge_relation_taxonomy_service import KnowledgeRelationTaxonomyService
from services.materialized_accepted_graph_bundle_reader import (
    MaterializedAcceptedGraphBundleReader,
)
from services.relation_usage_grounding_service import (
    _create_relation_usage_grounding_authority,
)
from storage.corpus_graph_schema_candidate_store import CorpusGraphSchemaCandidateStore
from storage.accepted_graph_release_origin import (
    record_accepted_release_origin,
    verify_exact_release_origin_reservation,
)
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.human_grounding_store import HumanGroundingStore
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.profile_relation_type_proposal_store import ProfileRelationTypeProposalStore
from storage.schema_decision_store import SchemaDecisionStore
from storage.source_structure_semantic_validation_store import (
    SourceStructureSemanticValidationStore,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CODE_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_STRUCTURE_MATERIALIZATION_AUTHORITY = object()


@dataclass(frozen=True)
class ResolveMaterializationInputsCommand:
    organization_id: str
    workspace_id: str
    source_identity_ref: str
    lineage_id: str
    graph_intent_ref: str
    graph_intent_hash: str
    schema_candidate_ref: str
    schema_candidate_hash: str
    schema_decision_ref: str
    schema_decision_hash: str
    extract_result_ref: str
    extract_result_hash: str
    predecessor_release_ref: str | None = None
    predecessor_release_hash: str | None = None
    source_structure_validation_ref: str | None = None
    source_structure_validation_hash: str | None = None
    _source_structure_authority: object | None = None


@dataclass(frozen=True)
class ResolveReviewedPatternInputsCommand:
    materialization: ResolveMaterializationInputsCommand
    candidate_graph_id: str
    human_label_event_ids: tuple[str, ...]
    canonical_pattern_predecessor_membership: tuple[tuple[str, str], ...] | None = None
    canonical_pattern_unchanged_relation_membership: tuple[
        tuple[str, str, str, str], ...
    ] | None = None


@dataclass(frozen=True)
class ResolvedMaterializationInputs:
    source_identity: SourceIdentity
    canonical_source_ir: CanonicalSourceDocument
    graph_intent: GraphIntentBrief
    schema_candidate: CorpusGraphSchemaCandidate
    schema_decision: SchemaDecision
    extract_result: ExtractResult
    witness_manifests: tuple[SourceSemanticWitnessSetManifestV1, ...]
    witnesses: tuple[SourceSemanticWitnessV1, ...]
    occurrence_groundings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ResolvedReviewedPatternInputs:
    materialization: ResolvedMaterializationInputs
    reviewed_selection: dict[str, Any]


class AcceptedSemanticGraphMaterializationService:
    """Resolve a complete exact-pinned input set without writing output state."""

    __slots__ = (
        "_runtime_root", "_workspace_root", "_db_path", "_organization_id",
        "_workspace_id", "_source_store", "_intent_store", "_candidate_store",
        "_decision_store", "_grounding_store",
        "_profile_registry", "_taxonomy_service",
        "_materialization_backend_configured", "_sealed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("materialization input authority is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self, *, runtime_root: str | Path, workspace_root: str | Path,
        db_path: str | Path, organization_id: str, workspace_id: str,
        graph_backend: GraphBackend | None = None,
        profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
    ) -> None:
        if type(self) is not AcceptedSemanticGraphMaterializationService:
            raise ValueError("materialization requires the exact sealed service")
        for field, value in (("organization_id", organization_id), ("workspace_id", workspace_id)):
            if not isinstance(value, str) or not value.isascii() or _SAFE_ID.fullmatch(value) is None:
                raise ValueError(f"{field} must be a safe ASCII identity")
        runtime = Path(runtime_root).resolve()
        workspace = Path(workspace_root).resolve()
        database = Path(db_path).resolve()
        if not workspace.is_relative_to(runtime):
            raise ValueError("workspace_root must remain inside governed runtime_root")
        if not database.is_relative_to(workspace):
            raise ValueError("db_path must remain inside governed workspace_root")
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_runtime_root", runtime)
        object.__setattr__(self, "_workspace_root", workspace)
        object.__setattr__(self, "_db_path", database)
        object.__setattr__(self, "_organization_id", organization_id)
        object.__setattr__(self, "_workspace_id", workspace_id)
        source_store = KnowledgeBuilderStore(database, workspace_root=workspace)
        profile_registry = profile_registry or CorpusGraphSchemaProfileRegistry(
            repository_root=_CODE_REPOSITORY_ROOT
        )
        proposal_store = ProfileRelationTypeProposalStore(
            workspace, workspace_id=workspace_id, organization_id=organization_id
        )
        taxonomy_service = KnowledgeRelationTaxonomyService(
            repository_root=_CODE_REPOSITORY_ROOT,
            profile_registry=profile_registry,
            profile_relation_type_proposal_store=proposal_store,
        )
        object.__setattr__(self, "_source_store", source_store)
        object.__setattr__(self, "_intent_store", GraphIntentBriefStore(
            workspace, organization_id=organization_id, workspace_id=workspace_id,
            source_registry=source_store,
        ))
        object.__setattr__(self, "_candidate_store", CorpusGraphSchemaCandidateStore(
            workspace, workspace_id=workspace_id
        ))
        object.__setattr__(self, "_decision_store", SchemaDecisionStore(
            workspace, workspace_id=workspace_id
        ))
        grounding_authority = _create_relation_usage_grounding_authority(
            workspace, organization_id=organization_id, workspace_id=workspace_id,
            profile_registry=profile_registry,
            profile_relation_type_proposal_store=proposal_store,
        )
        object.__setattr__(self, "_grounding_store", HumanGroundingStore(
            database,
            graph_backend=(
                graph_backend if graph_backend is not None else InMemoryGraphBackend()
            ),
            _relation_usage_grounding_authority=grounding_authority,
        ))
        object.__setattr__(self, "_profile_registry", profile_registry)
        object.__setattr__(self, "_taxonomy_service", taxonomy_service)
        object.__setattr__(
            self, "_materialization_backend_configured", graph_backend is not None
        )
        object.__setattr__(self, "_sealed", True)

    def resolve_materialization_inputs(
        self, command: ResolveMaterializationInputsCommand
    ) -> ResolvedMaterializationInputs:
        self._require_integrity(command)
        intent = GraphIntentBriefStore.get(
            self._intent_store, command.graph_intent_ref,
            graph_intent_hash=command.graph_intent_hash,
        )
        if intent.review.state not in {"accepted", "edited"}:
            raise ValueError("materialization GraphIntent must be accepted or edited")
        if GraphIntentBriefStore.successors(self._intent_store, intent.graph_intent_id):
            raise ValueError("materialization GraphIntent is stale/superseded")
        if GraphIntentBriefStore.is_superseded_for_new_compilation(
            self._intent_store, intent.graph_intent_id
        ):
            raise ValueError("materialization source identity is stale/superseded")
        source = KnowledgeBuilderStore.get_source_identity(
            self._source_store, source_identity_ref=command.source_identity_ref,
            organization_id=command.organization_id, workspace_id=command.workspace_id,
            lineage_id=command.lineage_id,
        )
        if source is None:
            raise ValueError("governed SourceIdentity is missing")
        self._match_source(intent, source, command)
        canonical = self._read_canonical(intent, source)

        candidate = CorpusGraphSchemaCandidate.from_dict(
            CorpusGraphSchemaCandidateStore.get_by_hash(
                self._candidate_store, command.schema_candidate_hash
            )
        )
        if candidate.schema_version != V4_SCHEMA_VERSION:
            raise ValueError("materialization requires Candidate.v4")
        if candidate.candidate_id != command.schema_candidate_ref:
            raise ValueError("schema candidate exact ref/hash mismatch")
        if CorpusGraphSchemaCandidateStore.successors(self._candidate_store, candidate.candidate_id):
            raise ValueError("schema candidate is stale/superseded")
        if candidate.graph_intent_binding != {
            "graph_intent_ref": intent.graph_intent_id,
            "graph_intent_hash": intent.graph_intent_hash,
            "review_state": intent.review.state,
            "source_identity_ref": source.source_identity_ref,
        }:
            raise ValueError("candidate GraphIntent binding mismatch")
        if candidate.source_binding != self._source_binding(intent):
            raise ValueError("candidate source binding mismatch")
        self._validate_compiler_provenance(candidate, intent)

        decision = SchemaDecisionStore.get_contract(
            self._decision_store, command.schema_decision_ref,
            taxonomy_service=self._taxonomy_service,
        )
        if decision.schema_version != V3_SCHEMA_VERSION:
            raise ValueError("materialization requires SchemaDecision.v3")
        if decision.schema_decision_hash != command.schema_decision_hash:
            raise ValueError("schema decision exact ref/hash mismatch")
        if decision.successor_schema_decision_refs or self._has_decision_successor(
            decision.schema_decision_id
        ):
            raise ValueError("schema decision is stale/superseded")
        if (
            decision.candidate_binding.get("candidate_id") != candidate.candidate_id
            or decision.candidate_binding.get("candidate_hash") != candidate.candidate_hash
            or decision.graph_intent_binding != candidate.graph_intent_binding
            or decision.source_binding != candidate.source_binding
            or decision.approved_schema != {
                "node_types": [dict(item) for item in candidate.proposed_node_types],
                "relation_types": [dict(item) for item in candidate.proposed_relation_types],
                "event_types": [dict(item) for item in candidate.proposed_event_types],
                "frame_types": [dict(item) for item in candidate.proposed_frame_types],
            }
            or list(decision.extraction_plan)
            != [dict(item) for item in candidate.extraction_plan]
            or list(decision.source_evidence_refs) != list(candidate.source_evidence_refs)
            or decision.semantica_binding != candidate.semantica_schema_evidence
        ):
            raise ValueError("decision exact Candidate/GraphIntent lineage mismatch")
        self._verify_semantica_shacl_evidence(candidate)

        extract = self._read_extract(command)
        if extract.status != "succeeded":
            raise ValueError("materialization requires succeeded ExtractResult")
        if (
            extract.schema_binding.get("schema_decision_ref") != decision.schema_decision_id
            or extract.schema_binding.get("schema_decision_hash") != decision.schema_decision_hash
            or extract.schema_binding.get("candidate_ref") != candidate.candidate_id
            or extract.schema_binding.get("candidate_hash") != candidate.candidate_hash
        ):
            raise ValueError("ExtractResult Candidate/Decision binding mismatch")
        expected_admissions = [
            {
                key: binding.get(key)
                for key in (
                    "relation_type_id", "taxonomy_ref", "taxonomy_hash",
                    "profile_ref", "profile_hash", "profile_relation_catalog_hash",
                )
            } | {
                "family_id": next(
                    item["relation_type_candidate"]["family_id"]
                    for item in decision.approved_schema["relation_types"]
                    if item["type_id"] == binding["relation_type_id"]
                ),
            }
            for binding in decision.admitted_relation_type_bindings
        ]
        if extract.schema_binding.get("admitted_relation_types") != expected_admissions:
            raise ValueError("ExtractResult admitted taxonomy binding mismatch")
        for field, expected in (
            ("lineage_id", command.lineage_id),
            ("canonical_source_ir_ref", intent.source_binding.canonical_source_ir_ref),
            ("canonical_source_hash", source.source_hash),
        ):
            if extract.source_binding.get(field) != expected:
                raise ValueError(f"ExtractResult source binding {field} mismatch")

        manifests, witnesses = self._read_witnesses(
            command, intent, candidate, canonical
        )
        groundings = self._read_terminal_groundings(command, extract, decision)
        return ResolvedMaterializationInputs(
            source, canonical, intent, candidate, decision, extract,
            manifests, witnesses, groundings,
        )

    def _verify_semantica_shacl_evidence(
        self, candidate: CorpusGraphSchemaCandidate
    ) -> None:
        evidence = candidate.semantica_schema_evidence
        if not isinstance(evidence, dict):
            raise ValueError("Candidate.v4 Semantica schema evidence is missing")
        report_ref = evidence.get("shacl_report_ref")
        report_hash = evidence.get("shacl_report_hash")
        if not isinstance(report_ref, str) or not isinstance(report_hash, str):
            raise ValueError("Candidate.v4 SHACL ref/hash is missing")
        report_ref_path = Path(report_ref)
        if report_ref_path.is_absolute():
            raise ValueError("Candidate.v4 SHACL artifact ref must be relative")
        workspace_root = (
            self._workspace_root / "workspaces" / self._workspace_id
        ).resolve()
        report_path = (workspace_root / report_ref_path).resolve()
        if not report_path.is_relative_to(workspace_root):
            raise ValueError("Candidate.v4 SHACL artifact escapes governed workspace")
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Candidate.v4 SHACL artifact is unavailable") from exc
        if "sha256:" + content_hash(report_payload) != report_hash:
            raise ValueError("Candidate.v4 SHACL artifact hash mismatch")
        if (
            report_payload.get("schema_version") != "semantica_shacl_report.v1"
            or report_payload.get("semantica_commit")
            != evidence.get("semantica_commit")
            or report_payload.get("input_hash") != evidence.get("input_hash")
            or report_payload.get("blocking_scope")
            != "semantica_shape_conformance"
        ):
            raise ValueError("Candidate.v4 SHACL artifact binding mismatch")
        report = report_payload.get("report")
        if not isinstance(report, dict) or report.get("conforms") is not True:
            raise ValueError("Candidate.v4 SHACL report does not conform")

    def derive_accepted_relation_instances(
        self, command: ResolveMaterializationInputsCommand
    ) -> tuple[KnowledgeRelationInstanceV1, ...]:
        """Derive release-independent assertions from exact grounded occurrences."""

        resolved = self.resolve_materialization_inputs(command)
        return tuple(sorted(
            (instance for instance, _grounding in self._derive_relation_pairs(
                command, resolved
            )),
            key=lambda item: item.relation_instance_id.encode("utf-8"),
        ))

    def _derive_relation_pairs(
        self,
        command: ResolveMaterializationInputsCommand,
        resolved: ResolvedMaterializationInputs,
    ) -> tuple[tuple[KnowledgeRelationInstanceV1, dict[str, Any]], ...]:
        """Keep each Layer-1 relation paired with its exact Grounding record."""

        occurrences = {
            item.occurrence_ref: item
            for item in (
                ExtractedRelationOccurrenceV1.model_validate(payload)
                for payload in resolved.extract_result.output["relation_occurrences"]
            )
        }
        admissions = {
            str(item["relation_type_id"]): dict(item)
            for item in resolved.schema_decision.admitted_relation_type_bindings
        }
        instances: list[tuple[KnowledgeRelationInstanceV1, dict[str, Any]]] = []
        for grounding in resolved.occurrence_groundings:
            occurrence = occurrences[str(grounding["extracted_occurrence_ref"])]
            if grounding["outcome"] == "corrected":
                decision = HumanGroundingStore.get_relation_usage_grounding(
                    self._grounding_store,
                    str(grounding["relation_usage_grounding_decision_ref"]),
                    organization_id=command.organization_id,
                    workspace_id=command.workspace_id,
                )
                corrected = decision.get("corrected_relation_instance")
                binding = decision.get("corrected_relation_binding")
                if not isinstance(corrected, dict) or not isinstance(binding, dict):
                    raise ValueError("corrected occurrence lacks exact W2e decision")
                if (
                    decision.get("grounding_decision_hash")
                    != grounding["relation_usage_grounding_decision_hash"]
                    or decision.get("organization_id") != command.organization_id
                    or decision.get("workspace_id") != command.workspace_id
                    or decision.get("extract_run_binding") != {
                        "extract_run_ref": resolved.extract_result.extract_run_id,
                        "extract_result_hash": resolved.extract_result.extract_result_hash,
                    }
                    or decision.get("schema_decision_binding") != {
                        "schema_decision_ref": resolved.schema_decision.schema_decision_id,
                        "schema_decision_hash": resolved.schema_decision.schema_decision_hash,
                    }
                ):
                    raise ValueError("corrected W2e decision exact lineage mismatch")
                matching_edges = [
                    dict(edge)
                    for edge in resolved.extract_result.output.get("edges", [])
                    if edge.get("relation_type_id") == occurrence.relation_type_id
                    and self._semantic_edge_endpoints(edge)
                    == (occurrence.source_node_id, occurrence.target_node_id)
                ]
                if len(matching_edges) != 1 or decision.get("subject_binding") != {
                    "subject_kind": "candidate_relation_instance",
                    "subject_ref": matching_edges[0].get("edge_id") if matching_edges else None,
                    "subject_hash": (
                        "sha256:" + content_hash(matching_edges[0])
                        if matching_edges else None
                    ),
                }:
                    raise ValueError("corrected W2e decision subject mismatch")
                source_node_id = corrected.get("source_node_id")
                target_node_id = corrected.get("target_node_id")
                if corrected.get("direction") == "target_to_source":
                    source_node_id, target_node_id = target_node_id, source_node_id
                occurrence = create_extracted_relation_occurrence(
                    organization_id=command.organization_id,
                    workspace_id=command.workspace_id,
                    relation_type_id=str(corrected["relation_type_id"]),
                    representation_kind=binding["representation_kind"],
                    projection_shape=str(binding["projection_shape"]),
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    frame_ref=None,
                    projection_source_refs=[],
                    evidence_refs=occurrence.evidence_refs,
                )
                if (
                    occurrence.occurrence_ref != grounding["corrected_occurrence_ref"]
                    or occurrence.occurrence_hash
                    != grounding["corrected_occurrence_hash"]
                ):
                    raise ValueError("corrected occurrence exact ref/hash mismatch")
            pins = admissions.get(occurrence.relation_type_id)
            if pins is None:
                raise ValueError("grounded occurrence relation type is not admitted")
            instance = create_knowledge_relation_instance(
                taxonomy_service=self._taxonomy_service,
                organization_id=command.organization_id,
                workspace_id=command.workspace_id,
                relation_type_id=occurrence.relation_type_id,
                representation_kind=occurrence.representation_kind,
                taxonomy_ref=str(pins["taxonomy_ref"]),
                taxonomy_hash=str(pins["taxonomy_hash"]),
                profile_ref=pins.get("profile_ref"),
                profile_hash=pins.get("profile_hash"),
                profile_relation_catalog_hash=pins.get(
                    "profile_relation_catalog_hash"
                ),
                source_node_id=occurrence.source_node_id,
                target_node_id=occurrence.target_node_id,
                frame_ref=occurrence.frame_ref,
                projection_source_refs=occurrence.projection_source_refs,
                evidence_refs=occurrence.evidence_refs,
            )
            instances.append((instance, grounding))
        return tuple(sorted(
            instances,
            key=lambda item: item[0].relation_instance_id.encode("utf-8"),
        ))

    def resolve_reviewed_pattern_inputs(
        self, command: ResolveReviewedPatternInputsCommand
    ) -> ResolvedReviewedPatternInputs:
        if type(command) is not ResolveReviewedPatternInputsCommand:
            raise ValueError("reviewed pattern resolution requires exact command type")
        if (
            _SAFE_ID.fullmatch(command.candidate_graph_id) is None
            or not command.human_label_event_ids
            or len(command.human_label_event_ids)
            != len(set(command.human_label_event_ids))
            or any(_SAFE_ID.fullmatch(item) is None for item in command.human_label_event_ids)
        ):
            raise ValueError("reviewed pattern selection identities are invalid")
        resolved = self.resolve_materialization_inputs(command.materialization)
        selected = HumanGroundingStore.resolve_reviewed_candidate_selection(
            self._grounding_store,
            candidate_graph_id=command.candidate_graph_id,
            human_event_ids=command.human_label_event_ids,
            organization_id=command.materialization.organization_id,
            workspace_id=command.materialization.workspace_id,
        )
        owner = selected["owner"]
        graph = HumanGroundingStore.get_candidate_graph(
            self._grounding_store, command.candidate_graph_id
        )
        source_rows = self._source_store._conn.execute(
            hosted.sql_query("SELECT source_id FROM knowledge_builder_sources "
            "WHERE organization_id=? AND workspace_id=? AND lineage_id=?", 'SELECT source_id FROM knowledge_builder_sources WHERE organization_id=%s AND workspace_id=%s AND lineage_id=%s'),
            (
                command.materialization.organization_id,
                command.materialization.workspace_id,
                command.materialization.lineage_id,
            ),
        ).fetchall()
        if len(source_rows) != 1:
            raise ValueError("reviewed candidate graph source lineage is ambiguous")
        legacy_source_id = str(source_rows[0]["source_id"])
        normalized_source_rows = self._source_store._conn.execute(
            hosted.sql_query("SELECT source_ref,source_hash FROM knowledge_builder_sources "
            "WHERE source_id=? AND organization_id=? AND workspace_id=?", 'SELECT source_ref,source_hash FROM knowledge_builder_sources WHERE source_id=%s AND organization_id=%s AND workspace_id=%s'),
            (
                owner.get("source_id"),
                command.materialization.organization_id,
                command.materialization.workspace_id,
            ),
        ).fetchall()
        representation_rows = self._source_store._conn.execute(
            hosted.sql_query("SELECT edge.artifact_path FROM source_representation_edges edge "
            "JOIN source_representation_lifecycle_events event "
            "ON event.event_id=(SELECT latest.event_id FROM "
            "source_representation_lifecycle_events latest WHERE "
            "latest.representation_id=edge.representation_id "
            "ORDER BY latest.created_at DESC,latest.rowid DESC LIMIT 1) "
            "WHERE edge.source_id=? AND edge.organization_id=? "
            "AND edge.workspace_id=? AND event.next_state='active'", "SELECT edge.artifact_path FROM source_representation_edges edge JOIN source_representation_lifecycle_events event ON event.event_id=(SELECT latest.event_id FROM source_representation_lifecycle_events latest WHERE latest.representation_id=edge.representation_id ORDER BY latest.created_at DESC,latest.event_sequence DESC LIMIT 1) WHERE edge.source_id=%s AND edge.organization_id=%s AND edge.workspace_id=%s AND event.next_state='active'"),
            (
                legacy_source_id,
                command.materialization.organization_id,
                command.materialization.workspace_id,
            ),
        ).fetchall()
        normalized_source = (
            dict(normalized_source_rows[0]) if len(normalized_source_rows) == 1 else {}
        )
        workflow_row = self._source_store._conn.execute(
            hosted.sql_query("SELECT workflow_kind FROM knowledge_builder_extraction_runs "
            "WHERE extraction_run_id=? AND organization_id=? AND workspace_id=?", 'SELECT workflow_kind FROM knowledge_builder_extraction_runs WHERE extraction_run_id=%s AND organization_id=%s AND workspace_id=%s'),
            (
                owner.get("extraction_run_id"),
                command.materialization.organization_id,
                command.materialization.workspace_id,
            ),
        ).fetchone()
        normalized_ref = str(normalized_source.get("source_ref") or "")
        exact_representation_bridge = [
            row for row in representation_rows
            if str(row["artifact_path"]) == normalized_ref
            or str(row["artifact_path"]).endswith("/" + normalized_ref)
        ]
        if (
            owner.get("source_identity_ref")
            != resolved.source_identity.source_identity_ref
            or owner.get("lineage_id") != owner.get("source_id")
            or len(exact_representation_bridge) != 1
            or graph.extraction_run_id != owner.get("extraction_run_id")
            or graph.source_document_hash
            != str(normalized_source.get("source_hash") or "")
        ):
            raise ValueError(
                "reviewed candidate graph exact source/extract lineage mismatch: "
                f"owner_source_identity={owner.get('source_identity_ref')}, "
                f"resolved_source_identity={resolved.source_identity.source_identity_ref}, "
                f"owner_source={owner.get('source_id')}, legacy_source={legacy_source_id}, "
                f"owner_lineage={owner.get('lineage_id')}, "
                f"graph_run={graph.extraction_run_id}, "
                f"owner_run={owner.get('extraction_run_id')}, "
                f"graph_source={graph.source_document_hash}, "
                f"resolved_source={resolved.source_identity.source_hash}, "
                f"normalized_ref={normalized_ref}, "
                f"normalized_hash={normalized_source.get('source_hash')}, "
                f"representation_paths={[row['artifact_path'] for row in representation_rows]}"
            )
        if workflow_row is not None and workflow_row["workflow_kind"] == "reference_graph_build":
            from plugins.tender_sensitive_term_detection_modeling.frame_semantics import (
                compile_frame_pattern_candidate,
            )

            schema_projection = {
                "review_state": resolved.schema_decision.candidate_binding[
                    "gate_outcome"
                ],
                "representation_type": "ConstraintFrameV01",
                "schema_candidate_id": resolved.schema_candidate.candidate_id,
            }
            enriched_rules = []
            for rule in selected["rules"]:
                if isinstance(rule.get("frame_pattern"), dict):
                    enriched_rules.append(rule)
                    continue
                anchors = tuple(
                    item for item in rule.get("source_anchors", [])
                    if item.get("anchor_id") and item.get("exact_text")
                )
                if not anchors:
                    raise ValueError(
                        "reference rule requires exact source text for frame compilation"
                    )
                try:
                    pattern = compile_frame_pattern_candidate(
                        source_term=" ".join(
                            str(item["exact_text"]) for item in anchors
                        ),
                        source_example=None,
                        source_evidence_refs=tuple(
                            str(item["anchor_id"]) for item in anchors
                        ),
                        accepted_schema_projection=schema_projection,
                        graph_release_ref=str(
                            owner.get("allowed_base_graph_version") or "KG_v0"
                        ),
                    )
                except ValueError as exc:
                    if str(exc) == "source term/example did not produce a constraint frame":
                        continue
                    raise
                enriched_rules.append({**rule, "frame_pattern": asdict(pattern)})
            if not enriched_rules:
                raise ValueError(
                    "accepted reference reviews contain no compilable constraint frames"
                )
            selected = {**selected, "rules": enriched_rules}
        return ResolvedReviewedPatternInputs(resolved, selected)

    def derive_accepted_canonical_patterns(
        self, command: ResolveReviewedPatternInputsCommand
    ) -> tuple[CanonicalRestrictionPatternV1, ...]:
        """Compile reviewed structured frame patterns into Layer-1 values."""

        resolved = self.resolve_reviewed_pattern_inputs(command)
        materialization = resolved.materialization
        positives = tuple(
            item.witness_id for item in materialization.witnesses
            if item.witness_kind == "positive_exemplar"
        )
        hard_negatives = tuple(
            item.witness_id for item in materialization.witnesses
            if item.witness_kind == "hard_negative"
        )
        patterns: list[CanonicalRestrictionPatternV1] = []

        fallback_evidence_refs = [
            dict(evidence)
            for occurrence in materialization.extract_result.output.get(
                "relation_occurrences", []
            )
            for evidence in occurrence.get("evidence_refs", [])
        ] or [
            evidence.model_dump(mode="json", by_alias=True)
            for witness in materialization.witnesses
            if witness.witness_kind == "positive_exemplar"
            for evidence in witness.evidence_refs
        ]

        def shared_payload(rule: dict[str, Any]) -> dict[str, Any]:
            return {
                "organization_id": command.materialization.organization_id,
                "workspace_id": command.materialization.workspace_id,
                "source_metadata": {
                    "source_identity_ref": materialization.source_identity.source_identity_ref,
                    "source_hash": materialization.source_identity.source_hash,
                    "canonical_source_ir_ref": materialization.graph_intent.source_binding.canonical_source_ir_ref,
                    "canonical_source_ir_hash": materialization.graph_intent.source_binding.canonical_source_ir_hash,
                    "evidence_refs": list(
                        rule.get("evidence_refs", []) or fallback_evidence_refs
                    ),
                    "source_classification": materialization.graph_intent.authority_boundary.source_role,
                },
                "witness_metadata": {
                    "positive_exemplar_refs": positives,
                    "hard_negative_refs": hard_negatives,
                },
                "lineage_metadata": {
                    "graph_intent_brief_ref": materialization.graph_intent.graph_intent_id,
                    "graph_intent_brief_hash": materialization.graph_intent.graph_intent_hash,
                    "schema_candidate_ref": materialization.schema_candidate.candidate_id,
                    "schema_candidate_hash": materialization.schema_candidate.candidate_hash,
                },
                "authority_metadata": {
                    "authority_role": "accepted_release_lineage_input",
                    "governed_rule_projection_ref": None,
                    "governed_rule_projection_hash": None,
                },
            }

        def atomic_pattern(
            legacy: FramePatternV01, rule: dict[str, Any]
        ) -> CanonicalRestrictionPatternV1:
            if (
                legacy.corpus_schema_candidate_ref
                != materialization.schema_candidate.candidate_id
                or legacy.lifecycle_state not in {"candidate", "accepted"}
            ):
                raise ValueError("reviewed frame pattern schema/lifecycle mismatch")
            frame = asdict(legacy.frame)
            frame_payload = {"frame_hash": "sha256:" + content_hash(frame), **frame}
            quantity = frame["value_constraint"].get("normalized_quantity")
            source_anchors = tuple(
                str(item["anchor_id"])
                for item in rule.get("source_anchors", [])
                if item.get("anchor_id")
            )
            return create_canonical_restriction_pattern_unverified(
                schema_version="canonical_restriction_pattern.v1",
                pattern_id=legacy.pattern_id,
                pattern_kind="atomic",
                frames=[frame_payload],
                composition={
                    "operator": "atomic", "member_pattern_refs": [],
                    "condition_pattern_refs": [], "exception_pattern_refs": [],
                },
                semantic_dimensions={
                    "target": {"type_ref": frame["target_type_ref"]},
                    "regulated_actor": None,
                    "constraint_dimension": frame["attribute_ref"],
                    "operator": frame["operator_ref"],
                    "value": frame["value_constraint"],
                    "threshold": quantity,
                    "unit": (
                        quantity.get("normalized_unit_ref")
                        if isinstance(quantity, dict) else None
                    ),
                    "modality": frame["modality_ref"],
                    "stage": frame["stage_ref"],
                    "scope": {
                        "condition_node_refs": list(rule.get("condition_node_ids", [])),
                        "exception_node_refs": list(rule.get("exception_node_ids", [])),
                        "source_anchor_refs": list(source_anchors),
                        "tender_family_ref": f"tender.family.{int(rule['family_id'])}",
                    },
                    "consequence": {"ref": frame["consequence_ref"]},
                },
                lexical_evidence_refs=tuple(dict.fromkeys((
                    *legacy.source_evidence_refs, *source_anchors,
                ))),
                **shared_payload(rule),
            )

        for rule in resolved.reviewed_selection["rules"]:
            raw_pattern = rule.get("frame_pattern")
            if not isinstance(raw_pattern, dict):
                raise ValueError(
                    "reviewed rule requires structured frame_pattern for v2 materialization"
                )
            version = raw_pattern.get("schema_version")
            if version == "frame_pattern.v0_1":
                patterns.append(atomic_pattern(
                    TypeAdapter(FramePatternV01).validate_python(raw_pattern), rule
                ))
                continue
            if version != "composed_frame_pattern.v0_1":
                raise ValueError("unsupported reviewed frame pattern version")
            legacy_composed = TypeAdapter(ComposedFramePatternV01).validate_python(
                raw_pattern
            )
            if legacy_composed.lifecycle_state not in {"candidate", "accepted"}:
                raise ValueError("reviewed composed pattern lifecycle mismatch")
            members = tuple(
                atomic_pattern(item, rule) for item in legacy_composed.constituents
            )
            patterns.extend(members)
            source_anchors = tuple(
                str(item["anchor_id"])
                for item in rule.get("source_anchors", [])
                if item.get("anchor_id")
            )
            patterns.append(create_canonical_restriction_pattern_unverified(
                schema_version="canonical_restriction_pattern.v1",
                pattern_id=legacy_composed.pattern_id,
                pattern_kind="composed",
                frames=[frame for member in members for frame in member.frames],
                composition={
                    "operator": "all_of",
                    "member_pattern_refs": [item.pattern_ref for item in members],
                    "condition_pattern_refs": [], "exception_pattern_refs": [],
                },
                semantic_dimensions={
                    "target": None, "regulated_actor": None,
                    "constraint_dimension": "composed_frame_set",
                    "operator": "ALL_OF", "value": None, "threshold": None,
                    "unit": None, "modality": "composed", "stage": None,
                    "scope": {
                        "condition_node_refs": list(rule.get("condition_node_ids", [])),
                        "exception_node_refs": list(rule.get("exception_node_ids", [])),
                        "source_anchor_refs": list(source_anchors),
                        "tender_family_ref": f"tender.family.{int(rule['family_id'])}",
                    },
                    "consequence": None,
                },
                lexical_evidence_refs=tuple(dict.fromkeys((
                    *legacy_composed.source_evidence_refs,
                    *(ref for item in legacy_composed.constituents for ref in item.source_evidence_refs),
                    *source_anchors,
                ))),
                component_artifacts=[
                    (canonical_pattern_bytes_unverified(item), item.pattern_hash)
                    for item in members
                ],
                **shared_payload(rule),
            ))
        refs = [item.pattern_ref for item in patterns]
        if len(refs) != len(set(refs)):
            raise ValueError("reviewed rules produce duplicate canonical pattern identity")
        return tuple(sorted(patterns, key=lambda item: item.pattern_ref.encode("utf-8")))

    @staticmethod
    def _require_canonical_pattern_correction_expectations(
        command: ResolveReviewedPatternInputsCommand,
        *,
        relation_membership: list[dict[str, Any]],
        pattern_membership: list[dict[str, Any]],
    ) -> None:
        expected_patterns = command.canonical_pattern_predecessor_membership
        expected_relations = command.canonical_pattern_unchanged_relation_membership
        if (expected_patterns is None) != (expected_relations is None):
            raise ValueError("canonical-pattern correction expectations must be complete")
        if expected_patterns is None:
            return
        actual_relations = tuple(
            (
                item["relation_instance_ref"],
                item["relation_instance_hash"],
                item["occurrence_grounding_ref"],
                item["occurrence_grounding_hash"],
            )
            for item in relation_membership
        )
        if actual_relations != expected_relations:
            raise ValueError(
                "canonical-pattern correction cannot change relation membership"
            )
        actual_patterns = tuple(
            (item["pattern_ref"], item["pattern_hash"])
            for item in pattern_membership
        )
        expected_set = set(expected_patterns)
        actual_set = set(actual_patterns)
        if (
            len(expected_set) != len(expected_patterns)
            or len(actual_set) != len(actual_patterns)
            or len(expected_set - actual_set) != 1
            or len(actual_set - expected_set) != 1
        ):
            raise ValueError(
                "canonical-pattern correction requires one exact pattern replacement"
            )

    def materialize_accepted_graph_release_v2(
        self, command: ResolveReviewedPatternInputsCommand
    ) -> dict[str, Any]:
        """Materialize one exact v2 release, then bind its Layer-1 members."""

        # One store instance owns one SQLite connection.  Serialize the full
        # read/derive/write journey so concurrent callers cannot interleave
        # replay reads on that connection before the transactional dedupe gate.
        with self._grounding_store._lock:
            return self._materialize_accepted_graph_release_v2_locked(command)

    def _materialize_accepted_graph_release_v2_locked(
        self, command: ResolveReviewedPatternInputsCommand
    ) -> dict[str, Any]:
        """Run the v2 journey while holding the store's re-entrant lock."""

        if not self._materialization_backend_configured:
            raise ValueError(
                "v2 materialization requires an explicitly configured graph backend"
            )
        self._require_integrity(command.materialization)

        prior = self._grounding_store.get_release_state_for_reviews(
            candidate_graph_id=command.candidate_graph_id,
            human_event_ids=tuple(sorted(
                command.human_label_event_ids, key=lambda item: item.encode("utf-8")
            )),
            organization_id=command.materialization.organization_id,
            workspace_id=command.materialization.workspace_id,
        )
        if prior is not None:
            if prior["release"].get("parent_graph_version") != (
                command.materialization.predecessor_release_ref
            ):
                raise ValueError("review selection predecessor release mismatch")
            if prior["lineage_contract"] != "exact_v4_v3":
                raise ValueError("review selection already belongs to a non-v2 release")
            if prior["status"] == "materialized":
                bundle = self._read_materialized_v2_bundle(prior["release"])
                self._require_canonical_pattern_correction_expectations(
                    command,
                    relation_membership=bundle["release"]["relation_membership"],
                    pattern_membership=bundle["release"]["pattern_membership"],
                )
                return bundle

        resolved = self.resolve_reviewed_pattern_inputs(command)
        self._require_predecessor_lineage(command.materialization, resolved.materialization)
        relation_pairs = self._derive_relation_pairs(
            command.materialization, resolved.materialization
        )
        patterns = self.derive_accepted_canonical_patterns(command)
        relation_membership = [
            {
                "relation_instance_ref": instance.relation_instance_id,
                "relation_instance_hash": instance.relation_instance_hash,
                "occurrence_grounding_ref": grounding["grounding_record_id"],
                "occurrence_grounding_hash": grounding["grounding_record_hash"],
            }
            for instance, grounding in relation_pairs
        ]
        pattern_membership = [
            {"pattern_ref": item.pattern_ref, "pattern_hash": item.pattern_hash}
            for item in patterns
        ]
        self._require_canonical_pattern_correction_expectations(
            command,
            relation_membership=relation_membership,
            pattern_membership=pattern_membership,
        )
        release_nodes = self._release_nodes_with_relation_subjects(
            selected_nodes=resolved.reviewed_selection["nodes"],
            relation_pairs=relation_pairs,
            extract_result=resolved.materialization.extract_result,
        )
        occurrence_manifest = OccurrenceGroundingManifestV1.create_unverified({
            "organization_id": command.materialization.organization_id,
            "workspace_id": command.materialization.workspace_id,
            "schema_decision_ref": resolved.materialization.schema_decision.schema_decision_id,
            "schema_decision_hash": resolved.materialization.schema_decision.schema_decision_hash,
            "extract_result_ref": resolved.materialization.extract_result.extract_run_id,
            "extract_result_hash": resolved.materialization.extract_result.extract_result_hash,
            "relation_membership": relation_membership,
        })
        pattern_manifest = CanonicalPatternManifestV1.create_unverified({
            "organization_id": command.materialization.organization_id,
            "workspace_id": command.materialization.workspace_id,
            "pattern_membership": pattern_membership,
        })
        selected = resolved.reviewed_selection
        attempt_id = f"attempt_{uuid4().hex}"
        review_command_hash = "sha256:" + content_hash({
            "materialization": {
                key: value
                for key, value in asdict(command.materialization).items()
                if key != "_source_structure_authority"
                and (not key.startswith("source_structure_") or value is not None)
            },
            "candidate_graph_id": command.candidate_graph_id,
            "human_label_event_ids": sorted(
                command.human_label_event_ids, key=lambda item: item.encode("utf-8")
            ),
            "canonical_pattern_predecessor_membership": (
                command.canonical_pattern_predecessor_membership
            ),
            "canonical_pattern_unchanged_relation_membership": (
                command.canonical_pattern_unchanged_relation_membership
            ),
        })
        store = self._grounding_store
        if prior is not None:
            return self._resume_v2_release(
                prior["release"], prior["status"], resolved=resolved,
                relation_pairs=relation_pairs, patterns=patterns,
                occurrence_manifest=occurrence_manifest,
                pattern_manifest=pattern_manifest,
            )
        with store._lock:
            hosted.begin_write(store._conn)
            try:
                existing = store._conn.execute(
                    hosted.sql_query("SELECT release_json,materialization_status FROM tender_graph_releases "
                    "WHERE review_command_hash=?", 'SELECT release_json,materialization_status FROM tender_graph_releases WHERE review_command_hash=%s'), (review_command_hash,),
                ).fetchone()
                if existing is not None:
                    release_payload = json.loads(existing["release_json"])
                    lifecycle = store._conn.execute(
                        hosted.sql_query("SELECT status FROM tender_graph_release_materialization_events "
                        "WHERE graph_version=? ORDER BY event_id DESC LIMIT 1", 'SELECT status FROM tender_graph_release_materialization_events WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
                        (release_payload["graph_version"],),
                    ).fetchone()
                    existing_status = (
                        lifecycle["status"] if lifecycle is not None
                        else existing["materialization_status"]
                    )
                    store._conn.commit()
                    return self._resume_v2_release(
                        release_payload, existing_status, resolved=resolved,
                        relation_pairs=relation_pairs, patterns=patterns,
                        occurrence_manifest=occurrence_manifest,
                        pattern_manifest=pattern_manifest,
                    )
                graph_version = store._allocate_graph_version_uncommitted(
                    command.candidate_graph_id
                )
                release = AcceptedGraphReleaseV2.create_unverified({
                    "graph_version": graph_version,
                    "parent_graph_version": (
                        command.materialization.predecessor_release_ref
                    ),
                    "candidate_graph_id": command.candidate_graph_id,
                    "graph_delta_id": selected["graph_delta_id"],
                    "human_label_event_ids": selected["human_label_event_ids"],
                    "nodes": release_nodes, "edges": selected["edges"],
                    "rules": selected["rules"],
                    "relation_membership": relation_membership,
                    "pattern_membership": pattern_membership,
                    "backend": "falkordb",
                })
                candidate_graph = store.get_candidate_graph(command.candidate_graph_id)
                lineage = AcceptedGraphReleaseLineageV2.create_unverified(
                    candidate=resolved.materialization.schema_candidate,
                    decision=resolved.materialization.schema_decision,
                    extract_result=resolved.materialization.extract_result,
                    candidate_graph=candidate_graph,
                    accepted_release=release,
                    occurrence_grounding_manifest=occurrence_manifest,
                    canonical_pattern_manifest=pattern_manifest,
                )
                for instance, _grounding in relation_pairs:
                    store._put_unverified_v2_artifact_uncommitted(
                        "knowledge_relation_instance",
                        instance.model_dump(mode="json", by_alias=True),
                    )
                patterns_by_ref = {item.pattern_ref: item for item in patterns}
                for pattern in patterns:
                    component_refs: set[str] = set()
                    pending = list(pattern.composition.member_pattern_refs)
                    while pending:
                        component_ref = pending.pop()
                        if component_ref in component_refs:
                            continue
                        component = patterns_by_ref.get(component_ref)
                        if component is None:
                            raise ValueError(
                                "canonical pattern component is absent from release membership"
                            )
                        component_refs.add(component_ref)
                        pending.extend(component.composition.member_pattern_refs)
                    store._put_unverified_v2_artifact_uncommitted(
                        "canonical_restriction_pattern",
                        pattern.model_dump(mode="json", by_alias=True),
                        component_artifacts=tuple(
                            (
                                canonical_pattern_bytes_unverified(
                                    patterns_by_ref[component_ref]
                                ),
                                patterns_by_ref[component_ref].pattern_hash,
                            )
                            for component_ref in sorted(component_refs)
                        ),
                    )
                store._put_unverified_v2_artifact_uncommitted(
                    "occurrence_grounding_manifest", occurrence_manifest.as_dict()
                )
                store._put_unverified_v2_artifact_uncommitted(
                    "canonical_pattern_manifest", pattern_manifest.as_dict()
                )
                release_payload = release.as_dict()
                store._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_releases "
                    "(graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,"
                    "release_json,content_hash,organization_id,workspace_id,"
                    "materialization_status,review_command_hash,lineage_contract) "
                    "VALUES (?,?,?,?,?,?,?,?,'pending',?,'exact_v4_v3')", "INSERT INTO tender_graph_releases (graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,release_json,content_hash,organization_id,workspace_id,materialization_status,review_command_hash,lineage_contract) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s,'exact_v4_v3')"),
                    (
                        graph_version,
                        command.materialization.predecessor_release_ref,
                        command.candidate_graph_id,
                        selected["graph_delta_id"],
                        json.dumps(release_payload, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")),
                        release.release_hash, command.materialization.organization_id,
                        command.materialization.workspace_id, review_command_hash,
                    ),
                )
                record_accepted_release_origin(
                    store._conn, graph_version=graph_version,
                    candidate_graph_id=command.candidate_graph_id,
                    source_id=str(selected["owner"]["source_id"]),
                    lineage_id=str(selected["owner"]["source_id"]),
                    extraction_run_id=str(selected["owner"]["extraction_run_id"]),
                    organization_id=command.materialization.organization_id,
                    workspace_id=command.materialization.workspace_id,
                    source_identity_ref=resolved.materialization.source_identity.source_identity_ref,
                    release_hash=release.release_hash, release_lineage=lineage,
                    created_at=store._now(), authority_state="candidate",
                )
                store._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_release_materialization_events "
                    "(graph_version,status,occurred_at,attempt_id,lease_expires_at) "
                    "VALUES (?,'pending',?,?,?)", "INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (%s,'pending',%s,%s,%s)"),
                    (graph_version, store._now(), attempt_id, store._lease_expires_at()),
                )
                store._conn.commit()
            except Exception:
                store._conn.rollback()
                raise
        relation_bindings = [
            self._relation_binding(instance, release)
            for instance, _grounding in relation_pairs
        ]
        pattern_bindings = [self._pattern_binding(item, release) for item in patterns]
        conformance_bundle = {
            "release": release_payload,
            "lineage": lineage.as_dict(),
            "occurrence_grounding_manifest": occurrence_manifest.as_dict(),
            "canonical_pattern_manifest": pattern_manifest.as_dict(),
            "accepted_relation_bindings": relation_bindings,
            "accepted_pattern_bindings": pattern_bindings,
            "knowledge_relation_instances": [
                instance.model_dump(mode="json", by_alias=True)
                for instance, _grounding in relation_pairs
            ],
            "canonical_restriction_patterns": [
                pattern.model_dump(mode="json", by_alias=True)
                for pattern in patterns
            ],
        }
        try:
            native_graph = native_graph_from_accepted_release_bundle(
                conformance_bundle
            )
            validate_accepted_release_conformance(
                bundle=conformance_bundle,
                graph=native_graph,
            )
            store._graph_backend.materialize(release_payload)
            if not store._graph_backend.verify_release(release_payload):
                raise ValueError("external v2 graph release missing after materialization")
        except Exception as exc:
            store._append_terminal_attempt(graph_version, attempt_id, "failed", str(exc))
            raise

        self._finalize_v2_release(
            release=release, attempt_id=attempt_id,
            allowed_attempt_states={"pending"},
            relation_bindings=relation_bindings,
            pattern_bindings=pattern_bindings,
        )
        return self._read_materialized_v2_bundle(release_payload)

    def _finalize_v2_release(
        self,
        *,
        release: AcceptedGraphReleaseV2,
        attempt_id: str,
        allowed_attempt_states: set[str],
        relation_bindings: list[dict[str, Any]],
        pattern_bindings: list[dict[str, Any]],
    ) -> None:
        store = self._grounding_store
        graph_version = release.graph_version
        with store._lock:
            hosted.begin_write(store._conn)
            try:
                latest = store._conn.execute(
                    hosted.sql_query("SELECT status,attempt_id FROM tender_graph_release_materialization_events "
                    "WHERE graph_version=? ORDER BY event_id DESC LIMIT 1", 'SELECT status,attempt_id FROM tender_graph_release_materialization_events WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
                    (graph_version,),
                ).fetchone()
                if (
                    latest is None
                    or latest["status"] not in allowed_attempt_states
                    or latest["attempt_id"] != attempt_id
                ):
                    raise ValueError("stale v2 materialization attempt")
                store._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_release_materialization_events "
                    "(graph_version,status,occurred_at,attempt_id) VALUES (?,'materialized',?,?)", "INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id) VALUES (%s,'materialized',%s,%s)"),
                    (graph_version, store._now(), attempt_id),
                )
                origin_id = verify_exact_release_origin_reservation(
                    store._conn, graph_version=graph_version
                )
                promoted = store._conn.execute(
                    hosted.sql_query("UPDATE contextgraph_artifact_origins SET authority_state='authoritative' "
                    "WHERE origin_id=? AND authority_state='candidate'", "UPDATE contextgraph_artifact_origins SET authority_state='authoritative' WHERE origin_id=%s AND authority_state='candidate'"), (origin_id,),
                )
                if promoted.rowcount != 1:
                    raise ValueError("v2 release origin promotion failed")
                for payload in relation_bindings:
                    store._put_unverified_v2_artifact_uncommitted(
                        "accepted_relation_binding", payload
                    )
                for payload in pattern_bindings:
                    store._put_unverified_v2_artifact_uncommitted(
                        "accepted_pattern_binding", payload
                    )
                store._conn.execute(
                    hosted.sql_query("UPDATE tender_graph_releases SET materialization_status='materialized' "
                    "WHERE graph_version=? AND materialization_status='pending'", "UPDATE tender_graph_releases SET materialization_status='materialized' WHERE graph_version=%s AND materialization_status='pending'"),
                    (graph_version,),
                )
                store._conn.commit()
            except Exception:
                store._conn.rollback()
                raise

    def _resume_v2_release(
        self,
        release_payload: dict[str, Any],
        status: str,
        *,
        resolved: ResolvedReviewedPatternInputs,
        relation_pairs: tuple[
            tuple[KnowledgeRelationInstanceV1, dict[str, Any]], ...
        ],
        patterns: tuple[CanonicalRestrictionPatternV1, ...],
        occurrence_manifest: OccurrenceGroundingManifestV1,
        pattern_manifest: CanonicalPatternManifestV1,
    ) -> dict[str, Any]:
        """Retry/finalize the same exact release; never allocate a successor."""

        stored = AcceptedGraphReleaseV2.parse_unverified(release_payload)
        selected = resolved.reviewed_selection
        release_nodes = self._release_nodes_with_relation_subjects(
            selected_nodes=selected["nodes"],
            relation_pairs=relation_pairs,
            extract_result=resolved.materialization.extract_result,
        )
        expected = AcceptedGraphReleaseV2.create_unverified({
            "graph_version": stored.graph_version,
            "parent_graph_version": stored.parent_graph_version,
            "candidate_graph_id": selected["candidate_graph_id"],
            "graph_delta_id": selected["graph_delta_id"],
            "human_label_event_ids": selected["human_label_event_ids"],
            "nodes": release_nodes, "edges": selected["edges"],
            "rules": selected["rules"],
            "relation_membership": occurrence_manifest.as_dict()["relation_membership"],
            "pattern_membership": pattern_manifest.as_dict()["pattern_membership"],
            "backend": "falkordb",
        })
        if stored != expected:
            raise ValueError("stored v2 release differs from exact reviewed inputs")
        if status == "materialized":
            return self._read_materialized_v2_bundle(release_payload)
        if status not in {"failed", "pending", "retry_pending"}:
            raise ValueError(f"v2 release is not retryable from state {status}")

        candidate_graph = self._grounding_store.get_candidate_graph(
            stored.candidate_graph_id
        )
        lineage = AcceptedGraphReleaseLineageV2.create_unverified(
            candidate=resolved.materialization.schema_candidate,
            decision=resolved.materialization.schema_decision,
            extract_result=resolved.materialization.extract_result,
            candidate_graph=candidate_graph,
            accepted_release=stored,
            occurrence_grounding_manifest=occurrence_manifest,
            canonical_pattern_manifest=pattern_manifest,
        )
        relation_bindings = [
            self._relation_binding(instance, stored)
            for instance, _grounding in relation_pairs
        ]
        pattern_bindings = [self._pattern_binding(item, stored) for item in patterns]
        conformance_bundle = {
            "release": stored.as_dict(),
            "lineage": lineage.as_dict(),
            "occurrence_grounding_manifest": occurrence_manifest.as_dict(),
            "canonical_pattern_manifest": pattern_manifest.as_dict(),
            "accepted_relation_bindings": relation_bindings,
            "accepted_pattern_bindings": pattern_bindings,
            "knowledge_relation_instances": [
                instance.model_dump(mode="json", by_alias=True)
                for instance, _grounding in relation_pairs
            ],
            "canonical_restriction_patterns": [
                pattern.model_dump(mode="json", by_alias=True)
                for pattern in patterns
            ],
        }

        store = self._grounding_store
        attempt_id = f"attempt_{uuid4().hex}"
        with store._lock:
            hosted.begin_write(store._conn)
            try:
                latest = store._conn.execute(
                    hosted.sql_query("SELECT status,lease_expires_at FROM "
                    "tender_graph_release_materialization_events "
                    "WHERE graph_version=? ORDER BY event_id DESC LIMIT 1", 'SELECT status,lease_expires_at FROM tender_graph_release_materialization_events WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
                    (stored.graph_version,),
                ).fetchone()
                if latest is None:
                    raise ValueError("v2 release materialization history is missing")
                if latest["status"] == "materialized":
                    store._conn.commit()
                    return self._read_materialized_v2_bundle(release_payload)
                if latest["status"] in {"pending", "retry_pending"} and store._lease_is_live(
                    latest["lease_expires_at"]
                ):
                    raise ValueError("v2 release materialization is already in progress")
                if latest["status"] not in {"failed", "pending", "retry_pending"}:
                    raise ValueError("v2 release materialization state is not retryable")
                verify_exact_release_origin_reservation(
                    store._conn, graph_version=stored.graph_version
                )
                store._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_release_materialization_events "
                    "(graph_version,status,occurred_at,attempt_id,lease_expires_at) "
                    "VALUES (?,'retry_pending',?,?,?)", "INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (%s,'retry_pending',%s,%s,%s)"),
                    (stored.graph_version, store._now(), attempt_id,
                     store._lease_expires_at()),
                )
                store._conn.commit()
            except Exception:
                if hosted.in_transaction(store._conn):
                    store._conn.rollback()
                raise
        try:
            native_graph = native_graph_from_accepted_release_bundle(
                conformance_bundle
            )
            validate_accepted_release_conformance(
                bundle=conformance_bundle,
                graph=native_graph,
            )
            if not store._graph_backend.verify_release(release_payload):
                store._graph_backend.materialize(release_payload)
                if not store._graph_backend.verify_release(release_payload):
                    raise ValueError("external v2 graph release missing after retry")
        except Exception as exc:
            store._append_terminal_attempt(
                stored.graph_version, attempt_id, "failed", str(exc)
            )
            raise
        self._finalize_v2_release(
            release=stored, attempt_id=attempt_id,
            allowed_attempt_states={"retry_pending"},
            relation_bindings=relation_bindings,
            pattern_bindings=pattern_bindings,
        )
        return self._read_materialized_v2_bundle(release_payload)

    @staticmethod
    def _release_nodes_with_relation_subjects(
        *, selected_nodes: list[dict[str, Any]],
        relation_pairs: tuple[
            tuple[KnowledgeRelationInstanceV1, dict[str, Any]], ...
        ],
        extract_result: ExtractResult,
    ) -> list[dict[str, Any]]:
        """Include every grounded relation endpoint/frame in v2 topology."""

        release_nodes = [dict(item) for item in selected_nodes]
        occupied = {str(item.get("node_id")) for item in release_nodes}
        extracted_nodes = {
            str(item.get("node_id")): item
            for item in extract_result.output.get("nodes", ())
            if isinstance(item, dict) and item.get("node_id")
        }
        extracted_frames = {
            str(item.get("frame_id")): item
            for item in extract_result.output.get("frames", ())
            if isinstance(item, dict) and item.get("frame_id")
        }
        required_nodes = {
            ref
            for instance, _grounding in relation_pairs
            for ref in (instance.source_node_id, instance.target_node_id)
            if ref is not None
        }
        required_frames = {
            instance.frame_ref
            for instance, _grounding in relation_pairs
            if instance.frame_ref is not None
        }
        for node_id in sorted(required_nodes, key=lambda item: item.encode("utf-8")):
            if node_id in occupied:
                continue
            source = extracted_nodes.get(node_id)
            if source is None:
                raise ValueError("accepted relation endpoint is absent from ExtractResult")
            release_nodes.append({
                "node_id": node_id,
                "node_type": str(source.get("type_id") or "concept"),
                "canonical_label": str(
                    source.get("canonical_label")
                    or source.get("display_name")
                    or node_id
                ),
                "semantic_payload": dict(source),
            })
            occupied.add(node_id)
        for frame_id in sorted(required_frames, key=lambda item: item.encode("utf-8")):
            if frame_id in occupied:
                continue
            source = extracted_frames.get(frame_id)
            if source is None:
                raise ValueError("accepted relation frame is absent from ExtractResult")
            release_nodes.append({
                "node_id": frame_id,
                "node_type": "semantic_constraint_frame",
                "canonical_label": frame_id,
                "semantic_payload": dict(source),
            })
            occupied.add(frame_id)
        return release_nodes

    @staticmethod
    def _relation_binding(
        instance: KnowledgeRelationInstanceV1, release: AcceptedGraphReleaseV2
    ) -> dict[str, Any]:
        payload = {
            "schemaVersion": "accepted_relation_binding.v1",
            "organizationId": instance.organization_id,
            "workspaceId": instance.workspace_id,
            "relationInstanceRef": instance.relation_instance_id,
            "relationInstanceHash": instance.relation_instance_hash,
            "acceptedGraphReleaseRef": release.graph_version,
            "acceptedGraphReleaseHash": release.release_hash,
        }
        digest = content_hash(payload)
        return AcceptedRelationBindingV1.model_validate({
            **payload, "bindingId": f"arb_{digest[:24]}",
            "bindingHash": f"sha256:{digest}",
        }).model_dump(mode="json", by_alias=True)

    @staticmethod
    def _pattern_binding(
        pattern: CanonicalRestrictionPatternV1, release: AcceptedGraphReleaseV2
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "accepted_pattern_binding.v1",
            "organization_id": pattern.organization_id,
            "workspace_id": pattern.workspace_id,
            "pattern_ref": pattern.pattern_ref, "pattern_hash": pattern.pattern_hash,
            "accepted_graph_release_ref": release.graph_version,
            "accepted_graph_release_hash": release.release_hash,
        }
        digest = content_hash(payload)
        return {**payload, "binding_id": f"apb_{digest[:24]}",
                "binding_hash": f"sha256:{digest}"}

    def _read_materialized_v2_bundle(
        self, release_payload: dict[str, Any]
    ) -> dict[str, Any]:
        return MaterializedAcceptedGraphBundleReader(
            store=self._grounding_store,
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
        ).read(release_payload)

    @staticmethod
    def _semantic_edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
        source = str(edge.get("source_node_id") or "")
        target = str(edge.get("target_node_id") or "")
        if edge.get("direction") == "target_to_source":
            return target, source
        if (
            edge.get("direction") == "undirected"
            and source.encode("utf-8") > target.encode("utf-8")
        ):
            return target, source
        return source, target

    def _require_integrity(self, command: ResolveMaterializationInputsCommand) -> None:
        if type(self) is not AcceptedSemanticGraphMaterializationService:
            raise ValueError("materialization requires the exact sealed service")
        if command.organization_id != self._organization_id or command.workspace_id != self._workspace_id:
            raise ValueError("materialization tenant scope mismatch")
        for value in (
            command.graph_intent_hash, command.schema_candidate_hash,
            command.schema_decision_hash, command.extract_result_hash,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("materialization pins require exact sha256 hashes")
        predecessor = (
            command.predecessor_release_ref,
            command.predecessor_release_hash,
        )
        if (predecessor[0] is None) != (predecessor[1] is None):
            raise ValueError("predecessor release ref/hash must align")
        if predecessor[0] is not None:
            if (
                re.fullmatch(r"KG_v[1-9]\d*", predecessor[0]) is None
                or _SHA256.fullmatch(str(predecessor[1])) is None
            ):
                raise ValueError("predecessor release identity is invalid")
            release = self._grounding_store.get_graph_release(
                predecessor[0],
                organization_id=command.organization_id,
                workspace_id=command.workspace_id,
            )
            if (
                release.get("schema_version") != "accepted_graph_release.v2"
                or release.get("release_hash") != predecessor[1]
            ):
                raise ValueError("predecessor release ref/hash mismatch")

    def _require_predecessor_lineage(
        self,
        command: ResolveMaterializationInputsCommand,
        resolved: ResolvedMaterializationInputs,
    ) -> None:
        if command.predecessor_release_ref is None:
            return
        predecessor = self._grounding_store.get_graph_release(
            command.predecessor_release_ref,
            organization_id=command.organization_id,
            workspace_id=command.workspace_id,
        )
        bundle = self._read_materialized_v2_bundle(predecessor)
        lineage = bundle["lineage"]
        expected = (
            command.organization_id,
            command.workspace_id,
            command.lineage_id,
            command.source_identity_ref,
            resolved.source_identity.source_hash,
        )
        actual = tuple(
            lineage[field]
            for field in (
                "organization_id",
                "workspace_id",
                "lineage_id",
                "source_identity_ref",
                "source_hash",
            )
        )
        if actual != expected:
            if not (
                command.source_structure_validation_ref
                and command.source_structure_validation_hash
                and command._source_structure_authority
                is _SOURCE_STRUCTURE_MATERIALIZATION_AUTHORITY
            ):
                raise ValueError("predecessor release source lineage mismatch")
            validation = SourceStructureSemanticValidationStore(
                self._workspace_root, workspace_id=command.workspace_id
            ).get(
                command.source_structure_validation_ref,
                validation_hash=command.source_structure_validation_hash,
            )
            predecessor_identity = self._source_store.get_source_identity(
                source_identity_ref=lineage["source_identity_ref"],
                organization_id=command.organization_id,
                workspace_id=command.workspace_id,
                lineage_id=command.lineage_id,
            )
            if predecessor_identity is None:
                raise ValueError("predecessor release SourceIdentity is missing")
            predecessor_identity_hash = "sha256:" + content_hash(
                predecessor_identity.model_dump(
                    mode="json", by_alias=True, exclude_none=False
                )
            )
            successor_identity_hash = "sha256:" + content_hash(
                resolved.source_identity.model_dump(
                    mode="json", by_alias=True, exclude_none=False
                )
            )
            proposal_ref = resolved.graph_intent.predecessor_graph_intent_ref
            if proposal_ref is None:
                raise ValueError("source-structure accepted GraphIntent proposal is missing")
            proposal = self._intent_store.get(proposal_ref)
            if (
                validation.outcome != "passed"
                or validation.organization_id != command.organization_id
                or validation.workspace_id != command.workspace_id
                or validation.predecessor_graph_intent_ref
                != lineage["graph_intent_ref"]
                or validation.predecessor_graph_intent_hash
                != lineage["graph_intent_hash"]
                or validation.successor_graph_intent_ref != proposal.graph_intent_id
                or validation.successor_graph_intent_hash != proposal.graph_intent_hash
                or validation.predecessor_source_identity_ref
                != lineage["source_identity_ref"]
                or validation.predecessor_source_identity_hash
                != predecessor_identity_hash
                or validation.successor_source_identity_ref
                != command.source_identity_ref
                or validation.successor_source_identity_hash
                != successor_identity_hash
                or validation.predecessor_canonical_source_ir_ref
                != lineage["canonical_source_ir_ref"]
                or validation.predecessor_canonical_source_ir_hash
                != lineage["canonical_source_ir_hash"]
                or validation.successor_canonical_source_ir_ref
                != resolved.graph_intent.source_binding.canonical_source_ir_ref
                or validation.successor_canonical_source_ir_hash
                != resolved.graph_intent.source_binding.canonical_source_ir_hash
                or proposal.source_binding != resolved.graph_intent.source_binding
                or proposal.review.state != "proposed"
                or resolved.graph_intent.review.state != "accepted"
            ):
                raise ValueError(
                    "predecessor release source-structure lineage mismatch"
                )

    @staticmethod
    def _source_binding(intent: GraphIntentBrief) -> dict[str, Any]:
        return intent.source_binding.as_dict()

    def _match_source(
        self, intent: GraphIntentBrief, source: SourceIdentity,
        command: ResolveMaterializationInputsCommand,
    ) -> None:
        binding = intent.source_binding
        expected = (
            command.organization_id, command.workspace_id, command.lineage_id,
            command.source_identity_ref, source.source_hash,
        )
        actual = (
            binding.organization_id, binding.workspace_id, binding.lineage_id,
            binding.source_identity_ref, binding.source_hash,
        )
        if actual != expected:
            raise ValueError("GraphIntent/SourceIdentity binding mismatch")
        if binding.source_version != source.source_version:
            supplied = source.model_copy(update={"source_version": binding.source_version})
            if not KnowledgeBuilderStore.permits_historical_null_source_version(
                self._source_store, supplied_identity=supplied,
                registered_identity=source, lineage_id=command.lineage_id,
            ):
                raise ValueError("GraphIntent/SourceIdentity source version mismatch")

    def _read_canonical(
        self, intent: GraphIntentBrief, source: SourceIdentity
    ) -> CanonicalSourceDocument:
        relative = Path(intent.source_binding.canonical_source_ir_ref)
        workspace_path = self._intent_store.workspace_path
        target = (workspace_path / relative).resolve()
        artifact_root = (workspace_path / "artifacts").resolve()
        if artifact_root not in target.parents or not target.is_file():
            raise ValueError("governed CanonicalSourceIR is missing")
        payload = json.loads(target.read_text(encoding="utf-8"))
        canonical = CanonicalSourceDocument.model_validate(payload)
        if (
            canonical.source_identity_ref != source.source_identity_ref
            or canonical_ir_content_hash(payload) != intent.source_binding.canonical_source_ir_hash
        ):
            raise ValueError("CanonicalSourceIR exact source/hash mismatch")
        return canonical

    def _read_extract(self, command: ResolveMaterializationInputsCommand) -> ExtractResult:
        target = (
            self._intent_store.workspace_path / "runs" / command.extract_result_ref
            / "artifacts" / "extract_result.v1.json"
        ).resolve()
        runs_root = (self._intent_store.workspace_path / "runs").resolve()
        if runs_root not in target.parents or not target.is_file():
            raise ValueError("governed ExtractResult artifact is missing")
        result = ExtractResult.from_dict(json.loads(target.read_text(encoding="utf-8")))
        if result.extract_run_id != command.extract_result_ref or result.extract_result_hash != command.extract_result_hash:
            raise ValueError("ExtractResult exact ref/hash mismatch")
        return result

    def _validate_compiler_provenance(
        self, candidate: CorpusGraphSchemaCandidate, intent: GraphIntentBrief
    ) -> None:
        provenance = dict(candidate.compiler_provenance or {})
        profile = CorpusGraphSchemaProfileRegistry.resolve(
            self._profile_registry, str(provenance.get("profile_id") or "")
        )
        if (
            provenance.get("profile_hash") != profile.content_hash
            or provenance.get("catalog_hash") != profile.graph_intent_catalog_hash
            or provenance.get("profile_relation_catalog_hash")
            != profile.knowledge_relation_catalog_hash
            or provenance.get("candidate_schema_version") != V4_SCHEMA_VERSION
        ):
            raise ValueError("candidate compiler profile provenance mismatch")
        taxonomy = KnowledgeRelationTaxonomyService.load_canonical(
            self._taxonomy_service,
            str(provenance.get("taxonomy_ref") or ""),
            expected_taxonomy_hash=str(provenance.get("taxonomy_hash") or ""),
        )
        if provenance.get("taxonomy_hash") != taxonomy.taxonomy_hash:
            raise ValueError("candidate compiler taxonomy provenance mismatch")

    def _has_decision_successor(self, decision_ref: str) -> bool:
        if not self._decision_store.root.is_dir():
            return False
        for path in self._decision_store.root.glob("*/schema_decision.v3.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("predecessor_schema_decision_ref") != decision_ref:
                continue
            candidate = KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
                self._taxonomy_service, payload
            )
            if candidate.predecessor_schema_decision_ref == decision_ref:
                return True
        return False

    def _read_witnesses(
        self, command: ResolveMaterializationInputsCommand, intent: GraphIntentBrief,
        candidate: CorpusGraphSchemaCandidate,
        canonical: CanonicalSourceDocument,
    ) -> tuple[tuple[SourceSemanticWitnessSetManifestV1, ...], tuple[SourceSemanticWitnessV1, ...]]:
        gates = [
            item.get("selected_value")
            for item in candidate.synthesis_decisions
            if item.get("decision_id") == "accepted-witness-exemplar-gate"
            and item.get("decision_kind") == "exemplar_reconstruction"
            and item.get("evidence_origin") == "accepted_source_semantic_witnesses"
        ]
        if len(gates) != 1 or not isinstance(gates[0], dict):
            raise ValueError("Candidate.v4 requires exactly one accepted exemplar gate")
        gate = gates[0]
        if (
            gate.get("schema_version") != "accepted_exemplar_gate.v1"
            or gate.get("positive_count") != 32
            or gate.get("hard_negative_count") != 6
            or gate.get("material_mismatches") != []
        ):
            raise ValueError("accepted exemplar gate 32+6 contract mismatch")
        pins = (
            (gate.get("positive_manifest_ref"), gate.get("positive_manifest_hash"), "positive_exemplar"),
            (gate.get("hard_negative_manifest_ref"), gate.get("hard_negative_manifest_hash"), "hard_negative"),
        )
        manifests = []
        witnesses = []
        seen = set()
        by_kind: dict[str, tuple[SourceSemanticWitnessV1, ...]] = {}
        trusted_by_kind: dict[str, Any] | None = None
        globally_replayed_kinds: set[str] = set()
        for manifest_ref, manifest_hash, expected_kind in pins:
            if not isinstance(manifest_ref, str) or not isinstance(manifest_hash, str):
                raise ValueError("accepted exemplar gate manifest pins are invalid")
            if manifest_ref in seen:
                raise ValueError("duplicate W1 witness manifest pin")
            seen.add(manifest_ref)
            artifact_ref = (
                f"artifacts/source-semantic-witness-set/{manifest_hash[7:]}/"
                "source_semantic_witness_set_manifest.v1.json"
            )
            trusted = None
            try:
                manifest = HumanGroundingStore.replay_source_semantic_witness_manifest(
                    self._grounding_store, organization_id=command.organization_id,
                    workspace_id=command.workspace_id, manifest_ref=manifest_ref,
                    content_hash=manifest_hash, artifact_ref=artifact_ref,
                )
            except ValueError as exc:
                if str(exc) != "witness manifest replay record is missing":
                    raise
                if trusted_by_kind is None:
                    from services.graph_schema_discovery_service import (
                        _resolve_trusted_witness_gate_package,
                        _resolve_trusted_witness_gate_packages_by_manifest_pins,
                    )

                    trusted_packages = (
                        _resolve_trusted_witness_gate_packages_by_manifest_pins(
                            positive_manifest_ref=str(pins[0][0]),
                            positive_manifest_hash=str(pins[0][1]),
                            hard_negative_manifest_ref=str(pins[1][0]),
                            hard_negative_manifest_hash=str(pins[1][1]),
                            positive_document=canonical,
                        )
                    )
                    trusted_by_kind = {
                        package.witness_kind: _resolve_trusted_witness_gate_package(
                            package
                        )
                        for package in trusted_packages
                    }
                trusted = trusted_by_kind.get(expected_kind)
                if trusted is None:
                    raise
                manifest = trusted.accepted_manifest
                members = trusted.accepted_witnesses
                globally_replayed_kinds.add(expected_kind)
            successor = self._grounding_store._conn.execute(
                hosted.sql_query("""SELECT 1 FROM source_semantic_witness_manifests
                   WHERE organization_id=? AND workspace_id=?
                     AND predecessor_manifest_ref=? LIMIT 1""", 'SELECT 1 FROM source_semantic_witness_manifests\n                   WHERE organization_id=%s AND workspace_id=%s\n                     AND predecessor_manifest_ref=%s LIMIT 1'),
                (command.organization_id, command.workspace_id, manifest.manifest_ref),
            ).fetchone()
            if successor is not None:
                raise ValueError("W1 witness manifest is stale/superseded")
            if trusted is None:
                members = tuple(
                    HumanGroundingStore.replay_source_semantic_witness(
                        self._grounding_store, organization_id=command.organization_id,
                        workspace_id=command.workspace_id, witness_ref=item.witness_ref,
                        witness_hash=item.witness_hash,
                    )
                    for item in manifest.witnesses
                )
            if any(item.review.state not in {"accepted", "edited"} for item in members):
                raise ValueError("W1 witness member is not accepted/edited")
            if any(item.witness_kind != expected_kind for item in members):
                raise ValueError("W1 witness manifest kind mismatch")
            by_kind[expected_kind] = members
            manifests.append(manifest)
            witnesses.extend(members)
        if (
            len(by_kind.get("positive_exemplar", ())) != 32
            or len(by_kind.get("hard_negative", ())) != 6
            or set(gate.get("hard_negative_witness_refs") or ())
            != {item.witness_id for item in by_kind["hard_negative"]}
            or not set(gate.get("representative_positive_witness_refs") or ()).issubset(
                {item.witness_id for item in by_kind["positive_exemplar"]}
            )
        ):
            raise ValueError("accepted exemplar gate member set mismatch")
        source_bound = [
            manifest for manifest in manifests
            if manifest.source_binding.source_identity_ref
            == intent.source_binding.source_identity_ref
            and manifest.source_binding.source_hash == intent.source_binding.source_hash
            and manifest.source_binding.canonical_source_ir_ref
            == intent.source_binding.canonical_source_ir_ref
            and manifest.source_binding.canonical_source_ir_hash
            == intent.source_binding.canonical_source_ir_hash
        ]
        if not source_bound and "positive_exemplar" in globally_replayed_kinds:
            canonical_text_hashes = {
                "sha256:" + sha256(
                    str(block.content).encode("utf-8")
                ).hexdigest()
                for block in canonical.blocks
            }
            portable = by_kind.get("positive_exemplar", ())
            if portable and all(
                witness.source_binding.source_hash == intent.source_binding.source_hash
                and witness.exact_text_hash in canonical_text_hashes
                for witness in portable
            ):
                source_bound = [manifests[0]]
        if len(source_bound) != 1:
            raise ValueError(
                "accepted W1 witness set must contain exactly one source-bound manifest"
            )
        return tuple(manifests), tuple(witnesses)

    def _read_terminal_groundings(
        self, command: ResolveMaterializationInputsCommand, extract: ExtractResult,
        decision: SchemaDecision,
    ) -> tuple[dict[str, Any], ...]:
        records = HumanGroundingStore.list_relation_occurrence_groundings(
            self._grounding_store, extract_result_ref=extract.extract_run_id,
            organization_id=command.organization_id, workspace_id=command.workspace_id,
        )
        raw_occurrences = extract.output.get("relation_occurrences")
        if not isinstance(raw_occurrences, list):
            raise ValueError(
                "ExtractResult lacks required relation_occurrences authority surface"
            )
        occurrences = [
            ExtractedRelationOccurrenceV1.model_validate(item)
            for item in raw_occurrences
        ]
        admitted_pins = {
            str(item["relation_type_id"]): dict(item)
            for item in decision.admitted_relation_type_bindings
        }
        admitted: dict[str, dict[str, Any]] = {}
        for item in decision.approved_schema["relation_types"]:
            relation = dict(item["relation_type_candidate"])
            relation_type_id = str(relation["relation_type_id"])
            pins = admitted_pins.get(relation_type_id)
            if pins is None:
                continue
            admitted[relation_type_id] = {**relation, **pins}
        if set(admitted) != set(admitted_pins) or not (
            GovernedExtractService._valid_relation_occurrence_surface(
                extract.output, admitted
            )
        ):
            raise ValueError("ExtractResult relation occurrence surface is invalid")
        occurrence_refs = [item.occurrence_ref for item in occurrences]
        occurrences_by_ref = {item.occurrence_ref: item for item in occurrences}
        record_refs = [item.get("extracted_occurrence_ref") for item in records]
        if (
            len(occurrence_refs) != len(set(occurrence_refs))
            or sorted(record_refs) != sorted(occurrence_refs)
        ):
            raise ValueError("missing, multiple, or extra occurrence Grounding records")
        if len(record_refs) != len(set(record_refs)):
            raise ValueError("multiple occurrence Grounding records are not terminal")
        if any(
            item.get("organization_id") != command.organization_id
            or item.get("workspace_id") != command.workspace_id
            or item.get("extract_result_ref") != extract.extract_run_id
            or item.get("extract_result_hash") != extract.extract_result_hash
            or item.get("schema_decision_ref") != decision.schema_decision_id
            or item.get("schema_decision_hash") != decision.schema_decision_hash
            or item.get("extracted_occurrence_hash")
            != occurrences_by_ref[str(item.get("extracted_occurrence_ref"))].occurrence_hash
            or item.get("outcome") not in {"accepted", "corrected"}
            for item in records
        ):
            raise ValueError(
                "occurrence Grounding is rejected/nonterminal or pinned to another schema"
            )
        correction_fields = (
            "corrected_occurrence_ref", "corrected_occurrence_hash",
            "supersedes_occurrence_ref", "supersedes_occurrence_hash",
            "relation_usage_grounding_decision_ref",
            "relation_usage_grounding_decision_hash",
        )
        for item in records:
            original = occurrences_by_ref[str(item["extracted_occurrence_ref"])]
            if item["outcome"] == "accepted":
                if any(item.get(field) is not None for field in correction_fields):
                    raise ValueError("accepted occurrence Grounding has correction fields")
            elif (
                any(item.get(field) is None for field in correction_fields)
                or item.get("supersedes_occurrence_ref") != original.occurrence_ref
                or item.get("supersedes_occurrence_hash") != original.occurrence_hash
            ):
                raise ValueError("corrected occurrence Grounding lineage mismatch")
        return tuple(sorted(records, key=lambda item: str(item["extracted_occurrence_ref"])))

    def resolve_accepted_graph_release_v2(
        self,
        release_ref: str,
        *,
        release_hash: str,
    ) -> dict[str, Any]:
        """Replay one exact materialized v2 release and its governed lineage."""

        if type(self) is not AcceptedSemanticGraphMaterializationService:
            raise ValueError("release replay requires the exact sealed service")
        if _SHA256.fullmatch(release_hash) is None:
            raise ValueError("release replay requires an exact sha256 hash")
        release = self._grounding_store.get_graph_release(
            release_ref,
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
        )
        if release.get("release_hash") != release_hash:
            raise ValueError("accepted release ref/hash mismatch")
        return self._read_materialized_v2_bundle(release)


__all__ = [
    "AcceptedSemanticGraphMaterializationService", "ResolveMaterializationInputsCommand",
    "ResolveReviewedPatternInputsCommand", "ResolvedMaterializationInputs",
    "ResolvedReviewedPatternInputs", "_SOURCE_STRUCTURE_MATERIALIZATION_AUTHORITY",
]
