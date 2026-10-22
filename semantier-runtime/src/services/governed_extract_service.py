"""Governed bounded Extract over CanonicalSourceIR and immutable SchemaDecision."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.corpus_graph_schema import (
    CorpusGraphSchemaCandidate,
    RelationTypeCandidate,
)
from contracts.governed_extract import ExtractRequest, ExtractResult
from contracts.extracted_relation_occurrence import (
    ExtractedRelationOccurrenceV1,
    create_extracted_relation_occurrence,
)
from contracts.schema_decision import SchemaDecision
from contracts.schema_gate_policy import SchemaGatePolicy
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    canonical_elements_from_ir,
)
from storage.corpus_graph_schema_candidate_store import (
    CorpusGraphSchemaCandidateStore,
)
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.profile_relation_type_proposal_store import (
    ProfileRelationTypeProposalStore,
)
from storage.schema_decision_store import SchemaDecisionStore
from storage.atomic_json_artifact import publish_immutable_json
from services.corpus_graph_schema_profile_registry import (
    CorpusGraphSchemaProfileRegistry,
)
from services.knowledge_relation_taxonomy_service import (
    KnowledgeRelationTaxonomyService,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class GovernedExtractService:
    """Fail-closed test-flight extraction; output remains candidate evidence."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        graph_intent_store: GraphIntentBriefStore | None = None,
        profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.workspace_dir = (
            self.workspace_root
            if self.workspace_root.name == workspace_id
            and self.workspace_root.parent.name == "workspaces"
            else self.workspace_root / "workspaces" / workspace_id
        )
        self.schema_decisions = SchemaDecisionStore(
            workspace_root, workspace_id=workspace_id
        )
        self.candidates = CorpusGraphSchemaCandidateStore(
            workspace_root, workspace_id=workspace_id
        )
        self.graph_intent_store = graph_intent_store
        self.profile_registry = profile_registry or CorpusGraphSchemaProfileRegistry()
        self.profile_relation_type_proposal_store = (
            ProfileRelationTypeProposalStore(
                workspace_root,
                workspace_id=workspace_id,
                organization_id=organization_id,
            )
        )

    def _taxonomy_service(self) -> KnowledgeRelationTaxonomyService:
        return KnowledgeRelationTaxonomyService(
            repository_root=Path(__file__).resolve().parents[2],
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.profile_relation_type_proposal_store
            ),
        )

    def extract(self, request: ExtractRequest) -> dict[str, Any]:
        if request.organization_id != self.organization_id:
            raise ValueError("ARTIFACT_SCOPE_MISMATCH: organization")
        if request.workspace_id != self.workspace_id:
            raise ValueError("ARTIFACT_SCOPE_MISMATCH: workspace")

        taxonomy_service = self._taxonomy_service()
        decision_payload = self.schema_decisions.get(
            request.schema_decision_ref,
            taxonomy_service=taxonomy_service,
        )
        decision = (
            KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
                taxonomy_service, decision_payload
            )
            if decision_payload.get("schema_version") == "schema_decision.v3"
            else SchemaDecision.from_dict(decision_payload)
        )
        if decision.schema_version != "schema_decision.v3":
            raise ValueError("historical SchemaDecision is read/replay-only")
        if decision.schema_decision_hash != request.schema_decision_hash:
            raise ValueError("SCHEMA_DECISION_HASH_MISMATCH")
        if decision.source_binding["lineage_id"] != request.lineage_id:
            raise ValueError("SOURCE_BINDING_MISMATCH: lineage")
        canonical_source_ref = decision.source_binding.get(
            "canonical_source_ir_ref",
            decision.source_binding.get("canonical_source_ref"),
        )
        if (
            canonical_source_ref != request.canonical_source_ir_ref
            or decision.source_binding["source_hash"]
            != request.canonical_source_hash
        ):
            raise ValueError("SOURCE_BINDING_MISMATCH")

        canonical_path = self._artifact_path(request.canonical_source_ir_ref)
        document = CanonicalSourceDocument.model_validate_json(
            canonical_path.read_text(encoding="utf-8")
        )
        elements = canonical_elements_from_ir(document)
        actual_hashes = {
            str(item["anchor"]["document_hash"]) for item in elements
        }
        if actual_hashes != {request.canonical_source_hash}:
            raise ValueError("SOURCE_BINDING_MISMATCH: canonical artifact hash")

        candidate = CorpusGraphSchemaCandidate.from_dict(
            self.candidates.get_by_hash(
                str(decision.candidate_binding["candidate_hash"])
            )
        )
        if candidate.candidate_id != decision.candidate_binding["candidate_id"]:
            raise ValueError("CANDIDATE_BINDING_MISMATCH")
        candidate_source = candidate.source_binding or {}
        candidate_canonical_ref = candidate_source.get(
            "canonical_source_ir_ref", candidate_source.get("canonical_source_ref")
        )
        if (
            candidate_source.get("lineage_id") != request.lineage_id
            or candidate_canonical_ref != request.canonical_source_ir_ref
            or candidate_source.get("source_hash") != request.canonical_source_hash
        ):
            raise ValueError("CANDIDATE_BINDING_MISMATCH: source")
        admitted_relation_types: list[dict[str, Any]] = []
        admitted_relation_directions: dict[str, bool] | None = None
        admitted_relation_occurrence_bindings: dict[str, dict[str, Any]] = {}
        if decision.schema_version == "schema_decision.v3":
            if candidate.schema_version != "corpus_graph_schema_candidate.v4":
                raise ValueError("TAXONOMY_ADMISSION_MISMATCH: candidate version")
            if candidate_source != decision.source_binding:
                raise ValueError("CANDIDATE_BINDING_MISMATCH: exact source pins")
            admitted_relation_directions = {}
            proposal_origin = (
                "system"
                if decision.candidate_binding["decided_by"] == "system"
                else "human"
            )
            for relation in decision.approved_schema["relation_types"]:
                relation_candidate = RelationTypeCandidate.model_validate(
                    relation["relation_type_candidate"]
                )
                admission = (
                    KnowledgeRelationTaxonomyService.resolve_candidate_admission(
                        taxonomy_service,
                        relation_candidate,
                        proposal_origin=proposal_origin,
                    )
                )
                admitted = admission.relation_type
                admitted_relation_directions[admitted.relation_type_id] = (
                    admitted.directed
                )
                admitted_relation_occurrence_bindings[admitted.relation_type_id] = {
                    "representation_kind": admitted.representation_kind,
                    "projection_shape": admitted.projection_shape,
                    "directed": admitted.directed,
                    "source_type_refs": tuple(admitted.source_type_refs),
                    "target_type_refs": tuple(admitted.target_type_refs),
                    "source_role_refs": tuple(admitted.source_role_refs),
                    "target_role_refs": tuple(admitted.target_role_refs),
                    "evidence_refs": [
                        item.model_dump(mode="json", by_alias=True)
                        for item in relation_candidate.evidence_refs
                    ],
                }
                admitted_relation_types.append(
                    {
                        "relation_type_id": admitted.relation_type_id,
                        "family_id": admitted.family_id,
                        "taxonomy_ref": admission.taxonomy_ref,
                        "taxonomy_hash": admission.taxonomy_hash,
                        "profile_ref": admission.profile_ref,
                        "profile_hash": admission.profile_hash,
                        "profile_relation_catalog_hash": (
                            admission.profile_relation_catalog_hash
                        ),
                    }
                )
            admitted_relation_types.sort(
                key=lambda item: item["relation_type_id"].encode("utf-8")
            )

        extract_run_id = f"extract-run-{content_hash(request.as_dict())[:24]}"
        existing_path = (
            self.workspace_dir
            / "runs"
            / extract_run_id
            / "artifacts"
            / "extract_result.v1.json"
        )
        if existing_path.is_file():
            existing = ExtractResult.from_dict(
                json.loads(existing_path.read_text(encoding="utf-8"))
            )

            def invalid_v3_edge(edge: dict[str, Any]) -> bool:
                relation_type_id = str(edge.get("relation_type_id") or "")
                admitted_directed = (admitted_relation_directions or {}).get(
                    relation_type_id
                )
                if (
                    admitted_directed is None
                    or "type_id" in edge
                    or "relation_family_id" in edge
                ):
                    return True
                direction = str(edge.get("direction") or "")
                if admitted_directed:
                    return direction not in {
                        "source_to_target",
                        "target_to_source",
                    }
                source_node_id = str(edge.get("source_node_id") or "")
                target_node_id = str(edge.get("target_node_id") or "")
                return (
                    direction != "undirected"
                    or not source_node_id
                    or not target_node_id
                    or source_node_id.encode("utf-8")
                    > target_node_id.encode("utf-8")
                    or "declared_source_type" in edge
                    or "declared_target_type" in edge
                )

            expected_source_binding = {
                "lineage_id": request.lineage_id,
                "canonical_source_ir_ref": request.canonical_source_ir_ref,
                "canonical_source_hash": request.canonical_source_hash,
            }
            if (
                existing.extract_run_id != extract_run_id
                or existing.source_binding != expected_source_binding
                or existing.schema_binding.get("schema_decision_ref")
                != decision.schema_decision_id
                or existing.schema_binding.get("schema_decision_hash")
                != decision.schema_decision_hash
                or existing.schema_binding.get("candidate_ref")
                != candidate.candidate_id
                or existing.schema_binding.get("candidate_hash")
                != candidate.candidate_hash
                or (
                    decision.schema_version == "schema_decision.v3"
                    and (
                        existing.schema_binding.get("admitted_relation_types")
                        != admitted_relation_types
                        or any(
                            invalid_v3_edge(edge)
                            for edge in existing.output.get("edges", [])
                        )
                        or not self._valid_relation_occurrence_surface(
                            existing.output,
                            admitted_relation_occurrence_bindings,
                        )
                    )
                )
            ):
                raise ValueError("EXTRACT_RESULT_ARTIFACT_CONFLICT")
            return existing.as_dict()

        started_at = _now()
        output, evidence_bindings, unresolved_relation_evidence = self._execute_plan(
            elements=elements,
            decision=decision,
            admitted_relation_directions=admitted_relation_directions,
        )
        if decision.schema_version == "schema_decision.v3":
            occurrence_unresolved = self._add_relation_occurrence_surface(
                output,
                admitted_relation_occurrence_bindings,
                evidence_bindings=evidence_bindings,
            )
            unresolved_relation_evidence.extend(occurrence_unresolved)
            unresolved_relation_evidence.sort(
                key=lambda item: (
                    str(item.get("relation_type_id") or "").encode("utf-8"),
                    tuple(
                        str(ref).encode("utf-8")
                        for ref in (item.get("evidence_refs") or [])
                    ),
                )
            )
        completed_at = _now()
        extraction_contract_hash = f"sha256:{content_hash({'approved_schema': decision.approved_schema, 'extraction_plan': list(decision.extraction_plan)})}"
        semantica = decision.semantica_binding
        result = ExtractResult.create(
            {
                "extract_run_id": extract_run_id,
                "status": "succeeded",
                "source_binding": {
                    "lineage_id": request.lineage_id,
                    "canonical_source_ir_ref": request.canonical_source_ir_ref,
                    "canonical_source_hash": request.canonical_source_hash,
                },
                "schema_binding": {
                    "schema_decision_ref": decision.schema_decision_id,
                    "schema_decision_hash": decision.schema_decision_hash,
                    "candidate_ref": candidate.candidate_id,
                    "candidate_hash": candidate.candidate_hash,
                    "semantica_commit": semantica["semantica_commit"],
                    "ontology_candidate_ref": semantica.get(
                        "ontology_candidate_ref"
                    ),
                    "ontology_candidate_hash": semantica.get(
                        "ontology_candidate_hash"
                    ),
                    "operating_mode": decision.candidate_binding[
                        "operating_mode"
                    ],
                    "policy_ref": decision.candidate_binding.get("policy_ref"),
                    "rationale_ref": decision.candidate_binding.get(
                        "rationale_ref"
                    ),
                    **(
                        {
                            "admitted_relation_types": admitted_relation_types,
                        }
                        if decision.schema_version == "schema_decision.v3"
                        else {}
                    ),
                },
                "output": output,
                "evidence_bindings": evidence_bindings,
                "validation": {
                    "shacl_report_ref": semantica.get("shacl_report_ref"),
                    "extraction_validation_refs": [],
                    "warnings": (
                        ["UNRESOLVED_RELATION_SEMANTICS"]
                        if unresolved_relation_evidence
                        else []
                    ),
                    "errors": [],
                    **(
                        {
                            "unresolved_relation_evidence": (
                                unresolved_relation_evidence
                            )
                        }
                        if decision.schema_version == "schema_decision.v3"
                        else {}
                    ),
                },
                "provenance": {
                    "provider_refs": ["semantica", "semantier-extraction-plan"],
                    "model_refs": [],
                    "extraction_contract_hash": extraction_contract_hash,
                    "started_at": started_at,
                    "completed_at": completed_at,
                },
            }
        )
        return self._persist(result).as_dict()

    def _add_relation_occurrence_surface(
        self,
        output: dict[str, list[dict[str, Any]]],
        admitted: dict[str, dict[str, Any]],
        *,
        evidence_bindings: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Create the strict representation-neutral surface from admitted output."""

        nodes = {
            str(node.get("node_id") or ""): str(node.get("type_id") or "")
            for node in output.get("nodes") or []
        }
        raw_frames = [dict(frame) for frame in output.get("frames") or []]
        frames = {
            str(frame.get("frame_id") or ""): dict(frame)
            for frame in raw_frames
        }
        retained_edges: list[dict[str, Any]] = []
        occurrences: list[ExtractedRelationOccurrenceV1] = []
        occurrence_source_refs: dict[str, list[str]] = {}
        unresolved: list[dict[str, Any]] = []
        for edge in output.get("edges") or []:
            relation_type_id = str(edge.get("relation_type_id") or "")
            binding = admitted.get(relation_type_id)
            if binding is None:
                raise ValueError(
                    "new relation occurrence is not admitted by SchemaDecision.v3"
                )
            source_ref, target_ref = self._semantic_occurrence_endpoints(
                edge=edge, binding=binding
            )
            self._validate_occurrence_endpoints(
                edge=edge,
                binding=binding,
                node_types=nodes,
            )
            kind = str(binding["representation_kind"])
            if kind == "derived_projection":
                edge_evidence = sorted(set(map(str, edge.get("evidence_refs") or [])))
                projection_sources = sorted(
                    set(map(str, edge.get("projection_source_refs") or [])),
                    key=lambda ref: ref.encode("utf-8"),
                )
                if not projection_sources:
                    unresolved.append(
                        {
                            "relation_type_id": relation_type_id,
                            "reason": "missing_governed_projection_sources",
                            "evidence_refs": sorted(edge_evidence),
                        }
                    )
                    continue
                if not self._valid_projection_sources(
                    projection_sources,
                    occurrences=occurrences,
                    frames=frames,
                ):
                    unresolved.append(
                        {
                            "relation_type_id": relation_type_id,
                            "reason": "invalid_governed_projection_sources",
                            "evidence_refs": edge_evidence,
                        }
                    )
                    continue
                projection_shape = str(binding["projection_shape"])
                frame_ref = edge.get("frame_ref") if projection_shape == "frame" else None
                derived_occurrence = create_extracted_relation_occurrence(
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    relation_type_id=relation_type_id,
                    representation_kind="derived_projection",
                    projection_shape=projection_shape,
                    source_node_id=(
                        source_ref if projection_shape == "binary" else None
                    ),
                    target_node_id=(
                        target_ref if projection_shape == "binary" else None
                    ),
                    frame_ref=frame_ref,
                    projection_source_refs=projection_sources,
                    evidence_refs=binding["evidence_refs"],
                )
                occurrences.append(derived_occurrence)
                occurrence_source_refs[derived_occurrence.occurrence_ref] = sorted(
                    edge_evidence
                )
                if projection_shape == "binary":
                    retained_edges.append(edge)
                continue
            frame_ref = None
            occurrence_source = source_ref
            occurrence_target = target_ref
            if kind == "frame_role":
                edge_evidence = set(map(str, edge.get("evidence_refs") or []))
                matching_frames = [
                    str(frame.get("frame_id") or "")
                    for frame in raw_frames
                    if self._validate_exact_frame_relation(
                        frame=frame,
                        relation_type_id=relation_type_id,
                        semantic_source=source_ref,
                        semantic_target=target_ref,
                        node_types=nodes,
                        binding=binding,
                        edge_evidence=edge_evidence,
                    )
                ]
                if len(matching_frames) != 1:
                    unresolved.append(
                        {
                            "relation_type_id": relation_type_id,
                            "reason": "missing_exact_governed_frame",
                            "evidence_refs": sorted(edge_evidence),
                        }
                    )
                    continue
                frame_ref = matching_frames[0]
                occurrence_source = None
                occurrence_target = None
            else:
                retained_edges.append(edge)
            occurrence = create_extracted_relation_occurrence(
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                relation_type_id=relation_type_id,
                representation_kind=kind,
                projection_shape=str(binding["projection_shape"]),
                source_node_id=occurrence_source,
                target_node_id=occurrence_target,
                frame_ref=frame_ref,
                projection_source_refs=[],
                evidence_refs=binding["evidence_refs"],
            )
            occurrences.append(occurrence)
            occurrence_source_refs[occurrence.occurrence_ref] = sorted(
                set(map(str, edge.get("evidence_refs") or []))
            )
        refs = [item.occurrence_ref for item in occurrences]
        if len(refs) != len(set(refs)):
            raise ValueError("new extraction produced duplicate relation occurrences")
        output["edges"] = retained_edges
        output["relation_occurrences"] = [
            item.model_dump(mode="json", by_alias=True)
            for item in sorted(
                occurrences, key=lambda item: item.occurrence_ref.encode("utf-8")
            )
        ]
        if not self._valid_relation_occurrence_surface(output, admitted):
            raise ValueError("new extraction produced invalid relation occurrence surface")
        if evidence_bindings is not None:
            live_output_refs = {
                str(item.get(id_field) or "")
                for output_key, id_field in (
                    ("nodes", "node_id"),
                    ("edges", "edge_id"),
                    ("events", "event_id"),
                    ("frames", "frame_id"),
                )
                for item in (output.get(output_key) or [])
            }
            evidence_bindings[:] = [
                item
                for item in evidence_bindings
                if str(item.get("output_ref") or "") in live_output_refs
            ]
            evidence_bindings.extend(
                {
                    "output_ref": occurrence_ref,
                    "evidence_refs": source_refs,
                    "source_block_refs": source_refs,
                }
                for occurrence_ref, source_refs in sorted(
                    occurrence_source_refs.items(),
                    key=lambda item: item[0].encode("utf-8"),
                )
            )
        return unresolved

    @staticmethod
    def _semantic_occurrence_endpoints(
        *, edge: dict[str, Any], binding: dict[str, Any]
    ) -> tuple[str, str]:
        source = str(edge.get("source_node_id") or "")
        target = str(edge.get("target_node_id") or "")
        direction = str(edge.get("direction") or "")
        if direction == "target_to_source":
            return target, source
        if direction == "undirected" and source.encode("utf-8") > target.encode("utf-8"):
            return target, source
        return source, target

    @staticmethod
    def _validate_exact_frame_relation(
        *,
        frame: dict[str, Any],
        relation_type_id: str,
        semantic_source: str,
        semantic_target: str,
        node_types: dict[str, str],
        binding: dict[str, Any],
        edge_evidence: set[str],
    ) -> bool:
        slots = frame.get("slots")
        if not isinstance(slots, dict) or len(slots) != len(set(slots.values())):
            return False
        expected_frame_id = "extract-frame-" + content_hash(
            {"type_id": str(frame.get("type_id") or ""), "slots": slots,
             "source_frame_node_ref": str(frame.get("source_frame_node_ref") or "")}
        )[:24]
        if frame.get("frame_id") != expected_frame_id:
            return False
        if frame.get("source_frame_node_ref") != semantic_source:
            return False
        participants = frame.get("relation_participants")
        if not isinstance(participants, dict) or set(participants) != {relation_type_id}:
            return False
        participant = participants[relation_type_id]
        if not isinstance(participant, dict) or set(participant) != {
            "source_node_id", "source_role_refs", "target_node_id",
            "target_role_refs", "evidence_refs",
        }:
            return False
        source_type = node_types.get(semantic_source)
        target_type = node_types.get(semantic_target)
        if not source_type or not target_type:
            return False
        if binding.get("source_type_refs") and source_type not in tuple(
            binding["source_type_refs"]
        ):
            return False
        if binding.get("target_type_refs") and target_type not in tuple(
            binding["target_type_refs"]
        ):
            return False
        if str(frame.get("type_id") or "") != source_type:
            return False
        if semantic_target not in set(map(str, slots.values())):
            if participant["target_node_id"] != semantic_target:
                return False
        if participant != {
            "source_node_id": semantic_source,
            "source_role_refs": list(binding.get("source_role_refs") or []),
            "target_node_id": semantic_target,
            "target_role_refs": list(binding.get("target_role_refs") or []),
            "evidence_refs": sorted(edge_evidence),
        }:
            return False
        if set(map(str, frame.get("evidence_refs") or [])) != edge_evidence:
            return False
        return frame.get("authority_state") == "candidate"

    @staticmethod
    def _valid_projection_sources(
        refs: list[str] | tuple[str, ...],
        *,
        occurrences: list[ExtractedRelationOccurrenceV1],
        frames: dict[str, dict[str, Any]],
        self_ref: str | None = None,
    ) -> bool:
        occurrences_by_ref = {item.occurrence_ref: item for item in occurrences}
        for ref in refs:
            if ref == self_ref:
                return False
            if ref in occurrences_by_ref:
                continue
            frame = frames.get(ref)
            if frame is None or frame.get("authority_state") != "candidate":
                return False
            slots = frame.get("slots")
            if not isinstance(slots, dict) or ref != "extract-frame-" + content_hash(
                {"type_id": str(frame.get("type_id") or ""), "slots": slots,
                 "source_frame_node_ref": str(frame.get("source_frame_node_ref") or "")}
            )[:24]:
                return False
        def cyclic(ref: str, path: set[str]) -> bool:
            if ref == self_ref or ref in path:
                return True
            occurrence = occurrences_by_ref.get(ref)
            if occurrence is None:
                return False
            return any(
                cyclic(child, path | {ref})
                for child in occurrence.projection_source_refs
            )
        return not any(cyclic(ref, set()) for ref in refs)

    @staticmethod
    def _validate_occurrence_endpoints(
        *,
        edge: dict[str, Any],
        binding: dict[str, Any],
        node_types: dict[str, str],
    ) -> None:
        source_ref = str(edge.get("source_node_id") or "")
        target_ref = str(edge.get("target_node_id") or "")
        if source_ref not in node_types or target_ref not in node_types:
            raise ValueError("relation occurrence endpoint is absent from ExtractResult")
        direction = str(edge.get("direction") or "")
        if binding["directed"]:
            if direction not in {"source_to_target", "target_to_source"}:
                raise ValueError("relation occurrence direction is invalid")
        elif (
            direction != "undirected"
            or source_ref.encode("utf-8") > target_ref.encode("utf-8")
        ):
            raise ValueError("relation occurrence direction is invalid")
        semantic_source, semantic_target = source_ref, target_ref
        if direction == "target_to_source":
            semantic_source, semantic_target = semantic_target, semantic_source
        source_types = tuple(binding.get("source_type_refs") or ())
        target_types = tuple(binding.get("target_type_refs") or ())
        if source_types and node_types[semantic_source] not in source_types:
            raise ValueError("relation occurrence source type is invalid")
        if target_types and node_types[semantic_target] not in target_types:
            raise ValueError("relation occurrence target type is invalid")

    @staticmethod
    def _valid_relation_occurrence_surface(
        output: dict[str, Any], admitted: dict[str, dict[str, Any]]
    ) -> bool:
        raw = output.get("relation_occurrences")
        if not isinstance(raw, list):
            return False
        try:
            occurrences = [
                ExtractedRelationOccurrenceV1.model_validate(item) for item in raw
            ]
        except ValueError:
            return False
        refs = [item.occurrence_ref for item in occurrences]
        if len(refs) != len(set(refs)):
            return False
        if not all(
            item.relation_type_id in admitted
            and item.representation_kind
            == admitted[item.relation_type_id]["representation_kind"]
            for item in occurrences
        ):
            return False
        edges = list(output.get("edges") or [])
        frames = {
            str(frame.get("frame_id") or ""): dict(frame)
            for frame in (output.get("frames") or [])
        }
        if len(frames) != len(output.get("frames") or []):
            return False
        nodes = {
            str(node.get("node_id") or ""): str(node.get("type_id") or "")
            for node in (output.get("nodes") or [])
        }
        for occurrence in occurrences:
            binding = admitted[occurrence.relation_type_id]
            try:
                exact_occurrence = create_extracted_relation_occurrence(
                    organization_id=occurrence.organization_id,
                    workspace_id=occurrence.workspace_id,
                    relation_type_id=occurrence.relation_type_id,
                    representation_kind=occurrence.representation_kind,
                    projection_shape=str(binding["projection_shape"]),
                    source_node_id=occurrence.source_node_id,
                    target_node_id=occurrence.target_node_id,
                    frame_ref=occurrence.frame_ref,
                    projection_source_refs=occurrence.projection_source_refs,
                    evidence_refs=binding["evidence_refs"],
                )
            except ValueError:
                return False
            if exact_occurrence != occurrence:
                return False
            if occurrence.representation_kind == "binary_edge":
                if str(binding["projection_shape"]) != "binary":
                    return False
                matches = []
                for edge in edges:
                    if edge.get("relation_type_id") != occurrence.relation_type_id:
                        continue
                    try:
                        semantic_endpoints = GovernedExtractService._semantic_occurrence_endpoints(
                            edge=edge, binding=binding
                        )
                        GovernedExtractService._validate_occurrence_endpoints(
                            edge=edge, binding=binding, node_types=nodes
                        )
                    except ValueError:
                        return False
                    if semantic_endpoints == (
                        occurrence.source_node_id,
                        occurrence.target_node_id,
                    ):
                        matches.append(edge)
                if len(matches) != 1:
                    return False
                if (
                    occurrence.source_node_id not in nodes
                    or occurrence.target_node_id not in nodes
                ):
                    return False
                if binding.get("source_type_refs") and nodes[
                    occurrence.source_node_id
                ] not in tuple(binding["source_type_refs"]):
                    return False
                if binding.get("target_type_refs") and nodes[
                    occurrence.target_node_id
                ] not in tuple(binding["target_type_refs"]):
                    return False
            elif occurrence.representation_kind == "frame_role":
                if str(binding["projection_shape"]) != "frame":
                    return False
                frame = frames.get(str(occurrence.frame_ref or ""))
                if frame is None:
                    return False
                # A frame occurrence is authoritative without a flattened edge;
                # its own schema must nevertheless remain structurally exact.
                slots = frame.get("slots")
                source_types = set(map(str, binding.get("source_type_refs") or []))
                if (
                    not isinstance(slots, dict)
                    or not slots
                    or len(slots) != len(set(map(str, slots.values())))
                    or frame.get("authority_state") != "candidate"
                    or not str(frame.get("type_id") or "")
                    or frame.get("frame_id")
                    != "extract-frame-"
                    + content_hash(
                        {"type_id": str(frame.get("type_id") or ""), "slots": slots,
                         "source_frame_node_ref": str(frame.get("source_frame_node_ref") or "")}
                    )[:24]
                    or (source_types and str(frame.get("type_id")) not in source_types)
                    or any(
                        node_ref not in nodes or nodes[node_ref] != slot_type
                        for slot_type, node_ref in slots.items()
                    )
                ):
                    return False
                participant = dict(frame.get("relation_participants") or {}).get(
                    occurrence.relation_type_id
                )
                if not isinstance(participant, dict):
                    return False
                if not GovernedExtractService._validate_exact_frame_relation(
                    frame=frame,
                    relation_type_id=occurrence.relation_type_id,
                    semantic_source=str(participant.get("source_node_id") or ""),
                    semantic_target=str(participant.get("target_node_id") or ""),
                    node_types=nodes,
                    binding=binding,
                    edge_evidence=set(map(str, participant.get("evidence_refs") or [])),
                ):
                    return False
            else:
                if not GovernedExtractService._valid_projection_sources(
                    occurrence.projection_source_refs,
                    occurrences=occurrences,
                    frames=frames,
                    self_ref=occurrence.occurrence_ref,
                ):
                    return False
                if (
                    str(binding["projection_shape"]) == "binary"
                    and len(
                        [
                            edge
                            for edge in edges
                            if edge.get("relation_type_id")
                            == occurrence.relation_type_id
                            and edge.get("source_node_id")
                            == occurrence.source_node_id
                            and edge.get("target_node_id")
                            == occurrence.target_node_id
                        ]
                    )
                    != 1
                ):
                    return False
                if occurrence.source_node_id is not None and (
                    occurrence.source_node_id not in nodes
                    or occurrence.target_node_id not in nodes
                ):
                    return False
        binary_edges = [
            edge
            for edge in edges
            if admitted.get(str(edge.get("relation_type_id") or ""), {}).get(
                "representation_kind"
            )
            in {"binary_edge", "derived_projection"}
            and admitted[str(edge.get("relation_type_id") or "")][
                "projection_shape"
            ]
            == "binary"
        ]
        return len(binary_edges) == len(
            [
                occurrence
                for occurrence in occurrences
                if occurrence.representation_kind == "binary_edge"
                or (
                    occurrence.representation_kind == "derived_projection"
                    and admitted[occurrence.relation_type_id]["projection_shape"]
                    == "binary"
                )
            ]
        )

    def read_result(self, extract_run_id: str) -> ExtractResult:
        """Read the immutable result inside the caller-owned governed workspace."""
        path = (
            self.workspace_dir
            / "runs"
            / extract_run_id
            / "artifacts"
            / "extract_result.v1.json"
        ).resolve()
        runs_root = (
            self.workspace_dir / "runs"
        ).resolve()
        if not path.is_relative_to(runs_root) or not path.is_file():
            raise ValueError("EXTRACT_RESULT_NOT_FOUND")
        result = ExtractResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if result.extract_run_id != extract_run_id:
            raise ValueError("EXTRACT_RESULT_REF_MISMATCH")
        return result

    def evaluate_grounding_gate(self, extract_run_id: str) -> dict[str, Any]:
        """Evaluate Gate 2 from persisted Extract evidence, never client claims."""
        result = self.read_result(extract_run_id)
        taxonomy_service = self._taxonomy_service()
        decision_payload = self.schema_decisions.get(
            str(result.schema_binding["schema_decision_ref"]),
            taxonomy_service=taxonomy_service,
        )
        decision = (
            KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
                taxonomy_service, decision_payload
            )
            if decision_payload.get("schema_version") == "schema_decision.v3"
            else SchemaDecision.from_dict(decision_payload)
        )
        schema_compatible = (
            result.schema_binding.get("schema_decision_hash")
            == decision.schema_decision_hash
            and result.schema_binding.get("candidate_hash")
            == decision.candidate_binding.get("candidate_hash")
        )
        outputs = [
            item
            for kind in ("nodes", "edges", "events", "frames")
            for item in (result.output.get(kind) or [])
        ]
        binding_refs = {
            str(item.get("output_ref") or "") for item in result.evidence_bindings
        }
        evaluation = SchemaGatePolicy.default().evaluate_gate_two(
            candidate_structurally_valid=bool(outputs)
            and all(isinstance(item, dict) for item in outputs),
            evidence_complete=all(
                item.get("evidence_refs")
                and str(
                    item.get("node_id")
                    or item.get("edge_id")
                    or item.get("event_id")
                    or item.get("frame_id")
                    or ""
                )
                in binding_refs
                for item in outputs
            ),
            schema_compatible=schema_compatible,
            authority_state=(
                "candidate"
                if all(item.get("authority_state") == "candidate" for item in outputs)
                else "authoritative"
            ),
            confidence=(
                1.0
                if not result.validation.get("errors")
                and not result.validation.get("warnings")
                else 0.0
            ),
            evidence_refs=tuple(
                sorted(
                    {
                        str(ref)
                        for item in result.evidence_bindings
                        for ref in (item.get("evidence_refs") or [])
                    }
                )
            ),
        )
        return {
            **evaluation.__dict__,
            "extract_run_ref": result.extract_run_id,
            "extract_result_hash": result.extract_result_hash,
            "schema_decision_ref": decision.schema_decision_id,
            "schema_decision_hash": decision.schema_decision_hash,
        }

    def _artifact_path(self, artifact_ref: str) -> Path:
        root = (
            self.workspace_dir / "artifacts"
        ).resolve()
        relative = artifact_ref.removeprefix("artifacts/")
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise ValueError("ARTIFACT_SCOPE_MISMATCH")
        if not target.is_file():
            raise ValueError("ARTIFACT_NOT_FOUND")
        return target

    @staticmethod
    def _execute_plan(
        *,
        elements: list[dict[str, Any]],
        decision: SchemaDecision,
        admitted_relation_directions: dict[str, bool] | None = None,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        node_type_by_role = {
            str(item.get("derived_from_role")): str(item["type_id"])
            for item in decision.approved_schema.get("node_types", [])
            if item.get("derived_from_role")
        }
        nodes: list[dict[str, Any]] = []
        bindings: list[dict[str, Any]] = []
        seen: set[str] = set()
        node_context: dict[str, dict[str, Any]] = {}
        element_order = {
            str((item.get("anchor") or {}).get("anchor_id") or ""): index
            for index, item in enumerate(elements)
        }
        row_nodes: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        grouped_type_nodes: dict[
            tuple[Any, ...], dict[str, list[dict[str, Any]]]
        ] = {}
        for plan in decision.extraction_plan:
            role_id = str(plan.get("role_id") or "")
            node_type = node_type_by_role.get(role_id)
            if not node_type:
                continue
            selector = dict(plan.get("selector") or {})
            column_specs = list(
                selector.get("columns") or selector.get("table_columns") or []
            )
            columns = {
                (item.get("table_index"), item.get("column_index"))
                for item in column_specs
            }
            column_header_refs = {
                str(item.get("evidence_ref") or "") for item in column_specs
            }
            paragraph_specs = list(
                selector.get("matches") or selector.get("paragraph_patterns") or []
            )
            paragraph_refs = {
                str(item.get("anchor_id"))
                for item in paragraph_specs
            }
            paragraph_values = {
                str(item.get("anchor_id")): str(item.get("value") or "").strip()
                for item in paragraph_specs
            }
            for element in elements:
                anchor = dict(element.get("anchor") or {})
                evidence_ref = str(anchor.get("anchor_id") or "")
                table_selected = (
                    bool(columns)
                    and evidence_ref not in column_header_refs
                    and (anchor.get("table_index"), anchor.get("cell_index"))
                    in columns
                )
                paragraph_selected = evidence_ref in paragraph_refs
                selected = (
                    table_selected or paragraph_selected
                )
                text = (
                    paragraph_values.get(evidence_ref)
                    if paragraph_selected
                    else str(element.get("text") or "").strip()
                )
                if not selected or not text:
                    continue
                node_id = f"extract-node-{content_hash({'role_id': role_id, 'type': node_type, 'text': text, 'evidence_ref': evidence_ref})[:24]}"
                if node_id in seen:
                    continue
                seen.add(node_id)
                node = {
                    "node_id": node_id,
                    "type_id": node_type,
                    "value": text,
                    "role_id": role_id,
                    "evidence_refs": [evidence_ref],
                    "authority_state": "candidate",
                }
                nodes.append(node)
                node_context[node_id] = {
                    "source_order": element_order[evidence_ref],
                    "source_anchor_ref": evidence_ref,
                }
                table_index = anchor.get("table_index")
                row_index = anchor.get("row_index")
                if isinstance(table_index, int) and isinstance(row_index, int):
                    row_nodes.setdefault((table_index, row_index), {})[role_id] = {
                        "node_id": node_id,
                        "evidence_ref": evidence_ref,
                    }
                    group_ref: tuple[Any, ...] = (
                        "table_row",
                        table_index,
                        row_index,
                    )
                else:
                    group_ref = ("source_anchor", evidence_ref)
                grouped_type_nodes.setdefault(group_ref, {}).setdefault(
                    node_type, []
                ).append(node)
                bindings.append(
                    {
                        "output_ref": node_id,
                        "evidence_refs": [evidence_ref],
                        "source_block_refs": [
                            str(element.get("source_block_id") or evidence_ref)
                        ],
                    }
                )
        return GovernedExtractService._execute_graph_intent_plan(
            decision=decision,
            nodes=nodes,
            bindings=bindings,
            grouped_type_nodes=grouped_type_nodes,
            elements=elements,
            node_context=node_context,
            admitted_relation_directions=admitted_relation_directions,
        )

    @staticmethod
    def _execute_graph_intent_plan(
        *,
        decision: SchemaDecision,
        nodes: list[dict[str, Any]],
        bindings: list[dict[str, Any]],
        grouped_type_nodes: dict[
            tuple[Any, ...], dict[str, list[dict[str, Any]]]
        ],
        elements: list[dict[str, Any]],
        node_context: dict[str, dict[str, Any]],
        admitted_relation_directions: dict[str, bool] | None,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Execute only source-grounded topology frozen in a GraphIntent decision."""

        element_by_ref = {
            str((item.get("anchor") or {}).get("anchor_id") or ""): item
            for item in elements
        }
        global_type_nodes: dict[str, list[dict[str, Any]]] = {}
        structural_nodes_by_type: dict[str, dict[str, dict[str, Any]]] = {}
        for instances_by_type in grouped_type_nodes.values():
            for type_id, type_nodes in instances_by_type.items():
                for node in type_nodes:
                    structural_nodes_by_type.setdefault(type_id, {})[
                        str(node["node_id"])
                    ] = node
        for type_id, type_nodes in structural_nodes_by_type.items():
            if len(type_nodes) == 1:
                global_type_nodes[type_id] = list(type_nodes.values())
        representation_groups: dict[
            tuple[Any, ...], dict[str, list[dict[str, Any]]]
        ] = {}
        events: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []

        def bind_output(output_ref: str, evidence_refs: list[str]) -> None:
            bindings.append(
                {
                    "output_ref": output_ref,
                    "evidence_refs": evidence_refs,
                    "source_block_refs": evidence_refs,
                }
            )

        def add_grounded_node(plan: dict[str, Any]) -> dict[str, Any] | None:
            source_refs = list(dict.fromkeys(map(str, plan.get("source_block_refs") or [])))
            grounded_elements = [
                element_by_ref[ref] for ref in source_refs if ref in element_by_ref
            ]
            if len(grounded_elements) != len(source_refs) or not grounded_elements:
                return None
            values = [str(item.get("text") or "").strip() for item in grounded_elements]
            if any(not value for value in values):
                return None
            type_id = str(plan["target_ref"])
            node_id = "extract-node-" + content_hash(
                {
                    "type_id": type_id,
                    "grounding_kind": "accepted_claim_evidence",
                    "source_block_refs": source_refs,
                    "values": values,
                }
            )[:24]
            node = {
                "node_id": node_id,
                "type_id": type_id,
                "value": values[0] if len(values) == 1 else values,
                "grounding_kind": "accepted_claim_evidence",
                "accepted_evidence_refs": list(
                    map(str, plan.get("accepted_evidence_refs") or [])
                ),
                "evidence_refs": source_refs,
                "authority_state": "candidate",
            }
            if not any(item["node_id"] == node_id for item in nodes):
                nodes.append(node)
                bind_output(node_id, source_refs)
            global_type_nodes.setdefault(type_id, []).append(node)
            return node

        for raw_plan in decision.extraction_plan:
            plan = dict(raw_plan)
            if plan.get("strategy") == "intent_evidence_node" or (
                plan.get("strategy") == "intent_slot"
                and plan.get("grounding_kind") == "accepted_claim_evidence"
            ):
                add_grounded_node(plan)

        frame_type_by_id = {
            str(item["type_id"]): dict(item)
            for item in decision.approved_schema.get("frame_types", [])
        }
        for raw_plan in decision.extraction_plan:
            plan = dict(raw_plan)
            if plan.get("strategy") != "intent_frame_constructor":
                continue
            frame_type_id = str(plan["target_ref"])
            slot_types = list(map(str, plan.get("required_slot_types") or []))
            slot_nodes = {
                type_id: global_type_nodes[type_id][0]
                for type_id in slot_types
                if global_type_nodes.get(type_id)
            }
            if set(slot_nodes) != set(slot_types):
                continue
            slots = {
                type_id: slot_nodes[type_id]["node_id"] for type_id in slot_types
            }
            evidence_refs = sorted(
                {
                    str(ref)
                    for node in slot_nodes.values()
                    for ref in node.get("evidence_refs") or []
                }
            )
            frame_node_id = "extract-node-" + content_hash(
                {
                    "type_id": frame_type_id,
                    "constructor": "intent_frame_constructor",
                    "slots": slots,
                }
            )[:24]
            frame_node = {
                "node_id": frame_node_id,
                "type_id": frame_type_id,
                "value": {
                    type_id: slot_nodes[type_id]["value"] for type_id in slot_types
                },
                "grounding_kind": "intent_frame_constructor",
                "evidence_refs": evidence_refs,
                "authority_state": "candidate",
            }
            nodes.append(frame_node)
            bind_output(frame_node_id, evidence_refs)
            global_type_nodes.setdefault(frame_type_id, []).append(frame_node)
            group_ref = ("intent_frame", frame_node_id)
            representation_groups[group_ref] = {
                **{type_id: [node] for type_id, node in slot_nodes.items()},
                frame_type_id: [frame_node],
            }
            frame_type = frame_type_by_id.get(frame_type_id, {})
            frame_id = "extract-frame-" + content_hash(
                {"type_id": frame_type_id, "slots": slots,
                 "source_frame_node_ref": frame_node_id}
            )[:24]
            frames.append(
                {
                    "frame_id": frame_id,
                    "type_id": frame_type_id,
                    "slots": slots,
                    "source_frame_node_ref": frame_node_id,
                    "semantica_representation": frame_type.get(
                        "semantica_representation"
                    ),
                    "runtime_adapter": frame_type.get("runtime_adapter"),
                    "evidence_refs": evidence_refs,
                    "authority_state": "candidate",
                }
            )
            bind_output(frame_id, evidence_refs)

        for raw_plan in decision.extraction_plan:
            plan = dict(raw_plan)
            if plan.get("strategy") != "intent_event_constructor":
                continue
            event_type_id = str(plan["target_ref"])
            required_types = list(
                map(str, plan.get("required_participant_types") or [])
            )
            rule = dict(plan.get("event_extraction_rule") or {})
            role_type_map = {
                str(role): str(type_id)
                for role, type_id in dict(plan.get("role_type_map") or {}).items()
            }
            boundary = str(rule.get("boundary_role") or "")
            required_roles = list(map(str, rule.get("required_roles") or []))
            inherited_roles = list(
                map(str, rule.get("inherit_context_roles") or [])
            )
            attached_roles = list(
                map(str, rule.get("attach_following_roles") or [])
            )
            sample_limit = int(rule.get("sample_limit") or 0)
            nodes_by_anchor: dict[str, list[dict[str, Any]]] = {}
            for node in nodes:
                role_id = str(node.get("role_id") or "")
                context = node_context.get(str(node["node_id"]))
                if role_id not in role_type_map or context is None:
                    continue
                nodes_by_anchor.setdefault(
                    str(context["source_anchor_ref"]), []
                ).append(node)
            ordered_anchors = sorted(
                nodes_by_anchor,
                key=lambda ref: min(
                    int(node_context[str(node["node_id"])]["source_order"])
                    for node in nodes_by_anchor[ref]
                ),
            )
            context_roles: dict[str, list[dict[str, Any]]] = {}
            pending: dict[str, list[dict[str, Any]]] | None = None
            pending_anchor: str | None = None
            produced = 0

            def finalize_event() -> None:
                nonlocal pending, pending_anchor, produced
                if pending is None or pending_anchor is None or produced >= sample_limit:
                    pending = None
                    pending_anchor = None
                    return
                participants_by_type: dict[str, dict[str, Any]] = {}
                for role_id, role_nodes in pending.items():
                    type_id = role_type_map.get(role_id)
                    if type_id and role_nodes and type_id not in participants_by_type:
                        participants_by_type[type_id] = role_nodes[0]
                if not set(required_roles) <= set(pending) or not set(
                    required_types
                ) <= set(participants_by_type):
                    pending = None
                    pending_anchor = None
                    return
                participants = {
                    type_id: participants_by_type[type_id]["node_id"]
                    for type_id in required_types
                }
                evidence_refs = sorted(
                    {
                        str(ref)
                        for type_id in required_types
                        for ref in participants_by_type[type_id].get(
                            "evidence_refs", []
                        )
                    }
                )
                event_node_id = "extract-node-" + content_hash(
                    {
                        "type_id": event_type_id,
                        "constructor": "intent_event_constructor",
                        "anchor": pending_anchor,
                        "participants": participants,
                    }
                )[:24]
                event_node = {
                    "node_id": event_node_id,
                    "type_id": event_type_id,
                    "value": {
                        type_id: participants_by_type[type_id]["value"]
                        for type_id in required_types
                    },
                    "grounding_kind": "intent_event_constructor",
                    "evidence_refs": evidence_refs,
                    "authority_state": "candidate",
                }
                nodes.append(event_node)
                bind_output(event_node_id, evidence_refs)
                group_ref = ("intent_event", pending_anchor)
                representation_groups[group_ref] = {
                    **{
                        type_id: [participants_by_type[type_id]]
                        for type_id in required_types
                    },
                    event_type_id: [event_node],
                }
                event_id = "extract-event-" + content_hash(
                    {
                        "type_id": event_type_id,
                        "anchor": pending_anchor,
                        "participants": participants,
                    }
                )[:24]
                events.append(
                    {
                        "event_id": event_id,
                        "type_id": event_type_id,
                        "participants": participants,
                        "evidence_refs": evidence_refs,
                        "authority_state": "candidate",
                    }
                )
                bind_output(event_id, evidence_refs)
                produced += 1
                pending = None
                pending_anchor = None

            for anchor_ref in ordered_anchors:
                anchor_nodes = nodes_by_anchor[anchor_ref]
                roles_at_anchor = {
                    str(node["role_id"]) for node in anchor_nodes
                }
                if boundary in roles_at_anchor:
                    finalize_event()
                    for role_id in inherited_roles:
                        values = [
                            node
                            for node in anchor_nodes
                            if node.get("role_id") == role_id
                        ]
                        if values:
                            context_roles[role_id] = values
                    pending = {
                        role_id: list(role_nodes)
                        for role_id, role_nodes in context_roles.items()
                    }
                    for node in anchor_nodes:
                        role_id = str(node["role_id"])
                        if role_id not in inherited_roles:
                            pending.setdefault(role_id, []).append(node)
                    pending_anchor = anchor_ref
                    continue
                for role_id in inherited_roles:
                    values = [
                        node
                        for node in anchor_nodes
                        if node.get("role_id") == role_id
                    ]
                    if values:
                        context_roles[role_id] = values
                if pending is not None:
                    for role_id in attached_roles:
                        values = [
                            node
                            for node in anchor_nodes
                            if node.get("role_id") == role_id
                        ]
                        pending.setdefault(role_id, []).extend(values)
            finalize_event()

        all_relationship_plans = [
            dict(item)
            for item in decision.extraction_plan
            if item.get("strategy") == "intent_relation"
        ]
        relation_roles = {
            str(item["relation_type_candidate"]["relation_type_id"]): (
                list(item["relation_type_candidate"].get("source_role_refs") or []),
                list(item["relation_type_candidate"].get("target_role_refs") or []),
            )
            for item in decision.approved_schema.get("relation_types", [])
        }
        unresolved_relation_evidence = []
        if admitted_relation_directions is None:
            relationship_plans = all_relationship_plans
        else:
            relationship_plans = []
            for plan in all_relationship_plans:
                relation_type_id = str(plan.get("target_ref") or "")
                if relation_type_id in admitted_relation_directions:
                    relationship_plans.append(plan)
                    continue
                unresolved_relation_evidence.append(
                    {
                        "relation_type_id": relation_type_id,
                        "reason": "not_admitted_by_schema_decision",
                        "evidence_refs": sorted(
                            set(map(str, plan.get("evidence_refs") or []))
                        ),
                    }
                )
            unresolved_relation_evidence.sort(
                key=lambda item: (
                    item["relation_type_id"].encode("utf-8"),
                    tuple(ref.encode("utf-8") for ref in item["evidence_refs"]),
                )
            )
        edges: list[dict[str, Any]] = []
        edge_keys: set[tuple[str, str, str]] = set()
        all_groups = {**grouped_type_nodes, **representation_groups}
        for group_ref, instances_by_type in sorted(
            all_groups.items(), key=lambda item: str(item[0])
        ):
            for relation_plan in relationship_plans:
                direction = str(relation_plan.get("direction") or "")
                relation_type_id = str(relation_plan["target_ref"])
                admitted_directed = (
                    admitted_relation_directions.get(relation_type_id)
                    if admitted_relation_directions is not None
                    else True
                )
                if admitted_directed and direction not in {
                    "source_to_target",
                    "target_to_source",
                }:
                    raise ValueError(
                        "GraphIntent primary relationship direction is invalid"
                    )
                if not admitted_directed and direction != "undirected":
                    raise ValueError(
                        "GraphIntent primary relationship direction is invalid"
                    )
                for declared_source in relation_plan.get("source_types") or []:
                    for declared_target in relation_plan.get("target_types") or []:
                        source_type = str(declared_source)
                        target_type = str(declared_target)
                        source_nodes = instances_by_type.get(
                            source_type
                        ) or global_type_nodes.get(source_type, [])
                        target_nodes = instances_by_type.get(
                            target_type
                        ) or global_type_nodes.get(target_type, [])
                        for source in source_nodes:
                            for target in target_nodes:
                                actual_source, actual_target = source, target
                                if direction == "target_to_source":
                                    actual_source, actual_target = target, source
                                elif direction == "undirected":
                                    actual_source, actual_target = sorted(
                                        (source, target),
                                        key=lambda node: str(node["node_id"]).encode(
                                            "utf-8"
                                        ),
                                    )
                                key = (
                                    str(relation_plan["target_ref"]),
                                    str(actual_source["node_id"]),
                                    str(actual_target["node_id"]),
                                )
                                if key in edge_keys:
                                    continue
                                edge_keys.add(key)
                                evidence_refs = sorted(
                                    {
                                        *map(str, source.get("evidence_refs") or []),
                                        *map(str, target.get("evidence_refs") or []),
                                    }
                                )
                                edge_id = "extract-edge-" + content_hash(
                                    {
                                        "group_ref": group_ref,
                                        "type_id": relation_plan["target_ref"],
                                        "source": actual_source["node_id"],
                                        "target": actual_target["node_id"],
                                        "direction": direction,
                                    }
                                )[:24]
                                edges.append(
                                    {
                                        "edge_id": edge_id,
                                        **(
                                            {
                                                "relation_type_id": str(
                                                    relation_plan["target_ref"]
                                                )
                                            }
                                            if decision.schema_version
                                            == "schema_decision.v3"
                                            else {
                                                "type_id": str(
                                                    relation_plan["target_ref"]
                                                )
                                            }
                                        ),
                                        "source_node_id": actual_source["node_id"],
                                        "target_node_id": actual_target["node_id"],
                                        **(
                                            {}
                                            if direction == "undirected"
                                            else {
                                                "declared_source_type": source_type,
                                                "declared_target_type": target_type,
                                            }
                                        ),
                                        "direction": direction,
                                        "evidence_refs": evidence_refs,
                                        "authority_state": "candidate",
                                    }
                                )
                                for frame in frames:
                                    if (
                                        frame.get("type_id") == source_type
                                        and frame.get("source_frame_node_ref")
                                        == source.get("node_id")
                                    ):
                                        source_roles, target_roles = relation_roles.get(
                                            relation_type_id, ([], [])
                                        )
                                        frame.setdefault("relation_participants", {})[
                                            relation_type_id
                                        ] = {
                                            "source_node_id": source["node_id"],
                                            "source_role_refs": source_roles,
                                            "target_node_id": target["node_id"],
                                            "target_role_refs": target_roles,
                                            "evidence_refs": evidence_refs,
                                        }
                                bind_output(edge_id, evidence_refs)
        return {
            "nodes": nodes,
            "edges": edges,
            "events": events,
            "frames": frames,
        }, bindings, unresolved_relation_evidence


    @staticmethod
    def _equivalent_extract_results(
        existing: dict[str, Any], proposed: dict[str, Any]
    ) -> bool:
        try:
            ExtractResult.from_dict(existing)
        except ValueError:
            return False

        def stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
            stable = dict(payload)
            stable.pop("extract_result_hash", None)
            provenance = dict(stable.get("provenance") or {})
            provenance.pop("started_at", None)
            provenance.pop("completed_at", None)
            stable["provenance"] = provenance
            return stable

        return stable_payload(existing) == stable_payload(proposed)

    def _persist(self, result: ExtractResult) -> ExtractResult:
        target = (
            self.workspace_dir
            / "runs"
            / result.extract_run_id
            / "artifacts"
            / "extract_result.v1.json"
        )
        published = publish_immutable_json(
            target,
            result.as_dict(),
            conflict_error="EXTRACT_RESULT_ARTIFACT_CONFLICT",
            equivalent=self._equivalent_extract_results,
        )
        return ExtractResult.from_dict(published)


__all__ = ["GovernedExtractService"]
