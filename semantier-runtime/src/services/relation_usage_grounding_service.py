"""Human Grounding enforcement for Decision-v3 admitted relation usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts.governed_extract import ExtractResult
from contracts.extracted_relation_occurrence import ExtractedRelationOccurrenceV1
from contracts.relation_usage_grounding import RelationUsageGroundingDecision
from contracts.schema_learning_feedback_v2 import SchemaLearningFeedbackV2
from eos.artifact_hashing import content_hash
from services.knowledge_relation_taxonomy_service import (
    KnowledgeRelationTaxonomyService,
)
from services.corpus_graph_schema_profile_registry import (
    CorpusGraphSchemaProfileRegistry,
)
from storage.human_grounding_store import HumanGroundingStore
from storage.profile_relation_type_proposal_store import (
    ProfileRelationTypeProposalStore,
)
from storage.schema_decision_store import SchemaDecisionStore
from storage.schema_learning_feedback_store import SchemaLearningFeedbackStore


_CORRECTION_FIELDS = {
    "relation_type_id",
    "source_node_id",
    "target_node_id",
    "direction",
}
_ROUTES = {
    "occurrence_error": "occurrence_supersession",
    "repeated_type_selection_error": "schema_successor_required",
    "missing_predicate": "profile_catalog_proposal_required",
}
_SCOPES = {
    "occurrence_error": "instance_local",
    "repeated_type_selection_error": "schema_understanding",
    "missing_predicate": "ontology_or_higher_layer",
    "family_definition_error": "ontology_or_higher_layer",
}


class RelationUsageGroundingService:
    """Validate candidate relation use and persist deterministic route evidence."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        db_path: str | Path,
        organization_id: str,
        workspace_id: str,
        graph_backend: Any = None,
        profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
    ) -> None:
        if not organization_id or not workspace_id:
            raise ValueError("organization_id and workspace_id are required")
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.workspace_dir = self.workspace_root / "workspaces" / workspace_id
        self.profile_registry = profile_registry or CorpusGraphSchemaProfileRegistry()
        self.profile_relation_type_proposal_store = (
            ProfileRelationTypeProposalStore(
                self.workspace_root,
                workspace_id=workspace_id,
                organization_id=organization_id,
            )
        )
        self.taxonomy = KnowledgeRelationTaxonomyService(
            repository_root=Path(__file__).resolve().parents[2],
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.profile_relation_type_proposal_store
            ),
        )
        self.decisions = SchemaDecisionStore(
            self.workspace_root, workspace_id=workspace_id
        )
        self.feedback = SchemaLearningFeedbackStore(
            self.workspace_root, workspace_id=workspace_id
        )
        self._authority = _create_relation_usage_grounding_authority(
            self.workspace_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.profile_relation_type_proposal_store
            ),
        )
        self.grounding = HumanGroundingStore(
            db_path,
            graph_backend=graph_backend,
            _relation_usage_grounding_authority=self._authority,
        )

    def review_relation_usage(
        self,
        *,
        extract_run_ref: str,
        finding_kind: str,
        subject_ref: str,
        subject_hash: str,
        correction: dict[str, Any] | None,
        reviewed_by: str,
        occurred_at: str,
    ) -> dict[str, Any]:
        result = self._load_extract_result(extract_run_ref)
        decision_payload = self.decisions.get(
            str(result.schema_binding.get("schema_decision_ref") or ""),
            taxonomy_service=self.taxonomy,
        )
        if decision_payload.get("schema_version") != "schema_decision.v3":
            raise ValueError("relation usage Grounding requires SchemaDecision.v3")
        decision = KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
            self.taxonomy, decision_payload
        )
        self._validate_ownership_and_pins(result=result, decision=decision)
        admitted = self._resolve_admitted_bindings(decision)
        expected_extract_bindings = [
            self._extract_binding(binding)
            for _, binding in sorted(admitted.items(), key=lambda item: item[0])
        ]
        if result.schema_binding.get("admitted_relation_types") != expected_extract_bindings:
            raise ValueError("extract admitted relation binding mismatch")
        self._validate_all_candidate_edges(result, admitted)

        subject_kind, subject, subject_relation_binding = self._resolve_subject(
            result=result,
            admitted=admitted,
            subject_ref=subject_ref,
            subject_hash=subject_hash,
        )
        if finding_kind == "missing_predicate":
            if subject_kind != "unresolved_relation_evidence":
                raise ValueError("missing predicate requires exact unresolved evidence")
        elif subject_kind != "candidate_relation_instance":
            raise ValueError("relation usage finding requires exact candidate occurrence")
        if finding_kind not in _SCOPES:
            raise ValueError("finding_kind is unsupported")

        required_route = _ROUTES.get(finding_kind)
        if finding_kind == "family_definition_error":
            if subject_relation_binding is None:
                raise ValueError("family definition error requires an admitted type")
            required_route = (
                "taxonomy_definition_successor_required"
                if subject_relation_binding["definition_scope"] == "canonical"
                else "profile_definition_successor_required"
            )

        corrected_relation_binding = None
        corrected_relation_instance = None
        if finding_kind == "occurrence_error":
            corrected_relation_binding, corrected_relation_instance = (
                self._correct_occurrence(
                    correction=correction,
                    original=subject,
                    original_hash=subject_hash,
                    admitted=admitted,
                    result=result,
                )
            )
        elif correction is not None:
            raise ValueError("systematic route cannot include an occurrence correction")

        command_material = {
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "extract_run_ref": result.extract_run_id,
            "extract_result_hash": result.extract_result_hash,
            "finding_kind": finding_kind,
            "required_route": required_route,
            "subject_ref": subject_ref,
            "subject_hash": subject_hash,
            "correction": correction,
            "reviewed_by": reviewed_by,
            "occurred_at": occurred_at,
        }
        command_ref = (
            "relation-usage-grounding-command:" + content_hash(command_material)
        )
        feedback = SchemaLearningFeedbackV2.create(
            {
                "source_binding": dict(decision.source_binding),
                "schema_decision_ref": decision.schema_decision_id,
                "candidate_ref": str(decision.candidate_binding["candidate_id"]),
                "extraction_run_ref": result.extract_run_id,
                "grounding_ref": command_ref,
                "feedback_scope": _SCOPES[finding_kind],
                "produced_by": "user_chat_clarification",
                "operating_mode": "assisted",
                "policy_ref": None,
                "rationale_ref": None,
                "decision_evidence_refs": tuple(
                    sorted(map(str, subject.get("evidence_refs") or []))
                ),
                "fallback_reason": None,
                "operation_ref": None,
                "before_state_ref": subject_hash,
                "after_state_ref": (
                    corrected_relation_instance.get("relation_instance_hash")
                    if corrected_relation_instance is not None
                    else None
                ),
                "attribution": {
                    "graph_intent": False,
                    "source_structure": False,
                    "ontology_induction": finding_kind
                    in {"missing_predicate", "family_definition_error"},
                    "synthesis": finding_kind == "repeated_type_selection_error",
                    "extraction_plan": finding_kind
                    == "repeated_type_selection_error",
                    "execution": finding_kind == "occurrence_error",
                },
                "findings": (
                    {
                        "finding_kind": finding_kind,
                        "required_route": required_route,
                        "subject_ref": subject_ref,
                        "subject_hash": subject_hash,
                    },
                ),
                "proposed_candidate_delta_ref": None,
                "created_at": occurred_at,
                "created_by": reviewed_by,
            }
        )
        grounding_decision = RelationUsageGroundingDecision.create(
            {
                "organization_id": self.organization_id,
                "workspace_id": self.workspace_id,
                "source_binding": dict(decision.source_binding),
                "extract_run_binding": {
                    "extract_run_ref": result.extract_run_id,
                    "extract_result_hash": result.extract_result_hash,
                },
                "schema_decision_binding": {
                    "schema_decision_ref": decision.schema_decision_id,
                    "schema_decision_hash": decision.schema_decision_hash,
                },
                "finding_kind": finding_kind,
                "required_route": required_route,
                "subject_binding": {
                    "subject_kind": subject_kind,
                    "subject_ref": subject_ref,
                    "subject_hash": subject_hash,
                },
                "subject_relation_binding": subject_relation_binding,
                "corrected_relation_binding": corrected_relation_binding,
                "original_relation_instance": (
                    dict(subject)
                    if subject_kind == "candidate_relation_instance"
                    else None
                ),
                "corrected_relation_instance": corrected_relation_instance,
                "schema_learning_feedback_ref": feedback.feedback_id,
                "reviewed_by": reviewed_by,
                "occurred_at": occurred_at,
            },
            authority=self._authority,
        )
        stored_feedback = self.feedback.put(feedback)
        stored_grounding = self.grounding.put_relation_usage_grounding(
            grounding_decision
        )
        return {
            "grounding_decision": stored_grounding,
            "schema_learning_feedback": stored_feedback,
            "successor_artifact": None,
        }

    def replay_grounding_decision(
        self, payload: dict[str, Any]
    ) -> RelationUsageGroundingDecision:
        """Rehydrate one decision through this service's governed authority."""

        return RelationUsageGroundingDecision.from_dict(
            payload, authority=self._authority
        )

    def _load_extract_result(self, extract_run_ref: str) -> ExtractResult:
        if not extract_run_ref.startswith("extract-run-"):
            raise ValueError("extract_run_ref is invalid")
        runs_root = (self.workspace_dir / "runs").resolve()
        path = (
            runs_root
            / extract_run_ref
            / "artifacts"
            / "extract_result.v1.json"
        ).resolve()
        if not path.is_relative_to(runs_root) or not path.is_file():
            raise ValueError("EXTRACT_RESULT_NOT_FOUND")
        result = ExtractResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if result.extract_run_id != extract_run_ref:
            raise ValueError("extract run identity mismatch")
        return result

    def _validate_ownership_and_pins(self, *, result: ExtractResult, decision: Any) -> None:
        source = decision.source_binding
        if source.get("organization_id") != self.organization_id:
            raise ValueError("grounding organization ownership mismatch")
        if source.get("workspace_id") != self.workspace_id:
            raise ValueError("grounding workspace ownership mismatch")
        if (
            result.schema_binding.get("schema_decision_ref")
            != decision.schema_decision_id
            or result.schema_binding.get("schema_decision_hash")
            != decision.schema_decision_hash
            or result.schema_binding.get("candidate_ref")
            != decision.candidate_binding.get("candidate_id")
            or result.schema_binding.get("candidate_hash")
            != decision.candidate_binding.get("candidate_hash")
        ):
            raise ValueError("extract SchemaDecision binding mismatch")
        if (
            result.source_binding.get("lineage_id") != source.get("lineage_id")
            or result.source_binding.get("canonical_source_ir_ref")
            != source.get("canonical_source_ir_ref")
            or result.source_binding.get("canonical_source_hash")
            != source.get("source_hash")
        ):
            raise ValueError("grounding source/run ownership mismatch")

    def _resolve_admitted_bindings(
        self, decision: Any, *, for_new_write: bool = True
    ) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        candidate_evidence = {
            str(item["relation_type_candidate"]["relation_type_id"]): list(
                item["relation_type_candidate"]["evidence_refs"]
            )
            for item in decision.approved_schema.get("relation_types", [])
        }
        proposal_origin = (
            "system"
            if decision.candidate_binding["decided_by"] == "system"
            else "human"
        )
        for pins in decision.admitted_relation_type_bindings:
            admission = KnowledgeRelationTaxonomyService.resolve_relation_type(
                self.taxonomy,
                relation_type_id=str(pins["relation_type_id"]),
                taxonomy_ref=str(pins["taxonomy_ref"]),
                taxonomy_hash=str(pins["taxonomy_hash"]),
                profile_ref=pins.get("profile_ref"),
                profile_hash=pins.get("profile_hash"),
                profile_relation_catalog_hash=pins.get(
                    "profile_relation_catalog_hash"
                ),
                for_new_write=for_new_write,
                proposal_origin=proposal_origin,
            )
            relation = admission.relation_type
            binding = {
                "relation_type_id": relation.relation_type_id,
                "family_id": relation.family_id,
                "definition_scope": relation.definition_scope,
                "namespace": relation.namespace,
                "representation_kind": relation.representation_kind,
                "projection_shape": relation.projection_shape,
                "directed": relation.directed,
                "taxonomy_ref": admission.taxonomy_ref,
                "taxonomy_hash": admission.taxonomy_hash,
                "profile_ref": admission.profile_ref,
                "profile_hash": admission.profile_hash,
                "profile_relation_catalog_hash": (
                    admission.profile_relation_catalog_hash
                ),
                "source_type_refs": relation.source_type_refs,
                "target_type_refs": relation.target_type_refs,
                "source_role_refs": relation.source_role_refs,
                "target_role_refs": relation.target_role_refs,
                "evidence_refs": candidate_evidence.get(relation.relation_type_id, []),
            }
            if relation.relation_type_id in resolved:
                raise ValueError("duplicate admitted relation type")
            resolved[relation.relation_type_id] = binding
        return resolved

    @staticmethod
    def _contract_binding(binding: dict[str, Any]) -> dict[str, Any]:
        return {
            key: binding[key]
            for key in (
                "relation_type_id",
                "family_id",
                "definition_scope",
                "namespace",
                "representation_kind",
                "projection_shape",
                "directed",
                "taxonomy_ref",
                "taxonomy_hash",
                "profile_ref",
                "profile_hash",
                "profile_relation_catalog_hash",
            )
        }

    @staticmethod
    def _extract_binding(binding: dict[str, Any]) -> dict[str, Any]:
        return {
            key: binding[key]
            for key in (
                "relation_type_id",
                "family_id",
                "taxonomy_ref",
                "taxonomy_hash",
                "profile_ref",
                "profile_hash",
                "profile_relation_catalog_hash",
            )
        }

    def _validate_all_candidate_edges(
        self, result: ExtractResult, admitted: dict[str, dict[str, Any]]
    ) -> None:
        for edge in result.output.get("edges") or []:
            relation_type_id = str(edge.get("relation_type_id") or "")
            binding = admitted.get(relation_type_id)
            if binding is None:
                raise ValueError("candidate relation type is not admitted by SchemaDecision.v3")
            expected_fields = {
                "edge_id",
                "relation_type_id",
                "source_node_id",
                "target_node_id",
                "direction",
                "evidence_refs",
                "authority_state",
            }
            if binding["directed"]:
                expected_fields.update(
                    {"declared_source_type", "declared_target_type"}
                )
            if set(edge) != expected_fields:
                raise ValueError(
                    "candidate relation contains unauthorized semantic fields"
                )
            RelationUsageGroundingService._validate_direction_and_representation(
                edge, binding
            )
            RelationUsageGroundingService._validate_endpoints(edge, binding, result)

    def _resolve_subject(
        self,
        *,
        result: ExtractResult,
        admitted: dict[str, dict[str, Any]],
        subject_ref: str,
        subject_hash: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        edges = [
            dict(edge)
            for edge in (result.output.get("edges") or [])
            if edge.get("edge_id") == subject_ref
        ]
        unresolved = []
        for item in result.validation.get("unresolved_relation_evidence") or []:
            payload = dict(item)
            item_ref = "unresolved-relation-evidence-" + content_hash(payload)[:24]
            if item_ref == subject_ref:
                unresolved.append(payload)
        if len(edges) + len(unresolved) != 1:
            raise ValueError("relation usage subject is not uniquely present in extract")
        subject = edges[0] if edges else unresolved[0]
        actual_hash = f"sha256:{content_hash(subject)}"
        if subject_hash != actual_hash:
            raise ValueError("subject hash mismatch")
        if edges:
            binding = admitted.get(str(subject.get("relation_type_id") or ""))
            if binding is None:
                raise ValueError("candidate relation type is not admitted by SchemaDecision.v3")
            return (
                "candidate_relation_instance",
                subject,
                RelationUsageGroundingService._contract_binding(binding),
            )
        return "unresolved_relation_evidence", subject, None

    def _correct_occurrence(
        self,
        *,
        correction: dict[str, Any] | None,
        original: dict[str, Any],
        original_hash: str,
        admitted: dict[str, dict[str, Any]],
        result: ExtractResult,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(correction, dict) or set(correction) != _CORRECTION_FIELDS:
            raise ValueError("correction must contain exact occurrence fields")
        relation_type_id = str(correction.get("relation_type_id") or "")
        binding = admitted.get(relation_type_id)
        if binding is None:
            raise ValueError("corrected relation type is not admitted by SchemaDecision.v3")
        candidate = {
            "relation_type_id": relation_type_id,
            "source_node_id": str(correction.get("source_node_id") or ""),
            "target_node_id": str(correction.get("target_node_id") or ""),
            "direction": str(correction.get("direction") or ""),
        }
        self._validate_direction_and_representation(candidate, binding)
        self._validate_endpoints(candidate, binding, result)
        if all(candidate.get(key) == original.get(key) for key in candidate):
            raise ValueError("occurrence correction must materially change the candidate")
        identity_payload = {
            **candidate,
            "evidence_refs": sorted(map(str, original.get("evidence_refs") or [])),
            "authority_state": "candidate",
            "supersedes_relation_instance_ref": str(original["edge_id"]),
            "supersedes_relation_instance_hash": original_hash,
        }
        instance_hash = f"sha256:{content_hash(identity_payload)}"
        corrected = {
            "relation_instance_ref": f"grounded-relation-candidate-{instance_hash[7:31]}",
            "relation_instance_hash": instance_hash,
            **identity_payload,
        }
        return self._contract_binding(binding), corrected

    @staticmethod
    def _validate_direction_and_representation(
        occurrence: dict[str, Any], binding: dict[str, Any]
    ) -> None:
        # Historical W2e records may bind frame-role compatibility edges. New W3
        # authority is resolved exclusively through relation_occurrences.
        if (
            binding["representation_kind"],
            binding["projection_shape"],
        ) not in {("binary_edge", "binary"), ("frame_role", "frame")}:
            raise ValueError("candidate edge violates admitted representation contract")
        direction = str(occurrence.get("direction") or "")
        if binding["directed"]:
            if direction not in {"source_to_target", "target_to_source"}:
                raise ValueError("direction does not match admitted relation type")
        else:
            source = str(occurrence.get("source_node_id") or "")
            target = str(occurrence.get("target_node_id") or "")
            if (
                direction != "undirected"
                or not source
                or not target
                or source.encode("utf-8") > target.encode("utf-8")
            ):
                raise ValueError("direction does not match admitted relation type")

    @staticmethod
    def _validate_endpoints(
        occurrence: dict[str, Any], binding: dict[str, Any], result: ExtractResult
    ) -> None:
        nodes = {
            str(node.get("node_id") or ""): str(node.get("type_id") or "")
            for node in (result.output.get("nodes") or [])
        }
        source_ref = str(occurrence["source_node_id"])
        target_ref = str(occurrence["target_node_id"])
        if source_ref not in nodes or target_ref not in nodes:
            raise ValueError("corrected relation endpoints must exist in bound extract")
        semantic_source = source_ref
        semantic_target = target_ref
        if occurrence["direction"] == "target_to_source":
            semantic_source, semantic_target = semantic_target, semantic_source
        source_types = tuple(binding.get("source_type_refs") or ())
        target_types = tuple(binding.get("target_type_refs") or ())
        if source_types and nodes[semantic_source] not in source_types:
            raise ValueError("corrected source endpoint violates admitted type constraints")
        if target_types and nodes[semantic_target] not in target_types:
            raise ValueError("corrected target endpoint violates admitted type constraints")


_AUTHORITY_CONSTRUCTION_TOKEN = object()


class _RelationUsageGroundingAuthority:
    """Sealed supported-composition authority for Grounding decision semantics."""

    __slots__ = (
        "workspace_root",
        "workspace_dir",
        "organization_id",
        "workspace_id",
        "taxonomy",
        "decisions",
        "_sealed",
    )

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        profile_registry: CorpusGraphSchemaProfileRegistry | None,
        profile_relation_type_proposal_store: (
            ProfileRelationTypeProposalStore | None
        ),
        _token: object,
    ) -> None:
        if _token is not _AUTHORITY_CONSTRUCTION_TOKEN:
            raise ValueError("relation usage Grounding authority composition is sealed")
        root = Path(workspace_root).resolve()
        object.__setattr__(self, "workspace_root", root)
        object.__setattr__(self, "workspace_dir", root / "workspaces" / workspace_id)
        object.__setattr__(self, "organization_id", organization_id)
        object.__setattr__(self, "workspace_id", workspace_id)
        governed_profile_registry = (
            profile_registry
            or CorpusGraphSchemaProfileRegistry(
                repository_root=Path(__file__).resolve().parents[2]
            )
        )
        governed_proposal_store = (
            profile_relation_type_proposal_store
            or ProfileRelationTypeProposalStore(
                root,
                workspace_id=workspace_id,
                organization_id=organization_id,
            )
        )
        object.__setattr__(
            self,
            "taxonomy",
            KnowledgeRelationTaxonomyService(
                repository_root=Path(__file__).resolve().parents[2],
                profile_registry=governed_profile_registry,
                profile_relation_type_proposal_store=governed_proposal_store,
            ),
        )
        object.__setattr__(
            self,
            "decisions",
            SchemaDecisionStore(root, workspace_id=workspace_id),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("relation usage Grounding authority is immutable")
        object.__setattr__(self, name, value)

    def validate(
        self,
        grounding: RelationUsageGroundingDecision,
        *,
        for_new_write: bool,
    ) -> None:
        if type(self) is not _RelationUsageGroundingAuthority:
            raise ValueError("relation usage Grounding requires exact governed authority")
        if type(grounding) is not RelationUsageGroundingDecision:
            raise ValueError(
                "relation usage Grounding requires the exact governed decision contract"
            )
        if (
            grounding.organization_id != self.organization_id
            or grounding.workspace_id != self.workspace_id
        ):
            raise ValueError("grounding tenant scope mismatch")

        result = RelationUsageGroundingService._load_extract_result(
            self, str(grounding.extract_run_binding["extract_run_ref"])
        )
        if result.extract_result_hash != grounding.extract_run_binding[
            "extract_result_hash"
        ]:
            raise ValueError("grounding extract result hash mismatch")
        decision_payload = self.decisions.get(
            str(grounding.schema_decision_binding["schema_decision_ref"]),
            taxonomy_service=self.taxonomy,
        )
        decision = KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
            self.taxonomy, decision_payload
        )
        if decision.schema_decision_hash != grounding.schema_decision_binding[
            "schema_decision_hash"
        ]:
            raise ValueError("grounding SchemaDecision hash mismatch")
        RelationUsageGroundingService._validate_ownership_and_pins(
            self, result=result, decision=decision
        )
        if grounding.source_binding != dict(decision.source_binding):
            raise ValueError("grounding source binding mismatch")

        admitted = RelationUsageGroundingService._resolve_admitted_bindings(
            self, decision, for_new_write=for_new_write
        )
        expected_extract_bindings = [
            RelationUsageGroundingService._extract_binding(binding)
            for _, binding in sorted(admitted.items(), key=lambda item: item[0])
        ]
        if result.schema_binding.get("admitted_relation_types") != (
            expected_extract_bindings
        ):
            raise ValueError("extract admitted relation binding mismatch")
        RelationUsageGroundingService._validate_all_candidate_edges(
            self, result, admitted
        )
        occurrence_surface = result.output.get("relation_occurrences")
        if for_new_write and not isinstance(occurrence_surface, list):
            raise ValueError(
                "new relation usage Grounding requires relation_occurrences authority surface"
            )
        if isinstance(occurrence_surface, list):
            from services.governed_extract_service import GovernedExtractService

            if not GovernedExtractService._valid_relation_occurrence_surface(
                result.output, admitted
            ):
                raise ValueError("ExtractResult relation occurrence surface is invalid")

        for binding in (
            grounding.subject_relation_binding,
            grounding.corrected_relation_binding,
        ):
            if binding is None:
                continue
            admitted_binding = admitted.get(str(binding["relation_type_id"]))
            if admitted_binding is None:
                raise ValueError(
                    "relation type is not admitted by bound SchemaDecision.v3"
                )
            if binding != RelationUsageGroundingService._contract_binding(
                admitted_binding
            ):
                raise ValueError("relation binding differs from bound SchemaDecision.v3")

        subject_kind, subject, subject_binding = (
            RelationUsageGroundingService._resolve_subject(
                self,
                result=result,
                admitted=admitted,
                subject_ref=str(grounding.subject_binding["subject_ref"]),
                subject_hash=str(grounding.subject_binding["subject_hash"]),
            )
        )
        if grounding.subject_binding["subject_kind"] != subject_kind:
            raise ValueError("grounding subject kind differs from bound extract")
        if subject_kind == "candidate_relation_instance":
            if grounding.original_relation_instance != subject:
                raise ValueError(
                    "original relation instance is not exact member of bound extract"
                )
            if grounding.subject_relation_binding != subject_binding:
                raise ValueError("subject relation binding differs from bound extract")
        elif grounding.finding_kind != "missing_predicate":
            raise ValueError("only missing predicate may bind unresolved extract evidence")

        if grounding.corrected_relation_instance is not None:
            corrected = grounding.corrected_relation_instance
            corrected_binding = admitted[str(corrected["relation_type_id"])]
            if corrected_binding["representation_kind"] != "binary_edge":
                raise ValueError(
                    "corrected frame/derived occurrence requires a governed successor"
                )
            RelationUsageGroundingService._validate_direction_and_representation(
                corrected, corrected_binding
            )
            RelationUsageGroundingService._validate_endpoints(
                corrected, corrected_binding, result
            )
            if corrected.get("evidence_refs") != subject.get("evidence_refs"):
                raise ValueError(
                    "corrected relation evidence must preserve bound extract evidence"
                )

    def resolve_occurrence(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        extract_result_ref: str,
        extract_result_hash: str,
        schema_decision_ref: str,
        schema_decision_hash: str,
        occurrence_ref: str,
        occurrence_hash: str,
        for_new_write: bool,
    ) -> tuple[ExtractResult, dict[str, Any]]:
        """Resolve one candidate edge through the same sealed W2e authority."""

        if type(self) is not _RelationUsageGroundingAuthority:
            raise ValueError("relation occurrence requires exact governed authority")
        if (
            organization_id != self.organization_id
            or workspace_id != self.workspace_id
        ):
            raise ValueError("relation occurrence tenant scope mismatch")
        result = RelationUsageGroundingService._load_extract_result(
            self, extract_result_ref
        )
        if result.extract_result_hash != extract_result_hash:
            raise ValueError("relation occurrence governed ExtractResult pin mismatch")
        decision_payload = self.decisions.get(
            schema_decision_ref, taxonomy_service=self.taxonomy
        )
        decision = KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
            self.taxonomy, decision_payload
        )
        if decision.schema_decision_hash != schema_decision_hash:
            raise ValueError("relation occurrence SchemaDecision pin mismatch")
        RelationUsageGroundingService._validate_ownership_and_pins(
            self, result=result, decision=decision
        )
        admitted = RelationUsageGroundingService._resolve_admitted_bindings(
            self, decision, for_new_write=for_new_write
        )
        expected_extract_bindings = [
            RelationUsageGroundingService._extract_binding(binding)
            for _, binding in sorted(admitted.items(), key=lambda item: item[0])
        ]
        if result.schema_binding.get("admitted_relation_types") != (
            expected_extract_bindings
        ):
            raise ValueError("extract admitted relation binding mismatch")
        RelationUsageGroundingService._validate_all_candidate_edges(
            self, result, admitted
        )
        occurrence = self._resolve_occurrence_surface(
            result=result,
            admitted=admitted,
            occurrence_ref=occurrence_ref,
            occurrence_hash=occurrence_hash,
        )
        return result, occurrence.model_dump(mode="json", by_alias=True)

    def _resolve_occurrence_surface(
        self,
        *,
        result: ExtractResult,
        admitted: dict[str, dict[str, Any]],
        occurrence_ref: str,
        occurrence_hash: str,
    ) -> ExtractedRelationOccurrenceV1:
        raw = result.output.get("relation_occurrences")
        if not isinstance(raw, list):
            raise ValueError(
                "ExtractResult lacks required relation_occurrences authority surface"
            )
        occurrences = [
            ExtractedRelationOccurrenceV1.model_validate(item) for item in raw
        ]
        from services.governed_extract_service import GovernedExtractService

        if not GovernedExtractService._valid_relation_occurrence_surface(
            result.output, admitted
        ):
            raise ValueError("ExtractResult relation occurrence surface is invalid")
        refs = [item.occurrence_ref for item in occurrences]
        if len(refs) != len(set(refs)):
            raise ValueError("ExtractResult contains duplicate relation occurrences")
        matches = [item for item in occurrences if item.occurrence_ref == occurrence_ref]
        if len(matches) != 1:
            raise ValueError("relation occurrence is not uniquely present in extract")
        occurrence = matches[0]
        if occurrence.occurrence_hash != occurrence_hash:
            raise ValueError("relation occurrence hash mismatch")
        if (
            occurrence.organization_id != self.organization_id
            or occurrence.workspace_id != self.workspace_id
        ):
            raise ValueError("relation occurrence tenant scope mismatch")
        binding = admitted.get(occurrence.relation_type_id)
        if binding is None:
            raise ValueError("relation occurrence type is not admitted")
        if occurrence.representation_kind != binding["representation_kind"]:
            raise ValueError("relation occurrence representation differs from admission")
        nodes = {
            str(node.get("node_id") or ""): str(node.get("type_id") or "")
            for node in (result.output.get("nodes") or [])
        }
        frames = {
            str(frame.get("frame_id") or "")
            for frame in (result.output.get("frames") or [])
        }
        if occurrence.representation_kind == "binary_edge":
            if (
                occurrence.source_node_id not in nodes
                or occurrence.target_node_id not in nodes
            ):
                raise ValueError("relation occurrence endpoints are absent")
            if binding.get("source_type_refs") and nodes[
                occurrence.source_node_id
            ] not in tuple(binding["source_type_refs"]):
                raise ValueError("relation occurrence source type is invalid")
            if binding.get("target_type_refs") and nodes[
                occurrence.target_node_id
            ] not in tuple(binding["target_type_refs"]):
                raise ValueError("relation occurrence target type is invalid")
            projections = [
                edge
                for edge in (result.output.get("edges") or [])
                if edge.get("relation_type_id") == occurrence.relation_type_id
                and GovernedExtractService._semantic_occurrence_endpoints(
                    edge=edge, binding=binding
                )
                == (occurrence.source_node_id, occurrence.target_node_id)
            ]
            if len(projections) != 1:
                raise ValueError(
                    "binary occurrence must have one exact compatibility edge"
                )
        elif occurrence.representation_kind == "frame_role":
            if occurrence.frame_ref not in frames:
                raise ValueError("relation occurrence frame is absent")
        else:
            governed_sources = set(refs) | frames
            if any(
                ref == occurrence.occurrence_ref or ref not in governed_sources
                for ref in occurrence.projection_source_refs
            ):
                raise ValueError("derived relation occurrence source is not governed")
            if occurrence.source_node_id is not None and (
                occurrence.source_node_id not in nodes
                or occurrence.target_node_id not in nodes
            ):
                raise ValueError("derived relation occurrence endpoints are absent")
            if occurrence.frame_ref is not None and occurrence.frame_ref not in frames:
                raise ValueError("derived relation occurrence frame is absent")
        return occurrence


def _create_relation_usage_grounding_authority(
    workspace_root: str | Path,
    *,
    organization_id: str,
    workspace_id: str,
    profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
    profile_relation_type_proposal_store: (
        ProfileRelationTypeProposalStore | None
    ) = None,
) -> _RelationUsageGroundingAuthority:
    return _RelationUsageGroundingAuthority(
        workspace_root,
        organization_id=organization_id,
        workspace_id=workspace_id,
        profile_registry=profile_registry,
        profile_relation_type_proposal_store=(
            profile_relation_type_proposal_store
        ),
        _token=_AUTHORITY_CONSTRUCTION_TOKEN,
    )


def _validate_relation_usage_grounding_decision(
    authority: object,
    grounding: RelationUsageGroundingDecision,
    *,
    for_new_write: bool,
) -> None:
    if type(authority) is not _RelationUsageGroundingAuthority:
        raise ValueError("relation usage Grounding requires exact governed authority")
    _RelationUsageGroundingAuthority.validate(
        authority, grounding, for_new_write=for_new_write
    )


def _resolve_relation_occurrence_grounding_inputs(
    authority: object,
    *,
    organization_id: str,
    workspace_id: str,
    extract_result_ref: str,
    extract_result_hash: str,
    schema_decision_ref: str,
    schema_decision_hash: str,
    occurrence_ref: str,
    occurrence_hash: str,
    for_new_write: bool,
) -> tuple[ExtractResult, dict[str, Any]]:
    if type(authority) is not _RelationUsageGroundingAuthority:
        raise ValueError("relation occurrence requires exact governed authority")
    return _RelationUsageGroundingAuthority.resolve_occurrence(
        authority,
        organization_id=organization_id,
        workspace_id=workspace_id,
        extract_result_ref=extract_result_ref,
        extract_result_hash=extract_result_hash,
        schema_decision_ref=schema_decision_ref,
        schema_decision_hash=schema_decision_hash,
        occurrence_ref=occurrence_ref,
        occurrence_hash=occurrence_hash,
        for_new_write=for_new_write,
    )


__all__ = ["RelationUsageGroundingService"]
