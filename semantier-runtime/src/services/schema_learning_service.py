"""Route typed Grounding feedback into append-only successor schema candidates."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate
from contracts.graph_intent_brief import GraphIntentReview
from contracts.material_correction_rerun import (
    MaterialCorrectionRerun,
    build_schema_feedback_retest,
    graph_intent_correction_material,
    material_correction_outcome_reason,
    relation_usage_route_material,
    schema_candidate_correction_material,
)
from contracts.schema_decision import SchemaDecision
from contracts.schema_learning_feedback import SchemaLearningFeedback
from contracts.schema_learning_feedback_v2 import SchemaLearningFeedbackV2
from contracts.semantic_release_retest import SemanticReleaseRetestV1
from contracts.source_structure_retest import SourceStructureRetestV1
from contracts.schema_correction_continuation import SchemaCorrectionContinuationV1
from contracts.schema_correction_release_finalization import (
    SchemaCorrectionReleaseFinalizationV1,
    SchemaCorrectionReleaseReservationV1,
)
from contracts.graph_intent_release_finalization import (
    GraphIntentReleaseFinalizationV1,
    GraphIntentReleaseReservationV1,
)
from contracts.governed_extract import ExtractRequest
from contracts.evidence_location import EvidenceRef
from eos.artifact_hashing import content_hash
from storage.corpus_graph_schema_candidate_store import (
    CorpusGraphSchemaCandidateStore,
)
from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore
from storage.schema_decision_store import SchemaDecisionStore
from storage.schema_learning_feedback_store import SchemaLearningFeedbackStore
from storage.source_structure_semantic_validation_store import (
    SourceStructureSemanticValidationStore,
)
from storage.material_correction_rerun_store import MaterialCorrectionRerunStore
from storage.atomic_json_artifact import publish_immutable_json
from storage.graph_intent_release_finalization_store import (
    GraphIntentReleaseFinalizationStore,
    _FINALIZATION_PUBLICATION_AUTHORITY,
)
from storage.schema_correction_continuation_store import (
    SchemaCorrectionContinuationStore,
    _SCHEMA_CONTINUATION_PUBLICATION_AUTHORITY,
)
from storage.schema_correction_release_finalization_store import (
    SchemaCorrectionReleaseFinalizationStore,
    _SCHEMA_RELEASE_FINALIZATION_AUTHORITY,
)
from services.governed_extract_service import GovernedExtractService
from services.graph_schema_discovery_service import (
    AcceptedWitnessGatePackage,
    GraphSchemaDiscoveryService,
    _ADJUDICATED_GRAPH_INTENT_SUCCESSOR_AUTHORITY,
    _TRUSTED_SCHEMA_ONLY_SUCCESSOR_DESCRIPTOR,
)
from services.graph_intent_witness_adjudication_service import (
    GraphIntentWitnessAdjudicationService,
)
from services.accepted_semantic_graph_materialization_service import (
    AcceptedSemanticGraphMaterializationService,
    ResolveMaterializationInputsCommand,
    ResolveReviewedPatternInputsCommand,
)
from services.semantic_impact_attribution_service import SemanticImpactAttributionService


_SUCCESSOR_PUBLICATION_LOCK = threading.RLock()
_GRAPH_INTENT_RELEASE_FINALIZATION_AUTHORITY = object()
_SCHEMA_CORRECTION_RELEASE_FINALIZATION_AUTHORITY = object()
_SOURCE_STRUCTURE_RETEST_AUTHORITY = object()


class SchemaLearningService:
    """Create feedback and successor proposals without rewriting accepted history."""

    _DELTA_FIELDS = {
        "extraction_plan",
        "unresolved_ambiguities",
        "assumptions",
        "proposed_node_types",
        "proposed_event_types",
        "proposed_frame_types",
        "representation_decisions",
    }

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        workspace_id: str,
        organization_id: str | None = None,
        discovery_service: GraphSchemaDiscoveryService | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.decisions = SchemaDecisionStore(workspace_root, workspace_id=workspace_id)
        self.candidates = CorpusGraphSchemaCandidateStore(
            workspace_root, workspace_id=workspace_id
        )
        self.feedback = SchemaLearningFeedbackStore(
            workspace_root, workspace_id=workspace_id
        )
        self.discovery_service = discovery_service
        self.discovery = (
            discovery_service.store
            if discovery_service is not None
            else GraphSchemaDiscoveryStore(workspace_root, workspace_id=workspace_id)
        )
        self.organization_id = organization_id
        self.graph_intent_store = (
            discovery_service.graph_intent_store
            if discovery_service is not None
            else None
        )
        self.material_reruns = (
            MaterialCorrectionRerunStore(
                workspace_root,
                organization_id=organization_id,
                workspace_id=workspace_id,
                discovery_service=discovery_service,
            )
            if organization_id is not None and discovery_service is not None
            else None
        )

    def record_and_create_successor(
        self,
        *,
        schema_decision_ref: str,
        feedback_content: dict[str, Any],
        candidate_delta: dict[str, Any] | None,
        graph_intent_changes: dict[str, Any] | None = None,
        witness_packages: tuple[AcceptedWitnessGatePackage, ...] | None = None,
    ) -> dict[str, Any]:
        with _SUCCESSOR_PUBLICATION_LOCK:
            if self.discovery_service is not None:
                decision = self.discovery_service.get_schema_decision_contract(
                    schema_decision_ref
                )
            else:
                decision_payload = self.decisions.get(schema_decision_ref)
                if decision_payload.get("schema_version") == "schema_decision.v3":
                    raise ValueError(
                        "Decision.v3 learning requires the governed discovery authority"
                    )
                decision = SchemaDecision.from_dict(decision_payload)
            if decision.schema_version != "schema_decision.v3":
                raise ValueError("historical SchemaDecision is read/replay-only")
            candidate = CorpusGraphSchemaCandidate.from_dict(
                self.candidates.get_by_hash(
                    str(decision.candidate_binding["candidate_hash"])
                )
            )
            feedback = self._feedback_from_content(feedback_content)
            self._validate_feedback_pins(
                feedback=feedback,
                decision=decision,
                candidate=candidate,
            )

            if (
                isinstance(feedback, SchemaLearningFeedbackV2)
                and feedback.grounding_ref is not None
                and any(
                    isinstance(item.get("required_route"), str)
                    for item in feedback.findings
                )
                and candidate_delta is None
                and graph_intent_changes is None
            ):
                return self._create_route_only(
                    feedback=feedback,
                    decision=decision,
                    candidate=candidate,
                )

            if feedback.feedback_scope == "instance_local":
                if candidate_delta is not None or graph_intent_changes is not None:
                    raise ValueError(
                        "instance_local feedback cannot create intent or schema successors"
                    )
                return {
                    "feedback": self.feedback.put(feedback),
                    "graph_intent": None,
                    "candidate": None,
                    "review_decision": None,
                }

            if feedback.feedback_scope != "schema_understanding":
                raise ValueError(
                    "only schema_understanding feedback creates a schema successor"
                )

            graph_intent_attributed = (
                isinstance(feedback, SchemaLearningFeedbackV2)
                and feedback.attribution["graph_intent"]
            )
            if graph_intent_attributed:
                return self._create_graph_intent_successor(
                    feedback=feedback,
                    decision=decision,
                    candidate=candidate,
                    candidate_delta=candidate_delta,
                    graph_intent_changes=graph_intent_changes,
                    witness_packages=witness_packages,
                )
            if graph_intent_changes is not None:
                raise ValueError(
                    "graph_intent_changes require graph_intent attribution"
                )
            if not isinstance(candidate_delta, dict):
                raise ValueError("schema feedback requires candidate_delta")
            if set(candidate_delta) - self._DELTA_FIELDS:
                raise ValueError("candidate_delta contains unsupported fields")
            canonical_material = schema_candidate_correction_material(
                candidate_delta
            )
            candidate_delta = dict(canonical_material["changes"])
            expected_delta_ref = f"sha256:{content_hash(candidate_delta)}"
            if feedback.proposed_candidate_delta_ref != expected_delta_ref:
                raise ValueError(
                    "proposed_candidate_delta_ref does not match candidate_delta"
                )
            return self._create_schema_successor(
                feedback=feedback,
                decision=decision,
                candidate=candidate,
                candidate_delta=candidate_delta,
            )

    @staticmethod
    def _feedback_from_content(
        content: dict[str, Any],
    ) -> SchemaLearningFeedback | SchemaLearningFeedbackV2:
        schema_version = content.get("schema_version")
        if schema_version == "schema_learning_feedback.v2":
            return SchemaLearningFeedbackV2.create(content)
        if schema_version in {None, "schema_learning_feedback.v1"}:
            return SchemaLearningFeedback.create(content)
        raise ValueError(
            f"unsupported schema learning feedback version: {schema_version}"
        )

    @staticmethod
    def _validate_feedback_pins(
        *,
        feedback: SchemaLearningFeedback | SchemaLearningFeedbackV2,
        decision: SchemaDecision,
        candidate: CorpusGraphSchemaCandidate,
    ) -> None:
        if feedback.schema_decision_ref != decision.schema_decision_id:
            raise ValueError("feedback schema_decision_ref mismatch")
        if feedback.candidate_ref != candidate.candidate_id:
            raise ValueError("feedback candidate_ref mismatch")
        if decision.candidate_binding.get("candidate_id") != candidate.candidate_id:
            raise ValueError("feedback decision candidate binding mismatch")
        if isinstance(feedback, SchemaLearningFeedbackV2):
            if feedback.source_binding != decision.source_binding:
                raise ValueError("feedback source binding mismatch")
        else:
            for key in ("lineage_id", "source_hash"):
                if feedback.source_binding.get(key) != decision.source_binding.get(key):
                    raise ValueError("feedback source binding mismatch")
        if candidate.schema_version in {
            "corpus_graph_schema_candidate.v3",
            "corpus_graph_schema_candidate.v4",
        }:
            if candidate.source_binding != decision.source_binding:
                raise ValueError("candidate and decision source pins do not match")
            if candidate.graph_intent_binding != decision.graph_intent_binding:
                raise ValueError("candidate and decision GraphIntent pins do not match")

    def _create_graph_intent_successor(
        self,
        *,
        feedback: SchemaLearningFeedbackV2,
        decision: SchemaDecision,
        candidate: CorpusGraphSchemaCandidate,
        candidate_delta: dict[str, Any] | None,
        graph_intent_changes: dict[str, Any] | None,
        witness_packages: tuple[AcceptedWitnessGatePackage, ...] | None,
    ) -> dict[str, Any]:
        if candidate_delta is not None:
            raise ValueError(
                "graph_intent correction recompiles instead of applying candidate_delta"
            )
        if not isinstance(graph_intent_changes, dict) or not graph_intent_changes:
            raise ValueError("graph_intent correction requires typed material changes")
        supported_pair = (
            decision.schema_version,
            candidate.schema_version,
        ) == ("schema_decision.v3", "corpus_graph_schema_candidate.v4")
        if not supported_pair or self.graph_intent_store is None or self.discovery_service is None:
            raise ValueError(
                "graph_intent correction requires the governed v3 discovery boundary"
            )
        intent_binding = dict(decision.graph_intent_binding or {})
        predecessor_intent = self.graph_intent_store.get(
            str(intent_binding.get("graph_intent_ref") or ""),
            graph_intent_hash=str(intent_binding.get("graph_intent_hash") or ""),
        )
        if predecessor_intent.source_binding.as_dict() != decision.source_binding:
            raise ValueError("source to GraphIntent pins do not match")
        source_identity = self.graph_intent_store.source_registry.get_source_identity(
            source_identity_ref=predecessor_intent.source_binding.source_identity_ref,
            organization_id=predecessor_intent.source_binding.organization_id,
            workspace_id=predecessor_intent.source_binding.workspace_id,
            lineage_id=predecessor_intent.source_binding.lineage_id,
        )
        if source_identity is None:
            raise ValueError("governed source identity not found")
        if (
            predecessor_intent.source_binding.source_version is None
            and source_identity.source_version is not None
        ):
            from storage.knowledge_builder_store import KnowledgeBuilderStore

            historical_identity = source_identity.model_copy(
                update={"source_version": None}
            )
            if not KnowledgeBuilderStore.permits_historical_null_source_version(
                self.graph_intent_store.source_registry,
                supplied_identity=historical_identity,
                registered_identity=source_identity,
                lineage_id=predecessor_intent.source_binding.lineage_id,
            ):
                raise ValueError(
                    "GraphIntent historical null source version is not row-derived"
                )
            source_identity = source_identity.model_copy(
                update={"source_version": None}
            )
        evidence_refs = tuple(
            dict.fromkeys(
                [
                    *feedback.decision_evidence_refs,
                    *(
                        str(ref)
                        for finding in feedback.findings
                        for ref in (finding.get("evidence_refs") or [])
                    ),
                ]
            )
        )
        reviewed_by = "system" if feedback.operating_mode == "auto_pilot" else "user"
        review = GraphIntentReview(
            state="edited",
            reviewed_by=reviewed_by,
            actor_ref=feedback.created_by,
            operating_mode=feedback.operating_mode,
            policy_ref=feedback.policy_ref,
            rationale_ref=(
                feedback.rationale_ref
                or f"schema-learning-feedback:{feedback.feedback_id}"
            ),
            decision_evidence_refs=evidence_refs,
            fallback_reason=feedback.fallback_reason,
        )
        feedback_payload = self.feedback.put(feedback)
        successor_intent = self.graph_intent_store.review(
            predecessor_intent.graph_intent_id,
            review=review,
            source_identity=source_identity,
            changes=graph_intent_changes,
        )
        if witness_packages:
            reconstructed = self.discovery_service.reconstruct_material_correction_witnesses(
                predecessor_graph_intent=predecessor_intent,
                successor_graph_intent=successor_intent,
                witness_packages=witness_packages,
                actor_ref=feedback.created_by,
            )
        else:
            gates = [
                item.get("selected_value")
                for item in candidate.synthesis_decisions
                if item.get("decision_kind") == "exemplar_reconstruction"
                and isinstance(item.get("selected_value"), dict)
            ]
            if len(gates) != 1:
                raise ValueError(
                    "Decision.v3 GraphIntent correction requires one trusted W1 gate"
                )
            gate = gates[0]
            reconstructed = self.discovery_service.reconstruct_persisted_material_correction_witnesses(
                predecessor_graph_intent=predecessor_intent,
                successor_graph_intent=successor_intent,
                positive_manifest_ref=str(gate.get("positive_manifest_ref") or ""),
                positive_manifest_hash=str(gate.get("positive_manifest_hash") or ""),
                hard_negative_manifest_ref=str(
                    gate.get("hard_negative_manifest_ref") or ""
                ),
                hard_negative_manifest_hash=str(
                    gate.get("hard_negative_manifest_hash") or ""
                ),
                actor_ref=feedback.created_by,
            )
        before_gate = reconstructed["before_gate"]
        positive_manifest = reconstructed["proposed_positive_manifest"]
        negative_manifest = reconstructed["proposed_hard_negative_manifest"]
        correction_material = graph_intent_correction_material(
            graph_intent_changes
        )
        outcome, reason = material_correction_outcome_reason(
            correction_material
        )
        if reconstructed["reason"] != reason:
            raise ValueError(
                "reconstructed witness outcome disagrees with correction material"
            )
        correction_material_hash = f"sha256:{content_hash(correction_material)}"
        rerun = MaterialCorrectionRerun.create(
            {
                "organization_id": successor_intent.source_binding.organization_id,
                "workspace_id": successor_intent.source_binding.workspace_id,
                "outcome": outcome,
                "reason": reason,
                "feedback_ref": feedback.feedback_id,
                "feedback_hash": f"sha256:{content_hash(feedback.as_dict())}",
                "correction_material_hash": correction_material_hash,
                "correction_material_ref": (
                    "artifacts/material-correction-material/"
                    + correction_material_hash.removeprefix("sha256:")
                    + ".json"
                ),
                "before": {
                    "graph_intent_ref": predecessor_intent.graph_intent_id,
                    "graph_intent_hash": predecessor_intent.graph_intent_hash,
                    "candidate_ref": candidate.candidate_id,
                    "candidate_hash": candidate.candidate_hash,
                    "schema_decision_ref": decision.schema_decision_id,
                    "schema_decision_hash": decision.schema_decision_hash,
                    "positive_manifest_ref": before_gate["positive_manifest_ref"],
                    "positive_manifest_hash": before_gate["positive_manifest_hash"],
                    "positive_count": before_gate["positive_count"],
                    "hard_negative_manifest_ref": before_gate[
                        "hard_negative_manifest_ref"
                    ],
                    "hard_negative_manifest_hash": before_gate[
                        "hard_negative_manifest_hash"
                    ],
                    "hard_negative_count": before_gate["hard_negative_count"],
                    "material_mismatches": before_gate["material_mismatches"],
                },
                "after": {
                    "graph_intent_ref": successor_intent.graph_intent_id,
                    "graph_intent_hash": successor_intent.graph_intent_hash,
                    "candidate_ref": None,
                    "candidate_hash": None,
                    "schema_decision_ref": None,
                    "schema_decision_hash": None,
                    "positive_manifest_ref": positive_manifest["manifestRef"],
                    "positive_manifest_hash": positive_manifest["contentHash"],
                    "positive_count": reconstructed["positive_count"],
                    "hard_negative_manifest_ref": negative_manifest["manifestRef"],
                    "hard_negative_manifest_hash": negative_manifest["contentHash"],
                    "hard_negative_count": reconstructed["hard_negative_count"],
                    "material_mismatches": reconstructed["material_mismatches"],
                },
                "comparison_ref": None,
                "comparison_hash": None,
                "created_at": feedback.created_at,
                "created_by": feedback.created_by,
            }
        )
        if self.material_reruns is None:
            raise ValueError("material correction rerun store is unavailable")
        self.material_reruns.put_pending(
            rerun,
            proposed_positive_witnesses=reconstructed[
                "proposed_positive_witnesses"
            ],
            proposed_hard_negative_witnesses=reconstructed[
                "proposed_hard_negative_witnesses"
            ],
            proposed_positive_manifest=positive_manifest,
            proposed_hard_negative_manifest=negative_manifest,
            correction_material=correction_material,
        )
        return {
            "feedback": feedback_payload,
            "graph_intent": successor_intent.as_dict(),
            "candidate": None,
            "review_decision": None,
            "material_correction_rerun": rerun.as_dict(),
            "proposed_witness_comparison": reconstructed,
        }


    def replay_material_correction_rerun(
        self, rerun_ref: str, *, rerun_hash: str
    ) -> dict[str, Any]:
        if self.material_reruns is None:
            raise ValueError("material correction rerun store is unavailable")
        return self.material_reruns.get(rerun_ref, rerun_hash=rerun_hash)

    def adjudicate_graph_intent_correction(
        self,
        rerun_ref: str,
        *,
        rerun_hash: str,
        witness_adjudication_ref: str,
        witness_adjudication_hash: str,
        requesting_actor_ref: str,
    ) -> dict[str, Any]:
        """Continue one pending GraphIntent correction after governed 32+6 review."""

        if self.material_reruns is None or self.discovery_service is None:
            raise ValueError("GraphIntent adjudication requires governed discovery")
        pending = self.material_reruns.get_pending_evidence(
            rerun_ref, rerun_hash=rerun_hash
        )
        rerun = MaterialCorrectionRerun.from_dict(pending["rerun"])
        feedback = SchemaLearningFeedbackV2.from_dict(
            self.feedback.get(rerun.feedback_ref)
        )
        predecessor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.candidates.get_by_id(str(rerun.before["candidate_ref"]))
        )
        predecessor_decision = self.discovery_service.get_schema_decision_contract(
            str(rerun.before["schema_decision_ref"])
        )
        successor_intent = self.graph_intent_store.get(
            str(rerun.after["graph_intent_ref"]),
            graph_intent_hash=str(rerun.after["graph_intent_hash"]),
        )

        adjudications = GraphIntentWitnessAdjudicationService(
            self.workspace_root,
            organization_id=rerun.organization_id,
            workspace_id=rerun.workspace_id,
            material_reruns=self.material_reruns,
        )
        adjudication = adjudications.get_verified(
            witness_adjudication_ref,
            adjudication_hash=witness_adjudication_hash,
            rerun_ref=rerun_ref,
            rerun_hash=rerun_hash,
        )
        if adjudication.actor_ref != requesting_actor_ref:
            raise PermissionError("witness adjudication belongs to another actor")
        positive_manifest, positive_witnesses = adjudications.replay_accepted_package(
            adjudication.accepted_positive_manifest
        )
        negative_manifest, negative_witnesses = adjudications.replay_accepted_package(
            adjudication.accepted_hard_negative_manifest
        )
        compiled = self.discovery_service.compile_adjudicated_graph_intent_successor(
            predecessor_candidate=predecessor_candidate,
            predecessor_decision=predecessor_decision,
            successor_graph_intent=successor_intent,
            proposed_positive_manifest=pending["proposed_positive_manifest"],
            proposed_positive_witnesses=pending["proposed_positive_witnesses"],
            accepted_positive_manifest=positive_manifest,
            accepted_positive_witnesses=positive_witnesses,
            proposed_hard_negative_manifest=pending[
                "proposed_hard_negative_manifest"
            ],
            proposed_hard_negative_witnesses=pending[
                "proposed_hard_negative_witnesses"
            ],
            accepted_hard_negative_manifest=negative_manifest,
            accepted_hard_negative_witnesses=negative_witnesses,
            clarification_ref=feedback.feedback_id,
            actor_ref=feedback.created_by,
            decided_at=feedback.created_at,
            _adjudication_authority=_ADJUDICATED_GRAPH_INTENT_SUCCESSOR_AUTHORITY,
        )
        successor_candidate = CorpusGraphSchemaCandidate.from_dict(
            compiled["candidate"]
        )
        successor_decision = self.discovery_service.get_schema_decision_contract(
            str(compiled["decision"]["schema_decision_id"])
        )
        comparison = self.comparable_retest(
            organization_id=rerun.organization_id,
            baseline_schema_decision_ref=predecessor_decision.schema_decision_id,
            successor_schema_decision_ref=successor_decision.schema_decision_id,
            requested_by=feedback.created_by,
        )
        return {
            "outcome": "pending_materialization",
            "graph_intent": successor_intent.as_dict(),
            "candidate": successor_candidate.as_dict(),
            "decision": successor_decision.as_dict(),
            "comparison": comparison,
            "material_correction_rerun": rerun.as_dict(),
            "exemplar_gate": compiled["exemplar_gate"],
        }

    def finalize_graph_intent_correction(
        self,
        rerun_ref: str,
        *,
        rerun_hash: str,
        witness_adjudication_ref: str,
        witness_adjudication_hash: str,
        requesting_actor_ref: str,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ids: tuple[str, ...],
        materializer: AcceptedSemanticGraphMaterializationService,
        _finalization_authority: object | None = None,
    ) -> dict[str, Any]:
        """Seal one GraphIntent correction only after an exact child release exists."""

        if _finalization_authority is not _GRAPH_INTENT_RELEASE_FINALIZATION_AUTHORITY:
            raise PermissionError("GraphIntent release finalization is capability-sealed")
        if type(materializer) is not AcceptedSemanticGraphMaterializationService:
            raise ValueError("GraphIntent finalization requires the exact materializer")

        continuation = self.adjudicate_graph_intent_correction(
            rerun_ref,
            rerun_hash=rerun_hash,
            witness_adjudication_ref=witness_adjudication_ref,
            witness_adjudication_hash=witness_adjudication_hash,
            requesting_actor_ref=requesting_actor_ref,
        )
        rerun = MaterialCorrectionRerun.from_dict(
            continuation["material_correction_rerun"]
        )
        feedback = SchemaLearningFeedbackV2.from_dict(
            self.feedback.get(rerun.feedback_ref)
        )
        successor_intent = self.graph_intent_store.get(
            str(continuation["graph_intent"]["graph_intent_id"]),
            graph_intent_hash=str(
                continuation["graph_intent"]["graph_intent_hash"]
            ),
        )
        successor_candidate = CorpusGraphSchemaCandidate.from_dict(
            continuation["candidate"]
        )
        successor_decision = self.discovery_service.get_schema_decision_contract(
            str(continuation["decision"]["schema_decision_id"])
        )
        comparison = dict(continuation["comparison"])
        materialization = ResolveMaterializationInputsCommand(
            organization_id=rerun.organization_id,
            workspace_id=rerun.workspace_id,
            source_identity_ref=successor_intent.source_binding.source_identity_ref,
            lineage_id=successor_intent.source_binding.lineage_id,
            graph_intent_ref=successor_intent.graph_intent_id,
            graph_intent_hash=successor_intent.graph_intent_hash,
            schema_candidate_ref=successor_candidate.candidate_id,
            schema_candidate_hash=successor_candidate.candidate_hash,
            schema_decision_ref=successor_decision.schema_decision_id,
            schema_decision_hash=successor_decision.schema_decision_hash,
            extract_result_ref=str(comparison["successor_extract_ref"]),
            extract_result_hash=str(comparison["successor_extract_hash"]),
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
        )
        predecessor_bundle = materializer.resolve_accepted_graph_release_v2(
            predecessor_release_ref,
            release_hash=predecessor_release_hash,
        )
        predecessor_lineage = predecessor_bundle["lineage"]
        expected_predecessor = {
            "graph_intent_ref": str(rerun.before["graph_intent_ref"]),
            "graph_intent_hash": str(rerun.before["graph_intent_hash"]),
            "schema_candidate_ref": str(rerun.before["candidate_ref"]),
            "schema_candidate_hash": str(rerun.before["candidate_hash"]),
            "schema_decision_ref": str(rerun.before["schema_decision_ref"]),
            "schema_decision_hash": str(rerun.before["schema_decision_hash"]),
        }
        if any(
            predecessor_lineage.get(field) != value
            for field, value in expected_predecessor.items()
        ):
            raise ValueError("predecessor release does not bind the rerun before-state")
        finalizations = GraphIntentReleaseFinalizationStore(
            self.workspace_root, workspace_id=self.workspace_id
        )
        reservation = finalizations.reserve(
            GraphIntentReleaseReservationV1.create(
                {
                    "schema_version": "graph_intent_release_reservation.v1",
                    "organization_id": rerun.organization_id,
                    "workspace_id": rerun.workspace_id,
                    "rerun_ref": rerun.rerun_ref,
                    "rerun_hash": rerun.rerun_hash,
                    "adjudication_ref": witness_adjudication_ref,
                    "adjudication_hash": witness_adjudication_hash,
                    "graph_intent_ref": successor_intent.graph_intent_id,
                    "graph_intent_hash": successor_intent.graph_intent_hash,
                    "schema_candidate_ref": successor_candidate.candidate_id,
                    "schema_candidate_hash": successor_candidate.candidate_hash,
                    "schema_decision_ref": successor_decision.schema_decision_id,
                    "schema_decision_hash": successor_decision.schema_decision_hash,
                    "extract_result_ref": str(comparison["successor_extract_ref"]),
                    "extract_result_hash": str(comparison["successor_extract_hash"]),
                    "predecessor_release_ref": predecessor_release_ref,
                    "predecessor_release_hash": predecessor_release_hash,
                    "candidate_graph_id": candidate_graph_id,
                    "human_label_event_ids": human_label_event_ids,
                }
            ),
            _authority=_FINALIZATION_PUBLICATION_AUTHORITY,
        )
        release_bundle = materializer.materialize_accepted_graph_release_v2(
            ResolveReviewedPatternInputsCommand(
                materialization=materialization,
                candidate_graph_id=candidate_graph_id,
                human_label_event_ids=human_label_event_ids,
            )
        )
        predecessor_release = predecessor_bundle["release"]
        successor_release = release_bundle["release"]
        release_comparison_content = {
            "schema_version": "semantic_release_retest.v1",
            "organization_id": rerun.organization_id,
            "workspace_id": rerun.workspace_id,
            "baseline_release_ref": predecessor_release_ref,
            "baseline_release_hash": predecessor_release_hash,
            "successor_release_ref": successor_release["graph_version"],
            "successor_release_hash": successor_release["release_hash"],
            "schema_feedback_retest_ref": comparison["comparison_id"],
            "schema_feedback_retest_hash": comparison["comparison_hash"],
            "pattern_membership": {
                "before": predecessor_release["pattern_membership"],
                "after": successor_release["pattern_membership"],
            },
            "relation_membership": {
                "before": predecessor_release["relation_membership"],
                "after": successor_release["relation_membership"],
            },
        }
        release_comparison_contract = SemanticReleaseRetestV1.create(
            release_comparison_content
        )
        release_comparison = release_comparison_contract.model_dump(mode="json")
        evaluation_id = release_comparison_contract.evaluation_id
        evaluation_path = (
            self.workspace_root
            / "workspaces"
            / self.workspace_id
            / "artifacts"
            / "schema-discovery"
            / "release-retests"
            / f"{evaluation_id.replace(':', '-')}.json"
        )
        release_comparison = SemanticReleaseRetestV1.from_dict(
            publish_immutable_json(
                evaluation_path,
                release_comparison,
                conflict_error="SEMANTIC_RELEASE_RETEST_ARTIFACT_CONFLICT",
            )
        )
        accepted_rerun = MaterialCorrectionRerun.create(
            {
                "organization_id": rerun.organization_id,
                "workspace_id": rerun.workspace_id,
                "outcome": "accepted",
                "reason": "graph_intent_successor_recompiled_cleanly",
                "feedback_ref": rerun.feedback_ref,
                "feedback_hash": rerun.feedback_hash,
                "correction_material_hash": rerun.correction_material_hash,
                "correction_material_ref": rerun.correction_material_ref,
                "before": rerun.before,
                "after": self._witness_gate_state(
                    graph_intent_ref=successor_intent.graph_intent_id,
                    graph_intent_hash=successor_intent.graph_intent_hash,
                    candidate=successor_candidate,
                    decision=successor_decision,
                    gate=continuation["exemplar_gate"],
                ),
                "comparison_ref": comparison["comparison_id"],
                "comparison_hash": comparison["comparison_hash"],
                "created_at": feedback.created_at,
                "created_by": feedback.created_by,
            }
        )
        if self.material_reruns is None:
            raise ValueError("material correction rerun store is unavailable")
        pending = self.material_reruns.get_pending_evidence(
            rerun_ref, rerun_hash=rerun_hash
        )
        predecessor_refs = (
            str(rerun.before["graph_intent_ref"]),
            str(rerun.before["candidate_ref"]),
            str(rerun.before["schema_decision_ref"]),
            *(item["pattern_ref"] for item in predecessor_release["pattern_membership"]),
            predecessor_release_ref,
        )
        successor_refs = (
            successor_intent.graph_intent_id,
            successor_candidate.candidate_id,
            successor_decision.schema_decision_id,
            *(item["pattern_ref"] for item in successor_release["pattern_membership"]),
            successor_release["graph_version"],
            evaluation_id,
        )
        attribution_evidence_by_ref = {
            evidence.evidence_ref: evidence
            for witness_key in (
                "proposed_positive_witnesses",
                "proposed_hard_negative_witnesses",
            )
            for witness in pending[witness_key]
            for evidence in witness.evidence_refs
        }
        attribution_service = SemanticImpactAttributionService(
            self.workspace_root, workspace_id=self.workspace_id
        )
        attribution = attribution_service.record(
            feedback_ref=feedback.feedback_id,
            feedback_hash=rerun.feedback_hash,
            semantic_impact_layer="graph_intent",
            target_artifact_ref=successor_intent.graph_intent_id,
            target_artifact_hash=successor_intent.graph_intent_hash,
            decision_evidence_refs=tuple(attribution_evidence_by_ref.values()),
            actor_ref=feedback.created_by,
            predecessor_refs=predecessor_refs,
            successor_refs=successor_refs,
        )
        finalization = finalizations.complete(
            rerun.rerun_ref,
            GraphIntentReleaseFinalizationV1.create(
                {
                    "schema_version": "graph_intent_release_finalization.v1",
                    "organization_id": rerun.organization_id,
                    "workspace_id": rerun.workspace_id,
                    "rerun_ref": rerun.rerun_ref,
                    "rerun_hash": rerun.rerun_hash,
                    "reservation_ref": reservation.reservation_ref,
                    "reservation_hash": reservation.reservation_hash,
                    "accepted_rerun_ref": accepted_rerun.rerun_ref,
                    "accepted_rerun_hash": accepted_rerun.rerun_hash,
                    "successor_release_ref": successor_release["graph_version"],
                    "successor_release_hash": successor_release["release_hash"],
                    "schema_feedback_retest_ref": comparison["comparison_id"],
                    "schema_feedback_retest_hash": comparison["comparison_hash"],
                    "semantic_release_retest_ref": evaluation_id,
                    "semantic_release_retest_hash": release_comparison.evaluation_hash,
                    "attribution_ref": attribution.attribution_id,
                    "attribution_hash": attribution.attribution_hash,
                }
            ),
            _authority=_FINALIZATION_PUBLICATION_AUTHORITY,
        )
        self.material_reruns.put_accepted(
            accepted_rerun,
            correction_material=pending["correction_material"],
            comparison=comparison,
        )
        finalization = finalizations.get_verified(
            rerun.rerun_ref,
            rerun_hash=rerun.rerun_hash,
            material_reruns=self.material_reruns,
            materializer=materializer,
            attribution_service=attribution_service,
            adjudications=GraphIntentWitnessAdjudicationService(
                self.workspace_root,
                organization_id=rerun.organization_id,
                workspace_id=rerun.workspace_id,
                material_reruns=self.material_reruns,
            ),
            _authority=_FINALIZATION_PUBLICATION_AUTHORITY,
        )
        return {
            **continuation,
            "outcome": "accepted",
            "material_correction_rerun": accepted_rerun.as_dict(),
            "accepted_graph_release": release_bundle,
            "release_comparison": release_comparison.model_dump(mode="json"),
            "semantic_impact_attribution": attribution.model_dump(mode="json"),
            "release_finalization": finalization.model_dump(mode="json"),
        }

    def finalize_schema_correction(
        self,
        continuation: SchemaCorrectionContinuationV1,
        *,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ids: tuple[str, ...],
        materializer: AcceptedSemanticGraphMaterializationService,
        _finalization_authority: object | None = None,
    ) -> dict[str, Any]:
        """Seal a schema correction only after an exact child release exists."""

        if (
            _finalization_authority
            is not _SCHEMA_CORRECTION_RELEASE_FINALIZATION_AUTHORITY
        ):
            raise PermissionError("schema release finalization is capability-sealed")
        if type(materializer) is not AcceptedSemanticGraphMaterializationService:
            raise ValueError("schema finalization requires the exact materializer")
        if self.discovery_service is None or self.material_reruns is None:
            raise ValueError("schema finalization requires governed discovery")

        feedback = SchemaLearningFeedbackV2.from_dict(
            self.feedback.get(continuation.feedback_ref)
        )
        intent = self.graph_intent_store.get(
            continuation.graph_intent_ref,
            graph_intent_hash=continuation.graph_intent_hash,
        )
        predecessor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.candidates.get_by_hash(continuation.predecessor_candidate_hash)
        )
        successor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.candidates.get_by_hash(continuation.successor_candidate_hash)
        )
        predecessor_decision = self.discovery_service.get_schema_decision_contract(
            continuation.predecessor_schema_decision_ref
        )
        successor_decision = self.discovery_service.get_schema_decision_contract(
            continuation.successor_schema_decision_ref
        )
        comparison = self.comparable_retest(
            organization_id=continuation.organization_id,
            baseline_schema_decision_ref=predecessor_decision.schema_decision_id,
            successor_schema_decision_ref=successor_decision.schema_decision_id,
            requested_by=continuation.created_by,
        )
        if (
            comparison["comparison_id"] != continuation.comparison_ref
            or comparison["comparison_hash"] != continuation.comparison_hash
        ):
            raise ValueError("schema continuation comparison mismatch")
        predecessor_bundle = materializer.resolve_accepted_graph_release_v2(
            predecessor_release_ref, release_hash=predecessor_release_hash
        )
        predecessor_lineage = predecessor_bundle["lineage"]
        expected_predecessor = {
            "graph_intent_ref": intent.graph_intent_id,
            "graph_intent_hash": intent.graph_intent_hash,
            "schema_candidate_ref": predecessor_candidate.candidate_id,
            "schema_candidate_hash": predecessor_candidate.candidate_hash,
            "schema_decision_ref": predecessor_decision.schema_decision_id,
            "schema_decision_hash": predecessor_decision.schema_decision_hash,
        }
        if any(
            predecessor_lineage.get(field) != value
            for field, value in expected_predecessor.items()
        ):
            raise ValueError("predecessor release does not bind schema continuation")

        command = ResolveMaterializationInputsCommand(
            organization_id=continuation.organization_id,
            workspace_id=continuation.workspace_id,
            source_identity_ref=intent.source_binding.source_identity_ref,
            lineage_id=intent.source_binding.lineage_id,
            graph_intent_ref=intent.graph_intent_id,
            graph_intent_hash=intent.graph_intent_hash,
            schema_candidate_ref=successor_candidate.candidate_id,
            schema_candidate_hash=successor_candidate.candidate_hash,
            schema_decision_ref=successor_decision.schema_decision_id,
            schema_decision_hash=successor_decision.schema_decision_hash,
            extract_result_ref=str(comparison["successor_extract_ref"]),
            extract_result_hash=str(comparison["successor_extract_hash"]),
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
        )
        finalizations = SchemaCorrectionReleaseFinalizationStore(
            self.workspace_root, workspace_id=self.workspace_id
        )
        reservation = finalizations.reserve(
            SchemaCorrectionReleaseReservationV1.create(
                {
                    "schema_version": "schema_correction_release_reservation.v1",
                    "organization_id": continuation.organization_id,
                    "workspace_id": continuation.workspace_id,
                    "continuation_ref": continuation.continuation_ref,
                    "continuation_hash": continuation.continuation_hash,
                    "graph_intent_ref": intent.graph_intent_id,
                    "graph_intent_hash": intent.graph_intent_hash,
                    "schema_candidate_ref": successor_candidate.candidate_id,
                    "schema_candidate_hash": successor_candidate.candidate_hash,
                    "schema_decision_ref": successor_decision.schema_decision_id,
                    "schema_decision_hash": successor_decision.schema_decision_hash,
                    "extract_result_ref": str(comparison["successor_extract_ref"]),
                    "extract_result_hash": str(comparison["successor_extract_hash"]),
                    "predecessor_release_ref": predecessor_release_ref,
                    "predecessor_release_hash": predecessor_release_hash,
                    "candidate_graph_id": candidate_graph_id,
                    "human_label_event_ids": human_label_event_ids,
                }
            ),
            _authority=_SCHEMA_RELEASE_FINALIZATION_AUTHORITY,
        )
        release_bundle = materializer.materialize_accepted_graph_release_v2(
            ResolveReviewedPatternInputsCommand(
                materialization=command,
                candidate_graph_id=candidate_graph_id,
                human_label_event_ids=human_label_event_ids,
            )
        )
        predecessor_release = predecessor_bundle["release"]
        successor_release = release_bundle["release"]
        release_retest = SemanticReleaseRetestV1.create(
            {
                "schema_version": "semantic_release_retest.v1",
                "organization_id": continuation.organization_id,
                "workspace_id": continuation.workspace_id,
                "baseline_release_ref": predecessor_release_ref,
                "baseline_release_hash": predecessor_release_hash,
                "successor_release_ref": successor_release["graph_version"],
                "successor_release_hash": successor_release["release_hash"],
                "schema_feedback_retest_ref": continuation.comparison_ref,
                "schema_feedback_retest_hash": continuation.comparison_hash,
                "pattern_membership": {
                    "before": predecessor_release["pattern_membership"],
                    "after": successor_release["pattern_membership"],
                },
                "relation_membership": {
                    "before": predecessor_release["relation_membership"],
                    "after": successor_release["relation_membership"],
                },
            }
        )
        release_retest = SemanticReleaseRetestV1.from_dict(
            publish_immutable_json(
                self.workspace_root / "workspaces" / self.workspace_id
                / "artifacts/schema-discovery/release-retests"
                / f"{release_retest.evaluation_id.replace(':', '-')}.json",
                release_retest.model_dump(mode="json"),
                conflict_error="SEMANTIC_RELEASE_RETEST_ARTIFACT_CONFLICT",
            )
        )
        gate = {
            "positive_manifest_ref": continuation.positive_manifest_ref,
            "positive_manifest_hash": continuation.positive_manifest_hash,
            "positive_count": continuation.positive_count,
            "hard_negative_manifest_ref": continuation.hard_negative_manifest_ref,
            "hard_negative_manifest_hash": continuation.hard_negative_manifest_hash,
            "hard_negative_count": continuation.hard_negative_count,
            "material_mismatches": [],
        }
        accepted_rerun = MaterialCorrectionRerun.create(
            {
                "organization_id": continuation.organization_id,
                "workspace_id": continuation.workspace_id,
                "outcome": "accepted",
                "reason": "schema_only_successor_recompiled_cleanly",
                "feedback_ref": continuation.feedback_ref,
                "feedback_hash": continuation.feedback_hash,
                "correction_material_hash": continuation.correction_material_hash,
                "correction_material_ref": continuation.correction_material_ref,
                "before": self._witness_gate_state(
                    graph_intent_ref=intent.graph_intent_id,
                    graph_intent_hash=intent.graph_intent_hash,
                    candidate=predecessor_candidate,
                    decision=predecessor_decision,
                    gate=gate,
                ),
                "after": self._witness_gate_state(
                    graph_intent_ref=intent.graph_intent_id,
                    graph_intent_hash=intent.graph_intent_hash,
                    candidate=successor_candidate,
                    decision=successor_decision,
                    gate=gate,
                ),
                "comparison_ref": continuation.comparison_ref,
                "comparison_hash": continuation.comparison_hash,
                "created_at": continuation.created_at,
                "created_by": continuation.created_by,
            }
        )
        material = schema_candidate_correction_material(
            {
                field: successor_candidate.as_dict()[field]
                for field in self._DELTA_FIELDS
                if successor_candidate.as_dict().get(field)
                != predecessor_candidate.as_dict().get(field)
            }
        )
        predecessor_refs = (
            intent.graph_intent_id,
            predecessor_candidate.candidate_id,
            predecessor_decision.schema_decision_id,
            *(item["pattern_ref"] for item in predecessor_release["pattern_membership"]),
            predecessor_release_ref,
        )
        successor_refs = (
            successor_candidate.candidate_id,
            successor_decision.schema_decision_id,
            *(item["pattern_ref"] for item in successor_release["pattern_membership"]),
            successor_release["graph_version"],
            release_retest.evaluation_id,
        )
        evidence_by_ref = {
            evidence.evidence_ref: evidence
            for relation in successor_candidate.proposed_relation_types
            for evidence in (
                EvidenceRef.model_validate(item)
                for item in relation["relation_type_candidate"]["evidence_refs"]
            )
        }
        attribution_service = SemanticImpactAttributionService(
            self.workspace_root, workspace_id=self.workspace_id
        )
        attribution = attribution_service.record(
            feedback_ref=feedback.feedback_id,
            feedback_hash=continuation.feedback_hash,
            semantic_impact_layer="schema",
            target_artifact_ref=successor_candidate.candidate_id,
            target_artifact_hash=successor_candidate.candidate_hash,
            decision_evidence_refs=tuple(evidence_by_ref.values()),
            actor_ref=feedback.created_by,
            predecessor_refs=predecessor_refs,
            successor_refs=successor_refs,
        )
        finalization = finalizations.complete(
            continuation.continuation_ref,
            SchemaCorrectionReleaseFinalizationV1.create(
                {
                    "schema_version": "schema_correction_release_finalization.v1",
                    "organization_id": continuation.organization_id,
                    "workspace_id": continuation.workspace_id,
                    "continuation_ref": continuation.continuation_ref,
                    "continuation_hash": continuation.continuation_hash,
                    "reservation_ref": reservation.reservation_ref,
                    "reservation_hash": reservation.reservation_hash,
                    "accepted_rerun_ref": accepted_rerun.rerun_ref,
                    "accepted_rerun_hash": accepted_rerun.rerun_hash,
                    "successor_release_ref": successor_release["graph_version"],
                    "successor_release_hash": successor_release["release_hash"],
                    "schema_feedback_retest_ref": continuation.comparison_ref,
                    "schema_feedback_retest_hash": continuation.comparison_hash,
                    "semantic_release_retest_ref": release_retest.evaluation_id,
                    "semantic_release_retest_hash": release_retest.evaluation_hash,
                    "attribution_ref": attribution.attribution_id,
                    "attribution_hash": attribution.attribution_hash,
                }
            ),
            _authority=_SCHEMA_RELEASE_FINALIZATION_AUTHORITY,
        )
        self.material_reruns.put_accepted(
            accepted_rerun, correction_material=material, comparison=comparison
        )
        return {
            "outcome": "accepted",
            "material_correction_rerun": accepted_rerun.as_dict(),
            "accepted_graph_release": release_bundle,
            "release_comparison": release_retest.model_dump(mode="json"),
            "semantic_impact_attribution": attribution.model_dump(mode="json"),
            "release_finalization": finalization.model_dump(mode="json"),
        }

    def replay_schema_correction_release_finalization(
        self,
        continuation: SchemaCorrectionContinuationV1,
        *,
        materializer: AcceptedSemanticGraphMaterializationService,
        _finalization_authority: object | None = None,
    ) -> SchemaCorrectionReleaseFinalizationV1:
        if (
            _finalization_authority
            is not _SCHEMA_CORRECTION_RELEASE_FINALIZATION_AUTHORITY
        ):
            raise PermissionError("schema release finalization replay is capability-sealed")
        if type(materializer) is not AcceptedSemanticGraphMaterializationService:
            raise ValueError("schema replay requires the exact materializer")
        if self.material_reruns is None or self.discovery_service is None:
            raise ValueError("schema replay requires governed correction storage")
        store = SchemaCorrectionReleaseFinalizationStore(
            self.workspace_root, workspace_id=self.workspace_id
        )
        reservation = store.get_reservation(
            continuation.continuation_ref,
            continuation_hash=continuation.continuation_hash,
        )
        finalization = store.get_finalization(
            continuation.continuation_ref,
            continuation_hash=continuation.continuation_hash,
        )
        if (
            finalization.schema_feedback_retest_ref
            != continuation.comparison_ref
            or finalization.schema_feedback_retest_hash
            != continuation.comparison_hash
        ):
            raise ValueError("schema finalization comparison binding mismatch")
        feedback = SchemaLearningFeedbackV2.from_dict(
            self.feedback.get(continuation.feedback_ref)
        )
        intent = self.graph_intent_store.get(
            continuation.graph_intent_ref,
            graph_intent_hash=continuation.graph_intent_hash,
        )
        predecessor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.candidates.get_by_hash(continuation.predecessor_candidate_hash)
        )
        successor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.candidates.get_by_hash(continuation.successor_candidate_hash)
        )
        predecessor_decision = self.discovery_service.get_schema_decision_contract(
            continuation.predecessor_schema_decision_ref
        )
        successor_decision = self.discovery_service.get_schema_decision_contract(
            continuation.successor_schema_decision_ref
        )
        comparison = self.comparable_retest(
            organization_id=continuation.organization_id,
            baseline_schema_decision_ref=predecessor_decision.schema_decision_id,
            successor_schema_decision_ref=successor_decision.schema_decision_id,
            requested_by=continuation.created_by,
        )
        if (
            comparison["comparison_id"] != continuation.comparison_ref
            or comparison["comparison_hash"] != continuation.comparison_hash
        ):
            raise ValueError("schema continuation comparison mismatch")
        fixed_reservation = {
            "organization_id": continuation.organization_id,
            "workspace_id": continuation.workspace_id,
            "continuation_ref": continuation.continuation_ref,
            "continuation_hash": continuation.continuation_hash,
            "graph_intent_ref": continuation.graph_intent_ref,
            "graph_intent_hash": continuation.graph_intent_hash,
            "schema_candidate_ref": continuation.successor_candidate_ref,
            "schema_candidate_hash": continuation.successor_candidate_hash,
            "schema_decision_ref": continuation.successor_schema_decision_ref,
            "schema_decision_hash": continuation.successor_schema_decision_hash,
            "extract_result_ref": comparison["successor_extract_ref"],
            "extract_result_hash": comparison["successor_extract_hash"],
        }
        if any(
            getattr(reservation, field) != value
            for field, value in fixed_reservation.items()
        ):
            raise ValueError("schema release reservation continuation mismatch")
        accepted = self.material_reruns.get(
            finalization.accepted_rerun_ref,
            rerun_hash=finalization.accepted_rerun_hash,
        )
        release_bundle = materializer.resolve_accepted_graph_release_v2(
            finalization.successor_release_ref,
            release_hash=finalization.successor_release_hash,
        )
        release = release_bundle["release"]
        lineage = release_bundle["lineage"]
        if (
            release.get("parent_graph_version") != reservation.predecessor_release_ref
            or release.get("candidate_graph_id") != reservation.candidate_graph_id
            or tuple(release.get("human_label_event_ids") or ())
            != reservation.human_label_event_ids
            or any(
                lineage.get(field) != value
                for field, value in {
                    "graph_intent_ref": reservation.graph_intent_ref,
                    "graph_intent_hash": reservation.graph_intent_hash,
                    "schema_candidate_ref": reservation.schema_candidate_ref,
                    "schema_candidate_hash": reservation.schema_candidate_hash,
                    "schema_decision_ref": reservation.schema_decision_ref,
                    "schema_decision_hash": reservation.schema_decision_hash,
                    "schema_validation_extract_run_ref": reservation.extract_result_ref,
                    "schema_validation_extract_result_hash": reservation.extract_result_hash,
                    "candidate_graph_materialization_extract_run_ref": reservation.extract_result_ref,
                }.items()
            )
        ):
            raise ValueError("schema finalization child release lineage mismatch")
        predecessor_bundle = materializer.resolve_accepted_graph_release_v2(
            reservation.predecessor_release_ref,
            release_hash=reservation.predecessor_release_hash,
        )
        predecessor_lineage = predecessor_bundle["lineage"]
        if any(
            predecessor_lineage.get(field) != value
            for field, value in {
                "graph_intent_ref": continuation.graph_intent_ref,
                "graph_intent_hash": continuation.graph_intent_hash,
                "schema_candidate_ref": continuation.predecessor_candidate_ref,
                "schema_candidate_hash": continuation.predecessor_candidate_hash,
                "schema_decision_ref": continuation.predecessor_schema_decision_ref,
                "schema_decision_hash": continuation.predecessor_schema_decision_hash,
            }.items()
        ):
            raise ValueError("schema finalization predecessor release lineage mismatch")
        gate = {
            "positive_manifest_ref": continuation.positive_manifest_ref,
            "positive_manifest_hash": continuation.positive_manifest_hash,
            "positive_count": continuation.positive_count,
            "hard_negative_manifest_ref": continuation.hard_negative_manifest_ref,
            "hard_negative_manifest_hash": continuation.hard_negative_manifest_hash,
            "hard_negative_count": continuation.hard_negative_count,
            "material_mismatches": [],
        }
        expected_accepted = MaterialCorrectionRerun.create(
            {
                "organization_id": continuation.organization_id,
                "workspace_id": continuation.workspace_id,
                "outcome": "accepted",
                "reason": "schema_only_successor_recompiled_cleanly",
                "feedback_ref": continuation.feedback_ref,
                "feedback_hash": continuation.feedback_hash,
                "correction_material_hash": continuation.correction_material_hash,
                "correction_material_ref": continuation.correction_material_ref,
                "before": self._witness_gate_state(
                    graph_intent_ref=intent.graph_intent_id,
                    graph_intent_hash=intent.graph_intent_hash,
                    candidate=predecessor_candidate,
                    decision=predecessor_decision,
                    gate=gate,
                ),
                "after": self._witness_gate_state(
                    graph_intent_ref=intent.graph_intent_id,
                    graph_intent_hash=intent.graph_intent_hash,
                    candidate=successor_candidate,
                    decision=successor_decision,
                    gate=gate,
                ),
                "comparison_ref": continuation.comparison_ref,
                "comparison_hash": continuation.comparison_hash,
                "created_at": continuation.created_at,
                "created_by": continuation.created_by,
            }
        )
        if accepted != expected_accepted.as_dict():
            raise ValueError("schema finalization accepted rerun binding mismatch")
        semantic_path = (
            store.root.parent / "release-retests"
            / f"{finalization.semantic_release_retest_ref.replace(':', '-')}.json"
        )
        semantic = SemanticReleaseRetestV1.model_validate_json(
            semantic_path.read_text(encoding="utf-8")
        )
        if (
            semantic.evaluation_id != finalization.semantic_release_retest_ref
            or semantic.evaluation_hash != finalization.semantic_release_retest_hash
            or semantic.baseline_release_ref != reservation.predecessor_release_ref
            or semantic.baseline_release_hash != reservation.predecessor_release_hash
            or semantic.successor_release_ref != finalization.successor_release_ref
            or semantic.successor_release_hash != finalization.successor_release_hash
            or semantic.schema_feedback_retest_ref != continuation.comparison_ref
            or semantic.schema_feedback_retest_hash != continuation.comparison_hash
            or semantic.pattern_membership.model_dump(mode="json")
            != {
                "before": predecessor_bundle["release"]["pattern_membership"],
                "after": release["pattern_membership"],
            }
            or semantic.relation_membership.model_dump(mode="json")
            != {
                "before": predecessor_bundle["release"]["relation_membership"],
                "after": release["relation_membership"],
            }
        ):
            raise ValueError("schema finalization semantic release retest mismatch")
        attribution = SemanticImpactAttributionService(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(
            finalization.attribution_ref,
            attribution_hash=finalization.attribution_hash,
        )
        predecessor_refs = (
            continuation.graph_intent_ref,
            continuation.predecessor_candidate_ref,
            continuation.predecessor_schema_decision_ref,
            *(
                item["pattern_ref"]
                for item in predecessor_bundle["release"]["pattern_membership"]
            ),
            reservation.predecessor_release_ref,
        )
        successor_refs = (
            reservation.schema_candidate_ref,
            reservation.schema_decision_ref,
            *(item["pattern_ref"] for item in release["pattern_membership"]),
            finalization.successor_release_ref,
            finalization.semantic_release_retest_ref,
        )
        if (
            attribution.semantic_impact_layer != "schema"
            or attribution.target_artifact_ref != reservation.schema_candidate_ref
            or attribution.target_artifact_hash != reservation.schema_candidate_hash
            or attribution.feedback_ref != continuation.feedback_ref
            or attribution.feedback_hash != continuation.feedback_hash
            or attribution.actor_ref != accepted["created_by"]
            or attribution.predecessor_refs != predecessor_refs
            or attribution.successor_refs != successor_refs
        ):
            raise ValueError("schema finalization attribution lineage mismatch")
        return finalization

    def replay_graph_intent_release_finalization(
        self,
        rerun_ref: str,
        *,
        rerun_hash: str,
        materializer: AcceptedSemanticGraphMaterializationService,
        _finalization_authority: object | None = None,
    ) -> GraphIntentReleaseFinalizationV1:
        if _finalization_authority is not _GRAPH_INTENT_RELEASE_FINALIZATION_AUTHORITY:
            raise PermissionError("GraphIntent release finalization replay is capability-sealed")
        if self.material_reruns is None:
            raise ValueError("material correction rerun store is unavailable")
        return GraphIntentReleaseFinalizationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get_verified(
            rerun_ref,
            rerun_hash=rerun_hash,
            material_reruns=self.material_reruns,
            materializer=materializer,
            attribution_service=SemanticImpactAttributionService(
                self.workspace_root, workspace_id=self.workspace_id
            ),
            adjudications=GraphIntentWitnessAdjudicationService(
                self.workspace_root,
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                material_reruns=self.material_reruns,
            ),
            _authority=_FINALIZATION_PUBLICATION_AUTHORITY,
        )

    def _create_route_only(
        self,
        *,
        feedback: SchemaLearningFeedbackV2,
        decision: SchemaDecision,
        candidate: CorpusGraphSchemaCandidate,
    ) -> dict[str, Any]:
        if (
            decision.schema_version != "schema_decision.v3"
            or candidate.schema_version != "corpus_graph_schema_candidate.v4"
            or self.discovery_service is None
            or self.material_reruns is None
        ):
            raise ValueError("route-only rerun requires governed v4/v3 authority")
        from services.relation_usage_grounding_service import (
            _create_relation_usage_grounding_authority,
        )
        from storage.human_grounding_store import HumanGroundingStore
        from semantier.knowledge_feature_packs.falkordb_backend import (
            InMemoryGraphBackend,
        )

        authority = _create_relation_usage_grounding_authority(
            self.workspace_root,
            organization_id=str(decision.source_binding["organization_id"]),
            workspace_id=self.workspace_id,
            profile_registry=self.discovery_service.profile_registry,
            profile_relation_type_proposal_store=(
                self.discovery_service.profile_relation_type_proposal_store
            ),
        )
        grounding_store = HumanGroundingStore(
            self.workspace_root / "eos.db",
            graph_backend=InMemoryGraphBackend(),
            _relation_usage_grounding_authority=authority,
        )
        grounding = grounding_store.get_relation_usage_grounding_for_feedback(
            feedback.feedback_id,
            organization_id=str(decision.source_binding["organization_id"]),
            workspace_id=self.workspace_id,
        )
        if (
            grounding["schema_decision_binding"]["schema_decision_ref"]
            != decision.schema_decision_id
            or grounding["schema_decision_binding"]["schema_decision_hash"]
            != decision.schema_decision_hash
            or grounding["schema_learning_feedback_ref"] != feedback.feedback_id
        ):
            raise ValueError("route-only Grounding authority binding mismatch")
        finding = next(
            (
                item
                for item in feedback.findings
                if item.get("required_route") == grounding["required_route"]
                and item.get("subject_ref")
                == grounding["subject_binding"]["subject_ref"]
            ),
            None,
        )
        if finding is None:
            raise ValueError("feedback does not preserve exact Grounding route")
        feedback_payload = self.feedback.put(feedback)
        feedback_hash = f"sha256:{content_hash(feedback_payload)}"
        material = relation_usage_route_material(
            {
                "feedback_ref": feedback.feedback_id,
                "feedback_hash": feedback_hash,
                "grounding_decision_ref": grounding["grounding_decision_ref"],
                "grounding_decision_hash": grounding["grounding_decision_hash"],
                "finding_kind": grounding["finding_kind"],
                "required_route": grounding["required_route"],
                "subject_ref": grounding["subject_binding"]["subject_ref"],
                "subject_hash": grounding["subject_binding"]["subject_hash"],
            }
        )
        outcome, reason = material_correction_outcome_reason(material)
        material_hash = f"sha256:{content_hash(material)}"
        gates = [
            item["selected_value"]
            for item in candidate.synthesis_decisions
            if item.get("decision_kind") == "exemplar_reconstruction"
            and isinstance(item.get("selected_value"), dict)
        ]
        if len(gates) != 1:
            raise ValueError("route-only baseline witness gate is missing")
        graph_binding = dict(candidate.graph_intent_binding or {})
        rerun = MaterialCorrectionRerun.create(
            {
                "organization_id": str(decision.source_binding["organization_id"]),
                "workspace_id": self.workspace_id,
                "outcome": outcome,
                "reason": reason,
                "feedback_ref": feedback.feedback_id,
                "feedback_hash": feedback_hash,
                "correction_material_hash": material_hash,
                "correction_material_ref": (
                    "artifacts/material-correction-material/"
                    + material_hash.removeprefix("sha256:")
                    + ".json"
                ),
                "before": self._witness_gate_state(
                    graph_intent_ref=str(graph_binding["graph_intent_ref"]),
                    graph_intent_hash=str(graph_binding["graph_intent_hash"]),
                    candidate=candidate,
                    decision=decision,
                    gate=gates[0],
                ),
                "after": None,
                "comparison_ref": None,
                "comparison_hash": None,
                "created_at": feedback.created_at,
                "created_by": feedback.created_by,
            }
        )
        self.material_reruns.put_route_only(
            rerun,
            correction_material=material,
        )
        return {
            "feedback": feedback_payload,
            "graph_intent": None,
            "candidate": None,
            "review_decision": None,
            "material_correction_rerun": rerun.as_dict(),
        }

    def _create_schema_successor(
        self,
        *,
        feedback: SchemaLearningFeedback | SchemaLearningFeedbackV2,
        decision: SchemaDecision,
        candidate: CorpusGraphSchemaCandidate,
        candidate_delta: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            decision.schema_version != "schema_decision.v3"
            or self.discovery_service is None
            or self.material_reruns is None
        ):
            raise ValueError(
                "Candidate.v4 schema correction requires governed Decision.v3 authority"
            )
        feedback_payload = self.feedback.put(feedback)
        if (
            GraphSchemaDiscoveryService.__dict__.get(
                "compile_schema_only_successor"
            )
            is not _TRUSTED_SCHEMA_ONLY_SUCCESSOR_DESCRIPTOR
        ):
            raise ValueError("schema-only correction authority was replaced")
        compiled = _TRUSTED_SCHEMA_ONLY_SUCCESSOR_DESCRIPTOR(
            self.discovery_service,
            predecessor_candidate=candidate,
            predecessor_decision=decision,
            candidate_delta=candidate_delta,
            clarification_ref=feedback.feedback_id,
            decided_by=(
                "system"
                if isinstance(feedback, SchemaLearningFeedbackV2)
                and feedback.operating_mode == "auto_pilot"
                else "user"
            ),
            actor_ref=feedback.created_by,
            decided_at=feedback.created_at,
            operating_mode=feedback.operating_mode,
            policy_ref=feedback.policy_ref,
            rationale_ref=feedback.rationale_ref,
        )
        successor = CorpusGraphSchemaCandidate.from_dict(compiled["candidate"])
        successor_decision = self.discovery_service.get_schema_decision_contract(
            compiled["decision"]["schema_decision_id"]
        )
        comparison = self.comparable_retest(
            organization_id=str(decision.source_binding["organization_id"]),
            baseline_schema_decision_ref=decision.schema_decision_id,
            successor_schema_decision_ref=successor_decision.schema_decision_id,
            requested_by=feedback.created_by,
        )
        graph_binding = dict(candidate.graph_intent_binding or {})
        material = schema_candidate_correction_material(candidate_delta)
        outcome, reason = material_correction_outcome_reason(material)
        material_hash = f"sha256:{content_hash(material)}"
        gate = compiled["exemplar_gate"]
        if outcome != "accepted" or reason != "schema_only_successor_recompiled_cleanly":
            raise ValueError("schema continuation requires a clean schema successor")
        material_ref = (
            "artifacts/material-correction-material/"
            + material_hash.removeprefix("sha256:")
            + ".json"
        )
        publish_immutable_json(
            self.workspace_root
            / "workspaces"
            / self.workspace_id
            / material_ref,
            material,
            conflict_error="MATERIAL_CORRECTION_MATERIAL_CONFLICT",
        )
        continuation = SchemaCorrectionContinuationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).publish(
            SchemaCorrectionContinuationV1.create(
                {
                    "schema_version": "schema_correction_continuation.v1",
                    "organization_id": str(decision.source_binding["organization_id"]),
                    "workspace_id": self.workspace_id,
                    "feedback_ref": feedback.feedback_id,
                    "feedback_hash": f"sha256:{content_hash(feedback.as_dict())}",
                    "correction_material_ref": material_ref,
                    "correction_material_hash": material_hash,
                    "graph_intent_ref": str(graph_binding["graph_intent_ref"]),
                    "graph_intent_hash": str(graph_binding["graph_intent_hash"]),
                    "predecessor_candidate_ref": candidate.candidate_id,
                    "predecessor_candidate_hash": candidate.candidate_hash,
                    "predecessor_schema_decision_ref": decision.schema_decision_id,
                    "predecessor_schema_decision_hash": decision.schema_decision_hash,
                    "successor_candidate_ref": successor.candidate_id,
                    "successor_candidate_hash": successor.candidate_hash,
                    "successor_schema_decision_ref": successor_decision.schema_decision_id,
                    "successor_schema_decision_hash": successor_decision.schema_decision_hash,
                    "positive_manifest_ref": gate["positive_manifest_ref"],
                    "positive_manifest_hash": gate["positive_manifest_hash"],
                    "positive_count": gate["positive_count"],
                    "hard_negative_manifest_ref": gate["hard_negative_manifest_ref"],
                    "hard_negative_manifest_hash": gate["hard_negative_manifest_hash"],
                    "hard_negative_count": gate["hard_negative_count"],
                    "comparison_ref": comparison["comparison_id"],
                    "comparison_hash": comparison["comparison_hash"],
                    "created_at": feedback.created_at,
                    "created_by": feedback.created_by,
                }
            ),
            _authority=_SCHEMA_CONTINUATION_PUBLICATION_AUTHORITY,
        )
        return {
            "outcome": "pending_materialization",
            "feedback": feedback_payload,
            "graph_intent": None,
            "candidate": successor.as_dict(),
            "review_decision": successor_decision.as_dict(),
            "comparison": comparison,
            "material_correction_rerun": None,
            "schema_correction_continuation": continuation.model_dump(mode="json"),
        }

    @staticmethod
    def _witness_gate_state(
        *,
        graph_intent_ref: str,
        graph_intent_hash: str,
        candidate: CorpusGraphSchemaCandidate,
        decision: SchemaDecision,
        gate: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "graph_intent_ref": graph_intent_ref,
            "graph_intent_hash": graph_intent_hash,
            "candidate_ref": candidate.candidate_id,
            "candidate_hash": candidate.candidate_hash,
            "schema_decision_ref": decision.schema_decision_id,
            "schema_decision_hash": decision.schema_decision_hash,
            "positive_manifest_ref": gate["positive_manifest_ref"],
            "positive_manifest_hash": gate["positive_manifest_hash"],
            "positive_count": gate["positive_count"],
            "hard_negative_manifest_ref": gate["hard_negative_manifest_ref"],
            "hard_negative_manifest_hash": gate["hard_negative_manifest_hash"],
            "hard_negative_count": gate["hard_negative_count"],
            "material_mismatches": gate["material_mismatches"],
        }


    def comparable_retest(
        self,
        *,
        organization_id: str,
        baseline_schema_decision_ref: str,
        successor_schema_decision_ref: str,
        requested_by: str,
        source_structure_validation_ref: str | None = None,
        source_structure_validation_hash: str | None = None,
        _source_structure_authority: object | None = None,
    ) -> dict[str, Any]:
        """Re-Extract the same pinned source under immutable A/B decisions."""
        if baseline_schema_decision_ref == successor_schema_decision_ref:
            raise ValueError("feedback retest decisions must be distinct")
        if self.discovery_service is not None:
            baseline = self.discovery_service.get_schema_decision_contract(
                baseline_schema_decision_ref
            )
            successor = self.discovery_service.get_schema_decision_contract(
                successor_schema_decision_ref
            )
        else:
            baseline = SchemaDecision.from_dict(
                self.decisions.get(baseline_schema_decision_ref)
            )
            successor = SchemaDecision.from_dict(
                self.decisions.get(successor_schema_decision_ref)
            )
        if successor.predecessor_schema_decision_ref != baseline.schema_decision_id:
            raise ValueError("successor decision does not reference baseline")
        validation_pins = (
            source_structure_validation_ref,
            source_structure_validation_hash,
        )
        if any(pin is None for pin in validation_pins) and any(
            pin is not None for pin in validation_pins
        ):
            raise ValueError("source-structure retest validation pins must be paired")
        if source_structure_validation_ref is None:
            if _source_structure_authority is not None:
                raise PermissionError("source-structure retest is sealed")
            if successor.source_binding != baseline.source_binding:
                raise ValueError("feedback retest source pins do not match")
        else:
            if _source_structure_authority is not _SOURCE_STRUCTURE_RETEST_AUTHORITY:
                raise PermissionError("source-structure retest is sealed")
            validation = SourceStructureSemanticValidationStore(
                self.workspace_root, workspace_id=self.workspace_id
            ).get(
                source_structure_validation_ref,
                validation_hash=str(source_structure_validation_hash),
            )
            baseline_source = baseline.source_binding
            successor_source = successor.source_binding
            baseline_intent_binding = dict(baseline.graph_intent_binding or {})
            successor_intent_binding = dict(successor.graph_intent_binding or {})
            successor_intent = self.graph_intent_store.get(
                str(successor_intent_binding.get("graph_intent_ref") or ""),
                graph_intent_hash=str(
                    successor_intent_binding.get("graph_intent_hash") or ""
                ),
            )
            if successor_intent.predecessor_graph_intent_ref is None:
                raise ValueError("source-structure retest GraphIntent proposal is missing")
            validated_proposal = self.graph_intent_store.get(
                successor_intent.predecessor_graph_intent_ref
            )
            predecessor_identity = self.graph_intent_store.source_registry.get_source_identity(
                source_identity_ref=str(baseline_source.get("source_identity_ref") or ""),
                organization_id=organization_id,
                workspace_id=self.workspace_id,
                lineage_id=str(baseline_source.get("lineage_id") or ""),
            )
            successor_identity = self.graph_intent_store.source_registry.get_source_identity(
                source_identity_ref=str(successor_source.get("source_identity_ref") or ""),
                organization_id=organization_id,
                workspace_id=self.workspace_id,
                lineage_id=str(successor_source.get("lineage_id") or ""),
            )
            if predecessor_identity is None or successor_identity is None:
                raise ValueError("source-structure retest SourceIdentity is missing")
            predecessor_identity_hash = "sha256:" + content_hash(
                predecessor_identity.model_dump(
                    mode="json", by_alias=True, exclude_none=False
                )
            )
            successor_identity_hash = "sha256:" + content_hash(
                successor_identity.model_dump(
                    mode="json", by_alias=True, exclude_none=False
                )
            )
            if (
                validation.organization_id != organization_id
                or validation.workspace_id != self.workspace_id
                or validation.created_by != requested_by
                or validation.predecessor_graph_intent_ref
                != baseline_intent_binding.get("graph_intent_ref")
                or validation.predecessor_graph_intent_hash
                != baseline_intent_binding.get("graph_intent_hash")
                or validation.successor_graph_intent_ref
                != validated_proposal.graph_intent_id
                or validation.successor_graph_intent_hash
                != validated_proposal.graph_intent_hash
                or validation.predecessor_source_identity_ref
                != baseline_source.get("source_identity_ref")
                or validation.predecessor_source_identity_hash
                != predecessor_identity_hash
                or validation.predecessor_canonical_source_ir_ref
                != baseline_source.get("canonical_source_ir_ref")
                or validation.predecessor_canonical_source_ir_hash
                != baseline_source.get("canonical_source_ir_hash")
                or validation.successor_source_identity_ref
                != successor_source.get("source_identity_ref")
                or validation.successor_source_identity_hash
                != successor_identity_hash
                or validation.successor_canonical_source_ir_ref
                != successor_source.get("canonical_source_ir_ref")
                or validation.successor_canonical_source_ir_hash
                != successor_source.get("canonical_source_ir_hash")
            ):
                raise ValueError("source-structure retest validation binding mismatch")
        service = GovernedExtractService(
            self.workspace_root,
            organization_id=organization_id,
            workspace_id=self.workspace_id,
            graph_intent_store=self.graph_intent_store,
            profile_registry=(
                self.discovery_service.profile_registry
                if self.discovery_service is not None
                else None
            ),
        )

        def execute(decision: SchemaDecision, label: str) -> dict[str, Any]:
            source = decision.source_binding
            return service.extract(
                ExtractRequest(
                    request_id=f"feedback-retest-{label}-{decision.schema_decision_id}",
                    organization_id=organization_id,
                    workspace_id=self.workspace_id,
                    lineage_id=str(source["lineage_id"]),
                    canonical_source_ir_ref=str(
                        source.get(
                            "canonical_source_ir_ref",
                            source.get("canonical_source_ref"),
                        )
                    ),
                    canonical_source_hash=str(source["source_hash"]),
                    schema_decision_ref=decision.schema_decision_id,
                    schema_decision_hash=decision.schema_decision_hash,
                    requested_by=requested_by,
                )
            )

        before = execute(baseline, "baseline")
        after = execute(successor, "successor")
        ordinary_comparison = build_schema_feedback_retest(
            baseline_decision=baseline.as_dict(),
            successor_decision=successor.as_dict(),
            baseline_extract=before,
            successor_extract=after,
        )
        if source_structure_validation_ref is None:
            comparison = ordinary_comparison
            comparison_directory = "feedback-retests"
        else:
            source_comparison_payload = dict(ordinary_comparison)
            source_comparison_payload.pop("comparison_id")
            source_comparison_payload.pop("comparison_hash")
            source_comparison_payload.pop("source_binding")
            source_comparison_payload["schema_version"] = "source_structure_retest.v1"
            source_comparison_payload["semantic_validation_ref"] = (
                source_structure_validation_ref
            )
            source_comparison_payload["semantic_validation_hash"] = (
                source_structure_validation_hash
            )
            source_comparison_payload["source_bindings"] = {
                "before": dict(baseline.source_binding),
                "after": dict(successor.source_binding),
            }
            comparison = SourceStructureRetestV1.create(
                source_comparison_payload
            ).model_dump(mode="json")
            comparison_directory = "source-structure-retests"
        target = (
            self.workspace_root
            / "workspaces"
            / self.workspace_id
            / "artifacts"
            / "schema-discovery"
            / comparison_directory
            / f"{comparison['comparison_id']}.json"
        )
        return publish_immutable_json(
            target,
            comparison,
            conflict_error="SCHEMA_FEEDBACK_RETEST_ARTIFACT_CONFLICT",
        )


__all__ = ["SchemaLearningService"]
