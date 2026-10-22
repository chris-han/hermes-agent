"""Authenticated channel-neutral facade for corpus graph-schema induction."""

from __future__ import annotations

from contracts.frame_mapping import FrameMapping

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any
from uuid import uuid4

from eos.artifact_hashing import content_hash
from contracts.governed_extract import ExtractRequest
from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.canonical_source_ir import content_hash as canonical_ir_content_hash
from contracts.graph_intent_brief import (
    GraphIntentBrief,
    GraphIntentReview,
    PrimaryRelationship,
    PreferredRepresentation,
    MaterialDimension,
    SourceRoleInterpretation,
)
from contracts.schema_decision import SchemaDecision
from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate
from contracts.schema_learning_feedback_v2 import SchemaLearningFeedbackV2
from contracts.schema_correction_continuation import SchemaCorrectionContinuationV1
from contracts.material_correction_rerun import schema_candidate_correction_material
from semantier.knowledge_feature_packs.source_elements import source_elements
from semantier.knowledge_feature_packs.falkordb_backend import GraphBackend
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.schema_correction_continuation_store import (
    SchemaCorrectionContinuationStore,
)

from .corpus_graph_schema_profile_registry import CorpusGraphSchemaProfileRegistry
from .graph_schema_discovery_service import GraphSchemaDiscoveryService
from .graph_schema_discovery_service import AcceptedWitnessGatePackage
from .graph_intent_discovery_service import GraphIntentDiscoveryService
from .graph_schema_llm_evidence_runner import (
    GraphSchemaLlmEvidenceRunner,
    resolve_runtime_llm_provider_model,
)
from .corpus_graph_schema_sample_executor import CorpusGraphSchemaSampleExecutor
from .governed_extract_service import GovernedExtractService
from .schema_learning_service import (
    SchemaLearningService,
    _GRAPH_INTENT_RELEASE_FINALIZATION_AUTHORITY,
    _SCHEMA_CORRECTION_RELEASE_FINALIZATION_AUTHORITY,
    _SOURCE_STRUCTURE_RETEST_AUTHORITY,
)
from .knowledge_relation_taxonomy_service import (
    DEFAULT_TAXONOMY_REF,
    KnowledgeRelationTaxonomyService,
)
from .graph_intent_witness_adjudication_service import (
    GraphIntentWitnessAdjudicationService,
    _WITNESS_ADJUDICATION_AUTHORITY,
)


class CorpusGraphSchemaCapabilityService:
    """Bind Web and Chat operations to one governed workspace authority scope."""

    capability_id = "corpus-graph-schema-induction"
    _EDITABLE_MATERIAL_SEMANTICS = frozenset(
        {
            "representation",
            "applicability",
            "topology",
            "direction",
            "grouping",
            "evidence_ownership",
            "extraction_behavior",
        }
    )

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        workspace_id: str,
        user_id: str,
        organization_id: str,
        profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
        graph_intent_llm_runner: GraphSchemaLlmEvidenceRunner | None = None,
        graph_intent_llm_provider_model: tuple[str, str] | None = None,
        source_registry: KnowledgeBuilderStore | None = None,
        graph_backend: GraphBackend | None = None,
        graph_intent_inference_enabled: bool = True,
    ) -> None:
        for label, value in (
            ("workspace_id", workspace_id),
            ("user_id", user_id),
            ("organization_id", organization_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"authenticated {label} is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.organization_id = organization_id
        self.graph_backend = graph_backend
        self.graph_intent_inference_enabled = graph_intent_inference_enabled
        self.profile_registry = profile_registry or CorpusGraphSchemaProfileRegistry()
        self.graph_intent_llm_provider_model = (
            (graph_intent_llm_provider_model
             if graph_intent_llm_provider_model is not None
             else resolve_runtime_llm_provider_model(workspace_root=self.workspace_root))
            if graph_intent_inference_enabled else None
        )
        self.graph_intent_llm_runner = graph_intent_llm_runner if graph_intent_inference_enabled else None
        if (
            self.graph_intent_llm_runner is None
            and self.graph_intent_llm_provider_model is not None
        ):
            self.graph_intent_llm_runner = GraphSchemaLlmEvidenceRunner(
                self.workspace_root,
                workspace_id=self.workspace_id,
            )
        self.discovery = GraphSchemaDiscoveryService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            profile_registry=self.profile_registry,
            organization_id=self.organization_id,
            source_registry=source_registry,
        )

    def profiles(self) -> dict[str, Any]:
        return {
            "capability": self.capability_id,
            "profiles": [
                {
                    "profileId": profile.profile_id,
                    "sourceKind": profile.source_kind,
                    "displayNameEn": profile.display_name_en,
                    "displayNameZh": profile.display_name_zh,
                }
                for profile in self.profile_registry.list_profiles()
            ],
        }

    def propose_graph_intent(
        self,
        *,
        lineage_id: str,
        source_identity_ref: str,
        canonical_source_ir_ref: str,
        profile_id: str,
        operating_mode: str,
        attempt_ref: str | None = None,
    ) -> dict[str, Any]:
        if not self.graph_intent_inference_enabled:
            raise ValueError("GRAPH_INTENT_INFERENCE_DISABLED")
        if operating_mode not in {"assisted", "auto_pilot"}:
            raise ValueError("operatingMode must be assisted or auto_pilot")
        store = self._graph_intent_store()
        source_identity = store.source_registry.get_source_identity(
            source_identity_ref=source_identity_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=lineage_id,
        )
        if source_identity is None:
            raise ValueError("governed source identity not found")
        canonical_path = self.discovery._resolve_canonical_source_ir_ref(  # noqa: SLF001
            canonical_source_ir_ref
        )
        canonical = CanonicalSourceDocument.model_validate(
            json.loads(canonical_path.read_text(encoding="utf-8"))
        )
        canonical_payload = canonical.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        canonical_hash = canonical_ir_content_hash(canonical_payload)
        discovery = GraphIntentDiscoveryService(
            self.workspace_root,
            intent_store=store,
            profile_registry=self.profile_registry,
            llm_evidence_runner=self.graph_intent_llm_runner,
        )
        provider_model = self.graph_intent_llm_provider_model
        brief = discovery.propose(
            source_identity=source_identity,
            lineage_id=lineage_id,
            canonical_source_ir_ref=canonical_source_ir_ref,
            canonical_source_ir_hash=canonical_hash,
            profile_id=profile_id,
            operating_mode=operating_mode,
            attempt_ref=attempt_ref,
            llm_provider=provider_model[0] if provider_model else None,
            llm_model=provider_model[1] if provider_model else None,
        )
        material_ids = self._material_question_ids(brief)
        fallback_reason = None
        effective_mode = operating_mode
        if operating_mode == "auto_pilot":
            if material_ids:
                effective_mode = "assisted"
                fallback_reason = "material_graph_intent_ambiguity"
            elif not brief.evidence_refs:
                effective_mode = "assisted"
                fallback_reason = "graph_intent_evidence_guard_failed"
            else:
                auto_review = GraphIntentReview(
                    state="accepted",
                    reviewed_by="system",
                    actor_ref="system:graph-intent-auto-pilot",
                    operating_mode="auto_pilot",
                    policy_ref="policy:graph-intent-auto-pilot.v1",
                    rationale_ref="rationale:graph-intent-guards-passed",
                    decision_evidence_refs=brief.evidence_refs,
                    fallback_reason=None,
                )
                brief = store.review(
                    brief.graph_intent_id,
                    review=auto_review,
                    source_identity=source_identity,
                )
        return self._graph_intent_envelope(
            brief,
            operating_mode=effective_mode,
            fallback_reason=fallback_reason,
        )

    def get_graph_intent(
        self, graph_intent_ref: str, *, graph_intent_hash: str | None = None
    ) -> dict[str, Any]:
        brief = self._graph_intent_store().get(
            graph_intent_ref, graph_intent_hash=graph_intent_hash
        )
        return self._graph_intent_envelope(
            brief,
            operating_mode=brief.review.operating_mode,
            fallback_reason=brief.review.fallback_reason,
        )

    @staticmethod
    def normalize_graph_intent_review(
        *,
        channel: str,
        state: str,
        operating_mode: str,
        relation_edit: dict[str, Any] | None,
        selected_question_ids: list[str],
    ) -> dict[str, Any]:
        if channel not in {"chat", "direct"}:
            raise ValueError("channel must be chat or direct")
        if state not in {"accepted", "edited", "rejected"}:
            raise ValueError("invalid GraphIntent review state")
        if operating_mode != "assisted":
            raise ValueError("user GraphIntent review must use assisted mode")
        if not isinstance(selected_question_ids, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in selected_question_ids
        ):
            raise ValueError("selectedQuestionIds must contain exact question ids")
        changes: dict[str, Any] = {}
        if state == "edited":
            if not isinstance(relation_edit, dict) or not {
                "primaryRelationship", "sourceRoleInterpretation",
            }.issubset(relation_edit) or set(relation_edit) - {
                "primaryRelationship", "sourceRoleInterpretation",
                "preferredRepresentation", "materialDimensions", "frameMapping",
            }:
                raise ValueError("edited GraphIntent requires a typed relationEdit")
            changes = {
                "primary_relationship": PrimaryRelationship.from_dict(
                    relation_edit["primaryRelationship"]
                ).as_dict(),
                "source_role_interpretation": SourceRoleInterpretation.from_dict(
                    relation_edit["sourceRoleInterpretation"]
                ).as_dict(),
            }
            if "frameMapping" in relation_edit:
                changes["frame_mapping"] = FrameMapping.from_dict(relation_edit["frameMapping"]).as_dict()
            if "preferredRepresentation" in relation_edit:
                changes["preferred_representation"] = PreferredRepresentation.from_dict(
                    relation_edit["preferredRepresentation"]
                ).as_dict()
            if "materialDimensions" in relation_edit:
                if not isinstance(relation_edit["materialDimensions"], list):
                    raise ValueError("materialDimensions must be a typed array")
                changes["material_dimensions"] = [
                    MaterialDimension.from_dict(item).as_dict()
                    for item in relation_edit["materialDimensions"]
                ]
        elif relation_edit is not None:
            raise ValueError("relationEdit is valid only for edited review")
        return {
            "state": state,
            "operating_mode": operating_mode,
            "selected_question_ids": list(dict.fromkeys(selected_question_ids)),
            "changes": changes,
        }

    def review_graph_intent(
        self,
        graph_intent_ref: str,
        *,
        graph_intent_hash: str,
        channel: str,
        state: str,
        operating_mode: str,
        relation_edit: dict[str, Any] | None,
        selected_question_ids: list[str],
    ) -> dict[str, Any]:
        normalized = self.normalize_graph_intent_review(
            channel=channel,
            state=state,
            operating_mode=operating_mode,
            relation_edit=relation_edit,
            selected_question_ids=selected_question_ids,
        )
        store = self._graph_intent_store()
        predecessor = store.get(graph_intent_ref, graph_intent_hash=graph_intent_hash)
        # This bounded review edits interpretation, never the admitted shape or
        # evidence identities. Existing proposal pins remain the provenance owner.
        changes = normalized["changes"]
        mapping = changes.get("frame_mapping")
        if mapping is not None:
            identity = store.source_registry.get_source_identity(
                source_identity_ref=predecessor.source_binding.source_identity_ref,
                organization_id=self.organization_id, workspace_id=self.workspace_id,
                lineage_id=predecessor.source_binding.lineage_id)
            if identity is None:
                raise ValueError("governed source identity not found")
            canonical = store.verify_source_binding(predecessor.source_binding, source_identity=identity)
            FrameMapping.from_dict(mapping).validate_review(predecessor, canonical)
            if store.successors(graph_intent_ref):
                raise ValueError("stale frame mapping proposal has a successor; reopen current review")
        representation = changes.get("preferred_representation")
        if representation is not None and any(
            representation[key] != predecessor.preferred_representation.as_dict()[key]
            for key in ("kind", "type_id")
        ):
            raise ValueError("review cannot change admitted representation identity")
        dimensions = changes.get("material_dimensions")
        if dimensions is not None and [
            (item["type_id"], item["evidence_refs"]) for item in dimensions
        ] != [
            (item.type_id, list(item.evidence_refs)) for item in predecessor.material_dimensions
        ]:
            raise ValueError("review cannot change material dimension identity or evidence")
        material_question_ids = self._material_question_ids(predecessor)
        selected_question_ids = normalized["selected_question_ids"]
        unknown_question_ids = sorted(
            set(selected_question_ids) - set(material_question_ids)
        )
        if unknown_question_ids:
            raise ValueError(
                "selectedQuestionIds must reference exact unresolved material question ids"
            )
        if normalized["state"] == "accepted" and material_question_ids:
            raise ValueError("cannot accept unresolved material GraphIntent questions")
        if normalized["state"] != "edited" and selected_question_ids:
            raise ValueError(
                "selectedQuestionIds are valid only with a typed edited review"
            )
        if selected_question_ids:
            changed_semantics = self._changed_relation_edit_semantics(
                predecessor,
                normalized["changes"],
            )
            selected_questions = {
                item.question_id: item
                for item in predecessor.unresolved_questions
                if item.question_id in selected_question_ids
            }
            for question_id in selected_question_ids:
                question = selected_questions[question_id]
                material_semantics = {
                    semantic
                    for semantic in question.affected_semantics
                    if GraphIntentDiscoveryService.is_material_question([semantic])
                }
                editable = self._EDITABLE_MATERIAL_SEMANTICS | ({"cardinality"} if mapping and mapping["question_id"] == question_id else set())
                unsupported = sorted(material_semantics - editable)
                if unsupported:
                    raise ValueError(
                        "selected material question is outside the typed relation "
                        f"edit surface: {question_id} ({', '.join(unsupported)})"
                    )
                unchanged = sorted(material_semantics - changed_semantics)
                if unchanged:
                    raise ValueError(
                        "typed edit does not resolve selected material question: "
                        f"{question_id} ({', '.join(unchanged)})"
                    )
            selected = set(selected_question_ids)
            normalized["changes"]["unresolved_questions"] = [
                item.as_dict()
                for item in predecessor.unresolved_questions
                if item.question_id not in selected
            ]
        if normalized["state"] == "edited" and not self._has_typed_relation_delta(
            predecessor,
            normalized["changes"],
        ):
            raise ValueError("edited GraphIntent requires a typed semantic change")
        binding = predecessor.source_binding
        source_identity = store.source_registry.get_source_identity(
            source_identity_ref=binding.source_identity_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=binding.lineage_id,
        )
        if source_identity is None:
            raise ValueError("governed source identity not found")
        review = GraphIntentReview(
            state=normalized["state"],
            reviewed_by="user",
            actor_ref=self.user_id,
            operating_mode=normalized["operating_mode"],
            policy_ref=None,
            rationale_ref=(
                "clarification:graph-intent:"
                + content_hash(
                    {
                        "graph_intent_ref": graph_intent_ref,
                        "graph_intent_hash": graph_intent_hash,
                        "selected_question_ids": selected_question_ids,
                        "changes": normalized["changes"],
                    }
                )
                if selected_question_ids
                else None
            ),
            decision_evidence_refs=predecessor.evidence_refs,
            fallback_reason=None,
        )
        successor = store.review(
            graph_intent_ref,
            review=review,
            source_identity=source_identity,
            changes=normalized["changes"],
        )
        return {
            **self._graph_intent_envelope(
                successor,
                operating_mode=successor.review.operating_mode,
                fallback_reason=successor.review.fallback_reason,
            ),
            "selectedQuestionIds": normalized["selected_question_ids"],
        }

    def _graph_intent_store(self):
        store = self.discovery.graph_intent_store
        if store is None:
            raise ValueError("authenticated organization scope is required")
        return store

    @staticmethod
    def _material_question_ids(brief: GraphIntentBrief) -> list[str]:
        return [
            item.question_id for item in brief.unresolved_questions if item.material
        ]

    @staticmethod
    def _has_typed_relation_delta(
        predecessor: GraphIntentBrief,
        changes: dict[str, Any],
    ) -> bool:
        return (
            ("frame_mapping" in changes and changes["frame_mapping"] != (predecessor.frame_mapping.as_dict() if predecessor.frame_mapping else None))
            or changes.get("primary_relationship")
            != predecessor.primary_relationship.as_dict()
            or changes.get("source_role_interpretation")
            != predecessor.source_role_interpretation.as_dict()
            or (
                "preferred_representation" in changes
                and changes["preferred_representation"]
                != predecessor.preferred_representation.as_dict()
            )
            or (
                "material_dimensions" in changes
                and changes["material_dimensions"]
                != [item.as_dict() for item in predecessor.material_dimensions]
            )
        )

    @staticmethod
    def _changed_relation_edit_semantics(
        predecessor: GraphIntentBrief,
        changes: dict[str, Any],
    ) -> set[str]:
        primary = dict(changes.get("primary_relationship") or {})
        roles = dict(changes.get("source_role_interpretation") or {})
        before_primary = predecessor.primary_relationship.as_dict()
        before_roles = predecessor.source_role_interpretation.as_dict()
        changed: set[str] = set()
        if "frame_mapping" in changes:
            changed.update({"grouping", "cardinality", "representation"})
        if any(
            primary.get(field) != before_primary[field]
            for field in ("type_id", "source_types", "target_types")
        ):
            changed.update({"topology", "extraction_behavior"})
        if primary.get("direction") != before_primary["direction"]:
            changed.update({"direction", "extraction_behavior"})
        if roles != before_roles:
            changed.update({"grouping", "evidence_ownership", "extraction_behavior"})
        if (
            "preferred_representation" in changes
            and changes["preferred_representation"]
            != predecessor.preferred_representation.as_dict()
        ):
            changed.update({"representation", "extraction_behavior"})
        if (
            "material_dimensions" in changes
            and changes["material_dimensions"]
            != [item.as_dict() for item in predecessor.material_dimensions]
        ):
            changed.add("extraction_behavior")
            before_dimensions = {
                item.type_id: item.description for item in predecessor.material_dimensions
            }
            if any(
                item["type_id"] in {"Applicability", "ApplicabilityScope"}
                and item["description"] != before_dimensions[item["type_id"]]
                for item in changes["material_dimensions"]
            ):
                changed.add("applicability")
        return changed

    def _graph_intent_profile_id(self, brief: GraphIntentBrief) -> str:
        return str(
            GraphIntentDiscoveryService(
                self.workspace_root,
                intent_store=self._graph_intent_store(),
                profile_registry=self.profile_registry,
            ).load_proposal_evidence(brief)["profile_id"]
        )

    def _graph_intent_envelope(
        self,
        brief: GraphIntentBrief,
        *,
        operating_mode: str,
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "status": brief.review.state,
            "graphIntent": brief.as_dict(),
            "pins": {
                "canonicalSourceIrRef": brief.source_binding.canonical_source_ir_ref,
                "canonicalSourceIrHash": brief.source_binding.canonical_source_ir_hash,
                "graphIntentRef": brief.graph_intent_id,
                "graphIntentHash": brief.graph_intent_hash,
            },
            "operatingMode": operating_mode,
            "profileId": self._graph_intent_profile_id(brief),
            "fallbackReason": fallback_reason,
            "unresolvedMaterialQuestionIds": self._material_question_ids(brief),
        }

    def induce(
        self,
        *,
        lineage_id: str | None = None,
        relative_path: str | None = None,
        canonical_source_ir_ref: str | None = None,
        source_ref: str | None = None,
        llm_proposal_evidence_id: str | None = None,
        profile_id: str | None,
        intended_use: str | None = None,
        competency_questions: list[dict[str, str]] | None = None,
        graph_intent_ref: str | None = None,
        graph_intent_hash: str | None = None,
    ) -> dict[str, Any]:
        graph_intent_shape = any(
            item is not None for item in (graph_intent_ref, graph_intent_hash)
        )
        legacy_shape = any(
            item is not None
            for item in (
                lineage_id,
                relative_path,
                source_ref,
                llm_proposal_evidence_id,
                intended_use,
                competency_questions,
            )
        )
        if graph_intent_shape:
            if legacy_shape:
                raise ValueError(
                    "mixed GraphIntent and legacy induction request shapes are forbidden"
                )
            if not all(
                isinstance(item, str) and item.strip()
                for item in (
                    graph_intent_ref,
                    graph_intent_hash,
                    canonical_source_ir_ref,
                    profile_id,
                )
            ):
                raise ValueError(
                    "GraphIntent induction requires graphIntentRef, graphIntentHash, "
                    "canonicalSourceIrRef, and exact profileId"
                )
            result = self.discovery.induce_graph_intent_schema_candidate(
                graph_intent_ref=str(graph_intent_ref),
                graph_intent_hash=str(graph_intent_hash),
                canonical_source_ir_ref=str(canonical_source_ir_ref),
                profile_id=str(profile_id),
                actor_id=self.user_id,
            )
            return self._envelope(
                result["decision"],
                candidate=result["candidate"],
                run=result["run"],
                status="candidate",
            )
        if not isinstance(intended_use, str) or not intended_use.strip():
            raise ValueError("intendedUse is required")
        if not isinstance(competency_questions, list):
            raise ValueError("competencyQuestions must be an array of objects")
        if canonical_source_ir_ref:
            normalized_lineage = str(lineage_id or "").strip()
            if not normalized_lineage:
                raise ValueError("lineageId is required for governed CanonicalSourceIR")
            profile = self.profile_registry.resolve(str(profile_id or ""))
            if profile.source_kind != "docx":
                raise ValueError("selected profile does not support DOCX sources")
            result = self.discovery.induce_corpus_schema_candidate(
                lineage_id=normalized_lineage,
                experiment_id=f"corpus-schema-{uuid4().hex}",
                actor_id=self.user_id,
                intended_use=intended_use.strip(),
                canonical_source_ir_ref=canonical_source_ir_ref,
                source_ref=source_ref or f"workspace://{canonical_source_ir_ref}",
                profile_id=profile.profile_id,
                competency_questions=tuple(competency_questions),
                llm_proposal_evidence_id=llm_proposal_evidence_id,
            )
            return self._envelope(
                result["decision"],
                candidate=result["candidate"],
                run=result["run"],
                status="candidate",
            )
        # Legacy upload compatibility keeps the explicit governed profile.
        # Resolve it before touching the source so path-like identifiers can
        # never influence filesystem selection.
        profile = self.profile_registry.resolve(str(profile_id or ""))
        if profile.source_kind != "docx":
            raise ValueError("selected profile does not support DOCX sources")
        source_path, normalized_relative = self._governed_docx(relative_path)
        result = self.discovery.induce_corpus_schema_candidate(
            lineage_id=(lineage_id or f"legacy-upload:{normalized_relative}"),
            experiment_id=f"corpus-schema-{uuid4().hex}",
            actor_id=self.user_id,
            intended_use=intended_use.strip(),
            source_path=source_path,
            source_ref=f"workspace://{normalized_relative}",
            profile_id=profile.profile_id,
            competency_questions=tuple(competency_questions),
        )
        return self._envelope(
            result["decision"],
            candidate=result["candidate"],
            run=result["run"],
            status="candidate",
        )

    def list(self, *, limit: int = 20) -> dict[str, Any]:
        return {
            "capability": self.capability_id,
            "candidates": [
                self._review_envelope(decision, status="candidate")
                for decision in self.discovery.list_corpus_schema_candidates(
                    limit=limit
                )
            ],
        }

    def get(self, decision_id: str) -> dict[str, Any]:
        decision = self.discovery.store.get_decision(decision_id)
        self._require_corpus_candidate(decision)
        return self._review_envelope(decision, status="candidate")

    def _review_envelope(
        self, review_decision: dict[str, Any], *, status: str
    ) -> dict[str, Any]:
        """Replay an exact v4/v3 promotion when this review already produced one."""

        matches: list[SchemaDecision] = []
        taxonomy_service = KnowledgeRelationTaxonomyService(
            repository_root=Path(__file__).resolve().parents[2],
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.discovery.profile_relation_type_proposal_store
            ),
        )
        root = self.discovery.schema_decision_store.root
        if root.is_dir():
            for path in sorted(root.glob("*/schema_decision.v3.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("candidate_binding", {}).get("review_ref") == (
                    review_decision["decision_id"]
                ):
                    matches.append(
                        KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
                            taxonomy_service, payload
                        )
                    )
        if len(matches) > 1:
            raise ValueError("review resolves multiple Decision.v3 artifacts")
        if not matches:
            return self._envelope(review_decision, status=status)
        decision = matches[0]
        candidate = self.discovery.candidate_store.get_by_id(
            str(decision.candidate_binding["candidate_id"])
        )
        graph_intent = self._graph_intent_store().get(
            str(decision.graph_intent_binding["graph_intent_ref"]),
            graph_intent_hash=str(
                decision.graph_intent_binding["graph_intent_hash"]
            ),
        )
        law_gates = [item["selected_value"] for item in candidate.get("synthesis_decisions", [])
            if item.get("decision_kind") == "exemplar_reconstruction"
            and item.get("selected_value", {}).get("schema_version") == "accepted_law_provision_gate.v1"]
        if law_gates:
            if len(law_gates) != 1:
                raise ValueError("ambiguous law provision qualification")
            gate, _, selection = GraphSchemaDiscoveryService._law_provision_compilation_basis(
                self.discovery, graph_intent, law_gates[0]["positive_manifest_hash"],
                law_gates[0]["hard_negative_manifest_hash"])
            if gate != law_gates[0]:
                raise ValueError("persisted law provision gate mismatch")
        else:
            selection = GraphSchemaDiscoveryService.registered_witness_selection(
                self.discovery, graph_intent
            )
        return self._envelope(
            review_decision,
            candidate=candidate,
            status="decided",
            schema_decision=decision.as_dict(),
            witness_selection=selection,
        )

    def review(
        self,
        decision_id: str,
        *,
        state: str,
        note: str = "",
        edited_candidate: dict[str, Any] | None = None,
        law_provision_manifest_pins: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        existing = self.discovery.store.get_decision(decision_id)
        self._require_corpus_candidate(existing)
        if state not in {"accepted", "edited", "rejected"}:
            raise ValueError("invalid review state")
        if state == "edited" and not isinstance(edited_candidate, dict):
            raise ValueError(
                "editedCandidate must be a complete candidate object for edited review"
            )
        if state != "edited" and edited_candidate is not None:
            raise ValueError("editedCandidate is only valid for edited review")
        normalized_note = note.strip() if isinstance(note, str) else ""
        existing_review = dict(existing.get("human_review") or {})
        existing_state = str(existing_review.get("state") or "unreviewed")
        terminal_retry = existing_state != "unreviewed"
        normalized_edit = None
        if state == "edited":
            original_candidate = existing["proposed_semantics"]["machine_readable"][
                "candidate"
            ]
            self.discovery._validate_edited_candidate_source_binding(
                original_candidate, edited_candidate
            )
            normalized_edit = self.discovery._authoritative_edited_candidate(
                edited_candidate
            )
        if terminal_retry:
            identical = (
                existing_state == state
                and existing_review.get("actor_id") == self.user_id
                and existing_review.get("note") == normalized_note
                and (
                    existing_review.get("edited_payload") == normalized_edit
                    if state == "edited"
                    else edited_candidate is None
                )
            )
            if not identical:
                raise ValueError(
                    "human review is terminal; transition or content drift rejected"
                )
            decision = existing
        else:
            decision = self.discovery.record_human_review(
                decision_id,
                state=state,
                actor_id=self.user_id,
                note=normalized_note,
                edited_candidate=edited_candidate,
            )
        self._require_corpus_candidate(decision)
        if decision["human_review"]["state"] not in {"accepted", "edited"}:
            return self._envelope(decision, status="reviewed")
        if terminal_retry:
            replayed = self._review_envelope(decision, status="reviewed")
            if "schemaDecision" in replayed:
                return replayed
        reviewed_payload = (
            decision["human_review"].get("edited_payload")
            if decision["human_review"]["state"] == "edited"
            else decision["proposed_semantics"]["machine_readable"]["candidate"]
        )
        reviewed_candidate = CorpusGraphSchemaCandidate.from_dict(reviewed_payload)
        if reviewed_candidate.schema_version == "corpus_graph_schema_candidate.v3":
            return self._compile_reviewed_candidate_v4(
                decision, reviewed_candidate, law_provision_manifest_pins=law_provision_manifest_pins
            )
        return self._envelope(decision, status="reviewed")

    def _compile_reviewed_candidate_v4(
        self,
        review_decision: dict[str, Any],
        candidate_v3: CorpusGraphSchemaCandidate,
        *, law_provision_manifest_pins: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Promote a reviewed v3 draft only through the sealed 32+6 compiler."""

        binding = dict(candidate_v3.graph_intent_binding or {})
        graph_intent = self._graph_intent_store().get(
            str(binding.get("graph_intent_ref") or ""),
            graph_intent_hash=str(binding.get("graph_intent_hash") or ""),
        )
        witness_packages = () if law_provision_manifest_pins is not None else (
            GraphSchemaDiscoveryService.resolve_registered_witness_packages(self.discovery, graph_intent)
        )
        proposal = self.discovery.profile_relation_type_proposal_store.find_for_graph_intent(
            graph_intent_ref=graph_intent.graph_intent_id,
            graph_intent_hash=graph_intent.graph_intent_hash,
            organization_id=self.organization_id,
        )
        taxonomy_service = KnowledgeRelationTaxonomyService(
            repository_root=Path(__file__).resolve().parents[2],
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.discovery.profile_relation_type_proposal_store
            ),
        )
        taxonomy = KnowledgeRelationTaxonomyService.load_canonical(
            taxonomy_service, DEFAULT_TAXONOMY_REF
        )
        result = GraphSchemaDiscoveryService.compile_exemplar_backed_relation_decision(
            self.discovery,
            graph_intent=graph_intent,
            candidate_v3=candidate_v3,
            witness_packages=witness_packages,
            law_provision_manifest_pins=law_provision_manifest_pins,
            taxonomy_ref=taxonomy.taxonomy_ref,
            taxonomy_hash=taxonomy.taxonomy_hash,
            decided_by="user",
            actor_ref=self.user_id,
            decided_at=str(review_decision["human_review"]["reviewed_at"]),
            relation_type_proposal_ref=(
                proposal["proposal_ref"] if proposal is not None else None
            ),
            relation_type_proposal_hash=(
                proposal["proposal_hash"] if proposal is not None else None
            ),
            review_ref=str(review_decision["decision_id"]),
        )
        if result.get("outcome") != "accepted":
            raise ValueError("reviewed Candidate v3 did not produce Candidate v4")
        selection = result["witness_selection"]
        return self._envelope(
            review_decision,
            candidate=result["candidate"],
            status="decided",
            schema_decision=result["decision"],
            witness_selection={
                "sourceSemanticWitnessSetRef": selection[
                    "source_semantic_witness_set_ref"
                ],
                "sourceSemanticWitnessSetHash": selection[
                    "source_semantic_witness_set_hash"
                ],
                "selectedSourceSemanticWitnessRef": selection[
                    "selected_source_semantic_witness_ref"
                ],
                "selectedSourceSemanticWitnessHash": selection[
                    "selected_source_semantic_witness_hash"
                ],
            },
        )

    def project(self, decision_id: str) -> dict[str, Any]:
        decision = self.discovery.store.get_decision(decision_id)
        self._require_corpus_candidate(decision)
        projection = self.discovery.accepted_corpus_schema_projection(decision_id)
        return {
            **self._envelope(decision, status="projected"),
            "projection": projection,
        }

    def decide(
        self,
        decision_id: str,
        *,
        operating_mode: str,
    ) -> dict[str, Any]:
        """Read the decision already sealed by governed review; never create authority."""
        if operating_mode not in {"assisted", "auto_pilot"}:
            raise ValueError("operatingMode must be assisted or auto_pilot")
        review_decision = self.discovery.store.get_decision(decision_id)
        self._require_corpus_candidate(review_decision)
        review = review_decision.get("human_review") or {}
        candidate = (
            review.get("edited_payload")
            if review.get("state") == "edited"
            else review_decision["proposed_semantics"]["machine_readable"]["candidate"]
        )
        if candidate.get("schema_version") == "corpus_graph_schema_candidate.v3":
            replayed = self._review_envelope(review_decision, status="decided")
            if "schemaDecision" in replayed:
                return replayed
            raise ValueError(
                "Candidate v3 is review-only; accepted new writes require the "
                "32+6 Candidate v4/Decision v3 review path"
            )
        raise ValueError(
            "Historical candidates are review-only; accepted new writes require the "
            "32+6 Candidate v4/Decision v3 review path"
        )

    def extract(self, schema_decision_id: str) -> dict[str, Any]:
        decision = self.discovery.get_schema_decision(schema_decision_id)
        source = decision["source_binding"]
        canonical_source_ref = source.get(
            "canonical_source_ir_ref", source.get("canonical_source_ref")
        )
        if not isinstance(canonical_source_ref, str) or not canonical_source_ref:
            raise ValueError("SchemaDecision canonical source binding is missing")
        request = ExtractRequest(
            request_id=f"extract-request-{uuid4().hex}",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=str(source["lineage_id"]),
            canonical_source_ir_ref=canonical_source_ref,
            canonical_source_hash=str(source["source_hash"]),
            schema_decision_ref=schema_decision_id,
            schema_decision_hash=str(decision["schema_decision_hash"]),
            requested_by=self.user_id,
        )
        return GovernedExtractService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_intent_store=self.discovery.graph_intent_store,
            profile_registry=self.profile_registry,
        ).extract(request)

    def read_extract_result(self, extract_run_id: str) -> dict[str, Any]:
        return GovernedExtractService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            profile_registry=self.profile_registry,
        ).read_result(extract_run_id).as_dict()

    def evaluate_grounding_gate(self, extract_run_id: str) -> dict[str, Any]:
        return GovernedExtractService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            profile_registry=self.profile_registry,
        ).evaluate_grounding_gate(extract_run_id)

    def record_schema_feedback(
        self,
        schema_decision_id: str,
        *,
        feedback: dict[str, Any],
        candidate_delta: dict[str, Any] | None,
        graph_intent_changes: dict[str, Any] | None = None,
        witness_packages: tuple[AcceptedWitnessGatePackage, ...] | None = None,
    ) -> dict[str, Any]:
        return SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        ).record_and_create_successor(
            schema_decision_ref=schema_decision_id,
            feedback_content=feedback,
            candidate_delta=candidate_delta,
            graph_intent_changes=graph_intent_changes,
            witness_packages=witness_packages,
        )

    def replay_schema_correction_continuation(
        self, continuation_ref: str, *, continuation_hash: str
    ) -> dict[str, Any]:
        continuation = SchemaCorrectionContinuationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(continuation_ref, continuation_hash=continuation_hash)
        if continuation.organization_id != self.organization_id:
            raise ValueError("schema correction continuation organization mismatch")
        learning = SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        )
        feedback = learning.feedback.get(continuation.feedback_ref)
        if "sha256:" + content_hash(feedback) != continuation.feedback_hash:
            raise ValueError("schema correction continuation feedback mismatch")
        intent = self._graph_intent_store().get(
            continuation.graph_intent_ref,
            graph_intent_hash=continuation.graph_intent_hash,
        )
        predecessor = CorpusGraphSchemaCandidate.from_dict(
            learning.candidates.get_by_hash(continuation.predecessor_candidate_hash)
        )
        successor = CorpusGraphSchemaCandidate.from_dict(
            learning.candidates.get_by_hash(continuation.successor_candidate_hash)
        )
        if (
            predecessor.candidate_id != continuation.predecessor_candidate_ref
            or successor.candidate_id != continuation.successor_candidate_ref
            or predecessor.graph_intent_binding != successor.graph_intent_binding
            or predecessor.graph_intent_binding.get("graph_intent_ref")
            != intent.graph_intent_id
        ):
            raise ValueError("schema correction continuation candidate lineage mismatch")
        predecessor_decision = self.discovery.get_schema_decision_contract(
            continuation.predecessor_schema_decision_ref
        )
        successor_decision = self.discovery.get_schema_decision_contract(
            continuation.successor_schema_decision_ref
        )
        if (
            predecessor_decision.schema_decision_hash
            != continuation.predecessor_schema_decision_hash
            or successor_decision.schema_decision_hash
            != continuation.successor_schema_decision_hash
            or predecessor_decision.candidate_binding.get("candidate_id")
            != predecessor.candidate_id
            or predecessor_decision.candidate_binding.get("candidate_hash")
            != predecessor.candidate_hash
            or successor_decision.candidate_binding.get("candidate_id")
            != successor.candidate_id
            or successor_decision.candidate_binding.get("candidate_hash")
            != successor.candidate_hash
            or successor.predecessor_candidate_ref != predecessor.candidate_id
            or successor_decision.predecessor_schema_decision_ref
            != predecessor_decision.schema_decision_id
        ):
            raise ValueError("schema correction continuation decision lineage mismatch")
        material_root = (
            self.workspace_root / "workspaces" / self.workspace_id
            / "artifacts/material-correction-material"
        ).resolve()
        material_path = (
            self.workspace_root / "workspaces" / self.workspace_id
            / continuation.correction_material_ref
        ).resolve()
        if material_path.parent != material_root:
            raise ValueError("schema correction material is outside governed root")
        material = json.loads(material_path.read_text(encoding="utf-8"))
        if (
            "sha256:" + content_hash(material) != continuation.correction_material_hash
            or schema_candidate_correction_material(material.get("changes")) != material
        ):
            raise ValueError("schema correction continuation material mismatch")
        successor_payload = successor.as_dict()
        if any(
            successor_payload.get(field) != value
            for field, value in material["changes"].items()
        ):
            raise ValueError("schema correction continuation material was not applied")
        feedback_contract = SchemaLearningFeedbackV2.from_dict(feedback)
        expected_source = dict(predecessor_decision.source_binding)
        exact_intent_binding = {
            "graph_intent_ref": intent.graph_intent_id,
            "graph_intent_hash": intent.graph_intent_hash,
        }
        if (
            feedback_contract.proposed_candidate_delta_ref
            != "sha256:" + content_hash(material["changes"])
            or feedback_contract.source_binding != expected_source
            or feedback_contract.schema_decision_ref
            != predecessor_decision.schema_decision_id
            or feedback_contract.candidate_ref != predecessor.candidate_id
            or feedback_contract.created_by != continuation.created_by
            or feedback_contract.created_at != continuation.created_at
            or feedback_contract.feedback_scope != "schema_understanding"
            or feedback_contract.attribution.get("graph_intent") is not False
            or feedback_contract.attribution.get("ontology_induction") is not True
            or any(
                feedback_contract.attribution.get(cause) is not False
                for cause in (
                    "source_structure",
                    "synthesis",
                    "extraction_plan",
                    "execution",
                )
            )
        ):
            raise ValueError("schema correction continuation feedback binding mismatch")
        if any(
            binding.get(field) != value
            for binding in (
                predecessor.graph_intent_binding,
                successor.graph_intent_binding,
                predecessor_decision.graph_intent_binding,
                successor_decision.graph_intent_binding,
            )
            for field, value in exact_intent_binding.items()
        ):
            raise ValueError("schema correction continuation GraphIntent binding mismatch")
        expected_gate = {
            "positive_manifest_ref": continuation.positive_manifest_ref,
            "positive_manifest_hash": continuation.positive_manifest_hash,
            "positive_count": continuation.positive_count,
            "hard_negative_manifest_ref": continuation.hard_negative_manifest_ref,
            "hard_negative_manifest_hash": continuation.hard_negative_manifest_hash,
            "hard_negative_count": continuation.hard_negative_count,
            "material_mismatches": [],
        }
        verified_gate = self.discovery.verify_persisted_accepted_witness_gate(
            graph_intent=intent,
            positive_manifest_ref=continuation.positive_manifest_ref,
            positive_manifest_hash=continuation.positive_manifest_hash,
            hard_negative_manifest_ref=continuation.hard_negative_manifest_ref,
            hard_negative_manifest_hash=continuation.hard_negative_manifest_hash,
        )
        if any(
            verified_gate.get(field) != value
            for field, value in expected_gate.items()
        ):
            raise ValueError("schema correction continuation witness authority mismatch")
        for item in (predecessor, successor):
            gates = [
                decision["selected_value"]
                for decision in item.synthesis_decisions
                if decision.get("decision_kind") == "exemplar_reconstruction"
                and isinstance(decision.get("selected_value"), dict)
            ]
            if len(gates) != 1 or any(
                gates[0].get(field) != value
                for field, value in expected_gate.items()
            ):
                raise ValueError("schema correction continuation witness gate mismatch")
        recomputed = learning.comparable_retest(
            organization_id=self.organization_id,
            baseline_schema_decision_ref=predecessor_decision.schema_decision_id,
            successor_schema_decision_ref=successor_decision.schema_decision_id,
            requested_by=continuation.created_by,
        )
        if (
            recomputed["comparison_id"] != continuation.comparison_ref
            or recomputed["comparison_hash"] != continuation.comparison_hash
        ):
            raise ValueError("schema correction continuation comparison mismatch")
        return continuation.model_dump(mode="json")

    def finalize_schema_correction(
        self,
        continuation_ref: str,
        *,
        continuation_hash: str,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        if self.graph_backend is None:
            raise ValueError("schema finalization requires configured graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )

        verified = self.replay_schema_correction_continuation(
            continuation_ref, continuation_hash=continuation_hash
        )
        learning = SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        )
        return learning.finalize_schema_correction(
            SchemaCorrectionContinuationV1.model_validate(verified),
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
            candidate_graph_id=candidate_graph_id,
            human_label_event_ids=human_label_event_ids,
            materializer=AcceptedSemanticGraphMaterializationService(
                runtime_root=self.workspace_root,
                workspace_root=self.workspace_root,
                db_path=self.workspace_root / "eos.db",
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                graph_backend=self.graph_backend,
                profile_registry=self.profile_registry,
            ),
            _finalization_authority=(
                _SCHEMA_CORRECTION_RELEASE_FINALIZATION_AUTHORITY
            ),
        )

    def replay_schema_correction_release_finalization(
        self, continuation_ref: str, *, continuation_hash: str
    ):
        if self.graph_backend is None:
            raise ValueError("schema finalization replay requires graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )

        verified = self.replay_schema_correction_continuation(
            continuation_ref, continuation_hash=continuation_hash
        )
        return SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        ).replay_schema_correction_release_finalization(
            SchemaCorrectionContinuationV1.model_validate(verified),
            materializer=AcceptedSemanticGraphMaterializationService(
                runtime_root=self.workspace_root,
                workspace_root=self.workspace_root,
                db_path=self.workspace_root / "eos.db",
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                graph_backend=self.graph_backend,
                profile_registry=self.profile_registry,
            ),
            _finalization_authority=(
                _SCHEMA_CORRECTION_RELEASE_FINALIZATION_AUTHORITY
            ),
        )

    def finalize_canonical_pattern_correction(
        self,
        *,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_id: str,
    ) -> dict[str, Any]:
        if self.graph_backend is None:
            raise ValueError("canonical-pattern correction requires graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )
        from services.canonical_pattern_correction_service import (
            CanonicalPatternCorrectionService,
            _CANONICAL_PATTERN_FINALIZATION_CAPABILITY,
        )

        materializer = AcceptedSemanticGraphMaterializationService(
            runtime_root=self.workspace_root,
            workspace_root=self.workspace_root,
            db_path=self.workspace_root / "eos.db",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_backend=self.graph_backend,
            profile_registry=self.profile_registry,
        )
        return CanonicalPatternCorrectionService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            discovery_service=self.discovery,
            materializer=materializer,
        ).finalize(
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
            candidate_graph_id=candidate_graph_id,
            human_label_event_id=human_label_event_id,
            requesting_actor_ref=self.user_id,
            _authority=_CANONICAL_PATTERN_FINALIZATION_CAPABILITY,
        )

    def create_source_structure_successor(
        self,
        *,
        predecessor_graph_intent_ref: str,
        predecessor_graph_intent_hash: str,
        corrected_canonical_source_ir: dict[str, Any],
    ) -> dict[str, Any]:
        from services.source_structure_successor_service import (
            SourceStructureSuccessorService,
            _SOURCE_STRUCTURE_SUCCESSOR_CAPABILITY,
        )

        return SourceStructureSuccessorService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_intent_store=self._graph_intent_store(),
        ).create(
            predecessor_graph_intent_ref=predecessor_graph_intent_ref,
            predecessor_graph_intent_hash=predecessor_graph_intent_hash,
            corrected_canonical_source_ir=corrected_canonical_source_ir,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            requesting_actor_ref=self.user_id,
            _authority=_SOURCE_STRUCTURE_SUCCESSOR_CAPABILITY,
        )

    def compile_source_structure_successor(
        self,
        *,
        predecessor_candidate_ref: str,
        predecessor_candidate_hash: str,
        predecessor_schema_decision_ref: str,
        predecessor_schema_decision_hash: str,
        successor_graph_intent_ref: str,
        successor_graph_intent_hash: str,
        semantic_validation_ref: str,
        semantic_validation_hash: str,
    ) -> dict[str, Any]:
        from services.graph_schema_discovery_service import (
            _SOURCE_STRUCTURE_SUCCESSOR_COMPILATION_AUTHORITY,
        )

        predecessor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.discovery.candidate_store.get_by_id(predecessor_candidate_ref)
        )
        if predecessor_candidate.candidate_hash != predecessor_candidate_hash:
            raise ValueError("source-structure predecessor candidate hash mismatch")
        predecessor_decision = self.discovery.get_schema_decision_contract(
            predecessor_schema_decision_ref
        )
        if predecessor_decision.schema_decision_hash != predecessor_schema_decision_hash:
            raise ValueError("source-structure predecessor decision hash mismatch")
        successor_intent = self._graph_intent_store().get(
            successor_graph_intent_ref,
            graph_intent_hash=successor_graph_intent_hash,
        )
        return self.discovery.compile_source_structure_successor(
            predecessor_candidate=predecessor_candidate,
            predecessor_decision=predecessor_decision,
            successor_graph_intent=successor_intent,
            semantic_validation_ref=semantic_validation_ref,
            semantic_validation_hash=semantic_validation_hash,
            clarification_ref=semantic_validation_ref,
            actor_ref=self.user_id,
            _authority=_SOURCE_STRUCTURE_SUCCESSOR_COMPILATION_AUTHORITY,
        )

    def replay_canonical_pattern_correction(
        self, reservation_ref: str, *, reservation_hash: str
    ):
        if self.graph_backend is None:
            raise ValueError("canonical-pattern replay requires graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )
        from services.canonical_pattern_correction_service import (
            CanonicalPatternCorrectionService,
            _CANONICAL_PATTERN_FINALIZATION_CAPABILITY,
        )

        materializer = AcceptedSemanticGraphMaterializationService(
            runtime_root=self.workspace_root,
            workspace_root=self.workspace_root,
            db_path=self.workspace_root / "eos.db",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_backend=self.graph_backend,
            profile_registry=self.profile_registry,
        )
        return CanonicalPatternCorrectionService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            discovery_service=self.discovery,
            materializer=materializer,
        ).replay(
            reservation_ref,
            reservation_hash=reservation_hash,
            _authority=_CANONICAL_PATTERN_FINALIZATION_CAPABILITY,
        )

    def record_graph_intent_correction(
        self,
        schema_decision_id: str,
        *,
        schema_decision_hash: str,
        graph_intent_ref: str,
        graph_intent_hash: str,
        channel: str,
        graph_intent_changes: dict[str, Any],
        selected_question_ids: list[str],
    ) -> dict[str, Any]:
        """Normalize a Studio correction while deriving every authority pin."""

        decision = self.discovery.get_schema_decision_contract(schema_decision_id)
        if (
            decision.schema_version != "schema_decision.v3"
            or decision.schema_decision_hash != schema_decision_hash
        ):
            raise ValueError("GraphIntent correction Decision.v3 pins mismatch")
        binding = dict(decision.graph_intent_binding or {})
        if (
            binding.get("graph_intent_ref") != graph_intent_ref
            or binding.get("graph_intent_hash") != graph_intent_hash
        ):
            raise ValueError("GraphIntent correction predecessor pins mismatch")
        if not isinstance(graph_intent_changes, dict) or set(
            graph_intent_changes
        ) != {"primary_relationship", "source_role_interpretation"}:
            raise ValueError(
                "GraphIntent correction must contain exact typed relation changes"
            )
        normalized = self.normalize_graph_intent_review(
            channel=channel,
            state="edited",
            operating_mode="assisted",
            relation_edit={
                "primaryRelationship": graph_intent_changes[
                    "primary_relationship"
                ],
                "sourceRoleInterpretation": graph_intent_changes[
                    "source_role_interpretation"
                ],
            },
            selected_question_ids=selected_question_ids,
        )
        predecessor = self._graph_intent_store().get(
            graph_intent_ref, graph_intent_hash=graph_intent_hash
        )
        if predecessor.source_binding.as_dict() != decision.source_binding:
            raise ValueError("GraphIntent correction source ownership mismatch")
        if not self._has_typed_relation_delta(predecessor, normalized["changes"]):
            raise ValueError("GraphIntent correction requires a typed semantic change")
        material_changes = {
            field: value
            for field, value in normalized["changes"].items()
            if value != getattr(predecessor, field).as_dict()
        }
        unknown_question_ids = set(normalized["selected_question_ids"]) - set(
            self._material_question_ids(predecessor)
        )
        if unknown_question_ids:
            raise ValueError(
                "selectedQuestionIds must reference exact unresolved material question ids"
            )
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        feedback = SchemaLearningFeedbackV2.create(
            {
                "source_binding": decision.source_binding,
                "schema_decision_ref": decision.schema_decision_id,
                "candidate_ref": str(decision.candidate_binding["candidate_id"]),
                "extraction_run_ref": None,
                "grounding_ref": None,
                "feedback_scope": "schema_understanding",
                "produced_by": "user_chat_clarification",
                "operating_mode": "assisted",
                "policy_ref": None,
                "rationale_ref": "rationale:studio-graph-intent-correction",
                "decision_evidence_refs": list(decision.source_evidence_refs),
                "fallback_reason": None,
                "operation_ref": None,
                "before_state_ref": None,
                "after_state_ref": None,
                "attribution": {
                    "graph_intent": True,
                    "source_structure": False,
                    "ontology_induction": False,
                    "synthesis": False,
                    "extraction_plan": False,
                    "execution": False,
                },
                "findings": [
                    {
                        "finding_kind": "material_graph_intent_correction",
                        "correction_channel": channel,
                        "evidence_refs": list(decision.source_evidence_refs),
                    }
                ],
                "proposed_candidate_delta_ref": None,
                "created_at": created_at,
                "created_by": self.user_id,
            }
        )
        result = self.record_schema_feedback(
            schema_decision_id,
            feedback=feedback.as_dict(),
            candidate_delta=None,
            graph_intent_changes=material_changes,
        )
        rerun = dict(result.get("material_correction_rerun") or {})
        if (
            rerun.get("outcome") != "pending_adjudication"
            or rerun.get("reason")
            != "graph_intent_successor_requires_witness_adjudication"
            or result.get("candidate") is not None
            or result.get("review_decision") is not None
        ):
            raise ValueError("GraphIntent correction must stop pending adjudication")
        return result

    def retest_schema_feedback(
        self,
        successor_schema_decision_id: str,
        *,
        baseline_schema_decision_id: str,
    ) -> dict[str, Any]:
        return SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        ).comparable_retest(
            organization_id=self.organization_id,
            baseline_schema_decision_ref=baseline_schema_decision_id,
            successor_schema_decision_ref=successor_schema_decision_id,
            requested_by=self.user_id,
        )

    def retest_source_structure_feedback(
        self,
        successor_schema_decision_id: str,
        *,
        baseline_schema_decision_id: str,
        semantic_validation_ref: str,
        semantic_validation_hash: str,
    ) -> dict[str, Any]:
        return SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        ).comparable_retest(
            organization_id=self.organization_id,
            baseline_schema_decision_ref=baseline_schema_decision_id,
            successor_schema_decision_ref=successor_schema_decision_id,
            requested_by=self.user_id,
            source_structure_validation_ref=semantic_validation_ref,
            source_structure_validation_hash=semantic_validation_hash,
            _source_structure_authority=_SOURCE_STRUCTURE_RETEST_AUTHORITY,
        )

    def continue_source_structure_correction(
        self,
        *,
        predecessor_candidate_ref: str,
        predecessor_candidate_hash: str,
        predecessor_schema_decision_ref: str,
        predecessor_schema_decision_hash: str,
        successor_graph_intent_ref: str,
        successor_graph_intent_hash: str,
        semantic_validation_ref: str,
        semantic_validation_hash: str,
    ) -> dict[str, Any]:
        from contracts.source_structure_correction_continuation import (
            SourceStructureCorrectionContinuationV1,
        )
        from storage.source_structure_correction_continuation_store import (
            SourceStructureCorrectionContinuationStore,
            _SOURCE_STRUCTURE_CONTINUATION_AUTHORITY,
        )
        from storage.source_structure_semantic_validation_store import (
            SourceStructureSemanticValidationStore,
        )

        compiled = self.compile_source_structure_successor(
            predecessor_candidate_ref=predecessor_candidate_ref,
            predecessor_candidate_hash=predecessor_candidate_hash,
            predecessor_schema_decision_ref=predecessor_schema_decision_ref,
            predecessor_schema_decision_hash=predecessor_schema_decision_hash,
            successor_graph_intent_ref=successor_graph_intent_ref,
            successor_graph_intent_hash=successor_graph_intent_hash,
            semantic_validation_ref=semantic_validation_ref,
            semantic_validation_hash=semantic_validation_hash,
        )
        successor_candidate = compiled["candidate"]
        successor_decision = compiled["decision"]
        comparison = self.retest_source_structure_feedback(
            successor_decision["schema_decision_id"],
            baseline_schema_decision_id=predecessor_schema_decision_ref,
            semantic_validation_ref=semantic_validation_ref,
            semantic_validation_hash=semantic_validation_hash,
        )
        validation = SourceStructureSemanticValidationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(semantic_validation_ref, validation_hash=semantic_validation_hash)
        gate = compiled["exemplar_gate"]
        continuation = SourceStructureCorrectionContinuationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).publish(
            SourceStructureCorrectionContinuationV1.create(
                {
                    "schema_version": "source_structure_correction_continuation.v1",
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id,
                    "semantic_validation_ref": semantic_validation_ref,
                    "semantic_validation_hash": semantic_validation_hash,
                    "predecessor_graph_intent_ref": validation.predecessor_graph_intent_ref,
                    "predecessor_graph_intent_hash": validation.predecessor_graph_intent_hash,
                    "successor_graph_intent_ref": successor_graph_intent_ref,
                    "successor_graph_intent_hash": successor_graph_intent_hash,
                    "predecessor_candidate_ref": predecessor_candidate_ref,
                    "predecessor_candidate_hash": predecessor_candidate_hash,
                    "successor_candidate_ref": successor_candidate["candidate_id"],
                    "successor_candidate_hash": successor_candidate["candidate_hash"],
                    "predecessor_schema_decision_ref": predecessor_schema_decision_ref,
                    "predecessor_schema_decision_hash": predecessor_schema_decision_hash,
                    "successor_schema_decision_ref": successor_decision[
                        "schema_decision_id"
                    ],
                    "successor_schema_decision_hash": successor_decision[
                        "schema_decision_hash"
                    ],
                    "positive_manifest_ref": gate["positive_manifest_ref"],
                    "positive_manifest_hash": gate["positive_manifest_hash"],
                    "positive_count": gate["positive_count"],
                    "hard_negative_manifest_ref": gate[
                        "hard_negative_manifest_ref"
                    ],
                    "hard_negative_manifest_hash": gate[
                        "hard_negative_manifest_hash"
                    ],
                    "hard_negative_count": gate["hard_negative_count"],
                    "comparison_ref": comparison["comparison_id"],
                    "comparison_hash": comparison["comparison_hash"],
                    "baseline_extract_ref": comparison["baseline_extract_ref"],
                    "baseline_extract_hash": comparison["baseline_extract_hash"],
                    "successor_extract_ref": comparison["successor_extract_ref"],
                    "successor_extract_hash": comparison["successor_extract_hash"],
                    "created_at": validation.created_at,
                    "created_by": validation.created_by,
                }
            ),
            _authority=_SOURCE_STRUCTURE_CONTINUATION_AUTHORITY,
        )
        return {
            **compiled,
            "outcome": "pending_materialization",
            "comparison": comparison,
            "source_structure_correction_continuation": continuation.model_dump(
                mode="json"
            ),
        }

    def finalize_source_structure_correction(
        self,
        continuation_ref: str,
        *,
        continuation_hash: str,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        if self.graph_backend is None:
            raise ValueError("source-structure finalization requires graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )
        from services.source_structure_correction_service import (
            SourceStructureCorrectionService,
            _SOURCE_STRUCTURE_FINALIZATION_CAPABILITY,
        )
        from storage.source_structure_correction_continuation_store import (
            SourceStructureCorrectionContinuationStore,
        )

        continuation = SourceStructureCorrectionContinuationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(continuation_ref, continuation_hash=continuation_hash)
        materializer = AcceptedSemanticGraphMaterializationService(
            runtime_root=self.workspace_root,
            workspace_root=self.workspace_root,
            db_path=self.workspace_root / "eos.db",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_backend=self.graph_backend,
            profile_registry=self.profile_registry,
        )
        return SourceStructureCorrectionService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            discovery_service=self.discovery,
            materializer=materializer,
        ).finalize(
            continuation,
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
            candidate_graph_id=candidate_graph_id,
            human_label_event_ids=human_label_event_ids,
            requesting_actor_ref=self.user_id,
            _authority=_SOURCE_STRUCTURE_FINALIZATION_CAPABILITY,
        )

    def replay_source_structure_release_finalization(
        self, continuation_ref: str, *, continuation_hash: str
    ):
        if self.graph_backend is None:
            raise ValueError("source-structure finalization replay requires graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )
        from services.source_structure_correction_service import (
            SourceStructureCorrectionService,
            _SOURCE_STRUCTURE_FINALIZATION_CAPABILITY,
        )
        from storage.source_structure_correction_continuation_store import (
            SourceStructureCorrectionContinuationStore,
        )

        continuation = SourceStructureCorrectionContinuationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(continuation_ref, continuation_hash=continuation_hash)
        return SourceStructureCorrectionService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            discovery_service=self.discovery,
            materializer=AcceptedSemanticGraphMaterializationService(
                runtime_root=self.workspace_root,
                workspace_root=self.workspace_root,
                db_path=self.workspace_root / "eos.db",
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                graph_backend=self.graph_backend,
                profile_registry=self.profile_registry,
            ),
        ).replay(
            continuation,
            _authority=_SOURCE_STRUCTURE_FINALIZATION_CAPABILITY,
        )

    def replay_material_correction_rerun(
        self, rerun_ref: str, *, rerun_hash: str
    ) -> dict[str, Any]:
        return SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        ).replay_material_correction_rerun(
            rerun_ref,
            rerun_hash=rerun_hash,
        )

    def adjudicate_graph_intent_correction(
        self,
        rerun_ref: str,
        *,
        rerun_hash: str,
        witness_adjudication_ref: str,
        witness_adjudication_hash: str,
    ) -> dict[str, Any]:
        return SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        ).adjudicate_graph_intent_correction(
            rerun_ref,
            rerun_hash=rerun_hash,
            witness_adjudication_ref=witness_adjudication_ref,
            witness_adjudication_hash=witness_adjudication_hash,
            requesting_actor_ref=self.user_id,
        )

    def finalize_graph_intent_correction(
        self,
        rerun_ref: str,
        *,
        rerun_hash: str,
        witness_adjudication_ref: str,
        witness_adjudication_hash: str,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        if self.graph_backend is None:
            raise ValueError("GraphIntent finalization requires configured graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )

        learning = SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        )
        materializer = AcceptedSemanticGraphMaterializationService(
            runtime_root=self.workspace_root,
            workspace_root=self.workspace_root,
            db_path=self.workspace_root / "eos.db",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_backend=self.graph_backend,
            profile_registry=self.profile_registry,
        )
        return learning.finalize_graph_intent_correction(
            rerun_ref,
            rerun_hash=rerun_hash,
            witness_adjudication_ref=witness_adjudication_ref,
            witness_adjudication_hash=witness_adjudication_hash,
            requesting_actor_ref=self.user_id,
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
            candidate_graph_id=candidate_graph_id,
            human_label_event_ids=human_label_event_ids,
            materializer=materializer,
            _finalization_authority=_GRAPH_INTENT_RELEASE_FINALIZATION_AUTHORITY,
        )

    def replay_graph_intent_release_finalization(
        self, rerun_ref: str, *, rerun_hash: str
    ):
        if self.graph_backend is None:
            raise ValueError("GraphIntent finalization replay requires graph backend")
        from services.accepted_semantic_graph_materialization_service import (
            AcceptedSemanticGraphMaterializationService,
        )

        learning = SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        )
        materializer = AcceptedSemanticGraphMaterializationService(
            runtime_root=self.workspace_root,
            workspace_root=self.workspace_root,
            db_path=self.workspace_root / "eos.db",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            graph_backend=self.graph_backend,
            profile_registry=self.profile_registry,
        )
        return learning.replay_graph_intent_release_finalization(
            rerun_ref,
            rerun_hash=rerun_hash,
            materializer=materializer,
            _finalization_authority=_GRAPH_INTENT_RELEASE_FINALIZATION_AUTHORITY,
        )

    def adjudicate_graph_intent_witnesses(
        self,
        rerun_ref: str,
        *,
        rerun_hash: str,
        justification: str,
    ) -> dict[str, Any]:
        """Commit this authenticated actor's exact reconstructed-witness review."""

        learning = SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        )
        if learning.material_reruns is None:
            raise ValueError("GraphIntent adjudication requires governed discovery")
        return GraphIntentWitnessAdjudicationService(
            self.workspace_root,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            material_reruns=learning.material_reruns,
        )._adjudicate_accept_all(
            rerun_ref,
            rerun_hash=rerun_hash,
            actor_ref=self.user_id,
            justification=justification,
            _authority=_WITNESS_ADJUDICATION_AUTHORITY,
        )

    def execute_sample(self, decision_id: str) -> dict[str, Any]:
        """Execute the authoritative reviewed contract on its governed source."""
        decision = self.discovery.store.get_decision(decision_id)
        envelope = self._envelope(decision, status="sampled")
        review = decision.get("human_review") or {}
        if review.get("state") not in {"accepted", "edited"}:
            raise ValueError("sample execution requires accepted or edited review")
        candidate = envelope["candidate"]
        source_ref = envelope["pins"]["sourceRef"]
        prefix = "workspace://"
        if not source_ref.startswith(prefix):
            raise ValueError("sample source is not a governed workspace upload")
        source_path, normalized = self._governed_docx(source_ref[len(prefix) :])
        if source_ref != f"workspace://{normalized}":
            raise ValueError("sample source reference is not canonical")
        parsed = source_elements(source_path)
        executor = CorpusGraphSchemaSampleExecutor()
        contract = executor.execution_contract(candidate)
        result = executor.execute(parsed_document=parsed, candidate=candidate)
        sample = self.discovery.store.persist_sample_result(
            decision_id=decision["decision_id"],
            decision_hash=decision["decision_hash"],
            authoritative_candidate_id=candidate["candidate_id"],
            authoritative_candidate_hash=candidate["candidate_hash"],
            source_ref=source_ref,
            source_hash=envelope["pins"]["sourceHash"],
            profile_id=envelope["pins"]["profileId"],
            engine_ref=executor.engine_ref,
            execution_contract_hash=f"sha256:{content_hash(contract)}",
            result=result,
        )
        return {**envelope, "sample": sample}

    def replay_sample(self, sample_id: str) -> dict[str, Any]:
        """Read a persisted result without invoking any live extraction boundary."""
        sample = self.discovery.store.get_sample_result(sample_id)
        decision = self.discovery.store.get_decision(sample["decision_id"])
        envelope = self._envelope(decision, status="replayed")
        pins = envelope["pins"]
        expected = {
            "decision_hash": pins["decisionHash"],
            "authoritative_candidate_id": pins["candidateId"],
            "authoritative_candidate_hash": pins["candidateHash"],
            "source_ref": pins["sourceRef"],
            "source_hash": pins["sourceHash"],
            "profile_id": pins["profileId"],
        }
        if any(sample.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "persisted sample pins do not match authoritative decision"
            )
        return {**envelope, "sample": sample}

    def feedback_retest(
        self,
        correction_decision_id: str,
        *,
        baseline_decision_id: str,
        requested_relation_type: str,
    ) -> dict[str, Any]:
        """Run and persist one governed accepted-baseline/edited-correction loop."""
        if correction_decision_id == baseline_decision_id:
            raise ValueError("feedback retest decisions must be distinct")
        if (
            not isinstance(requested_relation_type, str)
            or not requested_relation_type.strip()
        ):
            raise ValueError("requestedRelationType is required")
        baseline = self.discovery.store.get_decision(baseline_decision_id)
        correction = self.discovery.store.get_decision(correction_decision_id)
        self._require_corpus_candidate(baseline)
        self._require_corpus_candidate(correction)
        if (baseline.get("human_review") or {}).get("state") != "accepted":
            raise ValueError("feedback retest baseline must be accepted")
        if (correction.get("human_review") or {}).get("state") != "edited":
            raise ValueError("feedback retest correction must be edited")
        baseline_run = self.discovery.store.get_run(baseline["run_id"])
        correction_run = self.discovery.store.get_run(correction["run_id"])
        baseline_original = baseline["proposed_semantics"]["machine_readable"][
            "candidate"
        ]
        correction_original = correction["proposed_semantics"]["machine_readable"][
            "candidate"
        ]
        if (
            baseline_run.get("source_ref"),
            baseline_run.get("source_hash"),
            baseline_original.get("source_structure_profile", {}).get("profile_id"),
            baseline_original.get("candidate_hash"),
        ) != (
            correction_run.get("source_ref"),
            correction_run.get("source_hash"),
            correction_original.get("source_structure_profile", {}).get("profile_id"),
            correction_original.get("candidate_hash"),
        ):
            raise ValueError("feedback retest source pins do not match")
        baseline_sample = self.execute_sample(baseline_decision_id)["sample"]
        correction_sample = self.execute_sample(correction_decision_id)["sample"]
        comparison = self.discovery.store.persist_feedback_retest(
            baseline_decision_id=baseline_decision_id,
            correction_decision_id=correction_decision_id,
            baseline_sample_id=baseline_sample["sample_id"],
            correction_sample_id=correction_sample["sample_id"],
            actor_id=self.user_id,
            requested_relation_type=requested_relation_type.strip(),
        )
        return {
            "capability": self.capability_id,
            "status": "feedback-retested",
            "comparison": comparison,
        }

    def _governed_docx(self, relative_path: str) -> tuple[Path, str]:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("source must be a governed session DOCX upload")
        relative = Path(relative_path.strip())
        if relative.is_absolute():
            raise ValueError("source must be a governed session DOCX upload")
        source = (self.workspace_root / relative).resolve()
        sessions_root = (self.workspace_root / "sessions").resolve()
        normalized_parts = relative.parts
        if (
            not sessions_root.is_relative_to(self.workspace_root)
            or not source.is_relative_to(sessions_root)
            or len(normalized_parts) < 4
            or normalized_parts[0] != "sessions"
            or normalized_parts[2] != "uploads"
            or not source.is_file()
            or source.suffix.lower() != ".docx"
        ):
            raise ValueError("source must be a governed session DOCX upload")
        return source, source.relative_to(self.workspace_root).as_posix()

    @staticmethod
    def _require_corpus_candidate(decision: dict[str, Any]) -> None:
        if decision.get("discovery_type") != "corpus_graph_schema_candidate":
            raise ValueError("decision is not a corpus graph schema candidate")

    def _envelope(
        self,
        decision: dict[str, Any],
        *,
        candidate: dict[str, Any] | None = None,
        run: dict[str, Any] | None = None,
        status: str,
        schema_decision: dict[str, Any] | None = None,
        witness_selection: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._require_corpus_candidate(decision)
        review = decision.get("human_review") or {}
        original_candidate = decision["proposed_semantics"]["machine_readable"][
            "candidate"
        ]
        if candidate is None and review.get("state") == "edited":
            candidate = review.get("edited_payload")
        if candidate is None:
            candidate = original_candidate
        if run is None:
            run = self.discovery.store.get_run(decision["run_id"])
        if schema_decision is None:
            self.discovery._validate_edited_candidate_source_binding(
                original_candidate, candidate
            )
        elif (
            candidate.get("schema_version") != "corpus_graph_schema_candidate.v4"
            or schema_decision.get("schema_version") != "schema_decision.v3"
            or candidate.get("source_binding")
            != original_candidate.get("source_binding")
            or candidate.get("graph_intent_binding")
            != original_candidate.get("graph_intent_binding")
            or schema_decision.get("candidate_binding", {}).get("candidate_id")
            != candidate.get("candidate_id")
            or schema_decision.get("candidate_binding", {}).get("candidate_hash")
            != candidate.get("candidate_hash")
            or schema_decision.get("candidate_binding", {}).get(
                "candidate_schema_version"
            )
            != candidate.get("schema_version")
        ):
            raise ValueError("Candidate v4/Decision v3 envelope binding mismatch")
        source_profile = original_candidate["source_structure_profile"]
        source_evidence_refs = original_candidate["source_evidence_refs"]
        profile_id = source_profile["profile_id"]
        if run.get("source_ref") != source_profile.get("source_ref") or run.get(
            "source_hash"
        ) != source_profile.get("document_hash"):
            raise ValueError("candidate run source binding is missing or tampered")
        provenance = decision.get("provenance") or {}
        version = candidate.get("schema_version")
        is_v2 = version == "corpus_graph_schema_candidate.v2"
        is_v3 = version == "corpus_graph_schema_candidate.v3"
        is_v4 = version == "corpus_graph_schema_candidate.v4"
        expected_canonical_refs = (
            [candidate["source_binding"]["canonical_source_ref"]]
            if is_v2
            else (
                [candidate["source_binding"]["canonical_source_ir_ref"]]
                if is_v3 or is_v4
                else source_evidence_refs
            )
        )
        if provenance.get("canonical_source_ir_refs") != expected_canonical_refs or (
            not is_v2
            and not is_v3
            and not is_v4
            and provenance.get("model_or_engine_ref") != profile_id
        ):
            raise ValueError("candidate provenance binding is missing or tampered")
        if is_v3 or is_v4:
            compiler = candidate.get("compiler_provenance") or {}
            if (
                provenance.get("orchestration", {}).get("skill_ref")
                != compiler.get("orchestration_skill_ref")
                or provenance.get("compiler", {}).get("capability_ref")
                != compiler.get("compiler_capability_ref")
                or provenance.get("compiler", {}).get("profile_id")
                != compiler.get("profile_id")
                or provenance.get("compiler", {}).get("profile_hash")
                != compiler.get("profile_hash")
                or provenance.get("compiler", {}).get("catalog_hash")
                != compiler.get("catalog_hash")
                or provenance.get("semantica", {}).get("commit")
                != compiler.get("semantica_commit")
                or provenance.get("semantica", {}).get("input_hash")
                != compiler.get("semantica_input_hash")
                or provenance.get("graph_intent", {}).get("ref")
                != candidate.get("graph_intent_binding", {}).get("graph_intent_ref")
                or provenance.get("graph_intent", {}).get("hash")
                != candidate.get("graph_intent_binding", {}).get("graph_intent_hash")
            ):
                raise ValueError(
                    "v3 compiler provenance binding is missing or tampered"
                )
        self._validate_immutable_decision_hash(decision)
        pins = {
            "decisionId": decision["decision_id"],
            "decisionHash": decision["decision_hash"],
            "candidateId": candidate["candidate_id"],
            "candidateHash": candidate["candidate_hash"],
            "sourceRef": run["source_ref"],
            "sourceHash": run["source_hash"],
            "profileId": profile_id,
            "skillRef": run["skill_ref"],
        }
        if is_v3 or is_v4:
            graph_intent = candidate["graph_intent_binding"]
            pins.update(
                {
                    "graphIntentRef": graph_intent["graph_intent_ref"],
                    "graphIntentHash": graph_intent["graph_intent_hash"],
                    "compilerCapabilityRef": candidate["compiler_provenance"][
                        "compiler_capability_ref"
                    ],
                    "profileHash": candidate["compiler_provenance"]["profile_hash"],
                    "catalogHash": candidate["compiler_provenance"]["catalog_hash"],
                    "semanticaCommit": candidate["compiler_provenance"][
                        "semantica_commit"
                    ],
                }
            )
        return {
            "capability": self.capability_id,
            "status": status,
            "decision": decision,
            "candidate": candidate,
            "run": run,
            "pins": pins,
            **({"schemaDecision": schema_decision} if schema_decision else {}),
            **({"witnessSelection": witness_selection} if witness_selection else {}),
        }

    @staticmethod
    def _validate_immutable_decision_hash(decision: dict[str, Any]) -> None:
        immutable_decision = {
            field: decision[field]
            for field in (
                "schema_version",
                "decision_id",
                "run_id",
                "lineage_id",
                "discovery_type",
                "subject",
                "proposed_semantics",
                "rationale",
                "alternatives_considered",
                "confidence",
                "semantica_signals",
                "provenance",
                "created_at",
            )
        }
        immutable_decision["human_review"] = {
            "state": "unreviewed",
            "actor_id": None,
            "reviewed_at": None,
            "note": None,
        }
        immutable_decision["learning_eligibility"] = {
            "status": "excluded",
            "reason": "not human reviewed",
        }
        expected_hash = f"sha256:{content_hash(immutable_decision)}"
        if decision.get("decision_hash") != expected_hash:
            raise ValueError("immutable schema decision binding is missing or tampered")
