"""Production composition for graph-guided runtime semantic inspection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.ai_grounding import CompatibleStructuredGenerator
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_inspection_result import RuntimeSemanticInspectionResultV1, RuntimeSemanticInspectionContextV1
from contracts.observed_constraint_hypothesis import (
    EffectiveRestrictionRuntimeContextV1,
    RuntimeSemanticRetrievalSnapshotV1,
)
from contracts.runtime_semantic_source_region import source_region_to_legacy_dict
from eos.artifact_hashing import content_hash
from services.decision_local_projection_service import DecisionLocalProjectionService
from services.bounded_decision_projection_service import BoundedDecisionProjectionService
from services.effective_restriction_runtime_context_service import EffectiveRestrictionRuntimeContextService
from services.governed_frame_finding_service import GovernedFrameFindingService
from services.runtime_context_proposal_provider import RuntimeContextProposalProviderRegistry
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.runtime_frame_match_service import TenderRuntimeFrameMatchService
from services.runtime_semantic_hypothesis_generation_service import (
    RuntimeSemanticHypothesisGenerationService,
)
from services.runtime_semantic_retrieval_service import RuntimeSemanticRetrievalService
from services.runtime_semantic_source_region_service import (
    RuntimeSemanticSourceRegionService,
)
from services.tender_decision_context_formation_service import (
    TenderDecisionContextFormationService,
)
from services.tender_decision_local_projection_service import (
    TenderDecisionLocalProjectionService,
)
from storage.governed_rule_store import GovernedRuleStore
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
from storage.runtime_context_authority_store import RuntimeContextAuthorityStore
from storage.runtime_semantic_hypothesis_artifact_store import (
    RuntimeSemanticHypothesisArtifactStore,
)
from storage.runtime_working_context_store import RuntimeTaskContextStore


class RuntimeSemanticInspectionService:
    """Persist the complete semantic runtime chain for one governed target."""

    def __init__(
        self,
        *,
        runtime_root: str | Path,
        workspace_root: str | Path,
        db_path: str | Path,
        actor_id: str,
        generator: CompatibleStructuredGenerator | None = None,
    ) -> None:
        if not str(actor_id).strip():
            raise ValueError("runtime semantic inspection requires actor_id")
        self._runtime_root = Path(runtime_root)
        self._workspace_root = Path(workspace_root)
        self._db_path = Path(db_path)
        self._actor_id = str(actor_id)
        self._generator = generator

    @staticmethod
    def source_region_resolver(
        *, runtime_root: str | Path, db_path: str | Path
    ) -> RuntimeSemanticSourceRegionService:
        return RuntimeSemanticSourceRegionService(
            runtime_root=runtime_root, db_path=db_path
        )

    @staticmethod
    def _runtime_context_from_snapshot(
        snapshot: RuntimeSemanticRetrievalSnapshotV1,
    ) -> EffectiveRestrictionRuntimeContextV1:
        return EffectiveRestrictionRuntimeContextV1(
            organizationId=snapshot.organization_id,
            workspaceId=snapshot.workspace_id,
            acceptedGraphReleaseRef=snapshot.accepted_graph_release_ref,
            acceptedGraphReleaseHash=snapshot.accepted_graph_release_hash,
            governedRuleProjectionRef=snapshot.governed_rule_projection_ref,
            governedRuleProjectionHash=snapshot.governed_rule_projection_hash,
            orderedGovernedRuleVersionRefs=list(
                snapshot.ordered_governed_rule_version_refs
            ),
            orderedGovernedRuleVersionSetHash=(
                snapshot.ordered_governed_rule_version_set_hash
            ),
            activationSetSnapshotRef=snapshot.activation_set_snapshot_ref,
            activationSetSnapshotHash=snapshot.activation_set_snapshot_hash,
            activationResolverPolicyVersion=(
                snapshot.activation_resolver_policy_version
            ),
            resolvedActivationSetHash=snapshot.resolved_activation_set_hash,
            effectiveContextResolverHash=snapshot.effective_context_resolver_hash,
            effectiveAt=snapshot.effective_at,
        )

    def _decision_components(
        self,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> tuple[
        RuntimeContextDTO,
        TenderDecisionContextFormationService,
        BoundedDecisionProjectionService,
        RuntimeSemanticHypothesisArtifactStore,
    ]:
        build_store = OntologyBuildStore(
            ontology_build_store_path(self._workspace_root)
        )
        authority_store = RuntimeContextAuthorityStore(build_store)
        authority_resolver = RuntimeAuthorityResolver(
            graph_intent_store=None,
            ontology_build_store=build_store,
            runtime_context_store=RuntimeTaskContextStore(build_store),
            authority_store=authority_store,
            provider_registry=(
                RuntimeContextProposalProviderRegistry.compose_from_packaged_resources()
            ),
        )
        actor_context = RuntimeContextDTO(
            organization_id=organization_id,
            workspace_id=workspace_id,
            actor_id=self._actor_id,
        )
        formation = TenderDecisionContextFormationService(
            build_store=build_store,
            authority_resolver=authority_resolver,
        )
        rule_store = GovernedRuleStore(self._db_path)
        projection = TenderDecisionLocalProjectionService(
            projection_service=DecisionLocalProjectionService(
                build_store=build_store
            ),
            governed_rule_resolver=rule_store.get_rule_version,
        )
        artifact_store = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        return actor_context, formation, BoundedDecisionProjectionService(
            build_store=build_store, tender_compiler=projection), artifact_store

    async def inspect(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        normalized_document_artifact_ref: str,
        runtime_context_pin: ArtifactPinV1 | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request = self._inspection_request(
            organization_id=organization_id, workspace_id=workspace_id,
            normalized_document_artifact_ref=normalized_document_artifact_ref,
            runtime_context_pin=runtime_context_pin, request_id=request_id,
        )
        completed = self.get_completed_inspection(
            organization_id=organization_id, workspace_id=workspace_id,
            normalized_document_artifact_ref=normalized_document_artifact_ref,
            runtime_context_pin=runtime_context_pin, request_id=request_id,
        )
        if completed is not None:
            return completed.model_dump(mode="json", by_alias=True)
        if self._generator is None:
            raise ValueError("INSPECTION_GENERATOR_REQUIRED")
        prepared = await self.prepare_context(
            organization_id=organization_id, workspace_id=workspace_id,
            normalized_document_artifact_ref=normalized_document_artifact_ref,
            runtime_context_pin=runtime_context_pin, request_id=request_id,
        )
        run_id = prepared.run_id
        regions = tuple(source_region_to_legacy_dict(region) for region in prepared.source_regions)
        artifact_store = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root, organization_id=organization_id, workspace_id=workspace_id,
        )
        retrievals = [artifact_store.get_retrieval_snapshot(snapshot_ref=pin.ref, snapshot_hash=pin.hash)
                      for pin in prepared.retrieval_snapshot_refs]
        current = EffectiveRestrictionRuntimeContextService(self._db_path).resolve_current(
            organization_id=organization_id, workspace_id=workspace_id,
        )
        excluded = {"effective_at", "effective_context_resolver_hash"}
        if any(self._runtime_context_from_snapshot(item).model_dump(exclude=excluded)
               != current.model_dump(exclude=excluded) for item in retrievals):
            raise ValueError("PINNED_RUNTIME_AUTHORITY_CHANGED")
        generator = RuntimeSemanticHypothesisGenerationService(
            runtime_root=self._runtime_root,
            db_path=self._db_path,
            generator=self._generator,
        )
        matcher = TenderRuntimeFrameMatchService(
            runtime_root=self._runtime_root, db_path=self._db_path
        )
        finding_service = GovernedFrameFindingService(runtime_root=self._runtime_root)
        hypotheses = []
        matches = []
        findings = []
        for projection_pin in prepared.decision_projection_refs:
            hypothesis = await generator.generate_and_persist(
                organization_id=organization_id,
                workspace_id=workspace_id,
                run_id=run_id,
                decision_projection_ref=projection_pin.ref,
                decision_projection_hash=projection_pin.hash,
            )
            hypotheses.append(hypothesis)
            for pattern_ref in hypothesis.retrieved_pattern_refs:
                match = matcher.match(
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    hypothesis_ref=hypothesis.hypothesis_id,
                    hypothesis_hash=hypothesis.hypothesis_hash,
                    canonical_pattern_ref=pattern_ref,
                )
                finding = finding_service.create_and_persist(
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    frame_match_result_ref=match.result_id,
                    frame_match_result_hash=match.result_hash,
                )
                matches.append(match)
                findings.append(finding)
        binding = regions[0]["source_binding"]
        result = {
            "schema_version": "runtime_semantic_inspection_result.v1",
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "run_id": run_id,
            "source_identity_ref": binding["sourceIdentityRef"],
            "source_hash": binding["sourceHash"],
            "canonical_source_ir_ref": binding["canonicalSourceIrRef"],
            "canonical_source_ir_hash": binding["canonicalSourceIrHash"],
            "retrieval_snapshot_refs": [
                {"ref": item.snapshot_id, "hash": item.snapshot_hash}
                for item in retrievals
            ],
            "observed_hypothesis_refs": [
                {"ref": item.hypothesis_id, "hash": item.hypothesis_hash}
                for item in hypotheses
            ],
            "frame_match_result_refs": [
                {"ref": item.result_id, "hash": item.result_hash} for item in matches
            ],
            "governed_finding_refs": [
                {"ref": item.finding_id, "hash": item.finding_hash} for item in findings
            ],
            "governed_findings": [
                item.model_dump(mode="json", by_alias=True) for item in findings
            ],
        }
        completed = RuntimeSemanticInspectionResultV1.create(**result)
        return self._publish_result(completed, request=request).model_dump(mode="json", by_alias=True)

    def get_completed_inspection(
        self, *, organization_id, workspace_id, normalized_document_artifact_ref,
        runtime_context_pin=None, request_id=None,
    ) -> RuntimeSemanticInspectionResultV1 | None:
        request = self._inspection_request(
            organization_id=organization_id, workspace_id=workspace_id,
            normalized_document_artifact_ref=normalized_document_artifact_ref,
            runtime_context_pin=runtime_context_pin, request_id=request_id,
        )
        if request is None:
            return None
        store = OntologyBuildStore(ontology_build_store_path(self._workspace_root))
        try:
            pin = store.get_operation_result(**request, artifact_kind="runtime_semantic_inspection_result.v1")
        finally:
            store.close()
        return None if pin is None else self.resolve_result(
            organization_id=organization_id, workspace_id=workspace_id, pin=pin,
        )

    async def prepare_context(
        self, *, organization_id: str, workspace_id: str,
        normalized_document_artifact_ref: str, runtime_context_pin: ArtifactPinV1 | None = None,
        request_id: str | None = None,
    ) -> RuntimeSemanticInspectionContextV1:
        request = self._inspection_request(
            organization_id=organization_id, workspace_id=workspace_id,
            normalized_document_artifact_ref=normalized_document_artifact_ref,
            runtime_context_pin=runtime_context_pin, request_id=request_id,
        )
        if request is not None:
            request["operation"] = "runtime_semantic_inspection_context_with_perception"
            store = OntologyBuildStore(ontology_build_store_path(self._workspace_root))
            try:
                pin = store.get_operation_result(**request, artifact_kind="runtime_semantic_inspection_context.v1")
            finally:
                store.close()
            if pin is not None:
                return self.resolve_context(organization_id=organization_id, workspace_id=workspace_id, pin=pin)
        resolved_regions = self.source_region_resolver(
            runtime_root=self._runtime_root, db_path=self._db_path
        ).resolve(
            organization_id=organization_id,
            workspace_id=workspace_id,
            normalized_document_artifact_ref=normalized_document_artifact_ref,
        )
        regions = tuple(
            source_region_to_legacy_dict(region) for region in resolved_regions
        )
        run_id = (
            "semantic-inspection:"
            + content_hash(
                {
                    "organization_id": organization_id,
                    "workspace_id": workspace_id,
                    "normalized_document_artifact_ref": normalized_document_artifact_ref,
                    "source_binding": regions[0]["source_binding"],
                    **({"runtime_context_pin": runtime_context_pin.model_dump(mode="json")}
                       if runtime_context_pin is not None else {}),
                    **({"request_id": request_id, "actor_id": self._actor_id}
                       if request_id is not None else {}),
                }
            )[:24]
        )
        retriever = RuntimeSemanticRetrievalService(
            runtime_root=self._runtime_root, db_path=self._db_path,
            workspace_root=self._workspace_root,
        )
        actor_context, formation, projection_service, artifact_store = (
            self._decision_components(
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
        )
        from services.runtime_restriction_perception_service import RuntimeRestrictionPerceptionService
        perception_service = RuntimeRestrictionPerceptionService(generator=self._generator, workspace_root=self._workspace_root)
        perceptions = []
        retrievals = []
        working_contexts = []
        projections = []
        for typed_region, region in zip(resolved_regions, regions, strict=True):
            perception, _ = await perception_service.perceive_persisted(
                store=formation._store, authority_resolver=formation._authority,
                actor_context=actor_context, source_region=typed_region, run_id=run_id,
                request_id=run_id + ":" + typed_region.source_region_ref)
            perception_pin = perception_service.pin(perception)
            perceptions.append(perception_pin)
            retrieval = retriever.retrieve_and_persist(
                organization_id=organization_id,
                workspace_id=workspace_id,
                run_id=run_id,
                source_region=region,
                perception_pin=perception_pin, typed_source_region=typed_region,
                runtime_context_pin=runtime_context_pin,
            )
            runtime_context = self._runtime_context_from_snapshot(retrieval)
            formed_at = datetime.now(timezone.utc)
            canonical_patterns = tuple(
                item
                for item in retrieval.ordered_material
                if item.kind == "canonical_pattern"
            )
            if not canonical_patterns:
                raise ValueError("runtime semantic inspection requires a canonical pattern")
            for pattern in canonical_patterns:
                working_context = formation.form(
                    actor_context=actor_context,
                    source_region=typed_region,
                    retrieval_snapshot=retrieval,
                    canonical_pattern_ref=pattern.ref,
                    formed_at=formed_at,
                )
                decision_projection = projection_service.compile_runtime_bound(
                    working_context=working_context,
                    decision_subject_ref=typed_region.source_region_ref,
                    runtime_context=runtime_context,
                    retrieval_snapshot=retrieval,
                    canonical_pattern_ref=pattern.ref,
                    source_region=typed_region,
                )
                artifact_store.put_decision_projection(
                    run_id=run_id,
                    projection=decision_projection.legacy,
                    working_context=working_context,
                    retrieval_snapshot_ref=retrieval.snapshot_id,
                    retrieval_snapshot_hash=retrieval.snapshot_hash,
                    source_region=typed_region if typed_region.material_dependencies else None,
                )
                artifact_store.put_bounded_projection(decision_projection)

                working_contexts.append(working_context.pin())
                projections.append({"ref": decision_projection.projection_ref, "hash": decision_projection.projection_hash})
            retrievals.append(retrieval)
        prepared = RuntimeSemanticInspectionContextV1.create(
            schema_version="runtime_semantic_inspection_context.v1",
            organization_id=organization_id, workspace_id=workspace_id, run_id=run_id,
            normalized_document_artifact_ref=normalized_document_artifact_ref, request_id=request_id,
            runtime_context_pin=runtime_context_pin.model_dump(mode="json") if runtime_context_pin else None,
            source_regions=[region.model_dump(mode="json", by_alias=True) for region in resolved_regions],
            retrieval_snapshot_refs=[{"ref": item.snapshot_id, "hash": item.snapshot_hash} for item in retrievals],
            working_context_pins=[pin.model_dump(mode="json") for pin in working_contexts],
            decision_projection_refs=projections,
            perception_pins=[pin.model_dump(mode="json") for pin in perceptions],
        )
        store = OntologyBuildStore(ontology_build_store_path(self._workspace_root))
        try:
            pin = ArtifactPinV1(artifact_kind=prepared.schema_version, artifact_ref=prepared.context_ref, artifact_hash=prepared.context_hash)
            if request is None:
                store.put(prepared)
            else:
                pin = store.put_many_with_operation_result(**request, artifacts=(prepared,), result_pin=pin)
        finally:
            store.close()
        return self.resolve_context(organization_id=organization_id, workspace_id=workspace_id, pin=pin)

    def resolve_context(self, *, organization_id, workspace_id, pin):
        if pin.artifact_kind != "runtime_semantic_inspection_context.v1":
            raise ValueError("INSPECTION_CONTEXT_KIND_MISMATCH")
        store = OntologyBuildStore(ontology_build_store_path(self._workspace_root))
        try:
            context = store.get(organization_id=organization_id, workspace_id=workspace_id,
                                artifact_ref=pin.artifact_ref, contract_type=RuntimeSemanticInspectionContextV1)
        finally:
            store.close()
        if (
            context is None or context.context_ref != pin.artifact_ref or context.context_hash != pin.artifact_hash
            or context.organization_id != organization_id or context.workspace_id != workspace_id
        ):
            raise ValueError("INSPECTION_CONTEXT_PIN_MISMATCH")
        artifacts = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root, organization_id=organization_id, workspace_id=workspace_id,
        )
        perception_queries = {}
        if context.perception_pins:
            from services.runtime_restriction_perception_service import RuntimeRestrictionPerceptionService
            perception_store = OntologyBuildStore(ontology_build_store_path(self._workspace_root))
            try:
                for region, perception_pin in zip(context.source_regions, context.perception_pins, strict=True):
                    _, observed = RuntimeRestrictionPerceptionService.read_persisted(store=perception_store,
                        organization_id=organization_id, workspace_id=workspace_id,
                        source_region=region, response_pin=perception_pin, run_id=context.run_id)
                    perception_queries[region.source_region_ref] = RuntimeRestrictionPerceptionService.retrieval_query(observed, region)
            finally:
                perception_store.close()
        retrievals = {(item.ref, item.hash) for item in context.retrieval_snapshot_refs}
        for region, retrieval_pin in zip(context.source_regions, context.retrieval_snapshot_refs, strict=True):
            snapshot = artifacts.get_retrieval_snapshot(snapshot_ref=retrieval_pin.ref, snapshot_hash=retrieval_pin.hash)
            if context.perception_pins:
                selected = context.perception_pins[context.source_regions.index(region)]
                if (snapshot.query.get("perception_pin") != selected.model_dump(mode="json")
                    or snapshot.query.get("perception_query") != perception_queries[region.source_region_ref]):
                    raise ValueError("INSPECTION_PERCEPTION_LINEAGE_MISMATCH")
            if snapshot.run_id != context.run_id or snapshot.query.get("source_region_ref") != region.source_region_ref:
                raise ValueError("INSPECTION_CONTEXT_LINEAGE_MISMATCH")
        for projection_pin, working_pin in zip(context.decision_projection_refs, context.working_context_pins, strict=True):
            if projection_pin.ref.startswith("decision-local-projection:v2:"):
                bounded = artifacts.get_bounded_projection(
                    projection_ref=projection_pin.ref, projection_hash=projection_pin.hash)
                working, snapshot = bounded.working_context, bounded.retrieval_snapshot
            else:
                projection, working, snapshot = artifacts.get_decision_projection(
                    projection_ref=projection_pin.ref, projection_hash=projection_pin.hash)
            if working.pin() != working_pin or (snapshot.snapshot_id, snapshot.snapshot_hash) not in retrievals:
                raise ValueError("INSPECTION_CONTEXT_LINEAGE_MISMATCH")
        return context

    def _inspection_request(
        self, *, organization_id, workspace_id, normalized_document_artifact_ref,
        runtime_context_pin, request_id,
    ):
        if request_id is None:
            return None
        if not request_id.strip():
            raise ValueError("INSPECTION_REQUEST_ID_REQUIRED")
        return dict(
            organization_id=organization_id, workspace_id=workspace_id,
            actor_ref=self._actor_id, operation="runtime_semantic_inspection",
            request_id=request_id,
            request_hash="sha256:" + content_hash({
                "normalized_document_artifact_ref": normalized_document_artifact_ref,
                "runtime_context_pin": runtime_context_pin.model_dump(mode="json") if runtime_context_pin else None,
            }),
        )

    def _publish_result(self, completed, *, request):
        store = OntologyBuildStore(ontology_build_store_path(
            self._workspace_root,
        ))
        try:
            if request is None:
                store.put(completed)
                return completed
            winner = store.put_many_with_operation_result(
                **request, artifacts=(completed,), result_pin=ArtifactPinV1(
                    artifact_kind=completed.schema_version, artifact_ref=completed.result_ref,
                    artifact_hash=completed.result_hash,
                ),
            )
        finally:
            store.close()
        return self.resolve_result(
            organization_id=completed.organization_id, workspace_id=completed.workspace_id, pin=winner,
        )

    def resolve_basis_content(self, *, organization_id, workspace_id, finding) -> dict[str, dict]:
        """Read recorded interpretation/match bytes; never regenerate missing detail."""
        if (finding.organization_id, finding.workspace_id) != (organization_id, workspace_id):
            raise ValueError("BASIS_CONTENT_SCOPE_MISMATCH")
        artifacts = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root, organization_id=organization_id, workspace_id=workspace_id,
        )
        content = {}
        try:
            hypothesis = artifacts.get_hypothesis(
                hypothesis_ref=finding.observed_hypothesis_ref,
                hypothesis_hash=finding.observed_hypothesis_hash,
            )
            if (hypothesis.source_binding.source_identity_ref, hypothesis.source_binding.source_hash) != (
                finding.source_identity_ref, finding.source_hash,
            ):
                raise ValueError("BASIS_CONTENT_LINEAGE_MISMATCH")
            content[finding.observed_hypothesis_ref] = hypothesis.model_dump(mode="json", by_alias=True)
        except (OSError, ValueError):
            pass  # Optional inspection body is unavailable; the saved finding remains reviewable.
        try:
            match = artifacts.get_frame_match_result(
                result_ref=finding.frame_match_result_ref, result_hash=finding.frame_match_result_hash,
            )
            if (match.observed_hypothesis_ref, match.observed_hypothesis_hash, match.source_identity_ref, match.source_hash) != (
                finding.observed_hypothesis_ref, finding.observed_hypothesis_hash, finding.source_identity_ref, finding.source_hash,
            ):
                raise ValueError("BASIS_CONTENT_LINEAGE_MISMATCH")
            content[finding.frame_match_result_ref] = match.model_dump(mode="json", by_alias=True)
        except (OSError, ValueError):
            pass
        return content

    def resolve_result(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1,
    ) -> RuntimeSemanticInspectionResultV1:
        """Load an exact completed result without rerunning inspection."""
        if pin.artifact_kind != "runtime_semantic_inspection_result.v1":
            raise ValueError("INSPECTION_RESULT_KIND_MISMATCH")
        store = OntologyBuildStore(ontology_build_store_path(
            self._workspace_root,
        ))
        try:
            result = store.get(
                organization_id=organization_id, workspace_id=workspace_id,
                artifact_ref=pin.artifact_ref, contract_type=RuntimeSemanticInspectionResultV1,
            )
        finally:
            store.close()
        if result is None:
            raise ValueError("INSPECTION_RESULT_NOT_FOUND")
        if (
            result.result_ref != pin.artifact_ref or result.result_hash != pin.artifact_hash
            or result.organization_id != organization_id or result.workspace_id != workspace_id
        ):
            raise ValueError("INSPECTION_RESULT_PIN_MISMATCH")
        return result


__all__ = ["RuntimeSemanticInspectionService"]
