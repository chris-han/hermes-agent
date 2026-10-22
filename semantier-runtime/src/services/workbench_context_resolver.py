
from storage.hosted_postgres import sql_query,begin_write,in_transaction
"""Shared stateless resolver for exact saved workbench artifacts."""

from datetime import datetime, timezone
from agents.auth_session import RequestContext
from contracts.evidence_location import ResolvedLocation
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1, ExploratoryGraphSnapshotV1
from contracts.semantic_kernel import ChangeCandidate, ContextGraph, Evidence, ContextGraphOrigin, DecisionRecord, WorkspaceScope
from eos.artifact_hashing import content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from contracts.ontology_workbench_context import (
    EvidenceFocusPayloadV1,
    FocusEvidenceActionEnvelopeV6,
    WorkbenchActionAcknowledgementV6,
    WorkbenchArtifactProjectionV2,
    WorkbenchEvidenceFocusV1,
    WorkbenchProjectionCursorHeadV3,
    WorkbenchEvidenceMemberV1,
    WorkbenchEvidenceSelectionIntentV1,
    WorkbenchSelectionRequestV1,
    WorkbenchArtifactV2,
    WorkbenchComparisonV3,
    WorkbenchSelectionV3,
    WorkbenchCapabilityV3,
    WorkbenchRuntimeStartV1,
    WorkbenchRuntimeV3,
    WorkbenchLearningV1,
    RuntimeReportCommandV1,
    EvidenceActionBindingV1,
    EvidenceFocusProjectionV1,
    EvidenceSourceGroupV1,
    LifecycleArtifactBindingEntryV1,
    KnowledgeWorkbenchContextV6,
    UnifiedWorkbenchContextV3,
    projection_hash_basis,
    serialize_unified_workbench_context_v3,
)
from storage.ontology_build_store import OntologyBuildStore
from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore
from services.runtime_errors import RuntimeContextError
from services.runtime_temporal_source_service import RuntimeTemporalSourceService
from services.runtime_inspection_report_service import server_runtime_report_profile_pins
from services.runtime_report_export_service import server_runtime_export_profile
from contracts.runtime_inspection_report import (
    ResolvedRuntimeReportPolicyBasisV1,
    ResolutionRequirementV1,
    RuntimeReportFindingV1,
    VerifiedRuntimeReportInputsV1,
)
from contracts.runtime_temporal_source import RuntimeTemporalSourceGraphV1
from storage.runtime_inspection_report_store import RuntimeInspectionReportStore
from storage.runtime_semantic_evaluation_store import RuntimeSemanticEvaluationStore
from services.workspace_artifact_preparation_service import (
    WorkspaceArtifactPreparationService,
)
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_for_block,
    evidence_ref_pin,
)
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref


class WorkbenchContextResolver:
    def __init__(
        self,
        *,
        store: OntologyBuildStore,
        source_service: RuntimeTemporalSourceService | None = None,
        evaluation_store: RuntimeSemanticEvaluationStore | None = None,
        authority_resolver: RuntimeAuthorityResolver | None = None,
        preparation_service: WorkspaceArtifactPreparationService | None = None,
    ) -> None:
        if preparation_service is not None and (
            type(preparation_service) is not WorkspaceArtifactPreparationService
            or preparation_service._build is not store
            or preparation_service._artifacts._authority is not authority_resolver
        ):
            raise RuntimeContextError("INVALID_STATE")
        if authority_resolver is not None and (
            type(authority_resolver) is not RuntimeAuthorityResolver
            or getattr(
                authority_resolver._owners.get("exploratory_graph_snapshot"),
                "_store",
                None,
            )
            is not store
        ):
            raise RuntimeContextError("INVALID_STATE")
        temporal = (source_service, evaluation_store)
        if any(owner is not None for owner in temporal):
            if (
                type(source_service) is not RuntimeTemporalSourceService
                or type(evaluation_store) is not RuntimeSemanticEvaluationStore
                or source_service._build_store is not store
                or evaluation_store._build_store is not store
                or source_service._authority_resolver
                is not evaluation_store._authority_resolver
                or source_service._session_resolver
                is not evaluation_store._session_resolver
            ):
                raise RuntimeContextError("INVALID_STATE")
        elif authority_resolver is None:
            raise RuntimeContextError("INVALID_STATE")
        self._store = store
        self._source_service = source_service
        self._evaluation_store = evaluation_store
        self._authority_resolver = authority_resolver
        self._preparation_service = preparation_service

    def resolve_workbench_context_v3(
        self,
        *,
        ctx: RequestContext,
        cursor_store: WorkbenchProjectionCursorStore,
        artifact_service,
        selection: WorkbenchSelectionRequestV1,
    ) -> UnifiedWorkbenchContextV3:
        if (
            type(cursor_store) is not WorkbenchProjectionCursorStore
            or cursor_store._store is not self._store
            or cursor_store._authority is not self._authority_resolver
        ):
            raise RuntimeContextError("INVALID_STATE")
        selection = WorkbenchSelectionRequestV1.model_validate(selection.model_dump(mode="json"))
        previous = cursor_store.get(ctx=ctx, cursor_ref=selection.cursor_ref)
        if (
            previous.revision != selection.expected_material_context_revision
            or previous.projection_basis_hash != selection.expected_projection_hash
        ):
            raise ValueError("STALE_MATERIAL_CONTEXT")
        if previous.design_snapshot_pin != selection.design_snapshot_pin:
            connection = self._store._connection
            if in_transaction(connection):
                raise RuntimeError("context resolution requires its own cursor transaction")
            begin_write(connection)
            try:
                previous = cursor_store.select_design_snapshot_on_connection(
                    connection,
                    ctx=ctx,
                    cursor_ref=selection.cursor_ref,
                    expected_revision=selection.expected_material_context_revision,
                    expected_projection_basis_hash=selection.expected_projection_hash,
                    design_snapshot_pin=selection.design_snapshot_pin,
                    recorded_at=datetime.now(timezone.utc),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        artifact_workspace = artifact_service.list(ctx=ctx)
        head = cursor_store.get(ctx=ctx, cursor_ref=selection.cursor_ref)
        if (
            artifact_workspace.cursor_ref != head.cursor_ref
            or artifact_workspace.material_context_revision != head.revision
            or artifact_workspace.material_projection_hash != head.projection_basis_hash
        ):
            raise ValueError("CURSOR_PIN_MISMATCH")

        snapshot = cursor_store._design_snapshot(ctx, head.design_snapshot_pin)
        focus = (
            cursor_store._focus(cursor_store._scope(ctx), head.evidence_focus_pin)
            if selection.lifecycle == "DESIGN"
            else None
        )
        artifact = WorkbenchArtifactV2(
            build_ref=snapshot.build_ref if snapshot else None,
            source_ref=snapshot.source_identity_ref if snapshot else None,
            candidate_graph_ref=snapshot.snapshot_ref if snapshot else None,
            candidate_graph_hash=snapshot.snapshot_hash if snapshot else None,
            release_ref=None,
            runtime_evaluation_ref=None,
        )
        selected_object = focus.focus.object_pin if focus else None
        selected_anchor = focus.focus.canonical_anchor if focus else None
        selected_evidence = tuple(pin.artifact_ref for pin in focus.focus.evidence_refs) if focus else ()
        projected_selection = WorkbenchSelectionV3(
            object_ref=selected_object.artifact_ref if selected_object else None,
            node_ref=selected_object.artifact_ref if selected_object and selected_object.artifact_kind == "exploratory_graph_node" else None,
            edge_ref=selected_object.artifact_ref if selected_object and selected_object.artifact_kind == "exploratory_graph_edge" else None,
            evidence_refs=selected_evidence,
            finding_ref=None,
            quality_or_validation_ref=None,
            canonical_anchor_ref=selected_anchor.artifact_ref if selected_anchor else None,
            transition_ref=None,
            proof_ref=None,
        )
        focus_allowed = False
        design_source_entry = None
        reason_code = (
            "LIFECYCLE_MISMATCH"
            if selection.lifecycle != "DESIGN"
            else "DESIGN_SNAPSHOT_REQUIRED"
        )
        if selection.lifecycle == "DESIGN" and snapshot is not None:
            expected_ir = ArtifactPinV1(
                artifact_kind="canonical_source_ir",
                artifact_ref=snapshot.canonical_source_ir_ref,
                artifact_hash=snapshot.canonical_source_ir_hash,
            )
            matches = [
                entry for entry in artifact_workspace.design_sources.entries
                if entry.state == "ACTIVE"
                and entry.source_identity_pin is not None
                and entry.source_identity_pin.artifact_ref == snapshot.source_identity_ref
                and entry.canonical_source_ir_pin == expected_ir
                and entry.preparation_pin is not None
            ]
            focus_allowed = len(matches) == 1
            design_source_entry = matches[0] if focus_allowed else None
            reason_code = None if focus_allowed else "SOURCE_NOT_BOUND"
        focus_capability = WorkbenchCapabilityV3(
            operation="FOCUS_EVIDENCE",
            allowed=focus_allowed,
            reason_code=reason_code,
            governing_receipt_pin=None,
            expected_material_context_revision=head.revision,
            expected_projection_hash=head.projection_basis_hash,
        )
        evidence_intent = selection.evidence_focus
        if evidence_intent is None and focus is not None:
            evidence_intent = WorkbenchEvidenceSelectionIntentV1(
                object_kind=(
                    "node"
                    if focus.focus.object_pin.artifact_kind == "exploratory_graph_node"
                    else "edge"
                ),
                object_ref=focus.focus.object_pin.artifact_ref,
                evidence_ref=(
                    focus.focus.selected_evidence_ref.artifact_ref
                    if focus.focus.selected_evidence_ref is not None
                    else None
                ),
                occurrence_ref=focus.occurrence_ref,
            )
        evidence_projection = None
        if (
            evidence_intent is not None
            and focus_allowed
            and design_source_entry is not None
            and design_source_entry.preparation_pin is not None
            and snapshot is not None
            and head.design_snapshot_pin is not None
        ):
            selection_snapshot = artifact_workspace.selection
            evidence_projection = self.project_design_evidence(
                ctx=ctx,
                snapshot=snapshot,
                snapshot_pin=head.design_snapshot_pin,
                artifact_pin=design_source_entry.artifact_pin,
                preparation_pin=design_source_entry.preparation_pin,
                intent=evidence_intent,
                expected_selection_pin=ArtifactPinV1(
                    artifact_kind="workspace_artifact_selection",
                    artifact_ref=selection_snapshot.selection_ref,
                    artifact_hash=selection_snapshot.selection_hash,
                ),
                expected_selection_revision=selection_snapshot.revision,
                source_entries=tuple(artifact_workspace.design_sources.entries),
            )
            selected_object = evidence_projection.object_pin
            projected_selection = WorkbenchSelectionV3(
                object_ref=selected_object.artifact_ref,
                node_ref=(
                    selected_object.artifact_ref
                    if selected_object.artifact_kind == "exploratory_graph_node"
                    else None
                ),
                edge_ref=(
                    selected_object.artifact_ref
                    if selected_object.artifact_kind == "exploratory_graph_edge"
                    else None
                ),
                evidence_refs=tuple(
                    member.evidence_ref for member in evidence_projection.members
                ),
                finding_ref=None,
                quality_or_validation_ref=None,
                canonical_anchor_ref=evidence_projection.selected_occurrence_ref,
                transition_ref=None,
                proof_ref=None,
            )
        runtime_target_entries = [
            entry for entry in artifact_workspace.runtime_target.entries
            if entry.state == "ACTIVE"
        ]
        runtime_start = None
        runtime_start_reason = "LIFECYCLE_MISMATCH"
        if selection.lifecycle == "RUNTIME":
            if len(runtime_target_entries) != 1:
                runtime_start_reason = "RUNTIME_TARGET_REQUIRED"
            elif snapshot is None:
                runtime_start_reason = "DESIGN_SNAPSHOT_REQUIRED"
            else:
                entry = runtime_target_entries[0]
                if entry.preparation_pin is None:
                    runtime_start_reason = "TARGET_UNSUPPORTED"
                else:
                    from services.workspace_artifact_preparation_service import RuntimeTargetPreparationAdapter
                    try:
                        target_graph = RuntimeTargetPreparationAdapter(
                            artifact_service=artifact_service
                        ).resolve_ready_target(
                            ctx=ctx,
                            artifact_pin=entry.artifact_pin,
                            preparation_pin=entry.preparation_pin,
                        )
                    except ValueError as exc:
                        if str(exc) == "TARGET_UNSUPPORTED":
                            target_graph = None
                            runtime_start_reason = "TARGET_UNSUPPORTED"
                        else:
                            raise
                    if target_graph is not None:
                        actor_context = RuntimeContextDTO(
                            organization_id=ctx.organization_id,
                            workspace_id=ctx.workspace_id,
                            actor_id=ctx.user_id,
                            session_id=cursor_store._scope(ctx)["session_id"],
                        )
                        self._authority_resolver.resolve_pins(
                            actor_context=actor_context,
                            pins=(target_graph.ontology, *target_graph.evidence),
                        )
                        target = __import__(
                            "contracts.runtime_semantic_evaluation",
                            fromlist=["GovernedTargetPinV1"],
                        ).GovernedTargetPinV1(
                            schema_version="governed_target_pin.v1",
                            organization_id=ctx.organization_id,
                            workspace_id=ctx.workspace_id,
                            artifact=target_graph.artifact_pin(),
                            target_type="graph",
                            evidence=tuple(sorted(
                                (target_graph.source, target_graph.normalization),
                                key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash),
                            )),
                            scope_bridge=None,
                        )
                        runtime_start = WorkbenchRuntimeStartV1(
                            source_graph_pin=target_graph.artifact_pin(),
                            ontology_pin=target_graph.ontology,
                            evidence_pins=target_graph.evidence,
                            target=target,
                            selected_entity_ids=tuple(sorted(row.id for row in target_graph.entities)),
                            preparation_pin=entry.preparation_pin,
                        )
                        runtime_start_reason = None
        runtime_start_capability = WorkbenchCapabilityV3(
            operation="START_RUNTIME",
            allowed=runtime_start is not None,
            reason_code=runtime_start_reason,
            governing_receipt_pin=None,
            expected_material_context_revision=head.revision,
            expected_projection_hash=head.projection_basis_hash,
        )
        selected_report = None
        runtime_projection = (
            WorkbenchRuntimeV3(start=runtime_start)
            if selection.lifecycle == "RUNTIME" and selection.runtime_context_pin is None
            else None
        )
        if selection.lifecycle == "RUNTIME" and selection.runtime_context_pin is not None:
            if self._evaluation_store is None:
                raise RuntimeContextError("INVALID_STATE")
            actor_context = RuntimeContextDTO(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                actor_id=ctx.user_id,
                session_id=cursor_store._scope(ctx)["session_id"],
            )
            if selection.runtime_context_pin.artifact_kind != "context_graph":
                raise RuntimeContextError("UNSUPPORTED_ARTIFACT_KIND")
            resolved = self._authority_resolver.resolve_pins(
                actor_context=actor_context,
                pins=(selection.runtime_context_pin,),
            )[0]
            if not isinstance(resolved, ContextGraph) or resolved.state != "ADMITTED":
                raise RuntimeContextError("INVALID_STATE")
            runtime_context = resolved
            run = None
            state = None
            if selection.runtime_run_pin is not None:
                run = self._evaluation_store.get_run(
                    actor_context=actor_context,
                    pin=selection.runtime_run_pin,
                )
                admission = self._evaluation_store.get_admission(
                    actor_context=actor_context,
                    pin=run.admission,
                )
                if admission.context != selection.runtime_context_pin:
                    raise RuntimeContextError("HASH_MISMATCH")
                state = self._evaluation_store.get_current_state(
                    actor_context=actor_context,
                    run=selection.runtime_run_pin,
                )
            world = runtime_context.world
            world_time = (
                world.at_time.isoformat().replace("+00:00", "Z")
                if world.mode == "point" and world.at_time is not None
                else (
                    f"{world.start_time.isoformat().replace('+00:00', 'Z')}/{world.end_time.isoformat().replace('+00:00', 'Z')}"
                    if world.mode == "interval" and world.start_time is not None and world.end_time is not None
                    else None
                )
            )
            if isinstance(runtime_context, ContextGraph):
                context_ref = runtime_context.artifact_ref
                context_version = str(runtime_context.revision)
                context_hash = runtime_context.artifact_hash
                world_scope_ref = runtime_context.artifact_ref
                world_scope_hash = runtime_context.artifact_hash
            else:
                context_ref = runtime_context.context_ref
                context_version = str(runtime_context.version)
                context_hash = runtime_context.context_hash
                world_scope_ref = world.world_ref
                world_scope_hash = world.world_hash
            result_pin = state.results[0] if state is not None and len(state.results) == 1 else None
            semantic_pin = (
                state.decisions[0]
                if state is not None and len(state.decisions) == 1
                else result_pin
            )
            if selection.runtime_report_pin is not None:
                if run is None or selection.runtime_run_pin is None:
                    raise RuntimeContextError("INVALID_STATE")
                reports = RuntimeInspectionReportStore(self._store)
                selected_report = reports.get(
                    scope=WorkspaceScope(
                        organization_id=ctx.organization_id,
                        workspace_id=ctx.workspace_id,
                    ),
                    report_pin=selection.runtime_report_pin,
                )
                if (
                    selected_report.subject.run_pin != selection.runtime_run_pin
                    or selected_report.subject.context_pin != selection.runtime_context_pin
                ):
                    raise RuntimeContextError("HASH_MISMATCH")
            report_projector_pin, report_renderer_pin, report_template_pin, report_environment_pin = (
                server_runtime_report_profile_pins()
            )
            export_replay_inputs_pin = None
            export_profile_pin = None
            if selected_report is not None:
                export_profile = server_runtime_export_profile(
                    report=selected_report,
                    format="JSON",
                    access_projection="MEMBER",
                )
                export_replay_inputs_pin = selected_report.replay_access.replay_pin
                export_profile_pin = ArtifactPinV1(
                    artifact_kind="pinned_render_profile",
                    artifact_ref=export_profile.profile_ref,
                    artifact_hash=export_profile.profile_hash,
                )
            runtime_projection = WorkbenchRuntimeV3(
                context_ref=context_ref,
                context_version=context_version,
                context_hash=context_hash,
                world_scope_ref=world_scope_ref,
                world_scope_hash=world_scope_hash,
                world_time=world_time,
                target_ref=(admission.target.artifact.artifact_ref if run is not None else None),
                target_hash=(admission.target.artifact.artifact_hash if run is not None else None),
                run_ref=(run.run_ref if run is not None else None),
                run_hash=(run.run_hash if run is not None else None),
                root_run_ref=(run.root_run.artifact_ref if run is not None and run.root_run is not None else None),
                root_run_hash=(run.root_run.artifact_hash if run is not None and run.root_run is not None else None),
                parent_run_ref=(run.parent_run.artifact_ref if run is not None and run.parent_run is not None else None),
                parent_run_hash=(run.parent_run.artifact_hash if run is not None and run.parent_run is not None else None),
                evaluation_ref=(semantic_pin.artifact_ref if semantic_pin is not None else None),
                evaluation_hash=(semantic_pin.artifact_hash if semantic_pin is not None else None),
                report_ref=(selected_report.report_ref if selected_report is not None else None),
                report_hash=(selected_report.report_hash if selected_report is not None else None),
                export_replay_inputs_pin=export_replay_inputs_pin,
                export_profile_pin=export_profile_pin,
                correction_evidence_pins=(
                    runtime_context.evidence
                    if isinstance(runtime_context, ContextGraph) and run is not None and selected_report is not None
                    else admission.target.evidence
                    if run is not None and selected_report is not None
                    else ()
                ),
                report_projector_pin=report_projector_pin,
                report_renderer_pin=report_renderer_pin,
                report_template_pin=report_template_pin,
                report_environment_pin=report_environment_pin,
                execution_epoch_ref=None,
                semantic_release_ref=None,
                semantic_release_hash=None,
            )
        export_allowed = selection.lifecycle == "RUNTIME" and selected_report is not None
        export_capability = WorkbenchCapabilityV3(
            operation="EXPORT_REPORT",
            allowed=export_allowed,
            reason_code=None if export_allowed else (
                "LIFECYCLE_MISMATCH" if selection.lifecycle != "RUNTIME" else "REPORT_REQUIRED"
            ),
            governing_receipt_pin=None,
            expected_material_context_revision=head.revision,
            expected_projection_hash=head.projection_basis_hash,
        )
        learning_projection = None
        if selection.learning_candidate_pin is not None:
            candidate = self._store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=selection.learning_candidate_pin.artifact_ref,
                contract_type=ChangeCandidate,
            )
            if candidate is None or candidate.pin() != selection.learning_candidate_pin:
                raise RuntimeContextError("HASH_MISMATCH")
            origin_context_ref = None
            origin_context_hash = None
            if isinstance(candidate.origin, ContextGraphOrigin):
                origin_context_ref = candidate.origin.context.artifact_ref
                origin_context_hash = candidate.origin.context.artifact_hash
            verdict = candidate.assessment.verdict if candidate.assessment is not None else None
            disposition = candidate.assessment.disposition if candidate.assessment is not None else None
            if verdict == "BETTER" or disposition == "ACCEPTED_FOR_SUCCESSOR_INPUT":
                try:
                    if self._evaluation_store is None:
                        raise ValueError("REFERENCE_EVALUATION_OWNER_UNAVAILABLE")
                    from services.runtime_context_reasoning_evaluation_service import RuntimeContextReasoningEvaluationService
                    from services.runtime_semantica_reasoning_service import RuntimeSemanticaReasoningService
                    from services.working_context_projection_service import WorkingContextProjectionService
                    from services.change_candidate_service import ChangeCandidateService

                    evaluator = RuntimeContextReasoningEvaluationService(
                        store=self._evaluation_store,
                        reasoning_service=RuntimeSemanticaReasoningService(
                            build_store=self._store, authority_resolver=self._authority_resolver),
                        projection_service=WorkingContextProjectionService(
                            build_store=self._store, authority_resolver=self._authority_resolver,
                            runtime_actor_id="agent:semantica-runtime"))
                    actor = RuntimeContextDTO(organization_id=ctx.organization_id,
                        workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
                        session_id=cursor_store._scope(ctx)["session_id"])
                    ChangeCandidateService(store=self._store).verify_reference_assessment(
                        candidate=candidate, context_evaluator=evaluator, actor_context=actor)
                except ValueError:
                    verdict, disposition = "INDETERMINATE", "NOT_YET_ADMISSIBLE"
            learning_projection = WorkbenchLearningV1(
                candidate_ref=candidate.artifact_ref,
                candidate_hash=candidate.artifact_hash,
                revision=candidate.revision,
                stage=candidate.stage,
                target=candidate.target,
                failure_class=candidate.failure_class,
                origin_context_ref=origin_context_ref,
                origin_context_hash=origin_context_hash,
                verdict=verdict,
                disposition=disposition,
            )
        correction_allowed = (
            selection.lifecycle == "RUNTIME"
            and selected_report is not None
            and result_pin is not None
            and runtime_projection is not None
            and bool(runtime_projection.correction_evidence_pins)
        )
        correction_capability = WorkbenchCapabilityV3(
            operation="SUBMIT_RUNTIME_CORRECTION",
            allowed=correction_allowed,
            reason_code=None if correction_allowed else (
                "LIFECYCLE_MISMATCH" if selection.lifecycle != "RUNTIME" else "REPORT_EVIDENCE_REQUIRED"
            ),
            governing_receipt_pin=None,
            expected_material_context_revision=head.revision,
            expected_projection_hash=head.projection_basis_hash,
        )
        review_allowed = bool(
            correction_allowed or learning_projection is not None
        )
        review_capability = WorkbenchCapabilityV3(
            operation="REVIEW_RUNTIME_CORRECTION",
            allowed=review_allowed,
            reason_code=None if review_allowed else (
                "LIFECYCLE_MISMATCH" if selection.lifecycle != "RUNTIME" else "REPORT_EVIDENCE_REQUIRED"
            ),
            governing_receipt_pin=None,
            expected_material_context_revision=head.revision,
            expected_projection_hash=head.projection_basis_hash,
        )
        shared = dict(
            representation=selection.representation,
            artifact=artifact,
            artifact_workspace=artifact_workspace,
            lifecycle=selection.lifecycle,
            runtime=runtime_projection,
            learning=learning_projection,
            evidence=evidence_projection,
            comparison=WorkbenchComparisonV3(subject=None, operands=(), diff_ref=None),
            selection=projected_selection,
            capabilities=(focus_capability, runtime_start_capability, export_capability, correction_capability, review_capability),
            cursor_ref=head.cursor_ref,
            projection_hash=None,
            material_context_revision=head.revision,
        )
        draft = UnifiedWorkbenchContextV3(
            knowledge=KnowledgeWorkbenchContextV6(**shared),
            **shared,
        )
        wire = serialize_unified_workbench_context_v3(draft)
        digest = "sha256:" + content_hash(projection_hash_basis(wire))
        shared.update(projection_hash=digest)
        result = UnifiedWorkbenchContextV3(
            knowledge=KnowledgeWorkbenchContextV6(**shared),
            **shared,
        )
        return result

    def resolve_temporal_report_inputs(
        self,
        *,
        ctx: RequestContext,
        cursor_store: WorkbenchProjectionCursorStore,
        command: RuntimeReportCommandV1,
    ) -> VerifiedRuntimeReportInputsV1:
        if self._evaluation_store is None or self._authority_resolver is None:
            raise RuntimeContextError("INVALID_STATE")
        command = RuntimeReportCommandV1.model_validate(command.model_dump(mode="json"))
        scope = cursor_store._scope(ctx)
        actor_context = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
            session_id=scope["session_id"],
        )
        run = self._evaluation_store.get_run(
            actor_context=actor_context,
            pin=command.run_pin,
        )
        state = self._evaluation_store.get_current_state(
            actor_context=actor_context,
            run=command.run_pin,
        )
        admission = self._evaluation_store.get_admission(
            actor_context=actor_context,
            pin=run.admission,
        )
        if admission.context.artifact_kind != "context_graph":
            raise RuntimeContextError("UNSUPPORTED_ARTIFACT_KIND")
        resolved_context = self._authority_resolver.resolve_pins(
            actor_context=actor_context, pins=(admission.context,)
        )
        if (
            len(resolved_context) != 1
            or not isinstance(resolved_context[0], ContextGraph)
            or resolved_context[0].state != "ADMITTED"
            or admission.world != admission.context
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        runtime_context = resolved_context[0]
        kernel_context = True
        if tuple(state.decisions) != tuple(command.result_pins):
            raise RuntimeContextError("HASH_MISMATCH")
        request = self._evaluation_store.get_provider_request(
            actor_context=actor_context,
            pin=admission.provider_request,
        )
        if request.context != admission.context or request.world != admission.world:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        if command.disposition_pins:
            raise RuntimeContextError("INCOMPATIBLE_SUBJECT")
        expected_profile = server_runtime_report_profile_pins()
        if (
            command.projector_pin,
            command.renderer_pin,
            command.template_pin,
            command.environment_pin,
        ) != expected_profile:
            raise RuntimeContextError("HASH_MISMATCH")
        if len(request.source_pins) != 1:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        resolved_graph = self._authority_resolver.resolve_pins(
            actor_context=actor_context,
            pins=request.source_pins,
        )
        if len(resolved_graph) != 1 or not isinstance(resolved_graph[0], RuntimeTemporalSourceGraphV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")

        findings = ()
        unknown_results = ()
        resolution_requirements = ()
        if kernel_context:
            decisions: list[DecisionRecord] = []
            provider_result_keys = {
                (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in state.results
            }
            for pin in command.result_pins:
                resolved = self._authority_resolver.resolve_pins(
                    actor_context=actor_context, pins=(pin,)
                )
                if len(resolved) != 1 or not isinstance(resolved[0], DecisionRecord):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                decision = resolved[0]
                if (
                    decision.context != admission.context
                    or decision.purpose != "EVALUATION"
                    or decision.outcome_observation is None
                    or (
                        decision.outcome_observation.artifact_kind,
                        decision.outcome_observation.artifact_ref,
                        decision.outcome_observation.artifact_hash,
                    ) not in provider_result_keys
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                self._evaluation_store.get_result(
                    actor_context=actor_context, pin=decision.outcome_observation
                )
                decisions.append(decision)
            findings = tuple(
                RuntimeReportFindingV1(
                    result_pin=decision.pin(),
                    finding_pin=None,
                    target_evidence_pins=decision.evidence,
                    anchor=None,
                    excerpt=None,
                    observed_facts={
                        "semantic_result": decision.result_code,
                        "observation_ref": decision.outcome_observation.artifact_ref,
                    },
                    ambiguities=decision.reasoning.diagnostics,
                    outcome=(
                        decision.result_code
                        if decision.result_code in {"MATCH", "NO_MATCH", "UNKNOWN"}
                        else "UNKNOWN"
                    ),
                    candidate_rule_trace_pins=tuple(
                        rule
                        for invocation in decision.reasoning.invocations
                        for rule in invocation.rules
                    ),
                    predicate_result_pins=(),
                    exception_result_pins=(),
                    missing_inputs=(),
                    policy_basis=ResolvedRuntimeReportPolicyBasisV1(
                        status="RESOLVED", evidence_pins=decision.evidence
                    ),
                )
                for decision in decisions
            )
            unknown_results = tuple(
                decision.pin()
                for decision in decisions
                if decision.epistemic_status == "UNKNOWN" or decision.result_code == "UNKNOWN"
            )
            if unknown_results:
                resolution_requirements = (
                    ResolutionRequirementV1(
                        kind="EVIDENCE",
                        description="Provide additional evidence sufficient to resolve the UNKNOWN semantic result",
                        requested_evidence_type="additional_evidence",
                        decision_owner=None,
                        related_predicate_or_rule_pin=None,
                        evidence_pins=tuple(
                            sorted(
                                {pin for decision in decisions for pin in decision.evidence},
                                key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash),
                            )
                        ),
                    ),
                )
        else:
            for pin in command.result_pins:
                self._evaluation_store.get_result(actor_context=actor_context, pin=pin)

        status = (
            "COMPLETE"
            if state.status == "COMPLETE"
            else "PARTIAL"
            if state.status == "PARTIAL"
            else "BLOCKED"
        )
        return VerifiedRuntimeReportInputsV1(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            target_pin=request.target.artifact,
            source_pin=resolved_graph[0].source,
            document_title="Runtime temporal evaluation",
            run_pin=command.run_pin,
            root_run_pin=run.root_run,
            parent_run_pin=run.parent_run,
            epoch_pin=admission.epoch,
            certified_plan_pin=admission.certified_plan,
            provider_identity_pin=request.provider,
            graph_pin=request.source_pins[0],
            release_pin=None,
            normative_ir_pin=None,
            resolver_pin=None,
            rule_version_pins=(),
            evaluator_pin=request.profile,
            context_pin=admission.context,
            world_pin=admission.world,
            transition_pin=(None if kernel_context else runtime_context.transition),
            result_pins=tuple(command.result_pins),
            evaluated_competency_pins=(),
            findings=findings,
            unknown_results=unknown_results,
            resolution_requirements=resolution_requirements,
            status=status,
            predecessor_report_pin=command.predecessor_report_pin,
            disposition_pins=(),
            business_effective_at=admission.business_effective_at,
            recorded_at=state.recorded_at,
        )

    def resolve_design_object_evidence(
        self,
        *,
        actor_context: RuntimeContextDTO,
        snapshot_pin: ArtifactPinV1,
        object_pin: ArtifactPinV1,
    ) -> tuple[str, ...]:
        """Read node/edge membership, never infer evidence from source proximity."""
        _, selected = self._design_object(
            actor_context=actor_context,
            snapshot_pin=snapshot_pin,
            object_pin=object_pin,
        )
        return selected.evidence_refs

    def _design_object(self, *, actor_context, snapshot_pin, object_pin):
        if self._authority_resolver is None:
            raise RuntimeContextError("INVALID_STATE")
        if snapshot_pin.artifact_kind != "exploratory_graph_snapshot":
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        snapshot = self._authority_resolver.resolve_pins(
            actor_context=actor_context, pins=(snapshot_pin,)
        )[0]
        if not isinstance(snapshot, ExploratoryGraphSnapshotV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        if object_pin.artifact_kind == "exploratory_graph_node":
            matches = [
                node
                for node in snapshot.nodes
                if node.node_id == object_pin.artifact_ref
            ]
        elif object_pin.artifact_kind == "exploratory_graph_edge":
            matches = [
                edge
                for edge in snapshot.edges
                if edge.edge_id == object_pin.artifact_ref
            ]
        else:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        if len(matches) != 1:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        selected = matches[0]
        if object_pin.artifact_hash != "sha256:" + content_hash(
            selected.model_dump(mode="json")
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        self._authority_resolver.require_current_actor(actor_context=actor_context)
        return snapshot, selected

    def resolve_design_evidence_locations(
        self,
        *,
        ctx: RequestContext,
        snapshot_pin: ArtifactPinV1,
        object_pin: ArtifactPinV1,
        artifact_pin: ArtifactPinV1,
        preparation_pin: ArtifactPinV1,
    ) -> tuple[ResolvedLocation, ...]:
        """Resolve saved object membership through exact prepared source owners."""
        return tuple(
            member.location
            for member in self.resolve_design_evidence_members(
                ctx=ctx,
                snapshot_pin=snapshot_pin,
                object_pin=object_pin,
                artifact_pin=artifact_pin,
                preparation_pin=preparation_pin,
            )
        )

    def resolve_design_evidence_members(
        self,
        *,
        ctx: RequestContext,
        snapshot_pin: ArtifactPinV1,
        object_pin: ArtifactPinV1,
        artifact_pin: ArtifactPinV1,
        preparation_pin: ArtifactPinV1,
        selected_evidence_pin: ArtifactPinV1 | None = None,
        occurrence_ref: str | None = None,
    ) -> tuple[WorkbenchEvidenceMemberV1, ...]:
        """Keep complete membership while validating any singular focus pair.

        These are owner inputs, not a new transport. The action owner still owns
        selection/cursor CAS and publication; this method never changes focus.
        """
        if self._preparation_service is None:
            raise RuntimeContextError("INVALID_STATE")
        if (
            not isinstance(ctx, RequestContext)
            or not ctx.authenticated
            or not ctx.organization_id
        ):
            raise RuntimeContextError("FORBIDDEN_ACTOR")
        actor = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
        )
        snapshot, selected = self._design_object(
            actor_context=actor, snapshot_pin=snapshot_pin, object_pin=object_pin
        )
        identity, document = self._preparation_service.resolve_design_source(
            ctx=ctx,
            artifact_pin=artifact_pin,
            preparation_pin=preparation_pin,
            expected_ir_pin=ArtifactPinV1(
                artifact_kind="canonical_source_ir",
                artifact_ref=snapshot.canonical_source_ir_ref,
                artifact_hash=snapshot.canonical_source_ir_hash,
            ),
        )
        if (snapshot.source_identity_ref, snapshot.source_hash) != (
            identity.source_identity_ref,
            identity.source_hash,
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        evidence = {}
        for block in document.blocks:
            ref = evidence_ref_for_block(
                document,
                block,
                source_identity=identity,
                created_at=identity.created_at,
            )
            evidence.setdefault(ref.evidence_ref, []).append(ref)
        members = []
        for evidence_ref in selected.evidence_refs:
            matches = evidence.get(evidence_ref, [])
            if len(matches) == 1:
                ref = matches[0]
                pin = evidence_ref_pin(ref)
                occurrence = "evidence-occurrence:" + content_hash(
                    dict(
                        source_identity_ref=identity.source_identity_ref,
                        canonical_source_ir_pin=dict(
                            artifact_kind="canonical_source_ir",
                            artifact_ref=snapshot.canonical_source_ir_ref,
                            artifact_hash=snapshot.canonical_source_ir_hash,
                        ),
                        evidence_pin=pin.model_dump(mode="json"),
                        source_block_id=ref.provenance["source_block_id"],
                    )
                )
                location = resolve_evidence_ref(
                    ref,
                    document,
                    artifact_ref=identity.artifact_ref,
                    selector_hash=ref.selector_hash,
                )
            else:
                pin, occurrence = None, None
                location = ResolvedLocation(
                    schemaVersion="semantier.resolved_location.v1",
                    evidenceRef=evidence_ref,
                    sourceIdentityRef=identity.source_identity_ref,
                    representationType="docx_structural",
                    representationRef=identity.artifact_ref,
                    representationHash=identity.source_hash,
                    locationType="other",
                    status="unresolved",
                    location={
                        "kind": "unresolved",
                        "reason": "saved evidence has no unique canonical source member",
                    },
                    matchedText=None,
                    resolverVersion="workbench.source-membership.v1",
                    resolvedAt=datetime.now(timezone.utc),
                )
            members.append(
                WorkbenchEvidenceMemberV1(
                    schema_version="workbench_evidence_member.v1",
                    evidence_ref=evidence_ref,
                    evidence_pin=pin,
                    occurrence_ref=occurrence,
                    location=location,
                )
            )
        self._authority_resolver.require_current_actor(actor_context=actor)
        if selected_evidence_pin is not None or occurrence_ref is not None:
            selected_members = [
                member
                for member in members
                if selected_evidence_pin is not None
                and member.evidence_pin == selected_evidence_pin
                and occurrence_ref is not None
                and member.occurrence_ref == occurrence_ref
            ]
            if (
                len(selected_members) != 1
                or selected_members[0].location.status != "exact"
            ):
                raise RuntimeContextError("UNRESOLVABLE_REF")
        return tuple(members)

    def _resolve_design_evidence_members_from_bindings(
        self,
        *,
        ctx: RequestContext,
        snapshot_pin: ArtifactPinV1,
        object_pin: ArtifactPinV1,
        source_entries: tuple[LifecycleArtifactBindingEntryV1, ...],
        selected_evidence_pin: ArtifactPinV1 | None = None,
        occurrence_ref: str | None = None,
    ) -> tuple[
        tuple[WorkbenchEvidenceMemberV1, ...],
        dict[str, LifecycleArtifactBindingEntryV1],
        dict[str, tuple[object, LifecycleArtifactBindingEntryV1]],
    ]:
        """Resolve one object against the exact active DESIGN source set.

        The binding set is already a cursor-pinned authority input. This helper
        only projects evidence membership; it creates no new source or graph
        authority and preserves the existing single-source method for callers
        that intentionally bind one source.
        """
        if self._preparation_service is None:
            raise RuntimeContextError("INVALID_STATE")
        if (
            not isinstance(ctx, RequestContext)
            or not ctx.authenticated
            or not ctx.organization_id
        ):
            raise RuntimeContextError("FORBIDDEN_ACTOR")
        actor = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
        )
        snapshot, selected = self._design_object(
            actor_context=actor, snapshot_pin=snapshot_pin, object_pin=object_pin
        )
        active = tuple(
            entry
            for entry in source_entries
            if entry.lifecycle == "DESIGN"
            and entry.role == "SOURCE"
            and entry.state == "ACTIVE"
            and entry.preparation_pin is not None
            and entry.source_identity_pin is not None
            and entry.canonical_source_ir_pin is not None
        )
        if not active:
            raise RuntimeContextError("SOURCE_NOT_BOUND")
        primary = tuple(
            entry
            for entry in active
            if entry.source_identity_pin is not None
            and entry.source_identity_pin.artifact_ref == snapshot.source_identity_ref
            and entry.canonical_source_ir_pin
            == ArtifactPinV1(
                artifact_kind="canonical_source_ir",
                artifact_ref=snapshot.canonical_source_ir_ref,
                artifact_hash=snapshot.canonical_source_ir_hash,
            )
        )
        if not primary:
            raise RuntimeContextError("HASH_MISMATCH")
        if len(primary) != 1:
            raise RuntimeContextError("SOURCE_NOT_BOUND")

        resolved_sources: dict[
            str, tuple[object, object, LifecycleArtifactBindingEntryV1]
        ] = {}
        candidates: dict[
            str,
            list[
                tuple[
                    object,
                    object,
                    object,
                    LifecycleArtifactBindingEntryV1,
                ]
            ],
        ] = {}
        for entry in active:
            identity, document = self._preparation_service.resolve_design_source(
                ctx=ctx,
                artifact_pin=entry.artifact_pin,
                preparation_pin=entry.preparation_pin,
                expected_ir_pin=entry.canonical_source_ir_pin,
            )
            expected_identity_pin = ArtifactPinV1(
                artifact_kind="source_identity",
                artifact_ref=identity.source_identity_ref,
                artifact_hash="sha256:"
                + content_hash(
                    identity.model_dump(
                        mode="json", by_alias=True, exclude_none=False
                    )
                ),
            )
            if entry.source_identity_pin != expected_identity_pin:
                raise RuntimeContextError("HASH_MISMATCH")
            if entry == primary[0] and identity.source_hash != snapshot.source_hash:
                raise RuntimeContextError("HASH_MISMATCH")
            resolved_sources[identity.source_identity_ref] = (identity, document, entry)
            for block in document.blocks:
                ref = evidence_ref_for_block(
                    document,
                    block,
                    source_identity=identity,
                    created_at=identity.created_at,
                )
                if ref.evidence_ref in selected.evidence_refs:
                    candidates.setdefault(ref.evidence_ref, []).append(
                        (ref, identity, document, entry)
                    )

        primary_identity, _primary_document, primary_entry = resolved_sources[
            snapshot.source_identity_ref
        ]
        projected: list[
            tuple[WorkbenchEvidenceMemberV1, LifecycleArtifactBindingEntryV1]
        ] = []
        owner_by_ref: dict[str, LifecycleArtifactBindingEntryV1] = {}
        for evidence_ref in selected.evidence_refs:
            matches = candidates.get(evidence_ref, [])
            owner = primary_entry
            if len(matches) == 1:
                ref, identity, document, owner = matches[0]
                pin = evidence_ref_pin(ref)
                occurrence = "evidence-occurrence:" + content_hash(
                    dict(
                        source_identity_ref=identity.source_identity_ref,
                        canonical_source_ir_pin=owner.canonical_source_ir_pin.model_dump(
                            mode="json"
                        ),
                        evidence_pin=pin.model_dump(mode="json"),
                        source_block_id=ref.provenance["source_block_id"],
                    )
                )
                location = resolve_evidence_ref(
                    ref,
                    document,
                    artifact_ref=identity.artifact_ref,
                    selector_hash=ref.selector_hash,
                )
            else:
                pin, occurrence = None, None
                location = ResolvedLocation(
                    schemaVersion="semantier.resolved_location.v1",
                    evidenceRef=evidence_ref,
                    sourceIdentityRef=primary_identity.source_identity_ref,
                    representationType="docx_structural",
                    representationRef=primary_identity.artifact_ref,
                    representationHash=primary_identity.source_hash,
                    locationType="other",
                    status="unresolved",
                    location={
                        "kind": "unresolved",
                        "reason": "saved evidence has no unique bound source member",
                    },
                    matchedText=None,
                    resolverVersion="workbench.source-membership.v1",
                    resolvedAt=datetime.now(timezone.utc),
                )
            member = WorkbenchEvidenceMemberV1(
                schema_version="workbench_evidence_member.v1",
                evidence_ref=evidence_ref,
                evidence_pin=pin,
                occurrence_ref=occurrence,
                location=location,
            )
            projected.append((member, owner))
            owner_by_ref[evidence_ref] = owner

        # Source groups are ordered by first occurrence. Membership order inside
        # each group stays the object's original evidence order. This matches the
        # frozen multi-document projection contract and keeps one viewer per doc.
        source_order: list[str] = []
        grouped: dict[
            str, list[tuple[WorkbenchEvidenceMemberV1, LifecycleArtifactBindingEntryV1]]
        ] = {}
        for item in projected:
            source_ref = item[1].source_identity_pin.artifact_ref
            if source_ref not in grouped:
                source_order.append(source_ref)
                grouped[source_ref] = []
            grouped[source_ref].append(item)
        ordered_pairs = tuple(item for ref in source_order for item in grouped[ref])
        members = tuple(member for member, _entry in ordered_pairs)

        self._authority_resolver.require_current_actor(actor_context=actor)
        if selected_evidence_pin is not None or occurrence_ref is not None:
            selected_members = tuple(
                member
                for member in members
                if selected_evidence_pin is not None
                and member.evidence_pin == selected_evidence_pin
                and occurrence_ref is not None
                and member.occurrence_ref == occurrence_ref
            )
            if (
                len(selected_members) != 1
                or selected_members[0].location.status != "exact"
            ):
                raise RuntimeContextError("UNRESOLVABLE_REF")
        meta = {
            ref: (resolved_sources[ref][0], resolved_sources[ref][2])
            for ref in source_order
        }
        return members, owner_by_ref, meta

    def project_design_evidence(
        self,
        *,
        ctx: RequestContext,
        snapshot: ExploratoryGraphSnapshotV1,
        snapshot_pin: ArtifactPinV1,
        artifact_pin: ArtifactPinV1,
        preparation_pin: ArtifactPinV1,
        intent: WorkbenchEvidenceSelectionIntentV1,
        expected_selection_pin: ArtifactPinV1,
        expected_selection_revision: int,
        source_entries: tuple[LifecycleArtifactBindingEntryV1, ...] | None = None,
    ) -> EvidenceFocusProjectionV1:
        """Project exact DESIGN evidence/action inputs from governed owners only."""
        if self._preparation_service is None:
            raise RuntimeContextError("INVALID_STATE")
        candidates = (
            tuple(node for node in snapshot.nodes if node.node_id == intent.object_ref)
            if intent.object_kind == "node"
            else tuple(edge for edge in snapshot.edges if edge.edge_id == intent.object_ref)
        )
        if len(candidates) != 1:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        selected = candidates[0]
        object_pin = ArtifactPinV1(
            artifact_kind=(
                "exploratory_graph_node"
                if intent.object_kind == "node"
                else "exploratory_graph_edge"
            ),
            artifact_ref=intent.object_ref,
            artifact_hash="sha256:" + content_hash(selected.model_dump(mode="json")),
        )
        owner_by_ref: dict[str, LifecycleArtifactBindingEntryV1] = {}
        source_meta: dict[str, tuple[object, LifecycleArtifactBindingEntryV1]] = {}
        active_source_entries = tuple(
            entry
            for entry in (source_entries or ())
            if entry.lifecycle == "DESIGN"
            and entry.role == "SOURCE"
            and entry.state == "ACTIVE"
        )
        if len(active_source_entries) <= 1:
            members = self.resolve_design_evidence_members(
                ctx=ctx,
                snapshot_pin=snapshot_pin,
                object_pin=object_pin,
                artifact_pin=artifact_pin,
                preparation_pin=preparation_pin,
            )
            source_entries = None
        else:
            members, owner_by_ref, source_meta = (
                self._resolve_design_evidence_members_from_bindings(
                    ctx=ctx,
                    snapshot_pin=snapshot_pin,
                    object_pin=object_pin,
                    source_entries=active_source_entries,
                )
            )
        exact_members = tuple(
            member
            for member in members
            if member.evidence_pin is not None and member.location.status == "exact"
        )
        exact_pins = tuple(member.evidence_pin for member in exact_members)
        selected_member = None
        if intent.evidence_ref is not None:
            matches = tuple(
                member
                for member in exact_members
                if member.evidence_ref == intent.evidence_ref
            )
            if len(matches) != 1 or matches[0].occurrence_ref is None:
                raise RuntimeContextError("UNRESOLVABLE_REF")
            selected_member = matches[0]
            if (
                intent.occurrence_ref is not None
                and intent.occurrence_ref != selected_member.occurrence_ref
            ):
                raise RuntimeContextError("UNRESOLVABLE_REF")
        selected_pin = selected_member.evidence_pin if selected_member is not None else None
        selected_occurrence = (
            selected_member.occurrence_ref if selected_member is not None else None
        )
        canonical_anchor = (
            ArtifactPinV1(
                artifact_kind="source_anchor",
                artifact_ref=selected_occurrence,
                artifact_hash="sha256:"
                + selected_occurrence.removeprefix("evidence-occurrence:"),
            )
            if selected_occurrence is not None
            else None
        )
        if source_entries is None:
            source_identity, _ = self._preparation_service.resolve_design_source(
                ctx=ctx,
                artifact_pin=artifact_pin,
                preparation_pin=preparation_pin,
                expected_ir_pin=ArtifactPinV1(
                    artifact_kind="canonical_source_ir",
                    artifact_ref=snapshot.canonical_source_ir_ref,
                    artifact_hash=snapshot.canonical_source_ir_hash,
                ),
            )
            source_meta = {
                source_identity.source_identity_ref: (source_identity, None)
            }
            owner_by_ref = {
                member.evidence_ref: next(
                    (
                        entry
                        for entry in (source_entries or ())
                        if entry.artifact_pin == artifact_pin
                        and entry.preparation_pin == preparation_pin
                    ),
                    None,
                )
                for member in members
            }

        source_groups = []
        group_refs: dict[str, list[str]] = {}
        group_order: list[str] = []
        for member in members:
            source_ref = member.location.source_identity_ref
            if source_ref not in group_refs:
                group_order.append(source_ref)
                group_refs[source_ref] = []
            group_refs[source_ref].append(member.evidence_ref)
        for source_ref in group_order:
            source_identity, source_entry = source_meta[source_ref]
            media_type = source_identity.media_type.lower()
            if media_type == "application/pdf":
                document_kind = "pdf"
            elif media_type in {
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.ms-word.document.macroenabled.12",
            }:
                document_kind = "docx"
            elif "json" in media_type or "canonical" in media_type:
                document_kind = "canonical_source_ir"
            else:
                document_kind = "unknown"
            source_groups.append(
                EvidenceSourceGroupV1(
                    source_identity_ref=source_identity.source_identity_ref,
                    source_hash=source_identity.source_hash,
                    artifact_ref=source_identity.artifact_ref,
                    document_name=source_identity.original_name,
                    document_kind=document_kind,
                    evidence_refs=tuple(group_refs[source_ref]),
                )
            )
        payload = EvidenceFocusPayloadV1(
            command="FOCUS_EVIDENCE",
            object_pin=object_pin,
            evidence_refs=exact_pins,
            selected_evidence_ref=selected_pin,
            canonical_anchor=canonical_anchor,
            target_view="EVIDENCE",
            expected_selection_pin=expected_selection_pin,
            expected_selection_revision=expected_selection_revision,
        )
        return EvidenceFocusProjectionV1(
            status="READY",
            object_pin=object_pin,
            members=members,
            source_groups=tuple(source_groups),
            selected_evidence_ref=(
                selected_member.evidence_ref if selected_member is not None else None
            ),
            selected_occurrence_ref=selected_occurrence,
            action_binding=EvidenceActionBindingV1(payload=payload),
            error_code=None,
        )

    def publish_design_evidence_focus_on_connection(
        self,
        connection,
        *,
        ctx: RequestContext,
        cursor_store: WorkbenchProjectionCursorStore,
        cursor_ref: str,
        expected_cursor_pin: ArtifactPinV1,
        expected_material_context_revision: int,
        expected_projection_hash: str,
        snapshot_pin: ArtifactPinV1,
        artifact_pin: ArtifactPinV1,
        preparation_pin: ArtifactPinV1,
        focus: EvidenceFocusPayloadV1,
    ):
        """Append validated focus and cursor inside the action owner's UoW.

        No transport, commit, or request receipt is created here. The dispatcher
        must append its acknowledgement/index before committing this transaction.
        Any exception requires the caller to roll back the whole UoW.
        """
        if connection is not self._store._connection or not in_transaction(connection):
            raise RuntimeError("focus requires the active workspace transaction")
        if (
            type(cursor_store) is not WorkbenchProjectionCursorStore
            or cursor_store._store is not self._store
            or cursor_store._authority is not self._authority_resolver
        ):
            raise RuntimeContextError("INVALID_STATE")
        focus = EvidenceFocusPayloadV1.model_validate(focus.model_dump(mode="json"))
        scope = cursor_store._scope(ctx)
        previous = cursor_store.get(ctx=ctx, cursor_ref=cursor_ref)
        previous_pin = ArtifactPinV1(
            artifact_kind="workbench_projection_cursor_head",
            artifact_ref=previous.cursor_head_ref,
            artifact_hash=previous.cursor_head_hash,
        )
        if type(expected_material_context_revision) is not int or (
            expected_cursor_pin,
            expected_material_context_revision,
            expected_projection_hash,
        ) != (previous_pin, previous.revision, previous.projection_basis_hash):
            raise ValueError("STALE_MATERIAL_CONTEXT")
        selection, design, _ = cursor_store._facts(
            scope,
            previous.selection_pin,
            previous.design_binding_set_pin,
            previous.runtime_binding_set_pin,
        )
        if (focus.expected_selection_pin, focus.expected_selection_revision) != (
            previous.selection_pin,
            selection.revision,
        ):
            raise ValueError("STALE_SELECTION_HEAD")
        if self._preparation_service is None:
            raise RuntimeContextError("INVALID_STATE")
        artifacts = self._preparation_service._artifacts
        current_selection = artifacts._head(
            ctx,
            "workspace_artifact_selection_heads",
            type(selection),
            "selection_ref",
            "selection_hash",
        )
        current_design = artifacts._head(
            ctx,
            "workspace_artifact_binding_set_heads",
            type(design),
            "binding_set_ref",
            "binding_set_hash",
            extra=("DESIGN", "SOURCE"),
        )
        if current_selection != selection:
            raise ValueError("STALE_SELECTION_HEAD")
        if current_design != design:
            raise ValueError("STALE_BINDING_SET_HEAD")
        if not any(
            entry.artifact_pin == artifact_pin
            and entry.preparation_pin == preparation_pin
            and entry.state == "ACTIVE"
            for entry in design.entries
        ):
            raise ValueError("SOURCE_NOT_BOUND")
        active_source_entries = tuple(
            entry
            for entry in design.entries
            if entry.lifecycle == "DESIGN"
            and entry.role == "SOURCE"
            and entry.state == "ACTIVE"
        )
        if len(active_source_entries) <= 1:
            members = self.resolve_design_evidence_members(
                ctx=ctx,
                snapshot_pin=snapshot_pin,
                object_pin=focus.object_pin,
                artifact_pin=artifact_pin,
                preparation_pin=preparation_pin,
            )
            owner_by_ref = {
                member.evidence_ref: active_source_entries[0]
                for member in members
                if active_source_entries
            }
        else:
            members, owner_by_ref, _source_meta = (
                self._resolve_design_evidence_members_from_bindings(
                    ctx=ctx,
                    snapshot_pin=snapshot_pin,
                    object_pin=focus.object_pin,
                    source_entries=active_source_entries,
                )
            )
        # Caller pins are assertions, not membership authority. In aggregate mode
        # unresolved rows survive, but no resolvable member may be omitted.
        exact_pins = tuple(
            member.evidence_pin
            for member in members
            if member.evidence_pin is not None and member.location.status == "exact"
        )
        if focus.evidence_refs != exact_pins:
            raise RuntimeContextError("UNRESOLVABLE_REF")
        occurrence_ref = None
        if focus.selected_evidence_ref is None:
            if focus.canonical_anchor is not None:
                raise RuntimeContextError("UNRESOLVABLE_REF")
        else:
            candidates = [
                member
                for member in members
                if member.evidence_pin == focus.selected_evidence_ref
                and member.location.status == "exact"
            ]
            if not candidates or candidates[0].occurrence_ref is None:
                raise RuntimeContextError("UNRESOLVABLE_REF")
            # Current canonical-block projection has one occurrence per member.
            # Anchor identity is the existing server occurrence content address,
            # never a rectangle, DOM selector, or caller-provided membership.
            anchors = [
                ArtifactPinV1(
                    artifact_kind="source_anchor",
                    artifact_ref=member.occurrence_ref,
                    artifact_hash="sha256:"
                    + member.occurrence_ref.removeprefix("evidence-occurrence:"),
                )
                for member in candidates
            ]
            anchor = focus.canonical_anchor or anchors[0]
            if anchors.count(anchor) != 1:
                raise RuntimeContextError("UNRESOLVABLE_REF")
            occurrence_ref = anchor.artifact_ref
            focus = focus.model_copy(update={"canonical_anchor": anchor})
        focus_owner = (
            owner_by_ref.get(focus.selected_evidence_ref.artifact_ref)
            if focus.selected_evidence_ref is not None
            else None
        )
        fact_artifact_pin = (
            focus_owner.artifact_pin if focus_owner is not None else artifact_pin
        )
        fact_preparation_pin = (
            focus_owner.preparation_pin
            if focus_owner is not None and focus_owner.preparation_pin is not None
            else preparation_pin
        )
        predecessor_pin = getattr(previous, "evidence_focus_pin", None)
        predecessor = cursor_store._focus(scope, predecessor_pin)
        now = datetime.now(timezone.utc)
        basis = dict(
            **scope,
            schema_version="workbench_evidence_focus.v1",
            lifecycle="DESIGN",
            revision=predecessor.revision + 1 if predecessor else 0,
            predecessor_pin=predecessor_pin.model_dump(mode="json")
            if predecessor_pin
            else None,
            parent_pin=snapshot_pin.model_dump(mode="json"),
            artifact_pin=fact_artifact_pin.model_dump(mode="json"),
            preparation_pin=fact_preparation_pin.model_dump(mode="json"),
            occurrence_ref=occurrence_ref,
            focus=focus.model_dump(mode="json"),
            recorded_at=now.isoformat().replace("+00:00", "Z"),
        )
        digest = content_hash(basis)
        fact = WorkbenchEvidenceFocusV1.model_validate(
            {
                **basis,
                "focus_ref": "workbench-evidence-focus:" + digest,
                "focus_hash": "sha256:" + digest,
            }
        )
        fact_pin = ArtifactPinV1(
            artifact_kind="workbench_evidence_focus",
            artifact_ref=fact.focus_ref,
            artifact_hash=fact.focus_hash,
        )
        material = dict(
            **scope,
            cursor_ref=cursor_ref,
            revision=previous.revision + 1,
            selection_pin=previous.selection_pin.model_dump(mode="json"),
            design_binding_set_pin=previous.design_binding_set_pin.model_dump(
                mode="json"
            ),
            runtime_binding_set_pin=previous.runtime_binding_set_pin.model_dump(
                mode="json"
            ),
            evidence_focus_pin=fact_pin.model_dump(mode="json"),
            design_snapshot_pin=previous.design_snapshot_pin.model_dump(mode="json")
            if previous.design_snapshot_pin
            else None,
        )
        self._store._insert_prepared((self._store._prepare(fact),))
        head = cursor_store.advance_on_connection(
            connection,
            ctx=ctx,
            cursor_ref=cursor_ref,
            expected_head_pin=previous_pin,
            expected_revision=previous.revision,
            expected_projection_basis_hash=previous.projection_basis_hash,
            selection_pin=previous.selection_pin,
            design_binding_set_pin=previous.design_binding_set_pin,
            runtime_binding_set_pin=previous.runtime_binding_set_pin,
            evidence_focus_pin=fact_pin,
            successor_projection_basis_hash="sha256:" + content_hash(material),
            recorded_at=now,
        )
        cursor_store._scope(ctx)
        return fact, head, members

    def read_design_evidence_focus(
        self,
        *,
        ctx: RequestContext,
        cursor_store: WorkbenchProjectionCursorStore,
        cursor_ref: str,
    ) -> tuple[
        WorkbenchProjectionCursorHeadV3,
        WorkbenchEvidenceFocusV1 | None,
        tuple[WorkbenchEvidenceMemberV1, ...],
    ]:
        """Read the pinned focus in one snapshot without dispatching or repairing.

        This supplies stored navigation facts to the projection owner; it does
        not issue capabilities or claim that a historical binding is active.
        """
        if (
            type(cursor_store) is not WorkbenchProjectionCursorStore
            or cursor_store._store is not self._store
            or cursor_store._authority is not self._authority_resolver
        ):
            raise RuntimeContextError("INVALID_STATE")
        connection = self._store._connection
        owns_snapshot = not in_transaction(connection)
        if owns_snapshot:
            connection.execute("BEGIN")
        try:
            scope = cursor_store._scope(ctx)
            head = cursor_store.get(ctx=ctx, cursor_ref=cursor_ref)
            fact = cursor_store._focus(scope, getattr(head, "evidence_focus_pin", None))
            _selection, design, _runtime = cursor_store._facts(
                scope,
                head.selection_pin,
                head.design_binding_set_pin,
                head.runtime_binding_set_pin,
            )
            members = (
                self._resolve_saved_design_focus(
                    ctx=ctx,
                    fact=fact,
                    source_entries=tuple(design.entries),
                )
                if fact is not None
                else ()
            )
            cursor_store._scope(ctx)
            return head, fact, members
        finally:
            if owns_snapshot:
                connection.rollback()

    def _resolve_saved_design_focus(
        self,
        *,
        ctx,
        fact,
        source_entries: tuple[LifecycleArtifactBindingEntryV1, ...] | None = None,
    ):
        if fact.lifecycle != "DESIGN":
            raise RuntimeContextError("INVALID_STATE")
        active_source_entries = tuple(
            entry
            for entry in (source_entries or ())
            if entry.lifecycle == "DESIGN"
            and entry.role == "SOURCE"
            and entry.state == "ACTIVE"
        )
        if len(active_source_entries) <= 1:
            members = self.resolve_design_evidence_members(
                ctx=ctx,
                snapshot_pin=fact.parent_pin,
                object_pin=fact.focus.object_pin,
                artifact_pin=fact.artifact_pin,
                preparation_pin=fact.preparation_pin,
                selected_evidence_pin=fact.focus.selected_evidence_ref,
                occurrence_ref=fact.occurrence_ref,
            )
        else:
            members, _owners, _meta = (
                self._resolve_design_evidence_members_from_bindings(
                    ctx=ctx,
                    snapshot_pin=fact.parent_pin,
                    object_pin=fact.focus.object_pin,
                    source_entries=active_source_entries,
                    selected_evidence_pin=fact.focus.selected_evidence_ref,
                    occurrence_ref=fact.occurrence_ref,
                )
            )
            if not any(
                entry.artifact_pin == fact.artifact_pin
                and entry.preparation_pin == fact.preparation_pin
                for entry in active_source_entries
            ):
                raise ValueError("SOURCE_NOT_BOUND")
        if (
            tuple(
                member.evidence_pin
                for member in members
                if member.evidence_pin is not None and member.location.status == "exact"
            )
            != fact.focus.evidence_refs
        ):
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        expected_anchor = (
            ArtifactPinV1(
                artifact_kind="source_anchor",
                artifact_ref=fact.occurrence_ref,
                artifact_hash="sha256:"
                + fact.occurrence_ref.removeprefix("evidence-occurrence:"),
            )
            if fact.occurrence_ref
            else None
        )
        if fact.focus.canonical_anchor != expected_anchor:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        return members

    def focus_design_evidence(
        self,
        *,
        ctx: RequestContext,
        cursor_store: WorkbenchProjectionCursorStore,
        command: FocusEvidenceActionEnvelopeV6,
        snapshot_pin: ArtifactPinV1,
        artifact_pin: ArtifactPinV1,
        preparation_pin: ArtifactPinV1,
    ) -> WorkbenchActionAcknowledgementV6:
        """Durable DESIGN owner operation; the full v6 transport wraps this result."""
        command = FocusEvidenceActionEnvelopeV6.model_validate(
            command.model_dump(mode="json")
        )
        return self._store.focus_workbench_evidence(
            service=self,
            ctx=ctx,
            cursor_store=cursor_store,
            command=command,
            snapshot_pin=snapshot_pin,
            artifact_pin=artifact_pin,
            preparation_pin=preparation_pin,
        )

    def _focus_design_evidence_on_connection(
        self,
        connection,
        *,
        ctx,
        cursor_store,
        command,
        snapshot_pin,
        artifact_pin,
        preparation_pin,
    ):
        if connection is not self._store._connection or not in_transaction(connection):
            raise RuntimeError("focus requires the active workspace transaction")
        if (
            type(cursor_store) is not WorkbenchProjectionCursorStore
            or cursor_store._store is not self._store
            or cursor_store._authority is not self._authority_resolver
        ):
            raise RuntimeContextError("INVALID_STATE")
        command = FocusEvidenceActionEnvelopeV6.model_validate(
            command.model_dump(mode="json")
        )
        scope = cursor_store._scope(ctx)
        request_hash = "sha256:" + content_hash(command.model_dump(mode="json"))
        keys = (
            ("FOCUS_EVIDENCE", command.request_id),
            ("FOCUS_EVIDENCE_IDEMPOTENCY", command.idempotency_key),
        )
        rows = [
            connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                (
                    scope["organization_id"],
                    scope["workspace_id"],
                    scope["actor_ref"],
                    operation,
                    key,
                ),
            ).fetchone()
            for operation, key in keys
        ]
        if any(row is not None for row in rows):
            if any(row is not None and row[0] != request_hash for row in rows):
                raise ValueError("REQUEST_CONFLICT")
            if rows[0] is None or rows[0] != rows[1]:
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            acknowledgement = self._store.get(
                organization_id=scope["organization_id"],
                workspace_id=scope["workspace_id"],
                artifact_ref=rows[0][1],
                contract_type=WorkbenchActionAcknowledgementV6,
            )
            if acknowledgement is None or (
                acknowledgement.receipt_ref,
                acknowledgement.receipt_hash,
                acknowledgement.request_hash,
                acknowledgement.request_id,
                acknowledgement.idempotency_key,
                acknowledgement.command,
                acknowledgement.ingress,
            ) != (
                rows[0][1],
                rows[0][2],
                request_hash,
                command.request_id,
                command.idempotency_key,
                command.command,
                command.ingress,
            ):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            if any(
                getattr(acknowledgement, key) != value for key, value in scope.items()
            ):
                raise ValueError("REQUEST_CONFLICT")
            head = cursor_store.exact_head(ctx=ctx, pin=acknowledgement.cursor_pin)
            if (
                head.cursor_ref,
                head.revision,
                head.projection_basis_hash,
                getattr(head, "evidence_focus_pin", None),
            ) != (
                command.cursor_ref,
                acknowledgement.material_context_revision,
                acknowledgement.projection_hash,
                acknowledgement.result_pin,
            ):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            if (
                head.predecessor_pin is None
                or head.revision != command.expected_material_context_revision + 1
            ):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            before = cursor_store.exact_head(ctx=ctx, pin=head.predecessor_pin)
            if (before.revision, before.projection_basis_hash) != (
                command.expected_material_context_revision,
                command.expected_projection_hash,
            ):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            fact = cursor_store._focus(scope, acknowledgement.result_pin)
            if (fact.lifecycle, fact.parent_pin) != ("DESIGN", snapshot_pin):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            normalized = command.payload
            if (
                normalized.selected_evidence_ref is not None
                and normalized.canonical_anchor is None
            ):
                normalized = normalized.model_copy(
                    update={"canonical_anchor": fact.focus.canonical_anchor}
                )
            if fact.focus != normalized:
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            selection, design, _ = cursor_store._facts(
                scope,
                head.selection_pin,
                head.design_binding_set_pin,
                head.runtime_binding_set_pin,
            )
            if (head.selection_pin, selection.revision) != (
                command.payload.expected_selection_pin,
                command.payload.expected_selection_revision,
            ) or not any(
                entry.artifact_pin == artifact_pin
                and entry.preparation_pin == preparation_pin
                and entry.state == "ACTIVE"
                for entry in design.entries
            ) or not any(
                entry.artifact_pin == fact.artifact_pin
                and entry.preparation_pin == fact.preparation_pin
                and entry.state == "ACTIVE"
                for entry in design.entries
            ):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            self._resolve_saved_design_focus(
                ctx=ctx,
                fact=fact,
                source_entries=tuple(design.entries),
            )
            return acknowledgement
        previous = cursor_store.get(ctx=ctx, cursor_ref=command.cursor_ref)
        fact, head, _ = self.publish_design_evidence_focus_on_connection(
            connection,
            ctx=ctx,
            cursor_store=cursor_store,
            cursor_ref=command.cursor_ref,
            expected_cursor_pin=ArtifactPinV1(
                artifact_kind="workbench_projection_cursor_head",
                artifact_ref=previous.cursor_head_ref,
                artifact_hash=previous.cursor_head_hash,
            ),
            expected_material_context_revision=command.expected_material_context_revision,
            expected_projection_hash=command.expected_projection_hash,
            snapshot_pin=snapshot_pin,
            artifact_pin=artifact_pin,
            preparation_pin=preparation_pin,
            focus=command.payload,
        )
        basis = dict(
            **scope,
            schema_version="workbench_action_acknowledgement.v6",
            command=command.command,
            accepted=True,
            result_pin=head.evidence_focus_pin.model_dump(mode="json"),
            cursor_pin=dict(
                artifact_kind="workbench_projection_cursor_head",
                artifact_ref=head.cursor_head_ref,
                artifact_hash=head.cursor_head_hash,
            ),
            material_context_revision=head.revision,
            projection_hash=head.projection_basis_hash,
            request_id=command.request_id,
            idempotency_key=command.idempotency_key,
            request_hash=request_hash,
            ingress=command.ingress,
            recorded_at=fact.recorded_at.isoformat().replace("+00:00", "Z"),
        )
        digest = content_hash(basis)
        acknowledgement = WorkbenchActionAcknowledgementV6.model_validate(
            {
                **basis,
                "receipt_ref": "workbench-action-acknowledgement:" + digest,
                "receipt_hash": "sha256:" + digest,
            }
        )
        self._store._insert_prepared((self._store._prepare(acknowledgement),))
        for operation, key in keys:
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    scope["organization_id"],
                    scope["workspace_id"],
                    scope["actor_ref"],
                    operation,
                    key,
                    request_hash,
                    acknowledgement.receipt_ref,
                    acknowledgement.receipt_hash,
                ),
            )
        return acknowledgement

    def read_temporal_artifact(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> WorkbenchArtifactProjectionV2:
        if self._evaluation_store is None:
            raise RuntimeContextError("INVALID_STATE")
        if pin.artifact_kind in {"context_graph", "evidence"}:
            payload = self._authority_resolver.resolve_pins(
                actor_context=actor_context, pins=(pin,)
            )[0]
            if not isinstance(payload, (ContextGraph, Evidence)):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        elif pin.artifact_kind == "runtime_inspection_report":
            payload = RuntimeInspectionReportStore(self._store).get(
                scope=WorkspaceScope(
                    organization_id=actor_context.organization_id,
                    workspace_id=actor_context.workspace_id,
                ),
                report_pin=pin,
            )
        else:
            readers = {
                "runtime_temporal_source_archive": self._source_service.get_archive,
                "runtime_temporal_source_graph": self._source_service.get_graph,
                "runtime_provider_request": self._evaluation_store.get_provider_request,
                "runtime_evaluation_run": self._evaluation_store.get_run,
                "runtime_evaluation_state": self._evaluation_store.get_state,
                "runtime_provider_result": self._evaluation_store.get_result,
            }
            reader = readers.get(pin.artifact_kind)
            if reader is None:
                raise RuntimeContextError("OBJECT_NOT_FOUND")
            payload = reader(actor_context=actor_context, pin=pin)
        return WorkbenchArtifactProjectionV2(
            schema_version="workbench_artifact_projection.v2",
            artifact=pin,
            payload=payload,
        )
