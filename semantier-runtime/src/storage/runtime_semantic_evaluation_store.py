"""Durable scoped persistence and CAS projection for temporal evaluation."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

from typing import TypeVar
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import ContextGraph, DecisionRecord, Ontology
from contracts.runtime_producer_integrity import (
    RuntimeContextReasoningRequestV1,
    RuntimeContextReasoningCaptureV1,
    RuntimeContextExecutionAdmissionV2,
    RuntimeContextRequestIdentityV2,
    RuntimeContextRequestOutcomeV2,
    RuntimeDispatchEventV1,
    RuntimeEvaluationStateV2,
    RuntimeTemporalCaptureV2,
)
from contracts.runtime_semantic_evaluation import (
    RuntimeEvaluationDiagnosticV1,
    RuntimeEvaluationRequestIdentityV1,
    RuntimeEvaluationRequestOutcomeV1,
    RuntimeEvaluationRunV1,
    RuntimeEvaluationStateV1,
    RuntimeExecutionAdmissionV1,
    RuntimeProviderResultV1,
    RuntimeTemporalCaptureV1,
    RuntimeTemporalSupportedResultV1,
    RuntimeTemporalUnavailableResultV1,
    RuntimeTemporalWorldSnapshotV1,
    RuntimeTemporalProviderRequestV1,
    RuntimeTemporalProviderProfileV1,
    RuntimeTemporalProviderProfileV2,
    runtime_temporal_profile_model,
    RuntimeTemporalProviderDescriptorV1,
    TemporalQueryV1,
)
from contracts.runtime_temporal_source import (
    RuntimeTemporalNormalizationV1,
    RuntimeTemporalSourceArchiveV1,
    RuntimeTemporalSourceGraphV1,
)
from eos.artifact_hashing import canonical_json, content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.runtime_session_service import GovernedRuntimeSessionResolver
from storage.ontology_build_store import OntologyBuildStore
from services.runtime_errors import RuntimeContextError


T = TypeVar("T", bound=BaseModel)


def _same_pin(left: ArtifactPinV1, right: ArtifactPinV1) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def temporal_publication_hash(
    *,
    actor_id: str,
    state: RuntimeEvaluationStateV1,
    phase: str,
    inputs: tuple[ArtifactPinV1, ...],
) -> str:
    return "sha256:" + content_hash(
        {
            "organization_id": state.organization_id,
            "workspace_id": state.workspace_id,
            "actor_id": actor_id,
            "run": state.run.model_dump(mode="json"),
            "predecessor": state.artifact_pin().model_dump(mode="json"),
            "phase": phase,
            "inputs": [
                pin.model_dump(mode="json")
                for pin in sorted(
                    inputs,
                    key=lambda pin: (
                        pin.artifact_kind,
                        pin.artifact_ref,
                        pin.artifact_hash,
                    ),
                )
            ],
        }
    )


class RuntimeSemanticEvaluationStore:
    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
        session_resolver: GovernedRuntimeSessionResolver,
    ) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("build_store must be OntologyBuildStore")
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("authority_resolver must be RuntimeAuthorityResolver")
        if not isinstance(session_resolver, GovernedRuntimeSessionResolver):
            raise TypeError("session_resolver must be GovernedRuntimeSessionResolver")
        self._build_store = build_store
        self._connection = build_store._connection
        self._authority_resolver = authority_resolver
        self._session_resolver = session_resolver

    def _base_access(self, actor_context: RuntimeContextDTO) -> tuple[str, str, str]:
        self._authority_resolver.require_current_actor(actor_context=actor_context)
        workspace = actor_context.workspace_id
        session = actor_context.session_id
        if not workspace:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if not session:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        canonical = self._session_resolver.resolve(
            actor_context=actor_context, session_ref=session
        )
        if canonical != session:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        return actor_context.organization_id, workspace, session

    def _read(
        self,
        organization: str,
        workspace: str,
        pin: ArtifactPinV1,
        model: type[T],
    ) -> T:
        raw = self._build_store.get_raw(
            organization_id=organization,
            workspace_id=workspace,
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        if raw["artifact_hash"] != pin.artifact_hash:
            raise RuntimeContextError("HASH_MISMATCH")
        versions = {
            (
                RuntimeEvaluationStateV1,
                "runtime_evaluation_state.v2",
            ): RuntimeEvaluationStateV2,
            (
                RuntimeExecutionAdmissionV1,
                "runtime_execution_admission.v2",
            ): RuntimeContextExecutionAdmissionV2,
            (
                RuntimeEvaluationRequestOutcomeV1,
                "runtime_evaluation_request_outcome.v2",
            ): RuntimeContextRequestOutcomeV2,
            (
                RuntimeTemporalProviderRequestV1,
                "runtime_context_reasoning_request.v1",
            ): RuntimeContextReasoningRequestV1,
            (
                RuntimeTemporalCaptureV1,
                "runtime_context_reasoning_capture.v1",
            ): RuntimeContextReasoningCaptureV1,
            (
                RuntimeTemporalCaptureV1,
                "runtime_temporal_capture.v2",
            ): RuntimeTemporalCaptureV2,
        }
        model = versions.get((model, raw["schema_version"]), model)
        try:
            value = model.model_validate(raw["payload"])
        except (ValidationError, ValueError, TypeError) as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        if (
            raw["schema_version"] != value.schema_version
            or value.artifact_pin() != pin
            or canonical_json(value.model_dump(mode="json"))
            != canonical_json(raw["payload"])
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        return value

    def _read_profile(
        self, organization: str, workspace: str, pin: ArtifactPinV1
    ) -> RuntimeTemporalProviderProfileV1 | RuntimeTemporalProviderProfileV2:
        raw = self._build_store.get_raw(
            organization_id=organization,
            workspace_id=workspace,
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        try:
            model = runtime_temporal_profile_model(raw["schema_version"])
        except ValueError as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        return self._read(organization, workspace, pin, model)

    def _read_result(
        self, organization: str, workspace: str, pin: ArtifactPinV1
    ) -> RuntimeProviderResultV1:
        raw = self._build_store.get_raw(
            organization_id=organization,
            workspace_id=workspace,
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        state = raw["payload"].get("capability_state")
        model = (
            RuntimeTemporalSupportedResultV1
            if state == "supported"
            else RuntimeTemporalUnavailableResultV1
        )
        return self._read(organization, workspace, pin, model)

    def _access_run(
        self, *, actor_context: RuntimeContextDTO, run: RuntimeEvaluationRunV1
    ) -> RuntimeExecutionAdmissionV1:
        organization, workspace, session = self._base_access(actor_context)
        if (run.organization_id, run.workspace_id) != (organization, workspace):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        admission = self._read(
            organization, workspace, run.admission, RuntimeExecutionAdmissionV1
        )
        if (
            admission.session_ref != session
            or admission.actor_id != actor_context.actor_id
        ):
            raise RuntimeContextError("FORBIDDEN_SESSION")
        return admission

    def get_run(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeEvaluationRunV1:
        organization, workspace, _session = self._base_access(actor_context)
        if pin.artifact_kind != "runtime_evaluation_run":
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        run = self._read(organization, workspace, pin, RuntimeEvaluationRunV1)
        self._access_run(actor_context=actor_context, run=run)
        return run

    def get_provider_request(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeTemporalProviderRequestV1:
        """Read retained request inputs, never current provider readiness/configuration."""
        organization, workspace, _session = self._base_access(actor_context)
        request = self._read(
            organization, workspace, pin, RuntimeTemporalProviderRequestV1
        )
        if request.context.artifact_kind != "context_graph":
            raise RuntimeContextError("UNSUPPORTED_ARTIFACT_KIND")
        resolved = self._authority_resolver.resolve_pins(
            actor_context=actor_context, pins=(request.context,)
        )
        if (
            len(resolved) != 1
            or not isinstance(resolved[0], ContextGraph)
            or resolved[0].state != "ADMITTED"
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        context = resolved[0]
        if isinstance(request, RuntimeContextReasoningRequestV1):
            self._validate_context_request(organization, workspace, request)
            return request
        query = self._read(organization, workspace, request.query, TemporalQueryV1)
        profile = self._read_profile(organization, workspace, request.profile)
        self._read(
            organization,
            workspace,
            request.provider,
            RuntimeTemporalProviderDescriptorV1,
        )
        graph = self._read(
            organization, workspace, query.source_graph, RuntimeTemporalSourceGraphV1
        )
        expected_world = request.context
        if (
            request.world != expected_world
            or query.source_graph != context.world.source_graph
            or request.source_pins != (graph.artifact_pin(),)
            or request.requested_operation != query.operation
            or profile.provider != request.provider
            or set(request.target.evidence) != {graph.source, graph.normalization}
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return request

    def get_admission(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeExecutionAdmissionV1:
        organization, workspace, _session = self._base_access(actor_context)
        admission = self._read(
            organization, workspace, pin, RuntimeExecutionAdmissionV1
        )
        run_rows = self._connection.execute(
            sql_query("SELECT artifact_ref,artifact_hash FROM ontology_build_artifacts "
            "WHERE organization_id=? AND workspace_id=? AND schema_version='runtime_evaluation_run.v1' "
            "AND json_extract(payload_json,'$.admission.artifact_ref')=?","SELECT artifact_ref,artifact_hash FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version='runtime_evaluation_run.v1' AND (payload_json::jsonb #>> '{admission,artifact_ref}')=%s"),
            (organization, workspace, admission.admission_ref),
        ).fetchall()
        if len(run_rows) != 1:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        run = self._read(
            organization,
            workspace,
            ArtifactPinV1(
                artifact_kind="runtime_evaluation_run",
                artifact_ref=run_rows[0][0],
                artifact_hash=run_rows[0][1],
            ),
            RuntimeEvaluationRunV1,
        )
        self._access_run(actor_context=actor_context, run=run)
        return admission

    def get_capture(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeTemporalCaptureV1:
        organization, workspace, _session = self._base_access(actor_context)
        capture = self._read(organization, workspace, pin, RuntimeTemporalCaptureV1)
        run = self._read(organization, workspace, capture.run, RuntimeEvaluationRunV1)
        self._access_run(actor_context=actor_context, run=run)
        return capture

    def get_snapshot(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeTemporalWorldSnapshotV1:
        organization, workspace, _session = self._base_access(actor_context)
        snapshot = self._read(
            organization, workspace, pin, RuntimeTemporalWorldSnapshotV1
        )
        run = self._read(organization, workspace, snapshot.run, RuntimeEvaluationRunV1)
        self._access_run(actor_context=actor_context, run=run)
        return snapshot

    def get_result(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeProviderResultV1:
        organization, workspace, _session = self._base_access(actor_context)
        if pin.artifact_kind != "runtime_provider_result":
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        result = self._read_result(organization, workspace, pin)
        run = self._read(organization, workspace, result.run, RuntimeEvaluationRunV1)
        admission = self._access_run(actor_context=actor_context, run=run)
        self._validate_result_closure(organization, workspace, run, admission, result)
        return result

    def _validate_result_closure(
        self,
        organization: str,
        workspace: str,
        run: RuntimeEvaluationRunV1,
        admission: RuntimeExecutionAdmissionV1,
        result: RuntimeProviderResultV1,
    ) -> None:
        """Resolve persisted inputs only; historical reads never invoke the provider."""
        if (
            result.run != run.artifact_pin()
            or result.provider_request != admission.provider_request
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        request = self._read(
            organization,
            workspace,
            result.provider_request,
            RuntimeTemporalProviderRequestV1,
        )
        query = self._read(organization, workspace, request.query, TemporalQueryV1)
        graph = self._read(
            organization, workspace, query.source_graph, RuntimeTemporalSourceGraphV1
        )
        normalization = self._read(
            organization, workspace, graph.normalization, RuntimeTemporalNormalizationV1
        )
        archive = self._read(
            organization, workspace, graph.source, RuntimeTemporalSourceArchiveV1
        )
        if (
            normalization.source != graph.source
            or request.context != admission.context
            or request.world != admission.world
            or request.target != admission.target
            or graph.session_ref != admission.session_ref
            or archive.session_ref != admission.session_ref
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        expected = {
            graph.artifact_pin(),
            archive.artifact_pin(),
            normalization.artifact_pin(),
            query.artifact_pin(),
        }
        if not isinstance(result, RuntimeTemporalSupportedResultV1):
            if set(result.evidence) != expected:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            return
        capture = self._read(
            organization, workspace, result.provider_output, RuntimeTemporalCaptureV1
        )
        if (
            capture.run != run.artifact_pin()
            or capture.provider_request != result.provider_request
            or capture.operation != query.operation
            or result.payload.operation != query.operation
            or capture.source_graph != query.source_graph
            or capture.normalization != graph.normalization
            or result.payload.normalization != graph.normalization
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        expected.add(capture.artifact_pin())
        if set(result.evidence) != expected:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        if result.payload.world_snapshot is not None:
            snapshot = self._read(
                organization,
                workspace,
                result.payload.world_snapshot,
                RuntimeTemporalWorldSnapshotV1,
            )
            if (
                snapshot.run != run.artifact_pin()
                or snapshot.query != request.query
                or snapshot.source_graph != capture.source_graph
                or snapshot.capture != result.provider_output
                or snapshot.graph != capture.world_snapshot
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")

    def get_state(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeEvaluationStateV1:
        organization, workspace, _session = self._base_access(actor_context)
        if pin.artifact_kind != "runtime_evaluation_state":
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        state = self._read(organization, workspace, pin, RuntimeEvaluationStateV1)
        run = self._read(organization, workspace, state.run, RuntimeEvaluationRunV1)
        self._access_run(actor_context=actor_context, run=run)
        projected, _outcome = self._validate_projection(
            organization, workspace, run, through_revision=state.revision
        )
        if projected != state:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return state

    def _publication_command(
        self,
        organization: str,
        workspace: str,
        admission: RuntimeExecutionAdmissionV1,
        previous: RuntimeEvaluationStateV1,
        state: RuntimeEvaluationStateV1,
    ) -> tuple[str, str]:
        request = self._read(
            organization,
            workspace,
            admission.provider_request,
            RuntimeTemporalProviderRequestV1,
        )
        if (
            isinstance(state, RuntimeEvaluationStateV2)
            and len(state.attempt_events) == len(previous.attempt_events) + 1
        ):
            event = state.attempt_events[-1]
            phase = f"attempt:{event.kind}:{event.attempt_id}"
            inputs = (
                (event.request,)
                if event.capture is None
                else (event.request, event.capture)
            )
        elif isinstance(request, RuntimeContextReasoningRequestV1):
            if state.status == "RUNNING" and previous.status == "LOADING":
                phase, inputs = "start", (request.artifact_pin(),)
            elif state.status == "COMPLETE" and previous.capture is not None:
                phase, inputs = "complete", (previous.capture,)
            elif state.status == "FAILED":
                phase, inputs = "fail", (request.artifact_pin(),)
            else:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        elif state.status == "RUNNING" and previous.status in {"LOADING", "PARTIAL"}:
            phase, inputs = "start", (request.artifact_pin(), request.profile)
        elif (
            state.status == "RUNNING"
            and previous.capture is None
            and state.capture is not None
        ):
            query = self._read(organization, workspace, request.query, TemporalQueryV1)
            phase, inputs = "capture", (request.artifact_pin(), query.source_graph)
        elif state.status == "COMPLETE" and previous.capture is not None:
            phase, inputs = "complete", (previous.capture,)
        elif state.status in {"PARTIAL", "FAILED"}:
            phase = "unavailable" if state.status == "PARTIAL" else "fail"
            inputs = (request.artifact_pin(),)
        else:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return phase, temporal_publication_hash(
            actor_id=admission.actor_id, state=previous, phase=phase, inputs=inputs
        )

    def _validate_projection(
        self,
        organization: str,
        workspace: str,
        run: RuntimeEvaluationRunV1,
        *,
        through_revision: int | None = None,
    ) -> tuple[RuntimeEvaluationStateV1, RuntimeEvaluationRequestOutcomeV1]:
        admission = self._read(
            organization, workspace, run.admission, RuntimeExecutionAdmissionV1
        )
        rows = self._connection.execute(
            sql_query("SELECT revision,state_ref,state_hash,status,outcome_ref,outcome_hash "
            "FROM runtime_evaluation_state_index WHERE organization_id=? AND workspace_id=? AND run_ref=? "
            "AND (? IS NULL OR revision<=?) ORDER BY revision",'SELECT revision,state_ref,state_hash,status,outcome_ref,outcome_hash FROM runtime_evaluation_state_index WHERE organization_id=%s AND workspace_id=%s AND run_ref=%s AND (%s::integer IS NULL OR revision<=%s) ORDER BY revision'),
            (organization, workspace, run.run_ref, through_revision, through_revision),
        ).fetchall()
        if not rows or [row[0] for row in rows] != list(range(len(rows))):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        predecessor = None
        predecessor_outcome = None
        latest_state = None
        latest_outcome = None
        for revision, state_ref, state_hash, status, outcome_ref, outcome_hash in rows:
            state = self._read(
                organization,
                workspace,
                ArtifactPinV1(
                    artifact_kind="runtime_evaluation_state",
                    artifact_ref=state_ref,
                    artifact_hash=state_hash,
                ),
                RuntimeEvaluationStateV1,
            )
            outcome = self._read(
                organization,
                workspace,
                ArtifactPinV1(
                    artifact_kind="runtime_evaluation_request_outcome",
                    artifact_ref=outcome_ref,
                    artifact_hash=outcome_hash,
                ),
                RuntimeEvaluationRequestOutcomeV1,
            )
            expected_predecessor = (
                None if predecessor is None else predecessor.artifact_pin()
            )
            if (
                state.revision != revision
                or state.status != status
                or state.run != run.artifact_pin()
                or state.predecessor != expected_predecessor
                or outcome.run_pin != run.artifact_pin()
                or outcome.state_pin != state.artifact_pin()
                or outcome.result_pins != state.results
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if (
                state.organization_id,
                state.workspace_id,
                outcome.organization_id,
                outcome.workspace_id,
            ) != (organization, workspace, organization, workspace):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if isinstance(state, RuntimeEvaluationStateV2):
                context_mode = state.producer_mode == "CONTEXT_REASONING"
                if context_mode != isinstance(
                    admission, RuntimeContextExecutionAdmissionV2
                ) or context_mode != isinstance(
                    outcome, RuntimeContextRequestOutcomeV2
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if predecessor is None:
                if (
                    state.status != "LOADING"
                    or state.decisions
                    or (
                        isinstance(state, RuntimeEvaluationStateV2)
                        and state.attempt_events
                    )
                    or state.capture is not None
                    or state.results
                    or state.diagnostics
                    or outcome.execution_admission_pin != admission.artifact_pin()
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            else:
                phase, command_hash = self._publication_command(
                    organization, workspace, admission, predecessor, state
                )
                publication = self._connection.execute(
                    sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                    "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                    (
                        organization,
                        workspace,
                        admission.actor_id,
                        "PUBLISH_RUNTIME_EVALUATION_STATE",
                        f"{run.run_ref}:{predecessor.revision}:{phase}",
                    ),
                ).fetchone()
                if publication != (
                    command_hash,
                    outcome.outcome_ref,
                    outcome.outcome_hash,
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if isinstance(predecessor, RuntimeEvaluationStateV2) and not isinstance(
                    state, RuntimeEvaluationStateV2
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if isinstance(state, RuntimeEvaluationStateV2):
                    if not isinstance(predecessor, RuntimeEvaluationStateV2) or (
                        state.producer_mode,
                        state.max_dispatches,
                        state.dispatch_lease_seconds,
                    ) != (
                        predecessor.producer_mode,
                        predecessor.max_dispatches,
                        predecessor.dispatch_lease_seconds,
                    ):
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                    if (
                        state.attempt_events[: len(predecessor.attempt_events)]
                        != predecessor.attempt_events
                        or len(state.attempt_events)
                        > len(predecessor.attempt_events) + 1
                    ):
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                    if state.attempt_events != predecessor.attempt_events:
                        self._validate_attempt_transition(admission, predecessor, state)
                if (
                    predecessor.decisions
                    and state.decisions != predecessor.decisions
                    and predecessor.status != "PARTIAL"
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                allowed = {
                    "LOADING": {"RUNNING", "PARTIAL", "FAILED"},
                    "RUNNING": {"RUNNING", "PARTIAL", "FAILED", "COMPLETE"},
                    "PARTIAL": {"RUNNING"},
                    "FAILED": {"FAILED"}
                    if isinstance(state, RuntimeEvaluationStateV2)
                    else set(),
                    "COMPLETE": {"COMPLETE"}
                    if isinstance(state, RuntimeEvaluationStateV2)
                    else set(),
                }
                if state.status not in allowed[predecessor.status]:
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if (
                    predecessor.capture is not None
                    and state.capture != predecessor.capture
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if (
                    state.diagnostics[: len(predecessor.diagnostics)]
                    != predecessor.diagnostics
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                assert predecessor_outcome is not None
                immutable_outcome = (
                    "actor_ref",
                    "operation",
                    "request_id",
                    "normalized_request_hash",
                    "run_pin",
                    "execution_admission_pin",
                    "epoch_pin",
                    "authority_release_receipt_pin",
                )
                if any(
                    getattr(outcome, field) != getattr(predecessor_outcome, field)
                    for field in immutable_outcome
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if predecessor.results and state.results != predecessor.results:
                    old_result = self._read_result(
                        organization, workspace, predecessor.results[0]
                    )
                    new_result = (
                        None
                        if not state.results
                        else self._read_result(
                            organization, workspace, state.results[0]
                        )
                    )
                    if not (
                        isinstance(old_result, RuntimeTemporalUnavailableResultV1)
                        and isinstance(new_result, RuntimeTemporalSupportedResultV1)
                        and state.status == "COMPLETE"
                    ):
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if isinstance(state, RuntimeEvaluationStateV2):
                for event in state.attempt_events:
                    if event.request != admission.provider_request:
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                    if event.capture is not None:
                        retained = self._read(
                            organization,
                            workspace,
                            event.capture,
                            RuntimeTemporalCaptureV1,
                        )
                        if (
                            retained.run != run.artifact_pin()
                            or retained.provider_request != admission.provider_request
                            or retained.attempt_id != event.attempt_id
                        ):
                            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if state.capture is not None:
                capture = self._read(
                    organization, workspace, state.capture, RuntimeTemporalCaptureV1
                )
                if (
                    capture.run != run.artifact_pin()
                    or capture.provider_request != admission.provider_request
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            for diagnostic_pin in state.diagnostics:
                diagnostic = self._read(
                    organization,
                    workspace,
                    diagnostic_pin,
                    RuntimeEvaluationDiagnosticV1,
                )
                if diagnostic.run != run.artifact_pin():
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if state.results:
                result = self._read_result(organization, workspace, state.results[0])
                self._validate_result_closure(
                    organization, workspace, run, admission, result
                )
                if (
                    isinstance(result, RuntimeTemporalSupportedResultV1)
                    and state.capture != result.provider_output
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if (
                    result.run != run.artifact_pin()
                    or result.provider_request != admission.provider_request
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if state.status == "PARTIAL" and not isinstance(
                    result, RuntimeTemporalUnavailableResultV1
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if state.status == "COMPLETE" and not isinstance(
                    result, RuntimeTemporalSupportedResultV1
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            elif (
                isinstance(state, RuntimeEvaluationStateV2)
                and state.producer_mode == "CONTEXT_REASONING"
            ):
                request = self._read(
                    organization,
                    workspace,
                    admission.provider_request,
                    RuntimeTemporalProviderRequestV1,
                )
                self._validate_context_request(organization, workspace, request)
                if state.status == "COMPLETE":
                    if state.capture is None or len(state.decisions) != 1:
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                    decision = self._build_store.get(
                        organization_id=organization,
                        workspace_id=workspace,
                        artifact_ref=state.decisions[0].artifact_ref,
                        contract_type=DecisionRecord,
                    )
                    if (
                        decision is None
                        or decision.pin() != state.decisions[0]
                        or decision.purpose != "INFERENCE"
                        or decision.context != request.context
                        or decision.subject != request.context
                        or decision.ontology != request.working_context.ontology
                        or decision.actor_id
                        != request.working_context.responsibility.delegated_actor_id
                        or decision.outcome_observation != state.capture
                    ):
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                    capture = self._read(
                        organization, workspace, state.capture, RuntimeTemporalCaptureV1
                    )
                    raw = capture.native_output
                    expected_code = "UNKNOWN" if raw is None else raw["conclusion"]
                    expected_status = "UNKNOWN" if raw is None else "KNOWN"
                    if (
                        decision.result_code != expected_code
                        or decision.epistemic_status != expected_status
                        or decision.reasoning.working_context != request.working_context
                        or decision.reasoner_identity != request.executable.artifact_ref
                        or any(
                            item.executable_engine != request.executable
                            or item.outputs != (expected_code,)
                            for item in decision.reasoning.invocations
                        )
                    ):
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                    inputs = {
                        pin
                        for item in decision.reasoning.invocations
                        for pin in item.inputs
                    }
                    if not {request.artifact_pin(), state.capture} <= inputs:
                        raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            elif state.status in {"PARTIAL", "COMPLETE"}:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            predecessor = state
            predecessor_outcome = outcome
            latest_state, latest_outcome = state, outcome
        if through_revision is not None:
            if (
                latest_state is None
                or latest_outcome is None
                or latest_state.revision != through_revision
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            return latest_state, latest_outcome
        head = self._connection.execute(
            sql_query("SELECT revision,state_ref,state_hash,outcome_ref,outcome_hash FROM runtime_evaluation_heads "
            "WHERE organization_id=? AND workspace_id=? AND run_ref=?",'SELECT revision,state_ref,state_hash,outcome_ref,outcome_hash FROM runtime_evaluation_heads WHERE organization_id=%s AND workspace_id=%s AND run_ref=%s'),
            (organization, workspace, run.run_ref),
        ).fetchone()
        if (
            latest_state is None
            or latest_outcome is None
            or head
            != (
                latest_state.revision,
                latest_state.state_ref,
                latest_state.state_hash,
                latest_outcome.outcome_ref,
                latest_outcome.outcome_hash,
            )
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return latest_state, latest_outcome

    def get_current_state(
        self, *, actor_context: RuntimeContextDTO, run: ArtifactPinV1
    ) -> RuntimeEvaluationStateV1:
        value = self.get_run(actor_context=actor_context, pin=run)
        state, _outcome = self._validate_projection(
            value.organization_id, value.workspace_id, value
        )
        return state

    def get_request_outcome(
        self,
        *,
        actor_context: RuntimeContextDTO,
        identity: RuntimeEvaluationRequestIdentityV1,
    ) -> RuntimeEvaluationRequestOutcomeV1 | None:
        organization, workspace, _session = self._base_access(actor_context)
        if type(identity) not in {
            RuntimeEvaluationRequestIdentityV1,
            RuntimeContextRequestIdentityV2,
        }:
            raise RuntimeContextError("INVALID_STATE")
        identity = type(identity).model_validate(identity.model_dump(mode="json"))
        if (
            identity.organization_id,
            identity.workspace_id,
            identity.actor_ref,
        ) != (organization, workspace, actor_context.actor_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        key = (
            organization,
            workspace,
            identity.actor_ref,
            identity.operation,
            identity.request_id,
        )
        row = self._connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
            "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            key,
        ).fetchone()
        expected_hash = identity.normalized_request_hash()
        if row is None:
            orphan = self._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? "
                "AND schema_version IN ('runtime_evaluation_request_outcome.v1','runtime_evaluation_request_outcome.v2') "
                "AND json_extract(payload_json,'$.actor_ref')=? "
                "AND json_extract(payload_json,'$.operation')=? "
                "AND json_extract(payload_json,'$.request_id')=? LIMIT 1","SELECT 1 FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version IN ('runtime_evaluation_request_outcome.v1','runtime_evaluation_request_outcome.v2') AND (payload_json::jsonb #>> '{actor_ref}')=%s AND (payload_json::jsonb #>> '{operation}')=%s AND (payload_json::jsonb #>> '{request_id}')=%s LIMIT 1"),
                key,
            ).fetchone()
            if orphan is not None:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            return None
        if row[0] != expected_hash:
            raise RuntimeContextError("REQUEST_CONFLICT")
        initial = self._read(
            organization,
            workspace,
            ArtifactPinV1(
                artifact_kind="runtime_evaluation_request_outcome",
                artifact_ref=row[1],
                artifact_hash=row[2],
            ),
            RuntimeEvaluationRequestOutcomeV1,
        )
        if (
            initial.normalized_request_hash != expected_hash
            or initial.actor_ref != identity.actor_ref
            or initial.request_id != identity.request_id
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        run = self.get_run(actor_context=actor_context, pin=initial.run_pin)
        _state, current = self._validate_projection(organization, workspace, run)
        return current

    def admit(
        self,
        *,
        actor_context: RuntimeContextDTO,
        identity: RuntimeEvaluationRequestIdentityV1,
        admission: RuntimeExecutionAdmissionV1,
        run: RuntimeEvaluationRunV1,
        state: RuntimeEvaluationStateV1,
        outcome: RuntimeEvaluationRequestOutcomeV1,
        verify_before_commit,
    ) -> RuntimeEvaluationRequestOutcomeV1:
        organization, workspace, session = self._base_access(actor_context)
        expected_hash = identity.normalized_request_hash()
        if (
            admission.session_ref != session
            or admission.actor_id != actor_context.actor_id
            or run.admission != admission.artifact_pin()
            or state.run != run.artifact_pin()
            or state.revision != 0
            or state.status != "LOADING"
            or outcome.state_pin != state.artifact_pin()
            or outcome.run_pin != run.artifact_pin()
            or outcome.execution_admission_pin != admission.artifact_pin()
            or outcome.normalized_request_hash != expected_hash
        ):
            raise RuntimeContextError("INVALID_STATE")
        if isinstance(state, RuntimeEvaluationStateV2):
            req = self.get_provider_request(
                actor_context=actor_context, pin=identity.provider_request
            )
            context_mode = state.producer_mode == "CONTEXT_REASONING"
            if (
                context_mode != isinstance(req, RuntimeContextReasoningRequestV1)
                or context_mode
                != isinstance(admission, RuntimeContextExecutionAdmissionV2)
                or context_mode != isinstance(identity, RuntimeContextRequestIdentityV2)
                or context_mode != isinstance(outcome, RuntimeContextRequestOutcomeV2)
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if (
                (
                    admission.context,
                    admission.world,
                    admission.provider_request,
                    admission.target,
                    admission.business_effective_at,
                )
                != (
                    identity.context,
                    req.world,
                    identity.provider_request,
                    identity.target,
                    identity.business_effective_at,
                )
                or req.context != admission.context
                or req.target != admission.target
                or state.attempt_events
                or state.decisions
                or state.predecessor is not None
                or (outcome.actor_ref, outcome.operation, outcome.request_id)
                != (identity.actor_ref, identity.operation, identity.request_id)
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if context_mode and (
                state.max_dispatches != req.max_dispatches
                or req.working_context.responsibility.responsibility_bearer_id
                != actor_context.actor_id
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        winner: RuntimeEvaluationRequestOutcomeV1 | None = None
        key = (
            organization,
            workspace,
            actor_context.actor_id,
            identity.operation,
            identity.request_id,
        )

        def validate(connection) -> bool:
            nonlocal winner
            row = connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                key,
            ).fetchone()
            if row is not None:
                if row[0] != expected_hash:
                    raise RuntimeContextError("REQUEST_CONFLICT")
                winner = self._read(
                    organization,
                    workspace,
                    ArtifactPinV1(
                        artifact_kind="runtime_evaluation_request_outcome",
                        artifact_ref=row[1],
                        artifact_hash=row[2],
                    ),
                    RuntimeEvaluationRequestOutcomeV1,
                )
                return False
            orphan = connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? "
                "AND schema_version IN ('runtime_evaluation_request_outcome.v1','runtime_evaluation_request_outcome.v2') "
                "AND json_extract(payload_json,'$.actor_ref')=? "
                "AND json_extract(payload_json,'$.operation')=? "
                "AND json_extract(payload_json,'$.request_id')=? LIMIT 1","SELECT 1 FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version IN ('runtime_evaluation_request_outcome.v1','runtime_evaluation_request_outcome.v2') AND (payload_json::jsonb #>> '{actor_ref}')=%s AND (payload_json::jsonb #>> '{operation}')=%s AND (payload_json::jsonb #>> '{request_id}')=%s LIMIT 1"),
                key,
            ).fetchone()
            if orphan is not None:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if (
                connection.execute(
                    sql_query("SELECT 1 FROM runtime_evaluation_heads WHERE organization_id=? AND workspace_id=? AND run_ref=?",'SELECT 1 FROM runtime_evaluation_heads WHERE organization_id=%s AND workspace_id=%s AND run_ref=%s'),
                    (organization, workspace, run.run_ref),
                ).fetchone()
                is not None
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            verify_before_commit(connection)
            return True

        # Keep the SQL literal separate so a malformed callback cannot partially publish.
        def mutate_all(connection) -> None:
            connection.execute(
                sql_query("INSERT INTO runtime_evaluation_state_index "
                "(organization_id,workspace_id,run_ref,revision,state_ref,state_hash,status,outcome_ref,outcome_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",'INSERT INTO runtime_evaluation_state_index (organization_id,workspace_id,run_ref,revision,state_ref,state_hash,status,outcome_ref,outcome_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    organization,
                    workspace,
                    run.run_ref,
                    0,
                    state.state_ref,
                    state.state_hash,
                    state.status,
                    outcome.outcome_ref,
                    outcome.outcome_hash,
                ),
            )
            connection.execute(
                sql_query("INSERT INTO runtime_evaluation_heads "
                "(organization_id,workspace_id,run_ref,revision,state_ref,state_hash,outcome_ref,outcome_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO runtime_evaluation_heads (organization_id,workspace_id,run_ref,revision,state_ref,state_hash,outcome_ref,outcome_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    organization,
                    workspace,
                    run.run_ref,
                    0,
                    state.state_ref,
                    state.state_hash,
                    outcome.outcome_ref,
                    outcome.outcome_hash,
                ),
            )
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (*key, expected_hash, outcome.outcome_ref, outcome.outcome_hash),
            )

        self._build_store.atomic_write(
            artifacts=(admission, run, state, outcome),
            validate_before_insert=validate,
            mutate_indexes=mutate_all,
            verify_before_commit=lambda: (
                self._base_access(actor_context),
                self._validate_projection(organization, workspace, run)
                if isinstance(state, RuntimeEvaluationStateV2)
                else None,
            ),
        )
        return winner or outcome

    def publish(
        self,
        *,
        actor_context: RuntimeContextDTO,
        previous: RuntimeEvaluationStateV1,
        previous_outcome: RuntimeEvaluationRequestOutcomeV1,
        state: RuntimeEvaluationStateV1,
        outcome: RuntimeEvaluationRequestOutcomeV1,
        artifacts: tuple[BaseModel, ...],
        phase: str,
        request_hash: str,
    ) -> RuntimeEvaluationRequestOutcomeV1:
        organization, workspace, _session = self._base_access(actor_context)
        run = self.get_run(actor_context=actor_context, pin=previous.run)
        if (
            state.revision != previous.revision + 1
            or state.predecessor != previous.artifact_pin()
            or state.run != run.artifact_pin()
            or outcome.state_pin != state.artifact_pin()
            or outcome.run_pin != run.artifact_pin()
            or outcome.execution_admission_pin
            != previous_outcome.execution_admission_pin
            or outcome.normalized_request_hash
            != previous_outcome.normalized_request_hash
            or outcome.result_pins != state.results
        ):
            raise RuntimeContextError("INVALID_STATE")
        publication_request_id = f"{run.run_ref}:{previous.revision}:{phase}"
        key = (
            organization,
            workspace,
            actor_context.actor_id,
            "PUBLISH_RUNTIME_EVALUATION_STATE",
            publication_request_id,
        )
        winner: RuntimeEvaluationRequestOutcomeV1 | None = None

        def validate(connection) -> bool:
            nonlocal winner
            self._validate_projection(organization, workspace, run)
            admission = self._access_run(actor_context=actor_context, run=run)
            expected_phase, expected_hash = self._publication_command(
                organization, workspace, admission, previous, state
            )
            if phase != expected_phase or request_hash != expected_hash:
                raise RuntimeContextError("REQUEST_CONFLICT")
            indexed = connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                key,
            ).fetchone()
            if indexed is not None:
                if indexed[0] != request_hash:
                    raise RuntimeContextError("REQUEST_CONFLICT")
                winner = self._read(
                    organization,
                    workspace,
                    ArtifactPinV1(
                        artifact_kind="runtime_evaluation_request_outcome",
                        artifact_ref=indexed[1],
                        artifact_hash=indexed[2],
                    ),
                    RuntimeEvaluationRequestOutcomeV1,
                )
                return False
            head = connection.execute(
                sql_query("SELECT revision,state_ref,state_hash,outcome_ref,outcome_hash FROM runtime_evaluation_heads "
                "WHERE organization_id=? AND workspace_id=? AND run_ref=?",'SELECT revision,state_ref,state_hash,outcome_ref,outcome_hash FROM runtime_evaluation_heads WHERE organization_id=%s AND workspace_id=%s AND run_ref=%s'),
                (organization, workspace, run.run_ref),
            ).fetchone()
            expected = (
                previous.revision,
                previous.state_ref,
                previous.state_hash,
                previous_outcome.outcome_ref,
                previous_outcome.outcome_hash,
            )
            if head != expected:
                return False
            return True

        def mutate(connection) -> None:
            connection.execute(
                sql_query("INSERT INTO runtime_evaluation_state_index "
                "(organization_id,workspace_id,run_ref,revision,state_ref,state_hash,status,outcome_ref,outcome_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",'INSERT INTO runtime_evaluation_state_index (organization_id,workspace_id,run_ref,revision,state_ref,state_hash,status,outcome_ref,outcome_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    organization,
                    workspace,
                    run.run_ref,
                    state.revision,
                    state.state_ref,
                    state.state_hash,
                    state.status,
                    outcome.outcome_ref,
                    outcome.outcome_hash,
                ),
            )
            cursor = connection.execute(
                sql_query("UPDATE runtime_evaluation_heads SET revision=?,state_ref=?,state_hash=?,outcome_ref=?,outcome_hash=? "
                "WHERE organization_id=? AND workspace_id=? AND run_ref=? "
                "AND revision=? AND state_ref=? AND state_hash=? AND outcome_ref=? AND outcome_hash=?",'UPDATE runtime_evaluation_heads SET revision=%s,state_ref=%s,state_hash=%s,outcome_ref=%s,outcome_hash=%s WHERE organization_id=%s AND workspace_id=%s AND run_ref=%s AND revision=%s AND state_ref=%s AND state_hash=%s AND outcome_ref=%s AND outcome_hash=%s'),
                (
                    state.revision,
                    state.state_ref,
                    state.state_hash,
                    outcome.outcome_ref,
                    outcome.outcome_hash,
                    organization,
                    workspace,
                    run.run_ref,
                    previous.revision,
                    previous.state_ref,
                    previous.state_hash,
                    previous_outcome.outcome_ref,
                    previous_outcome.outcome_hash,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeContextError("STALE_PARENT")
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (*key, request_hash, outcome.outcome_ref, outcome.outcome_hash),
            )

        self._build_store.atomic_write(
            artifacts=(*artifacts, state, outcome),
            validate_before_insert=validate,
            mutate_indexes=mutate,
            verify_before_commit=lambda: (
                self._base_access(actor_context),
                self._validate_projection(organization, workspace, run)
                if isinstance(state, RuntimeEvaluationStateV2)
                else None,
            ),
        )
        if winner is not None:
            return winner
        current_run = self.get_run(actor_context=actor_context, pin=run.artifact_pin())
        _state, current_outcome = self._validate_projection(
            organization, workspace, current_run
        )
        return current_outcome

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def _validate_context_request(self, organization, workspace, request):
        if not isinstance(request, RuntimeContextReasoningRequestV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        graphs = []
        for pin in (request.context, request.case):
            graph = self._build_store.get(
                organization_id=organization,
                workspace_id=workspace,
                artifact_ref=pin.artifact_ref,
                contract_type=ContextGraph,
            )
            if graph is None or graph.pin() != pin or graph.state != "ADMITTED":
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            from contracts.semantic_kernel import validate_context_admission_transition

            if graph.predecessor is None or graph.admission_decision is None:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            proposed = self._build_store.get(
                organization_id=organization,
                workspace_id=workspace,
                artifact_ref=graph.predecessor.artifact_ref,
                contract_type=ContextGraph,
            )
            decision = self._build_store.get(
                organization_id=organization,
                workspace_id=workspace,
                artifact_ref=graph.admission_decision.artifact_ref,
                contract_type=DecisionRecord,
            )
            if (
                proposed is None
                or decision is None
                or proposed.pin() != graph.predecessor
                or decision.pin() != graph.admission_decision
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            try:
                validate_context_admission_transition(proposed, decision, graph)
            except ValueError as exc:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE") from exc
            ontology = self._build_store.get(
                organization_id=organization,
                workspace_id=workspace,
                artifact_ref=graph.ontology.artifact_ref,
                contract_type=Ontology,
            )
            if ontology is None or ontology.pin() != graph.ontology:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            graphs.append(graph)
        graph, case = graphs
        if (request.world_basis, request.intent, request.source_inputs) != (
            graph.world,
            graph.intent,
            graph.formation.source_inputs,
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        from services.runtime_semantica_reasoning_service import (
            RuntimeSemanticaReasoningService,
        )

        RuntimeSemanticaReasoningService(
            build_store=self._build_store
        )._validate_working_context(
            working_context=request.working_context, context=graph
        )
        wc = request.working_context
        assertions = sorted(graph.assertions, key=lambda item: item.assertion_id)
        facts = tuple(
            f"{a.predicate}({a.subject_ref}, {a.object_ref})" for a in assertions
        )
        selected_evidence = tuple(
            sorted(
                {p for a in assertions for p in a.evidence},
                key=lambda p: (p.artifact_kind, p.artifact_ref, p.artifact_hash),
            )
        )
        if (
            wc.ontology != graph.ontology
            or wc.selected_assertion_ids != tuple(a.assertion_id for a in assertions)
            or request.facts != facts
            or wc.evidence != selected_evidence
            or request.target.evidence != case.evidence
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        for pin in selected_evidence:
            from contracts.semantic_kernel import Evidence

            item = self._build_store.get(
                organization_id=organization,
                workspace_id=workspace,
                artifact_ref=pin.artifact_ref,
                contract_type=Evidence,
            )
            if item is None or item.pin() != pin:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")

    @staticmethod
    def _validate_attempt_transition(admission, previous, state):
        event = state.attempt_events[-1]
        if event.request != admission.provider_request or event.run != previous.run:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        registered = {
            e.attempt_id: e
            for e in previous.attempt_events
            if e.kind == "DISPATCH_REGISTERED"
        }
        ended = {
            e.attempt_id: e.kind
            for e in previous.attempt_events
            if e.kind != "DISPATCH_REGISTERED"
        }
        if event.kind == "DISPATCH_REGISTERED":
            if (
                previous.status != "RUNNING"
                or previous.capture is not None
                or any(k not in ended for k in registered)
                or event.fence != state.revision
                or event.lease_until
                != event.recorded_at
                + timedelta(seconds=previous.dispatch_lease_seconds)
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        elif event.attempt_id not in registered:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        elif event.kind == "OUTCOME_UNCERTAIN":
            if event.recorded_at < registered[event.attempt_id].lease_until:
                raise RuntimeContextError("ATTEMPT_IN_FLIGHT")
        elif event.kind == "RETURN_CAPTURED":
            latest = next(reversed(registered))
            expected = (
                event.capture
                if event.attempt_id == latest and previous.capture is None
                else previous.capture
            )
            if state.capture != expected:
                raise RuntimeContextError("STALE_ATTEMPT")
        if event.kind != "RETURN_CAPTURED" and state.capture != previous.capture:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        if (
            state.status != previous.status
            or state.results != previous.results
            or state.decisions != previous.decisions
            or state.diagnostics != previous.diagnostics
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")

    def _append_attempt_event(self, *, actor_context, previous, event, capture=None):
        run = self.get_run(actor_context=actor_context, pin=previous.run)
        for _ in range(8):
            current, outcome = self._validate_projection(
                previous.organization_id, previous.workspace_id, run
            )
            if current != previous:
                if event.kind == "DISPATCH_REGISTERED":
                    return current, False
                previous = current
            if any(
                e.kind == event.kind and e.attempt_id == event.attempt_id
                for e in previous.attempt_events
            ):
                return previous, False
            event = RuntimeDispatchEventV1.model_validate(
                {
                    **event.model_dump(mode="python"),
                    "sequence": len(previous.attempt_events),
                }
            )
            registrations = [
                e for e in previous.attempt_events if e.kind == "DISPATCH_REGISTERED"
            ]
            selected = previous.capture
            if (
                capture is not None
                and registrations[-1].attempt_id == event.attempt_id
                and selected is None
            ):
                selected = capture.artifact_pin()
            values = {
                k: getattr(previous, k)
                for k in type(previous).model_fields
                if k not in {"state_ref", "state_hash"}
            }
            values.update(
                revision=previous.revision + 1,
                predecessor=previous.artifact_pin(),
                capture=selected,
                attempt_events=(*previous.attempt_events, event),
                recorded_at=event.recorded_at,
            )
            state = RuntimeEvaluationStateV2.create(**values)
            vals = {
                k: getattr(outcome, k)
                for k in type(outcome).model_fields
                if k not in {"outcome_ref", "outcome_hash"}
            }
            vals.update(state_pin=state.artifact_pin(), recorded_at=event.recorded_at)
            next_outcome = type(outcome).create(**vals)
            phase = f"attempt:{event.kind}:{event.attempt_id}"
            inputs = (
                (event.request,)
                if event.capture is None
                else (event.request, event.capture)
            )
            published = self.publish(
                actor_context=actor_context,
                previous=previous,
                previous_outcome=outcome,
                state=state,
                outcome=next_outcome,
                artifacts=(() if capture is None else (capture,)),
                phase=phase,
                request_hash=temporal_publication_hash(
                    actor_id=actor_context.actor_id,
                    state=previous,
                    phase=phase,
                    inputs=inputs,
                ),
            )
            stable = self.get_state(
                actor_context=actor_context, pin=published.state_pin
            )
            if event in stable.attempt_events:
                return stable, True
            if event.kind == "DISPATCH_REGISTERED":
                return stable, False
            previous = stable
        raise RuntimeContextError("STALE_PARENT")

    def register_attempt(self, *, actor_context, run):
        state = self.get_current_state(actor_context=actor_context, run=run)
        if not isinstance(state, RuntimeEvaluationStateV2):
            raise RuntimeContextError("LEGACY_ATTEMPT_BASIS_UNAVAILABLE")
        if state.status != "RUNNING" or state.capture is not None:
            return state, None
        registrations = [
            e for e in state.attempt_events if e.kind == "DISPATCH_REGISTERED"
        ]
        ended = {
            e.attempt_id
            for e in state.attempt_events
            if e.kind != "DISPATCH_REGISTERED"
        }
        pending = [e for e in registrations if e.attempt_id not in ended]
        now = self._now()
        if pending:
            previous = pending[-1]
            if now < previous.lease_until:
                return state, None
            event = RuntimeDispatchEventV1(
                sequence=len(state.attempt_events),
                attempt_id=previous.attempt_id,
                ordinal=previous.ordinal,
                run=run,
                request=previous.request,
                fence=previous.fence,
                role="ACTOR",
                kind="OUTCOME_UNCERTAIN",
                lease_until=None,
                capture=None,
                diagnostic_code="DISPATCH_OUTCOME_UNCERTAIN",
                recorded_at=now,
            )
            state, _ = self._append_attempt_event(
                actor_context=actor_context, previous=state, event=event
            )
            if state.capture is not None or state.status != "RUNNING":
                return state, None
            registrations = [
                e for e in state.attempt_events if e.kind == "DISPATCH_REGISTERED"
            ]
            ended = {
                e.attempt_id
                for e in state.attempt_events
                if e.kind != "DISPATCH_REGISTERED"
            }
            if any(e.attempt_id not in ended for e in registrations):
                return state, None
        if len(registrations) >= state.max_dispatches:
            raise RuntimeContextError("DISPATCH_BUDGET_EXCEEDED")
        request = self._access_run(
            actor_context=actor_context,
            run=self.get_run(actor_context=actor_context, pin=run),
        ).provider_request
        ordinal = len(registrations) + 1
        event = RuntimeDispatchEventV1(
            sequence=len(state.attempt_events),
            attempt_id=f"attempt:{uuid4().hex}",
            ordinal=ordinal,
            run=run,
            request=request,
            fence=state.revision + 1,
            role="ACTOR",
            kind="DISPATCH_REGISTERED",
            lease_until=now + timedelta(seconds=state.dispatch_lease_seconds),
            capture=None,
            diagnostic_code=None,
            recorded_at=now,
        )
        stable, won = self._append_attempt_event(
            actor_context=actor_context, previous=state, event=event
        )
        return stable, event if won else None

    def capture_attempt(self, *, actor_context, run, attempt, capture):
        state = self.get_current_state(actor_context=actor_context, run=run)
        if (
            capture.run != run
            or capture.provider_request != attempt.request
            or capture.attempt_id != attempt.attempt_id
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        event = RuntimeDispatchEventV1(
            sequence=len(state.attempt_events),
            attempt_id=attempt.attempt_id,
            ordinal=attempt.ordinal,
            run=run,
            request=attempt.request,
            fence=attempt.fence,
            role="ACTOR",
            kind="RETURN_CAPTURED",
            lease_until=None,
            capture=capture.artifact_pin(),
            diagnostic_code=None,
            recorded_at=self._now(),
        )
        stable, _ = self._append_attempt_event(
            actor_context=actor_context, previous=state, event=event, capture=capture
        )
        return stable

    def record_attempt_exception(self, *, actor_context, run, attempt, code):
        state = self.get_current_state(actor_context=actor_context, run=run)
        event = RuntimeDispatchEventV1(
            sequence=len(state.attempt_events),
            attempt_id=attempt.attempt_id,
            ordinal=attempt.ordinal,
            run=run,
            request=attempt.request,
            fence=attempt.fence,
            role="ACTOR",
            kind="CALL_EXCEPTION",
            lease_until=None,
            capture=None,
            diagnostic_code=code,
            recorded_at=self._now(),
        )
        stable, _ = self._append_attempt_event(
            actor_context=actor_context, previous=state, event=event
        )
        return stable

    def attempt_report(self, *, actor_context, run):
        state = self.get_current_state(actor_context=actor_context, run=run)
        if not isinstance(state, RuntimeEvaluationStateV2):
            return {
                "trial_validity": "INCOMPLETE_EVIDENCE",
                "availability": "LEGACY_ATTEMPT_BASIS_UNAVAILABLE",
                "physical_execution_count": None,
            }
        registrations = {
            e.attempt_id
            for e in state.attempt_events
            if e.kind == "DISPATCH_REGISTERED"
        }
        captured = {
            e.attempt_id for e in state.attempt_events if e.kind == "RETURN_CAPTURED"
        }
        exceptions = {
            e.attempt_id for e in state.attempt_events if e.kind == "CALL_EXCEPTION"
        }
        unresolved = registrations - captured - exceptions
        return {
            "availability": "RETAINED",
            "registered_dispatches": len(registrations),
            "captured_returns": len(captured),
            "observed_exceptions": len(exceptions),
            "unresolved_dispatches": len(unresolved),
            "physical_execution_count": None,
            "cost": None,
            "trial_validity": "INCOMPLETE_EVIDENCE"
            if unresolved or state.status != "COMPLETE"
            else "VALID",
        }


__all__ = ["RuntimeSemanticEvaluationStore"]
