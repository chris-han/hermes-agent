"""Trusted composition and durable execution of the closed temporal provider."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import tomllib
from typing import Iterable

from pydantic import BaseModel, ValidationError

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    ContextGraph,
    DecisionRecord,
    Evidence,
    Ontology,
    ReasonerInvocation,
    ReasoningTrace,
    validate_context_admission_transition,
)
from contracts.runtime_context_reconstruction import CapturedJsonV1
from contracts.runtime_semantic_evaluation import (
    GovernedTargetPinV1,
    RuntimeEvaluationDiagnosticV1,
    RuntimeEvaluationRequestIdentityV1,
    RuntimeEvaluationRequestOutcomeV1,
    RuntimeEvaluationRunV1,
    RuntimeEvaluationStateV1,
    RuntimeExecutionAdmissionV1,
    RuntimeTemporalCaptureV1,
    RuntimeTemporalDependencyV2,
    RuntimeTemporalMetricsV1,
    RuntimeTemporalOccurrenceV1,
    RuntimeTemporalPathV1,
    RuntimeTemporalPatternV1,
    RuntimeTemporalPayloadV1,
    RuntimeTemporalProviderConfigV1,
    RuntimeTemporalProviderDescriptorV1,
    RuntimeTemporalProviderProfileV2,
    RuntimeTemporalProviderRequestV1,
    RuntimeTemporalRequestSettingsV1,
    RuntimeTemporalSupportedResultV1,
    RuntimeTemporalUnavailableResultV1,
    RuntimeTemporalWorldSnapshotV1,
    TemporalQueryV1,
)
from contracts.runtime_producer_integrity import RuntimeEvaluationStateV2, RuntimeTemporalCaptureV2
from contracts.runtime_temporal_source import RuntimeTemporalSourceGraphV1
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.semantica_adapter import (
    SEMANTICA_COMMIT,
    _ensure_vendored_semantica_importable,
)
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.runtime_temporal_source_service import RuntimeTemporalSourceService
from services.runtime_session_service import GovernedRuntimeSessionResolver
from storage.runtime_semantic_evaluation_store import (
    RuntimeSemanticEvaluationStore,
    temporal_publication_hash,
)
from services.runtime_errors import RuntimeContextError


_SEMANTICA_VERSION = "0.7.0"
_CONFIGURATION = RuntimeTemporalProviderConfigV1(
    temporal_granularity="second",
    pattern_gap_tolerance_seconds=1_576_800_000,
)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _capture_json(value: object, active: set[int] | None = None) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("captured JSON must be finite")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("captured datetime must be UTC")
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        seen = active if active is not None else set()
        identity = id(value)
        if identity in seen:
            raise ValueError("captured JSON must not be cyclic")
        seen.add(identity)
        try:
            if isinstance(value, dict):
                if not all(type(key) is str for key in value):
                    raise ValueError("captured JSON keys must be strings")
                return {key: _capture_json(item, seen) for key, item in value.items()}
            return [_capture_json(item, seen) for item in value]
        finally:
            seen.remove(identity)
    raise TypeError("captured output contains an unsupported value")


def _captured(value: object) -> CapturedJsonV1:
    encoded = canonical_json(_capture_json(value))
    return CapturedJsonV1(
        media_type="application/json",
        canonical_json=encoded,
        bytes_hash=_sha256_bytes(encoded.encode("utf-8")),
    )


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return pin.artifact_kind, pin.artifact_ref, pin.artifact_hash


def _canonical_pins(values: Iterable[ArtifactPinV1]) -> tuple[ArtifactPinV1, ...]:
    unique = {_pin_key(pin): pin for pin in values}
    return tuple(unique[key] for key in sorted(unique))


def _utc_seconds(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
        or value.microsecond
    ):
        raise RuntimeContextError("INVALID_STATE")
    return value


def _execute_temporal_kernel(**values: object):
    """Load the optional native kernel only after trusted dependency verification."""

    from services.runtime_temporal_provider import _execute_temporal_kernel as execute

    return execute(**values)


class RuntimeDocumentEvaluationService:
    def __init__(
        self,
        *,
        store: RuntimeSemanticEvaluationStore,
        source_service: RuntimeTemporalSourceService,
        authority_resolver: RuntimeAuthorityResolver,
        session_resolver: GovernedRuntimeSessionResolver,
    ) -> None:
        if not isinstance(store, RuntimeSemanticEvaluationStore):
            raise TypeError("store must be RuntimeSemanticEvaluationStore")
        if not isinstance(source_service, RuntimeTemporalSourceService):
            raise TypeError("source_service must be RuntimeTemporalSourceService")
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("authority_resolver must be RuntimeAuthorityResolver")
        if not isinstance(session_resolver, GovernedRuntimeSessionResolver):
            raise TypeError("session_resolver must be GovernedRuntimeSessionResolver")
        build_store = store._build_store
        if (
            source_service._build_store is not build_store
            or store._authority_resolver is not authority_resolver
            or source_service._authority_resolver is not authority_resolver
            or session_resolver._authority_resolver is not authority_resolver
        ):
            raise RuntimeContextError("INVALID_STATE")
        self._store = store
        self._source_service = source_service
        self._authority_resolver = authority_resolver
        self._session_resolver = session_resolver
        self._build_store = build_store
        self._dependency_ready = self._verify_dependency()
        names = (
            "runtime_temporal_provider.py",
            "runtime_temporal_source_service.py",
            "runtime_document_evaluation_service.py",
        )
        hashes = [
            {
                "module": f"services.{name.removesuffix('.py')}",
                "source_hash": _sha256_bytes(
                    resources.files("services").joinpath(name).read_bytes()
                ),
            }
            for name in names
        ]
        self._implementation_hash = "sha256:" + content_hash(
            sorted(hashes, key=lambda row: row["module"])
        )

    @staticmethod
    def _verify_dependency() -> bool:
        try:
            _ensure_vendored_semantica_importable()
            import semantica

            package_file = Path(semantica.__file__).resolve()
            root = package_file.parent.parent
            metadata = tomllib.loads(
                (root / "pyproject.toml").read_text(encoding="utf-8")
            )
            return (
                metadata.get("project", {}).get("version") == _SEMANTICA_VERSION
                and SEMANTICA_COMMIT == "701aa6a755899b8b63ce38ee971827b3e2c092e7"
                and package_file.is_relative_to(root)
            )
        except (OSError, RuntimeError, ValueError, ImportError, KeyError, TypeError):
            return False

    def _access(self, actor_context: RuntimeContextDTO) -> tuple[str, str, str]:
        return self._store._base_access(actor_context)

    def _raw_model(
        self,
        pin: ArtifactPinV1,
        model: type[BaseModel],
        *,
        organization: str,
        workspace: str,
    ):
        raw = self._build_store.get_raw(
            organization_id=organization,
            workspace_id=workspace,
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        if raw["artifact_hash"] != pin.artifact_hash:
            raise RuntimeContextError("HASH_MISMATCH")
        try:
            value = model.model_validate(raw["payload"])
        except (ValidationError, ValueError, TypeError) as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        if value.artifact_pin() != pin or canonical_json(
            value.model_dump(mode="json")
        ) != canonical_json(raw["payload"]):
            raise RuntimeContextError("HASH_MISMATCH")
        return value

    @staticmethod
    def _world_pin(context: ContextGraph) -> ArtifactPinV1:
        # ContextGraph.world is embedded in the exact ContextGraph identity;
        # workflow requests therefore pin the ContextGraph itself as world basis.
        return context.pin()

    def _kernel_pin(
        self,
        *,
        organization: str,
        workspace: str,
        pin: ArtifactPinV1,
        model: type[BaseModel],
    ):
        value = self._build_store.get(
            organization_id=organization,
            workspace_id=workspace,
            artifact_ref=pin.artifact_ref,
            contract_type=model,
        )
        if value is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        if not hasattr(value, "pin") or value.pin() != pin:
            raise RuntimeContextError("HASH_MISMATCH")
        return value

    def _context_closure(
        self, *, actor_context: RuntimeContextDTO, context_pin: ArtifactPinV1
    ) -> tuple[ContextGraph, None, RuntimeTemporalSourceGraphV1]:
        organization, workspace, _session = self._access(actor_context)
        if context_pin.artifact_kind != "context_graph":
            raise RuntimeContextError("UNSUPPORTED_ARTIFACT_KIND")
        context = self._kernel_pin(
            organization=organization,
            workspace=workspace,
            pin=context_pin,
            model=ContextGraph,
        )
        if context.state != "ADMITTED":
            raise RuntimeContextError("CONTEXT_NOT_ADMITTED")
        if context.predecessor is None or context.admission_decision is None:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        proposed = self._kernel_pin(
            organization=organization,
            workspace=workspace,
            pin=context.predecessor,
            model=ContextGraph,
        )
        decision = self._kernel_pin(
            organization=organization,
            workspace=workspace,
            pin=context.admission_decision,
            model=DecisionRecord,
        )
        try:
            validate_context_admission_transition(proposed, decision, context)
        except ValueError as exc:
            raise RuntimeContextError("INTEGRITY_UNPROVABLE") from exc
        self._kernel_pin(
            organization=organization,
            workspace=workspace,
            pin=context.ontology,
            model=Ontology,
        )
        for evidence_pin in context.evidence:
            self._kernel_pin(
                organization=organization,
                workspace=workspace,
                pin=evidence_pin,
                model=Evidence,
            )
        graph = self._source_service.get_graph(
            actor_context=actor_context, pin=context.world.source_graph
        )
        if graph.artifact_pin() != context.world.source_graph:
            raise RuntimeContextError("HASH_MISMATCH")
        return context, None, graph

    def _trusted_profile(self, organization: str, workspace: str):
        descriptor = RuntimeTemporalProviderDescriptorV1.create(
            schema_version="runtime_provider_descriptor.v1",
            organization_id=organization,
            workspace_id=workspace,
            provider_kind="semantica_temporal",
            implementation_id="semantier.runtime_temporal",
            implementation_version="v1",
            implementation_hash=self._implementation_hash,
        )
        profile = RuntimeTemporalProviderProfileV2.create(
            schema_version="runtime_provider_profile.v2",
            organization_id=organization,
            workspace_id=workspace,
            profile_id="semantica_temporal_local_v2",
            provider=descriptor.artifact_pin(),
            query_schema_version="runtime_temporal_query.v1",
            configuration=_CONFIGURATION,
            dependency=RuntimeTemporalDependencyV2(
                distribution="semantica",
                package_metadata_version=_SEMANTICA_VERSION,
                source_revision="701aa6a755899b8b63ce38ee971827b3e2c092e7",
            ),
        )
        return descriptor, profile

    def prepare_temporal_request(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context: ArtifactPinV1,
        target: GovernedTargetPinV1,
        query: TemporalQueryV1,
    ) -> RuntimeTemporalProviderRequestV1:
        organization, workspace, _session = self._access(actor_context)
        current, _selection, graph = self._context_closure(
            actor_context=actor_context, context_pin=context
        )
        query = TemporalQueryV1.model_validate(query.model_dump(mode="json"))
        if (
            query.organization_id,
            query.workspace_id,
            query.source_graph,
        ) != (organization, workspace, graph.artifact_pin()):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        archive = self._source_service.get_archive(
            actor_context=actor_context, pin=graph.source
        )
        evidence = _canonical_pins((archive.artifact_pin(), graph.normalization))
        target = GovernedTargetPinV1.model_validate(target.model_dump(mode="json"))
        if (
            target.organization_id,
            target.workspace_id,
            target.artifact,
            target.target_type,
            target.evidence,
            target.scope_bridge,
        ) != (organization, workspace, graph.artifact_pin(), "graph", evidence, None):
            raise RuntimeContextError("INVALID_STATE")
        descriptor, profile = self._trusted_profile(organization, workspace)
        request = RuntimeTemporalProviderRequestV1.create(
            schema_version="runtime_provider_request.v1",
            organization_id=organization,
            workspace_id=workspace,
            context=context,
            world=self._world_pin(current),
            target=target,
            provider=descriptor.artifact_pin(),
            profile=profile.artifact_pin(),
            query=query.artifact_pin(),
            source_pins=(graph.artifact_pin(),),
            model_pins=(),
            index_pins=(),
            namespace_pin=None,
            requested_operation=query.operation,
            settings=RuntimeTemporalRequestSettingsV1(),
        )
        self._build_store.put_many((descriptor, profile, query, request))
        return request

    def _request_closure(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context_pin: ArtifactPinV1,
        target: GovernedTargetPinV1,
        request_pin: ArtifactPinV1,
        require_head: bool,
        continuation: bool = False,
    ):
        organization, workspace, _session = self._access(actor_context)
        context, selection, graph = self._context_closure(
            actor_context=actor_context, context_pin=context_pin
        )
        request = self._raw_model(
            request_pin,
            RuntimeTemporalProviderRequestV1,
            organization=organization,
            workspace=workspace,
        )
        query = self._raw_model(
            request.query,
            TemporalQueryV1,
            organization=organization,
            workspace=workspace,
        )
        profile = self._store._read_profile(organization, workspace, request.profile)
        descriptor = self._raw_model(
            request.provider,
            RuntimeTemporalProviderDescriptorV1,
            organization=organization,
            workspace=workspace,
        )
        expected_descriptor, expected_profile = self._trusted_profile(
            organization, workspace
        )
        archive = self._source_service.get_archive(
            actor_context=actor_context, pin=graph.source
        )
        evidence = _canonical_pins((archive.artifact_pin(), graph.normalization))
        if (
            request.context != context_pin
            or request.world != self._world_pin(context)
            or request.target != target
            or request.source_pins != (graph.artifact_pin(),)
            or request.requested_operation != query.operation
            or query.source_graph != graph.artifact_pin()
            or target.artifact != graph.artifact_pin()
            or target.evidence != evidence
            or request.provider != profile.provider
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        if descriptor != expected_descriptor or profile != expected_profile:
            if continuation:
                raise RuntimeContextError(
                    "TEMPORAL_CONTINUATION_INCOMPATIBLE",
                    {"retained_profile": profile.profile_ref,
                     "current_profile": expected_profile.profile_ref},
                )
            raise RuntimeContextError("HASH_MISMATCH")
        return context, selection, graph, request, query, profile

    def evaluate(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context: ArtifactPinV1,
        target: GovernedTargetPinV1,
        provider_request: ArtifactPinV1,
        business_effective_at: datetime,
        request_id: str,
        parent_run: ArtifactPinV1 | None = None,
    ) -> RuntimeEvaluationRunV1:
        organization, workspace, session = self._access(actor_context)
        business_effective_at = _utc_seconds(business_effective_at)
        identity = RuntimeEvaluationRequestIdentityV1(
            schema_version="runtime_evaluation_request_identity.v1",
            organization_id=organization,
            workspace_id=workspace,
            actor_ref=actor_context.actor_id,
            operation="EVALUATE_RUNTIME_DOCUMENT",
            request_id=request_id,
            context=context,
            provider_request=provider_request,
            target=target,
            business_effective_at=business_effective_at,
            parent_run=parent_run,
        )
        existing = self._store.get_request_outcome(
            actor_context=actor_context, identity=identity
        )
        if existing is not None:
            return self._store.get_run(
                actor_context=actor_context, pin=existing.run_pin
            )
        current, _selection, _graph, _request, _query, _profile = self._request_closure(
            actor_context=actor_context,
            context_pin=context,
            target=target,
            request_pin=provider_request,
            require_head=True,
        )
        root = None
        if parent_run is not None:
            parent = self._store.get_run(actor_context=actor_context, pin=parent_run)
            root = parent.root_run or parent.artifact_pin()
        instant = datetime.now(timezone.utc)
        admission = RuntimeExecutionAdmissionV1.create(
            schema_version="runtime_execution_admission.v1",
            organization_id=organization,
            workspace_id=workspace,
            actor_id=actor_context.actor_id,
            session_ref=session,
            context=context,
            world=self._world_pin(current),
            provider_request=provider_request,
            target=target,
            context_stream_ref=current.artifact_ref,
            context_revision=current.revision,
            certified_plan=None,
            epoch=None,
            activation_snapshot=None,
            activation_events=(),
            business_effective_at=business_effective_at,
            admitted_at=instant,
        )
        run = RuntimeEvaluationRunV1.create(
            schema_version="runtime_evaluation_run.v1",
            organization_id=organization,
            workspace_id=workspace,
            admission=admission.artifact_pin(),
            root_run=root,
            parent_run=parent_run,
            request_id=request_id,
            created_by=actor_context.actor_id,
            created_at=instant,
        )
        state = RuntimeEvaluationStateV2.create(
            schema_version="runtime_evaluation_state.v2",
            producer_mode="TEMPORAL", max_dispatches=2,
            organization_id=organization,
            workspace_id=workspace,
            run=run.artifact_pin(),
            revision=0,
            predecessor=None,
            status="LOADING",
            capture=None,
            results=(),
            diagnostics=(),
            recorded_at=instant,
        )
        outcome = RuntimeEvaluationRequestOutcomeV1.create(
            schema_version="runtime_evaluation_request_outcome.v1",
            organization_id=organization,
            workspace_id=workspace,
            actor_ref=actor_context.actor_id,
            operation="EVALUATE_RUNTIME_DOCUMENT",
            request_id=request_id,
            normalized_request_hash=identity.normalized_request_hash(),
            run_pin=run.artifact_pin(),
            execution_admission_pin=admission.artifact_pin(),
            state_pin=state.artifact_pin(),
            epoch_pin=None,
            authority_release_receipt_pin=None,
            result_pins=(),
            recorded_at=instant,
        )

        def verify(connection) -> None:
            if connection is not self._build_store._connection:
                raise RuntimeContextError("INVALID_STATE")
            self._request_closure(
                actor_context=actor_context,
                context_pin=context,
                target=target,
                request_pin=provider_request,
                require_head=True,
            )

        winner = self._store.admit(
            actor_context=actor_context,
            identity=identity,
            admission=admission,
            run=run,
            state=state,
            outcome=outcome,
            verify_before_commit=verify,
        )
        stable = self._store.get_run(actor_context=actor_context, pin=winner.run_pin)
        if stable != run:
            return stable
        self.resume(actor_context=actor_context, run=stable.artifact_pin())
        return stable

    @staticmethod
    def _publication_hash(
        *,
        actor_context: RuntimeContextDTO,
        state: RuntimeEvaluationStateV1,
        phase: str,
        inputs: tuple[ArtifactPinV1, ...],
    ) -> str:
        return temporal_publication_hash(
            actor_id=actor_context.actor_id, state=state, phase=phase, inputs=inputs
        )

    def _next(
        self,
        *,
        actor_context: RuntimeContextDTO,
        previous: RuntimeEvaluationStateV1,
        previous_outcome: RuntimeEvaluationRequestOutcomeV1,
        status: str,
        capture: ArtifactPinV1 | None,
        results: tuple[ArtifactPinV1, ...],
        decisions: tuple[ArtifactPinV1, ...] | None = None,
        diagnostics: tuple[ArtifactPinV1, ...],
        phase: str,
        artifacts: tuple[BaseModel, ...],
        inputs: tuple[ArtifactPinV1, ...],
    ) -> RuntimeEvaluationStateV1:
        now = datetime.now(timezone.utc)
        state = type(previous).create(
            schema_version=previous.schema_version,
            **({k: getattr(previous, k) for k in ("producer_mode", "max_dispatches", "dispatch_lease_seconds", "attempt_events")} if isinstance(previous, RuntimeEvaluationStateV2) else {}),
            organization_id=previous.organization_id,
            workspace_id=previous.workspace_id,
            run=previous.run,
            revision=previous.revision + 1,
            predecessor=previous.artifact_pin(),
            status=status,
            capture=capture,
            results=results,
            decisions=previous.decisions if decisions is None else decisions,
            diagnostics=diagnostics,
            recorded_at=now,
        )
        outcome = RuntimeEvaluationRequestOutcomeV1.create(
            schema_version="runtime_evaluation_request_outcome.v1",
            organization_id=previous_outcome.organization_id,
            workspace_id=previous_outcome.workspace_id,
            actor_ref=previous_outcome.actor_ref,
            operation=previous_outcome.operation,
            request_id=previous_outcome.request_id,
            normalized_request_hash=previous_outcome.normalized_request_hash,
            run_pin=previous_outcome.run_pin,
            execution_admission_pin=previous_outcome.execution_admission_pin,
            state_pin=state.artifact_pin(),
            epoch_pin=None,
            authority_release_receipt_pin=None,
            result_pins=results,
            recorded_at=now,
        )
        published = self._store.publish(
            actor_context=actor_context,
            previous=previous,
            previous_outcome=previous_outcome,
            state=state,
            outcome=outcome,
            artifacts=artifacts,
            phase=phase,
            request_hash=self._publication_hash(
                actor_context=actor_context,
                state=previous,
                phase=phase,
                inputs=inputs,
            ),
        )
        return self._store.get_state(
            actor_context=actor_context, pin=published.state_pin
        )

    @staticmethod
    def _instant_sequence_cycle(graph: RuntimeTemporalSourceGraphV1) -> bool:
        groups: dict[datetime, list[tuple[str, str]]] = {}
        for edge in graph.relationships:
            if edge.valid_from == edge.valid_until:
                groups.setdefault(edge.valid_from, []).append(
                    (edge.source, edge.target)
                )
        for edges in groups.values():
            adjacency: dict[str, set[str]] = {}
            for source, target in edges:
                adjacency.setdefault(source, set()).add(target)
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(node: str) -> bool:
                if node in visiting:
                    return True
                if node in visited:
                    return False
                visiting.add(node)
                if any(visit(target) for target in adjacency.get(node, ())):
                    return True
                visiting.remove(node)
                visited.add(node)
                return False

            if any(visit(node) for node in tuple(adjacency)):
                return True
        return False

    def _semantic_decision(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context: ContextGraph,
        request: RuntimeTemporalProviderRequestV1,
        result: RuntimeTemporalSupportedResultV1 | RuntimeTemporalUnavailableResultV1,
    ) -> DecisionRecord:
        """Project one semantic conclusion from an operational provider observation."""

        if not context.evidence:
            raise RuntimeContextError("MISSING_EVIDENCE")
        if isinstance(result, RuntimeTemporalUnavailableResultV1):
            result_code = "UNKNOWN"
            epistemic_status = "UNKNOWN"
            diagnostics = tuple(result.limitations) or ("TEMPORAL_CAPABILITY_UNAVAILABLE",)
            proof_steps = (
                f"capability:{result.capability_state}",
                *(f"limitation:{item}" for item in result.limitations),
                f"observation:{result.result_ref}",
            )
        else:
            payload = result.payload
            has_match = bool(
                payload.entity_ids
                or payload.relationship_ids
                or payload.paths
                or payload.patterns
                or payload.world_snapshot is not None
                or (payload.metrics is not None and payload.metrics.count > 0)
            )
            result_code = "MATCH" if has_match else "NO_MATCH"
            epistemic_status = "KNOWN"
            diagnostics = ()
            proof_steps = (
                f"operation:{payload.operation}",
                f"entities:{','.join(payload.entity_ids)}",
                f"relationships:{','.join(payload.relationship_ids)}",
                f"observation:{result.result_ref}",
            )
        invocation_basis = {
            "context": context.pin().model_dump(mode="json"),
            "provider_request": request.artifact_pin().model_dump(mode="json"),
            "provider": request.provider.model_dump(mode="json"),
            "observation": result.artifact_pin().model_dump(mode="json"),
            "result_code": result_code,
        }
        invocation = ReasonerInvocation(
            invocation_id=f"temporal-evaluation:{content_hash(invocation_basis)}",
            strategy="temporal_provider",
            executable_engine=request.provider,
            rules=(),
            inputs=(context.pin(), request.artifact_pin()),
            outputs=(result_code,),
            proof_steps=proof_steps,
            evidence=context.evidence,
        )
        trace = ReasoningTrace(
            strategy="temporal_provider",
            reasoning_version=request.provider.artifact_hash,
            invocations=(invocation,),
            derived_assertion_ids=(),
            diagnostics=diagnostics,
        )
        return DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            purpose="EVALUATION",
            subject=request.target.artifact,
            context=context.pin(),
            ontology=context.ontology,
            result_code=result_code,
            epistemic_status=epistemic_status,
            evidence=context.evidence,
            admissibility=None,
            reasoning=trace,
            actor_id=actor_context.actor_id,
            reasoner_identity=request.provider.artifact_ref,
            authorization=None,
            outcome_observation=result.artifact_pin(),
            recorded_at=result.completed_at,
        )

    def _unavailable(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context: ContextGraph,
        state: RuntimeEvaluationStateV1,
        outcome: RuntimeEvaluationRequestOutcomeV1,
        request: RuntimeTemporalProviderRequestV1,
        graph: RuntimeTemporalSourceGraphV1,
        limitation: str,
        capability_state: str,
        diagnostic_code: str,
    ) -> RuntimeEvaluationStateV1:
        now = datetime.now(timezone.utc)
        result = RuntimeTemporalUnavailableResultV1.create(
            schema_version="runtime_provider_result.v1",
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
            run=state.run,
            provider_request=request.artifact_pin(),
            provider_output=None,
            kind="temporal",
            capability_state=capability_state,
            payload=None,
            evidence=_canonical_pins(
                (request.query, graph.artifact_pin(), graph.source, graph.normalization)
            ),
            proposal_pins=(),
            limitations=(limitation,),
            completed_at=now,
        )
        diagnostic = RuntimeEvaluationDiagnosticV1.create(
            schema_version="runtime_evaluation_diagnostic.v1",
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
            run=state.run,
            code=diagnostic_code,
            message="Temporal capability is unavailable for this request.",
            input_pins=(request.artifact_pin(),),
            evidence_pins=(result.artifact_pin(),),
            recorded_at=now,
        )
        decision = (
            self._semantic_decision(
                actor_context=actor_context,
                context=context,
                request=request,
                result=result,
            )
            if isinstance(context, ContextGraph)
            else None
        )
        return self._next(
            actor_context=actor_context,
            previous=state,
            previous_outcome=outcome,
            status="PARTIAL",
            capture=state.capture,
            results=(result.artifact_pin(),),
            decisions=(() if decision is None else (decision.pin(),)),
            diagnostics=(*state.diagnostics, diagnostic.artifact_pin()),
            phase="unavailable",
            artifacts=(result, diagnostic) if decision is None else (result, diagnostic, decision),
            inputs=(request.artifact_pin(),),
        )

    def _capture_model(
        self,
        *,
        state: RuntimeEvaluationStateV1,
        request: RuntimeTemporalProviderRequestV1,
        graph: RuntimeTemporalSourceGraphV1,
        kernel_capture,
        attempt=None,
    ) -> RuntimeTemporalCaptureV1:
        selected = None
        if kernel_capture.selected_relationships is not None:
            selected = tuple(
                sorted(
                    {str(row["id"]) for row in kernel_capture.selected_relationships}
                )
            )
        return (RuntimeTemporalCaptureV2 if attempt is not None else RuntimeTemporalCaptureV1).create(
            schema_version="runtime_temporal_capture.v2" if attempt is not None else "runtime_temporal_capture.v1",
            **({"attempt_id": attempt.attempt_id} if attempt is not None else {}),
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
            run=state.run,
            provider_request=request.artifact_pin(),
            source_graph=graph.artifact_pin(),
            normalization=graph.normalization,
            operation=request.requested_operation,
            raw_output=_captured(kernel_capture.raw_output),
            selected_relationship_ids=selected,
            world_snapshot=(
                None
                if kernel_capture.world_snapshot is None
                else _captured(kernel_capture.world_snapshot)
            ),
            limitations=kernel_capture.limitations,
            captured_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _relationship_maps(graph: RuntimeTemporalSourceGraphV1):
        entities = {row.id for row in graph.entities}
        relationships = {row.id: row for row in graph.relationships}
        return entities, relationships

    @staticmethod
    def _path(
        value: object,
        *,
        entities: set[str],
        relationships: dict[str, object],
    ) -> RuntimeTemporalPathV1:
        if not isinstance(value, dict):
            raise ValueError("temporal path must be an object")
        nodes = value.get("path")
        edges = value.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("temporal path fields are invalid")
        entity_ids = tuple(nodes)
        relationship_ids = tuple(
            edge.get("id") if isinstance(edge, dict) else None for edge in edges
        )
        if not all(isinstance(item, str) and item in entities for item in entity_ids):
            raise ValueError("temporal path references an unknown entity")
        if not all(
            isinstance(item, str) and item in relationships for item in relationship_ids
        ):
            raise ValueError("temporal path references an unknown relationship")
        path = RuntimeTemporalPathV1(
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
        )
        for index, relationship_id in enumerate(path.relationship_ids):
            relationship = relationships[relationship_id]
            if (
                relationship.source,
                relationship.target,
            ) != path.entity_ids[index : index + 2]:
                raise ValueError("temporal path edge does not join successive nodes")
        return path

    def _convert(
        self,
        *,
        state: RuntimeEvaluationStateV1,
        request: RuntimeTemporalProviderRequestV1,
        query: TemporalQueryV1,
        graph: RuntimeTemporalSourceGraphV1,
        archive: ArtifactPinV1,
        capture: RuntimeTemporalCaptureV1,
    ) -> tuple[RuntimeTemporalSupportedResultV1, RuntimeTemporalWorldSnapshotV1 | None]:
        raw = json.loads(capture.raw_output.canonical_json)
        entities, relationships = self._relationship_maps(graph)
        entity_ids: set[str] = set()
        relationship_ids: set[str] = set()
        paths: tuple[RuntimeTemporalPathV1, ...] = ()
        patterns: tuple[RuntimeTemporalPatternV1, ...] = ()
        metrics = None
        snapshot = None

        if query.operation in {"point", "range", "world_snapshot"}:
            selected = capture.selected_relationship_ids
            if selected is None:
                raise ValueError("selection capture is missing")
            relationship_ids.update(selected)
            for relationship_id in selected:
                relationship = relationships.get(relationship_id)
                if relationship is None:
                    raise ValueError("selection references an unknown relationship")
                entity_ids.update((relationship.source, relationship.target))
            if query.operation == "world_snapshot":
                if capture.world_snapshot is None:
                    raise ValueError("world snapshot capture is missing")
                captured_snapshot = json.loads(capture.world_snapshot.canonical_json)
                if not isinstance(captured_snapshot, dict):
                    raise ValueError("world snapshot must be an object")
                snapshot_entities = captured_snapshot.get("entities")
                snapshot_relationships = captured_snapshot.get("relationships")
                if not isinstance(snapshot_entities, list) or not isinstance(
                    snapshot_relationships, list
                ):
                    raise ValueError("world snapshot graph is invalid")
                if {
                    item.get("id")
                    for item in snapshot_entities
                    if isinstance(item, dict)
                } != entities or {
                    item.get("id")
                    for item in snapshot_relationships
                    if isinstance(item, dict)
                } != relationship_ids:
                    raise ValueError("world snapshot does not match captured selection")
                entity_ids = set(entities)
                snapshot = RuntimeTemporalWorldSnapshotV1.create(
                    schema_version="runtime_temporal_world_snapshot.v1",
                    organization_id=state.organization_id,
                    workspace_id=state.workspace_id,
                    run=state.run,
                    query=query.artifact_pin(),
                    source_graph=graph.artifact_pin(),
                    capture=capture.artifact_pin(),
                    graph=capture.world_snapshot,
                )
        elif query.operation == "path":
            if not isinstance(raw, dict) or not isinstance(raw.get("paths"), list):
                raise ValueError("native path output is invalid")
            paths = tuple(
                sorted(
                    (
                        self._path(
                            item,
                            entities=entities,
                            relationships=relationships,
                        )
                        for item in raw["paths"]
                    ),
                    key=lambda item: (item.entity_ids, item.relationship_ids),
                )
            )
            entity_ids.update(item for path in paths for item in path.entity_ids)
            relationship_ids.update(
                item for path in paths for item in path.relationship_ids
            )
        elif query.operation == "pattern":
            if not isinstance(raw, list):
                raise ValueError("native pattern output is invalid")
            converted_patterns = []
            for native_pattern in raw:
                if not isinstance(native_pattern, dict):
                    raise ValueError("native pattern is invalid")
                pattern_type = native_pattern.get("pattern_type")
                signature = native_pattern.get("signature")
                native_occurrences = native_pattern.get("occurrences")
                if (
                    pattern_type not in {"sequence", "cycle"}
                    or not isinstance(signature, list)
                    or not isinstance(native_occurrences, list)
                ):
                    raise ValueError("native pattern fields are invalid")
                frequency = native_pattern.get("frequency")
                if type(frequency) is not int or frequency != len(native_occurrences):
                    raise ValueError("native pattern frequency is invalid")
                provisional = []
                for native_occurrence in native_occurrences:
                    if not isinstance(native_occurrence, dict):
                        raise ValueError("native occurrence is invalid")
                    native_edges = native_occurrence.get("edges")
                    if not isinstance(native_edges, list):
                        raise ValueError("native occurrence edges are invalid")
                    path = self._path(
                        {"path": signature, "edges": native_edges},
                        entities=entities,
                        relationships=relationships,
                    )
                    if not path.relationship_ids:
                        raise ValueError("native occurrence must contain an edge")
                    first = relationships[path.relationship_ids[0]]
                    last = relationships[path.relationship_ids[-1]]
                    if native_occurrence.get("nodes") != list(path.entity_ids):
                        raise ValueError("native occurrence nodes are invalid")
                    for field, expected in (
                        ("start_time", first.valid_from),
                        ("end_time", last.valid_until),
                    ):
                        try:
                            span = datetime.fromisoformat(native_occurrence[field])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(
                                "native occurrence span is invalid"
                            ) from exc
                        if span.utcoffset() != timedelta(0) or span != expected:
                            raise ValueError("native occurrence span is invalid")
                    provisional.append(
                        (
                            first.valid_from,
                            last.valid_until,
                            path.entity_ids,
                            path.relationship_ids,
                        )
                    )
                provisional.sort()
                occurrences = []
                for occurrence_index, item in enumerate(provisional):
                    start_time, end_time, occurrence_entities, occurrence_edges = item
                    occurrence_ref = "runtime-temporal-occurrence:" + content_hash(
                        {
                            "capture": capture.artifact_pin().model_dump(mode="json"),
                            "pattern_type": pattern_type,
                            "signature": signature,
                            "occurrence_index": occurrence_index,
                        }
                    )
                    occurrences.append(
                        RuntimeTemporalOccurrenceV1(
                            occurrence_ref=occurrence_ref,
                            entity_ids=occurrence_entities,
                            relationship_ids=occurrence_edges,
                            start_time=start_time,
                            end_time=end_time,
                        )
                    )
                if (
                    query.min_frequency is None
                    or len(occurrences) < query.min_frequency
                ):
                    raise ValueError("native pattern violates requested frequency")
                converted_patterns.append(
                    RuntimeTemporalPatternV1(
                        pattern_type=pattern_type,
                        signature=tuple(signature),
                        frequency=len(occurrences),
                        occurrences=tuple(occurrences),
                    )
                )
            patterns = tuple(
                sorted(
                    converted_patterns,
                    key=lambda item: (item.pattern_type, item.signature),
                )
            )
            entity_ids.update(
                item
                for pattern in patterns
                for occurrence in pattern.occurrences
                for item in occurrence.entity_ids
            )
            relationship_ids.update(
                item
                for pattern in patterns
                for occurrence in pattern.occurrences
                for item in occurrence.relationship_ids
            )
        else:
            if not isinstance(raw, dict):
                raise ValueError("native evolution output is invalid")
            count = raw.get("count")
            diversity = raw.get("diversity")
            stability = raw.get("stability")
            num_relationships = raw.get("num_relationships")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or isinstance(diversity, bool)
                or not isinstance(diversity, int)
                or isinstance(num_relationships, bool)
                or not isinstance(num_relationships, int)
                or num_relationships != count
                or isinstance(stability, bool)
                or not isinstance(stability, (int, float))
                or not math.isfinite(float(stability))
                or float(stability) < 0
            ):
                raise ValueError("native evolution metrics are invalid")
            metrics = RuntimeTemporalMetricsV1(
                count=count,
                diversity=diversity,
                stability_seconds=float(stability),
            )

        payload = RuntimeTemporalPayloadV1(
            operation=query.operation,
            entity_ids=tuple(sorted(entity_ids)),
            relationship_ids=tuple(sorted(relationship_ids)),
            paths=paths,
            patterns=patterns,
            metrics=metrics,
            world_snapshot=None if snapshot is None else snapshot.artifact_pin(),
            normalization=capture.normalization,
            proof_state="query_evidence",
            proof_pins=(),
        )
        limitations = tuple(
            [*capture.limitations, "ENGINE_PROOF_NOT_AVAILABLE"]
            if "ENGINE_PROOF_NOT_AVAILABLE" not in capture.limitations
            else capture.limitations
        )
        evidence = _canonical_pins(
            (
                graph.artifact_pin(),
                archive,
                graph.normalization,
                query.artifact_pin(),
                capture.artifact_pin(),
            )
        )
        result = RuntimeTemporalSupportedResultV1.create(
            schema_version="runtime_provider_result.v1",
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
            run=state.run,
            provider_request=request.artifact_pin(),
            provider_output=capture.artifact_pin(),
            kind="temporal",
            capability_state="supported",
            payload=payload,
            evidence=evidence,
            proposal_pins=(),
            limitations=limitations,
            completed_at=datetime.now(timezone.utc),
        )
        return result, snapshot

    def _fail(
        self,
        *,
        actor_context: RuntimeContextDTO,
        state: RuntimeEvaluationStateV1,
        outcome: RuntimeEvaluationRequestOutcomeV1,
        request: RuntimeTemporalProviderRequestV1,
        code: str,
    ) -> RuntimeEvaluationStateV1:
        diagnostic = RuntimeEvaluationDiagnosticV1.create(
            schema_version="runtime_evaluation_diagnostic.v1",
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
            run=state.run,
            code=code,
            message="Temporal evaluation could not produce a valid result.",
            input_pins=(request.artifact_pin(),),
            evidence_pins=(() if state.capture is None else (state.capture,)),
            recorded_at=datetime.now(timezone.utc),
        )
        return self._next(
            actor_context=actor_context,
            previous=state,
            previous_outcome=outcome,
            status="FAILED",
            capture=state.capture,
            results=state.results,
            diagnostics=(*state.diagnostics, diagnostic.artifact_pin()),
            phase="fail",
            artifacts=(diagnostic,),
            inputs=(request.artifact_pin(),),
        )

    def resume(
        self, *, actor_context: RuntimeContextDTO, run: ArtifactPinV1
    ) -> RuntimeEvaluationStateV1:
        stable_run = self._store.get_run(actor_context=actor_context, pin=run)
        admission = self._store.get_admission(
            actor_context=actor_context, pin=stable_run.admission
        )
        while True:
            organization, workspace, _session = self._access(actor_context)
            state, outcome = self._store._validate_projection(
                organization, workspace, stable_run
            )
            if state.status in {"COMPLETE", "FAILED"}:
                return state
            _context, _selection, graph, request, query, profile = (
                self._request_closure(
                    actor_context=actor_context,
                    context_pin=admission.context,
                    target=admission.target,
                    request_pin=admission.provider_request,
                    require_head=False,
                    continuation=True,
                )
            )
            if state.status == "PARTIAL":
                if not state.results:
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                unavailable = self._store.get_result(
                    actor_context=actor_context, pin=state.results[0]
                )
                if not isinstance(unavailable, RuntimeTemporalUnavailableResultV1):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if unavailable.capability_state == "unsupported":
                    return state
                if not self._dependency_ready:
                    return state
                state = self._next(
                    actor_context=actor_context,
                    previous=state,
                    previous_outcome=outcome,
                    status="RUNNING",
                    capture=state.capture,
                    results=state.results,
                    decisions=(),
                    diagnostics=state.diagnostics,
                    phase="start",
                    artifacts=(),
                    inputs=(request.artifact_pin(), profile.artifact_pin()),
                )
                continue
            if state.status == "LOADING":
                if not self._dependency_ready:
                    return self._unavailable(
                        actor_context=actor_context,
                        context=_context,
                        state=state,
                        outcome=outcome,
                        request=request,
                        graph=graph,
                        limitation="TEMPORAL_DEPENDENCY_BLOCKED",
                        capability_state="dependency_blocked",
                        diagnostic_code="TEMPORAL_DEPENDENCY_BLOCKED",
                    )
                state = self._next(
                    actor_context=actor_context,
                    previous=state,
                    previous_outcome=outcome,
                    status="RUNNING",
                    capture=None,
                    results=(),
                    diagnostics=(),
                    phase="start",
                    artifacts=(),
                    inputs=(request.artifact_pin(), profile.artifact_pin()),
                )
                continue
            if state.capture is None:
                if (
                    query.operation == "pattern"
                    and query.pattern_type == "sequence"
                    and self._instant_sequence_cycle(graph)
                ):
                    return self._unavailable(
                        actor_context=actor_context,
                        context=_context,
                        state=state,
                        outcome=outcome,
                        request=request,
                        graph=graph,
                        limitation="TEMPORAL_INSTANT_SEQUENCE_CYCLE",
                        capability_state="unsupported",
                        diagnostic_code="TEMPORAL_OPERATION_UNSUPPORTED",
                    )
                attempt = None
                if isinstance(state, RuntimeEvaluationStateV2):
                    try:
                        state, attempt = self._store.register_attempt(actor_context=actor_context, run=run)
                    except RuntimeContextError as exc:
                        if exc.code != "DISPATCH_BUDGET_EXCEEDED":
                            raise
                        state, outcome = self._store._validate_projection(organization, workspace, stable_run)
                        return self._fail(actor_context=actor_context, state=state,
                            outcome=outcome, request=request, code="DISPATCH_BUDGET_EXCEEDED")
                    if attempt is None:
                        return state
                    state, outcome = self._store._validate_projection(organization, workspace, stable_run)
                self._access(actor_context)
                try:
                    kernel_capture = _execute_temporal_kernel(
                        query=query,
                        configuration=profile.configuration,
                        graph=self._source_service.kernel_graph(graph),
                    )
                except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
                    if attempt is not None:
                        self._store.record_attempt_exception(actor_context=actor_context, run=run, attempt=attempt, code="TEMPORAL_EXECUTION_FAILED")
                        state, outcome = self._store._validate_projection(organization, workspace, stable_run)
                        registrations = [event for event in state.attempt_events if event.kind == "DISPATCH_REGISTERED"]
                        if state.capture is not None or registrations[-1].attempt_id != attempt.attempt_id:
                            return state
                    return self._fail(actor_context=actor_context, state=state,
                        outcome=outcome, request=request, code="TEMPORAL_EXECUTION_FAILED")
                try:
                    capture = self._capture_model(
                        state=state, request=request, graph=graph,
                        kernel_capture=kernel_capture, attempt=attempt)
                except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
                    return self._fail(actor_context=actor_context, state=state,
                        outcome=outcome, request=request, code="TEMPORAL_CAPTURE_INVALID")
                if attempt is not None:
                    self._store.capture_attempt(actor_context=actor_context, run=run, attempt=attempt, capture=capture)
                    continue
                state = self._next(
                    actor_context=actor_context,
                    previous=state,
                    previous_outcome=outcome,
                    status="RUNNING",
                    capture=capture.artifact_pin(),
                    results=state.results,
                    diagnostics=state.diagnostics,
                    phase="capture",
                    artifacts=(capture,),
                    inputs=(request.artifact_pin(), graph.artifact_pin()),
                )
                continue
            capture = self._store.get_capture(
                actor_context=actor_context, pin=state.capture
            )
            archive = self._source_service.get_archive(
                actor_context=actor_context, pin=graph.source
            )
            try:
                result, snapshot = self._convert(
                    state=state,
                    request=request,
                    query=query,
                    graph=graph,
                    archive=archive.artifact_pin(),
                    capture=capture,
                )
            except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
                return self._fail(
                    actor_context=actor_context,
                    state=state,
                    outcome=outcome,
                    request=request,
                    code="TEMPORAL_OUTPUT_INVALID",
                )
            decision = (
                self._semantic_decision(
                    actor_context=actor_context,
                    context=_context,
                    request=request,
                    result=result,
                )
                if isinstance(_context, ContextGraph)
                else None
            )
            artifacts: tuple[BaseModel, ...] = (
                ((result,) if snapshot is None else (snapshot, result))
                if decision is None
                else ((result, decision) if snapshot is None else (snapshot, result, decision))
            )
            return self._next(
                actor_context=actor_context,
                previous=state,
                previous_outcome=outcome,
                status="COMPLETE",
                capture=capture.artifact_pin(),
                results=(result.artifact_pin(),),
                decisions=(() if decision is None else (decision.pin(),)),
                diagnostics=state.diagnostics,
                phase="complete",
                artifacts=artifacts,
                inputs=(capture.artifact_pin(),),
            )


__all__ = ["RuntimeDocumentEvaluationService"]
