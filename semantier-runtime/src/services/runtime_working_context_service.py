"""Governed session binding prerequisites for runtime task contexts.

This service owns task-persistent/session-scoped runtime state, not the decision-relative
semantic-kernel ``WorkingContext`` defined by the Working Context Runtime Contract.
"""

from __future__ import annotations

from agents.auth_session import RequestContext
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_temporal_source import RuntimeTemporalContextSelectionV1
from contracts.runtime_working_context import (
    RuntimeTaskContextV3,
    RuntimeWorldScopeV1,
)
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.runtime_errors import RuntimeContextError
from services.runtime_session_service import GovernedRuntimeSessionResolver
from storage.runtime_working_context_store import RuntimeTaskContextStore


class RuntimeTaskContextService:
    """Trusted composer for explicit temporal V3 context creation."""

    def __init__(
        self,
        *,
        store: RuntimeTaskContextStore,
        authority_resolver: RuntimeAuthorityResolver,
        session_resolver: GovernedRuntimeSessionResolver,
    ) -> None:
        if not isinstance(store, RuntimeTaskContextStore):
            raise TypeError("store must be RuntimeTaskContextStore")
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("authority_resolver must be RuntimeAuthorityResolver")
        if not isinstance(session_resolver, GovernedRuntimeSessionResolver):
            raise TypeError("session_resolver must be GovernedRuntimeSessionResolver")
        self._store = store
        self._authority_resolver = authority_resolver
        self._session_resolver = session_resolver

    def get_temporal_context(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeTaskContextV3:
        """Read the exact persisted selection without formation or live execution."""
        from services.runtime_temporal_source_service import (
            RuntimeTemporalSourceService,
        )

        self._authority_resolver.require_current_actor(actor_context=actor_context)
        workspace = actor_context.workspace_id
        session = actor_context.session_id
        if not workspace or not session:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        if (
            self._session_resolver.resolve(
                actor_context=actor_context, session_ref=session
            )
            != session
        ):
            raise RuntimeContextError("FORBIDDEN_SESSION")
        context = self._store.get_context(
            organization_id=actor_context.organization_id,
            workspace_id=workspace,
            pin=pin,
        )
        if not isinstance(context, RuntimeTaskContextV3):
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        selection = self._store._read_artifact(
            actor_context.organization_id, workspace, context.selection
        )
        if not isinstance(selection, RuntimeTemporalContextSelectionV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        if (
            selection.session_ref != session
            or selection.actor_id != actor_context.actor_id
        ):
            raise RuntimeContextError("FORBIDDEN_SESSION")
        source = RuntimeTemporalSourceService(
            build_store=self._store._build_store,
            authority_resolver=self._authority_resolver,
            session_resolver=self._session_resolver,
        )
        graph = source.get_graph(
            actor_context=actor_context, pin=selection.source_graph
        )
        if (
            context.goal != selection.goal
            or context.world != selection.world
            or context.world.source_graph != graph.artifact_pin()
            or context.semantic_pins != selection.semantic_pins
            or tuple(element.object_ref for element in context.elements)
            != selection.selected_entity_ids
            or any(
                element.admission != "candidate" or element.active or element.derived
                for element in context.elements
            )
        ):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return context

    def create_temporal_context(
        self,
        *,
        actor_context: RuntimeContextDTO,
        goal: ArtifactPinV1,
        source_graph: ArtifactPinV1,
        world: RuntimeWorldScopeV1,
        selected_entity_ids: tuple[str, ...],
        request_id: str,
        semantic_pins: tuple[ArtifactPinV1, ...] = (),
    ) -> RuntimeTaskContextV3:
        self._authority_resolver.require_current_actor(actor_context=actor_context)
        workspace = actor_context.workspace_id
        session = actor_context.session_id
        if not workspace:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if not session:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        if (
            self._session_resolver.resolve(
                actor_context=actor_context, session_ref=session
            )
            != session
        ):
            raise RuntimeContextError("FORBIDDEN_SESSION")
        self._authority_resolver.resolve_pins(
            actor_context=actor_context, pins=(goal, source_graph, *semantic_pins)
        )
        selection = RuntimeTemporalContextSelectionV1.create(
            schema_version="runtime_temporal_context_selection.v1",
            organization_id=actor_context.organization_id,
            workspace_id=workspace,
            profile_id="explicit_temporal_selection_v1",
            actor_id=actor_context.actor_id,
            session_ref=session,
            request_id=request_id,
            goal=goal,
            source_graph=source_graph,
            world=world,
            semantic_pins=semantic_pins,
            selected_entity_ids=selected_entity_ids,
        )
        receipt = self._store.publish_temporal_context(
            actor_context=actor_context,
            selection=selection,
            authority_resolver=self._authority_resolver,
            session_resolver=self._session_resolver,
        )
        context = self._store.get_context(
            organization_id=actor_context.organization_id,
            workspace_id=workspace,
            pin=receipt.successor,
        )
        if not isinstance(context, RuntimeTaskContextV3):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return context


def runtime_context_dto_from_request_context(
    *, ctx: RequestContext, session_id: str, channel: str | None
) -> RuntimeContextDTO:
    if not ctx.authenticated or not ctx.organization_id or not ctx.workspace_id:
        raise RuntimeContextError("MISSING_WORKSPACE")
    canonical = str(session_id or "").strip()
    if (
        not canonical
        or canonical.startswith("public:")
        or not canonical.startswith(f"{ctx.workspace_id}:")
    ):
        raise RuntimeContextError("FORBIDDEN_SESSION")
    return RuntimeContextDTO(
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
        actor_id=str(ctx.user_id),
        channel=channel,
        session_id=canonical,
    )


__all__ = [
    "GovernedRuntimeSessionResolver",
    "RuntimeTaskContextService",
    "runtime_context_dto_from_request_context",
]
