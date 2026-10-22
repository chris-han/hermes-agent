"""Deterministic WorkingContext projection over one exact admitted ContextGraph."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import ContextGraph, HumanResponsibilityContext, WorkingContext
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.working_context_selection_policy import working_context_selection_policy
from storage.ontology_build_store import OntologyBuildStore


class WorkingContextProjectionService:
    """Form one bounded runtime projection without creating a new semantic store."""

    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
        runtime_actor_id: str,
    ) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("working context projection requires the shared kernel store")
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("working context projection requires the runtime authority resolver")
        if not runtime_actor_id.strip():
            raise ValueError("runtime actor id is required")
        self._store = build_store
        self._authority = authority_resolver
        self._runtime_actor_id = runtime_actor_id

    @staticmethod
    def selection_policy() -> ArtifactPinV1:
        return working_context_selection_policy()

    def _load_context(
        self,
        *,
        actor_context: RuntimeContextDTO,
        pin: ArtifactPinV1,
    ) -> ContextGraph:
        if pin.artifact_kind != "context_graph":
            raise ValueError("CONTEXT_GRAPH_REQUIRED")
        self._authority.require_current_actor(actor_context=actor_context)
        context = self._store.get(
            organization_id=actor_context.organization_id,
            workspace_id=str(actor_context.workspace_id or ""),
            artifact_ref=pin.artifact_ref,
            contract_type=ContextGraph,
        )
        if context is None:
            raise ValueError("CONTEXT_NOT_FOUND")
        if context.pin() != pin:
            raise ValueError("HASH_MISMATCH")
        if context.state != "ADMITTED":
            raise ValueError("ADMITTED_CONTEXT_REQUIRED")
        return context

    def _responsibility(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context: ArtifactPinV1,
        delegated_operation: str,
        commitment_mode: Literal["AUTO", "NOTIFY", "CONFIRM"],
        captured_at: datetime,
    ) -> HumanResponsibilityContext:
        basis = self._authority.current_actor_authorization_basis(
            actor_context=actor_context,
            operation="FORM_WORKING_CONTEXT",
            subject=context,
            authorization_class="TENANT_MEMBER",
        )
        return HumanResponsibilityContext.create(
            organization_id=basis.organization_id,
            workspace_id=basis.workspace_id,
            responsibility_bearer_id=basis.actor_id,
            delegated_actor_id=self._runtime_actor_id,
            delegated_operation=delegated_operation,
            commitment_mode=commitment_mode,
            scope_ref=basis.scope_ref,
            captured_at=captured_at,
        )

    @staticmethod
    def _unique_ids(values: tuple[str, ...], label: str) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError(f"DUPLICATE_{label.upper()}_SELECTION")
        return tuple(sorted(values))

    def project(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context: ArtifactPinV1,
        task_ref: str,
        task_goal: str,
        selected_node_ids: tuple[str, ...],
        selected_edge_ids: tuple[str, ...],
        selected_assertion_ids: tuple[str, ...],
        delegated_operation: str,
        commitment_mode: Literal["AUTO", "NOTIFY", "CONFIRM"],
        formed_at: datetime,
    ) -> WorkingContext:
        graph = self._load_context(actor_context=actor_context, pin=context)
        node_ids = self._unique_ids(selected_node_ids, "node")
        edge_ids = self._unique_ids(selected_edge_ids, "edge")
        assertion_ids = self._unique_ids(selected_assertion_ids, "assertion")

        nodes = {item.node_id: item for item in graph.nodes}
        edges = {item.edge_id: item for item in graph.edges}
        assertions = {item.assertion_id: item for item in graph.assertions}
        if any(item not in nodes for item in node_ids):
            raise ValueError("UNKNOWN_NODE_SELECTION")
        if any(item not in edges for item in edge_ids):
            raise ValueError("UNKNOWN_EDGE_SELECTION")
        if any(item not in assertions for item in assertion_ids):
            raise ValueError("UNKNOWN_ASSERTION_SELECTION")

        selected_node_set = set(node_ids)
        for edge_id in edge_ids:
            edge = edges[edge_id]
            if edge.source_id not in selected_node_set or edge.target_id not in selected_node_set:
                raise ValueError("SELECTED_EDGE_ENDPOINTS_REQUIRED")

        evidence_by_key = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash): pin
            for assertion_id in assertion_ids
            for pin in assertions[assertion_id].evidence
        }
        graph_evidence_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in graph.evidence
        }
        if not set(evidence_by_key) <= graph_evidence_keys:
            raise ValueError("OUT_OF_GRAPH_EVIDENCE")
        evidence = tuple(evidence_by_key[key] for key in sorted(evidence_by_key))
        responsibility = self._responsibility(
            actor_context=actor_context,
            context=graph.pin(),
            delegated_operation=delegated_operation,
            commitment_mode=commitment_mode,
            captured_at=formed_at,
        )
        return WorkingContext.create(
            organization_id=graph.organization_id,
            workspace_id=graph.workspace_id,
            context_graph=graph.pin(),
            task_ref=task_ref,
            task_goal=task_goal,
            ontology=graph.ontology,
            selected_node_ids=node_ids,
            selected_edge_ids=edge_ids,
            selected_assertion_ids=assertion_ids,
            evidence=evidence,
            unresolved_questions=graph.intent.unresolved_questions,
            selection_policy=working_context_selection_policy(),
            responsibility=responsibility,
            formed_at=formed_at,
        )

    def project_for_reasoning(
        self,
        *,
        actor_context: RuntimeContextDTO,
        context: ArtifactPinV1,
        task_ref: str,
        task_goal: str,
        selected_assertion_ids: tuple[str, ...],
        delegated_operation: str,
        commitment_mode: Literal["AUTO", "NOTIFY", "CONFIRM"],
        formed_at: datetime,
    ) -> WorkingContext:
        graph = self._load_context(actor_context=actor_context, pin=context)
        assertion_ids = self._unique_ids(selected_assertion_ids, "assertion")
        if not assertion_ids:
            raise ValueError("REASONING_REQUIRES_CONTEXT_FACTS")
        assertions = {item.assertion_id: item for item in graph.assertions}
        if any(item not in assertions for item in assertion_ids):
            raise ValueError("UNKNOWN_ASSERTION_SELECTION")
        graph_node_ids = {item.node_id for item in graph.nodes}
        selected_nodes = tuple(
            sorted(
                {
                    ref
                    for assertion_id in assertion_ids
                    for ref in (
                        assertions[assertion_id].subject_ref,
                        assertions[assertion_id].object_ref,
                    )
                    if ref in graph_node_ids
                }
            )
        )
        return self.project(
            actor_context=actor_context,
            context=context,
            task_ref=task_ref,
            task_goal=task_goal,
            selected_node_ids=selected_nodes,
            selected_edge_ids=(),
            selected_assertion_ids=assertion_ids,
            delegated_operation=delegated_operation,
            commitment_mode=commitment_mode,
            formed_at=formed_at,
        )


__all__ = ["WorkingContextProjectionService"]
