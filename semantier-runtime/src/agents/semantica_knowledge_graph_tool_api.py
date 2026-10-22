from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from contracts.accepted_graph_release_v2 import (
    AcceptedGraphReleaseV2,
    parse_accepted_graph_release_unverified,
)
from knowledge_extraction.service import KnowledgeExtractionService
from runtime_paths import sqlite_db_path
from semantier.knowledge_feature_packs.candidate_graph import conservative_identity_candidates
from storage.human_grounding_store import HumanGroundingStore
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore


def _eos_path() -> Path:
    return Path(os.environ.get(
        "SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db")),
    ))


def _default_extraction_service() -> KnowledgeExtractionService:
    return KnowledgeExtractionService(_eos_path())


def _default_extraction_store() -> KnowledgeExtractionStore:
    return KnowledgeExtractionStore(_eos_path())


def _default_grounding_store() -> HumanGroundingStore:
    return HumanGroundingStore(_eos_path())


def _default_source_store() -> KnowledgeBuilderStore:
    return KnowledgeBuilderStore(_eos_path())


def _trusted_context(kwargs: dict[str, Any]) -> Any:
    context = kwargs.get("ctx") or kwargs.get("request_context")
    if context is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    if not str(getattr(context, "organization_id", "") or "").strip():
        raise ValueError("AUTHENTICATED_ORGANIZATION_CONTEXT_REQUIRED")
    return context


class SemanticaKnowledgeGraphToolAPI:
    """Authenticated plugin bridge; semantic authority remains in core services."""

    def __init__(
        self,
        *,
        context: Any,
        extraction_service_factory: Callable[[], KnowledgeExtractionService] = _default_extraction_service,
        extraction_store_factory: Callable[[], KnowledgeExtractionStore] = _default_extraction_store,
        grounding_store_factory: Callable[[], HumanGroundingStore] = _default_grounding_store,
        source_store_factory: Callable[[], KnowledgeBuilderStore] = _default_source_store,
    ) -> None:
        self._context = context
        self._extraction_service_factory = extraction_service_factory
        self._extraction_store_factory = extraction_store_factory
        self._grounding_store_factory = grounding_store_factory
        self._source_store_factory = source_store_factory

    def invoke(self, *, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_{operation}", None)
        if not callable(handler):
            raise ValueError(f"SEMANTICA_KNOWLEDGE_GRAPH_OPERATION_UNSUPPORTED:{operation}")
        return handler(payload)

    def _extract(self, payload: dict[str, Any]) -> dict[str, Any]:
        organization_id = str(self._context.organization_id).strip()
        workspace_id = str(getattr(self._context, "workspace_id", "") or "").strip() or None
        return self._extraction_service_factory().start(
            organization_id=organization_id,
            workspace_id=workspace_id,
            discovery_run_id=str(payload.get("discovery_run_id") or "").strip(),
            document_id=str(payload.get("document_id") or "").strip(),
            provider=str(payload.get("provider") or "semantica").strip(),
            package_id=str(payload.get("knowledge_scope_id") or "").strip(),
        )

    def _ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = payload.get("source") or {}
        if not isinstance(source, dict):
            raise ValueError("source is required")
        source_registry_id = str(source.get("source_registry_id") or "").strip()
        if not source_registry_id:
            raise ValueError(
                "source_registry_id is required; upload/artifact references must be admitted through the governed import boundary"
            )
        organization_id, workspace_id = self._scope()
        store = self._source_store_factory()
        admitted = store.get_source(source_registry_id)
        if admitted is None:
            admitted = store.find_source_by_governed_id(source_registry_id)
        if (admitted is None or admitted.get("organization_id") != organization_id or
                admitted.get("workspace_id") != workspace_id):
            raise ValueError("governed knowledge source not found for active authority scope")
        return admitted

    def _deduplicate(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(payload.get("candidate_graph_id") or "").strip()
        _, graph = self._candidate(candidate_id)
        candidates = conservative_identity_candidates(graph)
        requested = {str(item).strip() for item in payload.get("candidate_ids") or [] if str(item).strip()}
        if requested:
            candidates = {
                key: [item for item in values if item["candidate_id"] in requested]
                for key, values in candidates.items()
            }
        review_candidates = [*candidates["exact_aliases"], *candidates["review_required"]]
        node_ids = list(dict.fromkeys(
            stable_id
            for item in review_candidates
            for stable_id in (item["source_node_id"], item["target_node_id"])
        ))
        result = {
            "message": "Entity-resolution candidates require governed review.",
            "focus": {
                "nodeIds": node_ids,
                "edgeIds": [item["edge_id"] for item in review_candidates],
                "ruleIds": [],
                "sourceAnchors": [],
            },
            "candidateGraphId": candidate_id,
            "acceptedReleaseId": None,
            "proposedAction": "review_merge" if review_candidates else "inspect",
        }
        return {
            "knowledgeWorkbenchResult": result,
            "candidateReview": {
                "candidateGraphId": candidate_id,
                **candidates,
                "authority": "candidate_only",
            },
        }

    def _scope(self) -> tuple[str, str | None]:
        organization_id = str(self._context.organization_id).strip()
        workspace_id = str(getattr(self._context, "workspace_id", "") or "").strip() or None
        return organization_id, workspace_id

    def _candidate(self, candidate_graph_id: str):
        store = self._grounding_store_factory()
        organization_id, workspace_id = self._scope()
        owner = store.get_candidate_graph_owner(candidate_graph_id)
        if not owner or owner.get("organization_id") != organization_id:
            raise ValueError("candidate graph not found for organization")
        if owner.get("workspace_id") != workspace_id:
            raise ValueError("candidate graph not found for workspace")
        return store, store.get_candidate_graph(candidate_graph_id)

    def _query(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or "").strip().casefold()
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(int(payload.get("limit") or 50), 200))
        organization_id, workspace_id = self._scope()
        assertions = self._extraction_store_factory().list_assertions(
            organization_id=organization_id, workspace_id=workspace_id, limit=limit,
        )
        serialized = [item.model_dump(mode="json") for item in assertions]
        matches = [item for item in serialized if query in str(item).casefold()]
        return {"query": query, "results": matches[:limit], "total": len(matches)}

    def _provenance(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(payload.get("candidate_graph_id") or "").strip()
        _, graph = self._candidate(candidate_id)
        return {
            "candidate_graph_id": graph.candidate_graph_id,
            "extraction_run_id": graph.extraction_run_id,
            "source_document_hash": graph.source_document_hash,
            "provider": graph.machine_proposal.model_dump(mode="json"),
            "graph_hash": graph.graph_hash(),
        }

    def _validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(payload.get("candidate_graph_id") or "").strip()
        graph_version = str(payload.get("graph_version") or "").strip()
        if candidate_id:
            _, graph = self._candidate(candidate_id)
            return {"valid": True, "object_ref": candidate_id, "content_hash": graph.graph_hash()}
        if graph_version:
            store = self._grounding_store_factory()
            organization_id, workspace_id = self._scope()
            release = store.get_graph_release(
                graph_version, organization_id=organization_id, workspace_id=workspace_id,
            )
            return {"valid": True, "object_ref": graph_version, "content_hash": release.get("release_hash")}
        raise ValueError("candidate_graph_id or graph_version is required")

    def _visualize(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(payload.get("candidate_graph_id") or "").strip()
        graph_version = str(payload.get("graph_version") or "").strip()
        if candidate_id:
            _, graph = self._candidate(candidate_id)
            focus = {str(item) for item in payload.get("focus_ids") or []}
            node_ids = [item.node_id for item in graph.nodes if not focus or item.node_id in focus]
            edge_ids = [item.edge_id for item in graph.edges if not focus or item.edge_id in focus]
            rule_ids = [item.rule_id for item in graph.rules if not focus or item.rule_id in focus]
            anchors = []
            seen = set()
            for item in (*graph.nodes, *graph.edges, *graph.rules):
                stable_id = getattr(item, "node_id", None) or getattr(item, "edge_id", None) or getattr(item, "rule_id", None)
                if focus and stable_id not in focus:
                    continue
                for anchor in getattr(item, "source_anchors", ()):
                    if anchor.anchor_id not in seen:
                        anchors.append(anchor.model_dump(mode="json")); seen.add(anchor.anchor_id)
            return {
                "message": "Candidate graph ready for governed review.",
                "focus": {"nodeIds": node_ids, "edgeIds": edge_ids, "ruleIds": rule_ids, "sourceAnchors": anchors},
                "candidateGraphId": candidate_id,
                "acceptedReleaseId": None,
                "proposedAction": "inspect",
            }
        if graph_version:
            store = self._grounding_store_factory()
            organization_id, workspace_id = self._scope()
            release = store.get_graph_release(
                graph_version, organization_id=organization_id, workspace_id=workspace_id,
            )
            release_model = parse_accepted_graph_release_unverified(release)
            if not isinstance(release_model, AcceptedGraphReleaseV2):
                return {
                    "message": "Accepted release ready for comparison.",
                    "focus": {
                        "nodeIds": [], "edgeIds": [], "ruleIds": [],
                        "sourceAnchors": [],
                    },
                    "candidateGraphId": None,
                    "acceptedReleaseId": graph_version,
                    "proposedAction": "compare_release",
                }
            nodes = tuple(release_model.nodes)
            edges = tuple(release_model.edges)
            rules = tuple(release_model.rules)
            anchors = []
            seen_anchors = set()
            for item in (*nodes, *edges, *rules):
                for anchor in item.get("source_anchors") or ():
                    anchor_id = str(anchor.get("anchor_id") or "")
                    if anchor_id and anchor_id not in seen_anchors:
                        anchors.append(dict(anchor))
                        seen_anchors.add(anchor_id)
            focus = {
                "nodeIds": [str(item["node_id"]) for item in nodes],
                "edgeIds": [str(item["edge_id"]) for item in edges],
                "ruleIds": [str(item["rule_id"]) for item in rules],
                "sourceAnchors": anchors,
            }
            response = {
                "message": "Accepted release ready for comparison.",
                "focus": focus,
                "candidateGraphId": None,
                "acceptedReleaseId": graph_version,
                "proposedAction": "compare_release",
            }
            focus["relationInstanceIds"] = [
                item.relation_instance_ref for item in release_model.relation_membership
            ]
            focus["patternIds"] = [
                item.pattern_ref for item in release_model.pattern_membership
            ]
            response["releaseContract"] = release_model.schema_version
            return response
        raise ValueError("candidate_graph_id or graph_version is required")


def authenticated_tool_api(kwargs: dict[str, Any]) -> SemanticaKnowledgeGraphToolAPI:
    return SemanticaKnowledgeGraphToolAPI(context=_trusted_context(kwargs))
