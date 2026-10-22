"""Apply reviewed graph deltas through immutable Semantica graph artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.canonical_skill_registry import (
    SkillRegistryError,
    select_for_new_run,
)
from services.feedback_attribution_service import FeedbackAttributionService
from services.semantica_mvl_control_adapter import SemanticaMvlControlAdapter
from storage.graph_schema_learning_store import (
    GraphDeltaCandidate,
    GraphSchemaLearningStore,
)


class GraphSchemaLearningService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        domain_id: str = "canonical-list",
        graph_factory: Any | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.graph_factory = graph_factory
        self.control = SemanticaMvlControlAdapter(
            workspace_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
            domain_id=domain_id,
        )
        self.learning_store = GraphSchemaLearningStore(
            workspace_root, workspace_id=workspace_id
        )
        self.attribution_service = FeedbackAttributionService(
            workspace_root, workspace_id=workspace_id
        )

    def create_delta_from_reviewed_attribution(
        self,
        *,
        experiment_id: str,
        base_graph_version_ref: str,
        attribution_refs: list[str],
        feedback_refs: list[str],
        proposed_node_changes: list[dict[str, Any]],
        proposed_edge_changes: list[dict[str, Any]] | None = None,
        rationale_summary: str,
        base_ontology_version_ref: str | None = None,
    ) -> GraphDeltaCandidate:
        for attribution_ref in attribution_refs:
            attribution = self.attribution_service.get_attribution(attribution_ref)
            if attribution.review_state not in {"human_accepted", "human_edited"}:
                raise ValueError(
                    "graph mutation requires human-reviewed feedback attribution"
                )
        return self.learning_store.create_graph_delta_candidate(
            experiment_id=experiment_id,
            base_graph_version_ref=base_graph_version_ref,
            base_ontology_version_ref=base_ontology_version_ref,
            feedback_refs=feedback_refs,
            attribution_refs=attribution_refs,
            proposed_node_changes=proposed_node_changes,
            proposed_edge_changes=proposed_edge_changes,
            rationale_summary=rationale_summary,
        )

    def apply_reviewed_delta(
        self,
        *,
        delta_id: str,
        selected_by: str,
    ) -> dict[str, Any]:
        delta = self.learning_store.get_graph_delta_candidate(delta_id)
        if delta.review_state != "human_accepted":
            raise ValueError("only human_accepted graph deltas may be applied")
        base = self.control.resolve_graph_ref(delta.base_graph_version_ref)
        graph = self.control.repository.load_graph(
            base.graph_metadata.artifact_id,
            graph_factory=self.graph_factory,
        )
        _apply_changes(graph, delta)
        metadata = self.control.repository.create_checkpoint(
            graph,
            lineage_id=(
                base.graph_metadata.lineage_id
                or base.graph_metadata.lineage_root_artifact_id
            ),
            graph_id=base.graph_metadata.graph_id,
            graph_version=f"{base.graph_metadata.graph_version}-{delta.delta_id}",
            artifact_kind="candidate",
            semantica_commit=base.graph_metadata.semantica_commit,
            source_manifest_hash=base.graph_metadata.source_manifest_hash,
            created_by_user_id=delta.reviewed_by or selected_by,
            parent_artifact_id=base.graph_metadata.artifact_id,
            governance_ref=delta.delta_id,
        )
        selection = self.control.select_graph_for_mvl(
            experiment_id=delta.experiment_id,
            profile_ref=base.selection.profile_ref,
            profile_version=base.selection.profile_version,
            graph_artifact_id=metadata.artifact_id,
            selected_by=selected_by,
        )
        return {
            "delta": delta.as_dict(),
            "graph_artifact": metadata.as_dict(),
            "selection": selection.as_dict(),
        }

    def record_delta_review(
        self,
        delta_id: str,
        *,
        review_state: str,
        reviewed_by: str,
    ) -> GraphDeltaCandidate:
        return self.learning_store.record_graph_delta_review(
            delta_id,
            review_state=review_state,  # type: ignore[arg-type]
            reviewed_by=reviewed_by,
        )

    def create_learning_candidate(
        self,
        *,
        source_delta_ref: str,
        evidence_refs: list[str],
        hypothesis: str,
    ) -> dict[str, Any]:
        return self.learning_store.create_learning_candidate(
            source_delta_ref=source_delta_ref,
            evidence_refs=evidence_refs,
            hypothesis=hypothesis,
        )

    def promote_learning_candidate(
        self, candidate_id: str, *, promoted_by: str
    ) -> dict[str, Any]:
        return self.learning_store.promote_learning_candidate(
            candidate_id, promoted_by=promoted_by
        )

    def create_skill_patch_proposal(
        self,
        *,
        target_skill: str,
        base_skill_ref: str,
        learning_candidate_ref: str,
        evidence_refs: list[str],
        patch_unified_diff: str,
        proposed_by: str,
    ) -> dict[str, Any]:
        if target_skill != "knowledge-graph-building":
            raise ValueError(
                "new procedural skill patches must target knowledge-graph-building"
            )
        try:
            registered_base = select_for_new_run(skill_ref=base_skill_ref)
        except SkillRegistryError:
            if (
                self.learning_store.resolve_governed_overlay_base(
                    skill_name=target_skill,
                    skill_ref=base_skill_ref,
                )
                is None
            ):
                raise
        else:
            if registered_base is None or registered_base.name != target_skill:
                raise ValueError("base_skill_ref must match target_skill")
        return self.learning_store.create_skill_patch_proposal(
            target_skill=target_skill,
            base_skill_ref=base_skill_ref,
            learning_candidate_ref=learning_candidate_ref,
            evidence_refs=evidence_refs,
            patch_unified_diff=patch_unified_diff,
            proposed_by=proposed_by,
        )

    def record_skill_patch_review(
        self,
        proposal_id: str,
        *,
        review_state: str,
        reviewed_by: str,
        reviewed_semantic_version: str | None = None,
        reviewed_overlay_content: str | None = None,
    ) -> dict[str, Any]:
        return self.learning_store.record_skill_patch_review(
            proposal_id,
            review_state=review_state,
            reviewed_by=reviewed_by,
            reviewed_semantic_version=reviewed_semantic_version,
            reviewed_overlay_content=reviewed_overlay_content,
        )

    def activate_workspace_overlay(
        self,
        *,
        proposal_id: str,
        activated_by: str,
    ) -> dict[str, Any]:
        return self.learning_store.activate_reviewed_skill_patch(
            proposal_id=proposal_id,
            activated_by=activated_by,
        )

    def rollback_effective_skill(
        self,
        *,
        activated_by: str,
        skill_name: str = "knowledge-graph-building",
    ) -> dict[str, Any]:
        return self.learning_store.rollback_effective_skill_version(
            skill_name=skill_name,
            activated_by=activated_by,
        )

    def latest_effective_skill_version(
        self, *, skill_name: str = "knowledge-graph-building"
    ) -> dict[str, Any] | None:
        return self.learning_store.latest_effective_skill_version(skill_name=skill_name)


def _apply_changes(graph: Any, delta: GraphDeltaCandidate) -> None:
    nodes = getattr(graph, "nodes", None)
    edges = getattr(graph, "edges", None)
    if not isinstance(nodes, dict) or not isinstance(edges, dict):
        raise TypeError("graph delta application requires dict-like nodes and edges")
    for change in delta.proposed_node_changes:
        operation = change.get("operation")
        node_id = change.get("node_id")
        if not operation or not node_id:
            raise ValueError("node changes require operation and node_id")
        if operation == "upsert":
            prior = dict(nodes.get(node_id, {"id": node_id, "properties": {}}))
            replacement = {**prior, **change.get("node", {})}
            replacement.setdefault("id", node_id)
            replacement.setdefault("properties", {})
            if "properties" in prior or "properties" in change.get("node", {}):
                replacement["properties"] = {
                    **prior.get("properties", {}),
                    **change.get("node", {}).get("properties", {}),
                }
            nodes[node_id] = replacement
        elif operation == "delete":
            nodes.pop(node_id, None)
        else:
            raise ValueError(f"unsupported node change operation: {operation}")
    for change in delta.proposed_edge_changes:
        operation = change.get("operation")
        edge_id = change.get("edge_id")
        if not operation or not edge_id:
            raise ValueError("edge changes require operation and edge_id")
        if operation == "upsert":
            prior = dict(edges.get(edge_id, {"id": edge_id, "properties": {}}))
            replacement = {**prior, **change.get("edge", {})}
            replacement.setdefault("id", edge_id)
            replacement.setdefault("properties", {})
            if "properties" in prior or "properties" in change.get("edge", {}):
                replacement["properties"] = {
                    **prior.get("properties", {}),
                    **change.get("edge", {}).get("properties", {}),
                }
            edges[edge_id] = replacement
        elif operation == "delete":
            edges.pop(edge_id, None)
        else:
            raise ValueError(f"unsupported edge change operation: {operation}")
