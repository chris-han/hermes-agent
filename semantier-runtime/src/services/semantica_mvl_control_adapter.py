"""Semantier control adapter for MVL-selected Semantica artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantier.knowledge_feature_packs.semantica_adapter import (
    _ensure_vendored_semantica_importable,
)
from semantier.semantica_context_graph.artifact_repository import (
    ArtifactMetadata,
    ArtifactRepositoryError,
    ContextGraphArtifactRepository,
)
from storage.mvl_selection_store import MvlSelection, MvlSelectionStore


@dataclass(frozen=True)
class ResolvedMvlGraphSelection:
    selection: MvlSelection
    graph_metadata: ArtifactMetadata


class SemanticaMvlControlAdapter:
    """Resolve opaque MVL refs to immutable Semantica graph artifacts."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        domain_id: str = "tender-sensitive",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.domain_id = domain_id
        self.semantica_state_root = (
            self.workspace_root / ".semantier" / "semantica" / domain_id
        )
        self.semantica_state_root.mkdir(parents=True, exist_ok=True)
        _ensure_vendored_semantica_importable()
        from semantica.change_management.managers import (
            OntologyVersionManager,
            TemporalVersionManager,
        )

        self.temporal_version_manager = TemporalVersionManager(
            str(self.semantica_state_root / "graph_versions.sqlite3")
        )
        self.ontology_version_manager = OntologyVersionManager(
            str(self.semantica_state_root / "ontology_versions.sqlite3")
        )
        self.repository = ContextGraphArtifactRepository(
            self.workspace_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
            domain_id=domain_id,
        )
        self.selection_store = MvlSelectionStore(self.workspace_root, workspace_id=workspace_id)

    def graph_ref_for(self, semantica_version_id: str) -> str:
        if not semantica_version_id:
            raise ValueError("semantica_version_id is required")
        return f"urn:semantier:mvl:graph:{self.workspace_id}:{semantica_version_id}"

    def ontology_ref_for(self, semantica_version_id: str) -> str:
        if not semantica_version_id:
            raise ValueError("semantica_version_id is required")
        return f"urn:semantier:mvl:ontology:{self.workspace_id}:{semantica_version_id}"

    def create_native_graph_version(
        self,
        *,
        graph_payload: dict[str, Any],
        version_label: str,
        author: str,
        description: str,
        experiment_id: str,
        actor_id: str | None = None,
        source_ref: str | None = None,
        source_hash: str | None = None,
        canonical_source_ir_ref: str | None = None,
        session_id: str | None = None,
        semantic_stage: str = "proposed",
        control_action: str | None = None,
        provenance_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.temporal_version_manager.create_snapshot(
            graph_payload,
            version_label=version_label,
            author=author,
            description=description,
            metadata={
                "organization_id": self.organization_id,
                "workspace_id": self.workspace_id,
                "domain_id": self.domain_id,
                "experiment_id": experiment_id,
                "source_ref": source_ref,
                "source_hash": source_hash,
                "canonical_source_ir_ref": canonical_source_ir_ref,
                "session_id": session_id,
                "actor_id": actor_id,
                "semantic_stage": semantic_stage,
                "control_action": control_action,
                "provenance_refs": provenance_refs or [],
            },
        )
        self.temporal_version_manager.tag_version(
            version_label,
            f"mvl:{experiment_id}:graph",
        )
        return snapshot

    def create_native_ontology_version(
        self,
        *,
        ontology_payload: dict[str, Any],
        version_label: str,
        author: str,
        description: str,
        experiment_id: str,
        actor_id: str | None = None,
        source_ref: str | None = None,
        source_hash: str | None = None,
        canonical_source_ir_ref: str | None = None,
        session_id: str | None = None,
        semantic_stage: str = "proposed",
        control_action: str | None = None,
        provenance_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.ontology_version_manager.create_snapshot(
            ontology_payload,
            version_label=version_label,
            author=author,
            description=description,
            metadata={
                "organization_id": self.organization_id,
                "workspace_id": self.workspace_id,
                "domain_id": self.domain_id,
                "experiment_id": experiment_id,
                "source_ref": source_ref,
                "source_hash": source_hash,
                "canonical_source_ir_ref": canonical_source_ir_ref,
                "session_id": session_id,
                "actor_id": actor_id,
                "semantic_stage": semantic_stage,
                "control_action": control_action,
                "provenance_refs": provenance_refs or [],
            },
        )
        self.ontology_version_manager.storage.save_tag(
            f"mvl:{experiment_id}:ontology",
            version_label,
        )
        return snapshot

    def select_graph_for_mvl(
        self,
        *,
        experiment_id: str,
        profile_ref: str,
        profile_version: str,
        graph_artifact_id: str,
        selected_by: str,
        ontology_semantica_version_id: str | None = None,
        ontology_checksum: str | None = None,
    ) -> MvlSelection:
        metadata = self.repository.get(graph_artifact_id)
        graph_ref = self.graph_ref_for(metadata.artifact_id)
        ontology_ref = (
            self.ontology_ref_for(ontology_semantica_version_id)
            if ontology_semantica_version_id
            else None
        )
        return self.selection_store.create_selection(
            experiment_id=experiment_id,
            profile_ref=profile_ref,
            profile_version=profile_version,
            graph_version_ref=graph_ref,
            ontology_version_ref=ontology_ref,
            graph_semantica_version_id=metadata.artifact_id,
            ontology_semantica_version_id=ontology_semantica_version_id,
            graph_tag=f"mvl:{experiment_id}:graph",
            ontology_tag=f"mvl:{experiment_id}:ontology" if ontology_semantica_version_id else None,
            graph_checksum=metadata.content_hash,
            ontology_checksum=ontology_checksum,
            selected_by=selected_by,
        )

    def resolve_graph_ref(self, graph_version_ref: str) -> ResolvedMvlGraphSelection:
        selection = self.selection_store.resolve_graph_ref(graph_version_ref)
        metadata = self.repository.get(selection.graph_semantica_version_id)
        if metadata.content_hash != selection.graph_checksum:
            raise ArtifactRepositoryError("selected graph checksum no longer matches immutable artifact")
        expected_ref = self.graph_ref_for(metadata.artifact_id)
        if expected_ref != graph_version_ref:
            raise ArtifactRepositoryError("selected graph ref does not match native artifact identity")
        return ResolvedMvlGraphSelection(selection=selection, graph_metadata=metadata)

    def active_selection_for_experiment(self, experiment_id: str) -> ResolvedMvlGraphSelection:
        selection = self.selection_store.active_for_experiment(experiment_id)
        return self.resolve_graph_ref(selection.graph_version_ref)
