"""Runtime detection snapshot binding verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.semantica_mvl_control_adapter import SemanticaMvlControlAdapter
from storage.canonical_list_profile_store import CanonicalListProfileStore
from storage.runtime_detection_snapshot_store import RuntimeDetectionSnapshotStore


class RuntimeDetectionSnapshotService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        domain_id: str = "canonical-list",
    ) -> None:
        self.control = SemanticaMvlControlAdapter(
            workspace_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
            domain_id=domain_id,
        )
        self.profile_store = CanonicalListProfileStore(workspace_root, workspace_id=workspace_id)
        self.snapshot_store = RuntimeDetectionSnapshotStore(workspace_root, workspace_id=workspace_id)

    def create_snapshot(
        self,
        *,
        run_id: str,
        experiment_id: str,
        input_source_ref: str,
        input_source_hash: str,
        findings: list[dict[str, Any]],
        matcher_version: str = "canonical-list-matcher@0.1.0",
    ) -> dict[str, Any]:
        if not findings:
            raise ValueError("runtime detection snapshot requires at least one finding")
        first = findings[0]
        graph_ref = first["graph_version_ref"]
        resolved = self.control.resolve_graph_ref(graph_ref)
        profile = self.profile_store.get_profile_version(
            profile_ref=first["profile_ref"],
            version=first["profile_version"],
        )
        snapshot = self.snapshot_store.create_snapshot(
            run_id=run_id,
            experiment_id=experiment_id,
            input_source_ref=input_source_ref,
            input_source_hash=input_source_hash,
            profile_ref=profile.profile_ref,
            profile_version=profile.version,
            graph_version_ref=graph_ref,
            graph_checksum=resolved.selection.graph_checksum,
            ontology_version_ref=first.get("ontology_version_ref"),
            ontology_checksum=resolved.selection.ontology_checksum,
            skill_ref=first["skill_ref"],
            semantica_commit=first["semantica_commit"],
            matcher_version=matcher_version,
            findings=findings,
        )
        return snapshot.as_dict()

    def verify_historical_binding(self, run_id: str) -> dict[str, Any]:
        snapshot = self.snapshot_store.get_snapshot(run_id)
        profile = self.profile_store.get_profile_version(
            profile_ref=snapshot.profile_ref,
            version=snapshot.profile_version,
        )
        resolved = self.control.resolve_graph_ref(snapshot.graph_version_ref)
        if resolved.selection.graph_checksum != snapshot.graph_checksum:
            raise ValueError("snapshot graph checksum no longer matches selected graph")
        if snapshot.ontology_checksum != resolved.selection.ontology_checksum:
            raise ValueError("snapshot ontology checksum no longer matches selection")
        return {
            "snapshot": snapshot.as_dict(),
            "profile_content_hash": profile.content_hash,
            "graph_artifact_id": resolved.graph_metadata.artifact_id,
            "graph_checksum": resolved.graph_metadata.content_hash,
            "verified": True,
        }

