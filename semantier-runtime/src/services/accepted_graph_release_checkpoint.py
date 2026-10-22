"""Exact authoritative checkpoint verification for accepted graph releases."""

from __future__ import annotations

from typing import Any

from contracts.accepted_graph_release_lineage import (
    AcceptedGraphReleaseLineage,
    AcceptedGraphReleaseLineageV2,
)
from contracts.knowledge_feature_pack import content_hash
from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT
from semantier.semantica_context_graph.artifact_repository import (
    ArtifactConflict,
    ArtifactRepositoryError,
    ContextGraphArtifactRepository,
)
from semantier.semantica_context_graph.native_builder import GRAPH_ID


def accepted_release_source_manifest_hash(
    release: dict[str, Any],
    *,
    release_lineage: AcceptedGraphReleaseLineage | AcceptedGraphReleaseLineageV2 | None,
) -> str:
    """Hash the exact release inputs pinned by an authoritative checkpoint."""
    payload = {
        "candidate_graph_id": release.get("candidate_graph_id"),
        "accepted_graph_release_hash": release.get("release_hash"),
    }
    if release_lineage is not None:
        payload["release_lineage_hash"] = release_lineage.lineage_hash
    return content_hash(payload)


def verify_accepted_release_checkpoint(
    repository: ContextGraphArtifactRepository,
    *,
    graph: Any,
    release: dict[str, Any],
    lineage_id: str,
    release_lineage: AcceptedGraphReleaseLineage | AcceptedGraphReleaseLineageV2 | None,
) -> dict[str, Any]:
    """Reload and verify the immutable graph plus every authoritative metadata pin."""
    graph_version = str(release.get("graph_version") or "")
    metadata = repository.get(graph_version)
    expected = {
        "artifact_id": graph_version,
        "organization_id": repository.organization_id,
        "workspace_id": repository.workspace_id,
        "domain_id": repository.domain_id,
        "graph_id": GRAPH_ID,
        "graph_version": graph_version,
        "artifact_kind": "authoritative",
        "authority_state": "authoritative",
        "run_mode": "authoritative",
        "parent_artifact_id": None,
        "lineage_id": lineage_id,
        "lineage_root_artifact_id": lineage_id,
        "semantica_commit": (
            release_lineage.compiler_semantica_commit
            if release_lineage is not None
            else SEMANTICA_COMMIT
        ),
        "source_manifest_hash": accepted_release_source_manifest_hash(
            release,
            release_lineage=release_lineage,
        ),
        "grounding_manifest_hash": str(release.get("release_hash") or ""),
        "evaluation_manifest_hash": None,
        "governance_ref": f"AcceptedGraphRelease:{graph_version}",
        "lifecycle_state": "active",
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if getattr(metadata, field) != expected_value
    ]
    if mismatches:
        raise ArtifactConflict(
            "accepted graph release checkpoint exact pin mismatch: "
            + ", ".join(sorted(mismatches))
        )
    try:
        stored_graph = repository.load_graph(graph_version)
    except ArtifactRepositoryError as exc:
        raise ArtifactConflict(
            "accepted graph release checkpoint content verification failed"
        ) from exc
    if stored_graph.to_dict() != graph.to_dict():
        raise ArtifactConflict(
            "accepted graph release checkpoint content does not match release"
        )
    return metadata.as_dict()


__all__ = [
    "accepted_release_source_manifest_hash",
    "verify_accepted_release_checkpoint",
]
