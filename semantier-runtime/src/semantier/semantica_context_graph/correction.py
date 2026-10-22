"""Human-authorized, non-toolized Phase1-001 grounding correction."""

from __future__ import annotations

from typing import Any, Mapping

from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT

from .artifact_repository import ArtifactConflict, ContextGraphArtifactRepository


class GroundingCorrectionRejected(ValueError):
    """Raised when a correction is not backed by exact grounding evidence."""


def apply_grounding_correction(
    repository: ContextGraphArtifactRepository,
    *,
    source_artifact_id: str,
    source_graph_version: str,
    source_content_hash: str,
    corrected_node_id: str,
    grounding_manifest_hash: str,
    human_authorization: Mapping[str, Any],
    created_by_user_id: str,
) -> Any:
    """Apply exactly one evidence-backed correction during build/test only.

    This function intentionally has no Hermes registration surface and accepts
    no browser request.  The caller must provide an exact source pin and an
    independently reviewed grounding authorization record.
    """
    source = repository.get(source_artifact_id)
    if source.artifact_kind != "candidate" or source.graph_version != source_graph_version:
        raise GroundingCorrectionRejected("correction source must be the candidate artifact")
    if source.content_hash != source_content_hash:
        raise GroundingCorrectionRejected("correction source hash is stale")
    if not grounding_manifest_hash.startswith("sha256:"):
        raise GroundingCorrectionRejected("grounding manifest must be hash pinned")
    if human_authorization.get("manifest_role") != "grounding":
        raise GroundingCorrectionRejected("human authorization must reference grounding evidence")
    if human_authorization.get("decision") != "approved":
        raise GroundingCorrectionRejected("human authorization is required")
    if not str(human_authorization.get("reviewer_id") or "").strip():
        raise GroundingCorrectionRejected("reviewer identity is required")
    if not str(human_authorization.get("hypothesis") or "").strip():
        raise GroundingCorrectionRejected("correction hypothesis is required")

    graph = repository.load_graph(source_artifact_id)
    node = graph.nodes.get(corrected_node_id)
    if node is None:
        raise GroundingCorrectionRejected("corrected node is absent from source artifact")
    properties = dict(getattr(node, "properties", {}) or {})
    properties["phase1_grounding_correction"] = {
        "manifest_hash": grounding_manifest_hash,
        "reviewer_id": str(human_authorization["reviewer_id"]),
        "hypothesis": str(human_authorization["hypothesis"]),
    }
    node.properties = properties
    try:
        return repository.create_checkpoint(
            graph,
            lineage_id=source.lineage_id or source.lineage_root_artifact_id,
            graph_id=source.graph_id,
            graph_version="v0002-candidate-r001",
            artifact_kind="candidate",
            run_mode="build",
            semantica_commit=SEMANTICA_COMMIT,
            source_manifest_hash=source.source_manifest_hash,
            grounding_manifest_hash=grounding_manifest_hash,
            created_by_user_id=created_by_user_id,
            parent_artifact_id=source.artifact_id,
            artifact_id="phase1-001-v1-candidate",
        )
    except ArtifactConflict:
        raise
