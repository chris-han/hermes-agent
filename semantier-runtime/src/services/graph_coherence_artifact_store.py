"""Governed workspace persistence for immutable coherence analyses."""

from __future__ import annotations

from pathlib import Path

from contracts.graph_coherence_analysis import GraphCoherenceAnalysisV1
from eos.artifact_hashing import canonical_json


class GraphCoherenceArtifactConflict(ValueError):
    """An existing content identity resolves to different bytes."""


class GraphCoherenceArtifactStore:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        organization_id: str,
        workspace_id: str,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not organization_id or not organization_id.isascii():
            raise ValueError("organization_id must be an ASCII-stable identifier")
        if not workspace_id or not workspace_id.isascii():
            raise ValueError("workspace_id must be an ASCII-stable identifier")
        if root.name != workspace_id:
            raise ValueError("workspace_root must identify the governed workspace")
        self._root = root
        self._organization_id = organization_id
        self._workspace_id = workspace_id
        self._artifact_root = root / "artifacts" / "semantic-graph-coherence"

    def persist(self, analysis: GraphCoherenceAnalysisV1) -> dict[str, str]:
        if type(analysis) is not GraphCoherenceAnalysisV1:
            raise TypeError("canonical graph coherence analysis required")
        path = self._path(analysis.analysis_ref)
        payload = canonical_json(analysis.as_dict()).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as output:
                output.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise GraphCoherenceArtifactConflict(
                    "existing coherence analysis identity has conflicting bytes"
                )
        return {
            "analysis_ref": analysis.analysis_ref,
            "analysis_hash": analysis.analysis_hash,
            "artifact_path": path.relative_to(self._root).as_posix(),
            "organization_id": self._organization_id,
            "workspace_id": self._workspace_id,
        }

    def read(self, analysis_ref: str, analysis_hash: str) -> GraphCoherenceAnalysisV1:
        path = self._path(analysis_ref)
        if not path.is_file():
            raise ValueError("graph coherence analysis not found")
        try:
            analysis = GraphCoherenceAnalysisV1.model_validate_json(path.read_bytes())
        except ValueError as exc:
            raise GraphCoherenceArtifactConflict(
                "persisted coherence analysis failed integrity validation"
            ) from exc
        if analysis.analysis_hash != analysis_hash:
            raise GraphCoherenceArtifactConflict(
                "persisted coherence analysis hash does not match requested identity"
            )
        return analysis

    def _path(self, analysis_ref: str) -> Path:
        if (
            not isinstance(analysis_ref, str)
            or not analysis_ref.isascii()
            or not analysis_ref.startswith("gca_")
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in analysis_ref)
        ):
            raise ValueError("invalid graph coherence analysis reference")
        path = (self._artifact_root / f"{analysis_ref}.json").resolve()
        if self._artifact_root.resolve() not in path.parents:
            raise ValueError("coherence artifact path escaped workspace boundary")
        return path
