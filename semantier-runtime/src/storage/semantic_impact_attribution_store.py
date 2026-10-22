"""Append-only workspace store for semantic-impact attribution companions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from uuid import uuid4

from contracts.semantic_impact_attribution import SemanticImpactAttributionV1


_ATTRIBUTION_PUBLICATION_AUTHORITY = object()


def _workspace_artifact_root(
    workspace_root: str | Path, *, workspace_id: str
) -> Path:
    if (
        not isinstance(workspace_id, str)
        or not workspace_id
        or workspace_id in {".", ".."}
        or "/" in workspace_id
        or "\\" in workspace_id
    ):
        raise ValueError("workspace_id must be one safe path segment")
    workspaces = Path(workspace_root).resolve() / "workspaces"
    resolved = (workspaces / workspace_id / "artifacts").resolve()
    if not resolved.is_relative_to(workspaces.resolve()):
        raise ValueError("workspace_id escapes governed workspace root")
    return resolved


class SemanticImpactAttributionStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.root = (
            _workspace_artifact_root(workspace_root, workspace_id=workspace_id)
            / "semantic-impact-attributions"
        )

    def path_for(self, attribution_id: str) -> Path:
        if re.fullmatch(r"sia_[0-9a-f]{24}", attribution_id) is None:
            raise ValueError("invalid semantic impact attribution id")
        return self.root / f"{attribution_id}.json"

    def _publish(
        self,
        attribution: SemanticImpactAttributionV1,
        *,
        _authority: object,
    ) -> SemanticImpactAttributionV1:
        if _authority is not _ATTRIBUTION_PUBLICATION_AUTHORITY:
            raise PermissionError("semantic impact attribution publication is sealed")
        if type(attribution) is not SemanticImpactAttributionV1:
            raise TypeError("attribution must be an exact SemanticImpactAttributionV1")
        if attribution.workspace_id != self.workspace_id:
            raise ValueError("semantic impact attribution workspace mismatch")
        target = self.path_for(attribution.attribution_id)
        encoded = json.dumps(
            attribution.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_text(encoding="utf-8") == encoded:
                return attribution
            raise ValueError("SEMANTIC_IMPACT_ATTRIBUTION_CONFLICT")
        temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_text(encoding="utf-8") == encoded:
                    return attribution
                raise ValueError("SEMANTIC_IMPACT_ATTRIBUTION_CONFLICT") from None
            else:
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return attribution

    def get(
        self, attribution_id: str, *, attribution_hash: str | None = None
    ) -> SemanticImpactAttributionV1:
        target = self.path_for(attribution_id)
        if not target.is_file():
            raise ValueError(f"semantic impact attribution not found: {attribution_id}")
        restored = SemanticImpactAttributionV1.model_validate_json(
            target.read_text(encoding="utf-8")
        )
        if restored.attribution_id != attribution_id:
            raise ValueError("semantic impact attribution path/id mismatch")
        if restored.workspace_id != self.workspace_id:
            raise ValueError("semantic impact attribution workspace mismatch")
        if attribution_hash is not None and restored.attribution_hash != attribution_hash:
            raise ValueError("semantic impact attribution hash mismatch")
        return restored


__all__ = ["SemanticImpactAttributionStore"]
