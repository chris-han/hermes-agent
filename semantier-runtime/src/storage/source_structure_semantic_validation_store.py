"""Sealed append-only store for neutral source-structure validation."""

from __future__ import annotations

from pathlib import Path
import re

from contracts.source_structure_semantic_validation import (
    SourceStructureSemanticValidationV1,
)
from storage.atomic_json_artifact import publish_immutable_json


_SOURCE_STRUCTURE_VALIDATION_AUTHORITY = object()
_REF = re.compile(r"^ssv_[0-9a-f]{24}$")
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourceStructureSemanticValidationStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if (
            not isinstance(workspace_id, str)
            or not workspace_id
            or workspace_id in {".", ".."}
            or "/" in workspace_id
            or "\\" in workspace_id
        ):
            raise ValueError("workspace_id must be one safe path segment")
        workspaces = Path(workspace_root).resolve() / "workspaces"
        self.workspace_id = workspace_id
        self.root = (
            workspaces / workspace_id
            / "artifacts/schema-discovery/source-structure-validations"
        ).resolve()
        if not self.root.is_relative_to(workspaces.resolve()):
            raise ValueError("workspace_id escapes governed workspace root")

    def _path(self, validation_ref: str) -> Path:
        if _REF.fullmatch(validation_ref) is None:
            raise ValueError("validation_ref must be exact")
        return self.root / f"{validation_ref}.json"

    def publish(
        self,
        validation: SourceStructureSemanticValidationV1,
        *,
        _authority: object,
    ) -> SourceStructureSemanticValidationV1:
        if _authority is not _SOURCE_STRUCTURE_VALIDATION_AUTHORITY:
            raise PermissionError("source-structure validation publication is sealed")
        if type(validation) is not SourceStructureSemanticValidationV1:
            raise TypeError("validation must be exact")
        if validation.workspace_id != self.workspace_id:
            raise ValueError("source-structure validation workspace mismatch")
        return SourceStructureSemanticValidationV1.model_validate(
            publish_immutable_json(
                self._path(validation.validation_ref),
                validation.model_dump(mode="json"),
                conflict_error="SOURCE_STRUCTURE_SEMANTIC_VALIDATION_CONFLICT",
            )
        )

    def get(
        self, validation_ref: str, *, validation_hash: str
    ) -> SourceStructureSemanticValidationV1:
        if _SHA.fullmatch(validation_hash) is None:
            raise ValueError("validation_hash must be exact")
        path = self._path(validation_ref)
        if not path.is_file():
            raise ValueError("source-structure semantic validation is missing")
        validation = SourceStructureSemanticValidationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            validation.validation_ref != validation_ref
            or validation.validation_hash != validation_hash
            or validation.workspace_id != self.workspace_id
        ):
            raise ValueError("source-structure validation path binding mismatch")
        return validation


__all__ = [
    "SourceStructureSemanticValidationStore",
    "_SOURCE_STRUCTURE_VALIDATION_AUTHORITY",
]
