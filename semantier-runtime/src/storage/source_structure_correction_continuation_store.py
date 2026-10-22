"""Sealed append-only source-structure correction continuation store."""

from __future__ import annotations

from pathlib import Path
import re

from contracts.source_structure_correction_continuation import (
    SourceStructureCorrectionContinuationV1,
)
from storage.atomic_json_artifact import publish_immutable_json


_SOURCE_STRUCTURE_CONTINUATION_AUTHORITY = object()
_REF = re.compile(r"^sscc_[0-9a-f]{24}$")
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourceStructureCorrectionContinuationStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if (
            not workspace_id
            or workspace_id in {".", ".."}
            or "/" in workspace_id
            or "\\" in workspace_id
        ):
            raise ValueError("workspace_id must be one safe path segment")
        workspaces = Path(workspace_root).resolve() / "workspaces"
        self.workspace_id = workspace_id
        self.root = (
            workspaces
            / workspace_id
            / "artifacts/schema-discovery/source-structure-correction-continuations"
        ).resolve()
        if not self.root.is_relative_to(workspaces.resolve()):
            raise ValueError("workspace_id escapes governed workspace root")

    def publish(
        self,
        continuation: SourceStructureCorrectionContinuationV1,
        *,
        _authority: object,
    ) -> SourceStructureCorrectionContinuationV1:
        if _authority is not _SOURCE_STRUCTURE_CONTINUATION_AUTHORITY:
            raise PermissionError("source-structure continuation publication is sealed")
        if type(continuation) is not SourceStructureCorrectionContinuationV1:
            raise TypeError("source-structure continuation must be exact")
        if continuation.workspace_id != self.workspace_id:
            raise ValueError("source-structure continuation workspace mismatch")
        return SourceStructureCorrectionContinuationV1.model_validate(
            publish_immutable_json(
                self.root / f"{continuation.continuation_ref}.json",
                continuation.model_dump(mode="json"),
                conflict_error="SOURCE_STRUCTURE_CORRECTION_CONTINUATION_CONFLICT",
            )
        )

    def get(
        self, continuation_ref: str, *, continuation_hash: str
    ) -> SourceStructureCorrectionContinuationV1:
        if _REF.fullmatch(continuation_ref) is None or _SHA.fullmatch(continuation_hash) is None:
            raise ValueError("source-structure continuation pins are invalid")
        path = self.root / f"{continuation_ref}.json"
        if not path.is_file():
            raise ValueError("source-structure continuation is missing")
        continuation = SourceStructureCorrectionContinuationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            continuation.continuation_ref != continuation_ref
            or continuation.continuation_hash != continuation_hash
            or continuation.workspace_id != self.workspace_id
        ):
            raise ValueError("source-structure continuation binding mismatch")
        return continuation


__all__ = [
    "SourceStructureCorrectionContinuationStore",
    "_SOURCE_STRUCTURE_CONTINUATION_AUTHORITY",
]
