"""Sealed append-only persistence for schema-correction continuations."""

from __future__ import annotations

from pathlib import Path
import re

from contracts.schema_correction_continuation import SchemaCorrectionContinuationV1
from storage.atomic_json_artifact import publish_immutable_json


_SCHEMA_CONTINUATION_PUBLICATION_AUTHORITY = object()
_REF = re.compile(r"^scc_[0-9a-f]{24}$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class SchemaCorrectionContinuationStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id or workspace_id in {".", ".."} or "/" in workspace_id or "\\" in workspace_id:
            raise ValueError("workspace_id must be one safe path segment")
        workspaces = Path(workspace_root).resolve() / "workspaces"
        root = (workspaces / workspace_id / "artifacts/schema-discovery/schema-correction-continuations").resolve()
        if not root.is_relative_to(workspaces.resolve()):
            raise ValueError("workspace_id escapes governed workspace root")
        self.workspace_id = workspace_id
        self.root = root

    def publish(self, continuation: SchemaCorrectionContinuationV1, *, _authority: object) -> SchemaCorrectionContinuationV1:
        if _authority is not _SCHEMA_CONTINUATION_PUBLICATION_AUTHORITY:
            raise PermissionError("schema correction continuation publication is sealed")
        if type(continuation) is not SchemaCorrectionContinuationV1:
            raise TypeError("continuation must use the exact governed contract")
        if continuation.workspace_id != self.workspace_id:
            raise ValueError("schema correction continuation workspace mismatch")
        return SchemaCorrectionContinuationV1.model_validate(publish_immutable_json(
            self.root / f"{continuation.continuation_ref}.json",
            continuation.model_dump(mode="json"),
            conflict_error="SCHEMA_CORRECTION_CONTINUATION_CONFLICT",
        ))

    def get(self, continuation_ref: str, *, continuation_hash: str) -> SchemaCorrectionContinuationV1:
        if _REF.fullmatch(continuation_ref) is None or _HASH.fullmatch(continuation_hash) is None:
            raise ValueError("schema correction continuation pins are invalid")
        path = self.root / f"{continuation_ref}.json"
        if not path.is_file():
            raise ValueError("schema correction continuation is missing")
        continuation = SchemaCorrectionContinuationV1.model_validate_json(path.read_text(encoding="utf-8"))
        if (continuation.continuation_ref != continuation_ref
                or continuation.continuation_hash != continuation_hash
                or continuation.workspace_id != self.workspace_id):
            raise ValueError("schema correction continuation binding mismatch")
        return continuation


__all__ = ["SchemaCorrectionContinuationStore", "_SCHEMA_CONTINUATION_PUBLICATION_AUTHORITY"]
