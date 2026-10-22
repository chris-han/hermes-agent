from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from eos.artifact_hashing import canonical_json
from contracts.tender_sensitive_term_detection_modeling_core import GovernedArtifactDTO, RuntimeContextDTO


class GovernedDocumentService(Protocol):
    def extract_live(self, document_ref: str, context: RuntimeContextDTO) -> GovernedArtifactDTO: ...

    def extract_replay(
        self,
        normalized_artifact_ref: str,
        pins: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]: ...

    def resolve_governed_artifact(
        self,
        ref: str,
        context: RuntimeContextDTO,
    ) -> GovernedArtifactDTO: ...


def load_verified_normalized_document_artifact(
    *,
    normalized_document_artifact_ref: str,
    hermes_home: str | Path | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    artifact_path = Path(normalized_document_artifact_ref)
    if not artifact_path.is_absolute():
        effective_hermes_home = (
            str(hermes_home or "").strip()
            or os.environ.get("HERMES_HOME")
            or os.environ.get("SESSION_HERMES_HOME")
            or ""
        )
        effective_session_id = (
            str(session_id or "").strip()
            or os.environ.get("HERMES_SESSION_ID")
            or os.environ.get("SESSION_ID")
            or ""
        )
        if not effective_hermes_home or not effective_session_id:
            raise RuntimeError("SESSION_BINDING_REQUIRED")
        from runtime_paths import workspace_session_runs_root_from_hermes_home

        session_root = workspace_session_runs_root_from_hermes_home(
            Path(effective_hermes_home), effective_session_id
        ).parent
        artifact_path = session_root / normalized_document_artifact_ref
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise RuntimeError("INVALID_CANONICAL_DOCUMENT_ARTIFACT")
    expected_hash = artifact.get("artifact_hash")
    payload = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    actual_hash = "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if expected_hash != actual_hash:
        raise RuntimeError("NORMALIZED_DOCUMENT_ARTIFACT_HASH_MISMATCH")
    return artifact
