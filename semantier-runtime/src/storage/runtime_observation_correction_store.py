"""Sealed append-only storage for runtime-observation correction lineage."""

from __future__ import annotations

from storage.governed_file_io import exclusive_file_lock
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.runtime_observation_correction_lineage_v1 import (
    RuntimeObservationCorrectionLineageV1,
)
from eos.artifact_hashing import content_hash
from storage.runtime_semantic_hypothesis_artifact_store import _publish_committed_directory, _safe


_RUNTIME_OBSERVATION_CORRECTION_WRITE_CAPABILITY = object()
_RUN_ID = "runtime-observation-corrections"
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["runtime_observation_correction_lineage_manifest.v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    correction_ref: str
    correction_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor_hypothesis_ref: str
    predecessor_hypothesis_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    grounding_event_ref: str
    grounding_event_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    successor_hypothesis_ref: str
    successor_hypothesis_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_filename: Literal["runtime_observation_correction_lineage.v1.json"]

    @model_validator(mode="after")
    def validate_hash(self) -> "_Manifest":
        for field in (
            "organization_id", "workspace_id", "correction_ref",
            "predecessor_hypothesis_ref", "grounding_event_ref",
            "successor_hypothesis_ref",
        ):
            _safe(getattr(self, field), field)
        payload = self.model_dump(mode="json")
        payload.pop("manifest_hash")
        if self.manifest_hash != "sha256:" + content_hash(payload):
            raise ValueError("runtime observation correction lineage manifest hash mismatch")
        return self

    @classmethod
    def create(cls, **payload):
        return cls.model_validate(
            {**payload, "manifest_hash": "sha256:" + content_hash(payload)}
        )


class RuntimeObservationCorrectionStore:
    def __init__(
        self,
        runtime_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> None:
        self.organization_id = _safe(organization_id, "organization_id")
        self.workspace_id = _safe(workspace_id, "workspace_id")
        root = Path(runtime_root).resolve()
        workspace = (
            root
            if root.name == workspace_id and root.parent.name == "workspaces"
            else root / "workspaces" / workspace_id
        )
        self.root = (workspace / "runs" / _RUN_ID / "runtime-semantic").resolve()
        if not self.root.is_relative_to(workspace.resolve()):
            raise ValueError("runtime observation correction root escapes workspace")

    def _put(
        self,
        lineage: RuntimeObservationCorrectionLineageV1,
        *,
        capability: object,
    ) -> RuntimeObservationCorrectionLineageV1:
        if capability is not _RUNTIME_OBSERVATION_CORRECTION_WRITE_CAPABILITY:
            raise PermissionError("publication requires the runtime observation correction service")
        lineage = RuntimeObservationCorrectionLineageV1.model_validate(
            lineage.model_dump(mode="json", by_alias=True)
        )
        self._own(lineage.organization_id, lineage.workspace_id)
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f".{lineage.successor_hypothesis_hash[7:]}.lineage.lock"
        with exclusive_file_lock(lock_path):
            existing = self._find_by_successor(
                lineage.successor_hypothesis_ref,
                lineage.successor_hypothesis_hash,
                required=False,
            )
            if existing is not None and existing.correction_hash != lineage.correction_hash:
                raise ValueError("successor hypothesis already has different correction lineage")
            manifest = _Manifest.create(
                schema_version="runtime_observation_correction_lineage_manifest.v1",
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                correction_ref=lineage.correction_id,
                correction_hash=lineage.correction_hash,
                predecessor_hypothesis_ref=lineage.predecessor_hypothesis_ref,
                predecessor_hypothesis_hash=lineage.predecessor_hypothesis_hash,
                grounding_event_ref=lineage.grounding_event_ref,
                grounding_event_hash=lineage.grounding_event_hash,
                successor_hypothesis_ref=lineage.successor_hypothesis_ref,
                successor_hypothesis_hash=lineage.successor_hypothesis_hash,
                artifact_filename="runtime_observation_correction_lineage.v1.json",
            )
            _publish_committed_directory(
                self._directory(lineage.correction_hash),
                files={
                    "runtime_observation_correction_lineage.v1.json": lineage.model_dump(
                        mode="json", by_alias=True
                    ),
                    "runtime_observation_correction_lineage.manifest.v1.json": manifest.model_dump(mode="json"),
                },
                conflict_error="RUNTIME_OBSERVATION_CORRECTION_LINEAGE_CONFLICT",
            )
        return lineage

    def _find_by_successor(
        self,
        successor_ref: str,
        successor_hash: str,
        *,
        required: bool = True,
    ) -> RuntimeObservationCorrectionLineageV1 | None:
        _safe(successor_ref, "successor_ref")
        if _SHA.fullmatch(successor_hash) is None:
            raise ValueError("successor hash must be an exact lowercase sha256 ref")
        matches = []
        for path in self.root.glob(
            "*/runtime_observation_correction_lineage.manifest.v1.json"
        ):
            manifest = _Manifest.model_validate_json(path.read_text(encoding="utf-8"))
            if (
                manifest.successor_hypothesis_ref,
                manifest.successor_hypothesis_hash,
            ) == (successor_ref, successor_hash):
                matches.append(
                    self._get(manifest.correction_ref, manifest.correction_hash)
                )
        if len(matches) != 1:
            if not matches and not required:
                return None
            raise ValueError("corrected successor requires exactly one correction lineage")
        return matches[0]

    def _get(
        self, correction_ref: str, correction_hash: str
    ) -> RuntimeObservationCorrectionLineageV1:
        _safe(correction_ref, "correction_ref")
        directory = self._directory(correction_hash)
        manifest = _Manifest.model_validate_json(
            (
                directory
                / "runtime_observation_correction_lineage.manifest.v1.json"
            ).read_text(encoding="utf-8")
        )
        if (
            directory != self._directory(manifest.correction_hash)
            or (manifest.correction_ref, manifest.correction_hash)
            != (correction_ref, correction_hash)
        ):
            raise ValueError("runtime observation correction manifest identity mismatch")
        self._own(manifest.organization_id, manifest.workspace_id)
        lineage = RuntimeObservationCorrectionLineageV1.model_validate_json(
            (directory / manifest.artifact_filename).read_text(encoding="utf-8")
        )
        if (manifest.organization_id, manifest.workspace_id) != (
            lineage.organization_id,
            lineage.workspace_id,
        ):
            raise ValueError("runtime observation correction manifest scope mismatch")
        if (
            lineage.correction_id,
            lineage.correction_hash,
            lineage.predecessor_hypothesis_ref,
            lineage.predecessor_hypothesis_hash,
            lineage.grounding_event_ref,
            lineage.grounding_event_hash,
            lineage.successor_hypothesis_ref,
            lineage.successor_hypothesis_hash,
        ) != (
            manifest.correction_ref,
            manifest.correction_hash,
            manifest.predecessor_hypothesis_ref,
            manifest.predecessor_hypothesis_hash,
            manifest.grounding_event_ref,
            manifest.grounding_event_hash,
            manifest.successor_hypothesis_ref,
            manifest.successor_hypothesis_hash,
        ):
            raise ValueError("runtime observation correction lineage differs from manifest")
        self._own(lineage.organization_id, lineage.workspace_id)
        return lineage

    def _directory(self, digest: str) -> Path:
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            raise ValueError("correction hash must be an exact lowercase sha256 ref")
        target = (self.root / digest[7:]).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("correction lineage path escapes governed runs")
        return target

    def _own(self, organization_id: str, workspace_id: str) -> None:
        if (organization_id, workspace_id) != (self.organization_id, self.workspace_id):
            raise ValueError("runtime observation correction ownership mismatch")


__all__ = ["RuntimeObservationCorrectionStore"]
