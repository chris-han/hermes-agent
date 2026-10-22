"""Sealed append-only persistence for governed finding corrections."""

from __future__ import annotations

from storage.governed_file_io import exclusive_file_lock
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.governed_frame_finding_correction_v1 import GovernedFrameFindingCorrectionV1
from eos.artifact_hashing import content_hash
from storage.runtime_semantic_hypothesis_artifact_store import (
    _publish_committed_directory,
    _safe,
)


_FINDING_CORRECTION_WRITE_CAPABILITY = object()
_RUN_ID = "finding-corrections"
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


class _CorrectionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_frame_finding_correction_manifest.v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    run_id: Literal["finding-corrections"]
    correction_ref: str
    correction_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    predecessor_finding_ref: str
    predecessor_finding_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    grounding_event_ref: str
    grounding_event_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_filename: Literal["governed_frame_finding_correction.v1.json"]

    @model_validator(mode="after")
    def validate_manifest(self) -> "_CorrectionManifest":
        for field in (
            "organization_id", "workspace_id", "run_id", "correction_ref",
            "predecessor_finding_ref", "grounding_event_ref",
        ):
            _safe(getattr(self, field), field)
        payload = self.model_dump(mode="json")
        payload.pop("manifest_hash")
        if self.manifest_hash != "sha256:" + content_hash(payload):
            raise ValueError("governed finding correction manifest hash mismatch")
        return self

    @classmethod
    def create(cls, **payload):
        return cls.model_validate(
            {**payload, "manifest_hash": "sha256:" + content_hash(payload)}
        )


class GovernedFrameFindingCorrectionStore:
    def __init__(
        self, runtime_root: str | Path, *, organization_id: str, workspace_id: str
    ) -> None:
        self.organization_id = _safe(organization_id, "organization_id")
        self.workspace_id = _safe(workspace_id, "workspace_id")
        root = Path(runtime_root).resolve()
        workspace = (
            root
            if root.name == workspace_id and root.parent.name == "workspaces"
            else root / "workspaces" / workspace_id
        ).resolve()
        self.root = (workspace / "runs").resolve()
        if not self.root.is_relative_to(workspace):
            raise ValueError("correction artifact root escapes workspace")

    def _put(
        self, correction: GovernedFrameFindingCorrectionV1, *, capability: object
    ) -> GovernedFrameFindingCorrectionV1:
        if capability is not _FINDING_CORRECTION_WRITE_CAPABILITY:
            raise PermissionError("publication requires the governed correction service")
        correction = GovernedFrameFindingCorrectionV1.model_validate(
            correction.model_dump(mode="json", by_alias=True)
        )
        self._own(correction.organization_id, correction.workspace_id)
        filename = "governed_frame_finding_correction.v1.json"
        manifest = _CorrectionManifest.create(
            schema_version="governed_frame_finding_correction_manifest.v1",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            run_id=_RUN_ID,
            correction_ref=correction.correction_id,
            correction_hash=correction.correction_hash,
            predecessor_finding_ref=correction.predecessor_finding_ref,
            predecessor_finding_hash=correction.predecessor_finding_hash,
            grounding_event_ref=correction.grounding_event_ref,
            grounding_event_hash=correction.grounding_event_hash,
            artifact_filename=filename,
        )
        lock_root = self.root / _RUN_ID / "runtime-semantic"
        lock_root.mkdir(parents=True, exist_ok=True)
        predecessor_lock = lock_root / f".{correction.predecessor_finding_hash[7:]}.terminal.lock"
        with exclusive_file_lock(predecessor_lock):
            for path in lock_root.glob(
                "*/governed_frame_finding_correction.manifest.v1.json"
            ):
                existing = _CorrectionManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if (
                    existing.predecessor_finding_ref
                    == correction.predecessor_finding_ref
                    and existing.correction_hash != correction.correction_hash
                ):
                    raise ValueError(
                        "predecessor finding already has a terminal correction"
                    )
            _publish_committed_directory(
                self._directory(correction.correction_hash),
                files={
                    filename: correction.model_dump(mode="json", by_alias=True),
                    "governed_frame_finding_correction.manifest.v1.json": manifest.model_dump(mode="json"),
                },
                conflict_error="GOVERNED_FRAME_FINDING_CORRECTION_CONFLICT",
            )
        return correction

    def _get(
        self, *, correction_ref: str, correction_hash: str
    ) -> GovernedFrameFindingCorrectionV1:
        _safe(correction_ref, "correction_ref")
        directory = self._directory(correction_hash)
        manifest = _CorrectionManifest.model_validate_json(
            (directory / "governed_frame_finding_correction.manifest.v1.json").read_text(
                encoding="utf-8"
            )
        )
        self._own(manifest.organization_id, manifest.workspace_id)
        if directory != self._directory(manifest.correction_hash) or (
            manifest.correction_ref,
            manifest.correction_hash,
        ) != (correction_ref, correction_hash):
            raise ValueError("governed finding correction manifest identity mismatch")
        correction = GovernedFrameFindingCorrectionV1.model_validate_json(
            (directory / manifest.artifact_filename).read_text(encoding="utf-8")
        )
        if (
            correction.correction_id,
            correction.correction_hash,
            correction.predecessor_finding_ref,
            correction.predecessor_finding_hash,
            correction.grounding_event_ref,
            correction.grounding_event_hash,
        ) != (
            manifest.correction_ref,
            manifest.correction_hash,
            manifest.predecessor_finding_ref,
            manifest.predecessor_finding_hash,
            manifest.grounding_event_ref,
            manifest.grounding_event_hash,
        ):
            raise ValueError("governed finding correction differs from manifest")
        return correction

    def _directory(self, digest: str) -> Path:
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            raise ValueError("correction hash must be an exact lowercase sha256 ref")
        target = (self.root / _RUN_ID / "runtime-semantic" / digest[7:]).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("correction artifact path escapes governed runs")
        return target

    def _own(self, organization_id: str, workspace_id: str) -> None:
        if organization_id != self.organization_id:
            raise ValueError("correction artifact organization ownership mismatch")
        if workspace_id != self.workspace_id:
            raise ValueError("correction artifact workspace ownership mismatch")


__all__ = ["GovernedFrameFindingCorrectionStore"]
