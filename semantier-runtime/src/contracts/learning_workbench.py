"""Non-kernel exact learning artifacts; these records grant no authority."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime
import hashlib
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import SHA256_PATTERN, _ContentAddressed, _Contract


class SkillFileV1(_Contract):
    relative_path: str = Field(min_length=1)
    content_base64: str
    content_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("relative_path")
    @classmethod
    def confined_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (path.is_absolute() or "\\" in value or "\x00" in value
                or any(part in ("", ".", "..") for part in value.split("/"))
                or str(path) != value):
            raise ValueError("skill file must use a confined canonical relative path")
        return value

    def content_bytes(self) -> bytes:
        try:
            content = base64.b64decode(self.content_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("skill file content must be canonical base64") from exc
        if base64.b64encode(content).decode("ascii") != self.content_base64:
            raise ValueError("skill file content must be canonical base64")
        return content

    @model_validator(mode="after")
    def exact_bytes(self):
        if self.content_hash != "sha256:" + hashlib.sha256(self.content_bytes()).hexdigest():
            raise ValueError("skill file hash does not bind exact bytes")
        return self


class SkillArtifactVersionV1(_ContentAddressed):
    ARTIFACT_KIND = "skill_artifact_version"
    schema_version: Literal["skill_artifact_version.v1"] = "skill_artifact_version.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    skill_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    version: int = Field(ge=1)
    provenance: Literal["legacy", "managed"]
    predecessor: ArtifactPinV1 | None = None
    files: tuple[SkillFileV1, ...]
    recorded_at: datetime

    @property
    def package_state(self):
        return "present" if self.files else "absent"

    @field_validator("files", mode="before")
    @classmethod
    def canonical_files(cls, value):
        files = tuple(file if isinstance(file, SkillFileV1) else SkillFileV1.model_validate(file)
                      for file in value)
        paths = [file.relative_path for file in files]
        if (paths and "SKILL.md" not in paths) or len(paths) != len(set(paths)):
            raise ValueError("skill version requires SKILL.md and unique file paths")
        return tuple(sorted(files, key=lambda file: file.relative_path))

    @model_validator(mode="after")
    def predecessor_meaning(self):
        if self.version == 1 and self.predecessor is not None:
            raise ValueError("initial skill version cannot have a predecessor")
        if self.version > 1 and (self.predecessor is None or self.provenance != "managed"):
            raise ValueError("successor must be managed and bind its exact predecessor")
        if self.predecessor is not None and self.predecessor.artifact_kind != self.ARTIFACT_KIND:
            raise ValueError("skill predecessor must pin a skill artifact version")
        return self


LEARNING_ARTIFACT_MODELS = {"skill_artifact_version.v1": SkillArtifactVersionV1}


class ProcedureInspectionV1(_Contract):
    """Inspection of exact retained bytes, without inferred runtime receipts."""
    object_kind: Literal["procedure"] = "procedure"
    object_pin: ArtifactPinV1
    skill_id: str
    version: int
    files: tuple[SkillFileV1, ...]
    organization_id: str
    workspace_id: str
    leading_answer: str
    decisive_uncertainty: str
    qualification: Literal["unknown"] = "unknown"
    activation: Literal["unknown"] = "unknown"
    runtime_availability: Literal["unknown"] = "unknown"
    applicability: Literal["unknown"] = "unknown"
    recorded_use: Literal["unknown"] = "unknown"
    causal_attribution: Literal["unknown"] = "unknown"
    available_actions: tuple[Literal["inspect_files"], ...] = ("inspect_files",)
