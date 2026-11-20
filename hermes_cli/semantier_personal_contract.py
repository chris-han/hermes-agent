"""Personal package v1 wire DTOs for the native consumer.

Runtime remains the admission and authorization owner. These bounded copies of
its versioned wire schemas validate downloaded bytes without importing Runtime
into the native payload. Cross-owner conformance tests pin their serialization.
"""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import PurePosixPath
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()



class ArtifactPinV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_kind: str = Field(min_length=1)
    artifact_ref: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_datetimes_are_aware_utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class _ContentAddressed(_Contract):
    """One identity rule shared by all five kernel semantic objects."""

    ARTIFACT_KIND: ClassVar[str]
    artifact_ref: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=SHA256_PATTERN)

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "artifact_ref": "pending",
                "artifact_hash": "sha256:" + "0" * 64,
                **values,
            },
            context={"kernel_create_draft": True},
        )
        normalized = draft.model_dump(
            mode="python", exclude={"artifact_ref", "artifact_hash"}
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        return cls.model_validate(
            {
                "artifact_ref": f"{cls.ARTIFACT_KIND}:{digest}",
                "artifact_hash": f"sha256:{digest}",
                **normalized,
            }
        )

    @model_validator(mode="after")
    def identity_binds_exact_content(self, info: ValidationInfo):
        if info.context and info.context.get("kernel_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        if (
            self.artifact_ref != f"{type(self).ARTIFACT_KIND}:{digest}"
            or self.artifact_hash != f"sha256:{digest}"
        ):
            raise ValueError("kernel artifact ref/hash does not bind exact content")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind=type(self).ARTIFACT_KIND,
            artifact_ref=self.artifact_ref,
            artifact_hash=self.artifact_hash,
        )


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


class RetainedContextItemChangeV1(_ContentAddressed):
    ARTIFACT_KIND = 'retained_context_item_change'
    schema_version: Literal['retained_context_item_change.v1'] = 'retained_context_item_change.v1'
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    item_ref: str = Field(min_length=1)
    source_file: Literal['MEMORY.md', 'USER.md']
    revision: int = Field(ge=1)
    expected_revision: str
    prior_content: str
    content: str | None
    action: Literal['correct', 'suppress']
    request_id: str = Field(min_length=1)
    recorded_at: datetime
    origin: dict[str, str] | None = None

    @model_validator(mode='after')
    def action_binds_content(self):
        if (self.action == 'suppress') != (self.content is None):
            raise ValueError('action must bind exact correction/suppression content')
        return self


class PersonalCapabilityTransferV1(_ContentAddressed):
    ARTIFACT_KIND='personal_capability_transfer'
    schema_version: Literal['personal_capability_transfer.v1']='personal_capability_transfer.v1'
    organization_id: str
    workspace_id: str
    owner_ref: str
    source_store_incarnation: str=Field(pattern=r'^[a-f0-9]{32}$')
    source_object_id: str
    source_revision: str
    allowed_purpose: Literal['personal_capability_reuse']
    skill: SkillArtifactVersionV1
    decision_context: RetainedContextItemChangeV1
    context_basis: SkillArtifactVersionV1
    predecessor: ArtifactPinV1 | None=None
    recorded_at: datetime

    @model_validator(mode='after')
    def exact_source(self):
        scope=(self.organization_id,self.workspace_id,self.owner_ref)
        if any((obj.organization_id,obj.workspace_id,obj.owner_ref)!=scope for obj in (self.skill,self.decision_context,self.context_basis)):
            raise ValueError('source author/scope mismatch')
        identity=content_hash([self.skill.skill_id,self.decision_context.item_ref])
        if self.source_object_id!=identity or self.source_revision!=f'{self.skill.version}:{self.decision_context.revision}':
            raise ValueError('logical source identity mismatch')
        origin=self.decision_context.origin or {}
        if (origin.get('source_ref'),origin.get('source_hash'))!=(self.context_basis.artifact_ref,self.context_basis.artifact_hash):
            raise ValueError('decision context requires exact original source')
        if not self.decision_context.content:
            raise ValueError('suppressed context cannot be transferred')
        if self.predecessor and self.predecessor.artifact_kind!='personal_capability_import':
            raise ValueError('predecessor must pin a durable import acknowledgement')
        if not self.skill.files or any(len(version.files)>128 for version in (self.skill,self.context_basis)):
            raise ValueError('bounded selected Skill package required')
        total=0
        seen=set()
        for file in (*self.skill.files,*self.context_basis.files):
            path=PurePosixPath(file.relative_path)
            parts=[part.casefold() for part in path.parts]
            if (any(p.startswith('.env') or p in {'.aws','.ssh','credentials','credentials.json','sessions','state.db','auth.db'} for p in parts)
                    or path.suffix.casefold() in {'.db','.sqlite','.sqlite3','.pem','.key','.pfx'}
                    or path.name.casefold().endswith(('-wal','-shm','.wal','.lock'))):
                raise ValueError('credential/database/runtime files cannot be transferred')
            if (file.relative_path,file.content_hash) not in seen:
                total+=len(file.content_bytes())
                seen.add((file.relative_path,file.content_hash))
        if total>1048576 or len(self.decision_context.model_dump_json().encode())>32768:
            raise ValueError('personal transfer exceeds bounded payload')
        return self
