"""Exact cross-artifact identity shared by graph and quality contracts."""
from pydantic import BaseModel, ConfigDict, Field


class ArtifactPinV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_kind: str = Field(min_length=1)
    artifact_ref: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
