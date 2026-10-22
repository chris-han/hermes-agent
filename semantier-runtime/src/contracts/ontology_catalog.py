from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_seed_query import OntologySeedQuery


_EXTERNAL_REF = re.compile(
    r"^hf-dataset://(?P<repo>[^/@]+/[^/@]+)@(?P<commit>[0-9a-f]{40,64})/(?P<path>.+)$"
)


def parse_catalog_external_ref(external_ref: str) -> tuple[str, str, str]:
    match = _EXTERNAL_REF.fullmatch(external_ref)
    if match is None:
        raise ValueError("external_ref must be an immutable Hugging Face dataset ref")
    repo, commit, path = match.group("repo", "commit", "path")
    parsed_path = PurePosixPath(path)
    if path.startswith("/") or ".." in parsed_path.parts or "\\" in path:
        raise ValueError("external_ref contains an invalid ontology path")
    return repo, commit, path


class CatalogProviderError(RuntimeError):
    """Bounded, actionable failure from an external ontology catalog."""


class OntologyCatalogCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    external_ref: str
    upstream_id: str
    commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    file_path: str
    license: str | None = None
    description: str | None = None
    provenance: str
    authority: str = Field(default="REFERENCE", pattern=r"^REFERENCE$")
    profile_status: str = Field(default="unprofiled", pattern=r"^unprofiled$")
    lexical_score: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_external_identity(self) -> "OntologyCatalogCandidate":
        repo, commit, path = parse_catalog_external_ref(self.external_ref)
        if (repo, commit, path) != (self.upstream_id, self.commit, self.file_path):
            raise ValueError("external_ref does not agree with identity fields")
        if self.provenance != f"huggingface-dataset:{repo}@{commit}":
            raise ValueError("provenance does not agree with external_ref")
        return self


class OntologyCatalogMetadata(OntologyCatalogCandidate):
    pass


class OntologyCatalogPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    external_ref: str
    upstream_id: str
    commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    file_path: str
    content: bytes
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    authority: str = Field(default="REFERENCE", pattern=r"^REFERENCE$")

    @model_validator(mode="after")
    def validate_immutable_identity(self) -> "OntologyCatalogPayload":
        repo, commit, path = parse_catalog_external_ref(self.external_ref)
        if (repo, commit, path) != (self.upstream_id, self.commit, self.file_path):
            raise ValueError("external_ref does not agree with identity fields")
        if self.sha256 != "sha256:" + sha256(self.content).hexdigest():
            raise ValueError("sha256 does not match payload content")
        return self


class OntologyCatalogProvider(Protocol):
    async def search(self, query: OntologySeedQuery) -> tuple[OntologyCatalogCandidate, ...]: ...
    async def metadata(self, external_ref: str) -> OntologyCatalogMetadata: ...
    async def fetch(self, external_ref: str) -> OntologyCatalogPayload: ...
