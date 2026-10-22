"""Immutable raw ontology and actual Hub outputs; offline validation needs no parser."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from contracts.ontology_catalog import OntologyCatalogMetadata
from contracts.ontology_seeds import StructuralTripleV1
from eos.artifact_hashing import content_hash

_HASH = r"^sha256:[0-9a-f]{64}$"


class OntologyHubSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["ontology_hub_snapshot.v1"] = "ontology_hub_snapshot.v1"
    snapshot_ref: str
    snapshot_hash: str = Field(pattern=_HASH)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    ontology_ref: str = Field(min_length=1)
    version: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    provider: Literal["huggingface"] = "huggingface"
    origin: OntologyCatalogMetadata
    source_base64: str
    source_content_hash: str = Field(pattern=_HASH)
    graph_content_hash: str = Field(pattern=_HASH)
    canonical_inventory: tuple[StructuralTripleV1, ...]
    rdf_format: Literal["turtle", "nt", "xml"]
    semantica_ref: Literal["third_party/semantica"] = "third_party/semantica"
    semantica_version: str = Field(min_length=1)
    semantica_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    hub_metadata: dict[str, Any]
    hub_registry: dict[str, Any]
    hub_graph: dict[str, Any]
    hub_version: dict[str, Any]
    hub_versioned_ontology: dict[str, Any]
    projection_hashes: dict[str, str]
    authority: Literal["REFERENCE"] = "REFERENCE"
    created_at: datetime

    def source_bytes(self) -> bytes:
        return base64.b64decode(self.source_base64, validate=True)

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"snapshot_ref", "snapshot_hash"})

    @model_validator(mode="after")
    def validate_exact_outputs(self) -> "OntologyHubSnapshotV1":
        raw = self.source_bytes()
        if base64.b64encode(raw).decode("ascii") != self.source_base64:
            raise ValueError("source_base64 must use canonical encoding")
        if self.source_content_hash != "sha256:" + sha256(raw).hexdigest():
            raise ValueError("source bytes do not match source_content_hash")
        if self.version != self.origin.commit:
            raise ValueError("snapshot version must be exact upstream commit")
        inventory = [item.model_dump(mode="json") for item in self.canonical_inventory]
        if self.graph_content_hash != "sha256:" + content_hash(inventory):
            raise ValueError("canonical inventory does not match graph_content_hash")
        expected_hashes = {name: "sha256:" + content_hash(getattr(self, name)) for name in (
            "hub_metadata", "hub_registry", "hub_graph", "hub_version", "hub_versioned_ontology",
        )}
        if self.projection_hashes != expected_hashes:
            raise ValueError("Hub projection hashes do not match serialized outputs")
        registry, metadata, versioned = self.hub_registry, self.hub_metadata, self.hub_versioned_ontology
        version_metadata = self.hub_version.get("metadata", {})
        if (
            registry.get("uri") != metadata.get("uri")
            or registry.get("name") != metadata.get("name")
            or registry.get("source_url") != self.origin.external_ref
            or registry.get("version") != self.version
            or registry.get("format") != self.rdf_format
            or metadata.get("triple_count") != len(self.canonical_inventory)
            or version_metadata.get("source_content_hash") != self.source_content_hash
            or version_metadata.get("source_ontology_uri") != registry.get("uri")
            or version_metadata.get("structure") != self.hub_graph
            or versioned.get("nodes") != self.hub_graph.get("nodes")
            or versioned.get("edges") != self.hub_graph.get("edges")
            or versioned.get("uri") != self.hub_version.get("ontology_iri")
            or versioned.get("version") != self.version
            or versioned.get("versionInfo") != self.version
            or self.hub_version.get("version_info") != self.version
        ):
            raise ValueError("Hub output binding is inconsistent with source, registry, graph or version")
        expected_ontology_ref = "semantica-ontology:" + content_hash({
            "organization_id": self.organization_id, "workspace_id": self.workspace_id,
            "provider": "huggingface", "upstream_id": self.origin.upstream_id, "file_path": self.origin.file_path,
        })
        if self.ontology_ref != expected_ontology_ref:
            raise ValueError("ontology identity binding does not match tenant/provider origin")
        times = [self.created_at, datetime.fromisoformat(self.hub_version["created_at"]),
                 datetime.fromisoformat(self.hub_registry["loaded_at"])]
        if any(value.tzinfo is None or value.utcoffset() != timedelta(0) for value in times):
            raise ValueError("runtime-generated snapshot timestamps must be UTC")
        if self.hub_registry.get("status") != "external" or self.hub_version.get("version") != self.version:
            raise ValueError("Hub registry/version does not match imported reference")
        digest = content_hash(self.hash_payload())
        if self.snapshot_hash != f"sha256:{digest}" or self.snapshot_ref != f"ontology-hub-snapshot:{digest}":
            raise ValueError("snapshot ref/hash does not bind exact snapshot content")
        return self

    @classmethod
    def create(cls, **values: Any) -> "OntologyHubSnapshotV1":
        material = to_jsonable_python({"schema_version": "ontology_hub_snapshot.v1", "provider": "huggingface",
            "semantica_ref": "third_party/semantica", "authority": "REFERENCE", **values})
        digest = content_hash(material)
        return cls(snapshot_ref=f"ontology-hub-snapshot:{digest}", snapshot_hash=f"sha256:{digest}", **values)
