"""Offline exact-version catalogue over governed Semantica snapshots/profiles."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict

from contracts.ontology_hub_snapshot import OntologyHubSnapshotV1
from contracts.ontology_seed_query import OntologySeedQuery
from contracts.ontology_seeds import OntologyCapabilityProfileV1
from storage.ontology_build_store import OntologyBuildStore


class RegistryCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_ref: str
    snapshot_hash: str
    ontology_ref: str
    version: str
    name: str
    source_content_hash: str
    graph_content_hash: str
    profile_ref: str | None
    profile_hash: str | None
    profile_status: Literal["profiled", "unprofiled"]
    matching_eligible: bool
    exclusion_reason: Literal["unprofiled"] | None
    authority: Literal["REFERENCE", "ORGANIZATION", "DOMAIN_TEMPLATE"]


class RegistryCatalogPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    entries: tuple[RegistryCatalogEntry, ...]
    truncated: bool
    total_count: int
    max_results: int


class SemanticaRegistryCatalogProvider:
    """Discovery exposes each exact profile revision; replay never picks latest."""
    def __init__(self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str, max_results: int = 100):
        if max_results < 1 or max_results > 1000:
            raise ValueError("registry max_results must be between 1 and 1000")
        self._max_results = max_results
        self._store = store
        self._organization_id = organization_id
        self._workspace_id = workspace_id

    @staticmethod
    def _entry(snapshot: OntologyHubSnapshotV1, profile: OntologyCapabilityProfileV1 | None) -> RegistryCatalogEntry:
        return RegistryCatalogEntry(
            snapshot_ref=snapshot.snapshot_ref, snapshot_hash=snapshot.snapshot_hash,
            ontology_ref=snapshot.ontology_ref, version=snapshot.version,
            name=snapshot.hub_registry["name"], source_content_hash=snapshot.source_content_hash,
            graph_content_hash=snapshot.graph_content_hash,
            profile_ref=profile.profile_ref if profile else None, profile_hash=profile.profile_hash if profile else None,
            profile_status="profiled" if profile else "unprofiled", matching_eligible=profile is not None,
            exclusion_reason=None if profile else "unprofiled",
            authority=profile.authority.seed_class if profile else snapshot.authority,
        )

    async def search(self, query: OntologySeedQuery | None = None) -> tuple[RegistryCatalogEntry, ...]:
        return (await self.search_page(query)).entries

    async def search_page(self, query: OntologySeedQuery | None = None) -> RegistryCatalogPage:
        if query is not None:
            query = OntologySeedQuery.model_validate(query.model_dump(mode="json"))
            if (query.organization_id, query.workspace_id) != (self._organization_id, self._workspace_id):
                raise PermissionError("seed query is outside registry tenant scope")
        rows, total = self._store.list_hub_catalog_page(organization_id=self._organization_id,
            workspace_id=self._workspace_id, limit=self._max_results)
        return RegistryCatalogPage(entries=tuple(self._entry(snapshot, profile) for snapshot, profile in rows),
            truncated=total > self._max_results, total_count=total, max_results=self._max_results)

    async def fetch(self, snapshot_ref: str) -> OntologyHubSnapshotV1:
        snapshot = self._store.get(organization_id=self._organization_id, workspace_id=self._workspace_id,
                                  artifact_ref=snapshot_ref, contract_type=OntologyHubSnapshotV1)
        if snapshot is None:
            raise LookupError("ontology snapshot not found")
        return snapshot

    async def metadata(self, snapshot_ref: str, *, profile_ref: str | None = None) -> RegistryCatalogEntry:
        await self.fetch(snapshot_ref)
        rows, total = self._store.list_hub_catalog_page(organization_id=self._organization_id,
            workspace_id=self._workspace_id, limit=2, snapshot_ref=snapshot_ref, profile_ref=profile_ref)
        if total != 1:
            raise ValueError("exact profile_ref is required for ambiguous or missing profile revision")
        return self._entry(*rows[0])

    async def fetch_profile(self, snapshot_ref: str, profile_ref: str) -> OntologyCapabilityProfileV1:
        await self.fetch(snapshot_ref)
        rows, total = self._store.list_hub_catalog_page(organization_id=self._organization_id,
            workspace_id=self._workspace_id, limit=1, snapshot_ref=snapshot_ref, profile_ref=profile_ref)
        if total != 1 or rows[0][1] is None:
            raise LookupError("exact ontology snapshot profile not found")
        return rows[0][1]


RegistryCatalogProvider = SemanticaRegistryCatalogProvider

__all__ = ["RegistryCatalogEntry", "RegistryCatalogPage", "SemanticaRegistryCatalogProvider", "RegistryCatalogProvider"]
