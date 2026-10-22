"""Thin governed adapter over the pinned in-memory Semantica Ontology Hub."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from contracts.ontology_catalog import OntologyCatalogMetadata, OntologyCatalogPayload
from contracts.ontology_hub_snapshot import OntologyHubSnapshotV1
from contracts.ontology_seeds import OntologyCapabilityProfileV1, OntologySnapshotIdentityV1, StructuralTripleV1
from eos.artifact_hashing import content_hash
from services.ontology_capability_profile_service import OntologyCapabilityProfileService, graph_content_hash
from storage.ontology_build_store import OntologyBuildStore

MAX_IMPORT_BYTES = 5_000_000
MAX_IMPORT_TRIPLES = 50_000


@dataclass(frozen=True)
class OntologyHubImportResult:
    snapshot: OntologyHubSnapshotV1
    profile: OntologyCapabilityProfileV1


@dataclass(frozen=True)
class HydratedOntologyHub:
    """Dedicated runtime state for one exact tenant snapshot; never shared globally."""
    request: Any
    session: Any
    version_manager: Any


def _pinned_hub():
    from semantier.knowledge_feature_packs.semantica_adapter import _ensure_vendored_semantica_importable
    _ensure_vendored_semantica_importable()
    from semantica.explorer.routes import ontology
    return ontology


def _new_runtime():
    from fastapi import FastAPI
    from starlette.requests import Request
    from semantica.context.context_graph import ContextGraph
    from semantica.explorer.session import GraphSession
    from semantica.change_management.ontology_version_manager import VersionManager
    return Request({"type": "http", "app": FastAPI()}), GraphSession(ContextGraph(advanced_analytics=False)), VersionManager()


def _graph_output(session) -> dict[str, Any]:
    return {
        "nodes": sorted((node.to_dict() for node in session.graph.nodes.values()), key=lambda node: node["id"]),
        "edges": sorted((edge.to_dict() for edge in session.graph.edges), key=lambda edge: edge["id"]),
    }


def _same_origin(left: OntologyCatalogMetadata, right: OntologyCatalogMetadata) -> bool:
    # Discovery rank is query-specific metadata, not immutable source identity.
    return left.model_dump(exclude={"lexical_score"}) == right.model_dump(exclude={"lexical_score"})


def _utc_upstream_timestamp(value: str) -> str:
    # VersionManager uses host-local datetime.now(). Explicitly interpret that
    # generated local value before crossing the governed persistence boundary.
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()


class OntologyHubService:
    def import_payload(self, *, payload: OntologyCatalogPayload, metadata: OntologyCatalogMetadata,
                       organization_id: str, workspace_id: str, store: OntologyBuildStore) -> OntologyHubImportResult:
        payload = OntologyCatalogPayload.model_validate(payload.model_dump(mode="python"))
        metadata = OntologyCatalogMetadata.model_validate(metadata.model_dump(mode="python"))
        if payload.external_ref != metadata.external_ref:
            raise ValueError("catalog payload and metadata identity mismatch")
        if not organization_id or not workspace_id:
            raise ValueError("governed tenant context is required")
        if not payload.content or len(payload.content) > MAX_IMPORT_BYTES:
            raise ValueError("ontology payload exceeds bounded import size")
        formats = {".ttl": "turtle", ".nt": "nt", ".rdf": "xml", ".owl": "xml"}
        fmt = formats.get(PurePosixPath(payload.file_path).suffix.lower())
        if fmt is None:
            raise ValueError("unsupported ontology format; JSON-LD contexts and N3 execution are disabled")
        ontology_ref = "semantica-ontology:" + content_hash({
            "organization_id": organization_id, "workspace_id": workspace_id,
            "provider": "huggingface", "upstream_id": payload.upstream_id, "file_path": payload.file_path,
        })
        existing = store.find_hub_snapshot(organization_id=organization_id, workspace_id=workspace_id,
                                            ontology_ref=ontology_ref, version=payload.commit)
        if existing is not None:
            if existing.source_content_hash != payload.sha256 or not _same_origin(existing.origin, metadata):
                raise ValueError("ontology version cannot be reused with different bytes or origin metadata")
            return OntologyHubImportResult(existing, self._initial_profile(store, existing))

        hub = _pinned_hub()
        from semantica import __version__ as semantica_version
        from rdflib import Graph, Literal, URIRef
        from rdflib.compare import to_canonical_graph
        from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT
        graph = Graph()
        hub._safe_parse_rdf(graph, payload.content, fmt)
        if not graph or len(graph) > MAX_IMPORT_TRIPLES:
            raise ValueError("ontology parsed graph is empty or exceeds triple limit")
        if any(isinstance(term, URIRef) and str(term).startswith('file:') for triple in graph for term in triple):
            raise ValueError("local or relative file IRIs are unsupported")
        nodes, edges, parsed_metadata = hub._parse_rdf_sync(payload.content, fmt)
        request, session, version_manager = _new_runtime()
        session.add_nodes_and_edges(nodes, edges)
        now = datetime.now(timezone.utc)
        entry = hub.OntologyEntry(uri=parsed_metadata["uri"], name=parsed_metadata["name"],
            description=parsed_metadata.get("description"), format=fmt, status="external",
            source_url=payload.external_ref, version=payload.commit,
            class_count=sum(node.get("type") in hub._CLASS_TYPES for node in nodes),
            concept_count=sum(node.get("type") in hub._CONCEPT_TYPES for node in nodes),
            property_count=sum(node.get("type") in hub._PROPERTY_TYPES for node in nodes),
            loaded_at=now.isoformat())
        hub._get_registry(request)[entry.uri] = entry
        graph_output = _graph_output(session)
        versioned = {"uri": entry.uri, **deepcopy(graph_output)}
        version_record = version_manager.create_version(payload.commit, versioned,
            metadata={"structure": deepcopy(graph_output), "source_content_hash": payload.sha256, "source_ontology_uri": entry.uri})
        version_record.created_at = _utc_upstream_timestamp(version_record.created_at)
        outputs = dict(hub_metadata=parsed_metadata, hub_registry=hub._get_registry(request)[entry.uri].model_dump(mode="json"),
            hub_graph=graph_output, hub_version=asdict(version_manager.get_version(payload.commit)),
            hub_versioned_ontology=versioned)
        inventory = tuple(sorted((StructuralTripleV1(
            subject=str(s), predicate=str(p), object=o.n3() if isinstance(o, Literal) else str(o),
        ) for s, p, o in to_canonical_graph(graph)), key=lambda item: (item.subject, item.predicate, item.object)))
        snapshot = OntologyHubSnapshotV1.create(organization_id=organization_id, workspace_id=workspace_id,
            ontology_ref=ontology_ref, version=payload.commit, origin=metadata,
            source_base64=base64.b64encode(payload.content).decode("ascii"), source_content_hash=payload.sha256,
            graph_content_hash=graph_content_hash(graph), canonical_inventory=inventory,
            rdf_format=fmt, semantica_commit=SEMANTICA_COMMIT, semantica_version=semantica_version, **outputs,
            projection_hashes={name: "sha256:" + content_hash(value) for name, value in outputs.items()}, created_at=now)
        identity = OntologySnapshotIdentityV1(organization_id=organization_id, workspace_id=workspace_id,
            ontology_ref=ontology_ref, version=payload.commit, source_content_hash=payload.sha256,
            graph_content_hash=snapshot.graph_content_hash, name=entry.name, domain_primary="unspecified",
            domain_secondary=(), provenance={"snapshot_ref": snapshot.snapshot_ref, "snapshot_hash": snapshot.snapshot_hash,
                "external_ref": payload.external_ref, "upstream_provenance": metadata.provenance},
            license=metadata.license, seed_class="REFERENCE", authority_role="external_reference", governance_state="imported_reference")
        profile = OntologyCapabilityProfileService(generated_at=now).inspect(snapshot=identity, graph=graph, source_bytes=payload.content)
        try:
            store.put_many((snapshot, profile))
        except ValueError:
            winner = store.find_hub_snapshot(organization_id=organization_id, workspace_id=workspace_id,
                ontology_ref=ontology_ref, version=payload.commit)
            if winner is None or winner.source_content_hash != payload.sha256 or not _same_origin(winner.origin, metadata):
                raise
            return OntologyHubImportResult(winner, self._initial_profile(store, winner))
        return OntologyHubImportResult(snapshot, profile)

    @staticmethod
    def _initial_profile(store: OntologyBuildStore, snapshot: OntologyHubSnapshotV1) -> OntologyCapabilityProfileV1:
        profiles = store.list_hub_profiles(organization_id=snapshot.organization_id, workspace_id=snapshot.workspace_id,
                                          snapshot_ref=snapshot.snapshot_ref)
        matches = [profile for profile in profiles if profile.revision == 1]
        if len(matches) != 1:
            raise ValueError("imported snapshot requires its exact initial capability profile")
        return matches[0]

    def hydrate(self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
                snapshot_ref: str) -> HydratedOntologyHub:
        snapshot = store.get(organization_id=organization_id, workspace_id=workspace_id,
                             artifact_ref=snapshot_ref, contract_type=OntologyHubSnapshotV1)
        if snapshot is None:
            raise LookupError("ontology snapshot not found")
        hub = _pinned_hub()
        from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT
        if snapshot.semantica_commit != SEMANTICA_COMMIT:
            raise ValueError("snapshot runtime pin is unavailable for hydration")
        from semantica.change_management.ontology_version_manager import OntologyVersion
        request, session, manager = _new_runtime()
        session.add_nodes_and_edges(deepcopy(snapshot.hub_graph["nodes"]), deepcopy(snapshot.hub_graph["edges"]))
        entry = hub.OntologyEntry.model_validate(snapshot.hub_registry)
        hub._get_registry(request)[entry.uri] = entry
        manager.versions[snapshot.version] = OntologyVersion(**deepcopy(snapshot.hub_version))
        manager.latest_version = snapshot.version
        return HydratedOntologyHub(request, session, manager)
