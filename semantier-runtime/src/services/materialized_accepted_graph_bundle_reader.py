"""Public read-only validator for authoritative materialized graph bundles."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from contracts.accepted_graph_release_lineage import AcceptedGraphReleaseLineageV2
from contracts.accepted_graph_release_v2 import (
    AcceptedGraphReleaseV2,
    CanonicalPatternManifestV1,
    OccurrenceGroundingManifestV1,
)


ArtifactKind = Literal[
    "canonical_restriction_pattern",
    "knowledge_relation_instance",
    "occurrence_grounding_manifest",
    "canonical_pattern_manifest",
    "accepted_relation_binding",
    "accepted_pattern_binding",
]
BindingKind = Literal["accepted_relation_binding", "accepted_pattern_binding"]


class MaterializedAcceptedGraphBundleFailure(ValueError):
    """Stable failure classification for replacement projection APIs."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@runtime_checkable
class MaterializedAcceptedGraphBundleStore(Protocol):
    """Storage capabilities required by the authoritative bundle reader."""

    def read_materialized_release_record(
        self, graph_version: str, *, organization_id: str, workspace_id: str
    ) -> dict[str, Any]: ...

    def read_authoritative_release_lineage(
        self, graph_version: str, *, organization_id: str, workspace_id: str
    ) -> dict[str, Any]: ...

    def list_materialized_release_binding_refs(
        self,
        graph_version: str,
        *,
        binding_kind: BindingKind,
        organization_id: str,
        workspace_id: str,
    ) -> tuple[str, ...]: ...

    def read_materialized_v2_artifact(
        self,
        kind: ArtifactKind,
        identity: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]: ...


class MaterializedAcceptedGraphBundleReader:
    """Resolve a complete exact-pinned bundle without storage-shape access."""

    def __init__(
        self,
        *,
        store: MaterializedAcceptedGraphBundleStore,
        organization_id: str,
        workspace_id: str,
    ) -> None:
        if not isinstance(store, MaterializedAcceptedGraphBundleStore):
            raise TypeError("store does not implement MaterializedAcceptedGraphBundleStore")
        self._store = store
        self._organization_id = organization_id
        self._workspace_id = workspace_id

    def read(self, release_payload: dict[str, Any]) -> dict[str, Any]:
        release = AcceptedGraphReleaseV2.parse_unverified(release_payload)
        record = self._store.read_materialized_release_record(
            release.graph_version,
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
        )
        if (
            record["lineage_contract"] != "exact_v4_v3"
            or record["materialization_status"] != "materialized"
            or record["lifecycle_status"] != "materialized"
            or record["retirement_state"] != "active"
            or record["content_hash"] != release.release_hash
            or record["release"] != release.as_dict()
        ):
            raise MaterializedAcceptedGraphBundleFailure(
                "GRAPH_RELEASE_INTEGRITY_MISMATCH",
                "materialized v2 release record is invalid",
            )

        lineage_payload = self._store.read_authoritative_release_lineage(
            release.graph_version,
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
        )
        lineage = AcceptedGraphReleaseLineageV2.parse_unverified(lineage_payload)
        if (
            lineage.accepted_graph_release_ref != release.graph_version
            or lineage.accepted_graph_release_hash != release.release_hash
            or lineage.organization_id != self._organization_id
            or lineage.workspace_id != self._workspace_id
        ):
            raise MaterializedAcceptedGraphBundleFailure(
                "GRAPH_RELEASE_INTEGRITY_MISMATCH",
                "materialized v2 release lineage is invalid",
            )

        occurrence_payload = self._artifact(
            "occurrence_grounding_manifest",
            lineage.occurrence_grounding_manifest_ref,
        )
        pattern_manifest_payload = self._artifact(
            "canonical_pattern_manifest",
            lineage.canonical_pattern_manifest_ref,
        )
        occurrence = OccurrenceGroundingManifestV1.parse_unverified(occurrence_payload)
        pattern_manifest = CanonicalPatternManifestV1.parse_unverified(
            pattern_manifest_payload
        )
        if (
            occurrence.manifest_hash != lineage.occurrence_grounding_manifest_hash
            or pattern_manifest.manifest_hash != lineage.canonical_pattern_manifest_hash
        ):
            raise MaterializedAcceptedGraphBundleFailure(
                "GRAPH_RELEASE_INTEGRITY_MISMATCH",
                "v2 release lineage manifest hash mismatch",
            )
        occurrence.assert_matches_release(release)
        pattern_manifest.assert_matches_release(release)

        relation_bindings = self._bindings(
            release.graph_version, "accepted_relation_binding"
        )
        pattern_bindings = self._bindings(
            release.graph_version, "accepted_pattern_binding"
        )
        if any(
            item["organizationId"] != self._organization_id
            or item["workspaceId"] != self._workspace_id
            or item["acceptedGraphReleaseRef"] != release.graph_version
            or item["acceptedGraphReleaseHash"] != release.release_hash
            for item in relation_bindings
        ) or any(
            item["organization_id"] != self._organization_id
            or item["workspace_id"] != self._workspace_id
            or item["accepted_graph_release_ref"] != release.graph_version
            or item["accepted_graph_release_hash"] != release.release_hash
            for item in pattern_bindings
        ):
            raise MaterializedAcceptedGraphBundleFailure(
                "GRAPH_RELEASE_INTEGRITY_MISMATCH",
                "v2 release binding release pin or tenant scope mismatch",
            )
        expected_relations = {
            (item.relation_instance_ref, item.relation_instance_hash)
            for item in release.relation_membership
        }
        actual_relations = {
            (item["relationInstanceRef"], item["relationInstanceHash"])
            for item in relation_bindings
        }
        expected_patterns = {
            (item.pattern_ref, item.pattern_hash) for item in release.pattern_membership
        }
        actual_patterns = {
            (item["pattern_ref"], item["pattern_hash"]) for item in pattern_bindings
        }
        if actual_relations != expected_relations or len(relation_bindings) != len(
            expected_relations
        ):
            raise MaterializedAcceptedGraphBundleFailure(
                "GRAPH_RELEASE_BUNDLE_INCOMPLETE",
                "v2 release relation binding completeness mismatch",
            )
        if actual_patterns != expected_patterns or len(pattern_bindings) != len(
            expected_patterns
        ):
            raise MaterializedAcceptedGraphBundleFailure(
                "GRAPH_RELEASE_BUNDLE_INCOMPLETE",
                "v2 release pattern binding completeness mismatch",
            )

        relation_instances = [
            self._artifact("knowledge_relation_instance", item.relation_instance_ref)
            for item in release.relation_membership
        ]
        canonical_patterns = [
            self._artifact("canonical_restriction_pattern", item.pattern_ref)
            for item in release.pattern_membership
        ]
        for membership, instance in zip(
            release.relation_membership, relation_instances, strict=True
        ):
            if instance["relationInstanceHash"] != membership.relation_instance_hash:
                raise MaterializedAcceptedGraphBundleFailure(
                    "GRAPH_RELEASE_INTEGRITY_MISMATCH",
                    "v2 release relation instance hash mismatch",
                )
        for membership, pattern in zip(
            release.pattern_membership, canonical_patterns, strict=True
        ):
            if pattern["pattern_hash"] != membership.pattern_hash:
                raise MaterializedAcceptedGraphBundleFailure(
                    "GRAPH_RELEASE_INTEGRITY_MISMATCH",
                    "v2 release canonical pattern hash mismatch",
                )

        return {
            "release": release.as_dict(),
            "lineage": lineage.as_dict(),
            "occurrence_grounding_manifest": occurrence_payload,
            "canonical_pattern_manifest": pattern_manifest_payload,
            "accepted_relation_bindings": relation_bindings,
            "accepted_pattern_bindings": pattern_bindings,
            "knowledge_relation_instances": relation_instances,
            "canonical_restriction_patterns": canonical_patterns,
        }

    def _artifact(self, kind: ArtifactKind, identity: str) -> dict[str, Any]:
        return self._store.read_materialized_v2_artifact(
            kind,
            identity,
            organization_id=self._organization_id,
            workspace_id=self._workspace_id,
        )

    def _bindings(
        self, graph_version: str, kind: BindingKind
    ) -> list[dict[str, Any]]:
        return [
            self._artifact(kind, identity)
            for identity in self._store.list_materialized_release_binding_refs(
                graph_version,
                binding_kind=kind,
                organization_id=self._organization_id,
                workspace_id=self._workspace_id,
            )
        ]
