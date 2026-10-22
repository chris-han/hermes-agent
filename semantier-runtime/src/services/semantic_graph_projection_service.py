"""Authoritative-bundle to renderer-independent semantic projection."""

from __future__ import annotations

from collections import Counter
from typing import Any

from contracts.knowledge_relation_instance import KnowledgeRelationInstanceV1
from contracts.semantic_graph_visual_projection import (
    SemanticGraphVisualProjectionV1,
    create_semantic_graph_visual_projection,
)
from services.knowledge_relation_taxonomy_service import (
    KnowledgeRelationTaxonomyService,
)
from services.materialized_accepted_graph_bundle_reader import (
    MaterializedAcceptedGraphBundleReader,
)
from services.semantic_layout_engine import SemanticLayoutEngine
from storage.schema_decision_store import SchemaDecisionStore


class SemanticGraphProjectionFailure(ValueError):
    """Typed stable failure crossing the projection API boundary."""

    def __init__(
        self,
        *,
        code: str,
        status_code: int,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message
        self.retryable = retryable


class SemanticGraphProjectionService:
    """Project one exact materialized accepted release without semantic inference."""

    def __init__(
        self,
        *,
        bundle_reader: MaterializedAcceptedGraphBundleReader,
        taxonomy_service: KnowledgeRelationTaxonomyService,
        layout_engine: SemanticLayoutEngine,
        schema_decision_store: SchemaDecisionStore | None = None,
    ) -> None:
        if type(bundle_reader) is not MaterializedAcceptedGraphBundleReader:
            raise TypeError("projection requires the authoritative bundle reader")
        if type(taxonomy_service) is not KnowledgeRelationTaxonomyService:
            raise TypeError("projection requires the governed taxonomy service")
        if type(layout_engine) is not SemanticLayoutEngine:
            raise TypeError("projection requires the deterministic layout engine")
        if schema_decision_store is not None and type(schema_decision_store) is not SchemaDecisionStore:
            raise TypeError("projection requires the governed schema decision store")
        self._bundle_reader = bundle_reader
        self._taxonomy_service = taxonomy_service
        self._layout_engine = layout_engine
        self._schema_decision_store = schema_decision_store

    def project(
        self, release_payload: dict[str, Any]
    ) -> SemanticGraphVisualProjectionV1:
        bundle = self._bundle_reader.read(release_payload)
        release = bundle["release"]
        self._require_knowledge_plane(bundle)
        instances = tuple(
            KnowledgeRelationInstanceV1.model_validate(item)
            for item in bundle["knowledge_relation_instances"]
        )
        pin_set = {
            (
                item.taxonomy_ref,
                item.taxonomy_hash,
                item.profile_ref,
                item.profile_hash,
                item.profile_relation_catalog_hash,
            )
            for item in instances
        }
        if not instances:
            pin_set = self._empty_release_pin_set(bundle)
        if len(pin_set) != 1:
            raise ValueError("accepted release contains mixed taxonomy/profile pins")
        taxonomy_ref, taxonomy_hash, profile_ref, profile_hash, profile_catalog_hash = (
            next(iter(pin_set))
        )
        taxonomy = self._taxonomy_service.resolve_taxonomy(
            taxonomy_ref, taxonomy_hash
        )
        relation_types = {item.relation_type_id: item for item in taxonomy.relation_types}
        if profile_ref is not None:
            profile_catalog = self._taxonomy_service.load_profile_catalog(
                profile_ref=profile_ref,
                profile_hash=profile_hash,
                profile_relation_catalog_hash=profile_catalog_hash,
                taxonomy_ref=taxonomy_ref,
                taxonomy_hash=taxonomy_hash,
            )
            relation_types.update(
                {item.relation_type_id: item for item in profile_catalog.relation_types}
            )

        admissions = {}
        for instance in instances:
            admission = self._taxonomy_service.resolve_relation_type(
                relation_type_id=instance.relation_type_id,
                taxonomy_ref=instance.taxonomy_ref,
                taxonomy_hash=instance.taxonomy_hash,
                profile_ref=instance.profile_ref,
                profile_hash=instance.profile_hash,
                profile_relation_catalog_hash=instance.profile_relation_catalog_hash,
                for_new_write=False,
            )
            if admission.relation_type.representation_kind != instance.representation_kind:
                raise ValueError("relation representation differs from admitted type")
            admissions[instance.relation_instance_ref] = admission

        layout_nodes = [self._layout_node(item) for item in release["nodes"]]
        layout_edges = [
            self._layout_edge(instance, admissions[instance.relation_instance_ref])
            for instance in instances
        ]
        coordinate_seed = ":".join(
            (
                release["release_hash"],
                taxonomy_hash,
                profile_hash or "no-profile",
                self._layout_engine.profile.profile_hash,
                self._layout_engine.profile.engine_version,
            )
        )
        layout = self._layout_engine.layout(
            nodes=layout_nodes,
            edges=layout_edges,
            coordinate_seed=coordinate_seed,
        )
        degrees = Counter()
        for edge in layout_edges:
            degrees[edge["source"]] += 1
            degrees[edge["target"]] += 1
        maximum_degree = max(degrees.values(), default=1)
        projected_nodes = []
        for item in release["nodes"]:
            node_id = str(item["node_id"])
            coordinate = layout.coordinates[node_id]
            projected_nodes.append(
                {
                    "id": node_id,
                    "label": str(
                        item.get("canonical_label")
                        or item.get("display_name_zh")
                        or item.get("label")
                        or node_id
                    ),
                    "object_type_id": str(
                        item.get("object_type_id")
                        or item.get("type_id")
                        or item.get("node_type")
                        or "unclassified"
                    ),
                    "role_id": item.get("role_id"),
                    "x": coordinate.x,
                    "y": coordinate.y,
                    "z_semantic": coordinate.z_semantic,
                    "cluster_id": coordinate.cluster_id,
                    "importance": float(
                        item.get("importance", degrees[node_id] / maximum_degree)
                    ),
                    "properties": dict(item),
                }
            )

        projected_edges = []
        release_node_ids = {str(item["node_id"]) for item in release["nodes"]}
        for instance in instances:
            admission = admissions[instance.relation_instance_ref]
            edge = self._projected_edge(instance, admission)
            if edge["source"] not in release_node_ids or edge["target"] not in release_node_ids:
                raise ValueError("relation projection endpoint is absent from release nodes")
            projected_edges.append(edge)
        counts_by_family = Counter(
            edge["relation_family_id"] for edge in projected_edges
        )
        counts_by_type = Counter(
            edge["relation_type_id"] for edge in projected_edges
        )
        return create_semantic_graph_visual_projection(
            {
                "graph_release_ref": release["graph_version"],
                "graph_release_hash": release["release_hash"],
                "taxonomy_ref": taxonomy_ref,
                "taxonomy_hash": taxonomy_hash,
                "profile_ref": profile_ref,
                "profile_hash": profile_hash,
                "layout_profile_ref": self._layout_engine.profile.profile_ref,
                "layout_profile_hash": self._layout_engine.profile.profile_hash,
                "layout_engine_version": self._layout_engine.profile.engine_version,
                "coordinate_seed": coordinate_seed,
                "plane": "knowledge",
                "diagnostics": layout.diagnostics,
                "nodes": projected_nodes,
                "edges": projected_edges,
                "relation_catalog": {
                    "families": [
                        {
                            "family_id": family.family_id,
                            "label": family.display_name,
                        }
                        for family in taxonomy.families
                    ],
                    "relation_types": [
                        {
                            "relation_type_id": relation_type.relation_type_id,
                            "family_id": relation_type.family_id,
                            "label": relation_type.display_name,
                        }
                        for relation_type in relation_types.values()
                        if relation_type.lifecycle_state == "admitted"
                    ],
                    "counts_by_family": {
                        family.family_id: counts_by_family[family.family_id]
                        for family in taxonomy.families
                    },
                    "counts_by_type": {
                        relation_type_id: counts_by_type[relation_type_id]
                        for relation_type_id, relation_type in relation_types.items()
                        if relation_type.lifecycle_state == "admitted"
                    },
                },
            }
        )

    def _empty_release_pin_set(self, bundle: dict[str, Any]) -> set[tuple[Any, ...]]:
        if self._schema_decision_store is None:
            raise ValueError("empty accepted release requires governed schema decision pins")
        lineage = bundle["lineage"]
        decision = self._schema_decision_store.get_contract(
            lineage["schema_decision_ref"], taxonomy_service=self._taxonomy_service
        )
        if decision.schema_decision_hash != lineage["schema_decision_hash"]:
            raise ValueError("schema decision hash differs from release lineage")
        return {
            (
                item["taxonomy_ref"], item["taxonomy_hash"], item["profile_ref"],
                item["profile_hash"], item["profile_relation_catalog_hash"],
            )
            for item in decision.admitted_relation_type_bindings
        }

    @staticmethod
    def _require_knowledge_plane(bundle: dict[str, Any]) -> None:
        for collection in ("nodes", "edges"):
            for item in bundle["release"].get(collection, []):
                plane = item.get("plane")
                if plane is not None and plane != "knowledge":
                    raise SemanticGraphProjectionFailure(
                        code="GRAPH_RELEASE_PLANE_INVALID",
                        status_code=422,
                        message="Graph release contains mixed-plane topology",
                    )
        for instance in bundle["knowledge_relation_instances"]:
            plane = instance.get("plane")
            if plane is not None and plane != "knowledge":
                raise SemanticGraphProjectionFailure(
                    code="GRAPH_RELEASE_PLANE_INVALID",
                    status_code=422,
                    message="Graph release contains mixed-plane relation",
                )

    @staticmethod
    def _layout_node(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item["node_id"]),
            "object_type_id": str(
                item.get("object_type_id")
                or item.get("type_id")
                or item.get("node_type")
                or "unclassified"
            ),
            "properties": dict(item),
        }

    @staticmethod
    def _layout_edge(instance, admission) -> dict[str, Any]:
        return {
            "relation_instance_ref": instance.relation_instance_ref,
            "source": instance.source_node_id or instance.frame_ref,
            "target": instance.target_node_id or instance.frame_ref,
            "relation_family_id": admission.relation_type.family_id,
            "relation_type_id": instance.relation_type_id,
        }

    @staticmethod
    def _projected_edge(instance, admission) -> dict[str, Any]:
        frame_carrier = instance.source_node_id is None
        source = instance.frame_ref if frame_carrier else instance.source_node_id
        target = instance.frame_ref if frame_carrier else instance.target_node_id
        if source is None or target is None:
            raise ValueError("relation projection lacks governed visual endpoints")
        return {
            "id": f"projection-edge:{instance.relation_instance_ref}",
            "relation_instance_ref": instance.relation_instance_ref,
            "relation_instance_hash": instance.relation_instance_hash,
            "source": source,
            "target": target,
            "relation_type_id": instance.relation_type_id,
            "relation_family_id": admission.relation_type.family_id,
            "label": admission.relation_type.display_name,
            "representation_kind": instance.representation_kind,
            "frame_ref": instance.frame_ref,
            "projection_source_refs": list(instance.projection_source_refs),
            "evidence_refs": [
                item.model_dump(mode="json", by_alias=True)
                for item in instance.evidence_refs
            ],
            "properties": (
                {
                    "visualization_shape": "frame_carrier",
                    "semantic_endpoints_available": False,
                }
                if frame_carrier else {}
            ),
        }
