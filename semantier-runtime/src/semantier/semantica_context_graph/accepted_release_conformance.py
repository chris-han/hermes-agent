"""Release-level RDF/SHACL checks for accepted semantic graph checkpoints."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF, URIRef

from contracts.accepted_graph_release_lineage import AcceptedGraphReleaseLineageV2
from contracts.accepted_graph_release_v2 import AcceptedGraphReleaseV2
from contracts.knowledge_relation_instance import (
    AcceptedRelationBindingV1,
    KnowledgeRelationInstanceV1,
)
from eos.artifact_hashing import content_hash


SEM = Namespace("urn:semantier:accepted-graph:")
SHAPES_VERSION = "accepted_release_shapes.v1"
_SHAPES_TTL = r'''
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix sem: <urn:semantier:accepted-graph:> .

sem:RelationInstanceShape a sh:NodeShape ;
  sh:targetClass sem:KnowledgeRelationInstance ;
  sh:property [ sh:path sem:organizationId ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:workspaceId ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:relationTypeId ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:representationKind ; sh:minCount 1 ; sh:maxCount 1 ; sh:in ("binary_edge" "frame_role" "derived_projection") ] ;
  sh:property [ sh:path sem:sourceNodeId ; sh:class sem:GraphNode ] ;
  sh:property [ sh:path sem:targetNodeId ; sh:class sem:GraphNode ] ;
  sh:property [ sh:path sem:frameRef ; sh:class sem:GraphNode ] ;
  sh:property [ sh:path sem:evidenceRef ; sh:minCount 1 ] ;
  sh:property [ sh:path sem:acceptedBindingHash ; sh:minCount 1 ; sh:maxCount 1 ; sh:pattern "^sha256:[0-9a-f]{64}$" ] ;
  sh:sparql [
    sh:message "binary_edge requires exactly one source and target and no frame" ;
    sh:select """PREFIX sem: <urn:semantier:accepted-graph:>
    SELECT $this WHERE {
      $this sem:representationKind "binary_edge" .
      FILTER (
        NOT EXISTS { $this sem:sourceNodeId ?source } ||
        NOT EXISTS { $this sem:targetNodeId ?target } ||
        EXISTS { $this sem:frameRef ?frame }
      )
    }"""
  ] ;
  sh:sparql [
    sh:message "frame_role requires one frame and no binary endpoints" ;
    sh:select """PREFIX sem: <urn:semantier:accepted-graph:>
    SELECT $this WHERE {
      $this sem:representationKind "frame_role" .
      FILTER (
        NOT EXISTS { $this sem:frameRef ?frame } ||
        EXISTS { $this sem:sourceNodeId ?source } ||
        EXISTS { $this sem:targetNodeId ?target }
      )
    }"""
  ] ;
  sh:sparql [
    sh:message "derived_projection requires sources and exactly one endpoint shape" ;
    sh:select """PREFIX sem: <urn:semantier:accepted-graph:>
    SELECT $this WHERE {
      $this sem:representationKind "derived_projection" .
      FILTER (
        NOT EXISTS { $this sem:projectionSourceRef ?projection } ||
        (EXISTS { $this sem:sourceNodeId ?source } != EXISTS { $this sem:targetNodeId ?target }) ||
        (NOT EXISTS { $this sem:sourceNodeId ?source } && NOT EXISTS { $this sem:frameRef ?frame }) ||
        (EXISTS { $this sem:sourceNodeId ?source } && EXISTS { $this sem:frameRef ?frame })
      )
    }"""
  ] .

sem:PatternShape a sh:NodeShape ;
  sh:targetClass sem:CanonicalRestrictionPattern ;
  sh:property [ sh:path sem:organizationId ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:workspaceId ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:patternKind ; sh:minCount 1 ; sh:maxCount 1 ; sh:in ("atomic" "composed") ] ;
  sh:property [ sh:path sem:sourceIdentityRef ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:graphIntentRef ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:schemaCandidateRef ; sh:minCount 1 ; sh:maxCount 1 ; sh:minLength 1 ] ;
  sh:property [ sh:path sem:acceptedBindingHash ; sh:minCount 1 ; sh:maxCount 1 ; sh:pattern "^sha256:[0-9a-f]{64}$" ] ;
  sh:property [ sh:path sem:memberPattern ; sh:class sem:CanonicalRestrictionPattern ] ;
  sh:sparql [
    sh:message "atomic patterns cannot contain members" ;
    sh:select """PREFIX sem: <urn:semantier:accepted-graph:>
    SELECT $this WHERE {
      $this sem:patternKind "atomic" ; sem:memberPattern ?member .
    }"""
  ] ;
  sh:sparql [
    sh:message "composed patterns require members" ;
    sh:select """PREFIX sem: <urn:semantier:accepted-graph:>
    SELECT $this WHERE {
      $this sem:patternKind "composed" .
      FILTER NOT EXISTS { $this sem:memberPattern ?member }
    }"""
  ] .
'''
SHAPES_HASH = "sha256:" + content_hash(_SHAPES_TTL)


def _resource(kind: str, identity: str) -> URIRef:
    return URIRef(f"urn:semantier:accepted-graph:{kind}/{quote(identity, safe='')}")


def _add_optional(graph: Graph, subject: URIRef, predicate: URIRef, value: Any) -> None:
    if value is not None and value != "":
        graph.add((subject, predicate, Literal(str(value))))


def _rdf_projection(*, bundle: dict[str, Any], graph: Any) -> Graph:
    release = AcceptedGraphReleaseV2.parse_unverified(bundle["release"])
    lineage = AcceptedGraphReleaseLineageV2.parse_unverified(bundle["lineage"])
    if (
        lineage.accepted_graph_release_ref != release.graph_version
        or lineage.accepted_graph_release_hash != release.release_hash
    ):
        raise ValueError("accepted release conformance lineage mismatch")
    data = Graph()
    for topology_node in (*release.nodes, *release.rules):
        node_id = topology_node.get("node_id") or topology_node.get("rule_id")
        if not node_id:
            raise ValueError("accepted release topology identity is missing")
        data.add((_resource("node", node_id), RDF.type, SEM.GraphNode))
    relation_refs = {
        item.relation_instance_ref for item in release.relation_membership
    }
    pattern_refs = {item.pattern_ref for item in release.pattern_membership}
    relation_dtos = {
        item.relation_instance_id: item
        for item in (
            KnowledgeRelationInstanceV1.model_validate(payload)
            for payload in bundle.get("knowledge_relation_instances", ())
        )
    }
    relation_bindings = {
        item.relation_instance_ref: item
        for item in (
            AcceptedRelationBindingV1.model_validate(payload)
            for payload in bundle.get("accepted_relation_bindings", ())
        )
    }
    pattern_dtos = {
        str(payload.get("pattern_id")): payload
        for payload in bundle.get("canonical_restriction_patterns", ())
        if isinstance(payload, dict)
    }
    pattern_bindings = {
        str(payload.get("pattern_ref")): payload
        for payload in bundle.get("accepted_pattern_bindings", ())
        if isinstance(payload, dict)
    }
    if (
        set(relation_dtos) != relation_refs
        or set(relation_bindings) != relation_refs
        or set(pattern_dtos) != pattern_refs
        or set(pattern_bindings) != pattern_refs
    ):
        raise ValueError("accepted release conformance bundle membership mismatch")
    for relation_ref in sorted(relation_refs, key=lambda item: item.encode("utf-8")):
        node = graph.nodes.get(relation_ref)
        if node is None or getattr(node, "node_type", None) != "knowledge_relation_instance":
            raise ValueError("accepted release Semantica relation node mismatch")
        properties = node.properties
        relation = relation_dtos[relation_ref]
        binding = relation_bindings[relation_ref]
        expected_properties = {
            "organization_id": relation.organization_id,
            "workspace_id": relation.workspace_id,
            "relation_type_id": relation.relation_type_id,
            "representation_kind": relation.representation_kind,
            "source_node_id": relation.source_node_id,
            "target_node_id": relation.target_node_id,
            "frame_ref": relation.frame_ref,
            "projection_source_refs": list(relation.projection_source_refs),
            "evidence_refs": [
                item.model_dump(mode="json", by_alias=True)
                for item in relation.evidence_refs
            ],
            "accepted_relation_binding_id": binding.binding_id,
            "accepted_relation_binding_hash": binding.binding_hash,
        }
        if any(
            properties.get(key) != value
            for key, value in expected_properties.items()
        ):
            raise ValueError(
                "accepted release Semantica relation projection differs from exact bundle"
            )
        subject = _resource("relation", relation_ref)
        data.add((subject, RDF.type, SEM.KnowledgeRelationInstance))
        for predicate, key in (
            (SEM.organizationId, "organization_id"),
            (SEM.workspaceId, "workspace_id"),
            (SEM.relationTypeId, "relation_type_id"),
            (SEM.representationKind, "representation_kind"),
            (SEM.acceptedBindingHash, "accepted_relation_binding_hash"),
        ):
            _add_optional(data, subject, predicate, properties.get(key))
        for predicate, key in (
            (SEM.sourceNodeId, "source_node_id"),
            (SEM.targetNodeId, "target_node_id"),
            (SEM.frameRef, "frame_ref"),
        ):
            endpoint = properties.get(key)
            if endpoint:
                endpoint_resource = _resource("node", str(endpoint))
                data.add((subject, predicate, endpoint_resource))
        for evidence in properties.get("evidence_refs") or ():
            if isinstance(evidence, dict):
                evidence_ref = evidence.get("evidenceRef") or evidence.get("evidence_ref")
                _add_optional(data, subject, SEM.evidenceRef, evidence_ref)
        for projection_ref in properties.get("projection_source_refs") or ():
            _add_optional(data, subject, SEM.projectionSourceRef, projection_ref)

    for pattern_ref in sorted(pattern_refs, key=lambda item: item.encode("utf-8")):
        node = graph.nodes.get(pattern_ref)
        if node is None or getattr(node, "node_type", None) != "canonical_restriction_pattern":
            raise ValueError("accepted release Semantica pattern node mismatch")
        properties = node.properties
        pattern = pattern_dtos[pattern_ref]
        binding = pattern_bindings[pattern_ref]
        expected_properties = {
            "organization_id": pattern["organization_id"],
            "workspace_id": pattern["workspace_id"],
            "pattern_kind": pattern["pattern_kind"],
            "frames": pattern["frames"],
            "composition": pattern["composition"],
            "semantic_dimensions": pattern["semantic_dimensions"],
            "source_metadata": pattern["source_metadata"],
            "witness_metadata": pattern["witness_metadata"],
            "lexical_evidence_refs": pattern["lexical_evidence_refs"],
            "lineage_metadata": pattern["lineage_metadata"],
            "authority_metadata": pattern["authority_metadata"],
            "accepted_pattern_binding_id": binding["binding_id"],
            "accepted_pattern_binding_hash": binding["binding_hash"],
        }
        if any(
            properties.get(key) != value
            for key, value in expected_properties.items()
        ):
            raise ValueError(
                "accepted release Semantica pattern projection differs from exact bundle"
            )
        subject = _resource("pattern", pattern_ref)
        data.add((subject, RDF.type, SEM.CanonicalRestrictionPattern))
        source = properties.get("source_metadata") or {}
        lineage_metadata = properties.get("lineage_metadata") or {}
        for predicate, value in (
            (SEM.organizationId, properties.get("organization_id")),
            (SEM.workspaceId, properties.get("workspace_id")),
            (SEM.patternKind, properties.get("pattern_kind")),
            (SEM.sourceIdentityRef, source.get("source_identity_ref")),
            (SEM.graphIntentRef, lineage_metadata.get("graph_intent_brief_ref")),
            (SEM.schemaCandidateRef, lineage_metadata.get("schema_candidate_ref")),
            (SEM.acceptedBindingHash, properties.get("accepted_pattern_binding_hash")),
        ):
            _add_optional(data, subject, predicate, value)
        composition = properties.get("composition") or {}
        for member_ref in composition.get("member_pattern_refs") or ():
            data.add((subject, SEM.memberPattern, _resource("pattern", str(member_ref))))
    return data


def validate_accepted_release_conformance(
    *, bundle: dict[str, Any], graph: Any,
) -> dict[str, Any]:
    """Validate one exact v2 native projection with the pinned SHACL shape set."""

    data_graph = _rdf_projection(bundle=bundle, graph=graph)
    shapes_graph = Graph().parse(data=_SHAPES_TTL, format="turtle")
    conforms, _report_graph, report_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="rdfs",
        advanced=True,
        allow_infos=False,
        allow_warnings=False,
    )
    if conforms is not True:
        raise ValueError(
            "accepted release SHACL conformance failed: "
            + str(report_text).strip()
        )
    report_payload = {
        "schema_version": "accepted_release_conformance_report.v1",
        "shapes_version": SHAPES_VERSION,
        "shapes_hash": SHAPES_HASH,
        "release_hash": bundle["release"]["release_hash"],
        "conforms": True,
        "data_triple_count": len(data_graph),
    }
    return {
        **report_payload,
        "report_hash": "sha256:" + content_hash(report_payload),
    }


__all__ = [
    "SHAPES_HASH", "SHAPES_VERSION", "validate_accepted_release_conformance",
]
