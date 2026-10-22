"""Governed candidate artifact for one profile-defined relation type."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.evidence_location import EvidenceRef
from contracts.knowledge_relation_taxonomy import KnowledgeRelationType
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_identity,
)


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _ascii(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.strip():
        raise ValueError(f"{field} must be a nonempty ASCII-stable string")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact sha256 ref")
    return value


class ProfileRelationTypeProposalV1(BaseModel):
    """Content-addressed candidate; it is never an admission artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["profile_relation_type_proposal.v1"]
    proposal_ref: str
    proposal_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    lifecycle_state: Literal["candidate"]
    organization_id: str
    workspace_id: str
    source_binding: dict[str, Any]
    graph_intent_binding: dict[str, Any]
    predecessor_binding: dict[str, Any]
    proposed_relation_type: KnowledgeRelationType
    proposal_origin: Literal["system", "human"]
    actor_ref: str
    evidence_refs: tuple[EvidenceRef, ...]
    occurred_at: datetime
    required_lifecycle: Literal[
        "versioned_successor_profile_catalog_and_schema_decision_v3"
    ]

    @model_validator(mode="after")
    def validate_candidate(self) -> "ProfileRelationTypeProposalV1":
        if (
            self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
            or self.occurred_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("occurred_at must be timezone-aware UTC")
        if not all(
            isinstance(value, str) and value.isascii() and value.strip()
            for value in (
                self.organization_id,
                self.workspace_id,
                self.actor_ref,
            )
        ):
            raise ValueError("proposal ownership and actor fields must be ASCII-stable")
        required_source = {
            "organization_id",
            "workspace_id",
            "lineage_id",
            "source_identity_ref",
            "source_hash",
            "source_version",
            "canonical_source_ir_ref",
            "canonical_source_ir_hash",
        }
        if set(self.source_binding) != required_source:
            raise ValueError("proposal source_binding is invalid")
        if (
            self.source_binding["organization_id"] != self.organization_id
            or self.source_binding["workspace_id"] != self.workspace_id
        ):
            raise ValueError("proposal source ownership mismatch")
        for field, value in self.source_binding.items():
            if field == "source_version" and value is None:
                continue
            if field in {"source_hash", "canonical_source_ir_hash"}:
                _sha256(value, f"source_binding.{field}")
            else:
                _ascii(value, f"source_binding.{field}")
        graph_intent_binding_fields = {
            "graph_intent_ref",
            "graph_intent_hash",
            "relationship_ref",
            "source_type_refs",
            "target_type_refs",
            "direction",
            "semantic_description",
            "preferred_representation",
        }
        endpoint_role_fields = {"source_role_refs", "target_role_refs"}
        present_endpoint_role_fields = endpoint_role_fields.intersection(
            self.graph_intent_binding
        )
        if present_endpoint_role_fields:
            graph_intent_binding_fields.update(endpoint_role_fields)
        if (
            present_endpoint_role_fields
            and present_endpoint_role_fields != endpoint_role_fields
        ) or set(self.graph_intent_binding) != graph_intent_binding_fields:
            raise ValueError("proposal GraphIntent binding is invalid")
        if set(self.predecessor_binding) != {
            "profile_ref",
            "profile_hash",
            "profile_relation_catalog_hash",
            "taxonomy_ref",
            "taxonomy_hash",
        }:
            raise ValueError("proposal predecessor binding is invalid")
        for field in (
            "graph_intent_ref",
            "relationship_ref",
            "semantic_description",
        ):
            _ascii(self.graph_intent_binding[field], f"graph_intent_binding.{field}")
        _sha256(
            self.graph_intent_binding["graph_intent_hash"],
            "graph_intent_binding.graph_intent_hash",
        )
        for field in ("source_type_refs", "target_type_refs"):
            values = self.graph_intent_binding[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f"graph_intent_binding.{field} must be a nonempty array")
            for value in values:
                _ascii(value, f"graph_intent_binding.{field}[]")
        for field in endpoint_role_fields:
            values = self.graph_intent_binding.get(field, [])
            if present_endpoint_role_fields and (
                not isinstance(values, list) or not values
            ):
                raise ValueError(
                    f"graph_intent_binding.{field} must be a nonempty array"
                )
            for value in values:
                _ascii(value, f"graph_intent_binding.{field}[]")
            if len(values) != len(set(values)) or values != sorted(
                values, key=lambda item: item.encode("utf-8")
            ):
                raise ValueError(
                    f"graph_intent_binding.{field} must contain canonical unique refs"
                )
        if self.graph_intent_binding["direction"] not in {
            "source_to_target",
            "target_to_source",
            "undirected",
        }:
            raise ValueError("proposal GraphIntent direction is invalid")
        if self.graph_intent_binding["preferred_representation"] not in {
            "binary_relation",
            "frame",
            "associative_class",
            "event",
        }:
            raise ValueError("proposal GraphIntent representation is invalid")
        for field in ("profile_ref", "taxonomy_ref"):
            _ascii(self.predecessor_binding[field], f"predecessor_binding.{field}")
        for field in (
            "profile_hash",
            "profile_relation_catalog_hash",
            "taxonomy_hash",
        ):
            _sha256(
                self.predecessor_binding[field],
                f"predecessor_binding.{field}",
            )
        if (
            self.proposed_relation_type.definition_scope != "profile_defined"
            or self.proposed_relation_type.lifecycle_state != "proposed"
        ):
            raise ValueError("proposal requires a proposed profile-defined relation type")
        expected_representation = {
            "binary_relation": ("binary_edge", "binary"),
            "frame": ("frame_role", "frame"),
            "associative_class": ("frame_role", "frame"),
            "event": ("frame_role", "frame"),
        }.get(self.graph_intent_binding["preferred_representation"])
        if expected_representation != (
            self.proposed_relation_type.representation_kind,
            self.proposed_relation_type.projection_shape,
        ):
            raise ValueError("proposal representation mismatches GraphIntent")
        if (
            self.proposed_relation_type.description
            != self.graph_intent_binding["semantic_description"]
        ):
            raise ValueError("proposal description mismatches GraphIntent")
        directed = self.graph_intent_binding["direction"] != "undirected"
        if self.proposed_relation_type.directed != directed:
            raise ValueError("proposal direction mismatches GraphIntent")
        if len(self.proposed_relation_type.source_type_refs) != len(
            self.graph_intent_binding["source_type_refs"]
        ) or len(self.proposed_relation_type.target_type_refs) != len(
            self.graph_intent_binding["target_type_refs"]
        ):
            raise ValueError("proposal endpoint mapping mismatches GraphIntent")
        if (
            self.proposed_relation_type.source_role_refs
            != tuple(self.graph_intent_binding.get("source_role_refs", []))
            or self.proposed_relation_type.target_role_refs
            != tuple(self.graph_intent_binding.get("target_role_refs", []))
        ):
            raise ValueError(
                "proposal endpoint role constraints mismatch GraphIntent"
            )
        if not self.evidence_refs:
            raise ValueError("proposal requires exact EvidenceRefs")
        identities = [item.evidence_ref for item in self.evidence_refs]
        if identities != sorted(identities, key=lambda item: item.encode("utf-8")):
            raise ValueError("proposal EvidenceRefs must be canonically ordered")
        if len(identities) != len(set(identities)):
            raise ValueError("proposal EvidenceRefs contain duplicate identity")
        for evidence in self.evidence_refs:
            if (
                evidence.organization_id != self.organization_id
                or evidence.workspace_id != self.workspace_id
                or evidence.source_identity_ref
                != self.source_binding["source_identity_ref"]
                or evidence.selector_hash != evidence.selector.selector_hash()
                or evidence.selector_schema_version
                != evidence.selector.schema_version
                or evidence.evidence_ref
                != evidence_ref_identity(
                    organization_id=evidence.organization_id,
                    workspace_id=evidence.workspace_id,
                    source_identity_ref=evidence.source_identity_ref,
                    selector_hash=evidence.selector_hash,
                    evidence_role=evidence.evidence_role,
                )
            ):
                raise ValueError("proposal EvidenceRef identity or scope mismatch")
            for field, expected in (
                ("source_hash", self.source_binding["source_hash"]),
                (
                    "canonical_source_ir_ref",
                    self.source_binding["canonical_source_ir_ref"],
                ),
                (
                    "canonical_source_ir_hash",
                    self.source_binding["canonical_source_ir_hash"],
                ),
            ):
                if evidence.provenance.get(field) != expected:
                    raise ValueError(
                        f"proposal EvidenceRef {field} provenance mismatch"
                    )
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.proposal_hash != expected_hash:
            raise ValueError("proposal hash mismatch")
        if self.proposal_ref != f"profile-relation-type-proposal:{expected_hash[7:31]}":
            raise ValueError("proposal ref mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("proposal_ref")
        payload.pop("proposal_hash")
        return payload

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def create(cls, content: dict[str, Any]) -> "ProfileRelationTypeProposalV1":
        payload = dict(content)
        payload["schema_version"] = "profile_relation_type_proposal.v1"
        payload["lifecycle_state"] = "candidate"
        payload["required_lifecycle"] = (
            "versioned_successor_profile_catalog_and_schema_decision_v3"
        )
        payload["proposed_relation_type"] = KnowledgeRelationType.model_validate(
            payload["proposed_relation_type"]
        )
        payload["evidence_refs"] = tuple(sorted(
            (
                item
                if isinstance(item, EvidenceRef)
                else EvidenceRef.model_validate(item)
                for item in payload["evidence_refs"]
            ),
            key=lambda item: (
                item.evidence_ref
            ).encode("utf-8"),
        ))
        if isinstance(payload["occurred_at"], str):
            payload["occurred_at"] = datetime.fromisoformat(
                payload["occurred_at"].replace("Z", "+00:00")
            )
        payload.pop("proposal_ref", None)
        payload.pop("proposal_hash", None)
        hash_payload = cls.model_construct(
            proposal_ref="pending",
            proposal_hash="sha256:" + "0" * 64,
            **payload,
        ).model_dump(mode="json", by_alias=True)
        hash_payload.pop("proposal_ref")
        hash_payload.pop("proposal_hash")
        digest = content_hash(hash_payload)
        return cls.model_validate(
            {
                **payload,
                "proposal_ref": f"profile-relation-type-proposal:{digest[:24]}",
                "proposal_hash": f"sha256:{digest}",
            }
        )


__all__ = ["ProfileRelationTypeProposalV1"]
