"""Immutable governed Gate-1 compiler decision contract."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from datetime import datetime, timezone
import re
from typing import Any

from contracts.corpus_graph_schema import (
    RelationTypeCandidate,
    V4_SCHEMA_VERSION as CANDIDATE_V4_SCHEMA_VERSION,
)
from eos.artifact_hashing import content_hash


SCHEMA_VERSION = "schema_decision.v1"
V2_SCHEMA_VERSION = "schema_decision.v2"
V3_SCHEMA_VERSION = "schema_decision.v3"
_V2_FIELDS = {
    "schema_version",
    "schema_decision_id",
    "schema_decision_hash",
    "source_binding",
    "candidate_binding",
    "approved_schema",
    "extraction_plan",
    "source_evidence_refs",
    "semantica_binding",
    "decided_at",
    "predecessor_schema_decision_ref",
    "successor_schema_decision_refs",
    "graph_intent_binding",
}
_V3_FIELDS = _V2_FIELDS | {"admitted_relation_type_bindings"}
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELATION_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ADMITTED_BINDING_FIELDS = {
    "relation_type_id",
    "taxonomy_ref",
    "taxonomy_hash",
    "profile_ref",
    "profile_hash",
    "profile_relation_catalog_hash",
}
_PROPOSAL_LINEAGE_FIELDS = {
    "profile_relation_type_proposal_ref",
    "profile_relation_type_proposal_hash",
}
_V3_CANDIDATE_BINDING_FIELDS = {
    "candidate_id",
    "candidate_hash",
    "candidate_schema_version",
    "gate_outcome",
    "operating_mode",
    "decided_by",
    "actor_ref",
    "policy_ref",
    "rationale_ref",
    "decision_evidence_refs",
    "review_ref",
}
_GOVERNED_COMPOSITION_TOKEN = object()


def _dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a nonempty object")
    return dict(value)


def _utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("decided_at must be timezone-aware UTC ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("decided_at must be timezone-aware UTC ISO-8601")
    return value


def _sha256_ref(value: Any, field: str) -> str:
    text = str(value or "")
    if _SHA256_REF.fullmatch(text) is None:
        raise ValueError(f"{field} must be a sha256 ref")
    return text


def _canonical_admitted_bindings(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("admitted_relation_type_bindings must not be empty")
    canonical: list[dict[str, Any]] = []
    relation_ids: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) not in {
            frozenset(_ADMITTED_BINDING_FIELDS),
            frozenset(_ADMITTED_BINDING_FIELDS | _PROPOSAL_LINEAGE_FIELDS),
        }:
            raise ValueError("admitted relation binding must contain exact fields")
        binding = dict(item)
        relation_type_id = binding.get("relation_type_id")
        if (
            not isinstance(relation_type_id, str)
            or not relation_type_id.isascii()
            or _RELATION_IDENTIFIER.fullmatch(relation_type_id) is None
        ):
            raise ValueError("admitted relation_type_id is invalid")
        if not str(binding.get("taxonomy_ref") or "").strip():
            raise ValueError("admitted taxonomy_ref is required")
        _sha256_ref(binding.get("taxonomy_hash"), "admitted taxonomy_hash")
        profile_pins = (
            binding.get("profile_ref"),
            binding.get("profile_hash"),
            binding.get("profile_relation_catalog_hash"),
        )
        if any(pin is None for pin in profile_pins) and any(
            pin is not None for pin in profile_pins
        ):
            raise ValueError("admitted profile pins must be all null or all non-null")
        if all(pin is not None for pin in profile_pins):
            if not str(binding["profile_ref"]).strip():
                raise ValueError("admitted profile_ref is required")
            _sha256_ref(binding["profile_hash"], "admitted profile_hash")
            _sha256_ref(
                binding["profile_relation_catalog_hash"],
                "admitted profile_relation_catalog_hash",
            )
        if _PROPOSAL_LINEAGE_FIELDS <= set(binding):
            proposal_ref = binding["profile_relation_type_proposal_ref"]
            if (
                not isinstance(proposal_ref, str)
                or re.fullmatch(
                    r"profile-relation-type-proposal:[0-9a-f]{24}",
                    proposal_ref,
                )
                is None
            ):
                raise ValueError("admitted profile relation proposal_ref is invalid")
            _sha256_ref(
                binding["profile_relation_type_proposal_hash"],
                "admitted profile relation proposal_hash",
            )
            if any(pin is None for pin in profile_pins):
                raise ValueError(
                    "admitted profile relation proposal requires profile pins"
                )
        canonical.append(binding)
        relation_ids.append(relation_type_id)
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("admitted relation bindings contain duplicate relation_type_id")
    return tuple(
        sorted(
            canonical,
            key=lambda item: item["relation_type_id"].encode("utf-8"),
        )
    )


@dataclass(frozen=True)
class SchemaDecision:
    schema_decision_id: str
    schema_decision_hash: str
    source_binding: dict[str, Any]
    candidate_binding: dict[str, Any]
    approved_schema: dict[str, Any]
    extraction_plan: tuple[dict[str, Any], ...]
    source_evidence_refs: tuple[str, ...]
    semantica_binding: dict[str, Any]
    decided_at: str
    predecessor_schema_decision_ref: str | None = None
    successor_schema_decision_refs: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    graph_intent_binding: dict[str, Any] | None = None
    admitted_relation_type_bindings: tuple[dict[str, Any], ...] = ()
    taxonomy_service: InitVar[Any | None] = None
    governed_composition_token: InitVar[object | None] = None

    def __post_init__(
        self,
        taxonomy_service: Any | None,
        governed_composition_token: object | None,
    ) -> None:
        if self.schema_version not in {
            SCHEMA_VERSION,
            V2_SCHEMA_VERSION,
            V3_SCHEMA_VERSION,
        }:
            raise ValueError("schema_version is unsupported")
        if not self.schema_decision_id.startswith("schema-decision-"):
            raise ValueError("schema_decision_id is invalid")
        if self.schema_version in {V2_SCHEMA_VERSION, V3_SCHEMA_VERSION}:
            _sha256_ref(self.schema_decision_hash, "schema_decision_hash")
        elif not self.schema_decision_hash.startswith("sha256:"):
            raise ValueError("schema_decision_hash must be a sha256 ref")
        _dict(self.source_binding, "source_binding")
        binding = _dict(self.candidate_binding, "candidate_binding")
        _dict(self.approved_schema, "approved_schema")
        _dict(self.semantica_binding, "semantica_binding")
        if binding.get("gate_outcome") not in {"accepted", "edited"}:
            raise ValueError("gate_outcome must be accepted or edited")
        if binding.get("operating_mode") not in {"assisted", "auto_pilot"}:
            raise ValueError("operating_mode must be assisted or auto_pilot")
        if binding.get("decided_by") not in {"user", "system"}:
            raise ValueError("decided_by must be user or system")
        if not str(binding.get("actor_ref") or "").strip():
            raise ValueError("actor_ref is required")
        if binding["operating_mode"] == "auto_pilot":
            for field in ("policy_ref", "rationale_ref"):
                if not str(binding.get(field) or "").strip():
                    raise ValueError(f"auto_pilot requires {field}")
            if not binding.get("decision_evidence_refs"):
                raise ValueError("auto_pilot requires decision_evidence_refs")
            if binding["decided_by"] != "system":
                raise ValueError("auto_pilot decisions must be decided_by system")
        if self.schema_version in {V2_SCHEMA_VERSION, V3_SCHEMA_VERSION}:
            intent = _dict(self.graph_intent_binding, "graph_intent_binding")
            if set(intent) != {
                "graph_intent_ref",
                "graph_intent_hash",
                "review_state",
                "source_identity_ref",
            }:
                raise ValueError("graph_intent_binding is invalid for v2")
            if intent.get("review_state") not in {"accepted", "edited"}:
                raise ValueError("graph_intent_binding review_state is invalid")
            _sha256_ref(intent.get("graph_intent_hash"), "graph_intent_hash")
            _sha256_ref(binding.get("candidate_hash"), "candidate_hash")
            for field in ("source_hash", "canonical_source_ir_hash"):
                _sha256_ref(self.source_binding.get(field), f"source_binding.{field}")
            if self.source_binding.get("source_identity_ref") != intent.get(
                "source_identity_ref"
            ):
                raise ValueError("decision GraphIntent source identity mismatch")
        if self.schema_version == V3_SCHEMA_VERSION:
            if set(binding) != _V3_CANDIDATE_BINDING_FIELDS:
                raise ValueError("decision v3 requires an exact candidate binding")
            if binding.get("candidate_schema_version") != CANDIDATE_V4_SCHEMA_VERSION:
                raise ValueError("schema decision v3 requires candidate v4")
            candidate_hash = _sha256_ref(
                binding.get("candidate_hash"), "candidate_hash"
            )
            if binding.get("candidate_id") != f"cgsc_{candidate_hash[7:31]}":
                raise ValueError("candidate_id does not match pinned candidate_hash")
            canonical_bindings = _canonical_admitted_bindings(
                self.admitted_relation_type_bindings
            )
            if self.admitted_relation_type_bindings != canonical_bindings:
                raise ValueError("admitted relation bindings are not canonical")
            approved_relations = self.approved_schema.get("relation_types")
            if not isinstance(approved_relations, list) or not approved_relations:
                raise ValueError("approved_schema relation_types must not be empty")
            approved_bindings = []
            for relation in approved_relations:
                if not isinstance(relation, dict) or not isinstance(
                    relation.get("relation_type_candidate"), dict
                ):
                    raise ValueError(
                        "approved relation must contain relation_type_candidate"
                    )
                candidate = RelationTypeCandidate.model_validate(
                    relation["relation_type_candidate"]
                )
                candidate.validate_source_binding(self.source_binding)
                if relation.get("type_id") != candidate.relation_type_id:
                    raise ValueError(
                        "approved relation type_id does not match relation candidate"
                    )
                candidate.validate_outer_topology(relation)
                if relation["relation_type_candidate"] != candidate.as_dict():
                    raise ValueError("approved relation candidate is not canonical")
                approved_bindings.append(
                    {
                        field: candidate.as_dict()[field]
                        for field in (
                            _ADMITTED_BINDING_FIELDS
                            | (
                                _PROPOSAL_LINEAGE_FIELDS
                                if candidate.profile_relation_type_proposal_ref
                                is not None
                                else set()
                            )
                        )
                    }
                )
            if _canonical_admitted_bindings(approved_bindings) != canonical_bindings:
                raise ValueError("approved relation taxonomy admission mismatch")
        _utc_timestamp(self.decided_at)
        expected = f"sha256:{content_hash(self._hash_payload())}"
        if self.schema_decision_hash != expected:
            raise ValueError("schema_decision_hash does not match decision content")
        if self.schema_decision_id != f"schema-decision-{expected[7:31]}":
            raise ValueError("schema_decision_id does not match decision content")
        if self.schema_version == V3_SCHEMA_VERSION:
            if governed_composition_token is not _GOVERNED_COMPOSITION_TOKEN:
                raise ValueError(
                    "decision v3 must be constructed by the exact governed taxonomy service"
                )
            self._validate_governed_admissions(taxonomy_service)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "schema_decision_id": self.schema_decision_id,
            "schema_decision_hash": self.schema_decision_hash,
            "source_binding": dict(self.source_binding),
            "candidate_binding": dict(self.candidate_binding),
            "approved_schema": dict(self.approved_schema),
            "extraction_plan": [dict(item) for item in self.extraction_plan],
            "source_evidence_refs": list(self.source_evidence_refs),
            "semantica_binding": dict(self.semantica_binding),
            "decided_at": self.decided_at,
            "predecessor_schema_decision_ref": self.predecessor_schema_decision_ref,
            "successor_schema_decision_refs": list(
                self.successor_schema_decision_refs
            ),
        }
        if self.schema_version in {V2_SCHEMA_VERSION, V3_SCHEMA_VERSION}:
            payload["graph_intent_binding"] = dict(self.graph_intent_binding or {})
        if self.schema_version == V3_SCHEMA_VERSION:
            payload["admitted_relation_type_bindings"] = [
                dict(item) for item in self.admitted_relation_type_bindings
            ]
        return payload

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("schema_decision_id")
        payload.pop("schema_decision_hash")
        return payload

    def _validate_governed_admissions(self, taxonomy_service: Any) -> None:
        from services.knowledge_relation_taxonomy_service import (
            KnowledgeRelationTaxonomyService,
        )

        if type(taxonomy_service) is not KnowledgeRelationTaxonomyService:
            raise ValueError(
                "decision v3 requires the exact governed taxonomy service"
            )
        proposal_origin = (
            "system"
            if self.candidate_binding.get("decided_by") == "system"
            else "human"
        )
        for relation in self.approved_schema["relation_types"]:
            candidate = RelationTypeCandidate.model_validate(
                relation["relation_type_candidate"]
            )
            KnowledgeRelationTaxonomyService.resolve_candidate_admission(
                taxonomy_service,
                candidate,
                proposal_origin=proposal_origin,
            )
            KnowledgeRelationTaxonomyService.validate_candidate_proposal_lineage(
                taxonomy_service,
                candidate,
                source_binding=self.source_binding,
                graph_intent_binding=dict(self.graph_intent_binding or {}),
            )

    @classmethod
    def create(cls, content: dict[str, Any]) -> "SchemaDecision":
        payload = dict(content)
        payload["schema_version"] = SCHEMA_VERSION
        payload.pop("schema_decision_id", None)
        payload.pop("schema_decision_hash", None)
        digest = content_hash(payload)
        return cls.from_dict(
            {
                **payload,
                "schema_decision_id": f"schema-decision-{digest[:24]}",
                "schema_decision_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def create_v2(cls, content: dict[str, Any]) -> "SchemaDecision":
        payload = dict(content)
        payload["schema_version"] = V2_SCHEMA_VERSION
        payload.pop("schema_decision_id", None)
        payload.pop("schema_decision_hash", None)
        digest = content_hash(payload)
        return cls.from_dict(
            {
                **payload,
                "schema_decision_id": f"schema-decision-{digest[:24]}",
                "schema_decision_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def create_v3(
        cls,
        content: dict[str, Any],
        **_caller_authority: Any,
    ) -> "SchemaDecision":
        del content, _caller_authority
        raise ValueError(
            "decision v3 must be constructed by the exact governed taxonomy service"
        )

    @classmethod
    def _create_v3_with_governed_service(
        cls,
        content: dict[str, Any],
        *,
        taxonomy_service: Any,
    ) -> "SchemaDecision":
        payload = dict(content)
        payload["schema_version"] = V3_SCHEMA_VERSION
        payload.pop("schema_decision_id", None)
        payload.pop("schema_decision_hash", None)
        payload["admitted_relation_type_bindings"] = [
            dict(item)
            for item in _canonical_admitted_bindings(
                payload.get("admitted_relation_type_bindings")
            )
        ]
        digest = content_hash(payload)
        return cls._from_dict(
            {
                **payload,
                "schema_decision_id": f"schema-decision-{digest[:24]}",
                "schema_decision_hash": f"sha256:{digest}",
            },
            taxonomy_service=taxonomy_service,
            governed_composition_token=_GOVERNED_COMPOSITION_TOKEN,
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "SchemaDecision":
        if payload.get("schema_version") == V3_SCHEMA_VERSION:
            raise ValueError(
                "decision v3 replay requires the exact governed taxonomy service"
            )
        return cls._from_dict(payload)

    @classmethod
    def _replay_v3_with_governed_service(
        cls,
        payload: dict[str, Any],
        *,
        taxonomy_service: Any,
    ) -> "SchemaDecision":
        return cls._from_dict(
            payload,
            taxonomy_service=taxonomy_service,
            governed_composition_token=_GOVERNED_COMPOSITION_TOKEN,
        )

    @classmethod
    def _from_dict(
        cls,
        payload: dict[str, Any],
        *,
        taxonomy_service: Any | None = None,
        governed_composition_token: object | None = None,
    ) -> "SchemaDecision":
        if payload.get("schema_version") == V2_SCHEMA_VERSION and set(payload) != (
            _V2_FIELDS
        ):
            raise ValueError("decision payload must contain exact v2 fields")
        if payload.get("schema_version") == V3_SCHEMA_VERSION and set(payload) != (
            _V3_FIELDS
        ):
            raise ValueError("decision payload must contain exact v3 fields")
        decision = cls(
            schema_version=str(payload.get("schema_version") or ""),
            schema_decision_id=str(payload.get("schema_decision_id") or ""),
            schema_decision_hash=str(payload.get("schema_decision_hash") or ""),
            source_binding=_dict(payload.get("source_binding"), "source_binding"),
            candidate_binding=_dict(
                payload.get("candidate_binding"), "candidate_binding"
            ),
            approved_schema=_dict(payload.get("approved_schema"), "approved_schema"),
            extraction_plan=tuple(
                dict(item) for item in (payload.get("extraction_plan") or [])
            ),
            source_evidence_refs=tuple(
                str(ref) for ref in (payload.get("source_evidence_refs") or [])
            ),
            semantica_binding=_dict(
                payload.get("semantica_binding"), "semantica_binding"
            ),
            decided_at=str(payload.get("decided_at") or ""),
            predecessor_schema_decision_ref=(
                str(payload["predecessor_schema_decision_ref"])
                if payload.get("predecessor_schema_decision_ref") is not None
                else None
            ),
            successor_schema_decision_refs=tuple(
                str(ref)
                for ref in (payload.get("successor_schema_decision_refs") or [])
            ),
            graph_intent_binding=(
                dict(payload["graph_intent_binding"])
                if isinstance(payload.get("graph_intent_binding"), dict)
                else None
            ),
            admitted_relation_type_bindings=tuple(
                dict(item)
                for item in (payload.get("admitted_relation_type_bindings") or [])
            ),
            taxonomy_service=taxonomy_service,
            governed_composition_token=governed_composition_token,
        )
        return decision


__all__ = [
    "SCHEMA_VERSION",
    "V2_SCHEMA_VERSION",
    "V3_SCHEMA_VERSION",
    "SchemaDecision",
]
