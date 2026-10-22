"""Minimal CFIUS matter boundary contracts for the factual authority slice.

These contracts intentionally do not create legal authority or a CFIUS decision
engine. They only pin matter identity, factual graph content, and the
confidentiality projection used to present matter facts without mutating the
internal evidence identity that the governed kernel already owns.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_datetimes_are_aware_utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


def _unique_nonempty_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return tuple(value) if isinstance(value, tuple) else tuple()
    items = tuple(str(item) for item in value)
    if not items:
        return ()
    if any(not item.strip() for item in items):
        raise ValueError("reference collections must be nonempty and free of blanks")
    if len(items) != len(set(items)):
        raise ValueError("reference collections must be unique")
    return items


def _assert_same_matter_ref(ref: str, matter_id: str) -> None:
    if "matter-" in ref.lower() or "matter:" in ref.lower():
        matter_slug = matter_id.split(":")[-1].lower()
        if matter_slug not in ref.lower():
            raise ValueError(f"evidence ref {ref!r} does not belong to matter {matter_id!r}")


class MatterV1(_Contract):
    schema_version: Literal["cfius.matter.v1"] = "cfius.matter.v1"
    matter_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    status: Literal["OPEN", "ACTIVE", "CLOSED", "WITHDRAWN"]
    created_at: datetime
    source_refs: tuple[str, ...] = ()
    confidentiality_policy_ref: str = Field(min_length=1)
    matter_ref: str = Field(min_length=1)
    matter_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("source_refs", mode="before")
    @classmethod
    def sort_and_validate_sources(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        return tuple(sorted(refs))

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "matter_ref": "pending",
                "matter_hash": "sha256:" + "0" * 64,
                **values,
            },
            context={"cfius_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"matter_ref", "matter_hash"})
        )
        return cls.model_validate(
            {
                "matter_ref": f"matter:{digest}",
                "matter_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"matter_ref", "matter_hash"}),
            }
        )

    @model_validator(mode="after")
    def bind_exact_identity(self, info):
        if info.context and info.context.get("cfius_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"matter_ref", "matter_hash"})
        )
        if (
            self.matter_ref != f"matter:{digest}"
            or self.matter_hash != f"sha256:{digest}"
        ):
            raise ValueError("matter ref/hash must bind exact content")
        return self


class SourceAuthorityPinV1(_Contract):
    source_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    source_kind: Literal["regulation", "faq", "appendix", "supporting"]
    canonical_ref: str = Field(min_length=1)
    effective_from: datetime
    effective_until: datetime

    @model_validator(mode="after")
    def effectivity_window_is_valid(self):
        if self.effective_until <= self.effective_from:
            raise ValueError("source pin effective_until must be after effective_from")
        return self


class CFIUSBenchmarkManifestV1(_Contract):
    schema_version: Literal["cfius.benchmark_manifest.v1"] = "cfius.benchmark_manifest.v1"
    benchmark_id: str = Field(min_length=1)
    benchmark_ref: str = Field(min_length=1)
    benchmark_hash: str = Field(pattern=SHA256_PATTERN)
    source_pins: tuple[SourceAuthorityPinV1, ...] = Field(min_length=1)
    primary_fixture_ids: tuple[str, ...] = Field(min_length=1)
    freeze_status: Literal["FROZEN", "DRAFT"] = "FROZEN"
    created_at: datetime

    @field_validator("source_pins", mode="before")
    @classmethod
    def validate_source_pins(cls, value: object) -> tuple[SourceAuthorityPinV1, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("benchmark manifest requires at least one source pin")
        return tuple(value)

    @field_validator("primary_fixture_ids", mode="before")
    @classmethod
    def validate_fixture_ids(cls, value: object) -> tuple[str, ...]:
        fixture_ids = _unique_nonempty_strings(value)
        if not fixture_ids:
            raise ValueError("benchmark manifest requires at least one fixture id")
        return fixture_ids

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "benchmark_ref": "pending",
                "benchmark_hash": "sha256:" + "0" * 64,
                "created_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_benchmark_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"benchmark_ref", "benchmark_hash"})
        )
        return cls.model_validate(
            {
                "benchmark_ref": f"benchmark:{digest}",
                "benchmark_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"benchmark_ref", "benchmark_hash"}),
            }
        )

    @model_validator(mode="after")
    def bind_exact_benchmark_identity(self, info):
        if self.benchmark_id != "CFIUS_PRECHECK_BENCHMARK@1":
            raise ValueError("benchmark_id must be CFIUS_PRECHECK_BENCHMARK@1")
        if info.context and info.context.get("cfius_benchmark_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"benchmark_ref", "benchmark_hash"})
        )
        expected_ref = f"benchmark:{digest}"
        if self.benchmark_ref != expected_ref or self.benchmark_hash != f"sha256:{digest}":
            raise ValueError("benchmark ref/hash must bind exact content")
        if not self.source_pins or not self.primary_fixture_ids:
            raise ValueError("benchmark manifest requires fixture and source pins")
        return self


class TransactionGraphNodeV1(_Contract):
    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    matter_id: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def validate_evidence_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("transaction graph nodes require exact evidence refs")
        return refs

    @model_validator(mode="after")
    def matter_scope_is_exact(self):
        for ref in self.evidence_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        return self


class TransactionGraphEdgeV1(_Contract):
    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    matter_id: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def validate_evidence_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("transaction graph edges require exact evidence refs")
        return refs

    @model_validator(mode="after")
    def matter_scope_is_exact(self):
        for ref in self.evidence_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        return self


class TransactionGraphV1(_Contract):
    schema_version: Literal["cfius.transaction_graph.v1"] = "cfius.transaction_graph.v1"
    graph_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    matter_id: str = Field(min_length=1)
    nodes: tuple[TransactionGraphNodeV1, ...] = ()
    edges: tuple[TransactionGraphEdgeV1, ...] = ()
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    graph_ref: str = Field(min_length=1)
    graph_hash: str = Field(pattern=SHA256_PATTERN)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def validate_evidence_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("transaction graph requires evidence refs")
        return refs

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "graph_id": "pending",
                "graph_ref": "pending",
                "graph_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_graph_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"graph_id", "graph_ref", "graph_hash"})
        )
        return cls.model_validate(
            {
                "graph_id": f"transaction_graph:{digest}",
                "graph_ref": f"transaction_graph:{digest}",
                "graph_hash": f"sha256:{digest}",
                **draft.model_dump(
                    mode="python",
                    exclude={"graph_id", "graph_ref", "graph_hash"},
                ),
            }
        )

    @model_validator(mode="after")
    def validate_factual_boundary(self, info):
        if info.context and info.context.get("cfius_graph_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"graph_id", "graph_ref", "graph_hash"})
        )
        if (
            self.graph_id != f"transaction_graph:{digest}"
            or self.graph_ref != f"transaction_graph:{digest}"
            or self.graph_hash != f"sha256:{digest}"
        ):
            raise ValueError("transaction graph id/ref/hash must bind exact content")

        node_ids = tuple(node.node_id for node in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("transaction graph node ids must be unique")
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("transaction graph edge ids must be unique")

        node_set = set(node_ids)
        for edge in self.edges:
            if edge.source_id not in node_set or edge.target_id not in node_set:
                raise ValueError("transaction graph edges must reference existing nodes")
            if edge.matter_id != self.matter_id:
                raise ValueError("transaction graph edges must stay inside one matter")
            for ref in edge.evidence_refs:
                _assert_same_matter_ref(ref, self.matter_id)
        for node in self.nodes:
            if node.matter_id != self.matter_id:
                raise ValueError("transaction graph nodes must stay inside one matter")
            for ref in node.evidence_refs:
                _assert_same_matter_ref(ref, self.matter_id)
        for ref in self.evidence_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        return self

    def redacted_projection(self) -> "TransactionGraphV1":
        redacted_nodes = tuple(
            TransactionGraphNodeV1(
                node_id=node.node_id,
                node_type=node.node_type,
                label="[REDACTED]",
                matter_id=node.matter_id,
                evidence_refs=node.evidence_refs,
            )
            for node in self.nodes
        )
        return TransactionGraphV1.create(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            matter_id=self.matter_id,
            nodes=redacted_nodes,
            edges=self.edges,
            evidence_refs=self.evidence_refs,
            recorded_at=self.recorded_at,
        )


class _MatterGraphV1(_Contract):
    graph_id: str = Field(min_length=1)
    graph_ref: str = Field(min_length=1)
    graph_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    matter_id: str = Field(min_length=1)
    graph_type: str = Field(min_length=1)
    nodes: tuple[str, ...] = ()
    edges: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def validate_evidence_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("matter graphs require evidence refs")
        return refs

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "graph_id": "pending",
                "graph_ref": "pending",
                "graph_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_graph_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"graph_id", "graph_ref", "graph_hash"})
        )
        return cls.model_validate(
            {
                "graph_id": f"{draft.graph_type.lower()}_graph:{digest}",
                "graph_ref": f"{draft.graph_type.lower()}_graph:{digest}",
                "graph_hash": f"sha256:{digest}",
                **draft.model_dump(
                    mode="python",
                    exclude={"graph_id", "graph_ref", "graph_hash"},
                ),
            }
        )

    @model_validator(mode="after")
    def exact_graph_identity_and_scope(self, info):
        if info.context and info.context.get("cfius_graph_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"graph_id", "graph_ref", "graph_hash"})
        )
        expected_ref = f"{self.graph_type.lower()}_graph:{digest}"
        if self.graph_id != expected_ref or self.graph_ref != expected_ref:
            raise ValueError("graph id/ref must bind exact content")
        if self.graph_hash != f"sha256:{digest}":
            raise ValueError("graph hash must bind exact content")
        for ref in self.evidence_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        return self


class NormativeGraphV1(_MatterGraphV1):
    schema_version: Literal["cfius.normative_graph.v1"] = "cfius.normative_graph.v1"
    graph_type: Literal["NORMATIVE"] = "NORMATIVE"

    @model_validator(mode="after")
    def normative_type_is_exact(self):
        if self.graph_type != "NORMATIVE":
            raise ValueError("NormativeGraphV1 only accepts NORMATIVE graph_type")
        return self


class InstitutionalActionGraphV1(_MatterGraphV1):
    schema_version: Literal["cfius.institutional_action_graph.v1"] = (
        "cfius.institutional_action_graph.v1"
    )
    graph_type: Literal["INSTITUTIONAL_ACTION"] = "INSTITUTIONAL_ACTION"

    @model_validator(mode="after")
    def institutional_type_is_exact(self):
        if self.graph_type != "INSTITUTIONAL_ACTION":
            raise ValueError("InstitutionalActionGraphV1 only accepts INSTITUTIONAL_ACTION graph_type")
        return self


class EvidenceSufficiencyAssessmentV1(_Contract):
    schema_version: Literal["cfius.evidence_sufficiency_assessment.v1"] = (
        "cfius.evidence_sufficiency_assessment.v1"
    )
    assessment_id: str = Field(min_length=1)
    assessment_ref: str = Field(min_length=1)
    assessment_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    required_fact_slots: tuple[str, ...] = Field(min_length=1)
    satisfied_refs: tuple[str, ...] = ()
    missing_refs: tuple[str, ...] = ()
    conflicting_refs: tuple[str, ...] = ()
    disposition: Literal["SUFFICIENT", "INSUFFICIENT", "UNKNOWN"]
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("required_fact_slots", "satisfied_refs", "missing_refs", "conflicting_refs", mode="before")
    @classmethod
    def validate_ref_collections(cls, value: object) -> tuple[str, ...]:
        return _unique_nonempty_strings(value)

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "assessment_id": "pending",
                "assessment_ref": "pending",
                "assessment_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_assessment_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"assessment_id", "assessment_ref", "assessment_hash"})
        )
        return cls.model_validate(
            {
                "assessment_id": f"evidence_assessment:{digest}",
                "assessment_ref": f"evidence_assessment:{digest}",
                "assessment_hash": f"sha256:{digest}",
                **draft.model_dump(
                    mode="python",
                    exclude={"assessment_id", "assessment_ref", "assessment_hash"},
                ),
            }
        )

    @model_validator(mode="after")
    def evaluate_sufficiency(self, info):
        if info.context and info.context.get("cfius_assessment_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"assessment_id", "assessment_ref", "assessment_hash"})
        )
        expected_ref = f"evidence_assessment:{digest}"
        if self.assessment_id != expected_ref or self.assessment_ref != expected_ref:
            raise ValueError("assessment id/ref must bind exact content")
        if self.assessment_hash != f"sha256:{digest}":
            raise ValueError("assessment hash must bind exact content")
        for ref in self.missing_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        for ref in self.satisfied_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        for ref in self.conflicting_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        if self.disposition == "SUFFICIENT" and (
            not self.required_fact_slots
            or not self.satisfied_refs
            or self.missing_refs
            or len(self.satisfied_refs) < len(self.required_fact_slots)
        ):
            raise ValueError("SUFFICIENT requires required fact slots to be satisfied")
        return self


class MatterRemediationRequestV1(_Contract):
    schema_version: Literal["cfius.matter_remediation_request.v1"] = (
        "cfius.matter_remediation_request.v1"
    )
    request_id: str = Field(min_length=1)
    request_ref: str = Field(min_length=1)
    request_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    required_evidence_refs: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("required_evidence_refs", mode="before")
    @classmethod
    def validate_required_evidence_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("remediation requests require evidence refs")
        return refs

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "request_id": "pending",
                "request_ref": "pending",
                "request_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_remediation_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"request_id", "request_ref", "request_hash"})
        )
        return cls.model_validate(
            {
                "request_id": f"remediation:{digest}",
                "request_ref": f"remediation:{digest}",
                "request_hash": f"sha256:{digest}",
                **draft.model_dump(
                    mode="python",
                    exclude={"request_id", "request_ref", "request_hash"},
                ),
            }
        )

    @model_validator(mode="after")
    def exact_request_identity(self, info):
        if info.context and info.context.get("cfius_remediation_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"request_id", "request_ref", "request_hash"})
        )
        expected_ref = f"remediation:{digest}"
        if self.request_id != expected_ref or self.request_ref != expected_ref:
            raise ValueError("remediation request id/ref must bind exact content")
        if self.request_hash != f"sha256:{digest}":
            raise ValueError("remediation request hash must bind exact content")
        for ref in self.required_evidence_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        return self


class PredicateEvaluationV1(_Contract):
    schema_version: Literal["cfius.predicate_evaluation.v1"] = "cfius.predicate_evaluation.v1"
    predicate_id: str = Field(min_length=1)
    predicate_ref: str = Field(min_length=1)
    predicate_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    execution_mode: Literal["COMPUTABLE", "INTERPRETED", "HYBRID"]
    execution_status: Literal[
        "SUCCESS",
        "STRUCTURAL_UNKNOWN",
        "SEMANTIC_UNKNOWN",
        "PROVIDER_FAILURE",
    ] = "SUCCESS"
    semantic_result: Literal["TRUE", "FALSE", "UNKNOWN"]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def validate_evidence_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("predicate evaluations require evidence refs")
        return refs

    @model_validator(mode="after")
    def execution_status_must_be_distinct_from_structural_unknown(self):
        if self.execution_status in {"STRUCTURAL_UNKNOWN", "SEMANTIC_UNKNOWN", "PROVIDER_FAILURE"}:
            if self.semantic_result != "UNKNOWN":
                raise ValueError("unknown-like execution statuses require semantic_result=UNKNOWN")
        if self.execution_status == "SUCCESS" and self.semantic_result not in {"TRUE", "FALSE", "UNKNOWN"}:
            raise ValueError("execution_status SUCCESS requires a valid semantic_result")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "predicate_ref": "pending",
                "predicate_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_predicate_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"predicate_ref", "predicate_hash"})
        )
        return cls.model_validate(
            {
                "predicate_ref": f"predicate:{digest}",
                "predicate_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"predicate_ref", "predicate_hash"}),
            }
        )

    @model_validator(mode="after")
    def bind_exact_identity(self, info):
        if info.context and info.context.get("cfius_predicate_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"predicate_ref", "predicate_hash"})
        )
        if self.predicate_ref != f"predicate:{digest}" or self.predicate_hash != f"sha256:{digest}":
            raise ValueError("predicate ref/hash must bind exact content")
        for ref in self.evidence_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        return self


class DecisionRunV1(_Contract):
    schema_version: Literal["cfius.decision_run.v1"] = "cfius.decision_run.v1"
    run_id: str = Field(min_length=1)
    run_ref: str = Field(min_length=1)
    run_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    predicate_results: tuple[PredicateEvaluationV1, ...] = Field(min_length=1)
    decision_status: Literal["UNKNOWN", "REVIEW_REQUIRED", "NO_JURISDICTION", "FILED"]
    verification_ref: str = Field(min_length=1)
    previous_run_ref: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("verification_ref")
    @classmethod
    def validate_verification_ref(cls, value: str) -> str:
        if not value.startswith("decision_verification:"):
            raise ValueError("verification_ref must be a decision_verification_ref")
        return value

    @field_validator("previous_run_ref")
    @classmethod
    def validate_previous_run_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith("decision_run:"):
            raise ValueError("previous_run_ref must be a decision_run_ref")
        return value

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "run_id": "pending",
                "run_ref": "pending",
                "run_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_run_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"run_id", "run_ref", "run_hash"})
        )
        return cls.model_validate(
            {
                "run_id": f"decision_run:{digest}",
                "run_ref": f"decision_run:{digest}",
                "run_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"run_id", "run_ref", "run_hash"}),
            }
        )

    @model_validator(mode="after")
    def exact_run_identity(self, info):
        if info.context and info.context.get("cfius_run_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"run_id", "run_ref", "run_hash"})
        )
        if self.run_id != f"decision_run:{digest}" or self.run_ref != f"decision_run:{digest}":
            raise ValueError("decision run id/ref must bind exact content")
        if self.run_hash != f"sha256:{digest}":
            raise ValueError("decision run hash must bind exact content")
        if self.previous_run_ref is not None:
            if self.previous_run_ref == self.run_ref:
                raise ValueError("decision run cannot reference itself")
            if not self.previous_run_ref.startswith("decision_run:"):
                raise ValueError("decision run previous_run_ref must be a decision_run_ref")
        for result in self.predicate_results:
            if result.matter_id != self.matter_id:
                raise ValueError("decision runs must remain within one matter")
        predicate_ids = tuple(result.predicate_id for result in self.predicate_results)
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ValueError("decision run predicate results must have unique predicate_ids")
        return self


class MatterDispositionV1(_Contract):
    schema_version: Literal["cfius.matter_disposition.v1"] = "cfius.matter_disposition.v1"
    disposition_id: str = Field(min_length=1)
    disposition_ref: str = Field(min_length=1)
    disposition_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    decision_run_ref: str = Field(min_length=1)
    disposition: Literal["UNKNOWN", "REVIEW_REQUIRED", "NO_JURISDICTION", "FILED"]
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("decision_run_ref")
    @classmethod
    def validate_decision_run_ref(cls, value: str) -> str:
        if not value.startswith("decision_run:"):
            raise ValueError("decision_run_ref must be a decision_run_ref")
        return value

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def validate_evidence_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("matter disposition requires evidence refs")
        return refs

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "disposition_id": "pending",
                "disposition_ref": "pending",
                "disposition_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_disposition_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"disposition_id", "disposition_ref", "disposition_hash"})
        )
        return cls.model_validate(
            {
                "disposition_id": f"disposition:{digest}",
                "disposition_ref": f"disposition:{digest}",
                "disposition_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"disposition_id", "disposition_ref", "disposition_hash"}),
            }
        )

    @model_validator(mode="after")
    def bind_exact_identity(self, info):
        if info.context and info.context.get("cfius_disposition_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"disposition_id", "disposition_ref", "disposition_hash"})
        )
        if self.disposition_id != f"disposition:{digest}" or self.disposition_ref != f"disposition:{digest}":
            raise ValueError("disposition id/ref must bind exact content")
        if self.disposition_hash != f"sha256:{digest}":
            raise ValueError("disposition hash must bind exact content")
        for ref in self.evidence_refs:
            _assert_same_matter_ref(ref, self.matter_id)
        return self


class DecisionVerificationV1(_Contract):
    schema_version: Literal["cfius.decision_verification.v1"] = "cfius.decision_verification.v1"
    verification_id: str = Field(min_length=1)
    verification_ref: str = Field(min_length=1)
    verification_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    decision_run_ref: str = Field(min_length=1)
    status: Literal["VERIFIED", "INSUFFICIENT_BASIS", "INVALID_BASIS", "BINDING_MISMATCH", "PROVIDER_FAILURE", "STRUCTURALLY_RESOLVED"]
    semantic_result: Literal["TRUE", "FALSE", "UNKNOWN"] = "UNKNOWN"
    diagnostics: tuple[str, ...] = ()
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("diagnostics", mode="before")
    @classmethod
    def validate_diagnostics(cls, value: object) -> tuple[str, ...]:
        return _unique_nonempty_strings(value)

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "verification_id": "pending",
                "verification_ref": "pending",
                "verification_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_verification_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"verification_id", "verification_ref", "verification_hash"})
        )
        return cls.model_validate(
            {
                "verification_id": f"decision_verification:{digest}",
                "verification_ref": f"decision_verification:{digest}",
                "verification_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"verification_id", "verification_ref", "verification_hash"}),
            }
        )

    @model_validator(mode="after")
    def bind_exact_identity_and_run_binding(self, info):
        if info.context and info.context.get("cfius_verification_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"verification_id", "verification_ref", "verification_hash"})
        )
        expected_ref = f"decision_verification:{digest}"
        if self.verification_id != expected_ref or self.verification_ref != expected_ref:
            raise ValueError("decision verification id/ref must bind exact content")
        if self.verification_hash != f"sha256:{digest}":
            raise ValueError("decision verification hash must bind exact content")
        if not self.decision_run_ref.startswith("decision_run:"):
            raise ValueError("decision verification requires a bound decision_run_ref")
        if self.status == "PROVIDER_FAILURE" and self.semantic_result != "UNKNOWN":
            raise ValueError("PROVIDER_FAILURE requires UNKNOWN semantic_result")
        if self.semantic_result == "UNKNOWN" and not self.diagnostics:
            raise ValueError("UNKNOWN verification requires diagnostics")
        return self


class CounselBriefV1(_Contract):
    schema_version: Literal["cfius.counsel_brief.v1"] = "cfius.counsel_brief.v1"
    brief_id: str = Field(min_length=1)
    brief_ref: str = Field(min_length=1)
    brief_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    disposition_ref: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    observed_facts: tuple[str, ...] = Field(min_length=1)
    normative_claims: tuple[str, ...] = Field(min_length=1)
    unresolved_items: tuple[str, ...] = Field(min_length=1)
    remediation_refs: tuple[str, ...] = ()
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("observed_facts", "normative_claims", "unresolved_items", "remediation_refs", mode="before")
    @classmethod
    def validate_brief_collections(cls, value: object) -> tuple[str, ...]:
        return _unique_nonempty_strings(value)

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "brief_id": "pending",
                "brief_ref": "pending",
                "brief_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_brief_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"brief_id", "brief_ref", "brief_hash"})
        )
        return cls.model_validate(
            {
                "brief_id": f"counsel_brief:{digest}",
                "brief_ref": f"counsel_brief:{digest}",
                "brief_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"brief_id", "brief_ref", "brief_hash"}),
            }
        )

    @model_validator(mode="after")
    def bind_exact_identity_and_provenance(self, info):
        if info.context and info.context.get("cfius_brief_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"brief_id", "brief_ref", "brief_hash"})
        )
        if self.brief_id != f"counsel_brief:{digest}" or self.brief_ref != f"counsel_brief:{digest}":
            raise ValueError("brief id/ref must bind exact content")
        if self.brief_hash != f"sha256:{digest}":
            raise ValueError("brief hash must bind exact content")
        if not self.disposition_ref.startswith("disposition:"):
            raise ValueError("counsel brief requires a valid disposition_ref")
        for ref in self.remediation_refs:
            if not ref.startswith("remediation:"):
                raise ValueError("counsel brief remediation refs must be matter-scoped remediation pins")
            _assert_same_matter_ref(ref, self.matter_id)
        return self


class ReplayEnvelopeV1(_Contract):
    schema_version: Literal["cfius.replay_envelope.v1"] = "cfius.replay_envelope.v1"
    envelope_id: str = Field(min_length=1)
    envelope_ref: str = Field(min_length=1)
    replay_hash: str = Field(pattern=SHA256_PATTERN)
    matter_id: str = Field(min_length=1)
    decision_run_ref: str = Field(min_length=1)
    benchmark_ref: str = Field(min_length=1)
    benchmark_hash: str = Field(pattern=SHA256_PATTERN)
    source_refs: tuple[str, ...] = Field(min_length=1)
    source_hashes: tuple[str, ...] = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_refs", "source_hashes", mode="before")
    @classmethod
    def validate_source_refs(cls, value: object) -> tuple[str, ...]:
        refs = _unique_nonempty_strings(value)
        if not refs:
            raise ValueError("replay envelope requires source refs and corresponding hashes")
        return refs

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "envelope_id": "pending",
                "envelope_ref": "pending",
                "replay_hash": "sha256:" + "0" * 64,
                "recorded_at": datetime.now(timezone.utc),
                **values,
            },
            context={"cfius_replay_create_draft": True},
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"envelope_id", "envelope_ref", "replay_hash"})
        )
        return cls.model_validate(
            {
                "envelope_id": f"replay_envelope:{digest}",
                "envelope_ref": f"replay_envelope:{digest}",
                "replay_hash": f"sha256:{digest}",
                **draft.model_dump(mode="python", exclude={"envelope_id", "envelope_ref", "replay_hash"}),
            }
        )

    @model_validator(mode="after")
    def exact_replay_identity(self, info):
        if info.context and info.context.get("cfius_replay_create_draft"):
            expected_benchmark = self.benchmark_ref
            expected_digest = self.benchmark_hash[7:]
            if not expected_benchmark.startswith("benchmark:") or expected_benchmark != f"benchmark:{expected_digest}":
                raise ValueError("benchmark ref/hash pair must be exact and tamper-evident")
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"envelope_id", "envelope_ref", "replay_hash"})
        )
        expected_ref = f"replay_envelope:{digest}"
        if self.envelope_id != expected_ref or self.envelope_ref != expected_ref:
            raise ValueError("replay envelope id/ref must bind exact content")
        if self.replay_hash != f"sha256:{digest}":
            raise ValueError("replay hash must bind exact content")
        if not self.benchmark_ref.startswith("benchmark:") or self.benchmark_ref != f"benchmark:{self.benchmark_hash[7:]}":
            raise ValueError("benchmark tamper detected: ref and hash mismatch")
        if len(self.source_refs) != len(self.source_hashes):
            raise ValueError("source refs and hashes must match one-to-one")
        for source_ref, source_hash in zip(self.source_refs, self.source_hashes):
            if not source_ref or not source_hash.startswith("sha256:"):
                raise ValueError("source refs/hashes must be pinned and tamper-evident")
        return self
