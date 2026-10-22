"""Five-object semantic kernel contracts for the clean-break ContextGraph refactor.

Only Ontology, Evidence, ContextGraph, DecisionRecord and ChangeCandidate are
first-class semantic objects. The additional models in this module are typed
embedded values used to preserve intent, formation, reasoning and empirical
invariants without recreating independent artifact families.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from contracts.evidence_location import EvidenceSelector
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash

from contracts.candidate_review import CandidateReviewResultV1

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"

KernelKind: TypeAlias = Literal[
    "ontology", "evidence", "context_graph", "decision_record", "change_candidate"
]


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


def _require_pin_kind(pin: ArtifactPinV1, kind: str, field: str) -> ArtifactPinV1:
    if pin.artifact_kind != kind:
        raise ValueError(f"{field} must pin {kind}")
    return pin


def _ordered_unique_pins(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        return value
    pins = tuple(
        item if isinstance(item, ArtifactPinV1) else ArtifactPinV1.model_validate(item)
        for item in value
    )
    keys = tuple(_pin_key(pin) for pin in pins)
    if len(keys) != len(set(keys)):
        raise ValueError("pin collection contains duplicate identities")
    return tuple(sorted(pins, key=_pin_key))


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


class WorkspaceScope(_Contract):
    """Embedded tenant/workspace lookup key; never a semantic authority object."""

    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)


class _ContentAddressed(_Contract):
    """One identity rule shared by all five kernel semantic objects."""

    ARTIFACT_KIND: ClassVar[str]
    artifact_ref: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=SHA256_PATTERN)

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "artifact_ref": "pending",
                "artifact_hash": "sha256:" + "0" * 64,
                **values,
            },
            context={"kernel_create_draft": True},
        )
        normalized = draft.model_dump(
            mode="python", exclude={"artifact_ref", "artifact_hash"}
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        return cls.model_validate(
            {
                "artifact_ref": f"{cls.ARTIFACT_KIND}:{digest}",
                "artifact_hash": f"sha256:{digest}",
                **normalized,
            }
        )

    @model_validator(mode="after")
    def identity_binds_exact_content(self, info: ValidationInfo):
        if info.context and info.context.get("kernel_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        if (
            self.artifact_ref != f"{type(self).ARTIFACT_KIND}:{digest}"
            or self.artifact_hash != f"sha256:{digest}"
        ):
            raise ValueError("kernel artifact ref/hash does not bind exact content")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind=type(self).ARTIFACT_KIND,
            artifact_ref=self.artifact_ref,
            artifact_hash=self.artifact_hash,
        )


class _EmbeddedAddressed(_Contract):
    """Content-addressed embedded/transport value, never a sixth semantic object."""

    VALUE_KIND: ClassVar[str]
    value_ref: str = Field(min_length=1)
    value_hash: str = Field(pattern=SHA256_PATTERN)

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                "value_ref": "pending",
                "value_hash": "sha256:" + "0" * 64,
                **values,
            },
            context={"kernel_create_draft": True},
        )
        normalized = draft.model_dump(
            mode="python", exclude={"value_ref", "value_hash"}
        )
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"value_ref", "value_hash"})
        )
        return cls.model_validate(
            {
                "value_ref": f"{cls.VALUE_KIND}:{digest}",
                "value_hash": f"sha256:{digest}",
                **normalized,
            }
        )

    @model_validator(mode="after")
    def identity_binds_exact_content(self, info: ValidationInfo):
        if info.context and info.context.get("kernel_create_draft"):
            return self
        digest = content_hash(
            self.model_dump(mode="json", exclude={"value_ref", "value_hash"})
        )
        if (
            self.value_ref != f"{type(self).VALUE_KIND}:{digest}"
            or self.value_hash != f"sha256:{digest}"
        ):
            raise ValueError("embedded value ref/hash does not bind exact content")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind=type(self).VALUE_KIND,
            artifact_ref=self.value_ref,
            artifact_hash=self.value_hash,
        )


class IntentSnapshot(_EmbeddedAddressed):
    VALUE_KIND = "intent_snapshot"

    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    competency_questions: tuple[str, ...]
    hard_negatives: tuple[str, ...] = ()
    material_dimensions: tuple[str, ...] = ()
    source_roles: tuple[str, ...] = ()
    required_semantics: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    captured_at: datetime

    @field_validator(
        "competency_questions",
        "hard_negatives",
        "material_dimensions",
        "source_roles",
        "required_semantics",
        "unresolved_questions",
    )
    @classmethod
    def unique_text_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("intent values must be nonempty")
        if len(value) != len(set(value)):
            raise ValueError("intent values must be unique")
        return value


class FormationInput(_EmbeddedAddressed):
    VALUE_KIND = "formation_input"

    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    intent: IntentSnapshot
    ontology: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...]
    policy_pins: tuple[ArtifactPinV1, ...] = ()
    predecessor_context: ArtifactPinV1 | None = None
    captured_at: datetime

    @field_validator("evidence", "policy_pins", mode="before")
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_scope_and_kinds(self):
        _require_pin_kind(self.ontology, "ontology", "ontology")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "evidence")
        if self.predecessor_context is not None:
            _require_pin_kind(
                self.predecessor_context, "context_graph", "predecessor_context"
            )
        if (self.intent.organization_id, self.intent.workspace_id) != (
            self.organization_id,
            self.workspace_id,
        ):
            raise ValueError("formation input intent scope mismatch")
        return self


class HumanResponsibilityContext(_EmbeddedAddressed):
    VALUE_KIND = "human_responsibility_context"

    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    responsibility_bearer_id: str = Field(min_length=1)
    delegated_actor_id: str = Field(min_length=1)
    delegated_operation: str = Field(min_length=1)
    commitment_mode: Literal["AUTO", "NOTIFY", "CONFIRM"]
    scope_ref: str = Field(min_length=1)
    captured_at: datetime

    @model_validator(mode="after")
    def responsibility_scope_is_self_consistent(self):
        if self.scope_ref != f"workspace:{self.workspace_id}":
            raise ValueError("responsibility scope must equal workspace scope")
        if self.responsibility_bearer_id == self.delegated_actor_id:
            raise ValueError("agent/runtime actor cannot bear delegated human responsibility")
        return self


class WorkingContext(_EmbeddedAddressed):
    VALUE_KIND = "working_context"

    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    context_graph: ArtifactPinV1
    task_ref: str = Field(min_length=1)
    task_goal: str = Field(min_length=1)
    ontology: ArtifactPinV1
    selected_node_ids: tuple[str, ...] = ()
    selected_edge_ids: tuple[str, ...] = ()
    selected_assertion_ids: tuple[str, ...] = ()
    evidence: tuple[ArtifactPinV1, ...]
    unresolved_questions: tuple[str, ...] = ()
    selection_policy: ArtifactPinV1
    responsibility: HumanResponsibilityContext
    formed_at: datetime

    @field_validator(
        "selected_node_ids",
        "selected_edge_ids",
        "selected_assertion_ids",
        "unresolved_questions",
        mode="before",
    )
    @classmethod
    def canonicalize_text_sets(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        items = tuple(str(item) for item in value)
        if any(not item.strip() for item in items):
            raise ValueError("working context selections must be nonempty")
        if len(items) != len(set(items)):
            raise ValueError("working context selections must be unique")
        return tuple(sorted(items))

    @field_validator("evidence", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_working_context_boundary(self):
        _require_pin_kind(self.context_graph, "context_graph", "context_graph")
        _require_pin_kind(self.ontology, "ontology", "ontology")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "evidence")
        if (
            self.responsibility.organization_id,
            self.responsibility.workspace_id,
        ) != (self.organization_id, self.workspace_id):
            raise ValueError("responsibility scope must equal WorkingContext scope")
        return self


class OntologyEntityType(_Contract):
    type_id: str = Field(min_length=1)
    label: str = Field(min_length=1)


class OntologyRelationType(_Contract):
    type_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    directional: bool = True


class OntologyRule(_Contract):
    rule_id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    applicability: str | None = None


class OntologySemanticFrame(_Contract):
    frame_id: str = Field(min_length=1)
    frame_kind: Literal[
        "exception",
        "scope",
        "qualification",
        "precedence",
        "temporal_condition",
        "entity_distinction",
    ]
    subject_ref: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_ref: str = Field(min_length=1)
    evidence: tuple[ArtifactPinV1, ...] = ()

    @field_validator("evidence", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def evidence_is_kernel_evidence(self):
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "frame evidence")
        return self


class Ontology(_ContentAddressed):
    ARTIFACT_KIND = "ontology"

    schema_version: Literal["semantier.ontology.v1"] = "semantier.ontology.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    predecessor: ArtifactPinV1 | None = None
    entity_types: tuple[OntologyEntityType, ...]
    relation_types: tuple[OntologyRelationType, ...]
    rules: tuple[OntologyRule, ...] = ()
    semantic_frames: tuple[OntologySemanticFrame, ...] = ()
    provenance: tuple[ArtifactPinV1, ...] = ()
    recorded_at: datetime

    @field_validator("provenance", mode="before")
    @classmethod
    def canonicalize_provenance(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_semantic_identity(self):
        if self.predecessor is not None:
            _require_pin_kind(self.predecessor, "ontology", "predecessor")
        for values, label in (
            (tuple(item.type_id for item in self.entity_types), "entity type"),
            (tuple(item.type_id for item in self.relation_types), "relation type"),
            (tuple(item.rule_id for item in self.rules), "rule"),
            (tuple(item.frame_id for item in self.semantic_frames), "semantic frame"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} identities must be unique")
        return self


class Evidence(_ContentAddressed):
    ARTIFACT_KIND = "evidence"

    schema_version: Literal["semantier.evidence.v1"] = "semantier.evidence.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    selector: EvidenceSelector
    selector_hash: str = Field(pattern=SHA256_PATTERN)
    evidence_role: str = Field(min_length=1)
    provenance: tuple[ArtifactPinV1, ...] = ()
    observed_at: datetime | None = None
    source_time: datetime | None = None
    recorded_at: datetime

    @field_validator("provenance", mode="before")
    @classmethod
    def canonicalize_provenance(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def selector_identity_is_exact(self):
        if self.selector_hash != self.selector.selector_hash():
            raise ValueError("selector_hash must bind the exact EvidenceSelector")
        return self


class ContextWorld(_Contract):
    source_graph: ArtifactPinV1
    mode: Literal["current", "point", "interval", "prior_state", "causal_window"]
    at_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    reasoning_version: str = Field(min_length=1)
    governance_time: datetime

    @model_validator(mode="after")
    def valid_temporal_shape(self):
        if self.mode == "current":
            if any(value is not None for value in (self.at_time, self.start_time, self.end_time)):
                raise ValueError("current world must not invent explicit world-time bounds")
        elif self.mode == "point":
            if self.at_time is None or self.start_time is not None or self.end_time is not None:
                raise ValueError("point world requires only at_time")
        else:
            if self.at_time is not None or self.start_time is None or self.end_time is None:
                raise ValueError("windowed world requires start_time/end_time only")
            if self.start_time >= self.end_time:
                raise ValueError("world-time interval must be ordered")
        return self


class ContextNode(_Contract):
    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ContextEdge(_Contract):
    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)


class ContextAssertion(_Contract):
    assertion_id: str = Field(min_length=1)
    subject_ref: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_ref: str = Field(min_length=1)
    origin: Literal["SOURCE", "DERIVED"]
    evidence: tuple[ArtifactPinV1, ...]
    derivation: ArtifactPinV1 | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def preserve_fact_derivation_distinction(self):
        if not self.evidence:
            raise ValueError("consequential assertions require evidence")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "assertion evidence")
        if self.origin == "SOURCE" and self.derivation is not None:
            raise ValueError("source assertions must not claim a derivation")
        if self.origin == "DERIVED" and self.derivation is None:
            raise ValueError("derived assertions require exact derivation/proof binding")
        return self


class ExcludedFormationInput(_Contract):
    input: ArtifactPinV1
    reason: str = Field(min_length=1)


class ContextFormation(_Contract):
    source_inputs: tuple[ArtifactPinV1, ...]
    selected_inputs: tuple[ArtifactPinV1, ...]
    excluded_inputs: tuple[ExcludedFormationInput, ...]
    conflicts: tuple[str, ...] = ()
    formation_policy: ArtifactPinV1
    formation_policy_version: str = Field(min_length=1)
    predecessor_context: ArtifactPinV1 | None = None
    applied_change_candidate: ArtifactPinV1 | None = None

    @field_validator("source_inputs", "selected_inputs", mode="before")
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def selection_is_reproducible(self):
        source_keys = {_pin_key(pin) for pin in self.source_inputs}
        selected_keys = {_pin_key(pin) for pin in self.selected_inputs}
        excluded_keys = {_pin_key(item.input) for item in self.excluded_inputs}
        if not selected_keys <= source_keys or not excluded_keys <= source_keys:
            raise ValueError("selected/excluded inputs must come from exact source inputs")
        if selected_keys & excluded_keys:
            raise ValueError("one formation input cannot be selected and excluded")
        if len(excluded_keys) != len(self.excluded_inputs):
            raise ValueError("excluded formation inputs must be unique")
        if self.predecessor_context is not None:
            _require_pin_kind(
                self.predecessor_context, "context_graph", "predecessor_context"
            )
        if self.applied_change_candidate is not None:
            _require_pin_kind(
                self.applied_change_candidate,
                "change_candidate",
                "applied_change_candidate",
            )
        return self


class ContextGraph(_ContentAddressed):
    ARTIFACT_KIND = "context_graph"

    schema_version: Literal["semantier.context_graph.v1"] = "semantier.context_graph.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    state: Literal["PROPOSED", "ADMITTED"]
    predecessor: ArtifactPinV1 | None = None
    admission_decision: ArtifactPinV1 | None = None
    intent: IntentSnapshot
    ontology: ArtifactPinV1
    world: ContextWorld
    nodes: tuple[ContextNode, ...]
    edges: tuple[ContextEdge, ...]
    assertions: tuple[ContextAssertion, ...]
    evidence: tuple[ArtifactPinV1, ...]
    formation: ContextFormation
    recorded_at: datetime

    @field_validator("evidence", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_context_integrity(self):
        _require_pin_kind(self.ontology, "ontology", "ontology")
        if self.predecessor is not None:
            _require_pin_kind(self.predecessor, "context_graph", "predecessor")
        if self.state == "PROPOSED":
            if self.admission_decision is not None:
                raise ValueError("PROPOSED ContextGraph must not bind an admission decision")
        else:
            if self.predecessor is None:
                raise ValueError("ADMITTED ContextGraph requires exact PROPOSED predecessor")
            if self.admission_decision is None:
                raise ValueError("ADMITTED ContextGraph requires exact admission DecisionRecord")
            _require_pin_kind(
                self.admission_decision, "decision_record", "admission_decision"
            )
        if (self.intent.organization_id, self.intent.workspace_id) != (
            self.organization_id,
            self.workspace_id,
        ):
            raise ValueError("ContextGraph intent scope mismatch")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "evidence")
        evidence_keys = {_pin_key(pin) for pin in self.evidence}
        for assertion in self.assertions:
            if any(_pin_key(pin) not in evidence_keys for pin in assertion.evidence):
                raise ValueError("assertion evidence must be included in ContextGraph evidence")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("ContextGraph node ids must be unique")
        node_set = set(node_ids)
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("ContextGraph edge ids must be unique")
        if any(edge.source_id not in node_set or edge.target_id not in node_set for edge in self.edges):
            raise ValueError("ContextGraph edges must reference existing nodes")
        assertion_ids = tuple(item.assertion_id for item in self.assertions)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("ContextGraph assertion ids must be unique")
        if self.formation.predecessor_context != self.predecessor:
            raise ValueError("formation predecessor must equal ContextGraph predecessor")
        return self


class ReasonerInvocation(_Contract):
    invocation_id: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    executable_engine: ArtifactPinV1
    rules: tuple[ArtifactPinV1, ...]
    inputs: tuple[ArtifactPinV1, ...]
    outputs: tuple[str, ...]
    proof_steps: tuple[str, ...]
    evidence: tuple[ArtifactPinV1, ...]

    @field_validator("rules", "inputs", "evidence", mode="before")
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def exact_reasoning_basis(self):
        if not self.outputs:
            raise ValueError("reasoner invocation requires exact outputs")
        if not self.evidence:
            raise ValueError("reasoner invocation requires evidence binding")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "reasoner evidence")
        return self


class ReasoningTrace(_Contract):
    strategy: str = Field(min_length=1)
    reasoning_version: str = Field(min_length=1)
    invocations: tuple[ReasonerInvocation, ...]
    working_context: WorkingContext | None = None
    derived_assertion_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class AdmissibilityResult(_Contract):
    result: Literal["ADMISSIBLE", "NOT_ADMISSIBLE", "UNKNOWN"]
    reasons: tuple[str, ...]


class AuthorizationBasis(_Contract):
    operation: str = Field(min_length=1)
    subject: ArtifactPinV1
    scope_ref: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    authorization_class: Literal["TENANT_MEMBER", "TENANT_ADMIN"]
    membership_status: str = Field(default="active", min_length=1)
    can_change_settings: bool
    resolver_ref: str = Field(min_length=1)
    resolver_hash: str = Field(pattern=SHA256_PATTERN)
    governing_pins: tuple[ArtifactPinV1, ...] = ()

    @field_validator("governing_pins", mode="before")
    @classmethod
    def canonicalize_governing_pins(cls, value: object) -> object:
        return _ordered_unique_pins(value)


_AUTHORIZATION_PURPOSES = {
    "CONTEXT_ADMISSION",
    "ONTOLOGY_ADMISSION",
    "CONSEQUENTIAL_DECISION",
}


class DecisionRecord(_ContentAddressed):
    ARTIFACT_KIND = "decision_record"

    schema_version: Literal["semantier.decision_record.v1"] = (
        "semantier.decision_record.v1"
    )
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    purpose: Literal[
        "INFERENCE",
        "EVALUATION",
        "CONTEXT_ADMISSION",
        "ONTOLOGY_ADMISSION",
        "CONSEQUENTIAL_DECISION",
        "REVIEW",
    ]
    subject: ArtifactPinV1
    context: ArtifactPinV1
    ontology: ArtifactPinV1
    result_code: str = Field(min_length=1)
    epistemic_status: Literal["KNOWN", "UNKNOWN"]
    evidence: tuple[ArtifactPinV1, ...]
    admissibility: AdmissibilityResult | None = None
    reasoning: ReasoningTrace
    actor_id: str = Field(min_length=1)
    reasoner_identity: str = Field(min_length=1)
    authorization: AuthorizationBasis | None = None
    outcome_observation: ArtifactPinV1 | None = None
    candidate_review: CandidateReviewResultV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    recorded_at: datetime

    @field_validator("evidence", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_decision_boundary(self):
        if self.candidate_review is not None:
            projection = self.candidate_review.projection
            if (
                self.purpose != "REVIEW" or self.authorization is not None
                or (self.subject, self.context, self.ontology) != (projection.proposal_pin, projection.baseline_graph_pin, projection.ontology_pin)
                or (self.organization_id, self.workspace_id) != (projection.organization_id, projection.workspace_id)
            ):
                raise ValueError("REVIEW_DECISION_BINDING_MISMATCH")
            if set(self.evidence) != {w.pin for w in projection.supporting_evidence}:
                raise ValueError("REVIEW_EVIDENCE_BINDING_MISMATCH")
        _require_pin_kind(self.context, "context_graph", "context")
        _require_pin_kind(self.ontology, "ontology", "ontology")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "evidence")
        if self.epistemic_status == "UNKNOWN" and not self.reasoning.diagnostics:
            raise ValueError("UNKNOWN decision requires explicit diagnostics")
        if self.purpose in _AUTHORIZATION_PURPOSES:
            if self.authorization is None:
                raise ValueError("authorization-bearing purpose requires authority basis")
            if self.authorization.subject != self.subject:
                raise ValueError("authorization subject must equal DecisionRecord subject")
            if (
                self.authorization.organization_id,
                self.authorization.workspace_id,
                self.authorization.actor_id,
            ) != (self.organization_id, self.workspace_id, self.actor_id):
                raise ValueError("authorization scope/actor must equal DecisionRecord scope/actor")
        elif self.authorization is not None:
            raise ValueError("ordinary inference/evaluation/review cannot carry authority")
        if self.purpose == "CONTEXT_ADMISSION":
            _require_pin_kind(self.subject, "context_graph", "context admission subject")
        if self.purpose == "ONTOLOGY_ADMISSION":
            _require_pin_kind(self.subject, "ontology", "ontology admission subject")
        return self


class ContextGraphOrigin(_Contract):
    origin_kind: Literal["CONTEXT_GRAPH"] = "CONTEXT_GRAPH"
    context: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...] = ()
    decision: ArtifactPinV1 | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_origin(self):
        _require_pin_kind(self.context, "context_graph", "origin context")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "origin evidence")
        if self.decision is not None:
            _require_pin_kind(self.decision, "decision_record", "origin decision")
        return self


class PreContextOrigin(_Contract):
    origin_kind: Literal["PRE_CONTEXT"] = "PRE_CONTEXT"
    formation_input: FormationInput
    evidence: tuple[ArtifactPinV1, ...] = ()
    commentary_refs: tuple[ArtifactPinV1, ...] = ()

    @field_validator("evidence", "commentary_refs", mode="before")
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_origin(self):
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "origin evidence")
        if not self.evidence and not self.commentary_refs:
            raise ValueError("pre-context origin requires supporting evidence/commentary")
        return self


class WorkflowEditOrigin(_Contract):
    """Pinned technical workflow-edit basis for canonical ChangeCandidate learning.

    Workflow definitions are owned outside the semantic-stage kernel. The
    origin binds the exact predecessor WorkflowDefinition identity plus kernel
    Evidence receipts without pretending the workflow is a ContextGraph or
    semantic OwningStage.
    """

    origin_kind: Literal["WORKFLOW_EDIT"] = "WORKFLOW_EDIT"
    workflow: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...] = ()

    @field_validator("evidence", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_origin(self):
        _require_pin_kind(self.workflow, "workflow_definition", "origin workflow")
        if not self.evidence:
            raise ValueError("workflow edit origin requires evidence")
        for pin in self.evidence:
            _require_pin_kind(pin, "evidence", "origin evidence")
        return self


ChangeOrigin: TypeAlias = ContextGraphOrigin | PreContextOrigin | WorkflowEditOrigin


class ProtectedInput(_Contract):
    slot_id: str = Field(min_length=1)
    artifact: ArtifactPinV1


class ComparisonExpectation(_Contract):
    matched_baseline_result: str = Field(min_length=1)
    matched_successor_result: str = Field(min_length=1)
    holdout_policy: Literal["UNCHANGED"] = "UNCHANGED"


class EvaluationBasis(_Contract):
    evaluator: ArtifactPinV1
    obligation_suite: ArtifactPinV1
    comparison_policy: ArtifactPinV1
    protected_inputs: tuple[ProtectedInput, ...]
    runtime_pins: tuple[ArtifactPinV1, ...] = ()
    provider_pins: tuple[ArtifactPinV1, ...] = ()
    config_pins: tuple[ArtifactPinV1, ...] = ()
    candidate_delta_bounds: str = Field(min_length=1)
    comparison_expectation: ComparisonExpectation | None = None

    @field_validator("runtime_pins", "provider_pins", "config_pins", mode="before")
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def protected_input_slots_are_unique(self):
        slots = tuple(item.slot_id for item in self.protected_inputs)
        if len(slots) != len(set(slots)):
            raise ValueError("protected evaluation input slots must be unique")
        return self


class ChangeDelta(_Contract):
    description: str = Field(min_length=1)
    claims_semantic_graph_change: bool = False
    basis_refs: tuple[ArtifactPinV1, ...] = ()
    structural_diff: ArtifactPinV1 | None = None

    @field_validator("basis_refs", mode="before")
    @classmethod
    def canonicalize_basis_refs(cls, value: object) -> object:
        return _ordered_unique_pins(value)


class ChangeObservation(_Contract):
    case: ArtifactPinV1
    execution_attempt: ArtifactPinV1
    status: Literal["COMPLETE", "FAILED", "REJECTED", "PARTIAL", "INCONCLUSIVE"]
    metric_refs: tuple[ArtifactPinV1, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @field_validator("metric_refs", mode="before")
    @classmethod
    def canonicalize_metric_refs(cls, value: object) -> object:
        return _ordered_unique_pins(value)


class ChangeAssessment(_Contract):
    trial_validity: Literal["VALID", "INVALID_EVALUATION", "INCOMPLETE_EVIDENCE"]
    verdict: Literal[
        "BETTER",
        "WORSE",
        "INDETERMINATE",
        "LOCAL_FIX_VERIFIED",
    ]
    metric_refs: tuple[ArtifactPinV1, ...] = ()
    disposition: Literal[
        "REVIEW_CANDIDATE",
        "ACCEPTED_FOR_SUCCESSOR_INPUT",
        "REJECTED_FOR_USE",
        "NOT_YET_ADMISSIBLE",
    ]
    reason_code: str = Field(min_length=1)

    @field_validator("metric_refs", mode="before")
    @classmethod
    def canonicalize_metric_refs(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validity_limits_claim(self):
        if self.trial_validity != "VALID" and self.verdict != "INDETERMINATE":
            raise ValueError("invalid/incomplete trial cannot claim improvement")
        return self


class ChangeCandidate(_ContentAddressed):
    ARTIFACT_KIND = "change_candidate"

    schema_version: Literal["semantier.change_candidate.v1"] = (
        "semantier.change_candidate.v1"
    )
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    stage: Literal["PROPOSAL", "OBSERVATION", "ASSESSMENT"]
    predecessor: ArtifactPinV1 | None = None
    origin: ChangeOrigin
    failure_class: Literal["CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"]
    target: Literal["ONTOLOGY", "CONTEXT_FORMATION", "REASONING", "WORKFLOW", "POLICY"]
    hypothesis: str = Field(min_length=1)
    evaluation_basis: EvaluationBasis
    delta: ChangeDelta
    matched_cases: tuple[ArtifactPinV1, ...]
    independent_holdout_cases: tuple[ArtifactPinV1, ...]
    observations: tuple[ChangeObservation, ...] = ()
    assessment: ChangeAssessment | None = None
    successor_inputs: tuple[ArtifactPinV1, ...] = ()
    recorded_at: datetime

    @field_validator(
        "matched_cases",
        "independent_holdout_cases",
        "successor_inputs",
        mode="before",
    )
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_learning_boundary(self):
        if self.predecessor is not None:
            _require_pin_kind(self.predecessor, "change_candidate", "predecessor")
        origin_scope = (
            self.origin.context
            if isinstance(self.origin, ContextGraphOrigin)
            else self.origin.formation_input
            if isinstance(self.origin, PreContextOrigin)
            else None
        )
        if isinstance(origin_scope, FormationInput):
            scope = (origin_scope.organization_id, origin_scope.workspace_id)
        else:
            scope = None
        if scope is not None and scope != (self.organization_id, self.workspace_id):
            raise ValueError("pre-context ChangeCandidate origin scope mismatch")
        matched = {_pin_key(pin) for pin in self.matched_cases}
        holdout = {_pin_key(pin) for pin in self.independent_holdout_cases}
        if matched & holdout:
            raise ValueError("matched and independent holdout cases must be disjoint")
        if self.target == "ONTOLOGY" and self.failure_class != "STRUCTURE":
            raise ValueError("ontology change requires STRUCTURE failure attribution")
        if self.target == "POLICY" and self.failure_class != "STRUCTURE":
            raise ValueError("policy change requires STRUCTURE failure attribution")
        if self.delta.claims_semantic_graph_change and self.failure_class != "STRUCTURE":
            raise ValueError("semantic graph change requires STRUCTURE failure attribution")
        if self.stage == "PROPOSAL":
            if self.revision != 0 or self.predecessor is not None:
                raise ValueError("proposal is revision zero with no predecessor")
            if self.observations or self.assessment is not None:
                raise ValueError("proposal cannot contain future observations/assessment")
        elif self.stage == "OBSERVATION":
            if self.revision < 1 or self.predecessor is None or not self.observations:
                raise ValueError("observation revision requires predecessor and observations")
            if self.assessment is not None:
                raise ValueError("observation revision cannot contain future assessment")
        else:
            if self.revision < 2 or self.predecessor is None or not self.observations:
                raise ValueError("assessment revision requires prior observations")
            if self.assessment is None:
                raise ValueError("assessment revision requires assessment")
            if self.assessment.verdict == "BETTER" and not self.independent_holdout_cases:
                raise ValueError("generalized BETTER claim requires independent holdout")
        if any(
            pin.artifact_kind in {"context_graph", "decision_record", "change_candidate"}
            for pin in self.successor_inputs
        ):
            raise ValueError(
                "ChangeCandidate may bind successor inputs, not future semantic outputs"
            )
        return self


def validate_context_admission_transition(
    proposed: ContextGraph,
    decision: DecisionRecord,
    admitted: ContextGraph,
) -> None:
    """Validate one durable PROPOSED -> DecisionRecord -> ADMITTED transition."""

    if proposed.state != "PROPOSED" or admitted.state != "ADMITTED":
        raise ValueError("context admission transition requires PROPOSED -> ADMITTED")
    if decision.purpose != "CONTEXT_ADMISSION":
        raise ValueError("context admission transition requires CONTEXT_ADMISSION decision")
    if decision.subject != proposed.pin():
        raise ValueError("context admission decision must target exact proposed revision")
    if decision.context != proposed.pin():
        raise ValueError("context admission decision must bind exact proposed context")
    if admitted.predecessor != proposed.pin():
        raise ValueError("admitted context must pin exact proposed predecessor")
    if admitted.admission_decision != decision.pin():
        raise ValueError("admitted context must pin exact admission DecisionRecord")
    if admitted.revision != proposed.revision + 1:
        raise ValueError("admitted context revision must advance exactly once")
    if (proposed.organization_id, proposed.workspace_id) != (
        admitted.organization_id,
        admitted.workspace_id,
    ) or (decision.organization_id, decision.workspace_id) != (
        proposed.organization_id,
        proposed.workspace_id,
    ):
        raise ValueError("context admission scope mismatch")
    if proposed.ontology != admitted.ontology or decision.ontology != proposed.ontology:
        raise ValueError("context admission must preserve exact ontology basis")


def validate_change_candidate_revision(
    previous: ChangeCandidate, current: ChangeCandidate
) -> None:
    """Protect proposal/evaluation basis while appending observations/assessment."""

    if current.predecessor != previous.pin():
        raise ValueError("ChangeCandidate revision must pin exact predecessor")
    if current.revision != previous.revision + 1:
        raise ValueError("ChangeCandidate revision must advance exactly once")
    if (previous.stage, current.stage) not in {
        ("PROPOSAL", "OBSERVATION"),
        ("OBSERVATION", "ASSESSMENT"),
    }:
        raise ValueError("ChangeCandidate stages must advance A -> B -> C")
    immutable_fields = (
        "organization_id",
        "workspace_id",
        "origin",
        "failure_class",
        "target",
        "hypothesis",
        "evaluation_basis",
        "delta",
        "matched_cases",
        "independent_holdout_cases",
    )
    if any(getattr(previous, field) != getattr(current, field) for field in immutable_fields):
        raise ValueError("ChangeCandidate proposal/evaluation basis is immutable")
    if previous.observations and current.observations != previous.observations:
        raise ValueError("recorded ChangeCandidate observations cannot be rewritten")


__all__ = [
    "AdmissibilityResult",
    "AuthorizationBasis",
    "ChangeAssessment",
    "ChangeCandidate",
    "ChangeDelta",
    "ChangeObservation",
    "ComparisonExpectation",
    "ContextAssertion",
    "ContextEdge",
    "ContextFormation",
    "ContextGraph",
    "ContextGraphOrigin",
    "ContextNode",
    "ContextWorld",
    "Evidence",
    "EvaluationBasis",
    "ExcludedFormationInput",
    "FormationInput",
    "HumanResponsibilityContext",
    "IntentSnapshot",
    "Ontology",
    "OntologyEntityType",
    "OntologyRelationType",
    "OntologyRule",
    "OntologySemanticFrame",
    "PreContextOrigin",
    "ProtectedInput",
    "ReasonerInvocation",
    "ReasoningTrace",
    "WorkingContext",
    "WorkflowEditOrigin",
    "WorkspaceScope",
    "validate_change_candidate_revision",
    "validate_context_admission_transition",
]
