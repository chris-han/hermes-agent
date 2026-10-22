"""Derived post-model verification and handoff; neither has semantic authority."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from contracts.decision_local_projection import MaterialDependencyResolutionV1
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"
SemanticResult = Literal["TRUE", "FALSE", "UNKNOWN"]
FactorEpistemicState = Literal["PRESENT", "ABSENT", "UNKNOWN"]
ConflictMembership = Literal["CONFLICT", "COMPATIBLE", "UNKNOWN"]
ConflictResolution = Literal[
    "RESOLVED_TO_LEFT",
    "RESOLVED_TO_RIGHT",
    "UNRESOLVED_CONFLICT",
    "NO_RESOLUTION_NEEDED",
    "UNKNOWN",
]
PredicateEvaluationMode = Literal["COMPUTABLE", "INTERPRETED"]
VerificationStatus = Literal[
    "VERIFIED",
    "INSUFFICIENT_BASIS",
    "INVALID_BASIS",
    "BINDING_MISMATCH",
    "INVALID_MODEL_EVIDENCE",
    "COMPOSITION_UNAVAILABLE",
    "STRUCTURALLY_RESOLVED",
]
_POLICY = {
    "scope": "INTERPRETED_PREDICATE",
    "version": "tender-interpreted-verification.v1",
    "candidate_semantics": {"A": "TRUE", "B": "FALSE", "C": "UNKNOWN"},
    "materiality": "reuse DecisionLocalProjectionService.evaluate_semantic_proposal",
    "missing_material_dependency": "observed proposal normalizes to UNKNOWN",
    "missing_model_evidence": "INSUFFICIENT_BASIS without semantic result",
    "composition": "not supported by this interpreted-only policy",
}
VERIFICATION_POLICY = ArtifactPinV1(
    artifact_kind="verification_policy",
    artifact_ref="tender-interpreted-verification:v1:" + content_hash(_POLICY),
    artifact_hash="sha256:" + content_hash(_POLICY),
)

_COMPOSITION_POLICY_PAYLOAD = {
    "version": "tender-frozen-composition.v1",
    "benchmark_hash": "sha256:567f0d65e973a580f685b2cad938d2e0821c5f7bb1ccfbd672e8c08643ec970b",
    "predicate_ir_hash": "sha256:469e4bb992453773219256155896942a327d166f3e0500e32379bf0aa6b33cd6",
    "admitted_release_hash": "sha256:a9202b6fc6bf9195393784c04aaf9b439dbf128a07de0eb62d725fca6aa9036e",
    "truth_table_payload_hash": "sha256:ad29f7cbcb8b6f83bfceabd33a18b3fb8df06c7366f1058cff618b30c6cc5381",
    "operator_semantics_payload_hash": "sha256:b7c374c8e7956e4c485bd9b5470a2bbb862a4ad7b775b740b9afde3930a90a12",
    "normalization_payload_hash": "sha256:cce1b9f04db9d4aeee968ad6cc4e91f9996a4db08c5823bcbe2ca8e2a6d3e9a8",
    "ordinary_conditions": "UNKNOWN dominates FALSE; otherwise conjunction",
    "input_and_exception_conjunction": "FALSE dominates UNKNOWN; otherwise conjunction",
    "exception_union": "TRUE dominates UNKNOWN; otherwise FALSE",
    "final": (
        "input!=TRUE=>UNKNOWN; exception=TRUE=>FALSE; remaining UNKNOWN=>UNKNOWN; "
        "ordinary FALSE=>FALSE; else TRUE"
    ),
}
COMPOSITION_POLICY = ArtifactPinV1(
    artifact_kind="composition_policy",
    artifact_ref="tender-frozen-composition:v1:" + content_hash(_COMPOSITION_POLICY_PAYLOAD),
    artifact_hash="sha256:" + content_hash(_COMPOSITION_POLICY_PAYLOAD),
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InterpretedPredicateTraceV1(_Frozen):
    predicate_id: str = Field(min_length=1)
    selected_candidate_id: Literal["A", "B", "C"]
    proposed_semantic_result: SemanticResult
    material_dependencies: tuple[MaterialDependencyResolutionV1, ...]
    semantic_result: SemanticResult
    normalization: Literal["MODEL_SELECTION", "MATERIAL_DEPENDENCY_MISSING"]


class PredicateResultReceiptV1(_Frozen):
    schema_version: Literal["predicate_result_receipt.v1"] = "predicate_result_receipt.v1"
    receipt_ref: str
    receipt_hash: str = Field(pattern=_SHA)
    predicate_id: str = Field(min_length=1)
    evaluation_mode: PredicateEvaluationMode
    semantic_result: SemanticResult
    basis: tuple[ArtifactPinV1, ...] = Field(min_length=1)

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})

    @model_validator(mode="after")
    def verify_identity(self, info: ValidationInfo):
        digest = content_hash(self.hash_payload())
        if not (info.context and info.context.get("receipt_draft")) and (
            self.receipt_ref,
            self.receipt_hash,
        ) != ("predicate-result:v1:" + digest, "sha256:" + digest):
            raise ValueError("PREDICATE_RESULT_RECEIPT_IDENTITY_MISMATCH")
        if len(
            {(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in self.basis}
        ) != len(self.basis):
            raise ValueError("PREDICATE_RESULT_RECEIPT_BASIS_MUST_BE_UNIQUE")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                **values,
                "receipt_ref": "",
                "receipt_hash": "sha256:" + "0" * 64,
            },
            context={"receipt_draft": True},
        )
        payload = draft.hash_payload()
        digest = content_hash(payload)
        return cls.model_validate(
            {
                **payload,
                "receipt_ref": "predicate-result:v1:" + digest,
                "receipt_hash": "sha256:" + digest,
            }
        )

    @property
    def factor_state(self) -> FactorEpistemicState:
        return {
            "TRUE": "PRESENT",
            "FALSE": "ABSENT",
            "UNKNOWN": "UNKNOWN",
        }[self.semantic_result]

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="predicate_result_receipt",
            artifact_ref=self.receipt_ref,
            artifact_hash=self.receipt_hash,
        )


class ConflictMembershipReceiptV1(_Frozen):
    schema_version: Literal["conflict_membership_receipt.v1"] = "conflict_membership_receipt.v1"
    receipt_ref: str
    receipt_hash: str = Field(pattern=_SHA)
    decision_scope: str = Field(min_length=1)
    left_factor: ArtifactPinV1
    right_factor: ArtifactPinV1
    projection: ArtifactPinV1
    membership: ConflictMembership
    basis: tuple[ArtifactPinV1, ...] = Field(min_length=1)

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})

    @model_validator(mode="after")
    def verify_identity_and_basis(self, info: ValidationInfo):
        if self.left_factor == self.right_factor:
            raise ValueError("CONFLICT_MEMBERSHIP_REQUIRES_DISTINCT_FACTORS")
        if self.left_factor.artifact_kind != "predicate_result_receipt" or self.right_factor.artifact_kind != "predicate_result_receipt":
            raise ValueError("CONFLICT_MEMBERSHIP_REQUIRES_PREDICATE_RESULT_FACTORS")
        if self.projection.artifact_kind != "decision_graph_projection":
            raise ValueError("CONFLICT_MEMBERSHIP_REQUIRES_DECISION_GRAPH_PROJECTION")
        unique = {(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in self.basis}
        if len(unique) != len(self.basis):
            raise ValueError("CONFLICT_MEMBERSHIP_BASIS_MUST_BE_UNIQUE")
        required = {
            (self.left_factor.artifact_kind, self.left_factor.artifact_ref, self.left_factor.artifact_hash),
            (self.right_factor.artifact_kind, self.right_factor.artifact_ref, self.right_factor.artifact_hash),
            (self.projection.artifact_kind, self.projection.artifact_ref, self.projection.artifact_hash),
        }
        if not required.issubset(unique):
            raise ValueError("CONFLICT_MEMBERSHIP_BASIS_MUST_BIND_FACTORS_AND_PROJECTION")
        digest = content_hash(self.hash_payload())
        if not (info.context and info.context.get("receipt_draft")) and (
            self.receipt_ref,
            self.receipt_hash,
        ) != ("conflict-membership:v1:" + digest, "sha256:" + digest):
            raise ValueError("CONFLICT_MEMBERSHIP_RECEIPT_IDENTITY_MISMATCH")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {**values, "receipt_ref": "", "receipt_hash": "sha256:" + "0" * 64},
            context={"receipt_draft": True},
        )
        payload = draft.hash_payload()
        digest = content_hash(payload)
        return cls.model_validate({
            **payload,
            "receipt_ref": "conflict-membership:v1:" + digest,
            "receipt_hash": "sha256:" + digest,
        })

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="conflict_membership_receipt",
            artifact_ref=self.receipt_ref,
            artifact_hash=self.receipt_hash,
        )


class ConflictControllerBindingV1(_Frozen):
    schema_version: Literal["conflict_controller_binding.v1"] = "conflict_controller_binding.v1"
    projection: ArtifactPinV1
    authority_basis: ArtifactPinV1
    winner: Literal["LEFT", "RIGHT"]
    basis_kind: str = Field(min_length=1)

    @model_validator(mode="after")
    def verify_binding(self):
        if self.projection.artifact_kind != "decision_graph_projection":
            raise ValueError("CONTROLLER_BINDING_REQUIRES_DECISION_GRAPH_PROJECTION")
        return self


class ConflictResolutionReceiptV1(_Frozen):
    schema_version: Literal["conflict_resolution_receipt.v1"] = "conflict_resolution_receipt.v1"
    receipt_ref: str
    receipt_hash: str = Field(pattern=_SHA)
    conflict_membership: ArtifactPinV1
    projection: ArtifactPinV1
    controller: ConflictControllerBindingV1 | None = None
    resolution: ConflictResolution

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})

    @model_validator(mode="after")
    def verify_identity_and_state(self, info: ValidationInfo):
        if self.conflict_membership.artifact_kind != "conflict_membership_receipt":
            raise ValueError("RESOLUTION_REQUIRES_CONFLICT_MEMBERSHIP_RECEIPT")
        if self.projection.artifact_kind != "decision_graph_projection":
            raise ValueError("RESOLUTION_REQUIRES_DECISION_GRAPH_PROJECTION")
        if self.controller is not None and self.controller.projection != self.projection:
            raise ValueError("CONTROLLER_PROJECTION_MISMATCH")
        if self.resolution in ("RESOLVED_TO_LEFT", "RESOLVED_TO_RIGHT") and self.controller is None:
            raise ValueError("RESOLVED_CONFLICT_REQUIRES_CONTROLLER")
        digest = content_hash(self.hash_payload())
        if not (info.context and info.context.get("receipt_draft")) and (
            self.receipt_ref,
            self.receipt_hash,
        ) != ("conflict-resolution:v1:" + digest, "sha256:" + digest):
            raise ValueError("CONFLICT_RESOLUTION_RECEIPT_IDENTITY_MISMATCH")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {**values, "receipt_ref": "", "receipt_hash": "sha256:" + "0" * 64},
            context={"receipt_draft": True},
        )
        payload = draft.hash_payload()
        digest = content_hash(payload)
        return cls.model_validate({
            **payload,
            "receipt_ref": "conflict-resolution:v1:" + digest,
            "receipt_hash": "sha256:" + digest,
        })

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="conflict_resolution_receipt",
            artifact_ref=self.receipt_ref,
            artifact_hash=self.receipt_hash,
        )


class ExceptionCompositionGroupV1(_Frozen):
    node_id: str = Field(min_length=1)
    predicate_ids: tuple[str, ...] = Field(min_length=1)


class TenderCompositionSpecV1(_Frozen):
    schema_version: Literal["tender_composition_spec.v1"] = "tender_composition_spec.v1"
    decision_id: str = Field(min_length=1)
    input_predicate_ids: tuple[str, ...] = ()
    ordinary_predicate_ids: tuple[str, ...] = Field(min_length=1)
    exception_groups: tuple[ExceptionCompositionGroupV1, ...] = ()

    @model_validator(mode="after")
    def unique_nodes_and_predicates(self):
        if len({g.node_id for g in self.exception_groups}) != len(self.exception_groups):
            raise ValueError("COMPOSITION_NODE_IDS_MUST_BE_UNIQUE")
        for values in (self.input_predicate_ids, self.ordinary_predicate_ids):
            if len(set(values)) != len(values):
                raise ValueError("COMPOSITION_PREDICATE_IDS_MUST_BE_UNIQUE")
        for group in self.exception_groups:
            if len(set(group.predicate_ids)) != len(group.predicate_ids):
                raise ValueError("EXCEPTION_PREDICATE_IDS_MUST_BE_UNIQUE")
        return self

    def leaf_ids(self) -> set[str]:
        return set(self.input_predicate_ids) | set(self.ordinary_predicate_ids) | {
            predicate_id
            for group in self.exception_groups
            for predicate_id in group.predicate_ids
        }


class CompositionNodeTraceV1(_Frozen):
    node_id: str = Field(min_length=1)
    operator: Literal[
        "INPUT_CONJUNCTION",
        "ALL_OF",
        "EXCEPTION_CONJUNCTION",
        "ANY_OF",
        "FINAL",
    ]
    inputs: tuple[SemanticResult, ...]
    semantic_result: SemanticResult


class ComposedDecisionTraceV1(_Frozen):
    receipts: tuple[PredicateResultReceiptV1, ...] = Field(min_length=1)
    nodes: tuple[CompositionNodeTraceV1, ...] = Field(min_length=1)
    semantic_result: SemanticResult


class DecisionVerificationV1(_Frozen):
    schema_version: Literal["decision_verification.v1"] = "decision_verification.v1"
    verification_ref: str
    verification_hash: str = Field(pattern=_SHA)
    scope: Literal["INTERPRETED_PREDICATE", "COMPOSED_PREDICATE_SET"] = (
        "INTERPRETED_PREDICATE"
    )
    policy: ArtifactPinV1 = VERIFICATION_POLICY
    supplied_basis_hash: str = Field(pattern=_SHA)
    status: VerificationStatus
    semantic_result: SemanticResult | None = None
    projection: ArtifactPinV1 | None = None
    frontier: ArtifactPinV1 | None = None
    model_input: ArtifactPinV1 | None = None
    distribution: ArtifactPinV1 | None = None
    trace: InterpretedPredicateTraceV1 | None = None
    composition_policy: ArtifactPinV1 | None = None
    composition_spec: TenderCompositionSpecV1 | None = None
    composed_trace: ComposedDecisionTraceV1 | None = None
    diagnostics: tuple[str, ...] = ()

    def hash_payload(self):
        return self.model_dump(
            mode="json", exclude={"verification_ref", "verification_hash"}
        )

    @model_validator(mode="after")
    def verify_identity_and_state(self, info: ValidationInfo):
        digest = content_hash(self.hash_payload())
        if not (info.context and info.context.get("verification_draft")) and (
            self.verification_ref,
            self.verification_hash,
        ) != (
            "decision-verification:v1:" + digest,
            "sha256:" + digest,
        ):
            raise ValueError("VERIFICATION_IDENTITY_MISMATCH")
        if self.policy != VERIFICATION_POLICY:
            raise ValueError("UNTRUSTED_VERIFICATION_POLICY")
        if self.status == "STRUCTURALLY_RESOLVED":
            if self.scope != "INTERPRETED_PREDICATE" or self.semantic_result != "UNKNOWN":
                raise ValueError("STRUCTURAL_RESOLUTION_REQUIRES_INTERPRETED_UNKNOWN")
            if self.projection is None:
                raise ValueError("STRUCTURAL_RESOLUTION_REQUIRES_PROJECTION")
            if any(value is not None for value in (self.frontier, self.model_input, self.distribution, self.trace, self.composition_policy, self.composition_spec, self.composed_trace)):
                raise ValueError("STRUCTURAL_RESOLUTION_MUST_NOT_CARRY_MODEL_OR_COMPOSITION_BASIS")
            return self
        if self.status == "VERIFIED":
            if self.scope == "INTERPRETED_PREDICATE":
                if any(
                    value is None
                    for value in (
                        self.semantic_result,
                        self.trace,
                        self.projection,
                        self.frontier,
                        self.model_input,
                        self.distribution,
                    )
                ):
                    raise ValueError("VERIFIED_REQUIRES_EXACT_RESULT_AND_BASIS")
                if any(
                    value is not None
                    for value in (
                        self.composition_policy,
                        self.composition_spec,
                        self.composed_trace,
                    )
                ):
                    raise ValueError(
                        "INTERPRETED_VERIFICATION_CANNOT_CARRY_COMPOSITION"
                    )
                if self.trace.semantic_result != self.semantic_result:
                    raise ValueError("VERIFICATION_TRACE_RESULT_MISMATCH")
            else:
                if any(
                    value is None
                    for value in (
                        self.semantic_result,
                        self.composition_policy,
                        self.composition_spec,
                        self.composed_trace,
                    )
                ):
                    raise ValueError(
                        "COMPOSED_VERIFICATION_REQUIRES_EXACT_RESULT_AND_BASIS"
                    )
                if self.composition_policy != COMPOSITION_POLICY:
                    raise ValueError("UNTRUSTED_COMPOSITION_POLICY")
                if self.trace is not None:
                    raise ValueError(
                        "COMPOSED_VERIFICATION_CANNOT_CARRY_INTERPRETED_TRACE"
                    )
                if self.composed_trace.semantic_result != self.semantic_result:
                    raise ValueError("COMPOSED_TRACE_RESULT_MISMATCH")
            if self.semantic_result == "UNKNOWN" and not self.diagnostics:
                raise ValueError("UNKNOWN_REQUIRES_DIAGNOSTICS")
        elif (
            self.semantic_result is not None
            or self.trace is not None
            or self.composed_trace is not None
            or not self.diagnostics
        ):
            raise ValueError("FAILED_VERIFICATION_CANNOT_HAVE_SEMANTIC_RESULT")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(
            {
                **values,
                "verification_ref": "",
                "verification_hash": "sha256:" + "0" * 64,
            },
            context={"verification_draft": True},
        )
        payload = draft.hash_payload()
        digest = content_hash(payload)
        return cls.model_validate(
            {
                **payload,
                "verification_ref": "decision-verification:v1:" + digest,
                "verification_hash": "sha256:" + digest,
            }
        )

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="decision_verification",
            artifact_ref=self.verification_ref,
            artifact_hash=self.verification_hash,
        )


class VerifiedDecisionHandoffV1(_Frozen):
    """Ephemeral projection; the owner must derive it again, never trust caller claims."""

    schema_version: Literal["verified_decision_handoff.v1"] = (
        "verified_decision_handoff.v1"
    )
    scope: Literal["INTERPRETED_PREDICATE", "COMPOSED_PREDICATE_SET"] = (
        "INTERPRETED_PREDICATE"
    )
    organization_id: str
    workspace_id: str
    verification: DecisionVerificationV1
    predicate_id: str
    semantic_result: SemanticResult
    subject: ArtifactPinV1
    context: ArtifactPinV1
    ontology: ArtifactPinV1
    working_context: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...]
    diagnostics: tuple[str, ...]
