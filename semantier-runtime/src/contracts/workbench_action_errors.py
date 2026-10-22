"""Closed ACTIONS-01 v6 errors; producers must sanitize messages and scope pins."""

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import Sha256


class _ErrorContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def verify_pin_identifiers(self):
        for field in type(self).model_fields:
            value = getattr(self, field)
            if isinstance(value, ArtifactPinV1) and (
                not value.artifact_ref.isascii() or not value.artifact_kind.isascii()
            ):
                raise ValueError("v6 pin identifiers must be ASCII-stable")
        return self


class EmptyDetailsV6(_ErrorContract):
    pass


class FieldDetailsV6(_ErrorContract):
    field: str
    reason: str


class ResourceDetailsV6(_ErrorContract):
    resource_kind: str
    resource_pin: ArtifactPinV1 | None = None


class AuthorityDetailsV6(_ErrorContract):
    required_authorization_class: str
    authority_pin: ArtifactPinV1 | None = None


class ConflictDetailsV6(_ErrorContract):
    expected_value: str | int | None
    current_value: str | int | None
    artifact_pin: ArtifactPinV1 | None = None


class HashDetailsV6(_ErrorContract):
    expected_hash: Sha256
    actual_hash: Sha256
    artifact_kind: str


class PendingDetailsV6(_ErrorContract):
    operation_key: str
    receipt_pin: ArtifactPinV1


class CorrelationDetailsV6(_ErrorContract):
    correlation_ref: str


ErrorDetailsV6 = (
    EmptyDetailsV6
    | FieldDetailsV6
    | ResourceDetailsV6
    | AuthorityDetailsV6
    | ConflictDetailsV6
    | HashDetailsV6
    | PendingDetailsV6
    | CorrelationDetailsV6
)


class WorkbenchErrorCurrentContextV6(_ErrorContract):
    collection_pin: ArtifactPinV1 | None
    collection_revision: Annotated[int, Field(ge=0)] | None
    preparation_pin: ArtifactPinV1 | None
    preparation_revision: Annotated[int, Field(ge=0)] | None
    selection_pin: ArtifactPinV1 | None
    selection_revision: Annotated[int, Field(ge=0)] | None
    binding_set_pin: ArtifactPinV1 | None
    binding_set_revision: Annotated[int, Field(ge=0)] | None
    cursor_pin: ArtifactPinV1 | None
    material_projection_hash: Sha256 | None
    material_context_revision: Annotated[int, Field(ge=0)] | None


class _ActionErrorV6(_ErrorContract):
    schema_version: Literal["workbench_action_error.v6"]
    message: Annotated[str, Field(min_length=1)]
    retryable: bool
    scoped_not_found: bool
    current_context: WorkbenchErrorCurrentContextV6

    @property
    def http_status(self) -> int:
        # Only codes in the closed discriminated union reach this policy.
        for status, codes in _HTTP_EXCEPTIONS.items():
            if self.code in codes:
                return status
        return 409

    @property
    def chat_status(self) -> Literal["PENDING", "BLOCKED"]:
        return "PENDING" if self.http_status == 202 else "BLOCKED"

    @model_validator(mode="after")
    def verify_transport_flags(self):
        if self.retryable != (
            self.http_status in {202, 503}
        ) or self.scoped_not_found != (self.http_status == 404):
            raise ValueError("error flags must match the accepted transport policy")
        return self


class _EmptyActionErrorV6(_ActionErrorV6):
    code: Literal[
        "AUTHORITY_CONTEXT_REQUIRED",
        "WORKSPACE_REQUIRED",
        "FORBIDDEN",
        "AUTHORITY_REQUIRED",
        "AUTHORITY_REVOKED",
        "AUTHORITY_NOT_EFFECTIVE",
        "AUTHORITY_TRANSACTION_NESTED",
        "AUTHORITY_TRANSACTION_CLOSED",
        "SOURCE_NOT_PREPARED",
        "SOURCE_PREPARATION_FAILED",
        "TARGET_UNSUPPORTED",
        "BINDING_NOT_FOUND",
        "PREPARATION_RECONCILIATION_FAILED",
        "MISSING_POLICY_BASIS",
        "SCOPE_BRIDGE_REQUIRED",
        "TENDER_COMPATIBILITY_NOT_CERTIFIED",
        "SCOPE_BRIDGE_FORBIDDEN",
        "EXPORT_ACCESS_REQUIRED",
        "EXPORT_CLEANUP_REFUSED",
        "EXPORT_STAGE_UNLINK_FAILED",
        "EXPORT_STAGE_FSYNC_FAILED",
        "ORIGIN_INCOMPLETE",
        "FALSE_NEGATIVE_ANCHOR_REQUIRED",
        "REPORT_ABSENCE_INVALID",
        "REVIEW_OUTCOME_NOT_LEARNING",
        "UNSUPPORTED_REPAIR",
        "INDEPENDENT_SUITE_REQUIRED",
        "REQUIRED_CASE_MISSING",
        "INCOMPATIBLE_SUBJECT",
        "INCOMPLETE_CLOSURE",
        "UNSUPPORTED_SEMANTICS",
        "MISSING_COMPARABLE_BEHAVIOR",
        "INCOMPATIBLE_WORLD",
        "INCOMPATIBLE_LINEAGE",
        "INCOMPATIBLE_TYPE",
        "NEEDS_MORE_EVIDENCE",
        "INCONCLUSIVE",
        "INCOMPATIBLE_LIFECYCLE",
        "INCOMPATIBLE_COMPARISON_PAIR",
        "MISSING_ANCHOR",
        "ELIGIBILITY_REVOKED",
    ]
    details: EmptyDetailsV6


class _FieldActionErrorV6(_ActionErrorV6):
    code: Literal[
        "INVALID_MEDIA_TYPE",
        "INVALID_BYTE_LENGTH",
        "INVALID_LIFECYCLE_ROLE",
        "UNSUPPORTED_SOURCE_PREPARATION",
        "EXPORT_FORMAT_UNSUPPORTED",
        "EXPORT_PROFILE_NOT_APPROVED",
        "MALFORMED_COMMAND",
        "UNSUPPORTED_SCHEMA_VERSION",
        "UNSUPPORTED_COMMAND",
        "NEGOTIATION_REQUIRED",
        "UNSUPPORTED_MEDIA",
    ]
    details: FieldDetailsV6


class _ResourceActionErrorV6(_ActionErrorV6):
    code: Literal[
        "ARTIFACT_NOT_FOUND",
        "ARTIFACT_INACCESSIBLE",
        "ARTIFACT_RETIRED",
        "ACTIVATION_SNAPSHOT_NOT_FOUND",
        "CERTIFIED_PLAN_NOT_FOUND",
        "CONTEXT_NOT_FOUND",
        "TARGET_NOT_FOUND",
        "EXPORT_STAGE_NOT_FOUND",
        "EXPORT_NOT_FOUND",
        "REVIEW_AUTHORITY_NOT_FOUND",
        "PIN_NOT_FOUND",
        "MISSING_EXECUTABLE_PLAN",
        "UNRESOLVABLE_REF",
        "SCOPED_NOT_FOUND",
    ]
    details: ResourceDetailsV6


class _AuthorityActionErrorV6(_ActionErrorV6):
    code: Literal[
        "AUTHORITY_SCOPE_MISMATCH",
        "ARTIFACT_SCOPE_MISMATCH",
        "TARGET_SCOPE_MISMATCH",
        "WORKSPACE_BINDING_MISMATCH",
        "REVIEW_AUTHORITY_MISMATCH",
        "REVIEW_AUTHORITY_EXPIRED",
        "FORBIDDEN_SCOPE",
        "REVIEW_FORBIDDEN",
    ]
    details: AuthorityDetailsV6


class _ConflictActionErrorV6(_ActionErrorV6):
    code: Literal[
        "ARTIFACT_METADATA_CONFLICT",
        "ARTIFACT_PIN_MISMATCH",
        "STALE_COLLECTION_HEAD",
        "STALE_PREPARATION_HEAD",
        "STALE_SELECTION_HEAD",
        "STALE_BINDING_SET_HEAD",
        "STALE_MATERIAL_CONTEXT",
        "REQUEST_CONFLICT",
        "PERSISTENCE_CONFLICT",
        "ACTIVATION_EVENT_PIN_MISMATCH",
        "ACTIVATION_EVENT_CHAIN_INVALID",
        "CERTIFIED_PLAN_RULE_CLOSURE_MISMATCH",
        "CONTEXT_HEAD_MISMATCH",
        "CONTEXT_REVISION_MISMATCH",
        "SOURCE_REGION_INTEGRITY_ERROR",
        "SCOPE_BRIDGE_REQUEST_CONFLICT",
        "EXPORT_REQUEST_CONFLICT",
        "EXPORT_PREDECESSOR_MISMATCH",
        "EXPORT_STAGE_STALE",
        "EXPORT_ALREADY_PUBLISHED",
        "REPAIR_INPUT_MISMATCH",
        "ORACLE_INPUT_MISMATCH",
        "EVALUATOR_ENVIRONMENT_MISMATCH",
        "OWNER_OUTCOME_MISMATCH",
        "STALE_REVISION",
        "PREDECESSOR_MISMATCH",
        "PIN_MISMATCH",
        "STALE_CONTEXT",
        "STALE_CURSOR",
        "STALE_PROJECTION",
        "STALE_SELECTION",
        "STALE_BINDING",
        "REVISION_CONFLICT",
    ]
    details: ConflictDetailsV6


class _HashActionErrorV6(_ActionErrorV6):
    code: Literal[
        "ARTIFACT_HASH_MISMATCH",
        "ACTIVATION_SNAPSHOT_HASH_MISMATCH",
        "RULE_VERSION_SET_HASH_MISMATCH",
        "CERTIFIED_PLAN_HASH_MISMATCH",
        "CONTEXT_HASH_MISMATCH",
        "TARGET_HASH_MISMATCH",
        "SCOPE_BRIDGE_LEGACY_HASH_MISMATCH",
        "EXPORT_ARTIFACT_HASH_MISMATCH",
        "HASH_MISMATCH",
    ]
    details: HashDetailsV6


class _PendingActionErrorV6(_ActionErrorV6):
    code: Literal[
        "AUTHORITY_RELEASE_UNCONFIRMED",
        "EXPORT_RECOVERY_PENDING",
        "OWNER_OUTCOME_PENDING",
        "FUTURE_RUN_NOT_COMPLETE",
        "RECOVERY_PENDING",
    ]
    details: PendingDetailsV6


class _CorrelationActionErrorV6(_ActionErrorV6):
    code: Literal["TEMPORARY_STORAGE_FAILURE", "INTERNAL_ERROR"]
    details: CorrelationDetailsV6


_HTTP_EXCEPTIONS = {
    202: frozenset(get_args(_PendingActionErrorV6.model_fields["code"].annotation)),
    401: {"AUTHORITY_CONTEXT_REQUIRED"},
    403: {
        "FORBIDDEN",
        "AUTHORITY_REQUIRED",
        "AUTHORITY_REVOKED",
        "AUTHORITY_NOT_EFFECTIVE",
        "SCOPE_BRIDGE_FORBIDDEN",
        "EXPORT_ACCESS_REQUIRED",
        "REVIEW_AUTHORITY_MISMATCH",
        "REVIEW_AUTHORITY_EXPIRED",
        "FORBIDDEN_SCOPE",
        "REVIEW_FORBIDDEN",
    },
    404: frozenset(get_args(_ResourceActionErrorV6.model_fields["code"].annotation))
    - {"MISSING_EXECUTABLE_PLAN"},
    400: {
        "INVALID_MEDIA_TYPE",
        "INVALID_BYTE_LENGTH",
        "INVALID_LIFECYCLE_ROLE",
        "SOURCE_PREPARATION_FAILED",
        "MALFORMED_COMMAND",
    },
    422: {
        "UNSUPPORTED_SOURCE_PREPARATION",
        "TARGET_UNSUPPORTED",
        "EXPORT_FORMAT_UNSUPPORTED",
        "EXPORT_PROFILE_NOT_APPROVED",
        "ORIGIN_INCOMPLETE",
        "FALSE_NEGATIVE_ANCHOR_REQUIRED",
        "REPORT_ABSENCE_INVALID",
        "REVIEW_OUTCOME_NOT_LEARNING",
        "UNSUPPORTED_REPAIR",
        "INDEPENDENT_SUITE_REQUIRED",
        "REQUIRED_CASE_MISSING",
        "MISSING_EXECUTABLE_PLAN",
        "UNSUPPORTED_SEMANTICS",
        "MISSING_COMPARABLE_BEHAVIOR",
        "INCOMPATIBLE_SUBJECT",
        "INCOMPLETE_CLOSURE",
        "INCOMPATIBLE_WORLD",
        "INCOMPATIBLE_LINEAGE",
        "INCOMPATIBLE_TYPE",
        "UNSUPPORTED_SCHEMA_VERSION",
        "UNSUPPORTED_COMMAND",
        "UNSUPPORTED_MEDIA",
        "INCOMPATIBLE_LIFECYCLE",
        "INCOMPATIBLE_COMPARISON_PAIR",
        "MISSING_ANCHOR",
        "NEEDS_MORE_EVIDENCE",
        "INCONCLUSIVE",
        "ELIGIBILITY_REVOKED",
    },
    503: {"TEMPORARY_STORAGE_FAILURE"},
    500: {"INTERNAL_ERROR"},
}


class WorkbenchActionErrorV6(
    RootModel[
        Annotated[
            _EmptyActionErrorV6
            | _FieldActionErrorV6
            | _ResourceActionErrorV6
            | _AuthorityActionErrorV6
            | _ConflictActionErrorV6
            | _HashActionErrorV6
            | _PendingActionErrorV6
            | _CorrelationActionErrorV6,
            Field(discriminator="code"),
        ]
    ]
):
    model_config = ConfigDict(frozen=True)
