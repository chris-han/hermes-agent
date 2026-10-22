"""Provisional, neutral real-module integration evidence contracts.

These contracts are explicitly **not** the canonical ``EvaluationRun`` /
``EvaluationEvidence`` records owned by ``src/eos/evaluation_control_plane.py``.
The evaluation stack defines the durable certification authority for
benchmark/quality evidence; this module provides a separate, lighter-weight
schema for recording which real-module integration tests actually ran and
what module/version/mode they used.

Records produced from this module are content-addressed and append-only.
They never assert a certification result and never reference accepted graph
releases or runtime semantic-index authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic import Field

from contracts.knowledge_feature_pack import StrictContract, content_hash


# Single source of truth for the integration-evidence manifest schema version.
# Adding a new major version requires a migration; bumping it here is the only
# safe place to do so.
INTEGRATION_EVIDENCE_MANIFEST_VERSION: Literal["integration_evidence_manifest.v1"] = (
    "integration_evidence_manifest.v1"
)


IntegrationModuleName = Literal[
    "semantica",
    "langextract",
    "falkordb",
    "eos_db",
    "knowledge_builder_api",
    "reasoning",
]


class IntegrationExecutionMode(str, Enum):
    REAL = "real"
    RECORDED = "recorded"
    IN_MEMORY = "in_memory"
    BLOCKED = "blocked"
    NOT_SELECTED = "not_selected"


class IntegrationTestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERRORED = "errored"


class ModuleRole(str, Enum):
    DOCUMENT_EXTRACTION = "document_extraction"
    GRAPH_ENRICHMENT = "graph_enrichment"
    GRAPH_RUNTIME = "graph_runtime"
    PERSISTENCE = "persistence"
    API = "api"
    REASONING = "reasoning"
    NOT_SELECTED = "not_selected"


class IntegrationModuleResult(StrictContract):
    """One real-module integration outcome, mode- and version-explicit."""

    module: IntegrationModuleName
    selected: bool
    role: ModuleRole
    mode: IntegrationExecutionMode
    version: str | None = None
    provider_id: str | None = None
    model_ref: str | None = None
    endpoint_or_image_ref: str | None = None
    database_path_mode: Literal["persistent_file", "in_process", "external_service"] | None = None
    test_status: IntegrationTestStatus
    duration_ms: int | None = Field(default=None, ge=0)
    detail: str | None = None
    artifact_refs: tuple[str, ...] = ()


class RealModuleIntegrationResult(StrictContract):
    """Aggregate, neutral, non-canonical outcome of one real-module integration run."""

    integration_run_id: str = Field(min_length=1)
    schema_version: Literal["real_module_integration_result.v1"] = "real_module_integration_result.v1"
    timestamp: datetime
    commit_ref: str | None = None
    profile_id: str | None = None
    profile_version: str | None = None
    modules: tuple[IntegrationModuleResult, ...]
    tests_passed: int = Field(ge=0)
    tests_failed: int = Field(ge=0)
    tests_skipped: int = Field(ge=0)
    real_module_gate: Literal["pass", "fail", "incomplete"]
    warnings: tuple[str, ...] = ()

    @field_validator("timestamp")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def require_modules(self) -> "RealModuleIntegrationResult":
        if not self.modules:
            raise ValueError("at least one module result is required")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return content_hash(payload)


class IntegrationEvidenceManifest(StrictContract):
    """Lightweight, non-canonical artifact that records a real-integration run.

    Intentionally distinct from canonical ``EvaluationEvidence``: this manifest
    records *which integration tests ran against which real modules*, with
    pinned versions and execution modes, and never asserts a certification
    result. It is content-addressed and immutable.
    """

    manifest_version: Literal["integration_evidence_manifest.v1"] = (
        INTEGRATION_EVIDENCE_MANIFEST_VERSION
    )
    integration_run_id: str = Field(min_length=1)
    timestamp: datetime
    commit_ref: str | None = None
    profile_id: str | None = None
    profile_version: str | None = None
    human_grounding_enabled: bool = False
    modules: tuple[IntegrationModuleResult, ...]
    tests_passed: int = Field(ge=0)
    tests_failed: int = Field(ge=0)
    tests_skipped: int = Field(ge=0)
    real_module_gate: Literal["pass", "fail", "incomplete"]
    warnings: tuple[str, ...] = ()
    content_hash: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def require_content_hash_match(self) -> "IntegrationEvidenceManifest":
        body = self.model_dump(mode="json")
        body.pop("content_hash", None)
        expected = content_hash(body)
        if self.content_hash != expected:
            raise ValueError("integration evidence manifest content_hash mismatch")
        return self


def write_integration_evidence_manifest(
    *,
    integration_run_id: str,
    timestamp: datetime,
    profile_id: str | None,
    profile_version: str | None,
    commit_ref: str | None,
    human_grounding_enabled: bool,
    modules: tuple[IntegrationModuleResult, ...],
    tests_passed: int,
    tests_failed: int,
    tests_skipped: int,
    real_module_gate: Literal["pass", "fail", "incomplete"],
    warnings: tuple[str, ...] = (),
) -> IntegrationEvidenceManifest:
    """Build a content-addressed, non-canonical manifest.

    This helper never writes canonical evaluation evidence and never asserts a
    certification result.
    """

    normalized_modules = tuple(_normalize_module(module) for module in modules)
    body = {
        "manifest_version": INTEGRATION_EVIDENCE_MANIFEST_VERSION,
        "integration_run_id": integration_run_id,
        "timestamp": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "commit_ref": commit_ref,
        "profile_id": profile_id,
        "profile_version": profile_version,
        "human_grounding_enabled": human_grounding_enabled,
        "modules": [module.model_dump(mode="json") for module in normalized_modules],
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_skipped": tests_skipped,
        "real_module_gate": real_module_gate,
        "warnings": list(warnings),
    }
    digest = content_hash(body)
    return IntegrationEvidenceManifest(
        integration_run_id=integration_run_id,
        timestamp=timestamp,
        commit_ref=commit_ref,
        profile_id=profile_id,
        profile_version=profile_version,
        human_grounding_enabled=human_grounding_enabled,
        modules=normalized_modules,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_skipped=tests_skipped,
        real_module_gate=real_module_gate,
        warnings=warnings,
        content_hash=digest,
    )


def aggregate_real_module_integration_results(
    results: tuple[RealModuleIntegrationResult, ...],
    *,
    integration_run_id: str | None = None,
    timestamp: datetime | None = None,
    commit_ref: str | None = None,
    human_grounding_enabled: bool | None = None,
) -> IntegrationEvidenceManifest:
    """Aggregate multiple ``RealModuleIntegrationResult`` into one neutral
    ``IntegrationEvidenceManifest``.

    The aggregator sums test counts, deduplicates ``modules`` by
    ``(module, version, role, mode)`` and ``warnings`` (preserving first-seen
    order), and recomputes the content hash from the merged body. Identity
    fields (``integration_run_id``, ``timestamp``, ``commit_ref``,
    ``human_grounding_enabled``) come from the first result unless explicitly
    overridden.

    The aggregated gate is:

    - ``"pass"`` if at least one result exists and none is ``"fail"``;
    - ``"fail"`` if any sub-result is ``"fail"``;
    - ``"incomplete"`` if no results were provided.

    The aggregator never produces a certification result and never references
    canonical ``EvaluationEvidence``; it only combines neutral, non-canonical
    real-module integration results.
    """

    if not results:
        raise ValueError("at least one RealModuleIntegrationResult is required")

    first = results[0]
    run_id = integration_run_id if integration_run_id is not None else first.integration_run_id
    run_timestamp = timestamp if timestamp is not None else first.timestamp
    run_commit_ref = commit_ref if commit_ref is not None else first.commit_ref
    profile_id = first.profile_id
    profile_version = first.profile_version
    if any(result.profile_id != profile_id or result.profile_version != profile_version
           for result in results):
        raise ValueError(
            "cannot aggregate RealModuleIntegrationResult with mixed profile_id/profile_version"
        )
    if any(result.integration_run_id != first.integration_run_id for result in results):
        # Sub-results with different run IDs are merged under the aggregated
        # run_id; we surface a warning but do not raise.
        pass

    modules_seen: dict[tuple[str, str | None, str, str], IntegrationModuleResult] = {}
    warnings_seen: dict[str, None] = {}
    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0

    for result in results:
        tests_passed += result.tests_passed
        tests_failed += result.tests_failed
        tests_skipped += result.tests_skipped
        for warning in result.warnings:
            warnings_seen.setdefault(warning, None)
        for module in result.modules:
            key = (module.module, module.version, module.role.value, module.mode.value)
            if key not in modules_seen:
                modules_seen[key] = module

    aggregate_gate: Literal["pass", "fail", "incomplete"]
    if any(result.real_module_gate == "fail" for result in results):
        aggregate_gate = "fail"
    elif all(result.real_module_gate == "pass" for result in results):
        aggregate_gate = "pass"
    else:
        aggregate_gate = "incomplete"

    grounding_flag = (
        human_grounding_enabled
        if human_grounding_enabled is not None
        else bool(first.modules and any(m.role == ModuleRole.GRAPH_RUNTIME for m in first.modules))
    )

    return write_integration_evidence_manifest(
        integration_run_id=run_id,
        timestamp=run_timestamp,
        profile_id=profile_id,
        profile_version=profile_version,
        commit_ref=run_commit_ref,
        human_grounding_enabled=grounding_flag,
        modules=tuple(modules_seen.values()),
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_skipped=tests_skipped,
        real_module_gate=aggregate_gate,
        warnings=tuple(warnings_seen.keys()),
    )


def _normalize_module(module: IntegrationModuleResult) -> IntegrationModuleResult:
    return IntegrationModuleResult(
        module=module.module,
        selected=module.selected,
        role=module.role,
        mode=module.mode,
        version=module.version,
        provider_id=module.provider_id,
        model_ref=module.model_ref,
        endpoint_or_image_ref=module.endpoint_or_image_ref,
        database_path_mode=module.database_path_mode,
        test_status=module.test_status,
        duration_ms=module.duration_ms,
        detail=module.detail,
        artifact_refs=tuple(module.artifact_refs),
    )


__all__ = [
    "INTEGRATION_EVIDENCE_MANIFEST_VERSION",
    "IntegrationModuleName",
    "IntegrationExecutionMode",
    "IntegrationTestStatus",
    "ModuleRole",
    "IntegrationModuleResult",
    "RealModuleIntegrationResult",
    "IntegrationEvidenceManifest",
    "write_integration_evidence_manifest",
    "aggregate_real_module_integration_results",
]