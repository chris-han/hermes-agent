"""Phase 1 knowledge feature-pack contracts.

These contracts intentionally keep model and judge output in proposal fields.
Accepted graph state and deterministic verdicts are represented separately.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Recognized Phase 1 runtime-profile provider identifiers. Adding a new value
# requires a contract version bump; existing identifiers remain stable.
RUNTIME_PROFILE_DOCUMENT_EXTRACTION_PROVIDERS = (
    "legacy",
    "langextract",
    "semantica",
)


RUNTIME_PROFILE_GRAPH_ENRICHMENT_PROVIDERS = (
    "semantica",
    "none",
)


RUNTIME_PROFILE_GRAPH_RUNTIMES = (
    "falkordblite",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return f"sha256:{sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphBackendContract(StrictContract):
    backend: Literal["falkordb"]
    adapter_version: str = Field(min_length=1)
    graph_versioning: Literal[True]
    historical_export: Literal[True]


class RuntimeProfile(StrictContract):
    """Typed Phase 1 release-profile identifier.

    A manifest without ``runtime_profile`` is **legacy** and remains valid for
    replay/compatibility. ``has_runtime_profile`` is the gate-neutral
    predicate that downstream gates and consumers may consult; whether the
    presence of a ``runtime_profile`` is sufficient for any specific
    release-decision policy is owned by the consumer.
    """

    profile_id: str = Field(min_length=1)
    profile_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    document_extraction_providers: tuple[
        Literal["legacy", "langextract", "semantica"], ...
    ]
    graph_enrichment_providers: tuple[Literal["semantica", "none"], ...]
    human_grounding_enabled: bool
    graph_runtime: str = Field(min_length=1)

    @field_validator("document_extraction_providers")
    @classmethod
    def require_provider(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError(
                "runtime_profile must declare at least one document_extraction_providers entry"
            )
        known = set(RUNTIME_PROFILE_DOCUMENT_EXTRACTION_PROVIDERS)
        unknown = [item for item in value if item not in known]
        if unknown:
            raise ValueError(f"unknown document_extraction_providers: {unknown}")
        return tuple(value)

    @field_validator("graph_enrichment_providers")
    @classmethod
    def require_graph_provider(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        known = set(RUNTIME_PROFILE_GRAPH_ENRICHMENT_PROVIDERS)
        unknown = [item for item in value if item not in known]
        if unknown:
            raise ValueError(f"unknown graph_enrichment_providers: {unknown}")
        return tuple(value)


class KnowledgeFeaturePackManifest(StrictContract):
    pack_id: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    dependencies: tuple[str, ...] = ()
    feature_flags: tuple[str, ...] = ()
    contract_versions: dict[str, str]
    dataset_release_ids: tuple[str, ...]
    graph_backend: GraphBackendContract
    evaluator_config_version: str = Field(min_length=1)
    compatibility_mode: str = Field(min_length=1)
    activation_policy: str = Field(min_length=1)
    rollback_policy: str = Field(min_length=1)
    runtime_profile: RuntimeProfile | None = None

    @property
    def has_runtime_profile(self) -> bool:
        """Gate-neutral presence predicate.

        A manifest carries an explicit ``runtime_profile`` if and only if this
        returns ``True``. Legacy manifests (no ``runtime_profile``) are
        backward-readable but cannot satisfy any downstream gate that requires
        a profile; the gate semantics themselves are owned by the consumer.
        """

        return self.runtime_profile is not None

    @field_validator("contract_versions")
    @classmethod
    def require_core_contracts(cls, value: dict[str, str]) -> dict[str, str]:
        required = {
            "learning_event",
            "human_label_event",
            "feedback_dataset_release",
            "evaluator_configuration",
            "graph_delta",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"missing core contract versions: {missing}")
        if any(not version for version in value.values()):
            raise ValueError("contract versions must be non-empty")
        return value

    @field_validator("dataset_release_ids")
    @classmethod
    def require_dataset_release(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one dataset release is required")
        return value


class LearningEventType(str, Enum):
    INGESTION = "ingestion"
    MACHINE_PROPOSAL = "machine_proposal"
    HUMAN_ACCEPT = "human_accept"
    HUMAN_EDIT = "human_edit"
    HUMAN_REJECT = "human_reject"
    HUMAN_UNCERTAIN = "human_uncertain"
    JUDGE_CRITIQUE = "judge_critique"
    GRAPH_DELTA = "graph_delta"
    DOWNSTREAM_OUTCOME = "downstream_outcome"


class HumanDecision(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class ProcessLabel(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    EDITED = "edited"
    MISSING = "missing"
    UNCERTAIN = "uncertain"


class MachineProposal(StrictContract):
    proposal_id: str
    provider: str
    model: str
    prompt_version: str
    prompt_hash: str
    parameters: dict[str, Any]
    request_id: str | None = None
    run_id: str
    raw_output_ref: str
    raw_output_hash: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class HumanLabel(StrictContract):
    decision: HumanDecision
    certainty: str
    reason_code: str
    justification: str = Field(min_length=1)
    process_labels: dict[
        Literal["entity", "relation", "concept", "condition", "path", "verdict"],
        ProcessLabel,
    ] = Field(default_factory=dict)


class LearningEventEnvelope(StrictContract):
    event_id: str
    case_id: str
    event_type: LearningEventType
    occurred_at: datetime
    source_document_hash: str
    source_anchor_refs: tuple[str, ...]
    graph_version: str
    rule_version: str
    evaluation_corpus_version: str
    actor_ref: str | None = None
    machine_proposal: MachineProposal | None = None
    human_label: HumanLabel | None = None
    judge_critique: MachineProposal | None = None
    graph_delta_ref: str | None = None
    downstream_outcome: dict[str, Any] | None = None
    prior_event_hash: str | None = None
    reviewed_candidate_graph_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    reviewed_evidence_ref: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    reviewed_evidence_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    reviewed_graph_delta_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )

    @field_validator("occurred_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def require_event_payload(self) -> "LearningEventEnvelope":
        human_types = {
            LearningEventType.HUMAN_ACCEPT,
            LearningEventType.HUMAN_EDIT,
            LearningEventType.HUMAN_REJECT,
            LearningEventType.HUMAN_UNCERTAIN,
        }
        if self.event_type in human_types and self.human_label is None:
            raise ValueError("human events require human_label")
        if self.event_type == LearningEventType.MACHINE_PROPOSAL and self.machine_proposal is None:
            raise ValueError("machine proposal events require machine_proposal")
        if self.event_type == LearningEventType.JUDGE_CRITIQUE and self.judge_critique is None:
            raise ValueError("judge critique events require judge_critique")
        if self.event_type == LearningEventType.GRAPH_DELTA and not self.graph_delta_ref:
            raise ValueError("graph delta events require graph_delta_ref")
        review_pins = (
            self.reviewed_candidate_graph_hash,
            self.reviewed_evidence_ref,
            self.reviewed_evidence_hash,
        )
        if any(value is not None for value in review_pins):
            if not all(value is not None for value in review_pins):
                raise ValueError("review commitment fields must be all null or all non-null")
            if self.event_type not in {
                LearningEventType.HUMAN_ACCEPT,
                LearningEventType.HUMAN_EDIT,
            }:
                raise ValueError("review commitments belong only to accepted/edited events")
        if self.reviewed_graph_delta_hash is not None:
            if (
                self.event_type is not LearningEventType.HUMAN_EDIT
                or self.graph_delta_ref is None
            ):
                raise ValueError("reviewed GraphDelta hash requires an edited event and ref")
        return self

    def event_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class SplitPolicy(StrictContract):
    version: str
    grouping_keys: tuple[Literal["source_document", "project", "entity_family", "time_window"], ...]
    salt: str

    @field_validator("grouping_keys")
    @classmethod
    def require_leakage_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        required = {"source_document", "project", "entity_family"}
        if not required.issubset(value):
            raise ValueError("split policy must group document, project, and entity family")
        return value


class DatasetMember(StrictContract):
    event_id: str
    source_document: str
    project: str
    entity_family: str
    time_window: str
    split: Literal["train", "validation", "hidden_test"]

    def grouping_key(self) -> tuple[str, str, str, str]:
        return (self.source_document, self.project, self.entity_family, self.time_window)


class FeedbackDatasetRelease(StrictContract):
    dataset_key: str
    loop_scope: tuple[str, ...]
    members: tuple[DatasetMember, ...]
    split_policy: SplitPolicy
    annotation_schema_version: str
    adjudication_policy_version: str
    inclusion_policy_version: str
    exclusion_policy_version: str
    leakage_checks: tuple[str, ...]

    @model_validator(mode="after")
    def reject_group_leakage(self) -> "FeedbackDatasetRelease":
        splits_by_group: dict[tuple[str, str, str, str], set[str]] = {}
        for member in self.members:
            splits_by_group.setdefault(member.grouping_key(), set()).add(member.split)
        leaked = [group for group, splits in splits_by_group.items() if len(splits) > 1]
        if leaked:
            raise ValueError(f"dataset split leakage detected for groups: {leaked}")
        return self

    def source_event_manifest_hash(self) -> str:
        return content_hash(sorted(member.event_id for member in self.members))

    def release_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload["members"] = sorted(payload["members"], key=lambda item: item["event_id"])
        return content_hash(payload)

    def release_id(self) -> str:
        return f"fds_{self.release_hash().removeprefix('sha256:')[:24]}"


ERROR_TAXONOMY = (
    "false_positive",
    "false_negative",
    "wrong_entity_grounding",
    "wrong_relation",
    "wrong_concept",
    "wrong_reason",
    "missing_relation",
    "missing_condition",
    "missing_exception",
    "invalid_path",
    "ontology_error",
)


class EvaluatorConfiguration(StrictContract):
    evaluator_key: str
    version: str
    code_hash: str
    prompt_hash: str | None = None
    metric_semantics: dict[str, str]
    thresholds: dict[str, float]
    error_taxonomy: tuple[str, ...] = ERROR_TAXONOMY
    human_agreement_metrics: tuple[str, ...] = (
        "raw_agreement",
        "cohens_kappa",
        "weighted_kappa",
    )

    @field_validator("error_taxonomy")
    @classmethod
    def require_frozen_taxonomy(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(value) != ERROR_TAXONOMY:
            raise ValueError("Phase 1 error taxonomy must match the frozen contract")
        return value

    def configuration_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class HistoricalDecisionContext(StrictContract):
    case_id: str
    source_document_hash: str
    source_anchor_refs: tuple[str, ...]
    machine_proposal: MachineProposal | None
    graph_version: str
    rule_version: str
    human_label: HumanLabel | None
    graph_delta_ref: str | None
    evaluation_corpus_version: str
    event_ids: tuple[str, ...]
    event_hashes: tuple[str, ...]
