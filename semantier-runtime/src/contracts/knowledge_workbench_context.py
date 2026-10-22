"""Strict Python mirror for the cross-boundary KnowledgeWorkbenchContext.v4."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
V4_REF_HASH_PAIRS = (
    ("source_semantic_witness_set_ref", "source_semantic_witness_set_hash"),
    ("selected_source_semantic_witness_ref", "selected_source_semantic_witness_hash"),
    ("selected_exemplar_evaluation_ref", "selected_exemplar_evaluation_hash"),
    ("selected_canonical_pattern_ref", "selected_canonical_pattern_hash"),
    ("selected_observed_hypothesis_ref", "selected_observed_hypothesis_hash"),
    ("selected_finding_derivation_ref", "selected_finding_derivation_hash"),
)


class ActiveFunctionV4(BaseModel):
    """Exact Python mirror of the TypeScript ActiveFunction contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: str = Field(min_length=1)
    function: str = Field(min_length=1)
    tab: str = Field(min_length=1)


class UserIntentV4(BaseModel):
    """Exact Python mirror of the TypeScript UserIntent contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "open_function",
        "inspect_selection",
        "review_candidate",
        "trace_evidence",
        "inspect_finding",
        "compare_releases",
        "evaluate_graph",
    ]
    source: Literal["web_ui", "chat", "navigation"]
    explicitness: Literal["implicit", "explicit"]
    target_type: Literal[
        "node", "edge", "candidate", "evidence", "finding", "release"
    ] | None = Field(alias="targetType")
    target_ids: list[str] = Field(alias="targetIds")


class KnowledgeWorkbenchContextV4(BaseModel):
    """Authoritative mirror used only because the Python Chat boundary validates it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["knowledge_workbench_context.v4"] = Field(
        alias="schemaVersion"
    )
    active_function: ActiveFunctionV4 | None = Field(
        default=None, alias="activeFunction"
    )
    user_intent: UserIntentV4 | None = Field(default=None, alias="userIntent")
    graph_ref: str | None = Field(default=None, alias="graphRef")
    graph_version: str | None = Field(default=None, alias="graphVersion")
    graph_hash: str | None = Field(default=None, alias="graphHash")
    authority_state: Literal["candidate", "authoritative"] | None = Field(
        default=None, alias="authorityState"
    )
    run_mode: Literal["evaluation_baseline", "authoritative"] | None = Field(
        default=None, alias="runMode"
    )
    candidate_graph_id: str | None = Field(alias="candidateGraphId")
    accepted_release_id: str | None = Field(alias="acceptedReleaseId")
    accepted_release_version: str | None = Field(alias="acceptedReleaseVersion")
    selected_node_ids: list[str] = Field(alias="selectedNodeIds")
    selected_edge_ids: list[str] = Field(alias="selectedEdgeIds")
    selected_rule_ids: list[str] = Field(alias="selectedRuleIds")
    selected_candidate_id: str | None = Field(default=None, alias="selectedCandidateId")
    selected_evidence_refs: list[str] = Field(
        default_factory=list, alias="selectedEvidenceRefs"
    )
    active_source_identity_ref: str | None = Field(
        default=None, alias="activeSourceIdentityRef"
    )
    source_anchors: list[dict[str, Any]] = Field(alias="sourceAnchors")
    governance_state: Literal[
        "candidate", "validated", "approved", "active", "deprecated", "rejected"
    ] = Field(alias="governanceState")
    has_accepted_release: bool = Field(alias="hasAcceptedRelease")
    extraction_run_id: str | None = Field(alias="extractionRunId")
    provider_ref: str | None = Field(alias="providerRef")
    provider_commit: str | None = Field(alias="providerCommit")
    target_evidence_ref: str | None = Field(default=None, alias="targetEvidenceRef")
    active_rule_version_id: str | None = Field(
        default=None, alias="activeRuleVersionId"
    )
    graph_rule_id: str | None = Field(default=None, alias="graphRuleId")
    origin_evidence_ref: str | None = Field(default=None, alias="originEvidenceRef")
    schema_candidate_ref: str | None = Field(default=None, alias="schemaCandidateRef")
    schema_candidate_hash: str | None = Field(default=None, alias="schemaCandidateHash")
    schema_decision_ref: str | None = Field(default=None, alias="schemaDecisionRef")
    schema_decision_hash: str | None = Field(default=None, alias="schemaDecisionHash")
    selected_schema_ambiguity_ids: list[str] = Field(
        default_factory=list, alias="selectedSchemaAmbiguityIds"
    )
    gate_operating_mode: Literal["assisted", "auto_pilot"] | None = Field(
        default=None, alias="gateOperatingMode"
    )
    gate_fallback_reason: str | None = Field(default=None, alias="gateFallbackReason")
    canonical_source_ir_ref: str | None = Field(alias="canonicalSourceIrRef")
    canonical_source_ir_hash: str | None = Field(alias="canonicalSourceIrHash")
    graph_intent_brief_ref: str | None = Field(alias="graphIntentBriefRef")
    graph_intent_brief_hash: str | None = Field(alias="graphIntentBriefHash")
    graph_intent_review_state: Literal[
        "proposed", "accepted", "edited", "rejected"
    ] | None = Field(alias="graphIntentReviewState")
    graph_intent_selected_question_ids: list[str] = Field(
        alias="graphIntentSelectedQuestionIds"
    )
    graph_intent_unresolved_material_question_ids: list[str] = Field(
        alias="graphIntentUnresolvedMaterialQuestionIds"
    )
    source_semantic_witness_set_ref: str | None = Field(
        alias="sourceSemanticWitnessSetRef", min_length=1
    )
    source_semantic_witness_set_hash: str | None = Field(
        alias="sourceSemanticWitnessSetHash", pattern=SHA256_PATTERN
    )
    selected_source_semantic_witness_ref: str | None = Field(
        alias="selectedSourceSemanticWitnessRef", min_length=1
    )
    selected_source_semantic_witness_hash: str | None = Field(
        alias="selectedSourceSemanticWitnessHash", pattern=SHA256_PATTERN
    )
    selected_exemplar_evaluation_ref: str | None = Field(
        alias="selectedExemplarEvaluationRef", min_length=1
    )
    selected_exemplar_evaluation_hash: str | None = Field(
        alias="selectedExemplarEvaluationHash", pattern=SHA256_PATTERN
    )
    selected_canonical_pattern_ref: str | None = Field(
        alias="selectedCanonicalPatternRef", min_length=1
    )
    selected_canonical_pattern_hash: str | None = Field(
        alias="selectedCanonicalPatternHash", pattern=SHA256_PATTERN
    )
    selected_observed_hypothesis_ref: str | None = Field(
        alias="selectedObservedHypothesisRef", min_length=1
    )
    selected_observed_hypothesis_hash: str | None = Field(
        alias="selectedObservedHypothesisHash", pattern=SHA256_PATTERN
    )
    selected_finding_derivation_ref: str | None = Field(
        alias="selectedFindingDerivationRef", min_length=1
    )
    selected_finding_derivation_hash: str | None = Field(
        alias="selectedFindingDerivationHash", pattern=SHA256_PATTERN
    )

    @model_validator(mode="after")
    def complete_ref_hash_pairs(self) -> "KnowledgeWorkbenchContextV4":
        for ref_field, hash_field in V4_REF_HASH_PAIRS:
            if (getattr(self, ref_field) is None) != (getattr(self, hash_field) is None):
                raise ValueError(
                    f"{ref_field} and {hash_field} must both be null or both be present"
                )
        return self


__all__ = [
    "ActiveFunctionV4",
    "KnowledgeWorkbenchContextV4",
    "UserIntentV4",
    "V4_REF_HASH_PAIRS",
]
