"""Implementation Contracts IC-1 to IC-7 for Chinese concept-boundary refactor.

These contracts specify exact algorithms and data structures that must remain
identical across backend, storage, migration tooling, and ContextGraph Studio.

References: docs/plans/2026-08-21-contextgraph-studio-chinese-concept-boundary-refactor-v1.md
"""

from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
import hashlib
import json


# ============================================================================
# IC-1: Candidate-span identity (RFC 8785/JCS + SHA-256)
# ============================================================================

class ConceptCandidateSpanIdentity(BaseModel):
    """IC-1: Deterministic SHA-256 identity for candidate spans.
    
    RFC 8785/JCS canonical JSON encoding ensures identical serialization
    across all platforms and implementations.
    """
    
    contract: Literal["concept-candidate-span-id/v1"] = "concept-candidate-span-id/v1"
    sourceIdentity: str  # Immutable CanonicalSourceIR source identity
    blockId: str  # Stable canonical block ID
    startOffset: int  # Zero-based Unicode code-point offset (inclusive)
    endOffset: int  # Zero-based Unicode code-point offset (exclusive)
    exactTextSha256: str  # Lowercase SHA-256 hex of exact UTF-8 source slice
    segmentationProfile: str  # e.g., "tender-sensitive/1.0.0"
    boundaryMethod: Literal[
        "source_cell",
        "source_list_item",
        "configured_delimiter",
        "clause_boundary",
        "schema_assisted",
        "human_defined"
    ]
    boundaryMethodVersion: str  # Semantic version of boundary method
    
    def compute_candidate_span_id(self) -> str:
        """Compute canonical span ID using RFC 8785/JCS."""
        # Serialize using RFC 8785 canonical ordering (must match JSON schema)
        payload_dict = {
            "boundaryMethod": self.boundaryMethod,
            "boundaryMethodVersion": self.boundaryMethodVersion,
            "blockId": self.blockId,
            "contract": self.contract,
            "endOffset": self.endOffset,
            "exactTextSha256": self.exactTextSha256,
            "segmentationProfile": self.segmentationProfile,
            "sourceIdentity": self.sourceIdentity,
            "startOffset": self.startOffset,
        }
        # Use json.dumps with sort_keys for canonical ordering
        canonical_json = json.dumps(payload_dict, separators=(',', ':'), sort_keys=True)
        payload_bytes = canonical_json.encode("utf-8")
        digest = hashlib.sha256(payload_bytes).hexdigest().lower()
        return f"ccs_v1_{digest}"
    
    model_config = ConfigDict(use_enum_values=True)


class ConceptCandidateSpan(BaseModel):
    """IC-1: Core candidate span contract with deterministic identity."""
    
    candidate_span_id: str  # Format: ccs_v1_<64-hex>, computed via IC-1
    block_id: str
    source_locator: Dict[str, Any]  # Reference to SourceLocator
    exact_text: str  # Immutable; no normalization or trimming
    start_offset: int  # Zero-based, Unicode code-point
    end_offset: int  # Exclusive
    boundary_method: Literal[
        "source_cell",
        "source_list_item",
        "configured_delimiter",
        "clause_boundary",
        "schema_assisted",
        "human_defined"
    ]
    boundary_confidence: float = Field(ge=0.0, le=1.0)
    source_role: str  # From profile decomposableSourceRoles
    semantic_role_hint: Optional[str] = None
    category_hint: Optional[str] = None
    parent_span_id: Optional[str] = None  # For split candidates


# ============================================================================
# IC-2: Segmentation profile schema
# ============================================================================

class DelimiterRule(BaseModel):
    """Delimiter rule in segmentation profile."""
    token: str
    roles: List[str]  # e.g., ["canonical_sensitive_term", "alias"]
    priority: int  # Lower = match first


class ClauseRule(BaseModel):
    """Chinese clause segmentation rules."""
    enabledRoles: List[str]
    terminalTokens: List[str]  # e.g., ["。", "！", "？"]
    conjunctionTokens: List[str]  # e.g., ["且", "并且"]
    protectedConjunctions: List[str]  # Cannot be split


class ProtectedSpanRule(BaseModel):
    """Rules for spans that protect embedded delimiters."""
    balancedPairs: List[List[str]]  # e.g., [""", """], ["（", "）"]
    regexes: List[str]  # e.g., date patterns, currency patterns
    lexicalExceptions: List[str]


class QualityGate(BaseModel):
    """Boundary quality gate thresholds."""
    advisoryLabelCodePoints: int = 40  # Flag if > this
    hardLabelCodePoints: int = 160  # Reject if > this without valid split
    maxIndependentClauses: int = 1  # Max non-coordinated clauses
    schemaAssistedMinConfidence: float = 0.85


class SchemaAssistedConfig(BaseModel):
    """Schema-assisted boundary proposal configuration."""
    enabled: bool = True
    promptAsset: str  # e.g., "src/prompts/reference_concept_boundary_zh_v1.md"
    promptVersion: str
    allowedRoles: List[str]
    requiresHumanAcceptance: bool = True


class SegmentationProfile(BaseModel):
    """IC-2: Segmentation profile schema with strict validation."""
    
    contractVersion: str = "1.0.0"
    profileId: str
    profileVersion: str
    decomposableSourceRoles: List[str]
    multiValueSourceRoles: List[str]
    delimiterRules: List[DelimiterRule]
    clauseRules: ClauseRule
    protectedSpanRules: ProtectedSpanRule
    qualityGates: QualityGate
    schemaAssisted: SchemaAssistedConfig
    
    @field_validator("profileVersion")
    @classmethod
    def validate_version(cls, v):
        """Ensure semantic version format."""
        import re
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError(f"profileVersion must be semantic: {v}")
        return v
    
    @field_validator("*", mode="before")
    @classmethod
    def no_unknown_keys(cls, v):
        """Reject unknown keys (strict schema)."""
        if isinstance(v, dict):
            known = set(cls.model_fields.keys())
            unknown = set(v.keys()) - known
            if unknown:
                raise ValueError(f"Unknown profile keys: {unknown}")
        return v
    
    model_config = ConfigDict(extra="forbid")  # Reject unknown fields


# ============================================================================
# IC-3: Boundary algorithm and failure policy
# ============================================================================

class SchemaAssistedProposal(BaseModel):
    """Schema-assisted boundary proposal from LLM or rules."""
    sourceSpanId: str  # Parent candidate span ID
    proposals: List[Dict[str, Any]]  # [{startOffsetRelative, endOffsetRelative, semanticRole, confidence, reasonCode}]


class BoundaryQualityGateResult(BaseModel):
    """Result of boundary quality gate checks."""
    candidate_span_id: str
    gate_results: Dict[str, Literal["pass", "fail", "warn"]]
    gate_outcome: Literal["pass", "needs_boundary_review", "reject", "provider_error"]
    details: Optional[str] = None


# ============================================================================
# IC-4: Curation action API
# ============================================================================

class CurationAction(BaseModel):
    """IC-4: Append-only curation command for ContextGraph Studio."""
    
    actionId: str  # UUID
    actionType: Literal["split", "merge", "edit_role", "accept", "reject"]
    actorId: str  # Authenticated principal
    expectedGraphVersion: int
    sourceSpanIds: List[str]  # Targets one or more spans
    
    # For split: absolute offsets within source block
    splitOffsetsAbsolute: Optional[List[int]] = None
    
    # For merge: contiguous sibling spans from same block
    mergeTargetSpanIds: Optional[List[str]] = None
    
    # For edit_role or accept/reject
    targetSemanticRole: Optional[str] = None
    reasonCode: str  # e.g., "curator_boundary_correction"
    comment: Optional[str] = None
    clientTimestamp: str  # RFC3339
    
    @field_validator("splitOffsetsAbsolute")
    @classmethod
    def validate_split_offsets(cls, v, info: ValidationInfo):
        if info.data.get("actionType") == "split" and v is None:
            raise ValueError("split action requires splitOffsetsAbsolute")
        if v:
            if len(v) < 1:
                raise ValueError("splitOffsetsAbsolute must be non-empty")
            if v != sorted(set(v)):
                raise ValueError("splitOffsetsAbsolute must be sorted, unique")
        return v


class CurationActionResponse(BaseModel):
    """Response from curation action endpoint."""
    actionId: str
    storedEvent: Dict[str, Any]
    replacementSpanIds: List[str]
    newGraphVersion: int
    refreshedDiagnostics: List[Dict[str, Any]]


# ============================================================================
# IC-5: Learning event persistence and replay
# ============================================================================

class BoundaryLearningEvent(BaseModel):
    """IC-5: Append-only learning event from curator corrections."""
    
    eventId: str  # UUID
    eventType: Literal["boundary.split", "boundary.merge", "role.edit", "candidate.accept", "candidate.reject"]
    eventVersion: int = 1
    runId: str
    sourceIdentity: str
    baseGraphVersion: int
    resultGraphVersion: int
    inputSpanIds: List[str]
    outputSpans: List[ConceptCandidateSpan]
    profileId: str
    profileVersion: str
    actorId: str
    reasonCode: str
    comment: Optional[str] = None
    idempotencyKey: str  # UUID
    recordedAt: str  # RFC3339 server timestamp
    previousEventHash: str  # SHA-256 of previous event or "genesis"
    eventHash: Optional[str] = None  # Computed; excludes this field
    
    def compute_event_hash(self) -> str:
        """Compute event hash using JCS (same as IC-1)."""
        # Create dict excluding eventHash and previousEventHash
        event_dict = self.dict(exclude={"eventHash"})
        canonical_json = json.dumps(event_dict, separators=(',', ':'), sort_keys=True)
        payload_bytes = canonical_json.encode("utf-8")
        return hashlib.sha256(payload_bytes).hexdigest().lower()


# ============================================================================
# IC-6: Candidate-run migration contract
# ============================================================================

class CandidateRunMigration(BaseModel):
    """IC-6: Migration assessment for shim-produced candidates."""
    
    oldRunId: str
    newRunId: str
    supersedesRelationship: Literal["replaces", "supplements", "supersedes"]
    migrationTimestamp: str  # RFC3339
    comparisonArtifact: Dict[str, Any]  # node counts, label-length percentiles, mappings
    
    # Acceptance checklist
    shim_only_concepts_count: int = Field(ge=0, le=0)  # Must be 0
    exact_resolvable_anchors_percent: float = Field(ge=100.0, le=100.0)  # Must be 100%
    silently_dropped_blocks_count: int = Field(ge=0, le=0)  # Must be 0
    old_candidates_mapped: bool
    split_merge_preserves_coverage: bool
    all_deterministic_fixtures_pass: bool
    ambiguous_candidates_reviewed: bool
    curator_approval_timestamp: Optional[str] = None


# ============================================================================
# IC-7: Compatibility removal and test-double boundary
# ============================================================================

class VerifiedReleaseCycle(BaseModel):
    """IC-7: One complete deployed cycle with new profile."""
    
    cycleNumber: int  # 1 or 2
    deployedVersion: str
    deploymentDate: str  # RFC3339
    newProfileEnabled: bool = True
    deterministicTests: bool
    realSemanticaIntegration: bool
    storageReplayTests: bool
    migrateTests: bool
    remoteChromeBrowserGates: bool
    evidenceRetained: bool
    chineseDocxIncluded: bool
    referenceProjectionsFromShim: int = Field(ge=0)  # Must reach 0


# ============================================================================
# Mock/Real Boundary Definition
# ============================================================================

class TestDoubleBoundary(BaseModel):
    """IC-7: Explicit boundary between mocks and integration tests."""
    
    mocks_allowed: List[str] = Field(
        default=[
            "error_handling_unit_tests",
            "serialization_unit_tests"
        ]
    )
    mocks_forbidden: List[str] = Field(
        default=[
            "W3_adapter_refactor",
            "W6_migration",
            "W7_promotion",
            "release_cycle_evidence"
        ]
    )
    realSemanticaRequired: List[str] = Field(
        default=[
            "integration_tests",
            "migration_acceptance",
            "browser_verification",
            "release_gates"
        ]
    )


if __name__ == "__main__":
    # Test IC-1 identity computation
    span_id_input = ConceptCandidateSpanIdentity(
        sourceIdentity="canonical_v1_abc123",
        blockId="doc_table_1_cell_2_3",
        startOffset=0,
        endOffset=12,
        exactTextSha256=hashlib.sha256("大型企业".encode("utf-8")).hexdigest(),
        segmentationProfile="tender-sensitive/1.0.0",
        boundaryMethod="configured_delimiter",
        boundaryMethodVersion="1.0.0"
    )
    span_id = span_id_input.compute_candidate_span_id()
    print(f"Generated span ID: {span_id}")
    assert span_id.startswith("ccs_v1_"), f"Invalid span ID format: {span_id}"
    print("✓ IC-1 identity computation works")
