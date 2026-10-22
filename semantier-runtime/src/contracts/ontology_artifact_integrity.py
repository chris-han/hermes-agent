"""Prospective integrity for exploratory ontology graph snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1, ExploratoryGraphSnapshotV1
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
ProducerProfile = Literal[
    "receivables_descriptive_build.v1",
    "source_first_build.v1",
    "bounded_exploration.v1",
    "source_grounded_seed_successor.v1",
    "a1_certification_successor.v1",
    "advisory_trial_projection.v1",
]


class OntologyArtifactIntegrityError(ValueError):
    """Stable failure code for owner integrity validation."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ExploratoryGraphSnapshotIntegrityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["exploratory_graph_snapshot_integrity.v1"] = (
        "exploratory_graph_snapshot_integrity.v1"
    )
    integrity_ref: str = Field(min_length=1)
    integrity_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    subject: ArtifactPinV1
    producer_profile: ProducerProfile
    producer_version: str = Field(min_length=1)
    payload_hash: str = Field(pattern=SHA256_PATTERN)
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_contract_identity(self) -> "ExploratoryGraphSnapshotIntegrityV1":
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        if self.subject.artifact_kind != "exploratory_graph_snapshot":
            raise ValueError("integrity subject must be an exploratory graph snapshot")
        basis = self.model_dump(mode="json", exclude={"integrity_ref", "integrity_hash"})
        digest = content_hash(basis)
        if (self.integrity_ref, self.integrity_hash) != (
            f"ontology-artifact-integrity:{digest}",
            f"sha256:{digest}",
        ):
            raise ValueError("integrity identity mismatch")
        return self


def _source_first_build_basis(snapshot: ExploratoryGraphSnapshotV1) -> dict[str, object]:
    return {
        "organization_id": snapshot.organization_id,
        "workspace_id": snapshot.workspace_id,
        "build_ref": snapshot.build_ref,
        "source_identity_ref": snapshot.source_identity_ref,
        "source_hash": snapshot.source_hash,
        "canonical_source_ir_ref": snapshot.canonical_source_ir_ref,
        "canonical_source_ir_hash": snapshot.canonical_source_ir_hash,
        "nodes": [item.model_dump(mode="json") for item in snapshot.nodes],
        "edges": [item.model_dump(mode="json") for item in snapshot.edges],
        "unresolved_gap_ids": snapshot.unresolved_gap_ids,
        "change_cause": snapshot.change_cause,
        "triggering_ref": snapshot.triggering_ref,
    }


def _source_first_build_v2_basis(snapshot: ExploratoryGraphSnapshotV1) -> dict[str, object]:
    for item in (*snapshot.nodes, *snapshot.edges):
        for ref in item.evidence_refs:
            digest = ref.removeprefix("evidence:")
            if not ref.startswith("evidence:") or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise OntologyArtifactIntegrityError("INVALID_SOURCE_FIRST_EVIDENCE_IDENTITY")
    return _source_first_build_basis(snapshot)


def _bounded_exploration_basis(snapshot: ExploratoryGraphSnapshotV1) -> dict[str, object]:
    return {
        "compiler_version": "semantier_exploratory_v1",
        "organization_id": snapshot.organization_id,
        "workspace_id": snapshot.workspace_id,
        "build_ref": snapshot.build_ref,
        "source_identity_ref": snapshot.source_identity_ref,
        "source_hash": snapshot.source_hash,
        "canonical_source_ir_ref": snapshot.canonical_source_ir_ref,
        "canonical_source_ir_hash": snapshot.canonical_source_ir_hash,
        "nodes": [item.model_dump(mode="json") for item in snapshot.nodes],
        "edges": [item.model_dump(mode="json") for item in snapshot.edges],
        "unresolved_gap_ids": snapshot.unresolved_gap_ids,
    }


def _source_grounded_successor_basis(
    snapshot: ExploratoryGraphSnapshotV1,
) -> dict[str, object]:
    basis = snapshot.model_dump(
        mode="json", exclude={"snapshot_ref", "snapshot_hash"}
    )
    # The retained producer serialized the saved UTC timestamp with ``+00:00``
    # before hashing. Pydantic's JSON dump renders the same instant with ``Z``;
    # restore the producer's historical lexical form without invoking it.
    basis["created_at"] = snapshot.created_at.isoformat()
    return basis


def _a1_certification_successor_basis(
    snapshot: ExploratoryGraphSnapshotV1,
) -> dict[str, object]:
    if (
        snapshot.change_cause != "schema_rebuild"
        or snapshot.authority_state not in {"provisional", "certified"}
        or snapshot.completeness_state != "bounded_complete"
        or snapshot.predecessor_snapshot_ref is None
        or snapshot.predecessor_snapshot_hash is None
        or snapshot.unresolved_gap_ids
    ):
        raise OntologyArtifactIntegrityError("INVALID_A1_CERTIFICATION_SUCCESSOR")
    return snapshot.model_dump(
        mode="json", exclude={"snapshot_ref", "snapshot_hash"}
    )


def _advisory_trial_projection_basis(
    snapshot: ExploratoryGraphSnapshotV1,
) -> dict[str, object]:
    if (
        snapshot.change_cause != "controlled_trial_projection"
        or snapshot.authority_state != "provisional"
        or snapshot.completeness_state != "incomplete"
        or snapshot.predecessor_snapshot_ref is None
        or not snapshot.predecessor_snapshot_ref.startswith("exploratory-snapshot:")
        or snapshot.predecessor_snapshot_hash is None
        or not snapshot.build_ref.startswith("ontology-build:advisory-trial:")
        or snapshot.triggering_ref is None
        or not snapshot.triggering_ref.startswith("advisory-trial-build-inputs:")
    ):
        raise OntologyArtifactIntegrityError("INVALID_ADVISORY_TRIAL_PROJECTION")
    return snapshot.model_dump(
        mode="json", exclude={"snapshot_ref", "snapshot_hash"}
    )


def _receivables_descriptive_basis(
    snapshot: ExploratoryGraphSnapshotV1,
) -> dict[str, object]:
    if (
        snapshot.authority_state != "provisional"
        or not snapshot.build_ref.startswith("ontology-build:receivables:")
    ):
        raise OntologyArtifactIntegrityError("INVALID_RECEIVABLES_DESCRIPTIVE_BUILD")
    return snapshot.model_dump(mode="json", exclude={"snapshot_ref", "snapshot_hash"})


SnapshotSemanticBasis = Callable[[ExploratoryGraphSnapshotV1], dict[str, object]]

# Historical verifier keys and functions are literal. Current producer constants
# are deliberately not consulted when old artifacts are read or replayed.
EXPLORATORY_SNAPSHOT_VERIFIERS: Mapping[
    tuple[str, str], SnapshotSemanticBasis
] = MappingProxyType(
    {
        (
            "receivables_descriptive_build.v1",
            "receivables_descriptive_v1",
        ): _receivables_descriptive_basis,
        (
            "source_first_build.v1",
            "semantier_source_first_exploratory_v1",
        ): _source_first_build_basis,
        (
            "source_first_build.v1",
            "semantier_source_first_exploratory_v2",
        ): _source_first_build_v2_basis,
        ("source_first_build.v1", "semantier_source_first_exploratory_v3"): _source_first_build_v2_basis,
        (
            "bounded_exploration.v1",
            "semantier_exploratory_v1",
        ): _bounded_exploration_basis,
        (
            "source_grounded_seed_successor.v1",
            "semantier_source_grounded_seed_successor_v1",
        ): _source_grounded_successor_basis,
        (
            "a1_certification_successor.v1",
            "semantier_a1_certification_successor_v1",
        ): _a1_certification_successor_basis,
        (
            "advisory_trial_projection.v1",
            "runtime_learning_trial_paired_build_v1",
        ): _advisory_trial_projection_basis,
    }
)


def create_exploratory_snapshot_integrity(
    *,
    snapshot: ExploratoryGraphSnapshotV1,
    producer_profile: ProducerProfile,
    producer_version: str,
    recorded_at: datetime,
) -> ExploratoryGraphSnapshotIntegrityV1:
    """Create an immutable integrity artifact for newly produced snapshot bytes."""

    snapshot = ExploratoryGraphSnapshotV1.model_validate(snapshot.model_dump(mode="json"))
    basis = {
        "schema_version": "exploratory_graph_snapshot_integrity.v1",
        "organization_id": snapshot.organization_id,
        "workspace_id": snapshot.workspace_id,
        "subject": {
            "artifact_kind": "exploratory_graph_snapshot",
            "artifact_ref": snapshot.snapshot_ref,
            "artifact_hash": snapshot.snapshot_hash,
        },
        "producer_profile": producer_profile,
        "producer_version": producer_version,
        "payload_hash": "sha256:" + content_hash(snapshot.model_dump(mode="json")),
        "recorded_at": recorded_at,
    }
    digest = content_hash(
        {**basis, "recorded_at": recorded_at.isoformat().replace("+00:00", "Z")}
    )
    return ExploratoryGraphSnapshotIntegrityV1(
        integrity_ref=f"ontology-artifact-integrity:{digest}",
        integrity_hash=f"sha256:{digest}",
        **basis,
    )


def verify_exploratory_snapshot_integrity(
    *, snapshot: ExploratoryGraphSnapshotV1, integrity: ExploratoryGraphSnapshotIntegrityV1
) -> None:
    """Validate a snapshot against its typed prospective integrity artifact."""

    snapshot = ExploratoryGraphSnapshotV1.model_validate(snapshot.model_dump(mode="json"))
    integrity = ExploratoryGraphSnapshotIntegrityV1.model_validate(
        integrity.model_dump(mode="json")
    )
    if (
        integrity.organization_id,
        integrity.workspace_id,
        integrity.subject.artifact_ref,
        integrity.subject.artifact_hash,
    ) != (
        snapshot.organization_id,
        snapshot.workspace_id,
        snapshot.snapshot_ref,
        snapshot.snapshot_hash,
    ):
        raise OntologyArtifactIntegrityError("INTEGRITY_SUBJECT_MISMATCH")
    if integrity.payload_hash != "sha256:" + content_hash(snapshot.model_dump(mode="json")):
        raise OntologyArtifactIntegrityError("SNAPSHOT_PAYLOAD_HASH_MISMATCH")
    verifier = EXPLORATORY_SNAPSHOT_VERIFIERS.get(
        (integrity.producer_profile, integrity.producer_version)
    )
    if verifier is None:
        raise OntologyArtifactIntegrityError("UNSUPPORTED_INTEGRITY_VERIFIER")
    if (
        integrity.producer_profile,
        integrity.producer_version,
    ) == (
        "advisory_trial_projection.v1",
        "runtime_learning_trial_paired_build_v1",
    ) and integrity.recorded_at != snapshot.created_at:
        raise OntologyArtifactIntegrityError("ADVISORY_TRIAL_RECORDED_AT_MISMATCH")
    digest = content_hash(verifier(snapshot))
    if (snapshot.snapshot_ref, snapshot.snapshot_hash) != (
        f"exploratory-snapshot:{digest}",
        f"sha256:{digest}",
    ):
        raise OntologyArtifactIntegrityError("SNAPSHOT_SEMANTIC_HASH_MISMATCH")


__all__ = [
    "ExploratoryGraphSnapshotIntegrityV1",
    "OntologyArtifactIntegrityError",
    "EXPLORATORY_SNAPSHOT_VERIFIERS",
    "create_exploratory_snapshot_integrity",
    "verify_exploratory_snapshot_integrity",
]
