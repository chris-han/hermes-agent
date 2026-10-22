"""Stage-A-only residual routing. No scorer, runtime activation, or store writes.

Use the existing projection service and constraint function unchanged. A routing
receipt is advisory: callers still need independently qualified Stage B and
source-owned eligibility before consequential interpretation/consumption.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Literal

from contracts.ontology_building import ArtifactPinV1
from contracts.project_memory_residual import (
    ResidualSourceConstraintsV1,
    ResidualStageBInputV1,
)
from contracts.project_memory_state import ProjectStateProjectionV1, StateDimension
from contracts.runtime_context_reconstruction import MemoryCandidateV1
from eos.artifact_hashing import content_hash
from services.project_memory_projection_service import (
    ProjectMemoryProjectionService,
    _candidate_text,
    _source_role,
)
from services.project_memory_state_discriminator import FRONTIERS, _eligible_frontier


# Frozen pre-W0 semantics. The diagnostic runner verifies actual file hashes;
# runtime uses this supplied, content-addressed profile rather than reading disk.
FROZEN_STAGE_A_FILES = {
    "src/services/project_memory_state_lowerer.py": "8bcf8cb7bc4b5b18e96ad57ccdd7d135e88c59b9f28f547bfd7d7d4e8ed7a37b",
    "src/services/project_memory_projection_service.py": "9ff62c52753b1b62ffc8d655352a467e901c42c93aabc288df3c6892cbc36574",
    "src/services/project_memory_state_discriminator.py": "7f0b1cd6baf4bc6adf1058188b9270eaf3f3ce4a0926f7ae5e625d4149969ca3",
    "src/contracts/project_memory_state.py": "b751beac856c5006089e1f785104a4e70150d5288974ff0d6137ba6ed3c26375",
    "src/contracts/runtime_context_reconstruction.py": "cb6c61362747c7f0d4fcb87058733250942a802b830d10973a08914faf819ff8",
}
ROUTING_VERSION = "project-memory-residual-routing.v1"
_EVIDENCE_CLASSES = {
    "system_record": "SOURCE_RECORD",
    "authority_record": "AUTHORITY_RECORD",
    "execution_receipt": "EXECUTION_RECEIPT",
    "project_chat": "PROJECT_CHAT",
}
RouteDisposition = Literal[
    "STAGE_A_RESOLVED",
    "SOURCE_CONSTRAINT_FALLBACK",
    "SOURCE_DECLARED_UNKNOWN",
    "NO_SOURCE_TEXT",
    "SEMANTIC_RESIDUAL",
]


def residual_routing_profile_pin() -> ArtifactPinV1:
    digest = content_hash(
        {
            "routing_version": ROUTING_VERSION,
            "stage_a_files": FROZEN_STAGE_A_FILES,
            "scope": "diagnostic_routing_only",
            "exclude_singleton_frontiers": True,
            "preserve_source_declared_unknown": True,
        }
    )
    return ArtifactPinV1(
        artifact_kind="project_memory_residual_profile",
        artifact_ref="project-memory-residual-profile:" + digest[:24],
        artifact_hash="sha256:" + digest,
    )


@dataclass(frozen=True)
class ResidualRoutingDecision:
    dimension: StateDimension
    disposition: RouteDisposition
    base_state: str | None
    state: str | None
    receipt: ResidualStageBInputV1 | None = None


@dataclass(frozen=True)
class ResidualRoutingResult:
    projection: ProjectStateProjectionV1
    routes: tuple[ResidualRoutingDecision, ...]

    @property
    def residuals(self) -> tuple[ResidualStageBInputV1, ...]:
        return tuple(item.receipt for item in self.routes if item.receipt is not None)


class ProjectMemoryResidualRouter:
    """Compile frozen Stage A and identify residuals, without executing Stage B."""

    def route(
        self,
        *,
        candidate: MemoryCandidateV1,
        target_dimensions: Iterable[str] = (),
    ) -> ResidualRoutingResult:
        # Pydantic model_copy/model_construct may bypass model validators.
        candidate = MemoryCandidateV1.model_validate(candidate.model_dump(mode="json"))
        requested = tuple(sorted(set(target_dimensions)))
        if any(dimension not in FRONTIERS for dimension in requested):
            raise ValueError("unsupported target dimension")
        service = ProjectMemoryProjectionService()  # no discriminator, no store
        baseline, baseline_decisions = service.compile(candidate=candidate)
        projection, decisions = service.compile(
            candidate=candidate, target_dimensions=requested
        )
        if decisions or baseline_decisions or projection.discriminator_receipts:
            raise ValueError("W0 routing requires a Stage-A-only projection")
        base_derivations = {item.dimension: item for item in baseline.derivations}
        derivations = {item.dimension: item for item in projection.derivations}
        source_role = _source_role(candidate)
        text = _candidate_text(candidate)
        routes: list[ResidualRoutingDecision] = []
        for dimension in requested:
            base = getattr(baseline, dimension + "_state")
            state = getattr(projection, dimension + "_state")
            frontier = _eligible_frontier(
                dimension=dimension,
                frontier=FRONTIERS[dimension],
                source_role=source_role,
                epistemic_state=projection.epistemic_state,
            )
            receipt = None
            if base not in (None, "UNKNOWN"):
                disposition = "STAGE_A_RESOLVED"
            elif len(frontier) == 1:
                disposition = "SOURCE_CONSTRAINT_FALLBACK"
            elif (
                base == "UNKNOWN"
                and dimension in base_derivations
                and base_derivations[dimension].method == "EXACT_STRUCTURED_FIELD"
            ):
                disposition = "SOURCE_DECLARED_UNKNOWN"
            elif not text.strip():
                disposition = "NO_SOURCE_TEXT"
            else:
                derivation = derivations[dimension]
                if state != "UNKNOWN" or derivation.method != "UNRESOLVED":
                    raise ValueError(
                        "cannot produce a receipt without exact Stage-A UNRESOLVED basis"
                    )
                disposition = "SEMANTIC_RESIDUAL"
                receipt = ResidualStageBInputV1.create(
                    organization_id=candidate.organization_id,
                    workspace_id=candidate.workspace_id,
                    memory_candidate=projection.memory_candidate,
                    stage_a_projection=ArtifactPinV1(
                        artifact_kind="project_state_projection",
                        artifact_ref=projection.projection_ref,
                        artifact_hash=projection.projection_hash,
                    ),
                    compiler_profile=residual_routing_profile_pin(),
                    target_dimension=dimension,
                    stage_a_derivation=derivation,
                    unresolved_reason=(
                        "REQUESTED_DIMENSION_UNRESOLVED"
                        if base is None
                        else "LOWERER_UNRESOLVED"
                    ),
                    source_constraints=ResidualSourceConstraintsV1(
                        source_role=source_role,
                        evidence_class=_EVIDENCE_CLASSES.get(
                            source_role, "DERIVED_MEMORY"
                        ),
                        epistemic_state=projection.epistemic_state,
                        eligible_frontier=frontier,
                    ),
                    input_text_hash="sha256:"
                    + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            routes.append(
                ResidualRoutingDecision(dimension, disposition, base, state, receipt)
            )
        return ResidualRoutingResult(projection=projection, routes=tuple(routes))

    def verify(
        self,
        *,
        receipt: ResidualStageBInputV1,
        candidate: MemoryCandidateV1,
        target_dimensions: Iterable[str],
    ) -> None:
        """Rebind, then replay. A self-consistent hash alone never proves eligibility."""
        receipt = ResidualStageBInputV1.model_validate(receipt.model_dump(mode="json"))
        if receipt.compiler_profile != residual_routing_profile_pin():
            raise ValueError("stale residual compiler profile")
        result = self.route(candidate=candidate, target_dimensions=target_dimensions)
        if receipt not in result.residuals:
            raise ValueError(
                "residual receipt does not bind the exact candidate, request, and Stage-A result"
            )
