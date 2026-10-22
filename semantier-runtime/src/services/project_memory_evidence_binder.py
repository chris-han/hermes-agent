"""Exact rebinding from derived project memory to existing governed artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping

from contracts.ontology_building import ArtifactPinV1
from contracts.project_memory_state import ProjectStateProjectionV1
from contracts.runtime_context_reconstruction import MemoryCandidateV1


@dataclass(frozen=True, slots=True)
class ProjectMemoryBinding:
    projection_ref: str
    candidate_ref: str
    evidence: tuple[ArtifactPinV1, ...]
    admitted_assertion_refs: tuple[str, ...]


class ProjectMemoryEvidenceBinder:
    """Rebind exactly; never synthesize Evidence or semantic admission."""

    @staticmethod
    def bind(
        *,
        projection: ProjectStateProjectionV1,
        candidate: MemoryCandidateV1,
        admitted_assertion_refs: Collection[str] = (),
    ) -> ProjectMemoryBinding:
        if (
            projection.organization_id != candidate.organization_id
            or projection.workspace_id != candidate.workspace_id
        ):
            raise PermissionError("projection/candidate tenant mismatch")
        if (
            projection.memory_candidate.artifact_ref != candidate.candidate_ref
            or projection.memory_candidate.artifact_hash != candidate.candidate_hash
        ):
            raise ValueError("projection does not bind the exact MemoryCandidate")

        admitted = set(admitted_assertion_refs)
        selected = tuple(
            ref for ref in projection.semantic_subject_refs if ref in admitted
        )
        return ProjectMemoryBinding(
            projection_ref=projection.projection_ref,
            candidate_ref=candidate.candidate_ref,
            evidence=candidate.evidence,
            admitted_assertion_refs=selected,
        )

    @staticmethod
    def require_exact_evidence(
        *,
        binding: ProjectMemoryBinding,
        evidence_registry: Mapping[tuple[str, str, str], object],
    ) -> tuple[object, ...]:
        resolved: list[object] = []
        for pin in binding.evidence:
            key = (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            if key not in evidence_registry:
                raise ValueError("project memory evidence cannot exact-rebind")
            resolved.append(evidence_registry[key])
        return tuple(resolved)
