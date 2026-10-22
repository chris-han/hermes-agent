"""Compile and persist derived project-state projections.

This service never admits ContextGraph state and never creates Evidence. It
references an existing MemoryCandidateV1 and stores only derived, rebuildable
state plus bounded discriminator receipts.
"""

from __future__ import annotations

import json
from typing import Iterable

from contracts.ontology_building import ArtifactPinV1
from contracts.project_memory_state import (
    ProjectStateDerivationV1,
    ProjectStateProjectionV1,
    TypedStateDecisionV1,
    bind_project_state_projection_identity,
)
from contracts.runtime_context_reconstruction import MemoryCandidateV1
from services.project_memory_state_discriminator import ProjectStateDiscriminator
from services.project_memory_state_lowerer import ProjectStateLowerer
from storage.ontology_build_store import OntologyBuildStore


_DIMENSIONS = ("epistemic", "authority", "execution", "verification")


def _candidate_pin(candidate: MemoryCandidateV1) -> ArtifactPinV1:
    return ArtifactPinV1(
        artifact_kind="memory_candidate",
        artifact_ref=candidate.candidate_ref,
        artifact_hash=candidate.candidate_hash,
    )


def _decoded_payload(candidate: MemoryCandidateV1):
    return json.loads(candidate.payload.canonical_json)


def _candidate_text(candidate: MemoryCandidateV1) -> str:
    payload = _decoded_payload(candidate)
    if isinstance(payload, dict):
        for key in ("text", "content", "message", "excerpt", "summary"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if isinstance(payload, str):
        return payload
    return ""


def _source_role(candidate: MemoryCandidateV1) -> str:
    payload = _decoded_payload(candidate)
    if isinstance(payload, dict):
        role = payload.get("source_role") or payload.get("producer_role")
        if isinstance(role, str) and role:
            return role
    role = candidate.asserted_governance.role
    if role == "binding":
        return "authority_record"
    if role == "empirical":
        return "execution_receipt"
    return "project_chat"


def _structured_fields(candidate: MemoryCandidateV1) -> dict:
    payload = _decoded_payload(candidate)
    if not isinstance(payload, dict):
        return {}
    value = payload.get("structured_fields")
    return dict(value) if isinstance(value, dict) else {}


def _locators(candidate: MemoryCandidateV1) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.locator
                for item in candidate.source_objects
                if item.locator
            }
        )
    )


class ProjectMemoryProjectionService:
    def __init__(
        self,
        *,
        lowerer: ProjectStateLowerer | None = None,
        discriminator: ProjectStateDiscriminator | None = None,
        store: OntologyBuildStore | None = None,
    ) -> None:
        self._lowerer = lowerer or ProjectStateLowerer()
        self._discriminator = discriminator
        self._store = store

    def compile(
        self,
        *,
        candidate: MemoryCandidateV1,
        target_dimensions: Iterable[str] = (),
        semantic_subject_refs: tuple[str, ...] = (),
        semantic_predicate_refs: tuple[str, ...] = (),
    ) -> tuple[ProjectStateProjectionV1, tuple[TypedStateDecisionV1, ...]]:
        text = _candidate_text(candidate)
        source_role = _source_role(candidate)
        locators = _locators(candidate)
        lowered = self._lowerer.lower(
            text=text,
            source_role=source_role,
            source_locator_refs=locators,
            structured_fields=_structured_fields(candidate),
        )

        states: dict[str, str | None] = {
            "epistemic": lowered.epistemic_state,
            "authority": lowered.authority_state,
            "execution": lowered.execution_state,
            "verification": lowered.verification_state,
        }
        derivations: dict[str, ProjectStateDerivationV1] = {
            item.dimension: item for item in lowered.derivations
        }
        requested = tuple(sorted(set(target_dimensions)))
        if any(item not in _DIMENSIONS for item in requested):
            raise ValueError("target_dimensions contains an unsupported dimension")

        decisions: list[TypedStateDecisionV1] = []
        for dimension in requested:
            if dimension == "epistemic" or states[dimension] not in (None, "UNKNOWN"):
                continue
            if self._discriminator is None:
                # Applicable but unresolved: freeze UNKNOWN, not None.
                states[dimension] = "UNKNOWN"
                derivations[dimension] = ProjectStateDerivationV1(
                    dimension=dimension,
                    method="UNRESOLVED",
                    source_locator_refs=locators,
                )
                continue
            result = self._discriminator.decide(
                organization_id=candidate.organization_id,
                workspace_id=candidate.workspace_id,
                text=text,
                source_role=source_role,
                epistemic_state=lowered.epistemic_state,
                dimension=dimension,
                source_locator_refs=locators,
            )
            states[dimension] = result.decision.selected_state
            derivations[dimension] = result.derivation
            decisions.append(result.decision)

        decision_pins = tuple(
            ArtifactPinV1(
                artifact_kind="typed_state_decision",
                artifact_ref=item.decision_ref,
                artifact_hash=item.decision_hash,
            )
            for item in decisions
        )
        payload = bind_project_state_projection_identity(
            {
                "schema_version": "project_state_projection.v1",
                "organization_id": candidate.organization_id,
                "workspace_id": candidate.workspace_id,
                "memory_candidate": _candidate_pin(candidate).model_dump(mode="json"),
                "epistemic_state": states["epistemic"],
                "authority_state": states["authority"],
                "execution_state": states["execution"],
                "verification_state": states["verification"],
                "derivations": tuple(
                    derivations[key].model_dump(mode="json")
                    for key in _DIMENSIONS
                    if key in derivations
                ),
                "semantic_subject_refs": semantic_subject_refs,
                "semantic_predicate_refs": semantic_predicate_refs,
                "discriminator_receipts": tuple(
                    item.model_dump(mode="json") for item in decision_pins
                ),
                "diagnostics": lowered.diagnostics,
                "projection_version": "project-memory-compiler.v1",
            }
        )
        projection = ProjectStateProjectionV1.model_validate(payload)
        return projection, tuple(decisions)

    def compile_and_persist(
        self,
        *,
        candidate: MemoryCandidateV1,
        target_dimensions: Iterable[str] = (),
        semantic_subject_refs: tuple[str, ...] = (),
        semantic_predicate_refs: tuple[str, ...] = (),
    ) -> ProjectStateProjectionV1:
        if self._store is None:
            raise RuntimeError("compile_and_persist requires an OntologyBuildStore")
        projection, decisions = self.compile(
            candidate=candidate,
            target_dimensions=target_dimensions,
            semantic_subject_refs=semantic_subject_refs,
            semantic_predicate_refs=semantic_predicate_refs,
        )
        self._store.put_many((*decisions, projection))
        return projection
