"""Thin Semantica v0.7 adapter for project-memory time/support views."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from contracts.ontology_building import ArtifactPinV1
from contracts.project_memory_state import EffectivityViewV1, ProjectStateProjectionV1
from contracts.runtime_context_reconstruction import MemoryCandidateV1
from semantier.knowledge_feature_packs.semantica_adapter import _ensure_vendored_semantica_importable


class SemanticaProjectMemoryAdapter:
    """Translate source-owned temporal/support state into a read-time view.

    Semantica is used for normalization and optional truth-maintenance snapshots.
    Institutional authority is deliberately outside this adapter.
    """

    def __init__(self) -> None:
        _ensure_vendored_semantica_importable()
        from semantica.context.context_graph import normalize_temporal_input

        self._normalize_temporal_input = normalize_temporal_input

    def effectivity_view(
        self,
        *,
        projection: ProjectStateProjectionV1,
        candidate: MemoryCandidateV1,
        as_of: datetime,
        truth_snapshot: Any | None = None,
        truth_fact: str | None = None,
        superseded: bool = False,
        retracted: bool = False,
    ) -> EffectivityViewV1:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        as_of = as_of.astimezone(timezone.utc)

        if (
            projection.organization_id != candidate.organization_id
            or projection.workspace_id != candidate.workspace_id
            or projection.memory_candidate.artifact_ref != candidate.candidate_ref
            or projection.memory_candidate.artifact_hash != candidate.candidate_hash
        ):
            raise ValueError("projection does not bind the exact memory candidate")

        temporal = candidate.temporal
        valid_from = temporal.valid_from
        valid_until = temporal.valid_to

        # Exercise the same Semantica v0.7 normalization used by ContextGraph.
        if valid_from is not None:
            self._normalize_temporal_input(valid_from)
        if valid_until is not None:
            self._normalize_temporal_input(valid_until)
        self._normalize_temporal_input(as_of)

        supported = True
        if truth_snapshot is not None and truth_fact is not None:
            facts = getattr(truth_snapshot, "facts", frozenset())
            supported = truth_fact in facts

        if temporal.knowledge == "unknown":
            state = "UNKNOWN"
        elif valid_from is not None and as_of < valid_from:
            state = "FUTURE"
        elif valid_until is not None and as_of >= valid_until:
            state = "HISTORICAL"
        elif superseded:
            state = "SUPERSEDED"
        elif retracted:
            # Retraction is a support/history signal, not deletion.
            state = "HISTORICAL"
        else:
            state = "ACTIVE"

        return EffectivityViewV1(
            projection=ArtifactPinV1(
                artifact_kind="project_state_projection",
                artifact_ref=projection.projection_ref,
                artifact_hash=projection.projection_hash,
            ),
            as_of=as_of,
            state=state,
            valid_from=valid_from,
            valid_until=valid_until,
            superseded=superseded,
            retracted=retracted,
            supported=supported,
        )
