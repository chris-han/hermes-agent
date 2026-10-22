"""Runtime orchestration for the Project Memory Compiler three-clock policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from contracts.project_memory_state import ProjectStateProjectionV1
from contracts.runtime_context_reconstruction import MemoryCandidateV1
from services.project_memory_projection_service import ProjectMemoryProjectionService
from services.project_memory_retrieval_service import ProjectMemoryRetrievalService


@dataclass(frozen=True, slots=True)
class ProjectMemoryRuntimeOutcome:
    projection: ProjectStateProjectionV1
    stage_b_attempted: bool
    stage_b_qualified: bool
    checkpoint_requested: bool
    fallback_required: bool


class ProjectMemoryRuntimeService:
    """Three-clock runtime with no memory-owned governance checkpoints."""

    def __init__(
        self,
        *,
        stage_a_service: ProjectMemoryProjectionService,
        retrieval: ProjectMemoryRetrievalService | None = None,
        stage_b_service: ProjectMemoryProjectionService | None = None,
        stage_b_qualified: bool = False,
    ) -> None:
        if stage_b_qualified and stage_b_service is None:
            raise ValueError("qualified Stage B requires a Stage-B projection service")
        self._stage_a = stage_a_service
        self._stage_b = stage_b_service
        self._stage_b_qualified = stage_b_qualified
        self._retrieval = retrieval

    def realtime(
        self,
        *,
        candidate: MemoryCandidateV1,
        target_dimensions: Iterable[str] = (),
    ) -> ProjectMemoryRuntimeOutcome:
        # Realtime never invokes Stage B and never blocks normal task completion.
        projection, _ = self._stage_a.compile(
            candidate=candidate,
            target_dimensions=target_dimensions,
        )
        if self._retrieval is not None:
            self._retrieval.index_projection(projection)
        return ProjectMemoryRuntimeOutcome(
            projection=projection,
            stage_b_attempted=False,
            stage_b_qualified=self._stage_b_qualified,
            checkpoint_requested=False,
            fallback_required=any(
                state == "UNKNOWN"
                for state in (
                    projection.authority_state,
                    projection.execution_state,
                    projection.verification_state,
                )
            ),
        )

    def sleeptime(
        self,
        *,
        candidate: MemoryCandidateV1,
        target_dimensions: Iterable[str] = (),
    ) -> ProjectMemoryRuntimeOutcome:
        if not self._stage_b_qualified or self._stage_b is None:
            return self.realtime(
                candidate=candidate,
                target_dimensions=target_dimensions,
            )
        projection, _ = self._stage_b.compile(
            candidate=candidate,
            target_dimensions=target_dimensions,
        )
        if self._retrieval is not None:
            self._retrieval.index_projection(projection)
        return ProjectMemoryRuntimeOutcome(
            projection=projection,
            stage_b_attempted=True,
            stage_b_qualified=True,
            checkpoint_requested=False,
            fallback_required=any(
                state == "UNKNOWN"
                for state in (
                    projection.authority_state,
                    projection.execution_state,
                    projection.verification_state,
                )
            ),
        )

    def on_demand(
        self,
        *,
        candidate: MemoryCandidateV1,
        target_dimensions: Iterable[str],
    ) -> ProjectMemoryRuntimeOutcome:
        # Unqualified Stage B is never invoked synchronously merely to avoid UNKNOWN.
        return self.sleeptime(
            candidate=candidate,
            target_dimensions=target_dimensions,
        )

    @property
    def slow_clock_activation_allowed(self) -> bool:
        # Schema/rule/model/profile changes remain governance candidates elsewhere.
        return False
