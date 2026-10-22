"""Locator-only progressive retrieval over Semantica AgentMemory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from contracts.project_memory_state import ProjectStateProjectionV1
from semantier.knowledge_feature_packs.semantica_adapter import _ensure_vendored_semantica_importable


@dataclass(frozen=True, slots=True)
class ProjectMemoryLocator:
    level: str
    projection_ref: str | None
    candidate_ref: str | None
    score: float | None
    compact_text: str


def compact_projection_text(projection: ProjectStateProjectionV1) -> str:
    parts = [
        f"epistemic={projection.epistemic_state}",
        f"authority={projection.authority_state or 'NA'}",
        f"execution={projection.execution_state or 'NA'}",
        f"verification={projection.verification_state or 'NA'}",
    ]
    if projection.semantic_subject_refs:
        parts.append("subjects=" + ",".join(projection.semantic_subject_refs))
    if projection.semantic_predicate_refs:
        parts.append("predicates=" + ",".join(projection.semantic_predicate_refs))
    return " ".join(parts)


class ProjectMemoryRetrievalService:
    """Semantica is an index/locator; Semantier artifacts remain authoritative."""

    def __init__(self, *, agent_memory=None) -> None:
        if agent_memory is None:
            _ensure_vendored_semantica_importable()
            from semantica.context.agent_memory import AgentMemory

            agent_memory = AgentMemory()
        self._memory = agent_memory

    def index_projection(self, projection: ProjectStateProjectionV1) -> str:
        content = compact_projection_text(projection)
        memory_id = "l1:" + projection.projection_ref
        return self._memory.store(
            content,
            metadata={
                "type": "semantier_project_state_l1",
                "level": "L1",
                "organization_id": projection.organization_id,
                "workspace_id": projection.workspace_id,
                "projection_ref": projection.projection_ref,
                "candidate_ref": projection.memory_candidate.artifact_ref,
            },
            memory_id=memory_id,
            skip_graph=True,
        )

    def index_project_abstract(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        abstract_ref: str,
        compact_text: str,
    ) -> str:
        return self._memory.store(
            compact_text,
            metadata={
                "type": "semantier_project_abstract_l0",
                "level": "L0",
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "abstract_ref": abstract_ref,
            },
            memory_id="l0:" + abstract_ref,
            skip_graph=True,
        )

    def retrieve(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        query: str,
        max_results: int = 5,
        include_l0: bool = True,
    ) -> tuple[ProjectMemoryLocator, ...]:
        raw = self._memory.retrieve(query, max_results=max_results * 4)
        locators: list[ProjectMemoryLocator] = []
        for item in raw:
            metadata = item.get("metadata") or {}
            if (
                metadata.get("organization_id") != organization_id
                or metadata.get("workspace_id") != workspace_id
            ):
                continue
            level = metadata.get("level")
            if level == "L0" and not include_l0:
                continue
            if level not in {"L0", "L1"}:
                continue
            locators.append(
                ProjectMemoryLocator(
                    level=level,
                    projection_ref=metadata.get("projection_ref"),
                    candidate_ref=metadata.get("candidate_ref"),
                    score=item.get("score"),
                    compact_text=str(item.get("content") or ""),
                )
            )
            if len(locators) >= max_results:
                break
        return tuple(locators)

    def retrieve_with_l2_fallback(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        query: str,
        exact_fallback: Callable[[str], Iterable[ProjectMemoryLocator]],
        max_results: int = 5,
    ) -> tuple[ProjectMemoryLocator, ...]:
        locators = self.retrieve(
            organization_id=organization_id,
            workspace_id=workspace_id,
            query=query,
            max_results=max_results,
        )
        if any(item.level == "L1" for item in locators):
            return locators
        # Exact evidence/session fallback occurs before any user ask.
        return tuple(exact_fallback(query))
