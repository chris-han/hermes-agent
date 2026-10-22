"""Deterministic Work-Surface guidance triggers and bounded prompt context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contracts.unified_workbench_context import WorkSurface


class GuidanceInterventionLevel(str, Enum):
    PASSIVE = "passive"
    ASSISTIVE = "assistive"
    REQUIRED = "required"


class GuidanceTriggerV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["guidance_trigger.v1"] = "guidance_trigger.v1"
    trigger_type: str = Field(min_length=1)
    current_work_surface: WorkSurface
    owning_work_surface: WorkSurface
    intervention_level: GuidanceInterventionLevel
    producer: str = Field(min_length=1)
    refs: dict[str, str]
    allowed_typed_actions: tuple[str, ...]
    material_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    suppression_key: str = Field(min_length=1)


@dataclass(frozen=True)
class GuidanceCatalogEntry:
    trigger_type: str
    work_surface: WorkSurface
    authoritative_producer: str
    ref_requirements: tuple[str, ...]
    intervention_level: GuidanceInterventionLevel
    allowed_typed_actions: tuple[str, ...]
    gate_ref_required: bool


class GuidanceCatalog:
    def __init__(
        self,
        *,
        entries: tuple[GuidanceCatalogEntry, ...],
        non_trigger_events: frozenset[str],
        cross_surface_pairs: frozenset[tuple[str, WorkSurface, WorkSurface]],
    ) -> None:
        self.entries = entries
        self._by_type = {entry.trigger_type: entry for entry in entries}
        self._non_trigger_events = non_trigger_events
        self._cross_surface_pairs = cross_surface_pairs

    @classmethod
    def from_audit_fixture(cls, path: Path) -> "GuidanceCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(
            GuidanceCatalogEntry(
                trigger_type=item["trigger_type"],
                work_surface=WorkSurface(item["work_surface"]),
                authoritative_producer=item["authoritative_producer"],
                ref_requirements=tuple(item["ref_requirements"]),
                intervention_level=GuidanceInterventionLevel(item["intervention_level"]),
                allowed_typed_actions=tuple(item["allowed_typed_actions"]),
                gate_ref_required=item["gate_ref_required"],
            )
            for item in payload["guidance_trigger_catalog"]
        )
        policy = payload["guidance_policy"]
        pairs = frozenset(
            (
                item["trigger_type"],
                WorkSurface(item["current_work_surface"]),
                WorkSurface(item["owning_work_surface"]),
            )
            for item in policy["cross_surface_triggers"]
        )
        return cls(
            entries=entries,
            non_trigger_events=frozenset(policy["non_trigger_events"]),
            cross_surface_pairs=pairs,
        )

    def entry(self, trigger_type: str) -> GuidanceCatalogEntry:
        return self._by_type[trigger_type]

    def validate(self, trigger: GuidanceTriggerV1) -> GuidanceTriggerV1:
        entry = self._by_type.get(trigger.trigger_type)
        cross_identity = (
            trigger.trigger_type,
            trigger.current_work_surface,
            trigger.owning_work_surface,
        )
        if entry is None:
            if cross_identity not in self._cross_surface_pairs:
                raise ValueError("trigger is not an approved cross-surface trigger")
            entry = next(
                item for item in self.entries if item.work_surface is trigger.owning_work_surface
            )
        if trigger.producer != entry.authoritative_producer:
            raise ValueError("trigger did not originate from its authoritative producer")
        if cross_identity not in self._cross_surface_pairs and trigger.current_work_surface is not entry.work_surface:
            raise ValueError("trigger is on the wrong work surface")
        missing = [name for name in entry.ref_requirements if not trigger.refs.get(name)]
        if entry.gate_ref_required and not trigger.refs.get("gate_ref"):
            raise ValueError("required guidance trigger must include gate_ref")
        if missing:
            raise ValueError(f"missing required trigger refs: {', '.join(missing)}")
        if trigger.intervention_level is not entry.intervention_level:
            raise ValueError("intervention level cannot be upgraded or downgraded")
        if trigger.allowed_typed_actions != entry.allowed_typed_actions:
            raise ValueError("allowed typed actions must match the catalog")
        if trigger.owning_work_surface is not trigger.current_work_surface:
            if cross_identity not in self._cross_surface_pairs:
                raise ValueError("trigger is not an approved cross-surface trigger")
        return trigger

    def may_trigger_for_event(self, event: str) -> bool:
        return event not in self._non_trigger_events


def assemble_guidance_context(trigger: GuidanceTriggerV1) -> dict[str, object]:
    """Build model input from a validated trigger without granting model authority."""

    prompt_path = Path(__file__).resolve().parents[1] / "prompts/contextgraph_guidance_policy.txt"
    return {
        "triggerType": trigger.trigger_type,
        "currentWorkSurface": trigger.current_work_surface.value,
        "owningWorkSurface": trigger.owning_work_surface.value,
        "interventionLevel": trigger.intervention_level.value,
        "refs": trigger.refs,
        "allowedTypedActions": list(trigger.allowed_typed_actions),
        "policyText": prompt_path.read_text(encoding="utf-8").strip(),
    }


__all__ = [
    "GuidanceCatalog",
    "GuidanceCatalogEntry",
    "GuidanceInterventionLevel",
    "GuidanceTriggerV1",
    "assemble_guidance_context",
]
