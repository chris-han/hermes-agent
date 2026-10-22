"""Unified Context and bounded Chat-to-workbench presentation actions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.knowledge_workbench_context import KnowledgeWorkbenchContextV4


class WorkSurface(str, Enum):
    SOURCES = "sources"
    SCHEMA = "schema"
    EXTRACT = "extract"
    GROUND = "ground"
    GRAPH = "graph"
    INSPECT = "inspect"
    COMPARE = "compare"
    EVALUATE = "evaluate"


class WorkflowProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    workflow_id: str = Field(alias="workflowId", min_length=1)
    phase: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    revision: int = Field(ge=0)


class UnifiedWorkbenchContextV1(BaseModel):
    """Additive navigation envelope whose semantic authority remains v4."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["unified_workbench_context.v1"] = Field(
        default="unified_workbench_context.v1", alias="schemaVersion"
    )
    knowledge: KnowledgeWorkbenchContextV4
    workflow: WorkflowProjectionV1
    work_surface: WorkSurface = Field(alias="workSurface")
    lens: str = Field(min_length=1)
    view: str = Field(min_length=1)
    material_context_revision: int = Field(alias="materialContextRevision", ge=0)
    selection_refs: tuple[str, ...] = Field(alias="selectionRefs")
    inspection_ref: str | None = Field(default=None, alias="inspectionRef")
    review_ref: str | None = Field(default=None, alias="reviewRef")

    @model_validator(mode="after")
    def preserve_active_function_compatibility(self) -> "UnifiedWorkbenchContextV1":
        active = self.knowledge.active_function
        if active is not None and active.tab != self.work_surface.value:
            raise ValueError("activeFunction.tab must match workSurface")
        return self

    def canvas_to_chat_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload["compatibility"] = {
            "phase": self.workflow.phase,
            "stage": self.workflow.stage,
            "activeFunction": (
                self.knowledge.active_function.model_dump(mode="json")
                if self.knowledge.active_function is not None
                else None
            ),
        }
        return payload


class WorkbenchActionCommand(str, Enum):
    FOCUS_OBJECT = "FOCUS_OBJECT"
    OPEN_INSPECTOR = "OPEN_INSPECTOR"
    SET_RELATION_FILTER = "SET_RELATION_FILTER"
    OPEN_WORK_SURFACE = "OPEN_WORK_SURFACE"


class WorkbenchViewActionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["workbench_view_action.v1"] = Field(
        default="workbench_view_action.v1", alias="schemaVersion"
    )
    command: WorkbenchActionCommand
    context_ref: str = Field(alias="contextRef", min_length=1)
    material_context_revision: int = Field(alias="materialContextRevision", ge=0)
    work_surface: WorkSurface | None = Field(default=None, alias="workSurface")
    inspector_tab: Literal["overview", "evidence", "lineage", "history"] | None = (
        Field(default=None, alias="inspectorTab")
    )
    filter_ids: tuple[str, ...] = Field(default=(), alias="filterIds")

    @model_validator(mode="after")
    def command_arguments_match(self) -> "WorkbenchViewActionV1":
        required = {
            WorkbenchActionCommand.OPEN_WORK_SURFACE: self.work_surface,
            WorkbenchActionCommand.OPEN_INSPECTOR: self.inspector_tab,
        }
        if self.command in required and required[self.command] is None:
            raise ValueError(f"{self.command.value} is missing its typed argument")
        if self.command is WorkbenchActionCommand.SET_RELATION_FILTER and not self.filter_ids:
            raise ValueError("SET_RELATION_FILTER requires at least one filter id")
        return self


class ActionErrorCode(str, Enum):
    STALE_CONTEXT = "STALE_CONTEXT"
    UNRESOLVABLE_REF = "UNRESOLVABLE_REF"
    FORBIDDEN = "FORBIDDEN"


class WorkbenchActionAcknowledgementV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["workbench_action_acknowledgement.v1"] = (
        "workbench_action_acknowledgement.v1"
    )
    command: WorkbenchActionCommand
    accepted: bool
    context_ref: str
    error_code: ActionErrorCode | None = None


class WorkbenchActionResolver:
    """Validates presentation actions without changing semantic context."""

    def __init__(
        self,
        *,
        context: UnifiedWorkbenchContextV1,
        resolvable_refs: set[str],
        allowed_commands: set[WorkbenchActionCommand],
    ) -> None:
        self._context = context
        self._resolvable_refs = frozenset(resolvable_refs)
        self._allowed_commands = frozenset(allowed_commands)

    def resolve(
        self, action: WorkbenchViewActionV1
    ) -> WorkbenchActionAcknowledgementV1:
        error: ActionErrorCode | None = None
        if action.material_context_revision != self._context.material_context_revision:
            error = ActionErrorCode.STALE_CONTEXT
        elif action.context_ref not in self._resolvable_refs:
            error = ActionErrorCode.UNRESOLVABLE_REF
        elif action.command not in self._allowed_commands:
            error = ActionErrorCode.FORBIDDEN
        return WorkbenchActionAcknowledgementV1(
            command=action.command,
            accepted=error is None,
            context_ref=action.context_ref,
            error_code=error,
        )


__all__ = [
    "ActionErrorCode",
    "UnifiedWorkbenchContextV1",
    "WorkbenchActionAcknowledgementV1",
    "WorkbenchActionCommand",
    "WorkbenchActionResolver",
    "WorkbenchViewActionV1",
    "WorkflowProjectionV1",
    "WorkSurface",
]
