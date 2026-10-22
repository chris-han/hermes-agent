"""Typed, non-executing action proposals emitted by the Chat surface."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_ACTION_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_GOVERNED_OPERATION = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*$")


class ChatActionKind(str, Enum):
    ADVICE = "advice"
    VIEW_COMMAND = "view_command"
    GOVERNED_MUTATION_PROPOSAL = "governed_mutation_proposal"


class ChatActionProposalV1(BaseModel):
    """Validated intent only; invoking the named action is a separate boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["chat_action_proposal.v1"] = Field(alias="schemaVersion")
    action_kind: ChatActionKind = Field(alias="actionKind")
    action_name: str = Field(alias="actionName")
    message_ref: str = Field(alias="messageRef")
    object_refs: tuple[str, ...] = Field(default=(), alias="objectRefs")
    governed_operation: str | None = Field(default=None, alias="governedOperation")
    decision_ref: str | None = Field(default=None, alias="decisionRef")
    rationale: str | None = None
    justification_ref: str | None = Field(default=None, alias="justificationRef")

    @field_validator("action_name")
    @classmethod
    def action_name_is_stable(cls, value: str) -> str:
        if _ACTION_NAME.fullmatch(value) is None:
            raise ValueError("actionName must be an ASCII-stable uppercase identifier")
        return value

    @field_validator("message_ref", "decision_ref", "justification_ref")
    @classmethod
    def refs_are_ascii(cls, value: str | None, info) -> str | None:
        if value is not None and _ASCII_REF.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be an ASCII-stable reference")
        return value

    @field_validator("object_refs", mode="before")
    @classmethod
    def canonical_object_refs(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("objectRefs must be an array")
        refs = tuple(value)
        if any(not isinstance(ref, str) or _ASCII_REF.fullmatch(ref) is None for ref in refs):
            raise ValueError("objectRefs must contain ASCII-stable references")
        if len(refs) != len(set(refs)):
            raise ValueError("objectRefs contains a duplicate reference")
        return tuple(sorted(refs, key=lambda ref: ref.encode("utf-8")))

    @model_validator(mode="after")
    def action_shape_matches_kind(self) -> "ChatActionProposalV1":
        governed_fields = (
            self.governed_operation,
            self.decision_ref,
            self.rationale,
            self.justification_ref,
        )
        if self.action_kind is ChatActionKind.ADVICE:
            if self.object_refs or any(value is not None for value in governed_fields):
                raise ValueError("advice must not carry executable or governed fields")
        elif self.action_kind is ChatActionKind.VIEW_COMMAND:
            if not self.object_refs:
                raise ValueError("view command requires objectRefs")
            if any(value is not None for value in governed_fields):
                raise ValueError("view command must remain presentation-only")
        else:
            if self.governed_operation is None:
                raise ValueError("governedOperation is required")
            if _GOVERNED_OPERATION.fullmatch(self.governed_operation) is None:
                raise ValueError("governedOperation must name a governed object operation")
            if self.governed_operation == "ReviewItem.set_state":
                raise ValueError("direct Review Queue state mutation is forbidden")
            if self.decision_ref is None:
                raise ValueError("decisionRef is required")
            if not self.rationale or not self.rationale.strip():
                raise ValueError("rationale is required")
            if self.justification_ref != self.message_ref:
                raise ValueError("justificationRef must point to the proposing Chat message")
        return self


def parse_chat_action_proposal(
    chat_input: str,
    *,
    object_ref_exists: Callable[[str], bool],
    decision_ref_is_current: Callable[[str], bool],
) -> ChatActionProposalV1:
    """Parse and validate a proposal against current governed references."""

    try:
        payload = json.loads(chat_input)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Chat action proposal must be valid JSON") from exc
    proposal = ChatActionProposalV1.model_validate(payload)

    if proposal.action_kind is ChatActionKind.VIEW_COMMAND:
        if any(not object_ref_exists(ref) for ref in proposal.object_refs):
            raise ValueError("objectRefs contains a stale or missing reference")
    elif proposal.action_kind is ChatActionKind.GOVERNED_MUTATION_PROPOSAL:
        if not decision_ref_is_current(proposal.decision_ref or ""):
            raise ValueError("decisionRef is stale or missing")
        if any(not object_ref_exists(ref) for ref in proposal.object_refs):
            raise ValueError("objectRefs contains a stale or missing reference")
    return proposal


__all__ = ["ChatActionKind", "ChatActionProposalV1", "parse_chat_action_proposal"]
