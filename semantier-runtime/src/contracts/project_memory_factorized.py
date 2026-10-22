"""Factorized residual observations for project-memory Stage B."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


TriState = Literal["YES", "NO", "UNKNOWN"]
TerminalEvent = Literal["SUCCESS", "FAILURE", "NONE", "UNKNOWN"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuthorityObservationV1(_Contract):
    proposal_present: TriState
    authorization_present: TriState
    authorization_effective: TriState
    required_authorization_absent: TriState

    @model_validator(mode="after")
    def reject_contradictions(self):
        if self.proposal_present == "YES" and self.authorization_present == "YES":
            raise ValueError(
                "proposal and authorization cannot both be positively asserted"
            )
        if (
            self.authorization_effective == "YES"
            and self.authorization_present != "YES"
        ):
            raise ValueError(
                "effective authorization requires authorization_present=YES"
            )
        if (
            self.required_authorization_absent == "YES"
            and self.authorization_present == "YES"
        ):
            raise ValueError("required authorization cannot be both absent and present")
        return self


class ExecutionObservationV1(_Contract):
    run_started: TriState
    activity_current: TriState
    terminal_event: TerminalEvent

    @model_validator(mode="after")
    def reject_contradictions(self):
        if self.run_started == "NO" and self.activity_current == "YES":
            raise ValueError("current activity requires a started run")
        if self.run_started == "NO" and self.terminal_event in {"SUCCESS", "FAILURE"}:
            raise ValueError("terminal event requires a started run")
        if self.activity_current == "YES" and self.terminal_event in {
            "SUCCESS",
            "FAILURE",
        }:
            raise ValueError("current activity conflicts with terminal completion")
        return self
