"""Deterministic Stage-A lowering for project-memory state.

Only explicit, structured, replayable state is lowered here. Anything requiring
implication or pragmatic interpretation remains unresolved for Stage B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from contracts.project_memory_state import (
    ProjectStateDerivationV1,
)


@dataclass(frozen=True, slots=True)
class LoweredProjectState:
    epistemic_state: str
    authority_state: str | None
    execution_state: str | None
    verification_state: str | None
    derivations: tuple[ProjectStateDerivationV1, ...]
    unresolved_dimensions: tuple[str, ...]
    diagnostics: tuple[str, ...]


_CLAIM_TOKENS = (
    "claims",
    "claimed",
    "says",
    "reports",
    "chat message",
    "chat note",
    "operator says",
    "assistant says",
    "assistant reports",
    "群聊",
    "声称",
    "称",
    "助手",
    "据称",
)

_INFERRED_TOKENS = (
    "suggests that",
    "implies that",
    "appears to",
    "likely",
    "推断",
    "似乎",
    "可能意味着",
)

_UNAPPROVED_TOKENS = (
    "not approved",
    "no approval",
    "approval absent",
    "approval is still absent",
    "unapproved",
    "尚未批准",
    "未获批准",
    "仍没有生产批准",
    "批准仍然缺失",
)

_APPROVED_TOKENS = (
    "formally approved",
    "approval granted",
    "formally authorized",
    "正式批准",
    "已批准",
    "获准",
)

_PROPOSED_TOKENS = (
    "proposal",
    "proposed",
    "recommends",
    "recommendation",
    "建议",
    "提议",
    "推荐",
)

_ACTIVATED_TOKENS = (
    "activated",
    "is activated",
    "activation completed",
    "已激活",
    "已经激活",
)

_NOT_RUN_TOKENS = (
    "not_run",
    "not run",
    "not started",
    "has not been launched",
    "no run",
    "未运行",
    "未启动",
    "尚未启动",
    "没有启动",
)

_RUNNING_TOKENS = (
    "running",
    "in progress",
    "正在运行",
    "执行中",
)

_SUCCEEDED_TOKENS = (
    "deployment completed successfully",
    "deployed successfully",
    "execution succeeded",
    "successfully deployed",
    "部署任务已经成功完成",
    "已经成功部署",
    "执行成功",
)

_FAILED_TOKENS = (
    "execution failed",
    "deployment failed",
    "job failed",
    "执行失败",
    "部署失败",
)

_VERIFICATION_PASS_TOKENS = (
    "verification passed",
    "tests passed",
    "check passed",
    "checks passed",
    "verification report records a pass",
    "验收通过",
    "测试通过",
    "验收报告记录全部测试通过",
)

_VERIFICATION_FAILED_TOKENS = (
    "verification failed",
    "tests failed",
    "验收失败",
    "测试失败",
)


def _contains(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _structured_status(fields: Mapping[str, Any] | None, *names: str) -> str | None:
    if not fields:
        return None
    for name in names:
        value = fields.get(name)
        if value is not None:
            return str(value).strip().casefold()
    return None


class ProjectStateLowerer:
    """Lower explicit project state without semantic inference."""

    def lower(
        self,
        *,
        text: str,
        source_role: str,
        source_locator_refs: tuple[str, ...] = (),
        structured_fields: Mapping[str, Any] | None = None,
    ) -> LoweredProjectState:
        t = text.casefold()
        diagnostics: list[str] = []
        unresolved: set[str] = set()
        derivations: list[ProjectStateDerivationV1] = []

        if _contains(t, _CLAIM_TOKENS):
            epistemic = "CLAIMED"
            epi_method = "EXPLICIT_SOURCE_TOKEN"
        elif _contains(t, _INFERRED_TOKENS):
            epistemic = "INFERRED"
            epi_method = "EXPLICIT_SOURCE_TOKEN"
        else:
            epistemic = "OBSERVED"
            epi_method = "SOURCE_ROLE_CONSTRAINT"
        derivations.append(
            ProjectStateDerivationV1(
                dimension="epistemic",
                method=epi_method,
                source_locator_refs=source_locator_refs,
            )
        )

        authority = self._authority(
            text=t,
            source_role=source_role,
            fields=structured_fields,
            claimed=epistemic == "CLAIMED",
            locators=source_locator_refs,
            derivations=derivations,
            unresolved=unresolved,
            diagnostics=diagnostics,
        )
        execution = self._execution(
            text=t,
            source_role=source_role,
            fields=structured_fields,
            claimed=epistemic == "CLAIMED",
            locators=source_locator_refs,
            derivations=derivations,
            unresolved=unresolved,
            diagnostics=diagnostics,
        )
        verification = self._verification(
            text=t,
            source_role=source_role,
            fields=structured_fields,
            claimed=epistemic == "CLAIMED",
            locators=source_locator_refs,
            derivations=derivations,
            unresolved=unresolved,
            diagnostics=diagnostics,
        )

        return LoweredProjectState(
            epistemic_state=epistemic,
            authority_state=authority,
            execution_state=execution,
            verification_state=verification,
            derivations=tuple(derivations),
            unresolved_dimensions=tuple(sorted(unresolved)),
            diagnostics=tuple(sorted(set(diagnostics))),
        )

    def _authority(
        self,
        *,
        text: str,
        source_role: str,
        fields: Mapping[str, Any] | None,
        claimed: bool,
        locators: tuple[str, ...],
        derivations: list[ProjectStateDerivationV1],
        unresolved: set[str],
        diagnostics: list[str],
    ) -> str | None:
        raw = _structured_status(fields, "authority_state", "approval_state")
        structured_map = {
            "proposed": "PROPOSED",
            "unapproved": "UNAPPROVED",
            "approved": "APPROVED",
            "activated": "ACTIVATED",
            "unknown": "UNKNOWN",
        }
        if raw in structured_map:
            state = structured_map[raw]
            if state in {"APPROVED", "ACTIVATED"} and source_role != "authority_record":
                diagnostics.append("AUTHORITY_STATE_INELIGIBLE_SOURCE")
                state = "UNKNOWN"
            derivations.append(
                ProjectStateDerivationV1(
                    dimension="authority",
                    method="EXACT_STRUCTURED_FIELD",
                    source_locator_refs=locators,
                )
            )
            return state

        applicable = (
            _contains(text, _UNAPPROVED_TOKENS)
            or _contains(text, _APPROVED_TOKENS)
            or _contains(text, _PROPOSED_TOKENS)
            or _contains(text, _ACTIVATED_TOKENS)
            or "approval" in text
            or "批准" in text
        )
        if not applicable:
            return None

        if _contains(text, _UNAPPROVED_TOKENS):
            state = "UNAPPROVED"
        elif _contains(text, _ACTIVATED_TOKENS):
            state = "ACTIVATED"
        elif _contains(text, _APPROVED_TOKENS):
            state = "APPROVED"
        elif _contains(text, _PROPOSED_TOKENS):
            state = "PROPOSED"
        else:
            state = "UNKNOWN"
            unresolved.add("authority")

        if state in {"APPROVED", "ACTIVATED"} and (source_role != "authority_record" or claimed):
            diagnostics.append("AUTHORITY_STATE_INELIGIBLE_SOURCE")
            state = "UNKNOWN"
            unresolved.add("authority")

        derivations.append(
            ProjectStateDerivationV1(
                dimension="authority",
                method="UNRESOLVED" if state == "UNKNOWN" else "EXPLICIT_SOURCE_TOKEN",
                source_locator_refs=locators,
            )
        )
        return state

    def _execution(
        self,
        *,
        text: str,
        source_role: str,
        fields: Mapping[str, Any] | None,
        claimed: bool,
        locators: tuple[str, ...],
        derivations: list[ProjectStateDerivationV1],
        unresolved: set[str],
        diagnostics: list[str],
    ) -> str | None:
        raw = _structured_status(fields, "execution_state", "job_state", "deployment_state")
        structured_map = {
            "not_run": "NOT_RUN",
            "not-run": "NOT_RUN",
            "running": "RUNNING",
            "succeeded": "SUCCEEDED",
            "success": "SUCCEEDED",
            "failed": "FAILED",
            "unknown": "UNKNOWN",
        }
        if raw in structured_map:
            state = structured_map[raw]
            if state != "UNKNOWN" and source_role != "execution_receipt":
                diagnostics.append("EXECUTION_STATE_INELIGIBLE_SOURCE")
                state = "UNKNOWN"
            derivations.append(
                ProjectStateDerivationV1(
                    dimension="execution",
                    method="EXACT_STRUCTURED_FIELD",
                    source_locator_refs=locators,
                )
            )
            return state

        applicable = (
            _contains(text, _NOT_RUN_TOKENS)
            or _contains(text, _RUNNING_TOKENS)
            or _contains(text, _SUCCEEDED_TOKENS)
            or _contains(text, _FAILED_TOKENS)
            or "deploy" in text
            or "部署" in text
            or "execution" in text
            or "执行" in text
        )
        if not applicable:
            return None

        if _contains(text, _NOT_RUN_TOKENS):
            state = "NOT_RUN"
        elif _contains(text, _FAILED_TOKENS):
            state = "FAILED"
        elif _contains(text, _SUCCEEDED_TOKENS):
            state = "SUCCEEDED"
        elif _contains(text, _RUNNING_TOKENS):
            state = "RUNNING"
        else:
            state = "UNKNOWN"
            unresolved.add("execution")

        if state != "UNKNOWN" and (source_role != "execution_receipt" or claimed):
            diagnostics.append("EXECUTION_STATE_INELIGIBLE_SOURCE")
            state = "UNKNOWN"
            unresolved.add("execution")

        derivations.append(
            ProjectStateDerivationV1(
                dimension="execution",
                method="UNRESOLVED" if state == "UNKNOWN" else "EXPLICIT_SOURCE_TOKEN",
                source_locator_refs=locators,
            )
        )
        return state

    def _verification(
        self,
        *,
        text: str,
        source_role: str,
        fields: Mapping[str, Any] | None,
        claimed: bool,
        locators: tuple[str, ...],
        derivations: list[ProjectStateDerivationV1],
        unresolved: set[str],
        diagnostics: list[str],
    ) -> str | None:
        raw = _structured_status(fields, "verification_state", "test_state", "validation_state")
        structured_map = {
            "not_run": "NOT_RUN",
            "not-run": "NOT_RUN",
            "passed": "PASSED",
            "pass": "PASSED",
            "failed": "FAILED",
            "unknown": "UNKNOWN",
        }
        if raw in structured_map:
            state = structured_map[raw]
            if state != "UNKNOWN" and source_role != "execution_receipt":
                diagnostics.append("VERIFICATION_STATE_INELIGIBLE_SOURCE")
                state = "UNKNOWN"
            derivations.append(
                ProjectStateDerivationV1(
                    dimension="verification",
                    method="EXACT_STRUCTURED_FIELD",
                    source_locator_refs=locators,
                )
            )
            return state

        verification_context = any(
            token in text
            for token in (
                "test",
                "verification",
                "validation",
                "check",
                "测试",
                "验收",
                "验证",
                "检查",
            )
        )
        if not verification_context:
            return None

        if _contains(text, _NOT_RUN_TOKENS):
            state = "NOT_RUN"
        elif _contains(text, _VERIFICATION_FAILED_TOKENS):
            state = "FAILED"
        elif _contains(text, _VERIFICATION_PASS_TOKENS):
            state = "PASSED"
        else:
            state = "UNKNOWN"
            unresolved.add("verification")

        if state != "UNKNOWN" and (source_role != "execution_receipt" or claimed):
            diagnostics.append("VERIFICATION_STATE_INELIGIBLE_SOURCE")
            state = "UNKNOWN"
            unresolved.add("verification")

        derivations.append(
            ProjectStateDerivationV1(
                dimension="verification",
                method="UNRESOLVED" if state == "UNKNOWN" else "EXPLICIT_SOURCE_TOKEN",
                source_locator_refs=locators,
            )
        )
        return state
