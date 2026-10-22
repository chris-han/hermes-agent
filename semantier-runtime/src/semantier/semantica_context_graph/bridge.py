"""Pinned, single-flight bridge contract for the Phase1-001 Runtime Expert.

This module is infrastructure only.  It deliberately does not register Hermes
tools; the executable graph surface remains owned by the runtime plugin.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping


TOOL_CONTRACT_VERSION = "semantier.tender_contextgraph_runtime.v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_INTERACTION_COMMANDS = {"focus_node", "highlight_path"}


class TenderExpertContractError(ValueError):
    """Raised when a bridge request is not safely pinned or scoped."""


class TenderExpertBusyError(TenderExpertContractError):
    """Raised when a specialist session already has an in-flight turn."""


def specialist_session_id(
    *, organization_id: str, workspace_id: str, parent_session_id: str
) -> str:
    """Derive a collision-resistant, deterministic child session identifier."""
    values = (organization_id, workspace_id, parent_session_id)
    if any(not value or not _SAFE_ID.fullmatch(value) for value in values):
        raise TenderExpertContractError("invalid specialist session identity")
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:24]
    return f"tender-runtime-expert:{organization_id}:{workspace_id}:{digest}"


@dataclass(frozen=True)
class TenderExpertRequest:
    organization_id: str
    workspace_id: str
    parent_session_id: str
    graph_ref: str
    graph_version: str
    graph_hash: str
    semantica_commit: str
    message: str
    run_mode: str = "evaluation_baseline"
    tool_contract_version: str = TOOL_CONTRACT_VERSION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    request_id: str = ""
    selected_node_ids: tuple[str, ...] = ()
    selected_edge_ids: tuple[str, ...] = ()
    source_anchors: tuple[Mapping[str, Any], ...] = ()
    mapper_version: str = "semantier-tender-mapper.v1"

    def __post_init__(self) -> None:
        required = {
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "parent_session_id": self.parent_session_id,
            "graph_ref": self.graph_ref,
            "graph_version": self.graph_version,
            "graph_hash": self.graph_hash,
            "semantica_commit": self.semantica_commit,
            "message": self.message,
        }
        if any(not str(value).strip() for value in required.values()):
            raise TenderExpertContractError("request contains an empty required pin")
        if not self.mapper_version.strip():
            raise TenderExpertContractError("mapper version is required")
        if self.run_mode not in {"evaluation_baseline", "authoritative"}:
            raise TenderExpertContractError("unsupported Runtime Expert run mode")
        if self.tool_contract_version != TOOL_CONTRACT_VERSION:
            raise TenderExpertContractError("unsupported Runtime Expert tool contract")
        if not 0 < self.timeout_seconds <= DEFAULT_TIMEOUT_SECONDS:
            raise TenderExpertContractError("timeout must be in (0, 60] seconds")
        if self.request_id and not _SAFE_ID.fullmatch(self.request_id):
            raise TenderExpertContractError("invalid request id")

    @property
    def specialist_session_id(self) -> str:
        return specialist_session_id(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            parent_session_id=self.parent_session_id,
        )


@dataclass(frozen=True)
class TenderExpertResult:
    status: str
    request_id: str
    specialist_session_id: str
    message: str = ""
    evidence: tuple[Mapping[str, Any], ...] = ()
    graph_refs: tuple[Mapping[str, Any], ...] = ()
    interaction: Mapping[str, Any] | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "cancelled"}:
            raise TenderExpertContractError("invalid specialist result status")
        if self.status == "completed" and self.error_code:
            raise TenderExpertContractError("completed result cannot contain an error")

    @classmethod
    def from_chat_payload(
        cls, request: TenderExpertRequest, payload: Mapping[str, Any]
    ) -> "TenderExpertResult":
        """Convert the bounded Hermes response into the bridge result shape."""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return cls(
                status="failed",
                request_id=request.request_id,
                specialist_session_id=request.specialist_session_id,
                error_code="invalid_completion_payload",
            )
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            return cls(
                status="failed",
                request_id=request.request_id,
                specialist_session_id=request.specialist_session_id,
                error_code="missing_completed_message",
            )
        envelope = payload.get("tender_contextgraph")
        if not isinstance(envelope, Mapping):
            envelope = {}
        raw_evidence = envelope.get("evidence", ())
        raw_graph_refs = envelope.get("graph_refs", ())
        evidence = tuple(item for item in raw_evidence if isinstance(item, Mapping)) if isinstance(raw_evidence, list) else ()
        graph_refs = tuple(item for item in raw_graph_refs if isinstance(item, Mapping)) if isinstance(raw_graph_refs, list) else ()
        interaction = envelope.get("interaction")
        if isinstance(interaction, Mapping):
            command = str(interaction.get("command") or "")
            if command not in _INTERACTION_COMMANDS:
                interaction = None
            else:
                bounded = {"command": command}
                for field in ("nodeId", "sourceNodeId", "targetNodeId"):
                    value = interaction.get(field)
                    if isinstance(value, str) and _SAFE_ID.fullmatch(value):
                        bounded[field] = value
                for field in ("nodeIds", "edgeIds"):
                    values = interaction.get(field)
                    if isinstance(values, list) and all(isinstance(value, str) and _SAFE_ID.fullmatch(value) for value in values):
                        bounded[field] = values[:100]
                interaction = bounded
        else:
            interaction = None
        return cls(
            status="completed",
            request_id=request.request_id,
            specialist_session_id=request.specialist_session_id,
            message=content,
            evidence=evidence,
            graph_refs=graph_refs,
            interaction=interaction,
        )


class TenderExpertBridge:
    """Enforce one in-flight turn and bounded cancellation/retry semantics."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def _lock_for(self, session_id: str) -> asyncio.Lock:
        async with self._guard:
            return self._locks.setdefault(session_id, asyncio.Lock())

    async def run(
        self,
        request: TenderExpertRequest,
        handler: Callable[[TenderExpertRequest], Awaitable[TenderExpertResult]],
        *,
        retry_once: bool = True,
    ) -> TenderExpertResult:
        lock = await self._lock_for(request.specialist_session_id)
        if lock.locked():
            raise TenderExpertBusyError("specialist session already has an in-flight turn")
        async with lock:
            attempts = 2 if retry_once else 1
            for attempt in range(attempts):
                try:
                    result = await asyncio.wait_for(
                        handler(request), timeout=request.timeout_seconds
                    )
                except asyncio.CancelledError:
                    return TenderExpertResult(
                        status="cancelled",
                        request_id=request.request_id,
                        specialist_session_id=request.specialist_session_id,
                        error_code="cancelled",
                    )
                except asyncio.TimeoutError:
                    if attempt + 1 < attempts:
                        continue
                    return TenderExpertResult(
                        status="failed",
                        request_id=request.request_id,
                        specialist_session_id=request.specialist_session_id,
                        error_code="timeout",
                    )
                if result.status == "completed":
                    return result
                if result.status != "failed" or not result.error_code or attempt + 1 >= attempts:
                    return result
            raise AssertionError("unreachable retry state")
