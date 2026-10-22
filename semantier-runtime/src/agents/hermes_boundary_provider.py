from __future__ import annotations

import base64
import json
import hashlib
import re
from typing import Mapping

from gateway.execution_boundary import (
    BoundaryPaths,
    BoundaryPolicy,
    ExecutionBoundary,
    ExecutionBoundaryRequest,
)
from runtime_paths import (
    workspace_root_path,
    workspace_session_segment,
    workspace_wiki_root,
)


def _trusted_internal(request: ExecutionBoundaryRequest) -> bool:
    if not bool(request.metadata.get("trusted_internal_boundary")):
        return False
    trust = (
        request.headers.get("X-Hermes-Execution-Boundary-Trust")
        or request.headers.get("x-hermes-execution-boundary-trust")
        or ""
    ).strip()
    return trust == "semantier-internal"


def _decode_internal_boundary_header(headers: Mapping[str, str]) -> dict[str, str]:
    raw = headers.get("X-Hermes-Execution-Boundary") or headers.get(
        "x-hermes-execution-boundary"
    )
    if not raw:
        return {}
    decoded = base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8")
    payload = json.loads(decoded)
    return payload if isinstance(payload, dict) else {}


def durable_request_metadata(identity: object, original_message: str, supplied_hash: object = None) -> dict[str, str]:
    """Bind caller correlation to original intent without granting authority."""
    if identity is None:
        if supplied_hash is not None:
            raise ValueError("requestHash requires requestIdentity")
        return {}
    if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
        raise ValueError("invalid requestIdentity")
    digest = hashlib.sha256(original_message.encode("utf-8")).hexdigest()
    if supplied_hash is not None and supplied_hash != digest:
        raise ValueError("requestHash does not match original message")
    return {"durable_request_id": identity, "durable_request_hash": digest, "durable_request_text": original_message}


class SemantierExecutionBoundaryProvider:
    def resolve(self, request: ExecutionBoundaryRequest) -> ExecutionBoundary | None:
        payload: dict[str, str] = {}
        if _trusted_internal(request):
            payload.update(_decode_internal_boundary_header(request.headers))
            payload.pop("durable_request_text", None)
            original_text = request.metadata.get("durable_request_text")
            if original_text is not None:
                payload["durable_request_text"] = original_text
            payload.setdefault(
                "workspace_id",
                request.headers.get("X-Semantier-Workspace-Id", ""),
            )
            payload.setdefault("user_id", request.headers.get("X-Semantier-User-Id", ""))
            payload.setdefault(
                "session_id",
                request.headers.get("X-Semantier-Upload-Session-Id", ""),
            )

        request_id = payload.get("durable_request_id")
        request_hash = payload.get("durable_request_hash")
        if request_id is not None or request_hash is not None:
            if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
                raise ValueError("invalid durable_request_id")
            if not isinstance(request_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", request_hash):
                raise ValueError("invalid durable_request_hash")
            original_text = payload.get("durable_request_text")
            if not isinstance(original_text, str) or hashlib.sha256(original_text.encode("utf-8")).hexdigest() != request_hash:
                raise ValueError("durable request text does not match intent hash")

        workspace_id = str(payload.get("workspace_id") or request.workspace_id or "").strip()
        if not workspace_id:
            return None

        user_id = str(payload.get("user_id") or request.user_id or "").strip() or None
        if (request.source == "gateway_runner"
                and request.metadata.get("platform") == "feishu"):
            from agents.feishu_ingress_identity import resolve_feishu_ingress_owner
            owner = resolve_feishu_ingress_owner(open_id=user_id,
                chat_id=request.metadata.get("chat_id"),
                platform_session_key=request.metadata.get("session_key"))
            if owner.owner_workspace_id != workspace_id:
                raise PermissionError("FEISHU_BOUNDARY_OWNER_MISMATCH")
            user_id = owner.owner_user_id

        session_id = str(payload.get("session_id") or request.session_id or "").strip()
        if not session_id:
            raise ValueError("session_id required for Semantier execution boundary")

        from agents.wiki_task_runtime import wiki_profile_actor
        from services.wiki_mutation_service import PROFILE_TOOLS, current_entitlement, validated_wiki_root
        profile_actor = wiki_profile_actor(workspace_id, session_id)
        if profile_actor is not None:
            if not _trusted_internal(request) or profile_actor != user_id:
                raise PermissionError("wiki profile requires trusted authenticated actor")
            current_entitlement(user_id, workspace_id)
            validated_wiki_root(workspace_id)

        workspace = workspace_root_path(workspace_id).resolve()
        session_segment = workspace_session_segment(workspace_id, session_id)
        session_root = workspace / "sessions" / session_segment
        runs = session_root / "runs"
        uploads = session_root / "uploads"
        artifacts = session_root / "artifacts"
        logs = session_root / "logs"
        for path in (runs, uploads, artifacts, logs):
            path.mkdir(parents=True, exist_ok=True)
        resolved_workspace = workspace.resolve()
        resolved_runs = runs.resolve()
        resolved_uploads = uploads.resolve()
        resolved_artifacts = artifacts.resolve()
        resolved_logs = logs.resolve()
        resolved_wiki = workspace_wiki_root(workspace_id).resolve()

        return ExecutionBoundary(
            source=request.source,
            session_id=session_id,
            user_id=user_id,
            workspace_id=workspace_id,
            paths=BoundaryPaths(
                hermes_home=resolved_workspace,
                terminal_cwd=resolved_workspace,
                runs_root=resolved_runs,
                uploads_root=resolved_uploads,
                artifacts_root=resolved_artifacts,
                logs_root=resolved_logs,
            ),
            policy=BoundaryPolicy(
                provider_fallback_enabled=False,
                require_boundary=True,
                allowed_tool_names=PROFILE_TOOLS if profile_actor is not None else None,
                allowed_read_roots=(
                    resolved_workspace,
                    resolved_uploads,
                    resolved_artifacts,
                ),
                allowed_write_roots=(
                    resolved_runs,
                    resolved_uploads,
                    resolved_artifacts,
                    resolved_logs,
                ),
            ),
            audit_metadata={
                "authority_source": "semantier_authenticated_context",
                **({"wiki_task_profile": "documentary-wiki.v1"} if profile_actor is not None else {}),
                "transport": request.metadata.get("transport", "unknown"),
                **{key: payload[key] for key in ("durable_request_id", "durable_request_hash", "durable_request_text") if key in payload},
            },
            subprocess_env={
                "SEMANTIER_WORKSPACE_RUNS_DIR": str(resolved_runs),
                "SEMANTIER_WORKSPACE_ARTIFACTS_DIR": str(resolved_artifacts),
                "WIKI_PATH": str(resolved_wiki),
            },
        )
