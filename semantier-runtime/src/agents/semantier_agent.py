from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from agents.adapter import SemantierAdapter
from agents.governed_context import build_governed_runtime_context_prompt
from agents.sandbox_scope import SandboxScope, bind_sandbox_scope
from agents.launcher import (
    _bootstrap_runtime_root,
    bootstrap_workspace_runtime_home,
    load_agent_prompt,
    load_default_soul_md,
    load_default_governance_md,
)
from runtime_paths import (
    bind_workspace_env,
    bind_workspace_session_env,
    platform_runtime_root,
    workspace_hermes_home_path,
)


_RUNTIME_HOME_BOOTSTRAP_LOCK = threading.RLock()
_BOOTSTRAPPED_RUNTIME_HOME_PATHS: set[Path] = set()


def ensure_semantier_runtime_home() -> None:
    """Create the shared Semantier runtime bootstrap surfaces on first launch."""
    runtime_root = platform_runtime_root().expanduser().resolve()
    with _RUNTIME_HOME_BOOTSTRAP_LOCK:
        if runtime_root in _BOOTSTRAPPED_RUNTIME_HOME_PATHS:
            return
        _bootstrap_runtime_root(runtime_root)
        _BOOTSTRAPPED_RUNTIME_HOME_PATHS.add(runtime_root)


@contextlib.contextmanager
def _hermes_home_bound(
    target_home: Path,
    *,
    session_id: str | None = None,
) -> Generator[None, None, None]:
    """Context manager that pins HERMES_HOME to the requested runtime home.

    Sets both the context-var (for in-process code that reads it) and the
    environment variables (for thread executors and subprocess spawns). The
    env-var contract (HERMES_HOME + SEMANTIER_WORKSPACE_RUNS_DIR) is owned
    by `runtime_paths.bind_workspace_env` so every gateway adapter binds
    identically. Thread-safe: multiple concurrent contexts with different
    homes are isolated by context-var, and env mutations are scoped here.
    """
    ensure_semantier_runtime_home()
    target_home = target_home.expanduser().resolve()
    shared_runtime_home = platform_runtime_root().expanduser().resolve()
    if target_home == shared_runtime_home:
        raise ValueError("authenticated Hermes execution requires a workspace runtime home")
    if _workspace_id_from_hermes_home(target_home) is None:
        raise ValueError("authenticated Hermes execution requires a governed workspace runtime home")
    bootstrap_workspace_runtime_home(
        target_home,
        shared_skills_external_dir=shared_runtime_home / "skills",
        shared_plugins_external_dir=shared_runtime_home / "plugins",
        workspace_id=_workspace_id_from_hermes_home(target_home),
    )

    binder = bind_workspace_session_env(target_home, session_id) if session_id else bind_workspace_env(target_home)
    with binder:
        yield


def _build_eos_mcp_config() -> dict | None:
    """Compatibility shim: eos_db is authoritative runtime state, not MCP."""
    return None


def _register_eos_mcp() -> None:
    """Compatibility shim retained for existing startup call sites."""
    return None


DEFAULT_SEMANTIER_SOUL_MD = load_default_soul_md()
DEFAULT_SEMANTIER_GOVERNANCE_MD = load_default_governance_md()


# Governance principles prepended to all agent conversations
WRAPPER_SYSTEM_INSTRUCTION = (
    DEFAULT_SEMANTIER_GOVERNANCE_MD.strip()
    + "\n\n"
    + load_agent_prompt("WRAPPER_SYSTEM_INSTRUCTION.md").strip()
    + "\n\n"
    + load_agent_prompt("DIGITAL_CFO_USER_GUIDANCE.md").strip()
)


def _workspace_id_from_hermes_home(hermes_home: Path) -> str | None:
    resolved = hermes_home.expanduser().resolve()
    workspace_id = resolved.name.strip()
    if not workspace_id:
        return None
    try:
        expected = workspace_hermes_home_path(workspace_id).resolve()
    except ValueError:
        return None
    if resolved != expected:
        return None
    return workspace_id


def _compose_effective_system_message(
    hermes_home: Path,
    *,
    user_message: str = "",
    user_id: str | None = None,
    evaluation_section: str | None = None,
) -> str:
    parts = [WRAPPER_SYSTEM_INSTRUCTION]
    try:
        governed_context = build_governed_runtime_context_prompt(
            workspace_id=_workspace_id_from_hermes_home(hermes_home),
            user_id=user_id,
            user_message=user_message,
        )
    except TypeError:
        governed_context = build_governed_runtime_context_prompt(None, user_message)
    if governed_context:
        parts.append(governed_context)
    if evaluation_section:
        parts.append(evaluation_section)
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


class SemantierAgent:
    """Semantier-owned wrapper around upstream hermes-agent AIAgent."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-sonnet-4-5",
    ):
        self.adapter = SemantierAdapter()
        self._base_url = base_url
        self._api_key = api_key
        self._model = model

    def run(self, program, semantic_projection_input, ontology, org_id):
        payload = {
            "org_id": org_id,
            "program": program,
            "semantic_projection_input": semantic_projection_input,
            "ontology": ontology,
            "justification": {},
        }
        return self.adapter.execute(payload)

    def run_conversation(
        self,
        prompt: str,
        *,
        hermes_home: Path | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        organization_id: str | None = None,
        authority_context_ref: str | None = None,
        retained_context=None,
        enabled_toolsets: list[str] | None = None,
        max_tokens: int | None = None,
        run_budget_seconds: float | None = None,
        evaluation_prompt_binding=None,
        evaluation_request_overrides: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Execute a Hermes conversation via upstream hermes-agent when available."""
        if evaluation_request_overrides is not None and evaluation_prompt_binding is None:
            raise PermissionError("evaluation request options require an EOS binding")
        if evaluation_request_overrides is not None:
            options = evaluation_request_overrides
            if (set(options) != {"max_completion_tokens", "extra_body", "extra_headers"}
                    or type(options["max_completion_tokens"]) is not int
                    or not 1 <= options["max_completion_tokens"] <= 2038
                    or options["extra_body"] != {}
                    or enabled_toolsets != [] or max_tokens is not None):
                raise ValueError("unsupported evaluation call options")
            headers = options["extra_headers"]
            if (not isinstance(headers, dict) or not headers.get("Idempotency-Key")
                    or set(headers) - {"Idempotency-Key", "x-semantier-session-id", "x-semantier-run-id", "x-semantier-purpose"}
                    or any(not isinstance(value, str) or not 1 <= len(value) <= 200
                           for value in headers.values())):
                raise ValueError("unsupported evaluation correlation headers")
        if hermes_home is None and workspace_id is None:
            raise ValueError(
                "run_conversation requires workspace_id or governed workspace hermes_home"
            )

        derived_workspace_id = workspace_id
        if hermes_home is not None:
            derived_from_home = _workspace_id_from_hermes_home(hermes_home)
            if derived_from_home is None:
                raise ValueError("run_conversation requires a governed workspace hermes_home")
            if workspace_id is not None and workspace_id != derived_from_home:
                raise ValueError("workspace_id does not match hermes_home")
            derived_workspace_id = derived_from_home

        from gateway.execution_boundary import (
            ExecutionBoundaryRequest,
            GovernedExecutionBoundaryRequired,
            bind_execution_boundary,
            resolve_execution_boundary,
        )

        boundary = resolve_execution_boundary(
            ExecutionBoundaryRequest(
                source="semantier_agent",
                session_id=session_id,
                user_id=user_id,
                workspace_id=derived_workspace_id,
                metadata={
                    "transport": "embedded",
                    "trusted_internal_boundary": True,
                },
                headers={
                    "X-Hermes-Execution-Boundary-Trust": "semantier-internal",
                },
            )
        )
        if boundary is None:
            raise GovernedExecutionBoundaryRequired(
                "Execution boundary required for SemantierAgent"
            )

        target_home = boundary.paths.hermes_home or hermes_home
        if target_home is None:
            raise GovernedExecutionBoundaryRequired(
                "Execution boundary missing hermes_home for SemantierAgent"
            )
        evaluation_prompt = None
        if evaluation_prompt_binding is not None:
            from agents.evaluation_prompt import EvaluationPromptBinding
            if not isinstance(evaluation_prompt_binding, EvaluationPromptBinding):
                raise TypeError("internal EOS evaluation prompt binding required")
            execution_profile = {
                "model": self._model, "base_url": self._base_url or "",
                "enabled_toolsets": enabled_toolsets, "max_tokens": max_tokens,
                "run_budget_seconds": run_budget_seconds,
            }
            if evaluation_request_overrides is not None:
                execution_profile["evaluation_request_overrides"] = evaluation_request_overrides
            evaluation_prompt = evaluation_prompt_binding.resolve(boundary, execution_profile=execution_profile)
        scope = None
        resolved_workspace_id = boundary.workspace_id or _workspace_id_from_hermes_home(target_home)
        if resolved_workspace_id and session_id:
            scope = SandboxScope(
                workspace_id=resolved_workspace_id,
                lane="interactive_session",
                scope_id=session_id,
                adapter_key=None,
                network_enabled=False,
            )

        try:
            from run_agent import AIAgent  # type: ignore[import]
        except Exception as exc:
            raise ImportError(
                "hermes-agent is not installed. Add it as a workspace dependency and sync the environment."
            ) from exc

        with bind_execution_boundary(boundary), _hermes_home_bound(
            target_home,
            session_id=boundary.session_id or session_id,
        ):
            from agents.runtime_memory_boundary import reconcile_retained_context_for_boundary

            reconcile_retained_context_for_boundary(boundary, context=retained_context)
            from tools.skill_review_capture import pin_generation, bind_stage_actor
            from services.hermes_skill_review_service import bind_runtime_package_resolver
            with bind_sandbox_scope(scope), bind_runtime_package_resolver(boundary), pin_generation(), bind_stage_actor(boundary.user_id, resolved_workspace_id):
                effective_context_sink = None
                effective_context_session_db = None
                if organization_id and authority_context_ref:
                    from semantier.effective_context import (
                        ActiveSemanticContract,
                        EffectiveContextCheckpointService,
                    )
                    from agents.hermes_checkpoint_persistence import CheckpointSessionDB

                    effective_context_service = EffectiveContextCheckpointService()

                    def effective_context_sink(identity: dict[str, Any]) -> None:
                        content_hash = hashlib.sha256(
                            json.dumps(
                                {
                                    "role": identity.get("role"),
                                    "content": identity.get("content"),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ).encode("utf-8")
                        ).hexdigest()
                        effective_context_service.record_message_committed(
                            organization_id=organization_id,
                            workspace_id=resolved_workspace_id,
                            session_id=identity["session_id"],
                            message_id=identity["message_id"],
                            message_content_hash=content_hash,
                            actor_ref="hermes:message_persistence",
                            authority_context_ref=authority_context_ref,
                        )
                        if identity.get("role") != "assistant" or identity.get("tool_calls"):
                            return
                        transcript_snapshot = identity["transcript_snapshot"]
                        effective_context_service.materialize_checkpoint(
                            organization_id=organization_id,
                            workspace_id=resolved_workspace_id,
                            session_id=identity["session_id"],
                            boundary_message_id=identity["message_id"],
                            actor_ref="hermes:message_persistence",
                            knowledge_state={},
                            active_contract=ActiveSemanticContract(),
                            authority_pins={
                                "historical_admission_proof_ref": authority_context_ref,
                                "principal_ref": user_id,
                                "organization_ref": organization_id,
                                "workspace_ref": resolved_workspace_id,
                                "model_ref": self._model,
                                "context_engine_version_ref": "hermes-agent:persisted-message-boundary",
                                **({"native_store_identity": identity["native_store_identity"]}
                                   if identity.get("native_store_identity") else {}),
                            },
                            transcript_snapshot=transcript_snapshot,
                        )

                    effective_context_session_db = CheckpointSessionDB(sink=effective_context_sink)

                agent = None
                try:
                    agent = AIAgent(
                        base_url=self._base_url or "",
                        api_key=self._api_key or "",
                        model=self._model,
                        max_iterations=1 if evaluation_request_overrides is not None else 20,
                        enabled_toolsets=enabled_toolsets,
                        max_tokens=max_tokens,
                        run_budget_seconds=run_budget_seconds,
                        quiet_mode=True,
                        save_trajectories=True,
                        skip_context_files=True,
                        **({"request_overrides": evaluation_request_overrides,
                            "skip_memory": True, "skip_background_review": True}
                           if evaluation_request_overrides is not None else {}),
                        session_db=effective_context_session_db,
                        session_id=boundary.session_id or session_id,
                    )
                    _register_eos_mcp()
                    system_message = _compose_effective_system_message(
                        target_home,
                        user_message=prompt,
                        user_id=user_id,
                        evaluation_section=evaluation_prompt["text"] if evaluation_prompt else None,
                    )
                    result = agent.run_conversation(prompt, system_message=system_message)
                    if evaluation_prompt is not None:
                        result = {**result, "evaluation_prompt_receipt": {
                            **evaluation_prompt["receipt"],
                            "composed_system_sha256": hashlib.sha256(system_message.encode()).hexdigest(),
                        }}
                    if effective_context_session_db is not None:
                        effective_context_session_db.raise_checkpoint_failure()
                    return result
                finally:
                    try:
                        if agent is not None and callable(getattr(agent, "close", None)):
                            agent.close()
                    finally:
                        if effective_context_session_db is not None:
                            effective_context_session_db.close()
