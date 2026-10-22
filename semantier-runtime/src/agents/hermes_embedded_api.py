from __future__ import annotations

import asyncio
import secrets
import socket


class EmbeddedHermesAPIService:
    """Own an in-process Hermes API server for the Semantier web surface."""

    def __init__(self) -> None:
        self._adapter = None
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._lock = asyncio.Lock()
        self._loop = None
        self._continuation_tasks = set()
        self._chat_pool = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("embedded Hermes API server is not started")
        return self._base_url

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            raise RuntimeError("embedded Hermes API server is not started")
        return self._api_key

    def chat_execution_pool(self):
        if self._chat_pool is None:
            import os
            from agents.chat_execution_pool import ChatExecutionPool
            self._chat_pool = ChatExecutionPool(
                max_workers=int(os.environ.get('SEMANTIER_CHAT_MAX_WORKERS', '4')),
                queue_timeout=float(os.environ.get('SEMANTIER_CHAT_QUEUE_TIMEOUT', '60')),
                turn_timeout=float(os.environ.get('SEMANTIER_CHAT_TURN_TIMEOUT', '120')))
        return self._chat_pool

    async def ensure_started(self, *, reuse_provisioned_runtime: bool = False) -> None:
        if (
            self._adapter is not None
            and self._base_url is not None
            and self._api_key is not None
        ):
            return

        async with self._lock:
            if (
                self._adapter is not None
                and self._base_url is not None
                and self._api_key is not None
            ):
                return

            from agents.launcher import _ensure_repo_local_runtime
            from agents.semantier_agent import (
                _register_eos_mcp,
                ensure_semantier_runtime_home,
            )

            if reuse_provisioned_runtime:
                from runtime_paths import platform_runtime_root
                from agents.launcher import _ensure_repo_hermes_agent_on_path
                from agents.semantier_agent import _BOOTSTRAPPED_RUNTIME_HOME_PATHS
                from agents.cron_home_provider import register_semantier_cron_home_provider
                root = platform_runtime_root()
                if not (root / 'config.yaml').is_file():
                    raise RuntimeError('CHAT_SHARED_RUNTIME_NOT_READY')
                _ensure_repo_hermes_agent_on_path()
                register_semantier_cron_home_provider()
                _BOOTSTRAPPED_RUNTIME_HOME_PATHS.add(root)
            else:
                ensure_semantier_runtime_home()
                _ensure_repo_local_runtime()
            _register_eos_mcp()

            from agents.hermes_boundary_provider import (
                SemantierExecutionBoundaryProvider,
            )
            from gateway.config import PlatformConfig
            from gateway.execution_boundary import replace_execution_boundary_provider
            from gateway.platforms.api_server import APIServerAdapter
            from gateway.run import _load_gateway_config, _resolve_gateway_model

            # config.yaml is the application-owned source of truth.  Do not
            # let a caller's shell environment silently change the model used
            # by an authenticated embedded session.
            gateway_config = _load_gateway_config()
            configured_model = _resolve_gateway_model(gateway_config).strip() or None

            from agents.wiki_task_runtime import register_wiki_tools

            register_wiki_tools()
            replace_execution_boundary_provider(SemantierExecutionBoundaryProvider())
            port = _reserve_loopback_port()
            api_key = f"sk-semantier-{secrets.token_hex(16)}"
            adapter = APIServerAdapter(
                PlatformConfig(
                    enabled=True,
                    extra={
                        "host": "127.0.0.1",
                        "port": port,
                        "key": api_key,
                        # The embedded API receives chat payloads from the
                        # authenticated gateway without a per-request model.
                        # Bind its advertised/default model to config.yaml.
                        "model_name": configured_model,
                    },
                )
            )
            adapter._semantier_embedded_boundary_required = True
            from agents.meeting_api_origin import admitted_api_actor

            adapter._semantier_api_actor_scope = admitted_api_actor
            connected = await adapter.connect()
            if not connected:
                raise RuntimeError("failed to start embedded Hermes API server")

            self._loop = asyncio.get_running_loop()
            self._adapter = adapter
            self._base_url = f"http://127.0.0.1:{port}"
            self._api_key = api_key

    async def recover_meeting_continuations(self, state_db_path, *, session_key=None):
        """Bounded recovery over persisted admissions; no new scheduler or live calls."""
        from agents.internal_continuation import (
            ContinuationAdmissionStore,
            ContinuationDispatchService,
        )
        from feishu_meeting_coordinator.store import MeetingCoordinatorStore

        store = ContinuationAdmissionStore(state_db_path)
        provider = APIInternalContinuationProvider(
            self, MeetingCoordinatorStore(state_db_path)
        )
        results = []
        for record in store.pending():
            import json

            if (
                session_key is None or record.session_key == session_key
            ) and "api_actor_origin_hash" in json.loads(record.payload_json):
                results.append(
                    await ContinuationDispatchService(store).dispatch(
                        continuation_id=record.continuation_id, provider=provider
                    )
                )
        return results

    async def shutdown(self) -> None:
        if self._chat_pool is not None:
            await self._chat_pool.shutdown()
            self._chat_pool = None
        async with self._lock:
            adapter = self._adapter
            self._adapter = None
            self._base_url = None
            self._api_key = None
            if adapter is not None:
                await adapter.disconnect()


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


_SERVICE: EmbeddedHermesAPIService | None = None


def get_embedded_hermes_api_service() -> EmbeddedHermesAPIService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = EmbeddedHermesAPIService()
    return _SERVICE


class APIInternalContinuationProvider:
    """Core-owned native API dispatch; durable acceptance is separate from result."""

    revalidate_dispatched = True

    def __init__(self, service, domain_store):
        self.service = service
        self.domain_store = domain_store
        self.detail = None

    async def dispatch(self, record):
        import dataclasses
        import json
        from agents.internal_continuation import (
            UNAVAILABLE,
            REJECTED_STALE,
            REJECTED_AUTHORITY,
            ACCEPTED,
            ALREADY_ACCEPTED,
        )
        from agents.auth_session import RequestContext
        from agents.durable_event_arbitration import StaleWorkflowVersion
        from agents.gateway_identity import (
            get_user_record,
            get_user_organization_context,
            ensure_workspace_paths,
        )
        from contracts.workflow_execution import (
            authorize_meeting_requester_mutation,
            hash_value,
            verify_meeting_binding,
        )
        from gateway.config import GatewayConfig, Platform
        from gateway.session import SessionStore
        from gateway.execution_boundary import ExecutionBoundaryRequest
        from agents.hermes_boundary_provider import SemantierExecutionBoundaryProvider
        from agents.internal_continuation import ContinuationAdmissionStore, derive_continuation_id
        import hashlib

        admitted = ContinuationAdmissionStore(self.domain_store.path).get(
            record.continuation_id
        )
        if (
            admitted != record
            or hashlib.sha256(record.payload_json.encode()).hexdigest()
            != record.payload_hash
            or derive_continuation_id(
                workspace_id=record.workspace_id,
                workflow_id=record.workflow_id,
                event_id=record.event_id,
                session_key=record.session_key,
                payload_json=record.payload_json,
            )
            != record.continuation_id
        ):
            return REJECTED_STALE
        case = self.domain_store.get_negotiation(record.workflow_id)
        origin = json.loads(case.get("actor_origin_json") or "null")
        payload = json.loads(record.payload_json)
        if payload.get("kind") != "AGENT_CONTINUE":
            return REJECTED_STALE
        if (
            origin is None
            or origin.get("schema_version") != "meeting_api_actor_origin.v1"
            or origin.get("adapter_owner") != "semantier_embedded_api"
            or payload.get("api_actor_origin_hash") != hash_value(origin)
            or record.session_key != origin["session_key"]
            or record.workspace_id != origin["workspace_id"]
        ):
            return REJECTED_STALE
        user = get_user_record(origin["actor_user_id"])
        org = get_user_organization_context(origin["actor_user_id"]) or {}
        if (
            not user
            or user.get("workspace_id") != origin["workspace_id"]
            or org.get("organization_id") != origin["organization_id"]
        ):
            return REJECTED_AUTHORITY
        root, home = ensure_workspace_paths(origin["workspace_id"])
        if self.domain_store.path.resolve() != (home / "state.db").resolve():
            return REJECTED_STALE
        ctx = RequestContext(
            authenticated=True,
            user_id=origin["actor_user_id"],
            organization_id=origin["organization_id"],
            workspace_id=origin["workspace_id"],
            workspace_slug=user.get("workspace_slug") or origin["workspace_id"],
            workspace_root=root,
            hermes_home=home,
        )
        try:
            authorize_meeting_requester_mutation(case, ctx.user_id, ctx)
            binding = json.loads(case["workflow_binding_json"])
            verify_meeting_binding(
                binding,
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
            )
            if hash_value(binding) != origin["workflow_binding_hash"] or case[
                "status"
            ] in {"consented", "requester_decided", "cancelled", "expired", "failed"}:
                return REJECTED_STALE
        except (PermissionError, ValueError):
            return REJECTED_AUTHORITY
        adapter = self.service._adapter
        if adapter is None:
            return UNAVAILABLE
        db = adapter._ensure_session_db()
        routes = SessionStore(
            adapter._session_db_home() / "sessions", GatewayConfig(), session_db=db
        )
        routes._db = db
        entry = routes.lookup_by_session_key(record.session_key)
        row = db.get_session(origin["session_id"])
        if (
            entry is None
            or entry.session_id != origin["session_id"]
            or entry.origin is None
            or entry.origin.to_dict() != origin["source"]
            or entry.origin.platform != Platform.API_SERVER
            or not row
            or row.get("ended_at") is not None
            or row.get("session_key") != record.session_key
        ):
            return REJECTED_STALE
        # UNKNOWN is non-compensable. A dependent native clarification cannot turn
        # uncertain transport into a completed effect.
        from storage import hosted_postgres as hosted
        with self.domain_store._connect() as conn:
            if hosted.is_postgres(conn):
                hosted.require_tables(conn,"semantier_effect_outbox")
                unknown=conn.execute(
                    "SELECT 1 FROM semantier_effect_outbox WHERE workflow_id=%s AND state IN ('DISPATCHING','EFFECT_UNKNOWN') LIMIT 1",
                    (record.workflow_id,),
                ).fetchone()
            else:
                unknown = (
                    conn.execute(
                        "SELECT 1 FROM semantier_effect_outbox WHERE workflow_id=? AND state IN ('DISPATCHING','EFFECT_UNKNOWN') LIMIT 1",
                        (record.workflow_id,),
                    ).fetchone()
                    if conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE name='semantier_effect_outbox'"
                    ).fetchone()
                    else None
                )
        if unknown:
            self.detail = "unknown_effect_blocks_continuation"
            return UNAVAILABLE
        result = self.domain_store.get_api_continuation_result(
            record.workflow_id, record.continuation_id
        )
        if result is not None and (
            result["origin_hash"] != hash_value(origin)
            or result["input_hash"] != record.payload_hash
            or result["response_hash"] != hash_value(result["response"])
        ):
            return REJECTED_STALE
        was_completed = result is not None
        if result is None:
            if db.has_platform_message_id(entry.session_id, record.continuation_id):
                self.detail = "accepted_but_incomplete"
                return UNAVAILABLE
            # Persist acceptance before the model. Crash/lease expiry never resends
            # the internal turn; another owner sees this marker and parks.
            holder = f"pid={__import__('os').getpid()}:api-continuation:{__import__('uuid').uuid4()}"
            if not db.try_acquire_session_turn_lease(
                entry.session_id, holder, ttl_seconds=120
            ):
                return UNAVAILABLE
            try:
                if db.has_platform_message_id(entry.session_id, record.continuation_id):
                    return UNAVAILABLE
                db.append_message(
                    entry.session_id,
                    "user",
                    record.payload_json,
                    platform_message_id=record.continuation_id,
                    display_kind="hidden",
                    display_metadata={"internal": True, "allow_gateway_control": False},
                    turn_lease_holder=holder,
                )
                boundary = SemantierExecutionBoundaryProvider().resolve(
                    ExecutionBoundaryRequest(
                        source="api_server",
                        session_id=entry.session_id,
                        user_id=ctx.user_id,
                        workspace_id=ctx.workspace_id,
                        metadata={"transport": "internal_continuation"},
                    )
                )
                # This continuation produces clarification text only. It cannot
                # dispatch arbitrary tools or consequential external effects.
                boundary = dataclasses.replace(
                    boundary,
                    policy=dataclasses.replace(boundary.policy, allowed_tool_names=()),
                )
                history = db.get_messages_as_conversation(entry.session_id)
                output, _ = await adapter._run_agent(
                    record.payload_json,
                    history,
                    session_id=entry.session_id,
                    gateway_session_key=record.session_key,
                    execution_boundary=boundary,
                    internal_continuation_id=record.continuation_id,
                    internal_turn_lease_holder=holder,
                    resume_unanswered_turn=True,
                )
                if output.get("session_id", entry.session_id) != entry.session_id:
                    return REJECTED_STALE
                if (
                    output.get("interrupted")
                    or output.get("error")
                    or not output.get("final_response")
                ):
                    return UNAVAILABLE

                def commit(conn):
                    if not db._session_turn_lease_key_on_conn(conn, entry.session_id):
                        raise ValueError("API_LEASE_REQUIRED")
                    lease = conn.execute(
                        "SELECT holder,expires_at FROM session_turn_leases WHERE conversation_id=?",
                        (db._session_turn_lease_key_on_conn(conn, entry.session_id),),
                    ).fetchone()
                    if (
                        not lease
                        or lease["holder"] != holder
                        or lease["expires_at"] <= __import__("time").time()
                    ):
                        raise ValueError("API_STALE_OWNER")
                    current_route = conn.execute(
                        "SELECT entry_json FROM gateway_routing WHERE scope=? AND session_key=?",
                        (routes._routing_scope(), record.session_key),
                    ).fetchone()
                    current_row = conn.execute(
                        "SELECT ended_at FROM sessions WHERE id=?", (entry.session_id,)
                    ).fetchone()
                    if (
                        not current_route
                        or json.loads(current_route[0])["session_id"]
                        != entry.session_id
                        or not current_row
                        or current_row["ended_at"] is not None
                    ):
                        raise ValueError("API_STALE_ORIGIN")
                    return self.domain_store.commit_api_continuation_result(
                        record,
                        response=str(output.get("final_response") or ""),
                        runtime_context=ctx,
                    )

                try:
                    if getattr(db,"backend",None)=="postgresql":
                        with self.domain_store._connect() as conn:
                            hosted.begin_write(conn)
                            db.assert_api_continuation_origin_on_conn(
                                conn,session_id=entry.session_id,holder=holder,
                                routing_scope=routes._routing_scope(),session_key=record.session_key,
                            )
                            result=self.domain_store.commit_api_continuation_result(
                                record,response=str(output.get("final_response") or ""),
                                runtime_context=ctx,connection=conn,
                            )
                    else:
                        result = db._execute_write(commit)
                except PermissionError:
                    return REJECTED_AUTHORITY
                except (ValueError, StaleWorkflowVersion):
                    return REJECTED_STALE
                if result is None:
                    return REJECTED_STALE
            finally:
                db.release_session_turn_lease(entry.session_id, holder)
        # Current authority is checked again by the domain owner before returning
        # stored outcomes; projection is not a bypass around revoked membership.
        try:
            authorize_meeting_requester_mutation(
                self.domain_store.get_negotiation(record.workflow_id), ctx.user_id, ctx
            )
        except (PermissionError, ValueError):
            return REJECTED_AUTHORITY
        try:
            db.append_internal_result(
                entry.session_id,
                result["response"],
                result_id=record.continuation_id + ":result",
                result_hash=result["response_hash"],
            )
        except Exception:
            self.detail = "committed_result_projection_pending"
            return UNAVAILABLE
        return ALREADY_ACCEPTED if was_completed else ACCEPTED


async def recover_authenticated_api_session(ctx, session_id):
    """Scope cold recovery to existing admitted Meeting work for this authenticated conversation."""
    import sqlite3
    import json
    from agents.gateway_identity import get_user_record, get_user_organization_context

    path = ctx.hermes_home / "state.db"
    from storage import hosted_postgres as hosted
    if hosted.enabled():
        from agents.internal_continuation import ContinuationAdmissionStore
        store=ContinuationAdmissionStore(path)
        with store._connect() as conn:
            rows=conn.execute(
                "SELECT continuation_id,payload_json,state FROM semantier_internal_continuations WHERE session_key=%s",
                (session_id,),
            ).fetchall()
    else:
        if not path.is_file():
            return []
        with sqlite3.connect(path) as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='semantier_internal_continuations'"
            ).fetchone():
                return []
            rows = conn.execute(
                "SELECT continuation_id,payload_json,state FROM semantier_internal_continuations WHERE session_key=?",
                (session_id,),
            ).fetchall()
    api_rows = [r for r in rows if "api_actor_origin_hash" in json.loads(r[1])]
    if not api_rows:
        return []
    user = get_user_record(ctx.user_id)
    org = get_user_organization_context(ctx.user_id) or {}
    if (
        not ctx.authenticated
        or not user
        or user.get("workspace_id") != ctx.workspace_id
        or org.get("organization_id") != ctx.organization_id
    ):
        raise PermissionError("API_CONTINUATION_CURRENT_AUTHORITY_REQUIRED")
    if not any(r[2] == "PENDING" for r in api_rows):
        return []
    service = get_embedded_hermes_api_service()
    await service.ensure_started()
    return await service.recover_meeting_continuations(path, session_key=session_id)
