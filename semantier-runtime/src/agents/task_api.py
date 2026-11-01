"""Authenticated core HTTP composition: independent tasks and explicit fresh Chat."""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import json
import os
from agents.cloud_mode import require_managed_runtime

require_managed_runtime()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from agents.auth_session import request_context_from_request, auth_context_payload
from agents.task_service import TaskService, baseline
from agents import job_store
from agents.webapi.password_routes import router as passwords


def context(request):
    ctx = request_context_from_request(request)
    return validate_task_context(ctx)


def validate_task_context(ctx, *, hosted=False):
    if not ctx.authenticated:
        raise HTTPException(
            503 if ctx.auth_invalid_reason == "context_unavailable" else 401,
            "Authentication required",
        )
    if ctx.membership_status != "active" or not ctx.organization_id:
        raise HTTPException(403, "Active workspace membership required")
    from services.learning_trace_evidence_service import LearningTraceEvidenceService

    try:
        LearningTraceEvidenceService.public_scope(ctx)
    except PermissionError as exc:
        raise HTTPException(403, "Current workspace authority required") from exc
    from agents.auth_db_postgres import connection

    with connection() as c:
        grant = c.execute(
            "SELECT permissions FROM semantier_gateway.grants WHERE organization_id=%s AND user_id=%s",
            (ctx.organization_id, ctx.user_id),
        ).fetchone()
        access = c.execute(
            "SELECT 1 FROM semantier_gateway.account_access WHERE organization_id=%s AND user_id=%s AND status='ACTIVE'",
            (ctx.organization_id, ctx.user_id),
        ).fetchone()
        if hosted:
            from agents.workspace_storage_provisioning import registered_workspace_scope
            root=c.execute("SELECT workspace_root FROM semantier_identity.workspace_provisioning_contract WHERE singleton=1").fetchone()
            allocated=c.execute("SELECT b.user_id,b.workspace_id,b.tenant_id,b.source_store_id,b.schema_name,b.hermes_home,u.payload_json FROM semantier_identity.workspace_runtime_bindings b JOIN semantier_identity.users u USING(user_id) WHERE b.user_id=%s AND b.workspace_id=%s",(ctx.user_id,ctx.workspace_id)).fetchone()
            try:
                if not root or not allocated:raise ValueError('Missing allocation')
                registered_workspace_scope(tuple(allocated.values()),root['workspace_root'])
                binding={'workspace_root':allocated['hermes_home']}
            except ValueError as exc:raise HTTPException(403,'Current governed workspace allocation required') from exc
        else:
            binding = c.execute(
                "SELECT workspace_root FROM semantier_identity.workspace_runtime_bindings WHERE user_id=%s AND workspace_id=%s",
                (ctx.user_id, ctx.workspace_id),
            ).fetchone()
    if (
        not grant
        or "model:invoke" not in grant["permissions"]
        or not access
        or not binding
        or binding["workspace_root"] != str(ctx.workspace_root.resolve())
    ):
        raise HTTPException(403, "Current task execution access required")
    return ctx


def create_app(*, core_delivery_transport_factory=None, core_channel_transport_factory=None):
    service = TaskService()
    from services.core_work_execution_service import ManagedWorkExecutionService
    managed_execution = ManagedWorkExecutionService()
    from services.core_scheduled_work_service import CoreScheduledWorkService
    scheduled_work = CoreScheduledWorkService(managed_execution)
    from services.core_feishu_conversation_service import CoreFeishuConversationService
    channel_chat = CoreFeishuConversationService(managed_execution,transport_factory=core_channel_transport_factory)
    from agents.webapi.sdk_plugin_routes import WorkspacePlugins, plugin_router
    plugins = WorkspacePlugins()

    @asynccontextmanager
    async def lifespan(app):
        from agents.task_bootstrap import configure
        from agents.hermes_upstream import UpstreamInstallation

        execution_root = configure()
        from services.meeting_clarification_delivery_service import recover_configured_meeting_delivery
        recover_configured_meeting_delivery(execution_root)
        UpstreamInstallation.from_environment().verify()
        job_store.interrupt_independent_tasks()
        from services.meeting_followup_timer_service import configured_service
        from services.core_one_shot_timer_service import bounded_poll
        timer_owner = configured_service(execution_root)
        owned_timer_tasks = []
        if timer_owner is not None:
            def scan_timers():
                timer_owner.recover()
                return timer_owner.driver.scan()
            owned_timer_tasks = [
                asyncio.create_task(bounded_poll(scan_timers), name="core-meeting-timer-signals"),
                asyncio.create_task(bounded_poll(timer_owner.dispatch_pending), name="core-meeting-continuations"),
            ]
        from services.meeting_clarification_delivery_service import MeetingClarificationDeliveryService
        from feishu_meeting_coordinator.store import MeetingCoordinatorStore
        delivery_owner = MeetingClarificationDeliveryService(MeetingCoordinatorStore(execution_root / "state.db"))
        from services.core_meeting_continuation_service import CoreMeetingContinuationService
        general_delivery_owner = CoreMeetingContinuationService(delivery_owner.store)
        def dispatch_delivery_retries():
            delivery_owner.recover_expired_retries()
            retries = delivery_owner.retry_due(transport_factory=core_delivery_transport_factory)
            general_delivery_owner.recover_pending()
            general_delivery_owner.dispatch_pending(transport_factory=core_delivery_transport_factory)
            return retries
        owned_timer_tasks.append(asyncio.create_task(bounded_poll(dispatch_delivery_retries), name="core-meeting-delivery-retries"))
        def scheduled_signals():
            scheduled_work.recover()
            return scheduled_work.driver.scan()
        owned_timer_tasks.extend([
            asyncio.create_task(bounded_poll(scheduled_signals),name='core-scheduled-work-signals'),
            asyncio.create_task(bounded_poll(scheduled_work.dispatch_pending),name='core-scheduled-work-execution'),
        ])
        owned_timer_tasks.append(asyncio.create_task(bounded_poll(channel_chat.tick),name='core-feishu-channel-conversations'))
        app.state.core_timer_tasks = owned_timer_tasks
        try:
            yield
        finally:
            for task in owned_timer_tasks:
                task.cancel()
            await asyncio.gather(*owned_timer_tasks, return_exceptions=True)
            await asyncio.to_thread(managed_execution.shutdown)
            await asyncio.to_thread(service.shutdown)
            await asyncio.to_thread(plugins.close)

    app = FastAPI(title="Semantier Runtime", lifespan=lifespan)
    from fastapi import Depends
    from agents.webapi.operations_admin_routes import reserve_operations_login
    app.include_router(passwords, dependencies=[Depends(reserve_operations_login)])
    from agents.webapi.operations_admin_routes import router as operations_admin_router
    app.include_router(operations_admin_router)
    from agents.webapi.journey_routes import router as journey_router
    app.include_router(journey_router)
    from agents.webapi.core_feishu_channel_routes import router as channel_router
    app.include_router(channel_router(context=context,service=channel_chat))
    app.state.core_channel_chat=channel_chat
    from agents.webapi.core_conversation_history_routes import router as history_router
    app.include_router(history_router(context=context))
    from agents.webapi.core_provider_routes import router as provider_router
    app.include_router(provider_router(context=context))
    from agents.webapi.core_conversation_routes import router as conversation_router
    app.include_router(conversation_router(context=context,execution=managed_execution))
    from feishu_meeting_coordinator.webapi import task_runtime_router
    app.include_router(task_runtime_router)
    app.include_router(plugin_router(context=context, service=plugins), prefix="/api")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            x.strip()
            for x in os.environ.get(
                "SEMANTIER_CORS_ALLOWED_ORIGINS",
                "http://localhost:3300,http://127.0.0.1:3300",
            ).split(",")
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type"],
    )
    from storage.hosted_postgres import HostedStorageUnavailable
    import psycopg

    async def unavailable(request, exc):
        return JSONResponse(
            status_code=503, content={"detail": "Authoritative PostgreSQL unavailable"}
        )

    app.add_exception_handler(HostedStorageUnavailable, unavailable)
    app.add_exception_handler(psycopg.OperationalError, unavailable)

    # Managed work is distinct from the immediately executed /api/tasks path.
    from services.core_work_service import CoreWorkService
    app.state.core_work = None

    def work_service():
        if app.state.core_work is None:
            app.state.core_work = CoreWorkService()
        return app.state.core_work

    def meeting_board_service():
        from feishu_meeting_coordinator.store import MeetingCoordinatorStore
        from services.core_meeting_board_service import CoreMeetingBoardService
        return CoreMeetingBoardService(MeetingCoordinatorStore(job_store._db_path()))

    @app.get("/api/meeting-work")
    async def list_meeting_work(request: Request, limit: int = 25, cursor: str = ''):
        ctx = await asyncio.to_thread(context, request)
        try:
            return await asyncio.to_thread(meeting_board_service().list, ctx, limit=limit, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/meeting-work/{case_id}")
    async def get_meeting_work(case_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        try:
            return await asyncio.to_thread(meeting_board_service().get, ctx, case_id)
        except KeyError as exc:
            raise HTTPException(404, 'Meeting work not found') from exc

    async def append_meeting_annotation(case_id: str, request: Request, kind: str):
        ctx = await asyncio.to_thread(context, request)
        try:
            service=meeting_board_service()
            return await asyncio.to_thread(getattr(service,kind), ctx, case_id, await request.json())
        except KeyError as exc:
            raise HTTPException(404, 'Meeting work or reference not found') from exc
        except ValueError as exc:
            raise HTTPException(409 if 'conflict' in str(exc) else 400, str(exc)) from exc

    @app.post("/api/meeting-work/{case_id}/comments")
    async def comment_meeting_work(case_id: str, request: Request):
        return await append_meeting_annotation(case_id, request, 'comment')

    @app.post("/api/meeting-work/{case_id}/references")
    async def reference_meeting_work(case_id: str, request: Request):
        return await append_meeting_annotation(case_id, request, 'reference')

    @app.post("/api/meeting-work/{case_id}/decision")
    async def decide_meeting_work(case_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        body = await request.json()
        if (not isinstance(body, dict) or set(body) - {'action', 'slot_id'}
            or not isinstance(body.get('action'), str)
            or ('slot_id' in body and not isinstance(body['slot_id'], str))):
            raise HTTPException(400, 'Invalid meeting decision')
        from services.core_meeting_continuation_service import CoreMeetingContinuationService
        try:
            service = meeting_board_service()
            if body.get('action') == 'requester_cancel':
                from services.core_meeting_continuation_service import CONTRACT
                from services.meeting_clarification_delivery_service import MeetingClarificationDeliveryService
                owner = MeetingClarificationDeliveryService(service.store)
                if (await asyncio.to_thread(owner.binding, case_id))['contract'] == CONTRACT:
                    owner = CoreMeetingContinuationService(service.store)
                await asyncio.to_thread(owner.cancel, ctx, case_id)
            else:
                await asyncio.to_thread(CoreMeetingContinuationService(service.store).requester_decision,
                    ctx, case_id, body.get('action'), slot_id=body.get('slot_id'))
            return await asyncio.to_thread(service.get, ctx, case_id)
        except KeyError as exc:
            raise HTTPException(404, 'Meeting work not found') from exc
        except (PermissionError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    async def scheduled_call(request, operation, *args):
        from agents.durable_responsibility import DurableResponsibilityError
        ctx=await asyncio.to_thread(context,request)
        try:
            return await asyncio.to_thread(operation,ctx,*args)
        except KeyError as exc:raise HTTPException(404,'Scheduled work not found') from exc
        except DurableResponsibilityError as exc:raise HTTPException(409,str(exc)) from exc
        except ValueError as exc:raise HTTPException(409 if 'conflict' in str(exc) else 400,str(exc)) from exc
        except PermissionError as exc:raise HTTPException(403,'Current scheduled-work authority required') from exc

    @app.post('/api/scheduled-work')
    async def create_schedule(request: Request):
        row,created=await scheduled_call(request,scheduled_work.create,await request.json())
        return JSONResponse(status_code=201 if created else 200,content=row)

    @app.get('/api/scheduled-work')
    async def list_schedules(request: Request):
        return await scheduled_call(request,scheduled_work.list)

    @app.get('/api/scheduled-work/{work_id}')
    async def get_schedule(work_id: str,request: Request):
        return await scheduled_call(request,scheduled_work.get,work_id)

    @app.post('/api/scheduled-work/{work_id}/actions')
    async def schedule_action(work_id: str,request: Request):
        return await scheduled_call(request,scheduled_work.action,work_id,await request.json())

    @app.post("/api/work-items")
    async def create_work(request: Request):
        ctx = await asyncio.to_thread(context, request)
        try:
            payload = await request.json()
            row, created = await asyncio.to_thread(work_service().create, ctx, payload)
        except ValueError as exc:
            raise HTTPException(409 if 'conflict' in str(exc) else 400, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(404, 'Work item not found') from exc
        return JSONResponse(status_code=201 if created else 200, content=row)

    @app.get("/api/work-items")
    async def list_work(request: Request, limit: int = 25, cursor: str = ''):
        ctx = await asyncio.to_thread(context, request)
        try:
            return await asyncio.to_thread(work_service().list, ctx, limit, cursor)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/work-items/{work_id}")
    async def get_work(work_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        try:
            return await asyncio.to_thread(work_service().get, ctx, work_id)
        except KeyError as exc:
            raise HTTPException(404, 'Work item not found') from exc

    @app.post("/api/work-items/{work_id}/execute")
    async def execute_work(work_id: str, request: Request):
        from agents.durable_responsibility import DurableResponsibilityError
        ctx = await asyncio.to_thread(context, request)
        try:
            if await request.json() != {}:
                raise ValueError('execution body must be empty')
            item = await asyncio.to_thread(managed_execution.submit, ctx, work_id,
                                           resolve_context=lambda: context(request))
            return JSONResponse(status_code=202, content=item)
        except KeyError as exc:
            raise HTTPException(404, 'Work item not found') from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, 'Current task execution access required') from exc
        except DurableResponsibilityError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/work-items/{work_id}/comments")
    async def comment_work(work_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        try:
            return await asyncio.to_thread(work_service().comment, ctx, work_id, await request.json())
        except KeyError as exc:
            raise HTTPException(404, 'Work item not found') from exc
        except ValueError as exc:
            raise HTTPException(409 if 'conflict' in str(exc) else 400, str(exc)) from exc

    @app.post("/api/work-items/{work_id}/references")
    async def reference_work(work_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        try:
            return await asyncio.to_thread(work_service().reference, ctx, work_id, await request.json())
        except KeyError as exc:
            raise HTTPException(404, 'Work item not found') from exc
        except ValueError as exc:
            raise HTTPException(409 if 'conflict' in str(exc) else 400, str(exc)) from exc

    @app.post("/api/work-items/{work_id}/claim")
    async def claim_work(work_id: str, request: Request):
        from agents.durable_responsibility import DurableResponsibilityError
        ctx = await asyncio.to_thread(context, request)
        try:
            payload = await request.json()
            if payload != {}:
                raise ValueError('claim body must be empty')
            return await asyncio.to_thread(work_service().claim, ctx, work_id)
        except KeyError as exc:
            raise HTTPException(404, 'Work item not found') from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except DurableResponsibilityError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/work-items/{work_id}/complete")
    async def complete_work(work_id: str, request: Request):
        from agents.durable_responsibility import DurableResponsibilityError
        ctx = await asyncio.to_thread(context, request)
        try:
            return await asyncio.to_thread(work_service().complete, ctx, work_id, await request.json())
        except KeyError as exc:
            raise HTTPException(404, 'Work item not found') from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except DurableResponsibilityError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/work-items/{work_id}/cancel")
    async def cancel_work(work_id: str, request: Request):
        from agents.durable_responsibility import DurableResponsibilityError
        ctx = await asyncio.to_thread(context, request)
        try:
            if await request.json() != {}:
                raise ValueError('cancel body must be empty')
            item = await asyncio.to_thread(work_service().cancel, ctx, work_id)
            if 'execution' in item:
                stopped = await asyncio.to_thread(managed_execution.stop, ctx, work_id)
                item = await asyncio.to_thread(work_service().get, ctx, work_id)
                if not stopped:
                    return JSONResponse(status_code=503, content={'detail': 'Cancellation recorded; executor stop is unconfirmed. Refresh and inspect the work item.'})
                item['execution_stop'] = 'reaped'
                return item
            return item
        except KeyError as exc:
            raise HTTPException(404, 'Work item not found') from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except DurableResponsibilityError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/auth/context")
    async def auth(request: Request):
        return auth_context_payload(
            await asyncio.to_thread(request_context_from_request, request)
        )

    register_task_routes(app,service)

    return app


def register_task_routes(app, service, context_owner=context):
    """One route owner shared by explicitly configured real core compositions."""
    context=context_owner
    @app.get("/api/task-capabilities")
    async def capabilities(request: Request):
        ctx = await asyncio.to_thread(context, request)
        return {
            "mode": "independent_tasks",
            "conversation_history": False,
            "workspace_id": ctx.workspace_id,
            "baseline_skill": await asyncio.to_thread(baseline, ctx),
        }

    @app.post("/api/tasks")
    async def submit(request: Request):
        ctx = await asyncio.to_thread(context, request)
        body = await request.json()
        if not isinstance(body, dict) or set(body) - {
            "request_id",
            "instruction",
            "skill_pin",
            "monitoring_basis",
            "value_purpose",
        }:
            raise HTTPException(400, "Only independent task input is supported")
        if body.get('value_purpose','unknown') not in {'real','demo','unknown'}:
            raise HTTPException(400,'Unsupported task value purpose')
        try:
            task = await asyncio.to_thread(
                service.submit,
                ctx,
                request_id=body.get("request_id", ""),
                instruction=body.get("instruction"),
                skill_pin=body.get("skill_pin")
                or await asyncio.to_thread(baseline, ctx),
                monitoring_basis=body.get("monitoring_basis"),
                value_purpose=body.get("value_purpose","unknown"),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        return task

    def get(ctx, task_id):
        task = job_store.get_job(
            user_id=ctx.user_id, workspace_id=ctx.workspace_id, job_id=task_id
        )
        if task is None or task.get("kind") != "independent_task":
            raise HTTPException(404, "Task not found")
        return task

    @app.get("/api/tasks/{task_id}")
    async def result(task_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        task=await asyncio.to_thread(get, ctx, task_id)
        return task

    @app.post("/api/tasks/{task_id}/result-viewed")
    async def result_viewed(task_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        task = await asyncio.to_thread(get, ctx, task_id)
        if task.get('state') != 'completed':
            raise HTTPException(409, 'Completed result required')
        from agents.operations_store import capture_fact
        recorded = await asyncio.to_thread(
            capture_fact, name='result_viewed', user_id=ctx.user_id,
            producer='task', receipt_ref=task_id+':'+ctx.user_id,
            surface='workspace', evidence_type='client_observation', properties={'task_id':task_id,'basis':'visible_result_acknowledgement'})
        return {'recorded': recorded}

    @app.post("/api/tasks/{task_id}/feedback")
    async def feedback(task_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        task = await asyncio.to_thread(get, ctx, task_id)
        if task.get('state') != 'completed':
            raise HTTPException(409, 'Completed result required')
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'request_id','kind','value'}:
            raise HTTPException(422, 'Bounded feedback fields required')
        import uuid
        try:
            request_id=str(uuid.UUID(body['request_id']))
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(422, 'Feedback request identity required')
        choices={'feedback':{'useful','needs_revision','not_useful'},'action':{'used_result','continued_work','no_action'}}
        kind=body['kind'];value=body['value']
        if not isinstance(kind,str) or not isinstance(value,str) or value not in choices.get(kind,set()):
            raise HTTPException(422, 'Supported self-reported feedback required')
        from agents.operations_store import capture_fact
        recorded=await asyncio.to_thread(capture_fact,
            name='feedback_submitted' if kind=='feedback' else 'action_recorded',
            user_id=ctx.user_id,producer='task',receipt_ref=task_id+':'+ctx.user_id+':'+request_id,
            surface='workspace',evidence_type='self_reported',
            properties={'task_id':task_id,'kind':kind,'value':value})
        return {'recorded':recorded}

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel(task_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        await asyncio.to_thread(get, ctx, task_id)
        return {"cancellation_requested": service.cancel(ctx, task_id)}

    @app.get("/api/tasks/{task_id}/events")
    async def events(task_id: str, request: Request):
        ctx = await asyncio.to_thread(context, request)
        await asyncio.to_thread(get, ctx, task_id)

        async def stream():
            cursor = 0
            while True:
                if await request.is_disconnected():
                    return
                # Recheck live core authority during an open stream.
                ctx = await asyncio.to_thread(context, request)
                task = await asyncio.to_thread(get, ctx, task_id)
                for row in task.get("events", [])[cursor:]:
                    yield "event: progress\ndata: " + json.dumps(row) + "\n\n"
                    cursor += 1
                if task["state"] not in {"queued", "running"}:
                    yield "event: result\ndata: " + json.dumps(task) + "\n\n"
                    return
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )



def hosted_task_context(request):
    return validate_task_context(request_context_from_request(request),hosted=True)
