"""Core adapter for authenticated wiki-task selection and registered typed tools."""
import json
from pathlib import Path

from runtime_paths import workspace_session_segment
from services.wiki_mutation_service import WikiMutationService, current_entitlement
from storage.wiki_mutation_store import WikiMutationStore


_PROMPTS = Path(__file__).resolve().parents[1]/'prompts/agents'


def bind_wiki_task(ctx, session_id, skill):
    if skill not in ('llm-wiki','research/llm-wiki'): return False
    if not ctx.authenticated: raise PermissionError('authenticated wiki task required')
    current_entitlement(ctx.user_id,ctx.workspace_id)
    WikiMutationStore().bind(ctx.workspace_id,workspace_session_segment(ctx.workspace_id,session_id),ctx.user_id)
    return True


def prepare_wiki_task(ctx, session_id, skill, message):
    try:
        selected = bind_wiki_task(ctx,session_id,skill)
    except PermissionError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if selected:
        return message+'\n\n'+(_PROMPTS/'wiki_task_profile.md').read_text()
    return message


def wiki_profile_actor(workspace, session):
    return WikiMutationStore().binding(workspace,workspace_session_segment(workspace,session))


def _dispatch(name, args, **kwargs):
    from gateway.execution_boundary import current_execution_boundary
    service=WikiMutationService(current_execution_boundary())
    if name=='wiki_read': result=service.read(args.get('path'))
    elif name=='wiki_mutate': result=service.mutate(args,tool_call_id=kwargs.get('tool_call_id'))
    else: result=service.reconcile(args.get('operation_id'))
    return json.dumps(result,ensure_ascii=False)


def register_wiki_tools():
    from tools.registry import registry
    for schema in json.loads((_PROMPTS/'wiki_task_tools.json').read_text()):
        name=schema['name']
        registry.register(name=name,toolset='file',schema=schema,handler=lambda args,_name=name,**kw:_dispatch(_name,args,**kw))
