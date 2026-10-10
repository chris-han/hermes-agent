"""Server-owned tool policy closes registry, catalog and both inline doors."""
import json
from types import SimpleNamespace
import pytest
from gateway.execution_boundary import BoundaryPolicy, ExecutionBoundary, bind_execution_boundary


@pytest.mark.parametrize('name',['terminal','execute_code','delegate_task','memory','skill_view','tool_call','connectors__test__write'])
def test_profile_denies_all_execution_doors_even_with_skip_flags(name):
    from model_tools import handle_function_call
    from tools.registry import registry
    from agent.agent_runtime_helpers import invoke_tool
    from agent.tool_executor import _resolve_sequential_dispatch
    b=ExecutionBoundary(source='test-server-owned-profile',policy=BoundaryPolicy(allowed_tool_names=('read_file',)))
    with bind_execution_boundary(b):
        for result in [handle_function_call(name,{},skip_pre_tool_call_hook=True,skip_tool_request_middleware=True,skip_tool_execution_middleware=True),registry.dispatch(name,{}),invoke_tool(SimpleNamespace(),name,{},'one',pre_tool_block_checked=True,skip_tool_execution_middleware=True)]:
            assert json.loads(result)['error_type']=='execution_profile'
        ref=SimpleNamespace(name=name,args={},task_id='one',call_id='id',trace=[])
        dispatch=_resolve_sequential_dispatch(SimpleNamespace(),ref,[])
        assert json.loads(dispatch.execute({}))['error_type']=='execution_profile'


def test_context_isolation_and_default_compatibility(tmp_path):
    from tools.registry import ToolRegistry
    from tools.thread_context import propagate_context_to_thread
    from concurrent.futures import ThreadPoolExecutor
    r=ToolRegistry()
    r.register(name='probe',toolset='test',schema={'name':'probe','parameters':{'type':'object'}},handler=lambda args,**kw:json.dumps({'ran':True}))
    restricted=ExecutionBoundary(source='A',policy=BoundaryPolicy(allowed_tool_names=()))
    ordinary=ExecutionBoundary(source='B')
    with ThreadPoolExecutor(max_workers=1) as pool:
        for boundary,denied in [(restricted,True),(ordinary,False),(restricted,True)]:
            with bind_execution_boundary(boundary):dispatch=propagate_context_to_thread(lambda:r.dispatch('probe',{}))
            assert ('error' in json.loads(pool.submit(dispatch).result()))==denied
    assert json.loads(r.dispatch('probe',{}))['ran'] is True


def test_existing_positional_scratch_roots_preserve_ordinary_policy(tmp_path):
    policy = BoundaryPolicy(True, False, (), (), (tmp_path,))
    assert policy.scratch_roots == (tmp_path,)
    assert policy.allowed_tool_names is None
