"""One-turn host routes precede default resolution and do not cross sessions."""
from types import SimpleNamespace
from gateway.session_host_runtime import bind_host_runtime, host_runtime_for, host_boundary_for
from gateway.run_turn_prepare import GatewayTurnPrepareMixin


def test_host_runtime_is_exact_and_transient():
    runner, source, other = object(), object(), object()
    assert host_runtime_for(runner, source, 'chat') is None
    runtime = dict(provider='custom', api_key='fixture-only')
    boundary = object()
    with bind_host_runtime(runner, source, 'chat', 'governed', runtime, boundary=boundary):
        assert host_boundary_for(runner) is boundary
        assert host_boundary_for(other) is None
        model, selected = host_runtime_for(runner, source, 'chat')
        assert model == 'governed'
        selected['api_key'] = 'changed'
        assert host_runtime_for(runner, source, 'chat')[1] == runtime
        assert host_runtime_for(runner, other, 'chat') is None
        assert host_runtime_for(other, source, 'chat') is None
        assert host_runtime_for(runner, source, 'other') is None
    assert host_runtime_for(runner, source, 'chat') is None
    assert host_boundary_for(runner) is None


def test_host_runtime_precedes_local_policy_and_default_provider():
    runner, source = SimpleNamespace(), object()
    with bind_host_runtime(runner, source, 'chat', 'governed', {'provider':'custom'}):
        assert GatewayTurnPrepareMixin._resolve_session_agent_runtime(
            runner, source=source, session_key='chat') == ('governed', {'provider':'custom'})
    assert runner._pre_agent_fallback_notice is None


def test_host_cloud_recognition_requires_exact_transient_route():
    from gateway.session_host_runtime import host_cloud_route
    runner, source = object(), object()
    assert not host_cloud_route('http://fixture/v1', 'secret-fixture')
    with bind_host_runtime(runner, source, 'chat', 'model',
                           {'base_url':'http://fixture/v1','api_key':'secret-fixture'}, cloud_bound=True):
        assert host_cloud_route('http://fixture/v1/', 'secret-fixture')
        assert not host_cloud_route('http://foreign/v1', 'secret-fixture')
        assert not host_cloud_route('http://fixture/v1', 'wrong')
    assert not host_cloud_route('http://fixture/v1', 'secret-fixture')


def test_host_cloud_capture_retains_admitted_revision_and_original_lineage():
    from agent.semantier_gateway import capture_agent_route, bind_task_context
    runner, source = object(), object()
    parent = {'task_id': 'admission', 'root_run_id': 'workspace:original',
              'business_object_ref': 'winman:bot:workspace:original'}
    agent = SimpleNamespace(base_url='http://fixture/v1', api_key='fixture-key')
    with bind_host_runtime(runner, source, 'chat', 'model',
            {'base_url':agent.base_url, 'api_key':agent.api_key}, cloud_bound=True,
            cloud_revision='admitted-revision', task_context=parent):
        capture_agent_route(agent)
    assert agent._semantier_cloud_revision == 'admitted-revision'
    assert agent._semantier_parent_task_context == parent
    parent['task_id'] = 'changed'
    bind_task_context(agent, 'new-turn', 'run')
    assert agent._semantier_task_context == {'task_id':'admission', 'root_run_id':'workspace:original',
        'run_id':'run', 'business_object_ref':'winman:bot:workspace:original'}


def test_host_inventory_scope_is_limited_to_background_executor_probes():
    import asyncio
    from gateway.run import _resolve_handoff_watch_scopes
    from gateway.run_idle_gates import off_loop_gate
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    boundary, source = object(), object()
    runner._host_inventory_scope = lambda: bind_host_runtime(
        runner, source, 'inventory', '', {}, boundary=boundary)
    async def offload(probe, *args):
        assert host_boundary_for(runner) is boundary
        return [(None, None)] if args else True
    runner._run_in_executor_with_context = offload
    async def run():
        assert await _resolve_handoff_watch_scopes(runner) == [(None, None)]
        assert host_boundary_for(runner) is None
        assert await off_loop_gate(runner, lambda: True)
        assert host_boundary_for(runner) is None
    asyncio.run(run())
