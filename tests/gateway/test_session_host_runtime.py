"""One-turn host routes precede default resolution and do not cross sessions."""
from types import SimpleNamespace
from gateway.session_host_runtime import bind_host_runtime, host_runtime_for
from gateway.run_turn_prepare import GatewayTurnPrepareMixin


def test_host_runtime_is_exact_and_transient():
    runner, source, other = object(), object(), object()
    assert host_runtime_for(runner, source, 'chat') is None
    runtime = dict(provider='custom', api_key='fixture-only')
    with bind_host_runtime(runner, source, 'chat', 'governed', runtime):
        model, selected = host_runtime_for(runner, source, 'chat')
        assert model == 'governed'
        selected['api_key'] = 'changed'
        assert host_runtime_for(runner, source, 'chat')[1] == runtime
        assert host_runtime_for(runner, other, 'chat') is None
        assert host_runtime_for(other, source, 'chat') is None
        assert host_runtime_for(runner, source, 'other') is None
    assert host_runtime_for(runner, source, 'chat') is None


def test_host_runtime_precedes_local_policy_and_default_provider():
    runner, source = SimpleNamespace(), object()
    with bind_host_runtime(runner, source, 'chat', 'governed', {'provider':'custom'}):
        assert GatewayTurnPrepareMixin._resolve_session_agent_runtime(
            runner, source=source, session_key='chat') == ('governed', {'provider':'custom'})
    assert runner._pre_agent_fallback_notice is None
