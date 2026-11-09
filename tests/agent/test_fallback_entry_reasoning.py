"""Entry reasoning belongs to the active fallback, never to a global/provider override map."""
import pytest

from hermes_constants import resolve_reasoning_config


@pytest.mark.parametrize("entry,expected", [
    ({"reasoning_effort": "none"}, {"enabled": False}),
    ({"reasoning_effort": "high"}, {"enabled": True, "effort": "high"}),
    ({}, {"enabled": True, "effort": "low"}),
])
def test_entry_over_model_over_global(entry, expected):
    config = {"agent": {"reasoning_effort": "medium", "reasoning_overrides": {"gpt-5-mini": "low"}}}
    assert resolve_reasoning_config(config, "gpt-5-mini", fallback_entry=entry) == expected
    assert resolve_reasoning_config(config, "primary") == {"enabled": True, "effort": "medium"}


@pytest.mark.parametrize('effort,wire', [('high', 'high'), ('none', 'none'), (None, 'low')])
@pytest.mark.parametrize('primary_effort', ['medium', None])
def test_real_failing_primary_and_fallback_wire(tmp_path, monkeypatch, effort, wire, primary_effort):
    """Only the two external provider transports are synthetic; the agent/failover are real."""
    import contextlib
    import json
    from tests.hermes_cli.test_provider_model_journey import provider_fixture
    from run_agent import AIAgent

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({
        'agent': {'reasoning_effort': primary_effort, 'reasoning_overrides': {'gpt-5-mini': 'low'}},
        'model': {'streaming': False}, 'toolsets': [], 'compression': {'enabled': False},
    }))
    fixture = contextlib.contextmanager(provider_fixture.__wrapped__)
    with fixture() as (primary_url, primary), fixture() as (backup_url, backup):
        primary['inference_status'] = 429
        entry = {'provider': 'custom', 'model': 'gpt-5-mini', 'base_url': backup_url + '/v1', 'api_key': 'fixture-key'}
        if effort is not None:
            entry['reasoning_effort'] = effort
        primary_reasoning = {'enabled': True, 'effort': primary_effort} if primary_effort else None
        agent = AIAgent(model='gpt-5', provider='custom', base_url=primary_url + '/v1', api_key='fixture-key',
                        reasoning_config=primary_reasoning, fallback_model=[entry],
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
                        max_iterations=3, save_trajectories=False)
        result = agent.run_conversation('Reply OK with no tools.')
        posts = [body for path, body in backup['calls'] if path.endswith('/chat/completions')]
        assert posts, result
        assert posts[-1].get('reasoning_effort') == wire
        primary_posts = [body for path, body in primary['calls'] if path.endswith('/chat/completions')]
        primary_wire = primary_posts[-1].get('reasoning_effort')
        assert agent._primary_runtime['reasoning_config'] == primary_reasoning
        from agent.agent_runtime_helpers import restore_primary_runtime
        agent._rate_limited_until = 0  # controlled expiry of the existing primary cooldown
        assert restore_primary_runtime(agent)
        assert agent.reasoning_config == primary_reasoning
        primary['inference_status'] = 200
        agent.run_conversation('Reply OK again with no tools.')
        posts = [body for path, body in primary['calls'] if path.endswith('/chat/completions')]
        assert posts[-1].get('reasoning_effort') == primary_wire


def test_initial_credential_fallback_applies_selected_entry_before_snapshot(tmp_path, monkeypatch):
    import contextlib
    import json
    from tests.hermes_cli.test_provider_model_journey import provider_fixture
    from run_agent import AIAgent
    import agent.auxiliary_client as auxiliary
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'model': {'streaming': False}, 'agent': {'reasoning_effort': 'high'}, 'toolsets': []}))
    original = auxiliary.resolve_provider_client
    def credentials(provider, *args, **kwargs):
        if provider in {'openrouter', 'custom:unavailable'}:
            return None, None  # controlled missing credentials, not a replacement route binder
        return original(provider, *args, **kwargs)
    monkeypatch.setattr(auxiliary, 'resolve_provider_client', credentials)
    with contextlib.contextmanager(provider_fixture.__wrapped__)() as (url, state):
        chain = [{'provider': 'custom:unavailable', 'model': 'missing', 'reasoning_effort': 'high'},
                 {'provider': 'custom', 'model': 'gpt-5-mini', 'base_url': url + '/v1', 'api_key': 'fixture-key', 'reasoning_effort': 'none'}]
        agent = AIAgent(model='primary', provider='openrouter', fallback_model=chain,
                        reasoning_config={'enabled': True, 'effort': 'medium'}, quiet_mode=True,
                        skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=2,
                        save_trajectories=False)
        assert agent.reasoning_config == {'enabled': False}
        assert agent._primary_runtime['reasoning_config'] == {'enabled': False}
        agent.run_conversation('Reply OK.')
        posts = [body for path, body in state['calls'] if path.endswith('/chat/completions')]
        assert posts[-1].get('reasoning_effort') == 'none'


@pytest.mark.parametrize("effort,expected", [("none", {"enabled": False}), ("high", {"enabled": True, "effort": "high"})])
def test_gateway_credential_resolution_carries_entry_reasoning(monkeypatch, effort, expected):
    from tui_gateway import server as agent_factory
    from hermes_cli.auth import AuthError
    import hermes_cli.runtime_provider as providers
    def resolve(**kwargs):
        if kwargs.get("requested") == "primary":
            raise AuthError("synthetic missing credentials")
        return {"provider": "custom", "api_key": "fixture-key", "base_url": "http://fixture/v1"}
    monkeypatch.setattr(providers, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(agent_factory, "_load_fallback_model", lambda: [{"provider": "custom", "model": "same", "reasoning_effort": effort}])
    resolution = agent_factory._resolve_runtime_with_fallback({"requested": "primary"})
    assert resolution.runtime["_fallback_reasoning_config"] == expected
