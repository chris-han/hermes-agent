"""Auto-selected provider defaults must not erase the existing selector contract."""
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('kind',['stt','tts'])
def test_dynamic_provider_selector_survives_absent_default(monkeypatch,kind):
    import hermes_cli.web_server_config as config
    import hermes_cli.web_server_profiles as profiles
    import hermes_cli.config as loader
    import agent.transcription_registry as stt
    import agent.tts_registry as tts
    schema={k:v for k,v in config.CONFIG_SCHEMA.items() if k!=f'{kind}.provider'}
    monkeypatch.setattr(config,'CONFIG_SCHEMA',schema)
    monkeypatch.setattr(loader,'load_config',lambda:{kind:{'provider':'local'}})
    monkeypatch.setattr(profiles,'_plugin_terminal_backend_rows',lambda:[])
    monkeypatch.setattr(stt if kind=='stt' else tts,'list_providers',lambda:[SimpleNamespace(name='optional_voice')])
    result=config._schema_with_dynamic_provider_options()
    assert result[f'{kind}.provider']['type']=='select'
    assert 'optional_voice' in result[f'{kind}.provider']['options']
    assert 'local' in result[f'{kind}.provider']['options']
    assert f'{kind}.provider' not in schema
