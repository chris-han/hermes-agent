import builtins
import json
from pathlib import Path
import pytest


def test_local_bootstrap_has_no_saas_import_and_preserves_local_state(monkeypatch,tmp_path):
    from hermes_cli.standalone_bootstrap import bootstrap_standalone_runtime
    bundle=tmp_path/'bundle';source=bundle/'semantier-runtime/src/plugins/sensevoice_stt';source.mkdir(parents=True)
    for name,data in {'provider.py':'# fixture provider\n','__init__.py':'# fixture registration\n','plugin.yaml':'name: sensevoice_stt\n'}.items():(source/name).write_text(data)
    owner=tmp_path/'local';owner.mkdir();(owner/'config.yaml').write_text('model: saved-provider\n')
    (owner/'memories').mkdir();(owner/'memories/MEMORY.md').write_text('retained memory')
    monkeypatch.setenv('HERMES_HOME',str(owner))
    import pm.features,pm.paths
    monkeypatch.setattr(pm.features,'runtime_distribution_extras',lambda:['sensevoice'])
    monkeypatch.setattr(pm.paths,'repo_root',lambda:bundle)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='agents' or name.startswith('agents.'):raise AssertionError('standalone imported SaaS')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    bootstrap_standalone_runtime();bootstrap_standalone_runtime()
    assert (owner/'config.yaml').read_text()=='model: saved-provider\n'
    assert (owner/'memories/MEMORY.md').read_text()=='retained memory'
    assert (owner/'plugins/sensevoice_stt/provider.py').read_text()=='# fixture provider\n'
    assert not (owner/'profiles/tender-runtime-expert').exists()
    assert not (owner/'.semantier-home').exists()
    assert not (owner/'eos.db').exists()


def test_custom_plugin_is_not_overwritten(monkeypatch,tmp_path):
    from hermes_cli.standalone_bootstrap import bootstrap_standalone_runtime
    home=tmp_path/'local';target=home/'plugins/sensevoice_stt';target.mkdir(parents=True)
    (target/'provider.py').write_text('user-owned plugin')
    bundle=tmp_path/'bundle';source=bundle/'semantier-runtime/src/plugins/sensevoice_stt';source.mkdir(parents=True)
    for name in ['provider.py','__init__.py','plugin.yaml']:(source/name).write_text('new package')
    monkeypatch.setenv('HERMES_HOME',str(home));import pm.features,pm.paths
    monkeypatch.setattr(pm.features,'runtime_distribution_extras',lambda:['sensevoice']);monkeypatch.setattr(pm.paths,'repo_root',lambda:bundle)
    bootstrap_standalone_runtime()
    assert (target/'provider.py').read_text()=='user-owned plugin'
