"""Windows sealed distribution omits local LLM engines; source capability remains."""
import json
import pytest
from scripts.bundles import native
from hermes_cli.local_runtime import binaries,bootstrap,endpoint

@pytest.mark.parametrize('target',['win32-x64'])
def test_windows_selection_excludes_all_llamacpp_variants(monkeypatch,target):
 monkeypatch.setattr(native,'current_target',lambda:target)
 names=native._bundle_package_names()
 assert not any(n.startswith('llamacpp-') for n in names)
 assert 'python' in names and 'ffmpeg' in names

@pytest.mark.parametrize('target',['linux-x64','darwin-arm64','win32-arm64'])
def test_other_distribution_package_definitions_remain(monkeypatch,target):
 monkeypatch.setattr(native,'current_target',lambda:target)
 assert any(n.startswith('llamacpp-') for n in native._bundle_package_names())

def test_stale_facts_pruned_from_owned_stage(monkeypatch,tmp_path):
 from pm.lock import Facts
 store=tmp_path/'tools';store.mkdir();entry=store/'llamacpp-cuda-stale';entry.mkdir();(entry/'llama-server.exe').write_bytes(b'old')
 (store/'facts.json').write_text(json.dumps({'schema':1,'packages':{'llamacpp-cuda':{'entry':entry.name,'version':'0'}}}))
 native.prune_staged_store(store,[])
 assert not entry.exists() and Facts(store/'facts.json').get('llamacpp-cuda') is None

def test_bundled_windows_never_adopts_downloads_starts_or_discovers(monkeypatch):
 from hermes_cli import steward
 monkeypatch.setattr('pm.store.current_target',lambda:'win32-x64')
 monkeypatch.setattr(steward,'is_bundled_payload',lambda root:True)
 def forbidden(*a,**k):raise AssertionError('local LLM work forbidden in Windows distribution')
 monkeypatch.setattr(binaries.pm,'ensure',forbidden)
 monkeypatch.setattr(binaries,'_installed',forbidden)
 monkeypatch.setattr(bootstrap,'adopt_legacy_models',forbidden)
 monkeypatch.setattr(endpoint,'_state_endpoint',forbidden)
 assert binaries.installed_engine('auto') is None
 for backend in binaries.BACKEND_PACKAGES:
  assert 'not bundled' in binaries.unavailable_reason(backend)
  with pytest.raises(binaries.BinaryResolutionError,match='not bundled'):binaries.ensure_engine(backend)
 assert bootstrap.ensure_local_runtime({'local_runtime':{'enabled':True}},force=True) is None
 assert endpoint.resolve_llamacpp_endpoint({'model':{'provider':'llamacpp','default':'saved-model'}},0) is None

def test_windows_source_still_has_engine_capability(monkeypatch):
 from hermes_cli import steward
 monkeypatch.setattr('pm.store.current_target',lambda:'win32-x64');monkeypatch.setattr(steward,'is_bundled_payload',lambda root:False)
 assert binaries.unavailable_reason('cpu') is None

def test_saved_local_model_is_truthfully_unavailable_without_config_mutation(monkeypatch):
 from hermes_cli import steward,runtime_provider_custom
 from hermes_cli.web_routers import local_models
 cfg={'provider':'llamacpp','default':'saved-model'}
 monkeypatch.setattr('pm.store.current_target',lambda:'win32-x64');monkeypatch.setattr(steward,'is_bundled_payload',lambda root:True)
 monkeypatch.setattr(local_models,'_runtime_section',lambda:{'enabled':True})
 monkeypatch.setattr(local_models,'_active_llamacpp_model_id',lambda:'saved-model')
 status=local_models.local_models_status()
 assert status['runtime_installed'] is False and status['server_running'] is False
 assert status['active_model_id']=='saved-model' and 'not bundled' in status['unavailable_reason']
 with pytest.raises(ValueError,match='not bundled'):runtime_provider_custom._resolve_llamacpp_runtime('llamacpp',None)
 assert cfg=={'provider':'llamacpp','default':'saved-model'}
