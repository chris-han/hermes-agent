"""Product assembly must feed the real plugin discovery/admission boundary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('disabled', [False, True])
def test_staged_product_backend_is_discovered_without_user_activation(monkeypatch, tmp_path, disabled):
    from scripts.bundles.native import stage_distribution_assets
    import pm.features
    repo = tmp_path / 'repo'
    parent = repo / 'semantier-runtime'
    source = Path(__file__).resolve().parents[2] / 'semantier-runtime/src/plugins/sensevoice_stt'
    target = parent / 'src/plugins/sensevoice_stt'
    shutil.copytree(source, target)
    files = [{'path': str(p.relative_to(parent)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in target.iterdir() if p.is_file()]
    (parent / 'composition.json').write_text(json.dumps({'files': files, 'models': []}))
    monkeypatch.setattr(pm.features, 'distribution_extras', lambda *a, **kw: ['sensevoice'])
    stage_distribution_assets(repo, tmp_path / 'cache')
    assert (repo / 'plugins/sensevoice_stt/provider.py').read_bytes() == (source / 'provider.py').read_bytes()
    home = tmp_path / 'home'; home.mkdir()
    config = 'stt:\n  provider: sensevoice\nplugins:\n  enabled: []\n'
    if disabled:
        config += '  disabled: [sensevoice_stt]\n'
    (home / 'config.yaml').write_text(config)
    script = '''
from hermes_cli.plugins import PluginManager
from agent.transcription_registry import get_provider
import os
manager = PluginManager()
manager.discover_and_load()
provider = get_provider('sensevoice')
assert (provider is None) == (os.environ['EXPECT_DISABLED'] == '1')
if provider is not None:
    assert manager._plugins['sensevoice_stt'].manifest.source == 'bundled'
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
        env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_BUNDLED_PLUGINS': str(repo / 'plugins'),
             'EXPECT_DISABLED': str(int(disabled))})
    assert result.returncode == 0, result.stdout + result.stderr
    assert (home / 'config.yaml').read_text() == config


def test_ordinary_distribution_does_not_stage_product_plugin(monkeypatch, tmp_path):
    import pm.features
    from scripts.bundles.native import stage_distribution_assets
    monkeypatch.setattr(pm.features, 'distribution_extras', lambda *a, **kw: None)
    stage_distribution_assets(tmp_path, tmp_path / 'cache')
    assert not (tmp_path / 'plugins').exists()
