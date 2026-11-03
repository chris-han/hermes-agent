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
    product = repo / 'semantier-product'
    original = Path(__file__).resolve().parents[2] / 'semantier-product'
    shutil.copytree(original, product)
    source = product / 'semantier_product/sensevoice_stt'
    monkeypatch.setattr(pm.features, 'distribution_extras', lambda *a, **kw: ['sensevoice'])
    from pm.downloader import Download
    class AssetTransferStopped(Exception):
        pass
    def stop_before_transfer(self):
        raise AssetTransferStopped()
    monkeypatch.setattr(Download, 'run', stop_before_transfer)
    with pytest.raises(AssetTransferStopped):
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
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False,
        env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_BUNDLED_PLUGINS': str(repo / 'plugins'),
             'EXPECT_DISABLED': str(int(disabled))})
    assert result.returncode == 0, result.stdout + result.stderr
    assert (home / 'config.yaml').read_text() == config


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'hash', 'url'])
def test_staging_refuses_unbound_model_manifest_before_download(monkeypatch, tmp_path, mutation):
    import pm.features
    from scripts.bundles.native import stage_distribution_assets
    repo = tmp_path / 'repo'
    product = repo / 'semantier-product'
    shutil.copytree(Path(__file__).resolve().parents[2] / 'semantier-product', product)
    receipt_path = product / 'source-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if mutation == 'missing':
        receipt['models'] = []
    elif mutation == 'duplicate':
        receipt['models'][1] = receipt['models'][0].copy()
    elif mutation == 'hash':
        receipt['models'][0]['sha256'] = '0' * 64
    else:
        receipt['models'][0]['url'] = 'https://example.invalid/model'
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(pm.features, 'distribution_extras', lambda *a, **kw: ['sensevoice'])
    with pytest.raises(ValueError, match='required model assets'):
        stage_distribution_assets(repo, tmp_path / 'cache')


def test_ordinary_distribution_does_not_stage_product_plugin(monkeypatch, tmp_path):
    import pm.features
    from scripts.bundles.native import stage_distribution_assets
    monkeypatch.setattr(pm.features, 'distribution_extras', lambda *a, **kw: None)
    stage_distribution_assets(tmp_path, tmp_path / 'cache')
    assert not (tmp_path / 'plugins').exists()
