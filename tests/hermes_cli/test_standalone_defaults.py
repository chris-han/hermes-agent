"""Exercise product composition in a fresh interpreter, without leaking mutated defaults."""
import os
from pathlib import Path
import subprocess
import sys


def test_product_initial_connection_and_saved_choices(tmp_path):
    script = '''
import pm.features
pm.features.runtime_distribution_extras = lambda: ['sensevoice']
from hermes_cli.config import load_config, save_config
config = load_config()
assert config['model']['provider'] == 'semantier-cloud'
assert not config['model']['default']
assert not config['providers']['semantier-cloud']['base_url']
assert not config['providers']['semantier-cloud'].get('api_key')
assert config['stt']['provider'] == 'sensevoice'
from agent.semantier_gateway import billing_backend, billing_request
assert billing_backend() == 'semantier'
assert billing_request('state', schema_version='semantier.billing.v2')['error'] == 'provider_not_configured'
config['model'] = {'provider':'custom','default':'saved-model','base_url':'http://127.0.0.1:7777/v1'}
config['stt']['provider'] = 'local'
config['billing'] = {'backend':'nous'}
save_config(config)
again = load_config()
assert again['model'] == config['model']
assert again['stt']['provider'] == 'local'
assert again['billing']['backend'] == 'nous'
'''
    env = {**os.environ, 'HERMES_HOME': str(tmp_path), 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    subprocess.run([sys.executable, '-c', script], env=env, check=True, capture_output=True, text=True)
