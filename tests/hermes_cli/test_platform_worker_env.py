"""Internal PG workers retain their fixed store connection, never platform custody."""

import json
import os
import subprocess
import sys

from tests.hermes_cli.test_kanban_worker_terminal_scope import profile_b as profile_b, _spawn_env_for_profile_b
from tests.tools._child_env_fixtures import child_env as child_env
from tools.environments.local_env_policy import protect_platform_secret_names


def test_trusted_worker_keeps_pg_but_customer_descendant_cannot(child_env, profile_b, tmp_path, monkeypatch):
    platform = {
        'HERMES_STATE_BACKEND': 'postgresql', 'SEMANTIER_STORAGE_BACKEND': 'postgresql',
        'HERMES_DATABASE_URL': 'fake-native-store', 'SEMANTIER_DATABASE_URL': 'fake-core-store',
        'HERMES_STATE_SCOPE_MAP': '{}', 'SEMANTIER_AUTHORITY_STORE_MAP': '{}',
        'SEMANTIER_NATIVE_CREDENTIAL_KEY_REF': 'USER_CUSTODY_MATERIAL',
        'USER_CUSTODY_MATERIAL': 'fake-platform-only',
        'SEMANTIER_NATIVE_CREDENTIAL_ENCRYPTION_KEY': 'fake-platform-only',
        'SEMANTIER_MODEL_GATEWAY_PROXY_KEY': 'fake-platform-only',
        'SEMANTIER_SESSION_SECRET': 'fake-platform-only',
    }
    for name, value in platform.items():
        monkeypatch.setenv(name, value)
    protect_platform_secret_names(os.environ)
    env = _spawn_env_for_profile_b(monkeypatch, tmp_path)
    assert env['HERMES_DATABASE_URL'] == 'fake-native-store'
    assert env['SEMANTIER_DATABASE_URL'] == 'fake-core-store'
    assert env['HERMES_STATE_BACKEND'] == 'postgresql'
    denied = [name for name in platform if name not in {
        'HERMES_STATE_BACKEND', 'SEMANTIER_STORAGE_BACKEND', 'HERMES_DATABASE_URL',
        'SEMANTIER_DATABASE_URL', 'HERMES_STATE_SCOPE_MAP', 'SEMANTIER_AUTHORITY_STORE_MAP'}]
    assert not set(denied) & set(env)
    # This is a real trusted worker process importing its real customer-child
    # builder; only its external worker launch/supervision boundary was captured.
    code = '''import json,os,subprocess,sys
from tools.environments.local import _sanitize_subprocess_env
assert os.environ['HERMES_DATABASE_URL']=='fake-native-store'
child=_sanitize_subprocess_env(dict(os.environ))
result=subprocess.run([sys.executable,'-c',%r],env=child,capture_output=True,text=True,check=True)
print(result.stdout)
''' % ('import os,json;print(json.dumps({k:os.environ.get(k) for k in ' + repr(list(platform)) + '}))')
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                            text=True, check=True, timeout=30)
    observed = json.loads(result.stdout)
    assert all(observed[name] is None for name in [*denied, 'HERMES_DATABASE_URL', 'SEMANTIER_DATABASE_URL'])
