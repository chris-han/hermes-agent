"""Cron's trusted external-worker handoff has fixed PG authority, no custody key."""

import os

from tests.cron.test_restart_safe_worker import _stub_external_worker_launch
from tests.tools._child_env_fixtures import child_env as child_env


def test_cron_handoff_retains_only_fixed_pg_authority(child_env, monkeypatch):
    from cron import scheduler
    from tools.environments.local_env_policy import protect_platform_secret_names
    from tools.process_registry import GatewayChildDispatch
    platform = {
        'HERMES_STATE_BACKEND': 'postgresql', 'SEMANTIER_STORAGE_BACKEND': 'postgresql',
        'HERMES_DATABASE_URL': 'fake-native-store', 'SEMANTIER_DATABASE_URL': 'fake-core-store',
        'SEMANTIER_NATIVE_CREDENTIAL_KEY_REF': 'USER_CRON_CUSTODY',
        'USER_CRON_CUSTODY': 'fake-platform-only', 'SEMANTIER_SESSION_SECRET': 'fake-platform-only',
        '_HERMES_FORCE_USER_CRON_CUSTODY': 'fake-platform-only',
    }
    for name, value in platform.items():
        monkeypatch.setenv(name, value)
    protect_platform_secret_names(os.environ)
    monkeypatch.setattr(scheduler, '_get_hermes_home', lambda: child_env)
    monkeypatch.setattr('tools.process_registry.restart_safe_gateway_child_argv',
                        lambda command, **_: GatewayChildDispatch('scoped', ['scope', '--', *command]))
    spawned, _, _, _ = _stub_external_worker_launch(scheduler, monkeypatch)
    assert scheduler._launch_external_cron_worker({'id': 'cron-1', 'execution_id': 'exec-1', 'prompt': 'work'})
    env = spawned[0][1]['env']
    assert env['HERMES_DATABASE_URL'] == 'fake-native-store'
    assert env['SEMANTIER_DATABASE_URL'] == 'fake-core-store'
    assert env['HERMES_STATE_BACKEND'] == 'postgresql'
    assert not {'SEMANTIER_NATIVE_CREDENTIAL_KEY_REF', 'USER_CRON_CUSTODY',
                '_HERMES_FORCE_USER_CRON_CUSTODY', 'SEMANTIER_SESSION_SECRET'} & set(env)
