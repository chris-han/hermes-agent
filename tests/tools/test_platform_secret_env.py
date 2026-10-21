"""Platform custody stays outside real customer tool children, with fake secrets only."""

import json
import os
import shlex
import shutil
import sys

import pytest

from tests.tools._child_env_fixtures import child_env as child_env, observe_child, observe_terminal, run_code
from tools.environments import local
from tools.env_passthrough import register_env_passthrough


PLATFORM_NAMES = (
    'SEMANTIER_SESSION_SECRET', 'SEMANTIER_DATABASE_URL', 'HERMES_DATABASE_URL',
    'SEMANTIER_MODEL_GATEWAY_DATABASE_URL', 'SEMANTIER_MODEL_GATEWAY_PROXY_KEY',
    'SEMANTIER_NATIVE_CREDENTIAL_KEY', 'SEMANTIER_NATIVE_CREDENTIAL_ENCRYPTION_KEY',
    'SEMANTIER_BILLING_DATABASE_URL', 'SEMANTIER_MIGRATION_DATABASE_URL',
    'LITELLM_DATABASE_URL', 'LITELLM_MIGRATION_DATABASE_URL',
)
REFERENCES = {
    'SEMANTIER_NATIVE_CREDENTIAL_KEY_REF': 'USER_ACCOUNT_MATERIAL',
    'SEMANTIER_MODEL_GATEWAY_DATABASE_SECRET_REF': 'LANG_ACCOUNT_STORAGE',
    'SEMANTIER_MODEL_GATEWAY_PROXY_SECRET_REF': 'TERM_ACCOUNT_CONTROL',
}


@pytest.fixture
def platform_names(child_env, monkeypatch):
    names = (*PLATFORM_NAMES, *REFERENCES, *REFERENCES.values())
    # Names accepted before platform ownership is configured must become denied later.
    register_env_passthrough([*REFERENCES.values(), 'MY_APP_KEY'])
    for name in names:
        monkeypatch.setenv(name, REFERENCES.get(name, 'fake-platform-only'))
    from tools.environments.local_env_policy import protect_platform_secret_names
    protect_platform_secret_names(os.environ)
    monkeypatch.setenv('MY_APP_KEY', 'workspace-byok')
    monkeypatch.setenv('MY_CUSTOM_VAR', 'workspace-value')
    return names


@pytest.mark.parametrize('surface', [
    'foreground', 'background', 'nonterminal', 'served', 'served_base', 'factory_no_scrub', 'python', 'managed'])
def test_customer_children_cannot_inherit_platform_authority(child_env, platform_names, surface):
    names = [*platform_names, 'MY_APP_KEY', 'MY_CUSTOM_VAR']
    extras = {name: 'fake-override' for name in platform_names}
    extras.update({name: 'MY_CUSTOM_VAR' for name in REFERENCES})
    extras.update({'_HERMES_FORCE_' + name: 'fake-force' for name in platform_names})
    extras.update({name.lower(): 'fake-lowercase' for name in platform_names})
    if surface == 'foreground':
        env = local.LocalEnvironment(cwd=str(child_env), timeout=30, env=extras)
        try:
            observed = observe_terminal(env, [*names, *(name.lower() for name in platform_names)])
        finally:
            env.cleanup()
    elif surface == 'python':
        # Exercise execute_code's actual child/kernel wiring; no persistence owner is mocked.
        observed = run_code('import os,json; print(json.dumps({k:os.environ.get(k) for k in ' + repr(names) + '}))')
    elif surface == 'managed':
        from tools.environments.remote_common import resolve_passthrough_env
        forwarded, _ = resolve_passthrough_env(explicit_forward=platform_names)
        observed = observe_child(forwarded, names)
    elif surface == 'served':
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        token = set_secret_scope({**extras, 'MY_APP_KEY': 'workspace-byok'})
        try:
            observed = observe_child(local.served_profile_child_env(inherit_credentials=True), names)
        finally:
            reset_secret_scope(token)
    elif surface in {'served_base', 'factory_no_scrub'}:
        supplied = {**os.environ, **extras}
        env = (local.served_profile_child_env(base=supplied) if surface == 'served_base'
               else local.build_subprocess_env(base=supplied, scrub_secrets=False))
        observed = observe_child(env, names)
        assert not any(name.removeprefix('_HERMES_FORCE_') in platform_names for name in env)
    else:
        env = (local._sanitize_subprocess_env(dict(os.environ), extras) if surface == 'background'
               else local.hermes_subprocess_env(base_env={**os.environ, **extras}, inherit_credentials=True))
        observed = observe_child(env, [*names, *(name.lower() for name in platform_names)])
    assert {name: observed[name] for name in platform_names} == dict.fromkeys(platform_names)
    assert all(observed[name] is None for name in observed if name.lower() in {v.lower() for v in platform_names})
    if surface != 'managed':
        assert observed['MY_APP_KEY'] == 'workspace-byok'
    if surface in {'foreground', 'background', 'nonterminal'}:
        assert observed['MY_CUSTOM_VAR'] == 'workspace-value'
    assert os.environ[REFERENCES['SEMANTIER_NATIVE_CREDENTIAL_KEY_REF']] == 'fake-platform-only'


def test_reference_override_cannot_hide_platform_name(child_env, platform_names):
    from tools.code_execution_env import _scrub_child_env
    from tools.env_passthrough import get_all_passthrough
    assert not set(platform_names) & get_all_passthrough()
    supplied = {**os.environ, **{name: '' for name in REFERENCES}}
    supplied['_HERMES_FORCE_USER_ACCOUNT_MATERIAL'] = 'fake-force'
    # A callback that declares everything safe cannot bypass custody ownership.
    observed = observe_child(_scrub_child_env(supplied, is_passthrough=lambda _: True), platform_names)
    assert observed == dict.fromkeys(platform_names)


def test_profile_write_cannot_release_frozen_operator_reference(child_env, platform_names):
    from hermes_cli.config import save_env_value
    pointer = 'SEMANTIER_NATIVE_CREDENTIAL_KEY_REF'
    original_name = REFERENCES[pointer]
    save_env_value(pointer, 'MY_CUSTOM_VAR')
    assert os.environ[pointer] == 'MY_CUSTOM_VAR'
    env = local.LocalEnvironment(cwd=str(child_env), timeout=30)
    try:
        observed = observe_terminal(env, [pointer, original_name, 'MY_CUSTOM_VAR'])
    finally:
        env.cleanup()
    assert observed == {pointer: None, original_name: None, 'MY_CUSTOM_VAR': 'workspace-value'}


def test_config_reference_cannot_launder_platform_value_into_custom_child_alias(child_env, platform_names):
    from hermes_cli.config import _expand_env_vars
    env_values = _expand_env_vars({
        'MY_CUSTOM_ALIAS': '${USER_ACCOUNT_MATERIAL}',
        'MY_POINTER_ALIAS': '${SEMANTIER_NATIVE_CREDENTIAL_KEY_REF}',
        'MY_BYOK_ALIAS': '${MY_APP_KEY}',
    })
    assert env_values == {
        'MY_CUSTOM_ALIAS': '${USER_ACCOUNT_MATERIAL}',
        'MY_POINTER_ALIAS': '${SEMANTIER_NATIVE_CREDENTIAL_KEY_REF}',
        'MY_BYOK_ALIAS': 'workspace-byok',
    }
    env = local.LocalEnvironment(cwd=str(child_env), timeout=30, env=env_values)
    try:
        assert observe_terminal(env, env_values) == env_values
    finally:
        env.cleanup()


def test_explicit_docker_env_cannot_restore_platform_names(child_env, platform_names):
    from tools.environments.docker import _normalize_env_dict
    env = _normalize_env_dict({**dict.fromkeys(platform_names, 'fake-override'),
                               'MY_APP_KEY': 'workspace-byok', 'MY_CUSTOM_VAR': 'workspace-value'})
    assert env == {'MY_APP_KEY': 'workspace-byok', 'MY_CUSTOM_VAR': 'workspace-value'}
    assert observe_child(env, platform_names) == dict.fromkeys(platform_names)


@pytest.mark.parametrize('overrides', [False, True])
def test_transport_client_child_never_inherits_platform_custody(child_env, platform_names, overrides):
    from tools.environments.remote_common import client_env_with
    extras = dict.fromkeys(platform_names, 'fake-client-override') if overrides else {}
    env = client_env_with(extras)
    observed = observe_child(env, [*platform_names, 'MY_APP_KEY'])
    assert {name: observed[name] for name in platform_names} == dict.fromkeys(platform_names)
    assert observed['MY_APP_KEY'] == 'workspace-byok'


def test_real_ssh_proxycommand_never_receives_platform_custody(child_env, platform_names, monkeypatch):
    from tools.environments.remote_common import run_capture
    ssh = shutil.which('ssh')
    if ssh is None:
        pytest.skip('OpenSSH client is not installed')
    monkeypatch.setenv('SSH_AUTH_SOCK', str(child_env / 'fake-agent.sock'))
    observed_path = child_env / 'proxy-observed.json'
    names = [*platform_names, 'SSH_AUTH_SOCK', 'MY_APP_KEY']
    code = ('import json,os;from pathlib import Path;Path(' + repr(str(observed_path))
            + ').write_text(json.dumps({k:k in os.environ for k in ' + repr(names) + '}))')
    config = child_env / 'ssh-test-config'
    config.write_text('Host fixture\n HostName 127.0.0.1\n BatchMode yes\n'
                      ' ProxyCommand ' + shlex.join([sys.executable, '-c', code]) + '\n')
    # OpenSSH starts only the local ProxyCommand pipe, which exits without a
    # protocol banner or a socket connection. No remote SSH/provider is contacted.
    result = run_capture([ssh, '-F', str(config), 'fixture'], timeout=10)
    assert result.returncode != 0
    observed = json.loads(observed_path.read_text())
    assert {name: observed[name] for name in platform_names} == dict.fromkeys(platform_names, False)
    assert observed['SSH_AUTH_SOCK'] and observed['MY_APP_KEY']


@pytest.mark.parametrize('explicit', [False, True])
def test_common_popen_child_excludes_custody_even_with_explicit_env(child_env, platform_names, explicit):
    from tools.environments.base_output import _popen_bash
    names = [*platform_names, 'MY_APP_KEY']
    code = 'import json,os; print(json.dumps({k:os.environ.get(k) for k in ' + repr(names) + '}))'
    kwargs = {'env': dict(os.environ)} if explicit else {}
    proc = _popen_bash([sys.executable, '-c', code], **kwargs)
    output, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0
    observed = json.loads(output)
    assert {name: observed[name] for name in platform_names} == dict.fromkeys(platform_names)
    assert observed['MY_APP_KEY'] == 'workspace-byok'
