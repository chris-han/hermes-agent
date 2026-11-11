"""Installed browser API validation is pure; rejected payloads never launch a backend."""
import json
from pathlib import Path

import pytest

from tools.browser_exec_contract import installed_api, validate_code


def test_namespace_comes_from_the_installed_helper_definitions():
    api = installed_api()
    assert api['module'] == 'browser_harness.helpers'
    assert 'goto_url' in api['helpers'] and 'js' in api['helpers']
    assert api['source_sha256'] and api['version']
    assert validate_code("new_tab('http://localhost/fixture'); print(js('document.title'))", api=api) is None


@pytest.mark.parametrize('code', [
    'import math; print(math.sqrt(9))',
    'from json import dumps as serialize; print(serialize({"x": 1}))',
    'def custom(x):\n return x + 1\nprint(custom(2))',
    'def outer(fn):\n return fn(2)\nprint(outer(abs))',
    'values = [abs(x) for x in [-1, 2]]; print(values)',
])
def test_supported_python_and_lexical_bindings_are_not_a_module_allowlist(code):
    assert validate_code(code) is None


def test_unsupported_helper_import_has_actual_alternatives():
    error = validate_code('from browser_helpers import goto_url, js\ngoto_url("http://localhost")')
    assert error['code'] == 'UNSUPPORTED_HELPER_IMPORT'
    assert error['module'] == 'browser_helpers'
    assert 'goto_url' in error['supported_helpers']
    assert error['recovery']['same_context'] is True


def test_unknown_helper_and_unknown_installed_helper_export():
    assert validate_code('unavailable_browser_helper()')['code'] == 'UNKNOWN_HELPER'
    assert validate_code('from browser_harness.helpers import unavailable_browser_helper')['code'] == 'UNKNOWN_HELPER'


def test_backend_capability_unavailable_is_distinct_from_missing_api():
    error = validate_code('capture_screenshot()', unavailable_cdp={'Page.captureScreenshot'})
    assert error['code'] == 'BACKEND_CAPABILITY_UNAVAILABLE'
    assert error['helper'] == 'capture_screenshot'


def test_custom_helper_metadata_does_not_execute_workspace_python(tmp_path):
    marker = tmp_path / 'metadata-executed'
    (tmp_path / 'agent_helpers.py').write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\ndef custom_task_helper():\n return 'ok'\n", encoding='utf-8')
    api = installed_api(workspace=tmp_path)
    assert 'custom_task_helper' in api['helpers'] and not marker.exists()
    assert validate_code('print(custom_task_helper())', api=api) is None


def test_unverifiable_dynamic_namespace_is_reported_before_execution():
    assert validate_code("globals()['unknown_helper']()")['code'] == 'UNVERIFIABLE_HELPER_REFERENCE'


def test_preflight_precedes_url_resolution_backend_and_payload_side_effects(monkeypatch, tmp_path):
    from tools import browser_use_cli as cli
    calls = []
    monkeypatch.setattr(cli, '_code_url_verdict', lambda *_: calls.append('url') or pytest.fail('URL resolution before invalid API rejected'))
    monkeypatch.setattr(cli, '_route_backend', lambda *_a, **_k: pytest.fail('Backend before rejection'))
    monkeypatch.setattr(cli, '_run_cli_killing_process_group', lambda *_: pytest.fail('User payload must not run'))
    marker = tmp_path / 'payload-executed'
    code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\nfrom browser_helpers import goto_url\ngoto_url('http://localhost')"
    result = json.loads(cli.browser_exec(code, session='same-context', task_id='scope'))
    assert result['code'] == 'UNSUPPORTED_HELPER_IMPORT'
    assert not calls and not marker.exists()


def test_valid_correction_keeps_the_same_independent_session(monkeypatch):
    from tools import browser_use_cli as cli
    from types import SimpleNamespace
    seen = []
    import sys
    monkeypatch.setattr(cli, '_find_cli', lambda: [sys.executable, '-m', 'browser_harness.run'])
    monkeypatch.setattr(cli, '_code_url_verdict', lambda *_: SimpleNamespace(blocked=None, local_sidecar=False))
    monkeypatch.setattr(cli, '_route_backend', lambda env, session, task, *_a, **_kw: seen.append((session, task)))
    monkeypatch.setattr(cli, '_attach_vault_supervisor', lambda *_: None)
    monkeypatch.setattr(cli, '_workspace_dir', lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, '_run_cli_killing_process_group', lambda *_: SimpleNamespace(returncode=0, stdout='verified', stderr=''))
    invalid = json.loads(cli.browser_exec('from browser_helpers import js', session='owned', task_id='task'))
    assert invalid['code'] == 'UNSUPPORTED_HELPER_IMPORT' and not seen
    valid = json.loads(cli.browser_exec("print(js('document.title'))", session='owned', task_id='task'))
    assert valid.get('success') and seen == [('owned', 'task')], (valid, seen)


def test_helper_import_from_wrong_package_cannot_pass_as_an_ordinary_import():
    error = validate_code('from browser_harness import goto_url')
    assert error is not None and error['code'] == 'UNSUPPORTED_HELPER_IMPORT'


def test_builtin_alias_imports_are_supported_python():
    assert validate_code('import os.path; print(os.path.join("a", "b"))') is None


@pytest.mark.parametrize('code', [
    'def custom(globals):\n return globals()\nprint(custom(lambda: 1))',
    'def custom(capture_screenshot):\n return capture_screenshot()\nprint(custom(lambda: 1))',
])
def test_local_parameters_do_not_resolve_to_runtime_helpers(code):
    assert validate_code(code, unavailable_cdp={'Page.captureScreenshot'}) is None


def test_missing_installed_metadata_is_a_structured_contract_failure(monkeypatch):
    import tools.browser_exec_contract as contract
    def unavailable(_workspace):
        raise ImportError('package metadata absent')
    monkeypatch.setattr(contract, 'installed_api', unavailable)
    error = validate_code('print(1)')
    assert error['code'] == 'API_CONTRACT_UNAVAILABLE'


def test_invalid_python_is_distinct_from_unreadable_installed_source(monkeypatch):
    import tools.browser_exec_contract as contract
    api = installed_api()
    assert validate_code('print(', api=api)['code'] == 'INVALID_ARGUMENT'
    def broken_source(_workspace):
        raise SyntaxError('installed helpers invalid')
    monkeypatch.setattr(contract, 'installed_api', broken_source)
    assert validate_code('print(1)')['code'] == 'API_CONTRACT_UNAVAILABLE'


@pytest.mark.parametrize('code', [
    'operation = unavailable_browser_helper\noperation()',
    'from browser_harness import helpers as h\noperation = h.unavailable_browser_helper\noperation()',
])
def test_unavailable_helper_reference_is_rejected_before_indirect_call(code):
    assert validate_code(code)['code'] == 'UNKNOWN_HELPER'


def test_helper_module_exported_standard_module_is_not_a_helper_allowlist():
    assert validate_code('from browser_harness import helpers as h\nprint(h.os.path.basename("/fixture"))') is None


@pytest.mark.parametrize(('expression', 'code'), [
    ('getattr(h, "missing_browser_helper")', 'UNKNOWN_HELPER'),
    ('getattr(h, name)', 'UNVERIFIABLE_HELPER_REFERENCE'),
])
def test_helper_namespace_lookup_is_validated_without_executing_it(expression, code):
    source = f'from browser_harness import helpers as h\nname = "missing_browser_helper"\noperation = {expression}\noperation()'
    assert validate_code(source)['code'] == code


def test_known_literal_helper_lookup_and_ordinary_getattr_remain_supported():
    assert validate_code('from browser_harness import helpers as h\noperation = getattr(h, "goto_url")') is None
    assert validate_code('import math\nprint(getattr(math, "pi"))') is None



def test_private_helper_is_not_assumed_to_be_preimported():
    assert validate_code('_load_env()')['code'] == 'UNKNOWN_HELPER'
    assert validate_code('from browser_harness.helpers import _load_env') is None


def test_custom_private_names_are_not_copied_into_the_runtime_namespace(tmp_path):
    (tmp_path / 'agent_helpers.py').write_text('def _fixture_private():\n return 1\n')
    api = installed_api(workspace=tmp_path)
    assert '_fixture_private' not in api['globals']
    assert validate_code('_fixture_private()', api=api)['code'] == 'UNKNOWN_HELPER'


def test_existing_wrong_sdk_module_cannot_import_an_unavailable_helper():
    assert validate_code('from browser_harness.daemon import goto_url')['code'] == 'UNSUPPORTED_HELPER_IMPORT'
