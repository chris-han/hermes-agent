"""Interrupted reservations require acceptance of the exact signed closure."""
import json

import pytest

from scripts.releases.windows_allocation import admit_windows_allocation


def evidence(tmp_path):
    request = {'commit': 'a' * 40, 'version': '0.0.86', 'windowsVersion': '0.0.86.0',
               'sequence': 86, 'publicBase': 'https://download.semantier.com'}
    allocation = {'kind': 'semantier-windows-build-version-v1', 'version': '0.0.86',
                  'sourceCommit': 'a' * 40, 'reservedAt': '2026-11-05T05:33:01Z'}
    row = {'key': 'releases/channel-builds/' + 'b' * 32 + '/main.msix', 'size': 10, 'sha256': 'c' * 64}
    manifest = {'request': request, 'packages': [{'artifact': row}], 'installation': {'objects': [row]}}
    receipt = {'result': 'LOCAL_PACKAGE_REPAIR_PASS', 'version': '0.0.86.0', 'sourceCommit': 'a' * 40,
               'artifactSha256': 'c' * 64, 'profilePreserved': True, 'sessionsPreserved': True,
               'window': 'PASS', 'pty': 'PASS', 'moduleErrors': [],
               'verifiedPackages': [{'name': 'main.msix', 'size': 10, 'sha256': 'c' * 64}]}
    ui = {'result': 'WINDOW_PTY_PASS', 'backendReady': True, 'windowOpened': True, 'errors': [],
          'binding': {'version': '0.0.86', 'sourceCommit': 'a' * 40, 'artifactSha256': 'c' * 64},
          'identity': {'packaged': True, 'platform': 'win32'}}
    documents = {'build-version.json': allocation, 'ledger.json': allocation, 'acceptance.json': receipt,
                 'ui.json': ui, 'channel-request.json': {**request, 'publicBase': 'https://download.semantier.cn'},
                 'signed/receipt.json': {'signedSha256': 'c' * 64, 'sourceCommit': 'a' * 40}}
    for name, document in documents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding='utf-8')
    return manifest


def admit(root, manifest):
    return admit_windows_allocation(root, root / 'build-version.json', root / 'ledger.json',
                                    root / 'acceptance.json', root / 'ui.json', manifest)


def test_accepts_reserved_version_with_same_signed_bytes_for_delivery_alias(tmp_path):
    assert admit(tmp_path, evidence(tmp_path)) is True


@pytest.mark.parametrize('name,changes', [
    ('ledger.json', {'sourceCommit': 'd' * 40}),
    ('acceptance.json', {'artifactSha256': 'd' * 64}),
    ('acceptance.json', {'pty': 'PENDING'}),
    ('acceptance.json', {'sessionsPreserved': False}),
    ('ui.json', {'backendReady': False}),
    ('channel-request.json', {'sequence': 85}),
])
def test_rejects_conflicting_source_or_unobserved_acceptance(tmp_path, name, changes):
    manifest = evidence(tmp_path)
    path = tmp_path / name
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}), encoding='utf-8')
    with pytest.raises(ValueError):
        admit(tmp_path, manifest)
