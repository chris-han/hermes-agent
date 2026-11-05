"""Exercise the MSIX native-closure gate with a minimal package family."""
import json
import struct
import zipfile

import pytest

from scripts.check_windows_native_closure import BASE, REQUIRED, inspect


def package(tmp_path, *, omit=None, wrong_arch=False, mismatch=False, http3_problem=None, notice_problem=None, pruned_metadata=False, changed_metadata=None):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    root = {'files': {}}
    files = {'dist/electron-main.mjs': b'import "node-pty"'}
    from pathlib import Path
    notices = Path(__file__).resolve().parents[1] / 'apps/desktop/public/third-party/semantier-cloud-http3'
    for source in notices.glob('*.txt'):
        if notice_problem == 'missing' and source.name == 'curl.txt':
            continue
        files['public/third-party/semantier-cloud-http3/' + source.name] = (
            b'altered notice' if notice_problem == 'altered' and source.name == 'curl.txt'
            else source.read_bytes())
    pe = bytearray(128)
    pe[:2] = b'MZ'
    struct.pack_into('<I', pe, 60, 64)
    pe[64:68] = b'PE\0\0'
    struct.pack_into('<H', pe, 68, 0x14c if wrong_arch else 0x8664)
    for name in REQUIRED:
        data = (json.dumps({'name': 'node-pty', 'version': '1.1.0', 'main': 'lib/index.js'}).encode()
                if name == 'package.json' else bytes(pe) if name.endswith(('.node', '.dll', '.exe')) else b'module.exports = {}')
        if name == 'package.json' and pruned_metadata:
            metadata = json.loads(data)
            metadata.update(scripts={'install': 'fixture'}, keywords=['fixture'],
                            bugs={'url': 'https://example.invalid/issues'},
                            dependencies={'node-addon-api': '^8.0.0'})
            data = json.dumps(metadata).encode()
        target = prepared / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data + (b'changed' if mismatch and name == 'lib/index.js' else b''))
        if name == 'package.json' and pruned_metadata:
            metadata = json.loads(data)
            for key in ('scripts', 'keywords', 'bugs'):
                metadata.pop(key)
            if changed_metadata:
                metadata.update(changed_metadata)
            data = json.dumps(metadata, ensure_ascii=False, indent=2).encode()
        if name != omit:
            files[BASE + name] = data
    body = b''
    unpacked = {}
    for name, data in files.items():
        node = root
        parts = name.split('/')
        for part in parts[:-1]:
            node = node.setdefault('files', {}).setdefault(part, {'files': {}})
        if name.startswith(BASE):
            node['files'][parts[-1]] = {'size': len(data), 'unpacked': True}
            unpacked['app/resources/app.asar.unpacked/' + name] = data
        else:
            node['files'][parts[-1]] = {'size': len(data), 'offset': str(len(body))}
            body += data
    index = json.dumps(root, separators=(',', ':')).encode()
    header_size = len(index) + 8
    asar = struct.pack('<4I', 4, header_size, header_size - 4, len(index)) + index + body
    msix = tmp_path / 'fixture.msix'
    with zipfile.ZipFile(msix, 'w') as archive:
        archive.writestr('app/resources/app.asar', asar)
        for name, data in unpacked.items():
            archive.writestr(name, data)
        site = 'app/resources/agent-payload/venv/Lib/site-packages/'
        native = bytes(pe)
        if http3_problem == 'wrong-native-architecture':
            bad = bytearray(native)
            struct.pack_into('<H', bad, 68, 0x14c)
            native = bytes(bad)
        components = {
            'curl_cffi/__init__.py': b'fixture',
            'curl_cffi/requests/__init__.py': b'fixture',
            'curl_cffi/_wrapper.pyd': bytes(pe),
            'curl_cffi.libs/libcurl-impersonate-ec8ead0e422fe4ae62a99233a63f6007.dll': native,
            '_cffi_backend.cp314-win_amd64.pyd': bytes(pe),
            'certifi/cacert.pem': b'fixture certificate bundle',
            'curl_cffi-0.16.0.dist-info/METADATA': b'Name: curl-cffi\nVersion: 0.16.0\n',
        }
        if http3_problem == 'missing-wrapper':
            del components['curl_cffi/_wrapper.pyd']
        elif http3_problem == 'missing-dll':
            del components['curl_cffi.libs/libcurl-impersonate-ec8ead0e422fe4ae62a99233a63f6007.dll']
        elif http3_problem == 'wrong-version':
            components['curl_cffi-0.16.0.dist-info/METADATA'] = b'Name: curl-cffi\nVersion: 0.15.0\n'
        for name, data in components.items():
            archive.writestr(site + name, data)
    return msix, prepared


def test_valid_package_and_prepared_tree(tmp_path):
    msix, prepared = package(tmp_path)
    assert inspect(msix, prepared)['nodePtyFiles'] == len(REQUIRED)


@pytest.mark.parametrize(('kwargs', 'error'), [
    ({'omit': 'package.json'}, 'missing ASAR path'),
    ({'wrong_arch': True}, 'not Windows x64'),
    ({'mismatch': True}, 'content mismatch'),
])
def test_rejects_incomplete_or_unbound_native_package(tmp_path, kwargs, error):
    msix, prepared = package(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=error):
        inspect(msix, prepared)


@pytest.mark.parametrize('problem', ['missing-wrapper', 'missing-dll', 'wrong-native-architecture', 'wrong-version'])
def test_rejects_incomplete_http3_windows_payload(tmp_path, problem):
    msix, prepared = package(tmp_path, http3_problem=problem)
    with pytest.raises(ValueError, match='HTTP/3'):
        inspect(msix, prepared)


@pytest.mark.parametrize('problem', ['missing', 'altered'])
def test_rejects_missing_or_changed_http3_license_notices(tmp_path, problem):
    msix, prepared = package(tmp_path, notice_problem=problem)
    with pytest.raises(ValueError, match='HTTP/3 notice'):
        inspect(msix, prepared)


def test_accepts_electron_builder_nonruntime_metadata_pruning(tmp_path):
    msix, prepared = package(tmp_path, pruned_metadata=True)
    assert inspect(msix, prepared)['nodePtyFiles'] == len(REQUIRED)


@pytest.mark.parametrize('change', [
    {'version': '1.1.1'}, {'main': 'lib/other.js'},
    {'dependencies': {'node-addon-api': '^9.0.0'}}, {'license': 'changed'},
])
def test_pruning_does_not_allow_other_metadata_changes(tmp_path, change):
    msix, prepared = package(tmp_path, pruned_metadata=True, changed_metadata=change)
    with pytest.raises(ValueError):
        inspect(msix, prepared)


def test_pruning_does_not_allow_javascript_changes(tmp_path):
    msix, prepared = package(tmp_path, pruned_metadata=True, mismatch=True)
    with pytest.raises(ValueError, match='content mismatch'):
        inspect(msix, prepared)
