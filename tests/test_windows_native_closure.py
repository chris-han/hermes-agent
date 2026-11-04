"""Exercise the MSIX native-closure gate with a minimal package family."""
import json
import struct
import zipfile

import pytest

from scripts.check_windows_native_closure import BASE, REQUIRED, inspect


def package(tmp_path, *, omit=None, wrong_arch=False, mismatch=False):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    root = {'files': {}}
    files = {'dist/electron-main.mjs': b'import "node-pty"'}
    pe = bytearray(128)
    pe[:2] = b'MZ'
    struct.pack_into('<I', pe, 60, 64)
    pe[64:68] = b'PE\0\0'
    struct.pack_into('<H', pe, 68, 0x14c if wrong_arch else 0x8664)
    for name in REQUIRED:
        data = (json.dumps({'name': 'node-pty', 'version': '1.1.0', 'main': 'lib/index.js'}).encode()
                if name == 'package.json' else bytes(pe) if name.endswith(('.node', '.dll', '.exe')) else b'module.exports = {}')
        target = prepared / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data + (b'changed' if mismatch and name == 'lib/index.js' else b''))
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
